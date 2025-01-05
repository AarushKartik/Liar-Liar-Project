# src/bert_model.py (Now adapted for PyTorch)

import torch
import os
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from transformers import BertForSequenceClassification, BertConfig, AdamW
import torch.nn.functional as F

class BERTClassifier:
    def __init__(self,
                 num_classes=6,
                 num_epochs=3,
                 dropout_rate=0.1,
                 learning_rate=2e-5,
                 model_save_dir='weights/bert',
                 batch_size=8,
                 device=None):
        """
        num_classes: Number of classification labels
        num_epochs: How many epochs to train
        dropout_rate: Not directly used here since BertForSequenceClassification has its own dropout
        learning_rate: Learning rate for the AdamW optimizer
        model_save_dir: Directory to save the trained BERT model
        batch_size: Batch size for DataLoader
        device: Torch device (e.g. 'cuda' or 'cpu'). If None, we detect automatically.
        """
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.model_save_dir = model_save_dir
        self.batch_size = batch_size

        # Set up device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Build the BERT-based classification model
        self.model = self.build_model()
        self.model.to(self.device)

    def build_model(self):
        """
        Builds a BertForSequenceClassification model for multi-class classification.
        """
        # Load a BERT configuration with the desired number of labels
        config = BertConfig.from_pretrained("bert-base-uncased", num_labels=self.num_classes)
        # Initialize the BertForSequenceClassification model
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", config=config)
        return model

    def _create_dataloader(self, X, y=None, shuffle=False):
        """
        Creates a DataLoader from input_ids, attention_mask, and optional labels.
        X should be a tuple or dict containing 'input_ids' and 'attention_mask'.
        y can be one-hot or integer. We'll store integer labels for PyTorch classification.
        """
        if isinstance(X, dict):
            input_ids = X['input_ids']
            attention_mask = X['attention_mask']
        else:
            # Assume X is a tuple (input_ids, attention_mask)
            input_ids, attention_mask = X

        # Convert numpy arrays to torch tensors
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        if y is not None:
            # If y is one-hot, convert to integer
            if len(y.shape) > 1 and y.shape[1] > 1:
                y = np.argmax(y, axis=-1)
            y = torch.tensor(y, dtype=torch.long)
            dataset = TensorDataset(input_ids, attention_mask, y)
        else:
            dataset = TensorDataset(input_ids, attention_mask)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Trains the BERT classifier using a manual training loop.
        
        X_train, y_train: 
            - X_train -> a dict or tuple (input_ids, attention_mask)
            - y_train -> one-hot or integer labels

        X_val, y_val: Same format for validation (optional).
        """
        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)

        if X_val is not None and y_val is not None:
            val_loader = self._create_dataloader(X_val, y_val, shuffle=False)
        else:
            val_loader = None

        # Set up the optimizer
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)

        # We do not have a built-in early stopping callback here, but you could implement it
        # if desired. We do basic epoch training and track best val loss if needed.
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 3  # Used to mimic early stopping

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")

            # ---- Training ----
            self.model.train()
            total_train_loss = 0

            for batch in tqdm(train_loader, desc="Training"):
                optimizer.zero_grad()
                batch = [b.to(self.device) for b in batch]
                input_ids, attention_mask, labels = batch

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_train_loss += loss.item()

                loss.backward()
                optimizer.step()

            avg_train_loss = total_train_loss / len(train_loader)

            # ---- Validation ----
            if val_loader:
                self.model.eval()
                total_val_loss = 0
                with torch.no_grad():
                    for batch in tqdm(val_loader, desc="Validation"):
                        batch = [b.to(self.device) for b in batch]
                        input_ids, attention_mask, labels = batch

                        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                        total_val_loss += loss.item()

                avg_val_loss = total_val_loss / len(val_loader)
                print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

                # Early stopping logic (optional)
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    # Save best model weights if desired
                    # torch.save(self.model.state_dict(), 'best_model.pt')
                else:
                    patience_counter += 1
                    if patience_counter >= max_patience:
                        print("Early stopping triggered.")
                        break
            else:
                print(f"Train Loss: {avg_train_loss:.4f}")

    def predict(self, X):
        """
        Generates class predictions from the BERT model.
        X should be a dict or tuple with 'input_ids' and 'attention_mask'.
        """
        self.model.eval()
        data_loader = self._create_dataloader(X, y=None, shuffle=False)
        all_preds = []

        with torch.no_grad():
            for batch in data_loader:
                batch = [b.to(self.device) for b in batch]
                input_ids, attention_mask = batch

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                preds = torch.argmax(logits, dim=-1)
                all_preds.append(preds.cpu().numpy())

        return np.concatenate(all_preds, axis=0)

    def predict_proba(self, X):
        """
        Returns the predicted probabilities for each class.
        """
        self.model.eval()
        data_loader = self._create_dataloader(X, y=None, shuffle=False)
        all_probs = []

        with torch.no_grad():
            for batch in data_loader:
                batch = [b.to(self.device) for b in batch]
                input_ids, attention_mask = batch

                outputs = self.model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs, axis=0)

    def score(self, X, y):
        """
        Computes the accuracy score, assuming y is one-hot or integer indices.
        """
        # Convert y to class indices if one-hot
        if len(y.shape) > 1 and y.shape[1] > 1:
            y_true = np.argmax(y, axis=-1)
        else:
            y_true = y

        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred)

    def evaluate(self, X, y):
        """
        Prints and returns accuracy, confusion matrix, and classification report.
        """
        if len(y.shape) > 1 and y.shape[1] > 1:
            y_true = np.argmax(y, axis=-1)
        else:
            y_true = y

        y_pred = self.predict(X)
        accuracy = accuracy_score(y_true, y_pred)
        confusion = confusion_matrix(y_true, y_pred)
        classification_rep = classification_report(y_true, y_pred)

        print(f"Accuracy: {accuracy}")
        print("Confusion Matrix:")
        print(confusion)
        print("Classification Report:")
        print(classification_rep)

        return accuracy, confusion, classification_rep

    def save_model(self, save_dir=None):
        """
        Saves the trained model weights to the specified directory (default: weights/bert).
        """
        if save_dir is None:
            save_dir = self.model_save_dir

        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        # Save the entire model (config + weights) in Transformers format
        self.model.save_pretrained(save_dir)
        print(f"Model saved to {save_dir}")

    def __getattr__(self, name):
        """
        Allows calls to underlying model methods except for 'predict' and 'predict_proba'.
        """
        if name not in ['predict', 'predict_proba']:
            return getattr(self.model, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
