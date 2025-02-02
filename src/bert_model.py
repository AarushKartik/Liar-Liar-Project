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
        config = BertConfig.from_pretrained("bert-base-uncased", num_labels=self.num_classes)
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", config=config)
        return model

    def _create_dataloader(self, X, y=None, shuffle=False):
        if isinstance(X, dict):
            input_ids = X['input_ids']
            attention_mask = X['attention_mask']
        else:
            input_ids, attention_mask = X

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.long)

        if y is not None:
            # Convert one-hot to integer label if needed
            if len(y.shape) > 1 and y.shape[1] > 1:
                y = np.argmax(y, axis=-1)
            y = torch.tensor(y, dtype=torch.long)
            dataset = TensorDataset(input_ids, attention_mask, y)
        else:
            dataset = TensorDataset(input_ids, attention_mask)

        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        train_loader = self._create_dataloader(X_train, y_train, shuffle=True)
        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = self._create_dataloader(X_val, y_val, shuffle=False)

        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        os.makedirs("weights", exist_ok=True)

        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 3

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")

            # ---- Training ----
            self.model.train()
            total_train_loss = 0
            all_train_preds = []
            all_train_labels = []

            for batch in tqdm(train_loader, desc="Training"):
                optimizer.zero_grad()
                batch = [b.to(self.device) for b in batch]
                input_ids, attention_mask, labels = batch

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_train_loss += loss.item()
                logits = outputs.logits

                preds = torch.argmax(logits, dim=-1)
                all_train_preds.append(preds.cpu().numpy())
                all_train_labels.append(labels.cpu().numpy())

                loss.backward()
                optimizer.step()

            avg_train_loss = total_train_loss / len(train_loader)
            train_accuracy = accuracy_score(
                np.concatenate(all_train_labels), np.concatenate(all_train_preds)
            )

            # ---- Validation ----
            if val_loader:
                self.model.eval()
                total_val_loss = 0
                all_val_preds = []
                all_val_labels = []

                with torch.no_grad():
                    for batch in tqdm(val_loader, desc="Validation"):
                        batch = [b.to(self.device) for b in batch]
                        input_ids, attention_mask, labels = batch

                        outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                        total_val_loss += loss.item()

                        logits = outputs.logits
                        preds = torch.argmax(logits, dim=-1)
                        all_val_preds.append(preds.cpu().numpy())
                        all_val_labels.append(labels.cpu().numpy())

                avg_val_loss = total_val_loss / len(val_loader)
                val_accuracy = accuracy_score(
                    np.concatenate(all_val_labels), np.concatenate(all_val_preds)
                )
                print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
                print(f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

                # Early stopping check
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= max_patience:
                        print("Early stopping triggered.")
                        break
            else:
                print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

            # ---- Save checkpoint ----
            save_path = f"weights/bert_epoch_{epoch+1}.pth"
            torch.save(self.model.state_dict(), save_path)
            print(f"Model weights saved to {save_path}")

    def predict(self, X):
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
        if len(y.shape) > 1 and y.shape[1] > 1:
            y_true = np.argmax(y, axis=-1)
        else:
            y_true = y

        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred)

    def evaluate(self, X, y):
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
        if save_dir is None:
            save_dir = self.model_save_dir

        os.makedirs(save_dir, exist_ok=True)
        self.model.save_pretrained(save_dir)
        print(f"Model saved to {save_dir}")

    def get_features(self, X):
        """
        Returns the [CLS] embedding from the last hidden layer (or whichever layer you prefer).
        """
        self.model.eval()
        data_loader = self._create_dataloader(X, y=None, shuffle=False)
        all_features = []

        with torch.no_grad():
            for batch in data_loader:
                batch = [b.to(self.device) for b in batch]
                input_ids, attention_mask = batch

                # Forward pass through only the BERT encoder part
                outputs = self.model.bert(
                    input_ids, 
                    attention_mask=attention_mask, 
                    output_hidden_states=True, 
                    return_dict=True
                )
                # outputs.hidden_states is a tuple of [embeddings, hidden_layer_1, ..., hidden_layer_12]
                # So the last item is the final hidden layer
                last_hidden_state = outputs.hidden_states[-1]  # (batch_size, seq_len, hidden_dim)

                # Typically the [CLS] token embedding is at index 0
                cls_embeddings = last_hidden_state[:, 0, :]    # (batch_size, hidden_dim)

                all_features.append(cls_embeddings.cpu().numpy())

        # Concatenate all batch outputs
        all_features = np.concatenate(all_features, axis=0)
        return all_features

    def extract_feature_vectors(self, X, split_name="train"):
        """
        Convenience method:
        1. Gets the features for the given split
        2. Saves them to feature_vectors/<split_name>/bert/features.npy
        """
        features = self.get_features(X)

        out_dir = f"feature_vectors/{split_name}/bert"
        os.makedirs(out_dir, exist_ok=True)

        np.save(os.path.join(out_dir, "features.npy"), features)
        print(f"[{split_name.upper()}] Feature vectors saved to: {out_dir}/features.npy")

    def __getattr__(self, name):
        """
        Allows calls to underlying model methods except for 'predict' and 'predict_proba'.
        """
        if name not in ['predict', 'predict_proba', 'get_features', 'extract_feature_vectors']:
            return getattr(self.model, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
