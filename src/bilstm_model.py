import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, GlobalAveragePooling1D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, Callback

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import os
import shutil

class SaveModelWeightsCallback(Callback):
    def __init__(self, model_save_dir, model_name):
        super(SaveModelWeightsCallback, self).__init__()
        self.model_save_dir = model_save_dir
        self.model_name = model_name

    def on_epoch_end(self, epoch, logs=None):
        # Save model weights in .pth format
        weights_path_pth = os.path.join(self.model_save_dir, f'{self.model_name}_epoch_{epoch + 1}.pth')
        self.model.save_weights(weights_path_pth)
        print(f"Model weights saved to: {weights_path_pth}")

        # Save model weights in .txt format
        weights_path_txt = os.path.join(self.model_save_dir, f'{self.model_name}_epoch_{epoch + 1}_weights.txt')
        with open(weights_path_txt, 'w') as f:
            for layer in self.model.layers:
                weights = layer.get_weights()
                if not weights:
                    continue
                f.write(f"=== Layer: {layer.name} ===\n")
                for i, weight_array in enumerate(weights):
                    f.write(f"\nWeight {i} - Shape: {weight_array.shape}\n")
                    np.savetxt(f, weight_array.ravel(), fmt='%.6f', delimiter=' ', newline=' ')
                    f.write("\n\n")
            f.write("\n")
        print(f"Model weights saved in text format to: {weights_path_txt}")

class BiLSTMClassifier:
    def __init__(self, 
                 num_classes=6, 
                 num_epochs=3,
                 lstm_units=200,
                 dropout_rate=0.1, 
                 learning_rate=2e-5,
                 max_len=128,  # Max sequence length for padding/truncating input sequences
                 embedding_dim=100,  # Dimensionality of the word embeddings
                 vocab_size=5000,  # Vocabulary size for tokenization
                 model_save_dir='weights/bilstm',
                 model_name='bilstm'):
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.model_save_dir = model_save_dir
        self.model_name = model_name

        # Initialize tokenizer (not needed if using pre-tokenized sequences)
        self.tokenizer = None  # Remove tokenizer if not using raw text

        # Build the BiLSTM-based classification model
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        # Embedding layer
        model.add(Embedding(
            input_dim=self.vocab_size, 
            output_dim=self.embedding_dim, 
            input_length=self.max_len
        ))

        # BiLSTM layer
        model.add(Bidirectional(LSTM(64, return_sequences=True)))

        # Add pooling and dropout for regularization
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(self.dropout_rate))

        # Output layer
        model.add(Dense(self.num_classes, activation='softmax'))

        # Compile the model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer, 
            loss='categorical_crossentropy', 
            metrics=['accuracy']
        )
        return model

    def save_features(self, features, file_path):
        """
        Saves the extracted feature vectors to a file.
        :param features: The feature vectors (numpy array).
        :param file_path: Path to save the features (e.g., 'features.npy' or 'features.csv').
        """
        if file_path.endswith('.npy'):
            np.save(file_path, features)
        elif file_path.endswith('.csv'):
            np.savetxt(file_path, features, delimiter=',')
        else:
            raise ValueError("Unsupported file format. Use '.npy' or '.csv'.")
        print(f"Features saved to: {file_path}")

    def extract_features(self, texts, save_path=None):
        """
        Converts pre-tokenized sequences (lists of string IDs) to integer arrays.
        :param texts: Input texts or tokenized sequences.
        :param save_path: Optional path to save the extracted features (e.g., 'features.npy').
        :return: Padded sequences as numpy array.
        """
        sequences = []
        for text in texts:
            if isinstance(text, list):
                # Convert list of string IDs to integers
                seq = [int(token) for token in text if token.isdigit()]
            elif isinstance(text, str):
                # Split string into tokens (if formatted as space-separated IDs)
                seq = [int(token) for token in text.split() if token.isdigit()]
            else:
                seq = []
            sequences.append(seq)

        # Pad sequences
        padded_sequences = pad_sequences(
            sequences, 
            maxlen=self.max_len, 
            padding='post', 
            truncating='post'
        )

        # Save features if a path is provided
        if save_path:
            self.save_features(padded_sequences, save_path)

        return padded_sequences

    def fit(self, X_train, y_train, X_val=None, y_val=None, batch_size=32, **kwargs):
        # Convert text to padded sequences
        X_train = self.extract_features(X_train, save_path='train_features.npy')
        if X_val is not None:
            X_val = self.extract_features(X_val, save_path='val_features.npy')

        # Convert labels to one-hot encoding
        y_train = to_categorical(y_train, num_classes=self.num_classes)
        if y_val is not None:
            y_val = to_categorical(y_val, num_classes=self.num_classes)

        callbacks = kwargs.pop('callbacks', [])
        if X_val is not None and y_val is not None:
            early_stopping = EarlyStopping(
                monitor='val_loss', 
                patience=3, 
                restore_best_weights=True
            )
            callbacks.append(early_stopping)

        # Add the custom callback to save model weights
        save_weights_callback = SaveModelWeightsCallback(self.model_save_dir, self.model_name)
        callbacks.append(save_weights_callback)

        history = self.model.fit(
            X_train,
            y_train,
            epochs=self.num_epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            batch_size=batch_size,
            verbose=1,
            **kwargs
        )
        return history

    # Keep other methods (predict, evaluate, etc.) unchanged
    def predict(self, X, **kwargs):
        """
        Generates class predictions from the BiLSTM model. 
        X should be raw text sequences.
        """
        X = self.extract_features(X, save_path='test_features.npy')  # Convert text to tokenized sequences
        y_pred = self.model.predict(X, **kwargs)
        return y_pred.argmax(axis=-1)  # Return predicted class indices

    def predict_proba(self, X, **kwargs):
        """
        Returns the predicted probabilities for each class.
        """
        X = self.extract_features(X, save_path='test_features.npy')  # Convert text to tokenized sequences
        y_pred = self.model.predict(X, **kwargs)
        return y_pred  # Output softmax probabilities

    def score(self, X, y):
        """
        Computes the accuracy score, assuming y is one-hot or integer indices.
        """
        if len(y.shape) > 1 and y.shape[1] > 1:
            # Convert one-hot to class indices
            y_true = y.argmax(axis=-1)
        else:
            y_true = y
        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred)

    def evaluate(self, X, y):
        """
        Prints and returns accuracy, confusion matrix, and classification report.
        """
        if len(y.shape) > 1 and y.shape[1] > 1:
            y_true = y.argmax(axis=-1)
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
        Saves the trained model weights and architecture to the specified directory (default: weights/bilstm).
        """
        if save_dir is None:
            save_dir = self.model_save_dir
        
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        # Save the entire model (architecture + weights) in Keras format
        self.model.save(os.path.join(save_dir, 'bilstm_model.h5'))
        print(f"Model saved to {save_dir}")
     
    def save_model_weights(self):
        weights_path = os.path.join(self.model_save_dir, "bilstm_weights.h5")
        zip_path = os.path.join(self.model_save_dir, "weights.zip")
        
        self.model.save_weights(weights_path)
        print(f"Model weights saved to: {weights_path}")
        
        shutil.make_archive(weights_path.replace('.h5', ''), 'zip', self.model_save_dir)
        print(f"Model weights zipped and saved to: {zip_path}")

    def __getattr__(self, name):
        """
        Allows calls to underlying model methods except for 'predict' and 'predict_proba'.
        """
        if name not in ['predict', 'predict_proba']:
            return getattr(self.model, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
