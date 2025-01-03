from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout, GlobalAveragePooling1D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
import os

class BiLSTMClassifier:
    def __init__(self, 
                 num_classes=6, 
                 num_epochs=3, 
                 dropout_rate=0.1, 
                 learning_rate=2e-5,
                 max_len=128,  # Max sequence length for padding/truncating input sequences
                 embedding_dim=100,  # Dimensionality of the word embeddings
                 model_save_dir='weights/bilstm'):
        """
        num_classes: Number of classification labels
        num_epochs: How many epochs to train
        dropout_rate: Dropout rate to use in the network
        learning_rate: Learning rate for the Adam optimizer
        max_len: Maximum length for padding/truncating input sequences
        embedding_dim: Embedding dimension for the input
        model_save_dir: Directory to save the trained BiLSTM model weights
        """
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.model_save_dir = model_save_dir

        # Build the BiLSTM-based classification model
        self.model = self.build_model()

    def build_model(self):
        """
        Builds and compiles a BiLSTM-based classification model.
        """
        model = Sequential()

        # Embedding layer
        model.add(Embedding(input_dim=5000, output_dim=self.embedding_dim, input_length=self.max_len))

        # BiLSTM layer
        model.add(Bidirectional(LSTM(64, return_sequences=True)))

        # Add pooling and dropout for regularization
        model.add(GlobalAveragePooling1D())
        model.add(Dropout(self.dropout_rate))

        # Output layer
        model.add(Dense(self.num_classes, activation='softmax'))

        # Set up the optimizer and compile
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer, 
            loss='categorical_crossentropy',  # Use for one-hot labels
            metrics=['accuracy']
        )
        return model

    def fit(self, X_train, y_train, X_val=None, y_val=None, batch_size=32, **kwargs):
        """
        Trains the BiLSTM classifier. 
        
        X_train, y_train: 
            X_train -> Input sequences (padded)
            y_train -> One-hot or integer labels (depending on your setup)

        X_val, y_val: Same format for validation set (optional).
        """
        # Convert y_train and y_val to one-hot encoding if they are integers
        if len(y_train.shape) == 1 or y_train.shape[1] != self.num_classes:
            y_train = to_categorical(y_train, num_classes=self.num_classes)
        if y_val is not None and (len(y_val.shape) == 1 or y_val.shape[1] != self.num_classes):
            y_val = to_categorical(y_val, num_classes=self.num_classes)

        callbacks = kwargs.pop('callbacks', [])
        # Early stopping, similar to your original RNN script
        if X_val is not None and y_val is not None:
            early_stopping = EarlyStopping(
                monitor='val_loss', 
                patience=3, 
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
            history = self.model.fit(
                X_train,
                y_train,
                epochs=self.num_epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                batch_size=batch_size,  # Can be adjusted for BiLSTM models
                verbose=1,
                **kwargs
            )
        else:
            history = self.model.fit(
                X_train,
                y_train,
                epochs=self.num_epochs,
                batch_size=batch_size,
                verbose=1,
                callbacks=callbacks,
                **kwargs
            )
        
        return history

    def predict(self, X, **kwargs):
        """
        Generates class predictions from the BiLSTM model. 
        X should be padded input sequences.
        """
        y_pred = self.model.predict(X, **kwargs)
        return y_pred.argmax(axis=-1)  # Return predicted class indices

    def predict_proba(self, X, **kwargs):
        """
        Returns the predicted probabilities for each class.
        """
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

    def __getattr__(self, name):
        """
        Allows calls to underlying model methods except for 'predict' and 'predict_proba'.
        """
        if name not in ['predict', 'predict_proba']:
            return getattr(self.model, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

