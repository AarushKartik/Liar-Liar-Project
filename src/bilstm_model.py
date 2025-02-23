import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, GlobalAveragePooling1D, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.preprocessing.text import Tokenizer

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import os
import shutil
import torch  # For loading PyTorch weights

class SaveModelWeightsCallback(Callback):
    def __init__(self, model_save_dir, model_name):
        super(SaveModelWeightsCallback, self).__init__()
        self.model_save_dir = model_save_dir
        self.model_name = model_name

        # Create the directory if it doesn't exist
        os.makedirs(self.model_save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        # Save model weights in .h5 format
        weights_path_h5 = os.path.join(self.model_save_dir, f'{self.model_name}_epoch_{epoch + 1}.h5')
        self.model.save_weights(weights_path_h5, save_format='h5')  # Explicitly specify .h5 format
        print(f"Model weights saved to: {weights_path_h5}")

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
                 vocab_size=20000,  # Vocabulary size for tokenization
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

        # Initialize tokenizer
        self.tokenizer = Tokenizer(num_words=self.vocab_size)

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

    def extract_features(self, texts, split_name="train", save_dir="features"):
        """
        Extracts feature vectors from the trained embedding layer using pre-trained weights
        and saves them for different dataset splits (train, valid, test).
    
        :param texts: List of input text samples.
        :param split_name: Dataset split name ('train', 'valid', 'test').
        :param save_dir: Directory to save extracted feature vectors.
        :return: Extracted feature vectors as a NumPy array.
        """
        if self.model is None:
            raise ValueError("Model must be loaded before extracting features.")
    
        os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
    
        # Convert text to tokenized sequences
        if isinstance(texts[0], str):
            sequences = self.tokenizer.texts_to_sequences(texts)
        else:
            sequences = texts  # Assume already tokenized
    
        # ✅ Debugging: Print sample sequences
        print(f"First 5 sequences before processing: {sequences[:5]}")
    
        # ✅ Ensure all sequences are lists of integers
        processed_sequences = []
        for seq in sequences:
            if seq is None or len(seq) == 0:  
                processed_sequences.append([0])  # Replace empty sequences with [0]
            elif isinstance(seq, int):  
                processed_sequences.append([seq])  # Convert single integer into a list
            elif isinstance(seq, (list, np.ndarray)):  
                processed_sequences.append(list(seq))  # Ensure list format
            else:
                raise ValueError(f"Unexpected sequence format: {seq}")
    
        # ✅ Truncate long sequences
        processed_sequences = [seq[:self.max_len] for seq in processed_sequences]
    
        # ✅ Debugging: Print processed sequences before padding
        print(f"First 5 sequences after processing: {processed_sequences[:5]}")
    
        # ✅ Now pad sequences safely
        padded_sequences = pad_sequences(processed_sequences, maxlen=self.max_len, padding='post', truncating='post')
    
        # Get embedding layer weights
        embedding_layer = self.model.get_layer(index=0)  
        embedding_weights = embedding_layer.get_weights()[0]  
    
        # Extract feature vectors using the trained embeddings
        feature_vectors = np.zeros((len(padded_sequences), self.embedding_dim))
    
        for i, sequence in enumerate(padded_sequences):
            valid_tokens = [embedding_weights[token] for token in sequence if token < self.vocab_size]
            if len(valid_tokens) > 0:
                embedded_vector = np.mean(valid_tokens, axis=0)
                if embedded_vector.shape == (self.embedding_dim,):
                    feature_vectors[i] = embedded_vector
    
        print(f"Extracted feature vectors for {split_name} set: {feature_vectors.shape}")
    
        # Save feature vectors
        np.save(os.path.join(save_dir, f"{split_name}_features.npy"), feature_vectors)
        np.savetxt(os.path.join(save_dir, f"{split_name}_features.txt"), feature_vectors, fmt="%.6f")
    
        return feature_vectors






    def fit(self, X_train, y_train, X_val=None, y_val=None, batch_size=32, **kwargs):
        # Tokenize input data if it's raw text
        if isinstance(X_train[0], str):
            self.tokenizer.fit_on_texts(X_train)
            X_train = self.tokenizer.texts_to_sequences(X_train)
            if X_val is not None:
                X_val = self.tokenizer.texts_to_sequences(X_val)

        # Debug: Print tokenized sequences
        print("Tokenized sequences (X_train):", X_train[:5])

        # Convert text to padded sequences
        X_train = self.extract_features(X_train, save_path=f'{self.model_name}_train_features.npy')
        if X_val is not None:
            X_val = self.extract_features(X_val, save_path=f'{self.model_name}_val_features.npy')

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

    def predict(self, X, **kwargs):
        """
        Generates class predictions from the BiLSTM model. 
        X should be raw text sequences.
        """
        if isinstance(X[0], str):
            X = self.tokenizer.texts_to_sequences(X)
        X = self.extract_features(X, save_path=f'{self.model_name}_test_features.npy')  # Convert text to tokenized sequences
        y_pred = self.model.predict(X, **kwargs)
        return y_pred.argmax(axis=-1)  # Return predicted class indices

    def predict_proba(self, X, **kwargs):
        """
        Returns the predicted probabilities for each class.
        """
        if isinstance(X[0], str):
            X = self.tokenizer.texts_to_sequences(X)
        X = self.extract_features(X, save_path=f'{self.model_name}_test_features.npy')  # Convert text to tokenized sequences
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

    from google.colab import drive  # Import the drive module to mount Google Drive

    def save_model(self, save_dir=None):
        """
        Saves the trained model weights and architecture to the specified directory.
        Default directory: 'weights/weights_extraction' in Google Drive.
        """
        # Mount Google Drive (if not already mounted)
        drive.mount('/content/drive')
    
        if save_dir is None:
            # Default directory in Google Drive
            save_dir = '/content/drive/My Drive/weights/weights_extraction'
        
        # Create the directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save the entire model (architecture + weights) in Keras format
        model_path = os.path.join(save_dir, 'bilstm_model.h5')
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
    
    def save_model_weights(self):
        """
        Saves the trained model weights to a folder called 'weights_extraction'
        inside a larger folder called 'weights' in Google Drive.
        Also creates a zip archive of the weights.
        """
        # Mount Google Drive (if not already mounted)
        drive.mount('/content/drive')
    
        # Define the paths
        weights_dir = '/content/drive/My Drive/weights/weights_extraction'
        weights_path = os.path.join(weights_dir, 'bilstm.h5')
        zip_path = os.path.join(weights_dir, 'weights.zip')
        
        # Create the directory if it doesn't exist
        os.makedirs(weights_dir, exist_ok=True)
        
        # Save the model weights
        self.model.save_weights(weights_path)
        print(f"Model weights saved to: {weights_path}")
        
        # Create a zip archive of the weights
        shutil.make_archive(weights_path.replace('.h5', ''), 'zip', weights_dir)
        print(f"Model weights zipped and saved to: {zip_path}")

    def load_model(self, path=None):
        """
        Loads pretrained weights from the specified path.
        - Only supports .h5 files.
        - If no path is provided, uses the default model_save_dir.
        """
        if path is None:
            path = self.model_save_dir
    
        # Check if the path exists
        if not os.path.exists(path):
            raise ValueError(f"The specified path does not exist: {path}")
    
        # Ensure the file is a .h5 file
        if not path.endswith('.h5'):
            raise ValueError(f"Unsupported file format: {path}. Only '.h5' files are supported.")
    
        # Load the weights directly
        self.model.load_weights(path)
        print(f"Model weights loaded from: {path}")
    def __getattr__(self, name):
        """
        Allows calls to underlying model methods except for 'predict' and 'predict_proba'.
        """
        if name not in ['predict', 'predict_proba']:
            return getattr(self.model, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
