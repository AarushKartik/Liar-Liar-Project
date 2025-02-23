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

    def save_features(self, features, file_path):
        """
        Saves the extracted feature vectors to a file.
        :param features: The feature vectors (numpy array).
        :param file_path: Path to save the features (e.g., 'features.npy' or 'features.txt').
        """
        if file_path.endswith('.npy'):
            np.save(file_path, features)
        elif file_path.endswith('.txt'):
            with open(file_path, 'w') as f:
                np.savetxt(f, features, fmt='%.6f', delimiter=' ')
        else:
            raise ValueError("Unsupported file format. Use '.npy' or '.txt'.")
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

        # Debug: Print the first few sequences
        print("First few sequences:", sequences[:5])

        # Pad sequences
        padded_sequences = pad_sequences(
            sequences, 
            maxlen=self.max_len, 
            padding='post', 
            truncating='post'
        )

        # Debug: Print the shape and first few padded sequences
        print("Padded sequences shape:", padded_sequences.shape)
        print("First few padded sequences:", padded_sequences[:5])

        # Save features if a path is provided
        if save_path:
            # Save in .npy format
            self.save_features(padded_sequences, save_path)
            # Save in .txt format
            txt_path = save_path.replace('.npy', '.txt')
            self.save_features(padded_sequences, txt_path)

        return padded_sequences

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

    def load_model(self, path=None):
        """
        Loads pretrained weights from the specified path.
        - If the path is a .pth file, it converts PyTorch weights to TensorFlow format.
        - If the path is a .h5 file, it loads the weights directly.
        If no path is provided, uses the default model_save_dir.
        """
        if path is None:
            path = self.model_save_dir

        # Check if the path exists
        if not os.path.exists(path):
            raise ValueError(f"The specified path does not exist: {path}")

        # If the path is a .pth file, convert and load PyTorch weights
        if path.endswith('.pth'):
            # Load PyTorch weights
            pytorch_state_dict = torch.load(path, map_location='cpu')

            # Map PyTorch weights to TensorFlow/Keras layers
            for layer in self.model.layers:
                if isinstance(layer, tf.keras.layers.LSTM):
                    # Extract PyTorch LSTM weights
                    weight_ih = pytorch_state_dict['lstm.weight_ih_l0'].numpy()
                    weight_hh = pytorch_state_dict['lstm.weight_hh_l0'].numpy()
                    bias_ih = pytorch_state_dict['lstm.bias_ih_l0'].numpy()
                    bias_hh = pytorch_state_dict['lstm.bias_hh_l0'].numpy()

                    # Combine weights for TensorFlow
                    kernel = np.concatenate([weight_ih, weight_hh], axis=1).T
                    recurrent_kernel = np.zeros_like(kernel)  # Placeholder for recurrent weights
                    bias = np.concatenate([bias_ih, bias_hh])

                    # Set TensorFlow weights
                    layer.set_weights([kernel, recurrent_kernel, bias])

                elif isinstance(layer, tf.keras.layers.Dense):
                    # Extract PyTorch dense layer weights
                    weight = pytorch_state_dict['dense.weight'].numpy().T
                    bias = pytorch_state_dict['dense.bias'].numpy()

                    # Set TensorFlow weights
                    layer.set_weights([weight, bias])

            print("PyTorch weights successfully converted and loaded into TensorFlow model.")

        # If the path is a .h5 file, load the weights directly
        elif path.endswith('.h5'):
            self.model.load_weights(path)
            print(f"Model weights loaded from: {path}")

        else:
            raise ValueError(f"Unsupported file format: {path}. Use '.pth' or '.h5'.")

    def __getattr__(self, name):
        """
        Allows calls to underlying model methods except for 'predict' and 'predict_proba'.
        """
        if name not in ['predict', 'predict_proba']:
            return getattr(self.model, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
