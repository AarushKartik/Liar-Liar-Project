# src/rnn_model.py  (Now adapted for BERT)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from transformers import TFBertForSequenceClassification, BertConfig
import tensorflow as tf

class BERTClassifier:  # We keep the same class name for structural consistency
    def __init__(self, 
                 num_classes=6, 
                 num_epochs=3, 
                 dropout_rate=0.1, 
                 learning_rate=2e-5):
        """
        num_classes: Number of classification labels
        num_epochs: How many epochs to train
        dropout_rate: Not used directly in TFBertForSequenceClassification 
                      (BERT has its own dropout), but kept for structural consistency
        learning_rate: Learning rate for the Adam optimizer
        """
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate

        # Instead of building a Sequential LSTM model, build a BERT model
        self.model = self.build_model()

    def build_model(self):
        """
        Builds and compiles a TFBertForSequenceClassification model.
        """
        # Load a BERT configuration with the desired number of labels
        config = BertConfig.from_pretrained("bert-base-uncased", num_labels=self.num_classes)
        # Initialize the TFBertForSequenceClassification model
        model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", config=config)

        # Set up the optimizer and compile
        optimizer = Adam(learning_rate=self.learning_rate)
        # Use 'categorical_crossentropy' if you are training with one-hot labels
        model.compile(optimizer=optimizer, 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
        return model

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """
        Trains the BERT classifier. 
        
        X_train, y_train: Typically,
            X_train -> a dict or tuple containing (input_ids, attention_mask) 
            y_train -> one-hot or integer labels (depending on your setup)

        X_val, y_val: Same format for validation set (optional).
        """
        callbacks = kwargs.pop('callbacks', [])
        # Early stopping, similar to the RNN script
        if X_val is not None and y_val is not None:
            early_stopping = EarlyStopping(
                monitor='val_loss', 
                patience=3, 
                restore_best_weights=True
            )
            callbacks.append(early_stopping)
            return self.model.fit(
                X_train,
                y_train,
                epochs=self.num_epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                batch_size=8,  # Typically smaller batch size for BERT
                verbose=1,
                **kwargs
            )
        else:
            return self.model.fit(
                X_train,
                y_train,
                epochs=self.num_epochs,
                batch_size=8,
                verbose=1,
                callbacks=callbacks,
                **kwargs
            )

    def predict(self, X, **kwargs):
        """
        Generates class predictions from the BERT model. 
        X should be a dict or tuple with 'input_ids' and 'attention_mask'.
        """
        # TFBertForSequenceClassification returns a model output object
        outputs = self.model.predict(X, **kwargs)
        logits = outputs.logits  # shape: (batch_size, num_classes)
        return logits.argmax(axis=-1)  # Return the predicted class indices

    def predict_proba(self, X, **kwargs):
        """
        Returns the predicted probabilities for each class.
        """
        outputs = self.model.predict(X, **kwargs)
        logits = outputs.logits
        # Convert logits to probabilities via softmax
        probabilities = tf.nn.softmax(logits, axis=-1).numpy()
        return probabilities

    def score(self, X, y):
        """
        Computes the accuracy score, assuming y is one-hot or integer indices.
        """
        # If y is one-hot, convert to class indices:
        if len(y.shape) > 1 and y.shape[1] > 1:
            y_true = y.argmax(axis=-1)
        else:
            y_true = y
        y_pred = self.predict(X)
        return accuracy_score(y_true, y_pred)

    def evaluate(self, X, y):
        """
        Prints and returns accuracy, confusion matrix, and classification report.
        """
        # Convert y to class indices if it is one-hot
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

    def __getattr__(self, name):
        """
        Allows calls to underlying model methods except for 'predict' and 'predict_proba'.
        """
        if name not in ['predict', 'predict_proba']:
            return getattr(self.model, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
