import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

class RNNClassifier:
    def __init__(self, num_classes=6, num_epochs=30, lstm_units=50, dropout_rate=0.7):
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = self.build_model()

    def build_model(self):
        model = Sequential()
        model.add(LSTM(self.lstm_units, return_sequences=True, input_shape=(None, 1)))  # input_shape added
        model.add(Dropout(self.dropout_rate))
        model.add(LSTM(self.lstm_units))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.num_classes, activation='softmax'))  # Softmax for multi-class classification
        optimizer = Adam(learning_rate=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model

    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        if X_train is None or y_train is None:
            print("Arguments are none. Retry with correct arguments.")
            return None

        callbacks = kwargs.pop('callbacks', [])
        if X_val is not None and y_val is not None:
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            callbacks.append(early_stopping)
            return self.model.fit(X_train, y_train, epochs=self.num_epochs, validation_data=(X_val, y_val), callbacks=callbacks, batch_size=32, verbose=1, **kwargs)
        else:
            return self.model.fit(X_train, y_train, epochs=self.num_epochs, batch_size=32, verbose=1, callbacks=callbacks, **kwargs)

    def predict(self, *args, **kwargs):
        predictions = self.model.predict(*args, **kwargs)
        return predictions.argmax(axis=-1)

    def predict_proba(self, *args, **kwargs):
        return self.model.predict(*args, **kwargs)

    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        accuracy = accuracy_score(y, predictions)
        confusion = confusion_matrix(y, predictions)
        classification_rep = classification_report(y, predictions)

        print(f"Accuracy: {accuracy}")
        print("Confusion Matrix:")
        print(confusion)
        print("Classification Report:")
        print(classification_rep)

        return accuracy, confusion, classification_rep

    def __getattr__(self, name):
        if name != 'predict' and name != 'predict_proba':
            return getattr(self.model, name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")



