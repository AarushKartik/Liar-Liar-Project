# src/train_rnn.py

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from rnn_model import RNNClassifier
import matplotlib.pyplot as plt

def plot_training_history(history):
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Main function to train and evaluate the RNN model
def train(rnn, X_train, X_test, y_train, y_test):
    
    # Train the model
    history = rnn.fit(X_train, y_train, validation_data=(X_test, y_test))

    # Plot training history
    plot_training_history(history)

    # Convert one-hot encoded y_test back to class labels for evaluation
    y_test_labels = np.argmax(y_test, axis=1)
    
    # Evaluate the model
    rnn.evaluate(X_test, y_test_labels)

if __name__ == "__main__":
    main()
