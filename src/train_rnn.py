# src/train_rnn.py

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from rnn_model import RNNClassifier
import matplotlib.pyplot as plt

# Load your data
# Replace with your actual loading mechanism
def load_data():
    df_train = pd.read_csv('train.tsv', sep='\t', header=None)
    df_test = pd.read_csv('test.tsv', sep='\t', header=None)
    
    # Assuming train and test dataframes have the same structure
    df_train.rename(columns={0: 'ID', 1: 'Label', 2: 'Statement'}, inplace=True)
    df_test.rename(columns={0: 'ID', 1: 'Label', 2: 'Statement'}, inplace=True)
    
    return df_train, df_test

def prepare_data(df_train, df_test):
    vectorizer = CountVectorizer()

    # Vectorize statements
    X_train_vec = vectorizer.fit_transform(df_train['Statement'])
    X_test_vec = vectorizer.transform(df_test['Statement'])

    # Reshape for LSTM (batch_size, timesteps, features)
    X_train_reshaped = X_train_vec.toarray().reshape(X_train_vec.shape[0], 1, X_train_vec.shape[1])
    X_test_reshaped = X_test_vec.toarray().reshape(X_test_vec.shape[0], 1, X_test_vec.shape[1])

    # Encode labels to one-hot
    y_train_array = np.array(df_train['Label'])
    y_test_array = np.array(df_test['Label'])
    y_train_one_hot = to_categorical(y_train_array, num_classes=6)
    y_test_one_hot = to_categorical(y_test_array, num_classes=6)

    return X_train_reshaped, X_test_reshaped, y_train_one_hot, y_test_one_hot

def plot_training_history(history):
    plt.plot(history.history['accuracy'], label='train accuracy')
    plt.plot(history.history['val_accuracy'], label='val accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Main function to train and evaluate the RNN model
def main():
    df_train, df_test = load_data()
    X_train, X_test, y_train, y_test = prepare_data(df_train, df_test)
    
    # Instantiate the model
    rnn = RNNClassifier(num_epochs=5, lstm_units=200, dropout_rate=0.2)
    
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
