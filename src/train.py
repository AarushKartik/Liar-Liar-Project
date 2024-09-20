import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Example data
# Replace these with your actual data
data = pd.DataFrame({
    'text': ['sample text 1', 'sample text 2', 'sample text 3'],  # Replace with your texts
    'label': [0, 1, 2]  # Replace with your labels
})

# Split the dataset into training and testing sets
X = data['text']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text to numerical features using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Reshape the data for LSTM (batch_size, timesteps, features)
X_train_reshaped = X_train_vec.toarray().reshape(X_train_vec.shape[0], 1, X_train_vec.shape[1])
X_test_reshaped = X_test_vec.toarray().reshape(X_test_vec.shape[0], 1, X_test_vec.shape[1])

# Convert y_train and y_test to NumPy arrays
y_train_array = np.array(y_train)
y_test_array = np.array(y_test)

# One-hot encode the labels
num_classes = len(np.unique(y_train_array))
y_train_one_hot = to_categorical(y_train_array, num_classes=num_classes)
y_test_one_hot = to_categorical(y_test_array, num_classes=num_classes)

# Define the RNN model
def create_rnn_model(num_classes):
    model = Sequential()
    model.add(LSTM(200, input_shape=(1, X_train_vec.shape[1]), return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Train the model
rnn = create_rnn_model(num_classes)
rnn.fit(X_train_reshaped, y_train_one_hot, validation_data=(X_test_reshaped, y_test_one_hot), epochs=5)

# Convert one-hot encoded y_test_one_hot back to class labels
y_test_labels = np.argmax(y_test_one_hot, axis=1)

# Evaluate the model
loss, accuracy = rnn.evaluate(X_test_reshaped, y_test_one_hot)
print(f'Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}')


