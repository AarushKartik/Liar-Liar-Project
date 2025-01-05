from src.download_data import download
import tensorflow as tf

import traceback

# Roberta
from src.preprocessing_roberta import process_data_pipeline_roberta
from src.roberta_model import RoBERTaClassifier

# Enable mixed precision
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_policy(policy)

def roberta():
    print("Starting the pipeline...")

    # Step 1: Download the Liar Dataset
    print("Step 1: Downloading the dataset...")
    download()
    print("Dataset downloaded successfully.\n")

    # Step 2: Preprocess the data
    print("Step 2: Preprocessing the data...")
    X_train, y_train, X_test, y_test, train_encodings, test_encodings = process_data_pipeline_roberta(
        'train.tsv', 'test.tsv', 'valid.tsv'
    )
    print("Data preprocessing complete.\n")

    # Debugging: Check the type of train_encodings and test_encodings
    print(f"Type of train_encodings: {type(train_encodings)}")
    print(f"Type of test_encodings: {type(test_encodings)}")

    # Step 3: Initialize the model
    print("Step 3: Building the RoBERTa model...")
    model = RoBERTaClassifier(num_epochs=5, dropout_rate=0.2, learning_rate=2e-5)
    print("Model built successfully.\n")

    # Step 4: Prepare tokenized data for training
    print("Step 4: Preparing tokenized data...")
    train_data = {
        "input_ids": train_encodings["input_ids"],
        "attention_mask": train_encodings["attention_mask"]
    }
    test_data = {
        "input_ids": test_encodings["input_ids"],
        "attention_mask": test_encodings["attention_mask"]
    }
    
    print("Data preparation complete.\n")
    y_train = tf.convert_to_tensor(y_train, dtype=tf.int32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

    # Step 5: Train the model
    print("Step 5: Training the model...")
    try:
        model.fit(
            x=train_data,
            y=y_train,
            validation_data=(test_data, y_test),
            epochs=5,
            batch_size=2,
        )
        print("Model training complete.\n")
    except Exception as e:
        print("An error occurred during model training:")
        traceback.print_exc()

if __name__ == '__main__':
    roberta()
