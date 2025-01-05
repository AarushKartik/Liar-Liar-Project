from src.download_data import download

# Roberta
from src.preprocessing_roberta import process_data_pipeline_roberta
from src.roberta_model import RoBERTaClassifier

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
    model = RoBERTaClassifier(num_epochs=5, lstm_units=200, dropout_rate=0.2)
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

    # Step 5: Train the model
    print("Step 5: Training the model...")
    model.fit(train_data, y_train, X_val=test_data, y_val=y_test)
    print("Model training complete.\n")

if __name__ == '__main__':
    roberta()
