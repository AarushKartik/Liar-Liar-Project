from src.download_data import download

# Roberta
from src.preprocessing_roberta import process_data_pipeline_roberta
from src.roberta_model import RoBERTaClassifier
from transformers import RobertaTokenizer

def roberta():
    print("Starting the pipeline...")

    # Stage 1: Download the Liar Dataset
    print("Step 1: Downloading the dataset...")
    download()
    print("Dataset downloaded successfully.\n")

    # Stage 2: Process the data
    print("Step 2: Preprocessing the data...")
    X_train, y_train, X_test, y_test, train_encodings, test_encodings = process_data_pipeline_roberta(
        'train.tsv', 'test.tsv', 'valid.tsv'
    )
    print("Data preprocessing complete.\n")

    # Debugging: Check the type and size of X_train and y_train
    print(f"Number of samples in X_train: {len(X_train)}")
    print(f"Number of samples in y_train: {len(y_train)}")

    # Align the sizes of X_train and y_train
    if len(X_train) != len(y_train):
        min_samples = min(len(X_train), len(y_train))
        X_train = X_train[:min_samples]
        y_train = y_train[:min_samples]

    print(f"Aligned number of samples in X_train: {len(X_train)}")
    print(f"Aligned number of samples in y_train: {len(y_train)}\n")

    # Stage 3: Initialize the model
    print("Step 3: Building the RoBERTa model...")
    model = RoBERTaClassifier(num_epochs=5, lstm_units=200, dropout_rate=0.2)
    tokenizer = model.tokenizer  # Use the tokenizer from the model
    print("Model built successfully.\n")

    # Stage 4: Tokenize and prepare the data for training
    print("Step 4: Tokenizing the data...")
    train_encodings = tokenizer(X_train, padding=True, truncation=True, return_tensors="tf", max_length=512)
    test_encodings = tokenizer(X_test, padding=True, truncation=True, return_tensors="tf", max_length=512)

    # Convert BatchEncoding to Keras-compatible dictionaries
    train_data = {
        "input_ids": train_encodings["input_ids"],
        "attention_mask": train_encodings["attention_mask"]
    }
    test_data = {
        "input_ids": test_encodings["input_ids"],
        "attention_mask": test_encodings["attention_mask"]
    }

    # Debugging: Check sizes after tokenization
    print(f"Number of samples in train_data['input_ids']: {train_data['input_ids'].shape[0]}")
    print(f"Number of samples in y_train: {len(y_train)}")
    print("Data tokenization complete.\n")

    # Stage 5: Train the model
    print("Step 5: Training the model...")
    model.fit(train_data, y_train, X_val=test_data, y_val=y_test)
    print("Model training complete.\n")

if __name__ == '__main__':
    roberta()
