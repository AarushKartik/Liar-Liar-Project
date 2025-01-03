from src.download_data import download

# Roberta
from src.preprocessing_roberta import process_data_pipeline_roberta
from src.roberta_model import RoBERTaClassifier

def roberta():
    print("Starting the pipeline...")

    # Stage 1: Download the Liar Dataset
    print("Step 1: Downloading the dataset...")
    download()
    print("Dataset downloaded successfully.\n")

    # Stage 2: Process the data
    print("Step 2: Preprocessing the data...")
    X_train, y_train, X_test, y_test, train_encodings, test_encodings = process_data_pipeline_roberta('train.tsv', 'test.tsv', 'valid.tsv')
    print("Data preprocessing complete.\n")

    # Stage 3: Gather the model
    print("Step 3: Building the roBERTa model...")
    model = RoBERTaClassifier(num_epochs=5, lstm_units=200, dropout_rate=0.2)
    print("Model built successfully.\n")

    # Stage 4: Train the model
    print("Step 4: Training the model...")
    train_roberta(model, X_train, X_test, y_train, y_test)
    print("Model training complete.\n")



if __name__ == '__main__':
    
    roberta()
