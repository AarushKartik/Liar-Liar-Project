from src.download_data import download

# BiLSTM
from src.preprocessing_bilstm import process_data_pipeline_bilstm
from src.bilstm_model import BiLSTMClassifier

def bilstm():
    print("Starting the pipeline...")

    # Stage 1: Download the Liar Dataset
    print("Step 1: Downloading the dataset...")
    download()
    print("Dataset downloaded successfully.\n")

    # Stage 2: Process the data
    # Unpacking
    data, val_data = process_data_pipeline_bilstm('train.tsv', 'test.tsv', 'valid.tsv')

    # Unpack main data
    X_train, y_train, X_test, y_test = data

    # Unpack validation data if needed
    X_valid, y_valid = val_data

    # Stage 3: Gather the model
    print("Step 3: Building the BiLSTM model...")
    model = Bilstm(num_epochs=5, lstm_units=200, dropout_rate=0.2)
    print("Model built successfully.\n")

    # Stage 4: Train the model
    print("Step 4: Training the model...")
    train_bilstm(model, X_train, X_test, y_train, y_test)
    print("Model training complete.\n")

if __name__ == '__main__':
    
    bilstm()
