from src.download_data import download

# BERT
from src.preprocessing_bert import process_data_pipeline_bert
from src.bert_model import BERTClassifier
from src.train_bert import train_bert

def bert():
    print("Starting the pipeline...")

    # Stage 1: Download the Liar Dataset
    print("Step 1: Downloading the dataset...")
    download()
    print("Dataset downloaded successfully.\n")

    # Stage 2: Process the data
    print("Step 2: Preprocessing the data...")
    X_train, y_train, X_test, y_test = process_data_pipeline_bert('train.tsv', 'test.tsv', 'valid.tsv')
    print("Data preprocessing complete.\n")

    # Stage 3: Gather the model
    print("Step 3: Building the BERT model...")
    classifier = BERTClassifier(num_classes=6, num_epochs=3)  # BERT-based now
    print("Model built successfully.\n")

    # Stage 4: Train the model
    print("Step 4: Training the model...")
    classifier.fit(X_train, y_train, X_val, y_val)
    print("Model training complete.\n")

if __name__ == '__main__':
    
    bert()
