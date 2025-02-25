from src.download_data import download
from src.preprocessing_roberta import process_data_pipeline_roberta
from src.roberta_model import RoBERTaClassifier

def roberta():
    print("Starting the RoBERTa pipeline...")

    # Step 1: Download the dataset
    print("Step 1: Downloading the dataset...")
    download()
    print("Dataset downloaded successfully.\n")

    # Step 2: Preprocess the data
    print("Step 2: Preprocessing the data...")
    # (Assuming labels are not needed for feature extraction)
    X_train, _, X_test, _, X_valid, _ = process_data_pipeline_roberta(
        'train.tsv', 'test.tsv', 'valid.tsv'
    )
    print("Data preprocessing complete.\n")

    # Step 3: Build the RoBERTa model and load pretrained weights
    print("Step 3: Building the RoBERTa model...")
    classifier = RoBERTaClassifier(num_classes=6, num_epochs=3)
    
    # Load pretrained weights from Google Drive or local path
    classifier.load_model_weights(epoch=1)  # Load weights for epoch 1
    print("Pretrained RoBERTa model loaded successfully.\n")

    # Step 4: Extract and save feature vectors for each split
    print("Step 4: Extracting and saving feature vectors...")
    classifier.extract_feature_vectors(X_train, split_name="train")
    classifier.extract_feature_vectors(X_valid, split_name="valid")
    classifier.extract_feature_vectors(X_test, split_name="test")

if __name__ == '__main__':
    roberta()
