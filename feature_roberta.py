
from src.download_data import download
from src.preprocessing_bert import process_data_pipeline_bert
from src.bert_model import BERTClassifier

def bert():
    print("Starting the pipeline...")

    # Step 1: Download the dataset
    print("Step 1: Downloading the dataset...")
    download()
    print("Dataset downloaded successfully.\n")

    # Step 2: Preprocess the data
    print("Step 2: Preprocessing the data...")
    # (Assuming labels are not needed for feature extraction)
    X_train, _, X_test, _, X_valid, _ = process_data_pipeline_bert(
        'train.tsv', 'test.tsv', 'valid.tsv'
    )
    print("Data preprocessing complete.\n")

    # Step 3: Build the BERT model and load pretrained weights
    print("Step 3: Building the BERT model...")
    classifier = BERTClassifier(num_classes=6, num_epochs=3)
    classifier.load_model('weights/bert_epoch_3.pth')  # New method to load pretrained weights
    print("Pretrained model loaded successfully.\n")

    # Step 4: Extract and save feature vectors for each split
    print("Step 4: Extracting and saving feature vectors...")
    classifier.extract_feature_vectors(X_train, split_name="train")
    classifier.extract_feature_vectors(X_valid, split_name="valid")
    classifier.extract_feature_vectors(X_test,  split_name="test")

if __name__ == '__main__':
    bert()

