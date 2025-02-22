from src.download_data import download
from src.preprocessing_bilstm import process_data_pipeline_bilstm
from src.bilstm_model import BiLSTMClassifier

def bilstm():
    print("Starting the pipeline...")

    # Step 1: Download the dataset
    print("Step 1: Downloading the dataset...")
    download()
    print("Dataset downloaded successfully.\n")

    # Step 2: Preprocess the data
    print("Step 2: Preprocessing the data...")
    # (Assuming labels are not needed for feature extraction)
    X_train, X_test, X_valid = process_data_pipeline_bilstm(
        'train.tsv', 'test.tsv', 'valid.tsv'
    )
    print("Data preprocessing complete.\n")

    # Step 3: Build the BiLSTM model and load pretrained weights
    print("Step 3: Building the BiLSTM model...")
    classifier = BiLSTMClassifier(vocab_size=10000, embedding_dim=128, num_classes=6)
    weights_path = '/content/drive/MyDrive/weights/bilstm_epoch_5'
    classifier.load_model(weights_path)
    print("Pretrained model loaded successfully.\n")

    # Step 4: Extract and save feature vectors for each split
    print("Step 4: Extracting and saving feature vectors...")
    classifier.extract_feature_vectors(X_train, split_name="train")
    classifier.extract_feature_vectors(X_valid, split_name="valid")
    classifier.extract_feature_vectors(X_test,  split_name="test")

if __name__ == '__main__':
    bilstm()
