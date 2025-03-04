from src.download_data import download

# BERT
from src.preprocessing_bert import process_data_pipeline_bert
from src.bert_model import BERTClassifier

def bert():
    print("Starting the pipeline...")

    # Stage 1: Download the Liar Dataset
    print("Step 1: Downloading the dataset...")
    download()
    print("Dataset downloaded successfully.\n")

    # Stage 2: Process the data
    print("Step 2: Preprocessing the data...")
    X_train, y_train, X_test, y_test, X_valid, y_valid = process_data_pipeline_bert(
        'train.tsv', 'test.tsv', 'valid.tsv'
    )
    print("Data preprocessing complete.\n")

    # Stage 3: Gather the model
    print("Step 3: Building the BERT model...")
    classifier = BERTClassifier(num_classes=6, num_epochs=3)
    print("Model built successfully.\n")

    # Stage 4: Train the model
    print("Step 4: Training the model...")
    classifier.fit(X_train, y_train, X_valid, y_valid)
    print("Model training complete.\n")

    # Stage 5: Evaluate the model on the test set
    classifier.evaluate(X_test, y_test)
    acc = classifier.score(X_test, y_test)
    print(f"Final Test Accuracy: {acc}\n")

    # Stage 6: Save the model
    classifier.save_model()

    # Stage 7: Extract the feature vectors for each split
    print("Step 7: Extracting and saving feature vectors...")
    classifier.extract_feature_vectors(X_train, split_name="train")
    classifier.extract_feature_vectors(X_valid, split_name="valid")
    classifier.extract_feature_vectors(X_test,  split_name="test")

if __name__ == '__main__':
    bert()
