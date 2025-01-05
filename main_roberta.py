from src.download_data import download
import tensorflow as tf
import traceback
import os

# Google Drive integration
from google.colab import drive  # For Google Colab
from pydrive.auth import GoogleAuth  # For general Python environments
from pydrive.drive import GoogleDrive

# Roberta
from src.preprocessing_roberta import process_data_pipeline_roberta
from src.roberta_model import RoBERTaClassifier


class SaveWeightsCallback(tf.keras.callbacks.Callback):
    def __init__(self, save_interval, model, save_dir, drive_folder_name):
        super().__init__()
        self.save_interval = save_interval
        self.model = model
        self.save_dir = save_dir
        self.drive_folder_name = drive_folder_name

    def on_train_batch_end(self, batch, logs=None):
        total_batches = self.params['steps']
        interval_batches = total_batches // self.save_interval
        if batch % interval_batches == 0 or batch == total_batches - 1:
            weight_path = os.path.join(self.save_dir, f"weights_batch_{batch}.h5")
            self.model.save_weights(weight_path)
            print(f"Saved weights at batch {batch} to {weight_path}.")
            upload_to_drive(weight_path, self.drive_folder_name)


def upload_to_drive(local_path, drive_folder_name):
    """
    Uploads a file to a Google Drive folder.

    Args:
        local_path (str): Path to the file to upload.
        drive_folder_name (str): The name of the folder in Google Drive.
    """
    try:
        # Authenticate and create a PyDrive client
        gauth = GoogleAuth()
        gauth.LocalWebserverAuth()  # Creates local webserver for authentication
        drive = GoogleDrive(gauth)

        # Find or create the folder
        folder_list = drive.ListFile({'q': f"title='{drive_folder_name}' and mimeType='application/vnd.google-apps.folder'"}).GetList()
        if not folder_list:
            folder_metadata = {
                "title": drive_folder_name,
                "mimeType": "application/vnd.google-apps.folder",
            }
            folder = drive.CreateFile(folder_metadata)
            folder.Upload()
            folder_id = folder["id"]
        else:
            folder_id = folder_list[0]["id"]

        # Upload the file
        file = drive.CreateFile({"title": os.path.basename(local_path), "parents": [{"id": folder_id}]})
        file.SetContentFile(local_path)
        file.Upload()
        print(f"File uploaded to Google Drive folder '{drive_folder_name}' successfully!")
    except Exception as e:
        print("An error occurred while uploading to Google Drive:")
        traceback.print_exc()


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

    # Step 5: Train the model with the custom callback
    print("Step 5: Training the model...")
    try:
        save_dir = "./saved_weights"
        os.makedirs(save_dir, exist_ok=True)
        
        callback = SaveWeightsCallback(
            save_interval=5,
            model=model,
            save_dir=save_dir,
            drive_folder_name="weights"
        )

        model.fit(
            x=train_data,
            y=y_train,
            validation_data=(test_data, y_test),
            epochs=2,
            batch_size=2,
            callbacks=[callback]
        )
        print("Model training complete.\n")
    except Exception as e:
        print("An error occurred during model training:")
        traceback.print_exc()


if __name__ == '__main__':
    roberta()
