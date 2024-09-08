import os
import requests
import zipfile

# Define the dataset URL and file names
dataset_url = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
dataset_file = "liar_dataset.zip"

def download_file(url, filename):
    """Download a file from a URL to a local file."""
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Check for HTTP errors

    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def unzip_file(filename, extract_to='.'):
    """Unzip a file to a specified directory."""
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def main():
    # Check if the dataset file already exists
    if not os.path.exists(dataset_file):
        print(f"Downloading {dataset_file}...")
        download_file(dataset_url, dataset_file)
        print(f"Downloaded {dataset_file}. Now unzipping...")
        unzip_file(dataset_file)
        print("Unzipping complete.")
    else:
        print(f"{dataset_file} already exists. Skipping download.")

if __name__ == "__main__":
    main()
# Liar-Liar-Project
