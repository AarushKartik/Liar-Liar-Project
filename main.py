from src.download_data import download
from src.preprocessing import download

def main():
  

  # Downloads the Liar Dataset
  download()

  # Process the data
  X_train, y_train, X_test, y_test = process_data_pipeline('train.tsv', 'test.tsv', 'valid.tsv', truthiness_rank)

  # Train our model

if __name__ == '__main__':
  main()
