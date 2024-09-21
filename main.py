from src.download_data import download
from src.preprocessing import process_data_pipeline
from src.rnn_model import RNNClassifier
from src.train_rnn import train

def main():
  

  # Downloads the Liar Dataset
  download()

  # Process the data
  X_train, y_train, X_test, y_test = process_data_pipeline('train.tsv', 'test.tsv', 'valid.tsv', truthiness_rank)

  # Gather the model
  rnn = RNNClassifier(num_epochs=5, lstm_units=200, dropout_rate=0.2)

  # Train our model
  train(rnn, X_train, X_test, y_train, y_test)

if __name__ == '__main__':
  main()
