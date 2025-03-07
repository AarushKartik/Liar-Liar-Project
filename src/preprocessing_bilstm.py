import pandas as pd
import numpy as np
import gc
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# Define truthiness ranking
truthiness_rank = {
    'true': 0,
    'mostly-true': 1,
    'half-true': 2,
    'barely-true': 3,
    'false': 4,
    'pants-fire': 5
}

# Load TSV files and return processed dataframes
def load_tsv_files(train_file, test_file, valid_file):
    df_train = pd.read_csv(train_file, sep='\t', header=None)
    df_test = pd.read_csv(test_file, sep='\t', header=None)
    df_valid = pd.read_csv(valid_file, sep='\t', header=None)
    
    return df_train, df_test, df_valid

# Rename columns for consistency
def rename_columns(df):
    columns_mapping = {0: 'ID', 1: 'Label', 2: 'Statement', 3: 'Subject', 4: 'Speaker', 5: 'Job Title', 6: 'State Info', 7: 'Party Affiliation'}
    df.rename(columns=columns_mapping, inplace=True)
    return df

def encode_labels(df, truthiness_rank):
    """
    Encode the truthfulness labels into numeric values and return the DataFrame.
    Also, return the number of unique classes.
    """
    df['Label_Rank'] = df['Label'].map(truthiness_rank)
    
    # Get the number of unique classes
    num_classes = len(truthiness_rank)
    
    return df, num_classes



# Prepare training and test data for BiLSTM
def prepare_data_for_bilstm(df_train, df_test, df_valid, tokenizer, max_length=256):
    # Tokenize the statements using Keras Tokenizer
    def tokenize_statements(df):
        return tokenizer.texts_to_sequences(df['Statement'].tolist())

    # Tokenize the datasets
    train_sequences = tokenize_statements(df_train)
    test_sequences = tokenize_statements(df_test)
    valid_sequences = tokenize_statements(df_valid)

    # Pad sequences to the same length
    X_train = pad_sequences(train_sequences, padding='post', maxlen=max_length)
    X_test = pad_sequences(test_sequences, padding='post', maxlen=max_length)
    X_valid = pad_sequences(valid_sequences, padding='post', maxlen=max_length)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train['Label_Rank'])
    y_test = label_encoder.transform(df_test['Label_Rank'])
    y_valid = label_encoder.transform(df_valid['Label_Rank'])

    return X_train, y_train, X_test, y_test, X_valid, y_valid

# Main function to execute the full process for BiLSTM
def process_data_pipeline_bilstm(train_file, test_file, valid_file, batch_size=100, max_length=256, vocab_size=5000):
    # Load and preprocess data
    print("Loading TSV files...")
    df_train, df_test, df_valid = load_tsv_files(train_file, test_file, valid_file)
    
    print("Renaming columns...")
    df_train = rename_columns(df_train)
    df_test = rename_columns(df_test)
    df_valid = rename_columns(df_valid)
    

    print("Encoding labels...")
    df_train, num_classes = encode_labels(df_train, truthiness_rank)
    df_test = encode_labels(df_test, truthiness_rank)[0]  # Only need the DataFrame, not num_classes
    df_valid = encode_labels(df_valid, truthiness_rank)[0]  # Only need the DataFrame, not num_classes
    
    print("Preparing data for BiLSTM model input...")
    
    # Initialize the tokenizer for BiLSTM model (word-level tokenization)
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(df_train['Statement'].tolist())
    
    # Prepare the data
    X_train, y_train, X_test, y_test, X_valid, y_valid = prepare_data_for_bilstm(
        df_train, df_test, df_valid, tokenizer, max_length
    )

    print("Data preparation complete.")
    
    # Return the primary data (X_train, y_train, X_test, y_test)
    # Validation data is returned as a separate tuple for optional use
    return (X_train, y_train, X_test, y_test), (X_valid, y_valid), num_classes

