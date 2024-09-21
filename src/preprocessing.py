import pandas as pd
import numpy as np
import gc
from tqdm import tqdm
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.utils import to_categorical

# Define truthiness ranking
truthiness_rank = {
    'true': 0,
    'mostly-true': 1,
    'half-true': 2,
    'barely-true': 3,
    'false': 4,
    'pants-fire': 5
}

# Load Spacy Model
text_to_nlp = spacy.load("en_core_web_md")

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

# Compute statement length for each dataframe
def compute_statement_length(df):
    df['statement_length'] = df['Statement'].apply(len)
    return df

# Encode labels based on truthiness rank
def encode_labels(df, truthiness_rank):
    df['Label_Rank'] = df['Label'].map(truthiness_rank)
    return df

# Prepare training and test data
def prepare_data(df_train, df_test, df_valid):
    vectorizer = CountVectorizer()

    # Vectorize statements
    X_train_vec = vectorizer.fit_transform(df_train['Statement'])
    X_test_vec = vectorizer.transform(df_test['Statement'])
    X_valid_vec = vectorizer.transform(df_valid['Statement'])

    # Reshape for LSTM (batch_size, timesteps, features)
    X_train_reshaped = X_train_vec.toarray().reshape(X_train_vec.shape[0], 1, X_train_vec.shape[1])
    X_test_reshaped = X_test_vec.toarray().reshape(X_test_vec.shape[0], 1, X_test_vec.shape[1])
    X_valid_reshaped = X_valid_vec.toarray().reshape(X_valid_vec.shape[0], 1, X_valid_vec.shape[1])

    # One-hot encode labels
    y_train_one_hot = to_categorical(df_train['Label_Rank'], num_classes=6)
    y_test_one_hot = to_categorical(df_test['Label_Rank'], num_classes=6)
    y_valid_one_hot = to_categorical(df_valid['Label_Rank'], num_classes=6)

    return X_train_reshaped, y_train_one_hot, X_test_reshaped, y_test_one_hot, X_valid_reshaped, y_valid_one_hot

# Main function to execute the full process
def process_data_pipeline(train_file, test_file, valid_file, batch_size=100):
    # Load and preprocess data
    print("Loading TSV files...")
    df_train, df_test, df_valid = load_tsv_files(train_file, test_file, valid_file)
    
    print("Renaming columns...")
    df_train = rename_columns(df_train)
    df_test = rename_columns(df_test)
    df_valid = rename_columns(df_valid)
    
    print("Computing statement lengths...")
    df_train = compute_statement_length(df_train)
    df_test = compute_statement_length(df_test)
    df_valid = compute_statement_length(df_valid)

    print("Encoding labels...")
    df_train = encode_labels(df_train, truthiness_rank)
    df_test = encode_labels(df_test, truthiness_rank)
    df_valid = encode_labels(df_valid, truthiness_rank)
    
    print("Preparing data for model input...")
    X_train, y_train, X_test, y_test, X_valid, y_valid = prepare_data(df_train, df_test, df_valid)

    print("Data preparation complete.")
    
    return X_train, y_train, X_test, y_test#, X_valid, y_valid
