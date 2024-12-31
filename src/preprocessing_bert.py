import pandas as pd
import numpy as np
import gc
from tqdm import tqdm
import spacy
from sklearn.feature_extraction.text import CountVectorizer  # No longer used, but left for structural consistency
from tensorflow.keras.utils import to_categorical
from transformers import BertTokenizer

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
text_to_nlp = spacy.load("en_core_web_md")  # Not strictly necessary for BERT, but kept to preserve structure

# Load TSV files and return processed dataframes
def load_tsv_files(train_file, test_file, valid_file):
    df_train = pd.read_csv(train_file, sep='\t', header=None)
    df_test = pd.read_csv(test_file, sep='\t', header=None)
    df_valid = pd.read_csv(valid_file, sep='\t', header=None)
    
    return df_train, df_test, df_valid

# Rename columns for consistency
def rename_columns(df):
    columns_mapping = {
        0: 'ID', 
        1: 'Label', 
        2: 'Statement', 
        3: 'Subject', 
        4: 'Speaker', 
        5: 'Job Title', 
        6: 'State Info', 
        7: 'Party Affiliation'
    }
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

# Prepare data for BERT model
def prepare_data(df_train, df_test, df_valid, max_length=128):
    """
    Instead of using CountVectorizer (as done for RNN/LSTM),
    we will use BertTokenizer to convert text into input_ids,
    attention_masks, etc.
    """

    # Load a BERT tokenizer (you can choose another model if you prefer)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the statements
    train_encodings = tokenizer(
        list(df_train['Statement']),
        truncation=True,
        padding='max_length',
        max_length=max_length
    )
    test_encodings = tokenizer(
        list(df_test['Statement']),
        truncation=True,
        padding='max_length',
        max_length=max_length
    )
    valid_encodings = tokenizer(
        list(df_valid['Statement']),
        truncation=True,
        padding='max_length',
        max_length=max_length
    )

    # Convert to numpy arrays (or keep them as lists if your framework accepts lists)
    X_train_input_ids = np.array(train_encodings['input_ids'])
    X_train_attention_mask = np.array(train_encodings['attention_mask'])
    
    X_test_input_ids = np.array(test_encodings['input_ids'])
    X_test_attention_mask = np.array(test_encodings['attention_mask'])

    X_valid_input_ids = np.array(valid_encodings['input_ids'])
    X_valid_attention_mask = np.array(valid_encodings['attention_mask'])

    # If you also want token_type_ids (for next sentence tasks or specific BERT variants)
    # X_train_token_type_ids = np.array(train_encodings['token_type_ids'])
    # X_test_token_type_ids = np.array(test_encodings['token_type_ids'])
    # X_valid_token_type_ids = np.array(valid_encodings['token_type_ids'])

    # Convert labels to one-hot if you want a one-hot vector for your classification layer
    y_train_one_hot = to_categorical(df_train['Label_Rank'], num_classes=6)
    y_test_one_hot = to_categorical(df_test['Label_Rank'], num_classes=6)
    y_valid_one_hot = to_categorical(df_valid['Label_Rank'], num_classes=6)

    # Return BERT-specific inputs (input_ids, attention_mask, labels).
    # You can bundle them as needed; below is just one example.
    return (
        (X_train_input_ids, X_train_attention_mask), y_train_one_hot,
        (X_test_input_ids, X_test_attention_mask), y_test_one_hot,
        (X_valid_input_ids, X_valid_attention_mask), y_valid_one_hot
    )

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
    
    print("Preparing data for model input (BERT)...")
    X_train, y_train, X_test, y_test, X_valid, y_valid = prepare_data(df_train, df_test, df_valid)

    print("Data preparation complete.")
    
    return X_train, y_train, X_test, y_test, X_valid, y_valid
