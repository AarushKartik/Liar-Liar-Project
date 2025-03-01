# src/preprocessing_bert.py (Now adapted for PyTorch)

import pandas as pd
import numpy as np
from tqdm import tqdm
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

# Load TSV files and return processed dataframes
def load_tsv_files(train_file, test_file, valid_file):
    df_train = pd.read_csv(train_file, sep='\t', header=None)
    df_test = pd.read_csv(test_file, sep='\t', header=None)
    df_valid = pd.read_csv(valid_file, sep='\t', header=None)

    df_train.to_csv("saved_train.csv", index=False)
    df_test.to_csv("saved_test.csv", index=False)
    df_valid.to_csv("saved_valid.csv", index=False)
    
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

# Encode labels based on truthiness rank
def encode_labels(df, truthiness_rank):
    df['Label_Rank'] = df['Label'].map(truthiness_rank)
    return df

# Prepare data for BERT classification
def prepare_data(df_train, df_test, df_valid, max_length=128):
    """
    Prepares data for BERT classification:
    - Tokenizes statements
    - Pads/truncates to `max_length`
    - Returns `input_ids`, `attention_mask`, and labels
    """

    # Load BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def tokenize_data(df):
        encodings = tokenizer(
            list(df['Statement']),
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='np'  # or 'pt' if you prefer direct PyTorch. Using 'np' is still okay.
        )
        return encodings['input_ids'], encodings['attention_mask']

    # Tokenize train, test, and valid datasets
    X_train_input_ids, X_train_attention_mask = tokenize_data(df_train)
    X_test_input_ids, X_test_attention_mask = tokenize_data(df_test)
    X_valid_input_ids, X_valid_attention_mask = tokenize_data(df_valid)

    # Convert labels to one-hot encoding (if needed). Otherwise, you can keep them as integer.
    # For consistency with your original code, we keep one-hot. PyTorch model will handle them.
    num_classes = 6
    y_train = np.eye(num_classes)[df_train['Label_Rank'].values]
    y_test = np.eye(num_classes)[df_test['Label_Rank'].values]
    y_valid = np.eye(num_classes)[df_valid['Label_Rank'].values]

    return (
        (X_train_input_ids, X_train_attention_mask), y_train,
        (X_test_input_ids, X_test_attention_mask), y_test,
        (X_valid_input_ids, X_valid_attention_mask), y_valid
    )

# Main function to execute the full process
def process_data_pipeline_bert(train_file, test_file, valid_file, max_length=128):
    """
    Executes the full preprocessing pipeline for BERT classification.
    """
    print("Loading TSV files...")
    df_train, df_test, df_valid = load_tsv_files(train_file, test_file, valid_file)
    
    print("Renaming columns...")
    df_train = rename_columns(df_train)
    df_test = rename_columns(df_test)
    df_valid = rename_columns(df_valid)

    print("Encoding labels...")
    df_train = encode_labels(df_train, truthiness_rank)
    df_test = encode_labels(df_test, truthiness_rank)
    df_valid = encode_labels(df_valid, truthiness_rank)
    
    print("Preparing data for BERT classification...")
    X_train, y_train, X_test, y_test, X_valid, y_valid = prepare_data(
        df_train, df_test, df_valid, max_length=max_length
    )

    print("Data preparation complete.")
    return X_train, y_train, X_test, y_test, X_valid, y_valid
