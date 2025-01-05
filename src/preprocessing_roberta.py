import pandas as pd
import numpy as np
import gc
from tqdm import tqdm
import spacy
from transformers import RobertaTokenizer
from sklearn.preprocessing import LabelEncoder

# Define truthiness ranking
truthiness_rank = {
    'true': 0,
    'mostly-true': 1,
    'half-true': 2,
    'barely-true': 3,
    'false': 4,
    'pants-fire': 5
}

# Load Spacy Model (not used for RoBERTa, so optional)
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

# Prepare training and test data for RoBERTa
def prepare_data_for_roberta(df_train, df_test, df_valid, tokenizer, max_length=256):
    # Tokenize the statements using RoBERTa tokenizer
    def tokenize_statements(df):
        return tokenizer(
            df['Statement'].tolist(), 
            padding=True, 
            truncation=True, 
            max_length=max_length, 
            return_tensors='pt'
        )
    
    # Tokenize the datasets
    train_encodings = tokenize_statements(df_train)
    test_encodings = tokenize_statements(df_test)
    valid_encodings = tokenize_statements(df_valid)

    # Encode labels
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(df_train['Label_Rank'])
    y_test = label_encoder.transform(df_test['Label_Rank'])
    y_valid = label_encoder.transform(df_valid['Label_Rank'])

    return train_encodings, y_train, test_encodings, y_test, valid_encodings, y_valid

# Main function to execute the full process
def process_data_pipeline_roberta(train_file, test_file, valid_file):
    # Load and preprocess data
    df_train, df_test, df_valid = load_tsv_files(train_file, test_file, valid_file)
    df_train = rename_columns(df_train)
    df_test = rename_columns(df_test)
    df_valid = rename_columns(df_valid)

    # Ensure raw text is preserved
    df_train['Statement'] = df_train['Statement'].astype(str)
    df_test['Statement'] = df_test['Statement'].astype(str)
    df_valid['Statement'] = df_valid['Statement'].astype(str)

    # Extract raw text for tokenization in main script
    X_train = df_train['Statement'].tolist()
    X_test = df_test['Statement'].tolist()
    y_train = df_train['Label_Rank'].tolist()
    y_test = df_test['Label_Rank'].tolist()

    # Tokenize data
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    train_encodings = tokenizer(X_train, padding=True, truncation=True, return_tensors="tf", max_length=512)
    test_encodings = tokenizer(X_test, padding=True, truncation=True, return_tensors="tf", max_length=512)

    return X_train, y_train, X_test, y_test, train_encodings, test_encodings
