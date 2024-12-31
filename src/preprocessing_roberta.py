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
def process_data_pipeline_roberta(train_file, test_file, valid_file, batch_size=100, max_length=256):
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
    
    print("Preparing data for RoBERTa model input...")
    
    # Initialize the RoBERTa tokenizer
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    
    # Prepare the data
    train_encodings, y_train, test_encodings, y_test, valid_encodings, y_valid = prepare_data_for_roberta(
        df_train, df_test, df_valid, tokenizer, max_length
    )

    print("Data preparation complete.")
    
    return train_encodings, y_train, test_encodings, y_test, valid_encodings, y_valid

