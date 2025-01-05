import pandas as pd
import numpy as np
import gc
from tqdm import tqdm
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.utils import to_categorical
from transformers import RobertaTokenizer

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

# Initialize Roberta Tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

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

# Tokenize data for transformer models
def tokenize_statements(df, tokenizer):
    encodings = tokenizer(
        df['Statement'].tolist(), 
        padding=True, 
        truncation=True, 
        max_length=512, 
        return_tensors="np"
    )
    return encodings

# Vectorize data for traditional models
def vectorize_statements(df, vectorizer=None, fit=False):
    if fit:
        vectorizer = CountVectorizer()
        X_vec = vectorizer.fit_transform(df['Statement'])
    else:
        X_vec = vectorizer.transform(df['Statement'])
    
    # Reshape for LSTM (batch_size, timesteps, features)
    X_reshaped = X_vec.toarray().reshape(X_vec.shape[0], 1, X_vec.shape[1])
    return X_reshaped, vectorizer

# Prepare data for traditional models
def prepare_data_traditional(df_train, df_test, df_valid):
    vectorizer = CountVectorizer()
    X_train, vectorizer = vectorize_statements(df_train, vectorizer, fit=True)
    X_test, _ = vectorize_statements(df_test, vectorizer, fit=False)
    X_valid, _ = vectorize_statements(df_valid, vectorizer, fit=False)

    # One-hot encode labels
    y_train_one_hot = to_categorical(df_train['Label_Rank'], num_classes=6)
    y_test_one_hot = to_categorical(df_test['Label_Rank'], num_classes=6)
    y_valid_one_hot = to_categorical(df_valid['Label_Rank'], num_classes=6)

    return X_train, y_train_one_hot, X_test, y_test_one_hot, X_valid, y_valid_one_hot

# Prepare data for transformer models
def prepare_data_transformer(df_train, df_test, df_valid, tokenizer):
    train_encodings = tokenize_statements(df_train, tokenizer)
    test_encodings = tokenize_statements(df_test, tokenizer)
    valid_encodings = tokenize_statements(df_valid, tokenizer)

    X_train = {
        "input_ids": train_encodings["input_ids"],
        "attention_mask": train_encodings["attention_mask"]
    }
    X_test = {
        "input_ids": test_encodings["input_ids"],
        "attention_mask": test_encodings["attention_mask"]
    }
    X_valid = {
        "input_ids": valid_encodings["input_ids"],
        "attention_mask": valid_encodings["attention_mask"]
    }

    # Labels for transformers (not one-hot encoded)
    y_train = df_train['Label_Rank'].values
    y_test = df_test['Label_Rank'].values
    y_valid = df_valid['Label_Rank'].values

    return X_train, y_train, X_test, y_test, X_valid, y_valid

# Main function to execute the full process
def process_data_pipeline(train_file, test_file, valid_file, model_type='transformer'):
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

    if model_type == 'transformer':
        print("Preparing data for transformer model input...")
        return prepare_data_transformer(df_train, df_test, df_valid, tokenizer)
    elif model_type == 'traditional':
        print("Preparing data for traditional model input...")
        return prepare_data_traditional(df_train, df_test, df_valid)
    else:
        raise ValueError("Invalid model_type. Choose 'transformer' or 'traditional'.")
