import pandas as pd
import numpy as np
import gc
from tqdm import tqdm
import spacy

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
def prepare_data(df_train, df_test):
    X_train = df_train['Statement']
    y_train = df_train['Label_Rank']
    X_test = df_test['Statement']
    y_test = df_test['Label_Rank']
    
    return X_train, y_train, X_test, y_test

# Combine training and testing data for processing
def consolidate_data(X_train, X_test, y_train, y_test):
    X_text = pd.concat([X_train, X_test]).tolist()
    y = pd.concat([y_train, y_test]).tolist()
    return X_text, y

# Get the maximum length of tokens in the dataset
def get_global_max_length(X_text):
    max_length = 0
    for doc in text_to_nlp.pipe(X_text):
        max_length = max(max_length, len(doc))
    return max_length

# Standardize embeddings to a global length
def standardize_length_global(embeddings, max_length):
    embedding_dim = len(embeddings[0][0]) if embeddings[0] else 0
    padded_embeddings = [[np.zeros(embedding_dim)] * (max_length - len(tokens)) + tokens for tokens in embeddings]
    return padded_embeddings

# Tokenize and embed text in batches
def tokenize_and_embed_batch(text_batch):
    docs = list(text_to_nlp.pipe(text_batch))
    embeddings = [[token.vector for token in doc] for doc in docs]
    return embeddings

# Process text in batches to avoid running out of memory
def batch_process(X_text, batch_size=100):
    max_length = get_global_max_length(X_text)  
    embedding_dim = len(text_to_nlp(X_text[0])[0].vector)  

    X_mmap = np.memmap('embeddings.dat', dtype='float32', mode='w+', shape=(len(X_text), max_length, embedding_dim))

    for i in tqdm(range(0, len(X_text), batch_size)):
        text_batch = X_text[i:i + batch_size]
        embeddings = tokenize_and_embed_batch(text_batch)
        padded_embeddings = standardize_length_global(embeddings, max_length)
        X_batch = np.array(padded_embeddings)
        X_mmap[i:i + batch_size] = X_batch

        del embeddings, X_batch  
        gc.collect()  

    return X_mmap

# Main function to execute the full process
def process_data_pipeline(train_file, test_file, valid_file, batch_size=100):
    df_train, df_test, df_valid = load_tsv_files(train_file, test_file, valid_file)
    
    df_train = rename_columns(df_train)
    df_test = rename_columns(df_test)
    df_valid = rename_columns(df_valid)
    
    df_train = compute_statement_length(df_train)
    df_test = compute_statement_length(df_test)
    df_valid = compute_statement_length(df_valid)

    df_train = encode_labels(df_train, truthiness_rank)
    df_test = encode_labels(df_test, truthiness_rank)
    
    X_train, y_train, X_test, y_test = prepare_data(df_train, df_test)
    
    # X_text, y = consolidate_data(X_train, X_test, df_train['Label_Rank'], df_test['Label_Rank'])
    
    # X_processed = batch_process(X_text, batch_size=batch_size)
    
    # print(f"The shape of the processed dataset is: {X_processed.shape}")
    
    return X_train, y_train, X_test, y_test

