import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
import gc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load the TSV files
df_train = pd.read_csv('train.tsv', sep='\t', header=None)
df_test = pd.read_csv('test.tsv', sep='\t', header=None)
df_valid = pd.read_csv('valid.tsv', sep='\t', header=None)

# Rename columns
df_train.rename(columns={0: 'ID', 1: 'Label', 2: 'Statement', 3: 'Subject', 4: 'Speaker', 5: 'Job Title', 6: 'State Info', 7: 'Party Affiliation'}, inplace=True)
df_test.rename(columns={0: 'ID', 1: 'Label', 2: 'Statement', 3: 'Subject', 4: 'Speaker', 5: 'Job Title', 6: 'State Info', 7: 'Party Affiliation'}, inplace=True)
df_valid.rename(columns={0: 'ID', 1: 'Label', 2: 'Statement', 3: 'Subject', 4: 'Speaker', 5: 'Job Title', 6: 'State Info', 7: 'Party Affiliation'}, inplace=True)

# Compute statement lengths
df_train['statement_length'] = df_train['Statement'].apply(len)
df_test['statement_length'] = df_test['Statement'].apply(len)
df_valid['statement_length'] = df_valid['Statement'].apply(len)

# Define the ranking based on truthiness (0 is the most true, 5 is the least true)
truthiness_rank = {
    'true': 0,
    'mostly-true': 1,
    'half-true': 2,
    'barely-true': 3,
    'false': 4,
    'pants-fire': 5
}

# Convert the labels in the train and test datasets to their corresponding ranks
df_train['Label_Rank'] = df_train['Label'].map(truthiness_rank)
df_test['Label_Rank'] = df_test['Label'].map(truthiness_rank)

# Extract the new ranked labels
y_train_encoded = df_train['Label_Rank']
y_test_encoded = df_test['Label_Rank']

# Consolidate text data
X_train = df_train['Statement']
X_test = df_test['Statement']
X_text = pd.concat([X_train, X_test])
y = pd.concat([y_train_encoded, y_test_encoded])

# Convert the pandas Series to a list
X_text = X_text.tolist()
y = y.tolist()

# Initialize SpaCy for tokenization and embeddings
text_to_nlp = spacy.load('en_core_web_md')

def get_global_max_length(X_text):
    """
    Calculate the maximum length of tokens across all documents in the dataset.
    Args:
        X_text (list): A list of text strings to be processed.
    Returns:
        int: The maximum number of tokens in any document.
    """
    max_length = 0
    for doc in text_to_nlp.pipe(X_text):
        max_length = max(max_length, len(doc))
    return max_length

def standardize_length_global(embeddings, max_length):
    """
    Ensures all embedding lists are the same length by padding shorter ones with zero vectors.
    Args:
        embeddings (list): A list of lists of embeddings.
        max_length (int): The global maximum length to pad to.
    Returns:
        list: A list of lists with padded embeddings to ensure uniform length.
    """
    embedding_dim = len(embeddings[0][0]) if embeddings[0] else 0
    padded_embeddings = [[np.zeros(embedding_dim)] * (max_length - len(tokens)) + tokens for tokens in embeddings]
    return padded_embeddings

def batch_process(X_text, batch_size=100):
    """
    Processes the text data in batches to avoid running out of RAM.
    Args:
        X_text (list): A list of text strings to be processed.
        batch_size (int): The number of documents to process in each batch.
    Returns:
        numpy.ndarray: A numpy array containing the embeddings for all text data.
    """
    max_length = get_global_max_length(X_text)  # Get the global max length
    embedding_dim = len(text_to_nlp(X_text[0])[0].vector)  # Get the embedding dimension
    X_mmap = np.memmap('embeddings.dat', dtype='float32', mode='w+', shape=(len(X_text), max_length, embedding_dim))

    for i in tqdm(range(0, len(X_text), batch_size)):
        text_batch = X_text[i:i+batch_size]
        embeddings = []
        for doc in text_to_nlp.pipe(text_batch):
            tokens = [token.vector for token in doc]
            padding = [[0] * len(tokens[0])] * (max_length - len(tokens))
            embeddings.append(padding + tokens)
        X_batch = np.array(embeddings)
        X_mmap[i:i+batch_size] = X_batch

        del embeddings, X_batch  # Free up memory
        gc.collect()  # Force garbage collection

    return X_mmap

def tokenize_and_embed_batch(text_batch):
    """
    Tokenizes a batch of text data and converts it to word embeddings using SpaCy.
    Args:
        text_batch (list): A batch of text strings to be processed.
    Returns:
        list: A list of lists containing embeddings for each token in each document.
    """
    docs = list(text_to_nlp.pipe(text_batch))
    embeddings = [[token.vector for token in doc] for doc in docs]
    return embeddings

def convert_to_array(padded_embeddings):
    """
    Converts a list of padded embeddings into a numpy array.
    Args:
        padded_embeddings (list): A list of lists of padded embeddings.
    Returns:
        numpy.ndarray: A numpy array containing the embeddings suitable for machine learning input.
    """
    return np.array(padded_embeddings)

# Convert text data to numerical features
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Reshape the data for LSTM (batch_size, timesteps, features)
X_train_reshaped = X_train_vec.toarray().reshape(X_train_vec.shape[0], 1, X_train_vec.shape[1])
X_test_reshaped = X_test_vec.toarray().reshape(X_test_vec.shape[0], 1, X_test_vec.shape[1])

# Convert labels to NumPy arrays and one-hot encode them
y_train_array = np.array(y_train_encoded)
y_test_array = np.array(y_test_encoded)
y_train_one_hot = to_categorical(y_train_array, num_classes=6)
y_test_one_hot = to_categorical(y_test_array, num_classes=6)

print(f"Processed training and test data shapes: {X_train_reshaped.shape}, {X_test_reshaped.shape}")
print(f"Processed label shapes: {y_train_one_hot.shape}, {y_test_one_hot.shape}")
# Liar-Liar-Project
