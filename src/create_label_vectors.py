import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def load_tsv_files(train_file, test_file, valid_file):
    """Load TSV files and return processed dataframes"""
    df_train = pd.read_csv(train_file, sep='\t', header=None)
    df_test = pd.read_csv(test_file, sep='\t', header=None)
    df_valid = pd.read_csv(valid_file, sep='\t', header=None)
    
    return df_train, df_test, df_valid

def rename_columns(df):
    """Rename columns for consistency"""
    columns_mapping = {0: 'ID', 1: 'Label', 2: 'Statement', 3: 'Subject', 4: 'Speaker', 
                      5: 'Job Title', 6: 'State Info', 7: 'Party Affiliation'}
    df.rename(columns=columns_mapping, inplace=True)
    return df

def encode_labels(df, truthiness_rank):
    """Encode labels based on truthiness rank"""
    df['Label_Rank'] = df['Label'].map(truthiness_rank)
    return df

def save_labels_to_file(labels, split_name="train", data_num="1"):
    """
    Save labels to text files with similar structure to feature vectors.
    Format: feature_vectors/{split_name}/{split_name}_{data_num}_labels.txt
    """
    # Define output directory (generic, not model-specific)
    out_dir = os.path.join("feature_vectors", split_name)
    os.makedirs(out_dir, exist_ok=True)
    
    # Save labels in .txt format
    txt_path = os.path.join(out_dir, f"{split_name}_labels.txt")
    
    # Convert labels to numpy array if not already
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    # Save labels as integers, one per line
    np.savetxt(txt_path, labels, fmt='%d')
    print(f"[{split_name.upper()}] Labels saved to: {txt_path}")
    
    return txt_path

def main():
    # Define truthiness ranking
    truthiness_rank = {
        'true': 0,
        'mostly-true': 1,
        'half-true': 2,
        'barely-true': 3,
        'false': 4,
        'pants-fire': 5
    }
    
    print("Loading and processing data...")
    
    # Load and preprocess data
    df_train, df_test, df_valid = load_tsv_files('train.tsv', 'test.tsv', 'valid.tsv')
    df_train = rename_columns(df_train)
    df_test = rename_columns(df_test)
    df_valid = rename_columns(df_valid)
    
    # Encode labels
    df_train = encode_labels(df_train, truthiness_rank)
    df_test = encode_labels(df_test, truthiness_rank)
    df_valid = encode_labels(df_valid, truthiness_rank)
    
    # Get labels
    y_train = df_train['Label_Rank'].values
    y_test = df_test['Label_Rank'].values
    y_valid = df_valid['Label_Rank'].values
    
    print(f"Number of training samples: {len(y_train)}")
    print(f"Number of testing samples: {len(y_test)}")
    print(f"Number of validation samples: {len(y_valid)}")
    
    # Save labels to text files
    print("Saving labels to text files...")
    train_path = save_labels_to_file(y_train, split_name="train", data_num="1")
    test_path = save_labels_to_file(y_test, split_name="test", data_num="1")
    valid_path = save_labels_to_file(y_valid, split_name="valid", data_num="1")
    
    print("\nLabel files created successfully:")
    print(f"Training labels: {train_path}")
    print(f"Testing labels: {test_path}")
    print(f"Validation labels: {valid_path}")
    
    print("\nNote: These label files are generic and can be used with any model, not just RoBERTa.")

if __name__ == "__main__":
    main()
