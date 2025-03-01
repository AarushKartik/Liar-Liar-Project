import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

# Load precomputed feature vectors for train, test, and valid separately
bert_train = np.load("feature_vectors/train/bert/bert_train_1_features.npy")  
bert_test = np.load("feature_vectors/test/bert/bert_test_1_features.npy")  
bert_valid = np.load("feature_vectors/valid/bert/bert_valid_1_features.npy")  

roberta_train = np.load("feature_vectors/train/roberta/roberta_train_1_features.npy")  
roberta_test = np.load("feature_vectors/test/roberta/roberta_test_1_features.npy")  
roberta_valid = np.load("feature_vectors/valid/roberta/roberta_valid_1_features.npy")  

bilstm_train = np.load("feature_vectors/train/bilstm/bilstm_train_1_features.npy")  
bilstm_test = np.load("feature_vectors/test/bilstm/bilstm_test_1_features.npy")  
bilstm_valid = np.load("feature_vectors/valid/bilstm/bilstm_valid_1_features.npy")  

# Concatenate features
X_train = np.hstack([bert_train, roberta_train, bilstm_train])
X_test = np.hstack([bert_test, roberta_test, bilstm_test])
X_valid = np.hstack([bilstm_valid, roberta_valid, bilstm_valid])

# Print shapes for verification
print(f"✅ X_train shape: {X_train.shape}")
print(f"✅ X_test shape: {X_test.shape}")
print(f"✅ X_valid shape: {X_valid.shape}")

# Load labels

df = pd.DataFrame(data)
labels = df['Label_Rank'].values

# First, split into training (80%) and a temporary set (20%)
y_train, y_temp = train_test_split(
    labels, test_size=0.2, stratify=labels, random_state=42
)

# Next, split the temporary set equally into test (10%) and validation (10%)
y_test, y_valid = train_test_split(
    y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# Save the split labels as .npy files
np.save("labels_train.npy", y_train)
np.save("labels_test.npy", y_test)
np.save("labels_valid.npy", y_valid)

print("Labels saved successfully!")

y_train = np.load("labels_train.npy")
y_test = np.load("labels_test.npy")
y_valid = np.load("labels_valid.npy")




