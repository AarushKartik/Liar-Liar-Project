import numpy as np
import pandas as pd
import os
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # For saving/loading the model
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA
import umap.umap_ as umap
from scipy.spatial.distance import cosine

# ------------------- Load Precomputed Feature Vectors -------------------
# BERT features
bert_train = np.load("/content/drive/MyDrive/feature_vectors/train_bert/bert_train_1_features.npy")  
bert_test = np.load("/content/drive/MyDrive/feature_vectors/test_bert/bert_test_1_features.npy")  
bert_valid = np.load("/content/drive/MyDrive/feature_vectors/valid_bert/bert_valid_1_features.npy")  

# RoBERTa features
roberta_train = np.load("/content/drive/MyDrive/feature_vectors/train_roberta/roberta_train_1_features.npy")  
roberta_test = np.load("/content/drive/MyDrive/feature_vectors/test_roberta/roberta_test_1_features.npy")  
roberta_valid = np.load("/content/drive/MyDrive/feature_vectors/valid_roberta/roberta_valid_1_features.npy")  

# BiLSTM features
bilstm_train = np.load("/content/drive/MyDrive/feature_vectors/train_bilstm/bilstm/bilstm_train_1_features.npy")  
bilstm_test = np.load("/content/drive/MyDrive/feature_vectors/test_bilstm/bilstm/bilstm_test_1_features.npy")
bilstm_valid = np.load("/content/drive/MyDrive/feature_vectors/valid_bilstm/bilstm/bilstm_valid_1_features.npy")

# Concatenate features from all models
X_train = np.hstack([bert_train, roberta_train, bilstm_train])
X_test = np.hstack([bert_test, roberta_test, bilstm_test])
X_valid = np.hstack([bert_valid, roberta_valid, bilstm_valid])

# Track original feature dimensions for feature importance analysis
bert_dim = bert_train.shape[1]
roberta_dim = roberta_train.shape[1]
bilstm_dim = bilstm_train.shape[1]

# Create feature names for later use in feature importance
feature_names = (
    [f"bert_{i}" for i in range(bert_dim)] + 
    [f"roberta_{i}" for i in range(roberta_dim)] + 
    [f"bilstm_{i}" for i in range(bilstm_dim)]
)

# ------------------- Load Labels -------------------
y_train = np.loadtxt("feature_vectors/train/train_labels.txt", dtype=int)
y_test = np.loadtxt("feature_vectors/test/test_labels.txt", dtype=int)
y_valid = np.loadtxt("feature_vectors/valid/valid_labels.txt", dtype=int)

# ------------------- Feature Scaling -------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid)

# Save the scaler for future use
joblib.dump(scaler, "scaler.pkl")
print("✅ Feature scaling complete and scaler saved.")

# ------------------- Train LightGBM Model -------------------
# Define LightGBM model with fixed hyperparameters
lgb_model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=len(np.unique(y_train)),
    learning_rate=0.05,
    num_leaves=31,
    max_depth=5,
    min_data_in_leaf=50,
    lambda_l1=0.5,
    lambda_l2=0.5,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    verbosity=-1
)

# Train the model with the training data
print("\n🚀 Training LightGBM model...")
lgb_model.fit(X_train, y_train, feature_name=feature_names)

# Save the trained model
joblib.dump(lgb_model, "lightgbm_model.pkl")
print("✅ Model trained and saved.")


# ------------------- Model Evaluation -------------------
# Load the trained model (optional, since we already have it in memory)
model = joblib.load("lightgbm_model.pkl")

# Predict on all datasets to compare performance
y_train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

y_valid_pred = model.predict(X_valid)
valid_accuracy = accuracy_score(y_valid, y_valid_pred)

y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

# Print all accuracies for comparison
print(f"🎯 Final Training Accuracy: {train_accuracy:.4f}")
print(f"🎯 Final Validation Accuracy: {valid_accuracy:.4f}")
print(f"🎯 Final Test Accuracy: {test_accuracy:.4f}")

# Classification Report
print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred))

# Save predictions for analysis
np.save("y_pred_labels.npy", y_pred)
print("✅ Predictions saved.")

# ------------------- Confusion Matrix -------------------
# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("✅ Confusion matrix saved to 'confusion_matrix.png'")

# Additional analyses with BERT, RoBERTa and BiLSTM individually with LightGBM

os.makedirs("outputs", exist_ok=True)

# ------------------- Load Labels -------------------
y_train = np.loadtxt("feature_vectors/train/train_labels.txt", dtype=int)
y_test  = np.loadtxt("feature_vectors/test/test_labels.txt", dtype=int)
y_valid = np.loadtxt("feature_vectors/valid/valid_labels.txt", dtype=int)

# ------------------- Load Feature Sets -------------------
# BERT
bert_train = np.load("/content/drive/MyDrive/feature_vectors/train_bert/bert_train_1_features.npy")
bert_test  = np.load("/content/drive/MyDrive/feature_vectors/test_bert/bert_test_1_features.npy")
bert_valid = np.load("/content/drive/MyDrive/feature_vectors/valid_bert/bert_valid_1_features.npy")

# RoBERTa
roberta_train = np.load("/content/drive/MyDrive/feature_vectors/train_roberta/roberta_train_1_features.npy")
roberta_test  = np.load("/content/drive/MyDrive/feature_vectors/test_roberta/roberta_test_1_features.npy")
roberta_valid = np.load("/content/drive/MyDrive/feature_vectors/valid_roberta/roberta_valid_1_features.npy")

# BiLSTM
bilstm_train = np.load("/content/drive/MyDrive/feature_vectors/train_bilstm/bilstm/bilstm_train_1_features.npy")
bilstm_test  = np.load("/content/drive/MyDrive/feature_vectors/test_bilstm/bilstm/bilstm_test_1_features.npy")
bilstm_valid = np.load("/content/drive/MyDrive/feature_vectors/valid_bilstm/bilstm/bilstm_valid_1_features.npy")

# ------------------- Helper: Train/Eval Single-Feature Model -------------------
def train_eval_single(name, X_train, X_valid, X_test, y_train, y_valid, y_test):
    print(f"\n==== {name} + LightGBM ====")
    # Feature names (optional, for LightGBM)
    feat_dim = X_train.shape[1]
    feature_names = [f"{name.lower()}_{i}" for i in range(feat_dim)]

    # Scale (kept for consistency with your pipeline)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_valid_s = scaler.transform(X_valid)
    X_test_s  = scaler.transform(X_test)

    joblib.dump(scaler, f"outputs/{name.lower()}_scaler.pkl")

    # LightGBM model (same hyperparams as your combined run)
    lgb_model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=len(np.unique(y_train)),
        learning_rate=0.05,
        num_leaves=31,
        max_depth=5,
        min_data_in_leaf=50,
        lambda_l1=0.5,
        lambda_l2=0.5,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        verbosity=-1
    )

    print("Training...")
    lgb_model.fit(X_train_s, y_train, feature_name=feature_names)
    joblib.dump(lgb_model, f"outputs/{name.lower()}_lightgbm.pkl")
    print("Model saved.")

    # Evaluate
    def eval_split(split_name, Xs, ys):
        yp = lgb_model.predict(Xs)
        acc = accuracy_score(ys, yp)
        print(f"{split_name} Accuracy: {acc:.4f}")
        return yp, acc

    y_train_pred, train_acc = eval_split("Train", X_train_s, y_train)
    y_valid_pred, valid_acc = eval_split("Valid", X_valid_s, y_valid)
    y_test_pred,  test_acc  = eval_split("Test",  X_test_s,  y_test)

    # Classification report on test
    print("\nClassification Report (Test):")
    print(classification_report(y_test, y_test_pred))

    # Confusion matrix on test
    cm = confusion_matrix(y_test, y_test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{name} + LightGBM — Confusion Matrix (Test)')
    cm_path = f"outputs/{name.lower()}_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    print(f"Confusion matrix saved to '{cm_path}'")

    # Save predictions
    np.save(f"outputs/{name.lower()}_y_pred_train.npy", y_train_pred)
    np.save(f"outputs/{name.lower()}_y_pred_valid.npy", y_valid_pred)
    np.save(f"outputs/{name.lower()}_y_pred_test.npy",  y_test_pred)

    return {
        "name": name,
        "train_acc": train_acc,
        "valid_acc": valid_acc,
        "test_acc":  test_acc
    }

# ------------------- Run Analysis -------------------
results = []
results.append(train_eval_single("BERT",    bert_train,    bert_valid,    bert_test,    y_train, y_valid, y_test))
results.append(train_eval_single("RoBERTa", roberta_train, roberta_valid, roberta_test, y_train, y_valid, y_test))
results.append(train_eval_single("BiLSTM",  bilstm_train,  bilstm_valid,  bilstm_test,  y_train, y_valid, y_test))

# ------------------- Print Summary -------------------
print("\n==== Ablation Summary (Accuracy) ====")
for r in results:
    print(f"{r['name']:8s} | Train: {r['train_acc']:.4f}  Valid: {r['valid_acc']:.4f}  Test: {r['test_acc']:.4f}")

