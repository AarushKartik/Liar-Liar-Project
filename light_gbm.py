import numpy as np
import pandas as pd
import os
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib  # For saving/loading the model

# ------------------- Load Precomputed Feature Vectors -------------------
# BERT features
bert_train = np.load("/content/feature_vectors/train_bert/bert_train_1_features.npy")  
bert_test = np.load("/content/feature_vectors/test_bert/bert_test_1_features.npy")  
bert_valid = np.load("/content/feature_vectors/valid_bert/bert_valid_1_features.npy")  

# RoBERTa features
roberta_train = np.load("/content/feature_vectors/train_roberta/roberta_train_1_features.npy")  
roberta_test = np.load("/content/feature_vectors/test_roberta/roberta_test_1_features.npy")  
roberta_valid = np.load("/content/feature_vectors/valid_roberta/roberta_valid_1_features.npy")  

# BiLSTM features
bilstm_train = np.load("/content/feature_vectors/train_bilstm/bilstm/bilstm_train_1_features.npy")  
bilstm_test = np.load("/content/feature_vectors/test_bilstm/bilstm/bilstm_test_1_features.npy")  
bilstm_valid = np.load("/content/feature_vectors/valid_bilstm/bilstm/bilstm_valid_1_features.npy")  

# Concatenate features from all models
X_train = np.hstack([bert_train, roberta_train, bilstm_train])
X_test = np.hstack([bert_test, roberta_test, bilstm_test])
X_valid = np.hstack([bert_valid, roberta_valid, bilstm_valid])

# Print shapes for verification
print(f"✅ X_train shape: {X_train.shape}")
print(f"✅ X_test shape: {X_test.shape}")
print(f"✅ X_valid shape: {X_valid.shape}")

# ------------------- Load Labels -------------------
y_train = np.loadtxt("feature_vectors/train/train_labels.txt", dtype=int)
y_test = np.loadtxt("feature_vectors/test/test_labels.txt", dtype=int)
y_valid = np.loadtxt("feature_vectors/valid/valid_labels.txt", dtype=int)

# Print to verify shapes
print(f"✅ y_train shape: {y_train.shape}")
print(f"✅ y_test shape: {y_test.shape}")
print(f"✅ y_valid shape: {y_valid.shape}")

# ------------------- Feature Scaling -------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_valid = scaler.transform(X_valid)

# Save the scaler for future use
joblib.dump(scaler, "scaler.pkl")
print("✅ Feature scaling complete and scaler saved.")

# ------------------- Define Custom Callback for Accuracy Reporting -------------------
def accuracy_eval_callback(period=100):
    """
    Custom callback to report training and validation accuracies during LightGBM training.
    
    Args:
        period (int): Interval (in iterations) to compute and print accuracies
    """
    def callback(env):
        if env.iteration % period == 0:
            # Get current model
            model = env.model
            
            # Predict on training data
            y_train_pred = model.predict(X_train)
            y_train_pred_labels = np.argmax(y_train_pred, axis=1)
            train_acc = accuracy_score(y_train, y_train_pred_labels)
            
            # Predict on validation data
            y_valid_pred = model.predict(X_valid)
            y_valid_pred_labels = np.argmax(y_valid_pred, axis=1)
            valid_acc = accuracy_score(y_valid, y_valid_pred_labels)
            
            print(f"Iteration {env.iteration}: Train Accuracy: {train_acc:.4f}, Validation Accuracy: {valid_acc:.4f}")
    
    callback.order = 0
    return callback

# ------------------- Train LightGBM Model -------------------
# Define the LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

# Define parameters for LightGBM
params = {
    'objective': 'multiclass',  # Multi-class classification
    'num_class': len(np.unique(y_train)),  # Number of classes
    'metric': 'multi_logloss',  # Suitable metric for multi-class classification
    'boosting_type': 'gbdt',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'verbosity': -1,
    'seed': 42
}

# Train the LightGBM model with custom callback and 2000 rounds (twice as long)
print("🚀 Training LightGBM model...")
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, valid_data],
    valid_names=['train', 'valid'],
    num_boost_round=2000,  # Train for twice as long (2000 instead of 1000)
    callbacks=[accuracy_eval_callback(period=100)]  # Report accuracies every 100 iterations
)

# Save the trained model
joblib.dump(model, "lightgbm_model.pkl")
print("✅ Model training complete and saved.")

# ------------------- Model Evaluation -------------------
# Load trained model (to ensure we can reload it)
model = joblib.load("lightgbm_model.pkl")

# Predict on test set
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred_labels)
print(f"🎯 Test Accuracy: {accuracy:.4f}")

# Classification Report
print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred_labels))

# Save predictions for analysis
np.save("y_pred_labels.npy", y_pred_labels)
print("✅ Predictions saved.")
