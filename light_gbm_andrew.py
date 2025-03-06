import numpy as np
import pandas as pd
import os
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # For saving/loading the model

# ------------------- Add Cross-Validation Function -------------------
def perform_kfold_cv(X, y, params, n_splits=5):
    """
    Perform k-fold cross-validation to get robust performance estimates.
    
    Args:
        X: Feature matrix
        y: Target labels
        params: LightGBM parameters
        n_splits: Number of folds
        
    Returns:
        Mean accuracy and standard deviation across folds
    """
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]
        
        train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
        val_data = lgb.Dataset(X_fold_val, label=y_fold_val, reference=train_data)
        
        # Train model
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=500,
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=False)
            ]
        )
        
        # Predict
        y_pred = model.predict(X_fold_val)
        y_pred_labels = np.argmax(y_pred, axis=1)
        
        # Calculate accuracy
        acc = accuracy_score(y_fold_val, y_pred_labels)
        accuracies.append(acc)
        print(f"Fold {fold+1} Accuracy: {acc:.4f}")
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    print(f"Mean CV Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    
    return mean_acc, std_acc

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

# ------------------- Run Cross-Validation to Find Optimal Parameters -------------------
print("\n🔍 Running cross-validation to find optimal parameters...")

# Define optimal parameter search
cv_params = {
    'objective': 'multiclass',
    'num_class': len(np.unique(y_train)),
    'metric': 'multi_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,
    'num_leaves': 20,
    'max_depth': 5,
    'min_data_in_leaf': 50,
    'lambda_l1': 0.5,
    'lambda_l2': 0.5,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbosity': -1,
    'seed': 42
}

# Perform CV to find optimal parameters
cv_acc, cv_std = perform_kfold_cv(X_train, y_train, cv_params, n_splits=5)
print(f"✅ Cross-validation completed with accuracy: {cv_acc:.4f} ± {cv_std:.4f}")

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

# Define parameters for LightGBM with anti-overfitting measures
params = {
    'objective': 'multiclass',  # Multi-class classification
    'num_class': len(np.unique(y_train)),  # Number of classes
    'metric': 'multi_logloss',  # Suitable metric for multi-class classification
    'boosting_type': 'gbdt',
    'learning_rate': 0.01,  # Reduced learning rate for better generalization
    'num_leaves': 20,  # Reduced from 31 to decrease model complexity
    'max_depth': 5,  # Set a specific max depth to limit tree growth
    'min_data_in_leaf': 50,  # Increased to ensure more samples per leaf
    'lambda_l1': 0.5,  # L1 regularization to encourage sparsity
    'lambda_l2': 0.5,  # L2 regularization to prevent large coefficients
    'feature_fraction': 0.8,  # Use 80% of features in each iteration
    'bagging_fraction': 0.8,  # Use 80% of data in each iteration
    'bagging_freq': 5,  # Perform bagging every 5 iterations
    'verbosity': -1,
    'seed': 42
}

# Train the LightGBM model with custom callback, early stopping, and 2000 rounds
print("🚀 Training LightGBM model...")
model = lgb.train(
    params,
    train_data,
    valid_sets=[train_data, valid_data],
    valid_names=['train', 'valid'],
    num_boost_round=2000,  # Train for twice as long (2000 instead of 1000)
    callbacks=[
        accuracy_eval_callback(period=50),  # Report accuracies more frequently
        lgb.early_stopping(stopping_rounds=100, verbose=True)  # Stop if no improvement for 100 rounds
    ]
)

# Save the trained model
joblib.dump(model, "lightgbm_model.pkl")
print("✅ Model training complete and saved.")

# ------------------- Model Evaluation -------------------
# Load trained model (to ensure we can reload it)
model = joblib.load("lightgbm_model.pkl")

# Predict on all datasets to compare performance
y_train_pred = model.predict(X_train)
y_train_pred_labels = np.argmax(y_train_pred, axis=1)
train_accuracy = accuracy_score(y_train, y_train_pred_labels)

y_valid_pred = model.predict(X_valid)
y_valid_pred_labels = np.argmax(y_valid_pred, axis=1)
valid_accuracy = accuracy_score(y_valid, y_valid_pred_labels)

y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
test_accuracy = accuracy_score(y_test, y_pred_labels)

# Print all accuracies for comparison
print(f"🎯 Final Training Accuracy: {train_accuracy:.4f}")
print(f"🎯 Final Validation Accuracy: {valid_accuracy:.4f}")
print(f"🎯 Final Test Accuracy: {test_accuracy:.4f}")

# Calculate the gap between train and test accuracy
accuracy_gap = train_accuracy - test_accuracy
print(f"📊 Train-Test Accuracy Gap: {accuracy_gap:.4f}")

# Classification Report
print("\n🔍 Classification Report:")
print(classification_report(y_test, y_pred_labels))

# Save predictions for analysis
np.save("y_pred_labels.npy", y_pred_labels)
print("✅ Predictions saved.")

# ------------------- Feature Importance Analysis -------------------
# Get feature importances
importance = model.feature_importance(importance_type='gain')

# Create feature names (since we don't have actual names)
# Assuming features are concatenated in order: BERT, RoBERTa, BiLSTM
feature_count = X_train.shape[1]
bert_count = bert_train.shape[1]
roberta_count = roberta_train.shape[1]
bilstm_count = bilstm_train.shape[1]

feature_names = []
feature_names.extend([f'BERT_{i}' for i in range(bert_count)])
feature_names.extend([f'RoBERTa_{i}' for i in range(roberta_count)])
feature_names.extend([f'BiLSTM_{i}' for i in range(bilstm_count)])

# Create a dataframe of feature importances
feature_imp = pd.DataFrame({'Feature': feature_names[:len(importance)], 
                            'Importance': importance})
feature_imp = feature_imp.sort_values(by='Importance', ascending=False)

# Print top 20 most important features
print("\n🔍 Top 20 Most Important Features:")
print(feature_imp.head(20))

# Plot feature importances
plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=feature_imp.head(30))
plt.title('Top 30 Feature Importances')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("✅ Feature importance graph saved to 'feature_importance.png'")

# ------------------- Confusion Matrix -------------------
# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
print("✅ Confusion matrix saved to 'confusion_matrix.png'")
