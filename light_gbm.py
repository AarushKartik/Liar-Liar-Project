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

# ------------------- Feature Importance Analysis -------------------
# Get feature importances from the model
importance = lgb_model.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})

# Sort by importance and get top 10 features
top_features = feature_importance.sort_values(by='Importance', ascending=False).head(10)

# Print top 10 features
print("\n🔍 Top 10 Most Important Features:")
for index, row in top_features.iterrows():
    print(f"{row['Feature']}: {row['Importance']:.6f}")

# Visualize top 10 features
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=top_features)
plt.title('Top 10 Feature Importance')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("✅ Feature importance visualization saved to 'feature_importance.png'")

# Analyze which embedding type contributes most to the top features
embed_types = ['bert', 'roberta', 'bilstm']
embed_counts = {embed: 0 for embed in embed_types}
embed_importance = {embed: 0 for embed in embed_types}

for index, row in top_features.iterrows():
    feature_name = row['Feature']
    for embed in embed_types:
        if feature_name.startswith(embed):
            embed_counts[embed] += 1
            embed_importance[embed] += row['Importance']

print("\n📊 Embedding Type Analysis in Top 10 Features:")
for embed in embed_types:
    print(f"{embed.upper()}: {embed_counts[embed]} features, Total importance: {embed_importance[embed]:.6f}")

# Save top features to file
top_features.to_csv('top_features.csv', index=False)
print("✅ Top features saved to 'top_features.csv'")

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
