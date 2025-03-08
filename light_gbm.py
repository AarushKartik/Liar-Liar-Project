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

# ------------------- Detailed Feature Importance Analysis -------------------
# Get feature importances from the model
importance = lgb_model.feature_importances_
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
})

# Add model type to each feature
feature_importance['Model'] = feature_importance['Feature'].apply(
    lambda x: 'BERT' if x.startswith('bert') else ('RoBERTa' if x.startswith('roberta') else 'BiLSTM')
)

# Sort by importance and get top 10 features
top_features = feature_importance.sort_values(by='Importance', ascending=False).head(10)

# Print top 10 features with details
print("\n🔍 Top 10 Most Important Features:")
print("Rank | Feature | Model | Importance")
print("-" * 50)
for i, (index, row) in enumerate(top_features.iterrows(), 1):
    print(f"{i:2d}   | {row['Feature']:15s} | {row['Model']:7s} | {row['Importance']:.6f}")

# Visualize top 10 features with model color-coding (fixed approach)
plt.figure(figsize=(12, 6))
# Create a color map for the models
colors = {'BERT': 'royalblue', 'RoBERTa': 'forestgreen', 'BiLSTM': 'crimson'}
bar_colors = [colors[model] for model in top_features['Model']]

# Create the plot
bars = plt.barh(top_features['Feature'], top_features['Importance'], color=bar_colors)

# Add a legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=colors[model], label=model) for model in colors]
plt.legend(handles=legend_elements)

plt.title('Top 10 Feature Importance by Model Type')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('feature_importance.png')
print("✅ Feature importance visualization saved to 'feature_importance.png'")

# Calculate total importance contribution by model
model_importance = top_features.groupby('Model')['Importance'].sum().reset_index()
model_importance['Percentage'] = (model_importance['Importance'] / model_importance['Importance'].sum()) * 100
model_feature_count = top_features['Model'].value_counts().reset_index()
model_feature_count.columns = ['Model', 'Count']

# Merge counts and importance
model_summary = pd.merge(model_importance, model_feature_count, on='Model')

# Print overall model importance in top 10
print("\n📊 Model Contribution in Top 10 Features:")
print("Model   | Count | Total Importance | Percentage")
print("-" * 55)
for _, row in model_summary.iterrows():
    print(f"{row['Model']:7s} | {row['Count']:5d} | {row['Importance']:16.6f} | {row['Percentage']:8.2f}%")

# Visualize model contribution as pie charts
plt.figure(figsize=(10, 6))

# Create pie chart for feature count
plt.subplot(1, 2, 1)
plt.pie(
    model_summary['Count'], 
    labels=model_summary['Model'], 
    autopct='%1.1f%%', 
    colors=[colors[m] for m in model_summary['Model']]
)
plt.title('Feature Count Distribution in Top 10')

# Create pie chart for importance distribution
plt.subplot(1, 2, 2)
plt.pie(
    model_summary['Importance'], 
    labels=model_summary['Model'], 
    autopct='%1.1f%%', 
    colors=[colors[m] for m in model_summary['Model']]
)
plt.title('Importance Distribution in Top 10')

plt.tight_layout()
plt.savefig('model_contribution.png')
print("✅ Model contribution visualization saved to 'model_contribution.png'")

# Save top features to file with detailed info
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
