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

# Add feature index for later analysis
feature_importance['Index'] = feature_importance['Feature'].apply(
    lambda x: int(x.split('_')[1])
)

# Sort by importance and get top 10 features
top_features = feature_importance.sort_values(by='Importance', ascending=False).head(10)

# Print top 10 features with details
print("\n🔍 Top 10 Most Important Features:")
print("Rank | Feature | Model | Importance")
print("-" * 50)
for i, (index, row) in enumerate(top_features.iterrows(), 1):
    print(f"{i:2d}   | {row['Feature']:15s} | {row['Model']:7s} | {row['Importance']:.6f}")

# Visualize top 10 features with model color-coding
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

# ------------------- Feature Interpretation -------------------
# Attempt to interpret what each important feature represents
# This is a simplified approach and will need to be adapted to your specific dataset

# 1. Function to load your original text data (adapt this to your data source)
def load_original_texts():
    try:
        # Try to load the original texts from appropriate sources
        # This is a placeholder - replace with actual loading code
        print("Attempting to load original text data...")
        
        # Example: Load from CSV or text files
        # texts = pd.read_csv('your_original_texts.csv')['text'].tolist()
        
        # For now, we'll simulate with a message
        print("ℹ️ Original text data not available for direct feature interpretation.")
        print("ℹ️ Will proceed with feature analysis without direct word mapping.")
        return None
    except Exception as e:
        print(f"Could not load original texts: {e}")
        return None

# 2. Function to analyze feature representation in embedding space
def analyze_feature_representation(feature_indices, model_types, embeddings_dict):
    """
    Analyze what features might represent by examining feature patterns
    
    Args:
        feature_indices: List of feature indices
        model_types: List of corresponding model types
        embeddings_dict: Dictionary with keys 'bert', 'roberta', 'bilstm' 
                         containing the respective embeddings
    """
    results = []
    
    # Map to track feature offsets
    offsets = {
        'BERT': 0,
        'RoBERTa': bert_dim,
        'BiLSTM': bert_dim + roberta_dim
    }
    
    for i, (feature_idx, model_type) in enumerate(zip(feature_indices, model_types)):
        # Get the appropriate embeddings based on model type
        if model_type.lower() == 'bert':
            embedding_matrix = embeddings_dict['bert']
            true_idx = feature_idx
        elif model_type.lower() == 'roberta':
            embedding_matrix = embeddings_dict['roberta']
            true_idx = feature_idx
        else:  # BiLSTM
            embedding_matrix = embeddings_dict['bilstm']
            true_idx = feature_idx
        
        # Extract the feature column values for all training examples
        feature_values = embedding_matrix[:, true_idx]
        
        # Find top examples where this feature is most activated
        top_examples_idx = np.argsort(feature_values)[-5:][::-1]
        top_values = feature_values[top_examples_idx]
        
        # Find examples where feature is least activated
        bottom_examples_idx = np.argsort(feature_values)[:5]
        bottom_values = feature_values[bottom_examples_idx]
        
        # Correlation with target variable (simplified analysis)
        # This helps identify if the feature is associated with specific classes
        class_means = {}
        for class_id in np.unique(y_train):
            class_mask = (y_train == class_id)
            class_mean = np.mean(feature_values[class_mask])
            class_means[class_id] = class_mean
        
        # Find the class with the highest mean for this feature
        most_associated_class = max(class_means.items(), key=lambda x: x[1])[0]
        class_correlation = {cls: val for cls, val in class_means.items()}
        
        # Store results
        feature_interpretation = {
            'rank': i+1,
            'feature_name': f"{model_type.lower()}_{feature_idx}",
            'model_type': model_type,
            'most_associated_class': most_associated_class,
            'class_correlations': class_correlation,
            'top_activation_values': top_values.tolist(),
            'bottom_activation_values': bottom_values.tolist(),
            'highest_activated_examples': top_examples_idx.tolist(),
            'lowest_activated_examples': bottom_examples_idx.tolist(),
        }
        
        results.append(feature_interpretation)
    
    return results

# 3. Function to map class IDs to class names (if available)
def get_class_names():
    # Replace with actual class names if available
    # This is a placeholder - modify based on your dataset
    try:
        # Try to load class names
        # class_names = pd.read_csv('class_names.csv')['name'].tolist()
        # return class_names
        
        # For now, we'll simulate with default class names
        unique_classes = sorted(np.unique(y_train))
        class_names = {cls_id: f"Class {cls_id}" for cls_id in unique_classes}
        return class_names
    except:
        # Default to generic class names
        unique_classes = sorted(np.unique(y_train))
        class_names = {cls_id: f"Class {cls_id}" for cls_id in unique_classes}
        return class_names

# 4. Prepare data for feature interpretation
original_texts = load_original_texts()
class_names = get_class_names()

# Create embeddings dictionary using unscaled embeddings for better interpretability
embeddings_dict = {
    'bert': bert_train,
    'roberta': roberta_train,
    'bilstm': bilstm_train
}

# Analyze top features
feature_interpretations = analyze_feature_representation(
    top_features['Index'].values,
    top_features['Model'].values,
    embeddings_dict
)

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
