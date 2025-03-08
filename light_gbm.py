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
from sklearn.metrics.pairwise import cosine_similarity

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

# ------------------- Enhanced Feature Interpretation -------------------
# Load original texts (adapt this path to your actual data source)
def load_original_texts():
    try:
        # Modify this path to point to your original text data
        # For example, from CSV: pd.read_csv('your_texts.csv')['text'].tolist()
        # Or from text files in a directory
        text_path = "/content/drive/MyDrive/original_texts.txt"
        
        if os.path.exists(text_path):
            with open(text_path, 'r', encoding='utf-8') as f:
                texts = f.readlines()
            print(f"✅ Loaded {len(texts)} original texts for interpretation")
            return texts
        else:
            # Alternative: try to load from a CSV or other source
            try:
                texts_df = pd.read_csv("/content/drive/MyDrive/original_texts.csv")
                texts = texts_df['text'].tolist()  # Adjust column name as needed
                print(f"✅ Loaded {len(texts)} original texts from CSV")
                return texts
            except:
                print("⚠️ Original text files not found. Using placeholder approach.")
                return None
    except Exception as e:
        print(f"⚠️ Could not load original texts: {e}")
        return None

# Function to get class names if available
def get_class_names():
    try:
        # Try to load class names from a file
        class_names_path = "/content/drive/MyDrive/class_names.txt"
        if os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                class_names = {i: name.strip() for i, name in enumerate(f.readlines())}
            return class_names
        else:
            # Alternative: try CSV
            try:
                names_df = pd.read_csv("/content/drive/MyDrive/class_names.csv")
                class_names = dict(zip(names_df['id'], names_df['name']))
                return class_names
            except:
                # Default to generic class names
                unique_classes = sorted(np.unique(y_train))
                class_names = {cls_id: f"Class {cls_id}" for cls_id in unique_classes}
                return class_names
    except:
        # Default to generic class names
        unique_classes = sorted(np.unique(y_train))
        class_names = {cls_id: f"Class {cls_id}" for cls_id in unique_classes}
        return class_names

# Load original texts and class names
original_texts = load_original_texts()
class_names = get_class_names()

# Create embeddings dictionary using unscaled embeddings for better interpretability
embeddings_dict = {
    'bert': bert_train,
    'roberta': roberta_train,
    'bilstm': bilstm_train
}

# --------------------- Method 1: Activation Maximization ---------------------
def analyze_feature_representation(feature_indices, model_types, embeddings_dict, original_texts=None):
    """
    Analyze what features might represent by examining feature patterns
    
    Args:
        feature_indices: List of feature indices
        model_types: List of corresponding model types
        embeddings_dict: Dictionary with keys 'bert', 'roberta', 'bilstm' 
                         containing the respective embeddings
        original_texts: List of original text documents (optional)
    """
    results = []
    
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
        
        # If original texts are available, include them in the analysis
        top_texts = None
        bottom_texts = None
        if original_texts is not None:
            try:
                top_texts = [original_texts[idx] for idx in top_examples_idx]
                bottom_texts = [original_texts[idx] for idx in bottom_examples_idx]
            except:
                print(f"⚠️ Could not retrieve original texts for some indices")
                top_texts = None
                bottom_texts = None
        
        # Correlation with target variable (simplified analysis)
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
            'top_example_texts': top_texts,
            'bottom_example_texts': bottom_texts
        }
        
        results.append(feature_interpretation)
    
    return results

# --------------------- Method 2: Word-Feature Correlation ---------------------
def analyze_word_feature_correlations(top_features, embeddings_dict, original_texts):
    """
    Correlate important features with word presence in documents
    """
    if original_texts is None:
        print("⚠️ Cannot perform word correlation analysis without original texts")
        return None
    
    print("\n🔍 Analyzing correlations between top features and words...")
    
    # Create a word presence matrix
    vectorizer = CountVectorizer(max_features=5000, stop_words='english')
    try:
        word_matrix = vectorizer.fit_transform(original_texts)
        word_features = vectorizer.get_feature_names_out()
    except:
        print("⚠️ Error creating word matrix. Check if texts are valid.")
        return None
    
    # Prepare results dataframe
    word_correlation_results = pd.DataFrame()
    
    # For each important embedding feature
    for i, (idx, row) in enumerate(top_features.iterrows()):
        feature_idx = row['Index']
        model_type = row['Model']
        
        print(f"Analyzing word correlations for {model_type}_{feature_idx} (Rank {i+1})...")
        
        # Get the appropriate embeddings
        if model_type == 'BERT':
            embedding_values = embeddings_dict['bert'][:, feature_idx]
        elif model_type == 'RoBERTa':
            embedding_values = embeddings_dict['roberta'][:, feature_idx]
        else:  # BiLSTM
            embedding_values = embeddings_dict['bilstm'][:, feature_idx]
        
        # Compute correlation between feature and word presence
        word_correlations = []
        sample_size = min(len(embedding_values), word_matrix.shape[0])
        for word_idx in range(word_matrix.shape[1]):
            word_presence = word_matrix[:sample_size, word_idx].toarray().flatten()
            try:
                correlation = np.corrcoef(embedding_values[:sample_size], word_presence)[0, 1]
                if not np.isnan(correlation):
                    word_correlations.append((word_features[word_idx], correlation))
            except:
                continue
        
        # Get top correlated words (both positive and negative)
        top_positive = sorted(word_correlations, key=lambda x: x[1], reverse=True)[:10]
        top_negative = sorted(word_correlations, key=lambda x: x[1])[:10]
        
        # Add to results
        result_row = {
            'Feature': f"{model_type}_{feature_idx}",
            'Rank': i+1,
            'Positive_Correlated_Words': ', '.join([f"{w} ({c:.3f})" for w, c in top_positive]),
            'Negative_Correlated_Words': ', '.join([f"{w} ({c:.3f})" for w, c in top_negative])
        }
        word_correlation_results = pd.concat([word_correlation_results, pd.DataFrame([result_row])], ignore_index=True)
        
        print(f"  Positive correlations: {', '.join([w for w, _ in top_positive[:5]])}")
        print(f"  Negative correlations: {', '.join([w for w, _ in top_negative[:5]])}")
    
    # Save results
    word_correlation_results.to_csv('word_correlations.csv', index=False)
    print("✅ Word correlation analysis saved to 'word_correlations.csv'")
    
    return word_correlation_results

# --------------------- Method 3: Topic Modeling ---------------------
def analyze_topics_for_features(top_features, embeddings_dict, original_texts, n_topics=3):
    """
    Apply topic modeling to examples where specific features are highly activated
    """
    if original_texts is None:
        print("⚠️ Cannot perform topic modeling without original texts")
        return None
    
    print("\n🔍 Extracting topics from high-activation examples...")
    
    topic_results = pd.DataFrame()
    
    for i, (idx, row) in enumerate(top_features.iterrows()):
        feature_idx = row['Index']
        model_type = row['Model']
        
        print(f"Analyzing topics for {model_type}_{feature_idx} (Rank {i+1})...")
        
        # Get appropriate feature values
        if model_type == 'BERT':
            feature_values = embeddings_dict['bert'][:, feature_idx]
        elif model_type == 'RoBERTa':
            feature_values = embeddings_dict['roberta'][:, feature_idx]
        else:  # BiLSTM
            feature_values = embeddings_dict['bilstm'][:, feature_idx]
        
        # Get top examples where this feature is most activated
        try:
            num_examples = min(100, len(original_texts))
            top_examples_idx = np.argsort(feature_values)[-num_examples:][::-1]
            top_example_texts = [original_texts[i] for i in top_examples_idx]
            
            # Apply topic modeling to these texts
            vectorizer = CountVectorizer(max_features=1000, stop_words='english')
            X = vectorizer.fit_transform(top_example_texts)
            
            # Use LDA for topic modeling
            lda = LDA(n_components=n_topics)
            lda.fit(X)
            
            # Extract topics
            feature_name = f"{model_type.lower()}_{feature_idx}"
            print(f"Topics detected in high-activation examples for {feature_name}:")
            
            topic_words = []
            vocab = vectorizer.get_feature_names_out()
            for topic_idx, topic in enumerate(lda.components_):
                top_words = [vocab[i] for i in topic.argsort()[:-11:-1]]
                topic_words.append(' '.join(top_words))
                print(f"  Topic {topic_idx+1}: {' '.join(top_words)}")
            
            # Add to results
            result_row = {
                'Feature': f"{model_type}_{feature_idx}",
                'Rank': i+1,
                'Topic_1': topic_words[0] if len(topic_words) > 0 else "",
                'Topic_2': topic_words[1] if len(topic_words) > 1 else "",
                'Topic_3': topic_words[2] if len(topic_words) > 2 else ""
            }
            topic_results = pd.concat([topic_results, pd.DataFrame([result_row])], ignore_index=True)
            
        except Exception as e:
            print(f"⚠️ Error in topic modeling for {feature_name}: {e}")
            continue
    
    # Save results
    topic_results.to_csv('feature_topics.csv', index=False)
    print("✅ Topic analysis saved to 'feature_topics.csv'")
    
    return topic_results

# --------------------- Method 4: Nearest Neighbors ---------------------
def analyze_nearest_neighbors(top_features, embeddings_dict, original_texts):
    """
    Find nearest neighbors in embedding space for high-activation examples
    """
    if original_texts is None:
        print("⚠️ Cannot perform nearest neighbors analysis without original texts")
        return None
    
    print("\n🔍 Finding nearest neighbors in embedding space...")
    
    nn_results = pd.DataFrame()
    
    for i, (idx, row) in enumerate(top_features.iterrows()):
        feature_idx = row['Index']
        model_type = row['Model']
        
        if model_type not in ['BERT', 'RoBERTa']:  # Skip BiLSTM for this analysis
            continue
            
        print(f"Analyzing embedding space for {model_type}_{feature_idx} (Rank {i+1})...")
        
        # Get appropriate embedding matrix
        embedding_matrix = embeddings_dict[model_type.lower()]
        
        # Extract feature values
        feature_values = embedding_matrix[:, feature_idx]
        
        try:
            # Get top examples where this feature is most activated
            top_examples_idx = np.argsort(feature_values)[-20:][::-1]
            
            # Get average embedding for these examples (centroid)
            high_activation_embeddings = embedding_matrix[top_examples_idx]
            centroid = np.mean(high_activation_embeddings, axis=0)
            
            # Find nearest neighbors in the embedding space
            similarities = cosine_similarity([centroid], embedding_matrix)[0]
            neighbor_indices = np.argsort(similarities)[-20:][::-1]
            
            # Extract texts
            neighbor_texts = [original_texts[i] if i < len(original_texts) else "Unknown" for i in neighbor_indices]
            
            # Look for common words in these texts
            if len(neighbor_texts) > 0:
                # Create mini corpus of neighbor texts
                vectorizer = CountVectorizer(max_features=100, stop_words='english')
                X = vectorizer.fit_transform(neighbor_texts)
                
                # Get word frequencies
                word_freq = np.sum(X.toarray(), axis=0)
                words = vectorizer.get_feature_names_out()
                
                # Sort by frequency
                sorted_indices = np.argsort(word_freq)[::-1]
                common_words = [words[i] for i in sorted_indices[:15]]
                
                print(f"  Common words in neighbors: {', '.join(common_words)}")
                
                # Add to results
                result_row = {
                    'Feature': f"{model_type}_{feature_idx}",
                    'Rank': i+1,
                    'Common_Words': ', '.join(common_words),
                    'Neighbor_Count': len(neighbor_texts)
                }
                nn_results = pd.concat([nn_results, pd.DataFrame([result_row])], ignore_index=True)
                
        except Exception as e:
            print(f"⚠️ Error in nearest neighbors analysis for {model_type}_{feature_idx}: {e}")
            continue
    
    # Save results
    if not nn_results.empty:
        nn_results.to_csv('nearest_neighbors.csv', index=False)
        print("✅ Nearest neighbors analysis saved to 'nearest_neighbors.csv'")
    
    return nn_results

# --------------------- Run All Feature Interpretation Methods ---------------------
print("\n🚀 Starting enhanced feature interpretation...")

# Get feature interpretations from activation maximization (original method)
feature_interpretations = analyze_feature_representation(
    top_features['Index'].values,
    top_features['Model'].values,
    embeddings_dict,
    original_texts
)

# Save basic interpretations
interpretation_df = pd.DataFrame()
for interp in feature_interpretations:
    # Save to dataframe for export
    row = {
        'Rank': interp['rank'],
        'Feature': interp['feature_name'],
        'Model': interp['model_type'],
        'Most_Associated_Class': class_names[interp['most_associated_class']],
        'Class_Correlation': str(interp['class_correlations']),
        'Avg_High_Activation': np.mean(interp['top_activation_values']),
        'Avg_Low_Activation': np.mean(interp['bottom_activation_values']),
    }
    
    # Add example texts if available
    if interp['top_example_texts']:
        row['Example_High_Activation'] = interp['top_example_texts'][0][:100] + "..." if len(interp['top_example_texts'][0]) > 100 else interp['top_example_texts'][0]
        row['Example_Low_Activation'] = interp['bottom_example_texts'][0][:100] + "..." if len(interp['bottom_example_texts'][0]) > 100 else interp['bottom_example_texts'][0]
    
    interpretation_df = pd.concat([interpretation_df, pd.DataFrame([row])], ignore_index=True)

# Save interpretations to CSV
interpretation_df.to_csv('feature_interpretations.csv', index=False)
print("✅ Basic feature interpretations saved to 'feature_interpretations.csv'")

# Run word correlation analysis if original texts are available
if original_texts:
    word_correlations = analyze_word_feature_correlations(top_features, embeddings_dict, original_texts)
    
    # Run topic modeling on examples with high feature activation
    topic_analysis = analyze_topics_for_features(top_features, embeddings_dict, original_texts)
    
    # Run nearest neighbors analysis in embedding space
    nn_analysis = analyze_nearest_neighbors(top_features, embeddings_dict, original_texts)
    
    # Create a comprehensive feature interpretation report
    print("\n📊 Creating comprehensive feature interpretation report...")
    
    # Combine all analyses
    comprehensive_df = interpretation_df.copy()
    
    # Add word correlations if available
    if word_correlations is not None and not word_correlations.empty:
        comprehensive_df = pd.merge(comprehensive_df, 
                                    word_correlations[['Feature', 'Positive_Correlated_Words']], 
                                    on='Feature', how='left')
    
    # Add topic information if available
    if topic_analysis is not None and not topic_analysis.empty:
        comprehensive_df = pd.merge(comprehensive_df, 
                                    topic_analysis[['Feature', 'Topic_1']], 
                                    on='Feature', how='left')
    
    # Add nearest neighbors info if available
    if nn_analysis is not None and not nn_analysis.empty:
        comprehensive_df = pd.merge(comprehensive_df, 
                                   nn_analysis[['Feature', 'Common_Words']], 
                                   on='Feature', how='left')
    
    # Save comprehensive report
    comprehensive_df.to_csv('comprehensive_feature_interpretation.csv', index=False)
    print("✅ Comprehensive feature interpretation saved to 'comprehensive_feature_interpretation.csv'")
    
    # Visualize relationships with interpretations
    plt.figure(figsize=(15, 10))
    for i, (_, row) in enumerate(comprehensive_df.head(5).iterrows()):
        plt.subplot(1, 5, i+1)
        
        # Create a word cloud-like visualization
        if 'Positive_Correlated_Words' in row and isinstance(row['Positive_Correlated_Words'], str):
            words = [w.split(' (')[0] for w in row['Positive_Correlated_Words'].split(', ')]
            y_pos = np.arange(len(words))
            plt.barh(y_pos, range(len(words), 0, -1))
            plt.yticks(y_pos, words)
            plt.title(f"{row['Feature']}\n{row['Most_Associated_Class']}")
            plt.tight_layout()
    
    plt.savefig('feature_words_relationship.png')
    print("✅ Feature-words relationship visualization saved to 'feature_words_relationship.png'")
else:
    print("⚠️ Skipping advanced feature interpretation methods that require original texts")
    print("⚠️ To enable comprehensive feature interpretation, please ensure original_texts.txt or original_texts.csv is available")

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
