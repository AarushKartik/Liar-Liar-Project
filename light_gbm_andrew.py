import numpy as np
import pandas as pd
import os
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from joblib import Parallel, delayed
from functools import partial
import time
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# ------------------- Define Base Models for Stacking -------------------
class BaseLGBMModel:
    """Base LightGBM model for the first level of stacking"""
    
    def __init__(self, params, feature_name=""):
        self.params = params
        self.model = None
        self.feature_name = feature_name
    
    def train(self, X_train, y_train, X_valid=None, y_valid=None):
        train_data = lgb.Dataset(X_train, label=y_train)
    
        # Set up validation datasets
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_valid is not None and y_valid is not None:
            valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)
            valid_sets.append(valid_data)
            valid_names.append('valid')
        
        # Train without early stopping
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=200,  # Fixed number of boosting rounds
            valid_sets=valid_sets,
            valid_names=valid_names
        )
    
        return self
    
    def predict_proba(self, X):
        if self.model is None:
            raise Exception("Model not trained yet")
        return self.model.predict(X)
    
    def save(self, path):
        joblib.dump(self.model, path)
        
    def load(self, path):
        self.model = joblib.load(path)
        return self

# ------------------- Define Stacking Ensemble -------------------
class StackingEnsemble:
    """Stacking ensemble using LightGBM as meta-learner"""
    
    def __init__(self, base_models, meta_model_params):
        self.base_models = base_models
        self.meta_model_params = meta_model_params
        self.meta_model = None
        self.scalers = []  # Store one scaler per base model
        self.pcas = []     # Store one PCA per base model
    
    def _get_oof_predictions(self, X, y, n_splits=3):
        """Generate out-of-fold predictions for training the meta-learner"""
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        oof_preds = []
        
        for i, model in enumerate(self.base_models):
            print(f"Generating OOF predictions for {model.feature_name} model...")
            X_current = X[i]  # Get features for current model
            oof_pred = np.zeros((X_current.shape[0], len(np.unique(y))))
            
            # Create a scaler for this model's features
            self.scalers.append(StandardScaler())
            # We'll create PCA for each fold separately to determine appropriate components
            self.pcas.append(None)  # Placeholder, will be set during fold processing
            
            for fold, (train_idx, val_idx) in enumerate(kf.split(X_current, y)):
                print(f"Processing fold {fold+1}/{n_splits}")
                X_train_fold, X_val_fold = X_current[train_idx], X_current[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                # Scale features
                X_train_fold = self.scalers[i].fit_transform(X_train_fold)
                X_val_fold = self.scalers[i].transform(X_val_fold)
                
                # Determine maximum possible components for this fold
                max_possible_components = min(X_train_fold.shape[0], X_train_fold.shape[1])
                n_components = min(100, max_possible_components)
                print(f"Using {n_components} PCA components for fold {fold+1}")
                
                # Create and apply PCA for this fold
                pca = PCA(n_components=n_components)
                X_train_fold = pca.fit_transform(X_train_fold)
                X_val_fold = pca.transform(X_val_fold)
                
                # Store the PCA object for the last fold (we'll use it for final predictions)
                if fold == n_splits - 1:
                    self.pcas[i] = pca
                
                # Train model
                model_clone = BaseLGBMModel(model.params, model.feature_name)
                model_clone.train(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
                
                # Predict on validation fold
                oof_pred[val_idx] = model_clone.predict_proba(X_val_fold)
                
            oof_preds.append(oof_pred)
        
        # Combine all OOF predictions
        meta_features = np.hstack(oof_preds)
        return meta_features
        
    def _get_test_meta_features(self, X, X_test, y):
        """Generate meta-features for test data"""
        test_meta_features = []
        
        for i, model in enumerate(self.base_models):
            print(f"Training full {model.feature_name} model for test prediction...")
            X_current = X[i]  # Now X is properly passed as a parameter
            
            # Scale features using the corresponding scaler
            X_current_scaled = self.scalers[i].transform(X_current)
            X_test_scaled = self.scalers[i].transform(X_test[i])
            
            # Apply PCA
            X_current_pca = self.pcas[i].transform(X_current_scaled)
            X_test_pca = self.pcas[i].transform(X_test_scaled)
            
            # Train on full data and predict on test
            model_clone = BaseLGBMModel(model.params, model.feature_name)
            model_clone.train(X_current_pca, y)
            test_preds = model_clone.predict_proba(X_test_pca)
            test_meta_features.append(test_preds)
        
        # Combine all test predictions
        test_meta_features = np.hstack(test_meta_features)
        return test_meta_features
    
    def train(self, X, y, X_test=None, n_splits=3):  # Reduced from 5 to 3 folds
        """
        Train the stacking ensemble
        
        Args:
            X: List of feature matrices for each base model
            y: Target labels
            X_test: List of test feature matrices for each base model
            n_splits: Number of folds for cross-validation
        """
        # Generate meta-features through cross-validation
        start_time = time.time()
        print("Generating meta-features through cross-validation...")
        meta_features = self._get_oof_predictions(X, y, n_splits)
        print(f"Meta-features shape: {meta_features.shape}")
        cv_time = time.time() - start_time
        print(f"Time to generate meta-features: {cv_time:.2f} seconds")
        
        # Train meta-model on meta-features
        print("Training meta-model...")
        meta_train_data = lgb.Dataset(meta_features, label=y)
        
        # Create a validation set for the meta-model (20% of data)
        X_meta_train, X_meta_val, y_meta_train, y_meta_val = train_test_split(
            meta_features, y, test_size=0.2, random_state=42, stratify=y
        )
        
        meta_train_data = lgb.Dataset(X_meta_train, label=y_meta_train)
        meta_val_data = lgb.Dataset(X_meta_val, label=y_meta_val, reference=meta_train_data)
        
        self.meta_model = lgb.train(
            self.meta_model_params,
            meta_train_data,
            valid_sets=[meta_train_data, meta_val_data],
            valid_names=['train', 'valid'],
            num_boost_round=200  # Reduced from 1000 to 200
        )
        
        # Generate test meta-features if provided
        # Generate test meta-features if provided
        self.test_meta_features = None
        if X_test is not None:
            print("Generating test meta-features...")
            self.test_meta_features = self._get_test_meta_features(X, X_test, y)
            print(f"Test meta-features shape: {self.test_meta_features.shape}")
        
        return self
    
    def predict_proba(self, X=None):
        """
        Predict probabilities using the meta-model
        
        Args:
            X: If provided, generate meta-features from X and predict
               If None, use pre-computed test meta-features
        """
        if X is not None:
            # Generate meta-features from X
            # Note: we can't call this because we don't have y labels for prediction
            # meta_features = self._get_test_meta_features(X)
            raise ValueError("Prediction on new data not implemented. Use pre-computed test meta-features.")
        elif self.test_meta_features is not None:
            # Use pre-computed test meta-features
            meta_features = self.test_meta_features
        else:
            raise ValueError("No test meta-features available")
        
        return self.meta_model.predict(meta_features)
    def save(self, path_prefix):
        """Save all models, scalers and PCAs"""
        # Save meta-model
        meta_model_path = f"{path_prefix}_meta_model.pkl"
        joblib.dump(self.meta_model, meta_model_path)
        
        # Save scalers
        scaler_path = f"{path_prefix}_scalers.pkl"
        joblib.dump(self.scalers, scaler_path)
        
        # Save PCAs
        pca_path = f"{path_prefix}_pcas.pkl"
        joblib.dump(self.pcas, pca_path)
        
        # Save test meta-features if available
        if self.test_meta_features is not None:
            test_meta_features_path = f"{path_prefix}_test_meta_features.npy"
            np.save(test_meta_features_path, self.test_meta_features)
            
        print(f"Ensemble saved with prefix: {path_prefix}")
    
    def load(self, path_prefix):
        """Load all models, scalers and PCAs"""
        # Load meta-model
        meta_model_path = f"{path_prefix}_meta_model.pkl"
        self.meta_model = joblib.load(meta_model_path)
        
        # Load scalers
        scaler_path = f"{path_prefix}_scalers.pkl"
        self.scalers = joblib.load(scaler_path)
        
        # Load PCAs
        pca_path = f"{path_prefix}_pcas.pkl"
        self.pcas = joblib.load(pca_path)
        
        # Try to load test meta-features
        test_meta_features_path = f"{path_prefix}_test_meta_features.npy"
        if os.path.exists(test_meta_features_path):
            self.test_meta_features = np.load(test_meta_features_path)
        
        print(f"Ensemble loaded from prefix: {path_prefix}")
        return self

# ------------------- Parallelized Grid Search with CV -------------------
def evaluate_params(param_set, X, y, n_splits=3):
    """
    Evaluate a parameter set using k-fold cross-validation
    
    Args:
        param_set: Dictionary of LightGBM parameters
        X: Feature matrix
        y: Target labels
        n_splits: Number of folds
        
    Returns:
        Mean accuracy and parameters
    """
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []
    
    # Apply PCA once for all folds
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Get the maximum possible number of components
    max_components = min(X.shape[0], X.shape[1])
    
    # Adjust n_components if necessary
    actual_components = min(100, max_components)  # Original was hardcoded to 100
    if actual_components < 100:
        print(f"Warning: Using {actual_components} PCA components instead of 100")
    
    pca = PCA(n_components=actual_components)
    X_pca = pca.fit_transform(X_scaled)
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_pca, y)):
        X_train_fold, X_val_fold = X_pca[train_idx], X_pca[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
        val_data = lgb.Dataset(X_val_fold, label=y_val_fold, reference=train_data)
        
        # Train model
        model = lgb.train(
            param_set,
            train_data,
            valid_sets=[val_data],
            num_boost_round=200,  # Fixed number of rounds
            # Early stopping removed as requested
        )
        
        # Predict
        y_pred = model.predict(X_val_fold)
        y_pred_labels = np.argmax(y_pred, axis=1)
        
        # Calculate accuracy
        acc = accuracy_score(y_val_fold, y_pred_labels)
        accuracies.append(acc)
    
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    
    # Return mean accuracy and parameters
    return {
        'params': param_set,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc
    }
def grid_search_parallel(param_grid, X, y, n_splits=3, n_jobs=2):  # Reduced from 5 to 3 folds, limited to 2 jobs
    """
    Perform grid search with parallelization
    
    Args:
        param_grid: Dictionary of parameter lists
        X: Feature matrix
        y: Target labels
        n_splits: Number of folds
        n_jobs: Number of parallel jobs
        
    Returns:
        List of results sorted by mean accuracy
    """
    # Generate all parameter combinations
    param_combinations = []
    
    # Create base parameter dictionary with all constants
    base_params = {
        'objective': 'multiclass',
        'num_class': len(np.unique(y)),
        'metric': 'multi_logloss',
        'verbosity': -1,
        'seed': 42
    }
    
    # Recursive function to generate all parameter combinations
    def generate_combinations(param_grid, current_params=None, param_names=None, i=0):
        if current_params is None:
            current_params = base_params.copy()
        if param_names is None:
            param_names = list(param_grid.keys())
        
        if i >= len(param_names):
            param_combinations.append(current_params.copy())
            return
        
        param_name = param_names[i]
        param_values = param_grid[param_name]
        
        for value in param_values:
            current_params[param_name] = value
            generate_combinations(param_grid, current_params, param_names, i+1)
    
    # Generate all combinations
    generate_combinations(param_grid)
    print(f"Generated {len(param_combinations)} parameter combinations")
    
    # Evaluate each parameter combination in parallel
    start_time = time.time()
    results = Parallel(n_jobs=n_jobs)(  # Limited to 2 jobs
        delayed(evaluate_params)(params, X, y, n_splits) for params in param_combinations
    )
    
    # Sort results by mean accuracy (descending)
    results.sort(key=lambda x: x['mean_accuracy'], reverse=True)
    
    # Calculate total time
    total_time = time.time() - start_time
    print(f"Grid search completed in {total_time:.2f} seconds")
    
    return results

# ------------------- Apply PCA to features -------------------
def apply_pca_to_features(features, n_components=100):
    """Apply PCA to reduce feature dimensionality"""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Get the maximum possible number of components
    max_components = min(features.shape[0], features.shape[1])
    
    # Adjust n_components if necessary
    actual_components = min(n_components, max_components)
    if actual_components < n_components:
        print(f"Warning: Requested {n_components} components, but only {actual_components} are available.")
    
    pca = PCA(n_components=actual_components)
    features_pca = pca.fit_transform(features_scaled)
    explained_variance = np.sum(pca.explained_variance_ratio_) * 100
    print(f"Explained variance with {actual_components} components: {explained_variance:.2f}%")
    return features_pca, pca, scaler

# ------------------- Main Script -------------------

if __name__ == "__main__":
    print("Starting optimized LightGBM stacking ensemble with PCA and reduced parameter search...")
    start_time = time.time()
    
    # Create a timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"ensemble_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved to: {output_dir}")
    
    # ------------------- Load Feature Vectors -------------------
    print("\n🔍 Loading feature vectors...")
    
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
    
    # Print shapes for verification
    print(f"✅ BERT train shape: {bert_train.shape}")
    print(f"✅ RoBERTa train shape: {roberta_train.shape}")
    print(f"✅ BiLSTM train shape: {bilstm_train.shape}")
    
    # ------------------- Apply PCA to reduce dimensionality -------------------
    print("\n🔍 Applying PCA to reduce feature dimensionality...")
    
    # Apply PCA to BERT features
    print("Applying PCA to BERT features...")
    bert_train_pca, bert_pca, bert_scaler = apply_pca_to_features(bert_train, n_components=100)
    bert_test_pca = bert_pca.transform(bert_scaler.transform(bert_test))
    bert_valid_pca = bert_pca.transform(bert_scaler.transform(bert_valid))
    
    # Apply PCA to RoBERTa features
    print("Applying PCA to RoBERTa features...")
    roberta_train_pca, roberta_pca, roberta_scaler = apply_pca_to_features(roberta_train, n_components=100)
    roberta_test_pca = roberta_pca.transform(roberta_scaler.transform(roberta_test))
    roberta_valid_pca = roberta_pca.transform(roberta_scaler.transform(roberta_valid))
    
    # Apply PCA to BiLSTM features
    print("Applying PCA to BiLSTM features...")
    bilstm_train_pca, bilstm_pca, bilstm_scaler = apply_pca_to_features(bilstm_train, n_components=100)
    bilstm_test_pca = bilstm_pca.transform(bilstm_scaler.transform(bilstm_test))
    bilstm_valid_pca = bilstm_pca.transform(bilstm_scaler.transform(bilstm_valid))
    
    # Save PCA and scalers
    joblib.dump(bert_pca, f"{output_dir}/bert_pca.pkl")
    joblib.dump(bert_scaler, f"{output_dir}/bert_scaler.pkl")
    joblib.dump(roberta_pca, f"{output_dir}/roberta_pca.pkl")
    joblib.dump(roberta_scaler, f"{output_dir}/roberta_scaler.pkl")
    joblib.dump(bilstm_pca, f"{output_dir}/bilstm_pca.pkl")
    joblib.dump(bilstm_scaler, f"{output_dir}/bilstm_scaler.pkl")
    
    # Print reduced dimensions
    print(f"✅ BERT PCA train shape: {bert_train_pca.shape}")
    print(f"✅ RoBERTa PCA train shape: {roberta_train_pca.shape}")
    print(f"✅ BiLSTM PCA train shape: {bilstm_train_pca.shape}")
    
    # ------------------- Load Labels -------------------
    print("\n🔍 Loading labels...")
    y_train = np.loadtxt("/content/drive/MyDrive/feature_vectors/train/train_labels.txt", dtype=int)
    y_test = np.loadtxt("/content/drive/MyDrive/feature_vectors/test/test_labels.txt", dtype=int)
    y_valid = np.loadtxt("/content/drive/MyDrive/feature_vectors/valid/valid_labels.txt", dtype=int)
    
    # Print to verify shapes
    print(f"✅ y_train shape: {y_train.shape}")
    print(f"✅ y_test shape: {y_test.shape}")
    print(f"✅ y_valid shape: {y_valid.shape}")
    
    # ------------------- Grid Search for Base Models -------------------
    print("\n🔍 Starting grid search for base models with reduced parameter space...")
    
    # Significantly reduced parameter grid
    minimal_param_grid = {
        'boosting_type': ['gbdt'],  # Only use gbdt
        'learning_rate': [0.05],    # Only one learning rate
        'num_leaves': [20],         # Only one num_leaves value
        'max_depth': [5],           # Only one max_depth value
        'min_data_in_leaf': [50],   # Only one min_data_in_leaf value
        'lambda_l1': [0.5],         # Only one lambda_l1 value
        'lambda_l2': [0.5],         # Only one lambda_l2 value
        'feature_fraction': [0.8],  # Only one feature_fraction value
        'bagging_fraction': [0.8],  # Only one bagging_fraction value
        'bagging_freq': [5]         # Only one bagging_freq value
    }
    
    print("Running grid search for BERT features...")
    bert_results = grid_search_parallel(minimal_param_grid, bert_train_pca, y_train, n_splits=3, n_jobs=2)
    best_bert_params = bert_results[0]['params']
    print(f"Best BERT parameters: {best_bert_params}")
    print(f"Best BERT accuracy: {bert_results[0]['mean_accuracy']:.4f} ± {bert_results[0]['std_accuracy']:.4f}")
    
    print("\nRunning grid search for RoBERTa features...")
    roberta_results = grid_search_parallel(minimal_param_grid, roberta_train_pca, y_train, n_splits=3, n_jobs=2)
    best_roberta_params = roberta_results[0]['params']
    print(f"Best RoBERTa parameters: {best_roberta_params}")
    print(f"Best RoBERTa accuracy: {roberta_results[0]['mean_accuracy']:.4f} ± {roberta_results[0]['std_accuracy']:.4f}")
    
    print("\nRunning grid search for BiLSTM features...")
    bilstm_results = grid_search_parallel(minimal_param_grid, bilstm_train_pca, y_train, n_splits=3, n_jobs=2)
    best_bilstm_params = bilstm_results[0]['params']
    print(f"Best BiLSTM parameters: {best_bilstm_params}")
    print(f"Best BiLSTM accuracy: {bilstm_results[0]['mean_accuracy']:.4f} ± {bilstm_results[0]['std_accuracy']:.4f}")
    
    # Save base model grid search results
    with open(f"{output_dir}/base_model_results.json", "w") as f:
        json.dump({
            "bert": bert_results,
            "roberta": roberta_results,
            "bilstm": bilstm_results
        }, f, indent=4, default=str)
    
    # ------------------- Create Base Models with Best Parameters -------------------
    print("\n🔍 Creating base models with best parameters...")
    
    base_models = [
        BaseLGBMModel(best_bert_params, "BERT"),
        BaseLGBMModel(best_roberta_params, "RoBERTa"),
        BaseLGBMModel(best_bilstm_params, "BiLSTM")
    ]

# ------------------- Grid Search for Meta-Learner -------------------
print("\n🔍 Starting grid search for meta-learner...")

# Generate meta-features for grid search
def generate_meta_features_for_grid_search(base_models, X, y, X_val, y_val, n_splits=3):
    """Generate meta-features for tuning the meta-learner"""
    meta_features = []
    meta_features_val = []
    
    for i, model in enumerate(base_models):
        print(f"Generating meta-features for {model.feature_name} model...")
        
        # Train model on full training data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X[i])
        X_val_scaled = scaler.transform(X_val[i])
        
        # Apply PCA
        pca = PCA(n_components=min(100, min(X_scaled.shape[0], X_scaled.shape[1])))
        X_pca = pca.fit_transform(X_scaled)
        X_val_pca = pca.transform(X_val_scaled)
        
        # Train model
        model_clone = BaseLGBMModel(model.params, model.feature_name)
        model_clone.train(X_pca, y)
        
        # Get predictions for training and validation
        train_preds = model_clone.predict_proba(X_pca)
        val_preds = model_clone.predict_proba(X_val_pca)
        
        meta_features.append(train_preds)
        meta_features_val.append(val_preds)
    
    # Combine all predictions
    meta_features = np.hstack(meta_features)
    meta_features_val = np.hstack(meta_features_val)
    
    return meta_features, meta_features_val

    # Generate meta-features for grid search
    meta_features_train, meta_features_val = generate_meta_features_for_grid_search(
        base_models, X, y_train, X_valid, y_valid
    )
    print(f"Meta-features train shape: {meta_features_train.shape}")
    print(f"Meta-features validation shape: {meta_features_val.shape}")
    
    # Define a more comprehensive parameter grid for meta-learner
    meta_learner_param_grid = {
        'boosting_type': ['gbdt'],
        'learning_rate': [0.01, 0.05, 0.1],
        'num_leaves': [15, 31, 50],
        'max_depth': [3, 5, 7],
        'min_data_in_leaf': [20, 50, 100],
        'lambda_l1': [0.0, 0.5, 1.0],
        'lambda_l2': [0.0, 0.5, 1.0],
        'feature_fraction': [0.7, 0.8, 0.9],
        'bagging_fraction': [0.7, 0.8, 0.9],
        'bagging_freq': [5]
    }
    
    # Run grid search for meta-learner
    print("Running grid search for meta-learner...")
    meta_learner_results = grid_search_parallel(
        meta_learner_param_grid, meta_features_train, y_train, n_splits=3, n_jobs=2
    )
    best_meta_learner_params = meta_learner_results[0]['params']
    print(f"Best meta-learner parameters: {best_meta_learner_params}")
    print(f"Best meta-learner accuracy: {meta_learner_results[0]['mean_accuracy']:.4f} ± {meta_learner_results[0]['std_accuracy']:.4f}")
    
    # Save meta-learner grid search results
    with open(f"{output_dir}/meta_learner_results.json", "w") as f:
        json.dump({"meta_learner": meta_learner_results}, f, indent=4, default=str)

    
    # ------------------- Create and Train Stacking Ensemble -------------------
    print("\n🚀 Training stacking ensemble...")
    
    # Create list of features for each base model (use PCA-reduced features)
    X = [bert_train_pca, roberta_train_pca, bilstm_train_pca]
    X_test = [bert_test_pca, roberta_test_pca, bilstm_test_pca]
    X_valid = [bert_valid_pca, roberta_valid_pca, bilstm_valid_pca]
    
    # Create ensemble with base models
    ensemble = StackingEnsemble(base_models, best_bert_params)  # Use bert params as default for meta-learner
    
    # Train ensemble
    ensemble.train(X, y_train, X_test, n_splits=3)  # Reduced from 5 to 3 folds
    
    # Save ensemble
    ensemble.save(f"{output_dir}/stacking_ensemble")
    
    # ------------------- Evaluate Ensemble -------------------
    print("\n📊 Evaluating ensemble performance...")

    # Get predictions for test set (which was used during training)
    test_preds = ensemble.predict_proba()
    test_pred_labels = np.argmax(test_preds, axis=1)
    test_accuracy = accuracy_score(y_test, test_pred_labels)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Now train a separate ensemble for validation predictions
    print("Training ensemble for validation predictions...")
    ensemble_valid = StackingEnsemble(base_models, best_bert_params)
    ensemble_valid.train(X, y_train, X_valid, n_splits=3)
    valid_preds = ensemble_valid.predict_proba()
    valid_pred_labels = np.argmax(valid_preds, axis=1)
    valid_accuracy = accuracy_score(y_valid, valid_pred_labels)
    print(f"Validation Accuracy: {valid_accuracy:.4f}")
    
    # Generate classification report for test set
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, test_pred_labels))
    
    # Generate classification report for validation set
    print("\nClassification Report (Validation Set):")
    print(classification_report(y_valid, valid_pred_labels))
    
    # Save predictions
    np.save(f"{output_dir}/valid_pred_labels.npy", valid_pred_labels)
    np.save(f"{output_dir}/test_pred_labels.npy", test_pred_labels)
    
    # Save validation ensemble
    ensemble_valid.save(f"{output_dir}/stacking_ensemble_valid")
    
    # ------------------- Compare with Base Models -------------------
    print("\n📊 Comparing ensemble with individual base models...")
    
    # Train and evaluate BERT model
    print("Evaluating BERT model...")
    bert_model = BaseLGBMModel(best_bert_params, "BERT")
    bert_model.train(bert_train_pca, y_train)
    bert_preds = bert_model.predict_proba(bert_test_pca)
    bert_pred_labels = np.argmax(bert_preds, axis=1)
    bert_accuracy = accuracy_score(y_test, bert_pred_labels)
    
    # Train and evaluate RoBERTa model
    print("Evaluating RoBERTa model...")
    roberta_model = BaseLGBMModel(best_roberta_params, "RoBERTa")
    roberta_model.train(roberta_train_pca, y_train)
    roberta_preds = roberta_model.predict_proba(roberta_test_pca)
    roberta_pred_labels = np.argmax(roberta_preds, axis=1)
    roberta_accuracy = accuracy_score(y_test, roberta_pred_labels)
    
    # Train and evaluate BiLSTM model
    print("Evaluating BiLSTM model...")
    bilstm_model = BaseLGBMModel(best_bilstm_params, "BiLSTM")
    bilstm_model.train(bilstm_train_pca, y_train)
    bilstm_preds = bilstm_model.predict_proba(bilstm_test_pca)
    bilstm_pred_labels = np.argmax(bilstm_preds, axis=1)
    bilstm_accuracy = accuracy_score(y_test, bilstm_pred_labels)
    
    # Compare results
    print("\nModel Comparison:")
    print(f"BERT Model Accuracy: {bert_accuracy:.4f}")
    print(f"RoBERTa Model Accuracy: {roberta_accuracy:.4f}")
    print(f"BiLSTM Model Accuracy: {bilstm_accuracy:.4f}")
    print(f"Stacking Ensemble Accuracy: {test_accuracy:.4f}")
    
    # Save comparison results
    with open(f"{output_dir}/model_comparison.json", "w") as f:
        json.dump({
            "bert_accuracy": bert_accuracy,
            "roberta_accuracy": roberta_accuracy,
            "bilstm_accuracy": bilstm_accuracy,
            "ensemble_accuracy": test_accuracy
        }, f, indent=4)
    
    # ------------------- Calculate Total Time -------------------
    total_time = time.time() - start_time
    print(f"\n✅ Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"All results saved to: {output_dir}")
