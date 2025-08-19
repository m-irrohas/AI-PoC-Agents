#!/usr/bin/env python3
"""
Model Comparison Example
Compare multiple classification algorithms on the same dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import time
import argparse

def load_data(data_path, target_column=None):
    """Load and preprocess dataset"""
    df = pd.read_csv(data_path)
    
    if target_column is None:
        target_column = df.columns[-1]
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Handle categorical features
    categorical_columns = X.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    # Encode target
    le_target = LabelEncoder()
    y_encoded = le_target.fit_transform(y)
    
    return X, y_encoded

def compare_models(X_train, X_test, y_train, y_test):
    """Compare multiple classification models"""
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42),
        'Naive Bayes': GaussianNB()
    }
    
    results = {}
    
    print("Comparing models...")
    print("-" * 60)
    
    for name, model in models.items():
        # Time training
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Make predictions
        start_time = time.time()
        y_pred = model.predict(X_test)
        prediction_time = time.time() - start_time
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        results[name] = {
            'accuracy': accuracy,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'training_time': training_time,
            'prediction_time': prediction_time
        }
        
        print(f"{name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  CV Score: {cv_mean:.4f} (+/- {cv_std*2:.4f})")
        print(f"  Training time: {training_time:.4f}s")
        print(f"  Prediction time: {prediction_time:.4f}s")
        print()
    
    # Find best model
    best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
    print(f"Best model: {best_model} (Accuracy: {results[best_model]['accuracy']:.4f})")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Model Comparison for Classification')
    parser.add_argument('data_path', help='Path to the CSV dataset')
    parser.add_argument('--target-column', help='Name of the target column')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    
    args = parser.parse_args()
    
    print("Loading data...")
    X, y = load_data(args.data_path, args.target_column)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Compare models
    results = compare_models(X_train_scaled, X_test_scaled, y_train, y_test)

if __name__ == '__main__':
    main()