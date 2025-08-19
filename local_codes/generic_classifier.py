#!/usr/bin/env python3
"""
Generic Classification Model Example
Works with any CSV dataset for multi-class classification
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import argparse

def load_and_preprocess_data(data_path, target_column=None):
    """Load and preprocess the dataset"""
    # Load data
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Auto-detect target column if not specified
    if target_column is None:
        # Assume last column is target
        target_column = df.columns[-1]
        print(f"Auto-detected target column: {target_column}")
    
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {y.nunique()}")
    print(f"Class distribution:\n{y.value_counts()}")
    
    # Handle categorical features
    categorical_columns = X.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
        print(f"Encoded categorical column: {col}")
    
    # Encode target labels
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    
    return X, y_encoded, target_encoder, label_encoders

def train_classifier(X_train, y_train, model_type='rf'):
    """Train a classification model"""
    models = {
        'rf': RandomForestClassifier(n_estimators=100, random_state=42),
        'gb': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'lr': LogisticRegression(random_state=42, max_iter=1000),
        'svm': SVC(random_state=42, probability=True)
    }
    
    if model_type not in models:
        raise ValueError(f"Unsupported model type. Choose from: {list(models.keys())}")
    
    model = models[model_type]
    print(f"Training {model_type.upper()} model...")
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, target_encoder):
    """Evaluate the trained model"""
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Get class names
    class_names = target_encoder.classes_
    
    # Print results
    print(f"\nAccuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    return accuracy, y_pred

def main():
    parser = argparse.ArgumentParser(description='Generic Classification Model')
    parser.add_argument('data_path', help='Path to the CSV dataset')
    parser.add_argument('--target-column', help='Name of the target column (default: last column)')
    parser.add_argument('--model', choices=['rf', 'gb', 'lr', 'svm'], default='rf',
                       help='Model type: rf (Random Forest), gb (Gradient Boosting), lr (Logistic Regression), svm (SVM)')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (default: 0.2)')
    parser.add_argument('--no-scaling', action='store_true',
                       help='Skip feature scaling')
    
    args = parser.parse_args()
    
    print("Loading and preprocessing data...")
    X, y, target_encoder, feature_encoders = load_and_preprocess_data(args.data_path, args.target_column)
    
    print("\nSplitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )
    
    # Feature scaling (optional)
    if not args.no_scaling:
        print("Scaling features...")
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train
        X_test_scaled = X_test
        print("Skipping feature scaling...")
    
    # Train model
    model = train_classifier(X_train_scaled, y_train, args.model)
    
    # Evaluate model
    print("\nEvaluating model...")
    accuracy, predictions = evaluate_model(model, X_test_scaled, y_test, target_encoder)
    
    print(f"\nModel training completed successfully!")
    print(f"Final accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()