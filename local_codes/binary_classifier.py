#!/usr/bin/env python3
"""
Binary Classification Model Example
For datasets with 2 classes (e.g., normal vs anomaly)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import argparse

def load_and_preprocess_data(data_path, target_column=None):
    """Load and preprocess the dataset for binary classification"""
    df = pd.read_csv(data_path)
    
    print(f"Dataset shape: {df.shape}")
    
    # Auto-detect target column if not specified
    if target_column is None:
        target_column = df.columns[-1]
        print(f"Auto-detected target column: {target_column}")
    
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    print(f"Original classes: {y.unique()}")
    print(f"Class distribution:\n{y.value_counts()}")
    
    # Handle categorical features
    categorical_columns = X.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    # Encode target labels for binary classification
    if y.nunique() == 2:
        # Already binary
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
    else:
        # Convert multi-class to binary (normal vs attack)
        # Assume 'normal' is the positive class, everything else is attack
        y_binary = y.apply(lambda x: 1 if 'normal' in str(x).lower() else 0)
        y_encoded = y_binary
        print("Converted to binary: 1=normal, 0=attack")
    
    return X, y_encoded

def train_binary_classifier(X_train, y_train, model_type='rf'):
    """Train a binary classification model"""
    if model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'lr':
        model = LogisticRegression(random_state=42, max_iter=1000)
    else:
        raise ValueError("Unsupported model type")
    
    model.fit(X_train, y_train)
    return model

def evaluate_binary_model(model, X_test, y_test):
    """Evaluate the binary classification model"""
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Attack', 'Normal']))
    
    return accuracy, auc_score, y_pred, y_pred_proba

def main():
    parser = argparse.ArgumentParser(description='Binary Classification Model')
    parser.add_argument('data_path', help='Path to the CSV dataset')
    parser.add_argument('--target-column', help='Name of the target column')
    parser.add_argument('--model', choices=['rf', 'lr'], default='rf',
                       help='Model type: rf (Random Forest) or lr (Logistic Regression)')
    
    args = parser.parse_args()
    
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data(args.data_path, args.target_column)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training {args.model.upper()} model...")
    model = train_binary_classifier(X_train_scaled, y_train, args.model)
    
    print("Evaluating model...")
    accuracy, auc, predictions, probabilities = evaluate_binary_model(model, X_test_scaled, y_test)
    
    print(f"\nBinary classification completed!")
    print(f"Final accuracy: {accuracy:.4f}")
    print(f"Final AUC: {auc:.4f}")

if __name__ == '__main__':
    main()