import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(data_path, target_column=None):
    """Load and preprocess the dataset for classification."""
    df = pd.read_csv(data_path)
    
    # Auto-detect target column if not specified
    if target_column is None:
        target_column = df.columns[-1]
    
    # Separate features and target
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Handle categorical features
    categorical_columns = X.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
    
    return X, y

def train_model(X_train, y_train):
    """Train ensemble models and return them."""
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    rf_model.fit(X_train, y_train)
    gb_model.fit(X_train, y_train)
    
    return rf_model, gb_model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return metrics."""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    return accuracy, precision, recall, f1

def main():
    data_path = '/mnt/d/sakana_work/work5_v3/AI-PoC-Agents-v2/data/kdd_cup_99/train.csv'  # Hardcoded data path
    X, y = load_and_preprocess_data(data_path)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    rf_model, gb_model = train_model(X_train, y_train)
    
    # Evaluate Random Forest model
    rf_metrics = evaluate_model(rf_model, X_test, y_test)
    print(f"Random Forest Model Metrics:\nAccuracy: {rf_metrics[0]:.5f}, Precision: {rf_metrics[1]:.5f}, Recall: {rf_metrics[2]:.5f}, F1 Score: {rf_metrics[3]:.5f}")
    
    # Evaluate Gradient Boosting model
    gb_metrics = evaluate_model(gb_model, X_test, y_test)
    print(f"Gradient Boosting Model Metrics:\nAccuracy: {gb_metrics[0]:.5f}, Precision: {gb_metrics[1]:.5f}, Recall: {gb_metrics[2]:.5f}, F1 Score: {gb_metrics[3]:.5f}")

if __name__ == "__main__":
    main()