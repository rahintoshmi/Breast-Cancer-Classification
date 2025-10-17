import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

def load_and_combine_data():
    """Load and combine datasets"""
    print("Loading datasets...")
    
    try:
        df1 = pd.read_csv('data/data.csv')
        df2 = pd.read_csv('data/breast_cancer.csv')
        
        print(f"Dataset 1: {df1.shape}")
        print(f"Dataset 2: {df2.shape}")
        
        # Combine and remove duplicates
        combined = pd.concat([df1, df2], ignore_index=True).drop_duplicates()
        print(f"Combined (after removing duplicates): {combined.shape}\n")
        
        return combined
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Make sure data/data.csv and data/breast_cancer.csv exist")
        return None

def preprocess_data(df):
    """Preprocess data"""
    print("Preprocessing data...")
    
    # Clean columns
    df = df.drop('id', axis=1, errors='ignore')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.iloc[:, ~df.columns.duplicated()]
    df = df.dropna()
    
    # Separate features and target
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis'].map({'M': 1, 'B': 0})
    
    print(f"Dataset shape: {X.shape}")
    print(f"Benign: {sum(y == 0)}, Malignant: {sum(y == 1)}\n")
    
    return X, y

# ============================================================================
# TRAINING
# ============================================================================

def train_and_save_models():
    """Complete training pipeline"""
    
    print("=" * 80)
    print("BREAST CANCER DETECTION - MODEL TRAINING")
    print("=" * 80 + "\n")
    
    # Create models directory
    if not os.path.exists('models'):
        os.makedirs('models')
        print("Created 'models' directory\n")
    
    # Load data
    combined_df = load_and_combine_data()
    if combined_df is None:
        return False
    
    # Preprocess
    X, y = preprocess_data(combined_df)
    
    # Split data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}\n")
    
    # Scale
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Scaling complete\n")
    
    # Train models
    print("=" * 80)
    print("TRAINING MODELS")
    print("=" * 80 + "\n")
    
    print("1. Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, lr_pred)
    print(f"   ✓ Accuracy: {lr_acc:.4f}\n")
    
    print("2. Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"   ✓ Accuracy: {rf_acc:.4f}\n")
    
    print("3. Training XGBoost...")
    xgb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    xgb.fit(X_train_scaled, y_train)
    xgb_pred = xgb.predict(X_test_scaled)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    print(f"   ✓ Accuracy: {xgb_acc:.4f}\n")
    
    # Save models
    print("=" * 80)
    print("SAVING MODELS")
    print("=" * 80 + "\n")
    
    try:
        joblib.dump(lr, 'models/logistic_regression.pkl')
        print("✓ Saved: models/logistic_regression.pkl")
        
        joblib.dump(rf, 'models/random_forest.pkl')
        print("✓ Saved: models/random_forest.pkl")
        
        joblib.dump(xgb, 'models/xgboost.pkl')
        print("✓ Saved: models/xgboost.pkl")
        
        joblib.dump(scaler, 'models/scaler.pkl')
        print("✓ Saved: models/scaler.pkl")
        
        joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')
        print("✓ Saved: models/feature_names.pkl\n")
        
        print("=" * 80)
        print("SUCCESS! All models saved.")
        print("Now run: streamlit run app.py")
        print("=" * 80 + "\n")
        
        return True
    
    except Exception as e:
        print(f"ERROR saving models: {e}")
        return False

if __name__ == "__main__":
    train_and_save_models()
