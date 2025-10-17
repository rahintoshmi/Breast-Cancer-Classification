import sys
import os
sys.path.append('..')

import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report

# Import preprocessor
from scripts.preprocess import DataPreprocessor

def create_models_directory():
    """Create models directory if it doesn't exist"""
    if not os.path.exists('models'):
        os.makedirs('models')
        print("✓ Created 'models' directory\n")
    else:
        print("✓ 'models' directory exists\n")

def train_models(X_train, X_test, y_train, y_test):
    """Train all three models"""
    print("\n" + "=" * 80)
    print("TRAINING MODELS")
    print("=" * 80)
    
    models = {}
    
    # Model 1: Logistic Regression
    print("\n1. Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
    lr.fit(X_train, y_train)
    models['logistic_regression'] = {
        'model': lr,
        'name': 'Logistic Regression'
    }
    print("   ✓ Logistic Regression trained")
    
    # Model 2: Random Forest
    print("\n2. Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['random_forest'] = {
        'model': rf,
        'name': 'Random Forest'
    }
    print("   ✓ Random Forest trained")
    
    # Model 3: XGBoost (Gradient Boosting)
    print("\n3. Training XGBoost...")
    xgb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    xgb.fit(X_train, y_train)
    models['xgboost'] = {
        'model': xgb,
        'name': 'XGBoost'
    }
    print("   ✓ XGBoost trained")
    
    return models

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n{model_name}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def save_models(models, scaler, feature_names):
    """Save all models and components to disk"""
    print("\n" + "=" * 80)
    print("SAVING MODELS AND COMPONENTS")
    print("=" * 80)
    
    # Save each model
    for model_key, model_data in models.items():
        filename = f'models/{model_key}.pkl'
        joblib.dump(model_data['model'], filename)
        print(f"\n✓ Saved: {filename}")
    
    # Save scaler
    scaler_filename = 'models/scaler.pkl'
    joblib.dump(scaler, scaler_filename)
    print(f"✓ Saved: {scaler_filename}")
    
    # Save feature names
    features_filename = 'models/feature_names.pkl'
    joblib.dump(feature_names, features_filename)
    print(f"✓ Saved: {features_filename}")
    
    print("\n" + "=" * 80)
    print("ALL MODELS SAVED SUCCESSFULLY!")
    print("=" * 80)

def print_summary(models, X_test, y_test):
    """Print final summary"""
    print("\n" + "=" * 80)
    print("FINAL MODEL PERFORMANCE SUMMARY")
    print("=" * 80)
    
    print(f"\n{'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 73)
    
    for model_key, model_data in models.items():
        metrics = evaluate_model(model_data['model'], X_test, y_test, model_data['name'])
        print(f"{model_data['name']:<25} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
              f"{metrics['recall']:<12.4f} {metrics['f1_score']:<12.4f}")

def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("BREAST CANCER DETECTION - MODEL TRAINING AND SAVING")
    print("=" * 80)
    
    # Step 1: Create models directory
    create_models_directory()
    
    # Step 2: Preprocess data
    preprocessor = DataPreprocessor(data_dir='data')
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocessor.preprocess_pipeline(
        file1='data.csv',
        file2='breast_cancer.csv',
        test_size=0.2
    )
    
    if X_train is None:
        print("\nError: Could not preprocess data. Exiting.")
        return
    
    # Step 3: Train models
    models = train_models(X_train, X_test, y_train, y_test)
    
    # Step 4: Evaluate models
    print("\n" + "=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)
    print_summary(models, X_test, y_test)
    
    # Step 5: Save models
    save_models(models, scaler, feature_names)
    
    print("\n" + "=" * 80)
    print("MODELS READY FOR DEPLOYMENT")
    print("You can now run: streamlit run app.py")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()