import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, auc,
                             precision_recall_curve, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Breast Cancer Detection",
    page_icon="🏥",
    layout="wide"
)

# ============================================================================
# LOAD MODELS
# ============================================================================

@st.cache_resource
def load_models():
    """Load all models"""
    try:
        models = {
            'Logistic Regression': joblib.load('models/logistic_regression.pkl'),
            'Random Forest': joblib.load('models/random_forest.pkl'),
            'XGBoost': joblib.load('models/xgboost.pkl')
        }
        
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        
        return models, scaler, feature_names, True
    
    except FileNotFoundError as e:
        return None, None, None, False

# Check if models exist
models, scaler, feature_names, models_exist = load_models()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    auc_score = roc_auc_score(y_true, y_pred_proba) if y_pred_proba is not None else None
    
    return {
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'specificity': specificity, 'f1_score': f1, 'auc_score': auc_score,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }

def preprocess_new_data(data, feature_names, scaler):
    """Preprocess new data"""
    data = data[feature_names]
    data_scaled = scaler.transform(data)
    return data_scaled

# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    ax.set_title('Confusion Matrix', fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    return fig

def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

st.markdown("# Breast Cancer Detection System")
st.markdown("---")

if not models_exist:
    st.error("ERROR: Models not found!")
    st.warning("Please run this first:")
    st.code("python train_models.py", language="bash")
    st.stop()

# Navigation
mode = st.radio(
    "Select Mode:",
    ["Single Patient Prediction", "Batch Testing"],
    horizontal=True
)

# ============================================================================
# PAGE 1: SINGLE PATIENT
# ============================================================================

if mode == "Single Patient Prediction":
    st.markdown("## Enter Patient Data")
    
    col1, col2 = st.columns(2)
    patient_data = {}
    
    for idx, feature in enumerate(feature_names):
        if idx % 2 == 0:
            with col1:
                patient_data[feature] = st.number_input(feature, value=0.0, step=0.1)
        else:
            with col2:
                patient_data[feature] = st.number_input(feature, value=0.0, step=0.1)
    
    if st.button("Get Prediction"):
        # Preprocess
        df_patient = pd.DataFrame([patient_data])
        patient_scaled = preprocess_new_data(df_patient, feature_names, scaler)
        
        st.markdown("---")
        st.markdown("## Results")
        
        results_data = []
        
        for model_name, model in models.items():
            pred = model.predict(patient_scaled)[0]
            proba = model.predict_proba(patient_scaled)[0]
            
            results_data.append({
                'Model': model_name,
                'Prediction': 'Malignant' if pred == 1 else 'Benign',
                'Confidence': f"{max(proba)*100:.2f}%",
                'Benign %': f"{proba[0]*100:.2f}%",
                'Malignant %': f"{proba[1]*100:.2f}%"
            })
        
        df_results = pd.DataFrame(results_data)
        st.dataframe(df_results, use_container_width=True)
        
        # Consensus
        malignant_count = sum(1 for r in results_data if r['Prediction'] == 'Malignant')
        
        if malignant_count >= 2:
            st.error(f"Consensus: MALIGNANT ({malignant_count}/3 models)")
        else:
            st.success(f"Consensus: BENIGN ({3-malignant_count}/3 models)")

# ============================================================================
# PAGE 2: BATCH TESTING
# ============================================================================

elif mode == "Batch Testing":
    st.markdown("## Upload Patient Data CSV")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.markdown("### Data Preview")
            st.write(f"Loaded {len(df)} patients")
            st.dataframe(df.head())
            
            # Check if has diagnosis
            has_diagnosis = 'diagnosis' in df.columns
            
            # Preprocess
            df_features = df[feature_names].copy()
            df_scaled = scaler.transform(df_features)
            
            # Get predictions
            results = []
            
            for idx in range(len(df)):
                sample = df_scaled[idx:idx+1]
                
                preds = {}
                for model_name, model in models.items():
                    pred = model.predict(sample)[0]
                    proba = model.predict_proba(sample)[0]
                    preds[f"{model_name}_Pred"] = 'Malignant' if pred == 1 else 'Benign'
                    preds[f"{model_name}_Conf"] = f"{max(proba)*100:.2f}%"
                
                results.append(preds)
            
            results_df = pd.DataFrame(results)
            
            st.markdown("### Predictions")
            st.dataframe(results_df, use_container_width=True)
            
            # If has diagnosis, calculate metrics
            if has_diagnosis:
                st.markdown("---")
                st.markdown("### Model Evaluation")
                
                y_true = df['diagnosis'].map({'M': 1, 'B': 0})
                
                metrics_list = []
                for model_name, model in models.items():
                    y_pred = model.predict(df_scaled)
                    y_proba = model.predict_proba(df_scaled)[:, 1]
                    
                    metrics = calculate_metrics(y_true, y_pred, y_proba)
                    metrics_list.append({
                        'Model': model_name,
                        'Accuracy': f"{metrics['accuracy']:.4f}",
                        'Precision': f"{metrics['precision']:.4f}",
                        'Recall': f"{metrics['recall']:.4f}",
                        'F1-Score': f"{metrics['f1_score']:.4f}"
                    })
                
                metrics_df = pd.DataFrame(metrics_list)
                st.dataframe(metrics_df, use_container_width=True)
                
                # Confusion matrices
                st.markdown("### Confusion Matrices")
                cols = st.columns(3)
                
                for idx, (model_name, model) in enumerate(models.items()):
                    y_pred = model.predict(df_scaled)
                    with cols[idx]:
                        fig = plot_confusion_matrix(y_true, y_pred)
                        st.pyplot(fig)
                
                # ROC curves
                st.markdown("### ROC Curves")
                cols = st.columns(3)
                
                for idx, (model_name, model) in enumerate(models.items()):
                    y_proba = model.predict_proba(df_scaled)[:, 1]
                    with cols[idx]:
                        fig = plot_roc_curve(y_true, y_proba)
                        st.pyplot(fig)
            
            # Download
            csv = results_df.to_csv(index=False)
            st.download_button(
                "Download Predictions",
                csv,
                "predictions.csv",
                "text/csv"
            )
        
        except Exception as e:
            st.error(f"Error: {e}")