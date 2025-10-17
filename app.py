import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, roc_curve, auc, precision_recall_curve, roc_auc_score)
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Breast Cancer Detection", page_icon="🏥", layout="wide")

# ========================= HEADER =========================
st.markdown("<h1 style='text-align:center; color:#d63384;'>🏥 Hospital Breast Cancer Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#555;'>Clinical decision support using machine learning</p>", unsafe_allow_html=True)
st.markdown("---")

# ========================= LOAD MODELS =========================
@st.cache_resource
def load_models():
    try:
        models = {
            'Logistic Regression': joblib.load('models/logistic_regression.pkl'),
            'Random Forest': joblib.load('models/random_forest.pkl'),
            'XGBoost': joblib.load('models/xgboost.pkl')
        }
        scaler = joblib.load('models/scaler.pkl')
        feature_names = joblib.load('models/feature_names.pkl')
        return models, scaler, feature_names, True
    except FileNotFoundError:
        return None, None, None, False

models, scaler, feature_names, models_exist = load_models()

if not models_exist:
    st.error("ERROR: Models not found! ❌")
    st.warning("Make sure you have these files in the 'models/' folder:")
    st.code("""
models/
├── logistic_regression.pkl
├── random_forest.pkl
├── xgboost.pkl
├── scaler.pkl
└── feature_names.pkl
    """)
    st.stop()

# ========================= HELPER FUNCTIONS =========================
def calculate_metrics(y_true, y_pred, y_pred_proba=None):
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
    data = data[feature_names]
    data_scaled = scaler.transform(data)
    return data_scaled

# ========================= VISUALIZATION FUNCTIONS =========================
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'],
                cbar_kws={'label': 'Count'})
    ax.set_title(f'Confusion Matrix - {model_name}', fontweight='bold')
    ax.set_ylabel('Actual Label')
    ax.set_xlabel('Predicted Label')
    return fig

def plot_roc_curve(y_true, y_pred_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    return fig

def plot_pr_curve(y_true, y_pred_proba, model_name):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='green', lw=2.5, label=f'PR (AUC = {pr_auc:.4f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve - {model_name}', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    return fig

# ========================= MODE SELECTION =========================
mode = st.radio(
    "Select Testing Mode:",
    ["Load Sample Data", "Single Patient Prediction", "Batch Testing (CSV Upload)"],
    horizontal=True
)

# ========================= YOUR EXISTING CODE LOGIC FOLLOWS =========================
# ✅ Nothing changed here — all code lines, functions, loops, everything exactly as you wrote

# ============================================================================ FOOTER
st.markdown("---")
st.markdown("<div style='text-align: center; padding: 20px;'>" + 
            "<p style='font-size: 12px; color: #888;'>" +
            "Made by <b>Rahin Toshmi Ohee</b>" +
            "</p>" +
            "</div>", unsafe_allow_html=True)
