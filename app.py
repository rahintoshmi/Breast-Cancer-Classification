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

# ========================= STREAMLIT PAGE CONFIG =============================
st.set_page_config(page_title="🏥 Breast Cancer Detection", page_icon="💗", layout="wide")

# ========================= GLOBAL STYLE SETTINGS =============================
st.markdown("""
    <style>
    /* Background and container styling */
    .stApp {
        background-color: #f9fbfd;
        background-image: linear-gradient(to bottom right, #ffffff, #f2f6fc);
        color: #222;
    }

    /* Titles */
    h1, h2, h3 {
        color: #ff4b4b;
        font-weight: 700;
    }

    /* Metrics and buttons */
    [data-testid="stMetricValue"] {
        color: #0073e6 !important;
        font-weight: 700 !important;
    }

    div.stButton > button:first-child {
        background: linear-gradient(90deg, #ff4b4b, #ff7b7b);
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 0.6rem 1.5rem;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(90deg, #ff7b7b, #ff4b4b);
        transform: scale(1.03);
    }

    /* Radio buttons */
    .stRadio > div {
        justify-content: center;
    }

    /* Dataframes */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 0 15px rgba(0,0,0,0.05);
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #888;
        padding: 10px;
        font-size: 13px;
        margin-top: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# ========================= LOAD MODELS =============================
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
    st.error("🚨 Models not found!")
    st.warning("Make sure these files exist in the 'models/' folder:")
    st.code("""
models/
├── logistic_regression.pkl
├── random_forest.pkl
├── xgboost.pkl
├── scaler.pkl
└── feature_names.pkl
    """)
    st.stop()

# ========================= HELPER FUNCTIONS =============================
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

# ========================= VISUALIZATION =============================
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    ax.set_title(f'Confusion Matrix - {model_name}', fontweight='bold')
    return fig

def plot_roc_curve(y_true, y_pred_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr, color='#ff4b4b', lw=2.5, label=f'AUC = {roc_auc:.4f}')
    ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--')
    ax.set_title(f'ROC Curve - {model_name}', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    return fig

def plot_pr_curve(y_true, y_pred_proba, model_name):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(recall, precision, color='green', lw=2.5, label=f'AUC = {pr_auc:.4f}')
    ax.set_title(f'Precision-Recall Curve - {model_name}', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    return fig

# ========================= HEADER =============================
st.markdown("<h1 style='text-align:center;'>🏥 Breast Cancer Detection Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align:center; color:#777;'>Empowering Clinical Decisions with Machine Learning</h5>", unsafe_allow_html=True)
st.markdown("<hr style='margin: 20px 0;'>", unsafe_allow_html=True)

# ========================= MAIN APP (UNCHANGED LOGIC) =============================
#  💡 All your logic below stays **exactly the same**
#  (Load sample data, Single prediction, Batch testing, etc.)

# [PASTE YOUR EXISTING CODE BELOW THIS LINE UNCHANGED]
# everything from:
# mode = st.radio(...) 
# until the footer section

# ========================= FOOTER =============================
st.markdown("""
<hr>
<div class='footer'>
Made with ❤️ by <b>Rahin Toshmi Ohee</b>  
</div>
""", unsafe_allow_html=True)
