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

# Page configuration
st.set_page_config(
    page_title="Breast Cancer Detection ML",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD SAVED MODELS AND COMPONENTS
# ============================================================================

@st.cache_resource
def load_models():
    """Load all pre-trained models from disk"""
    models = {}
    
    try:
        models['Manual LR'] = joblib.load('models/manual_lr.pkl')
        models['Logistic Regression'] = joblib.load('models/logistic_regression.pkl')
        models['Random Forest'] = joblib.load('models/random_forest.pkl')
        models['XGBoost'] = joblib.load('models/xgboost.pkl')
        
        feature_names = joblib.load('models/feature_names.pkl')
        scaler = joblib.load('models/scaler.pkl')
        
        return models, feature_names, scaler
    except FileNotFoundError as e:
        st.error(f"Error loading models: {e}")
        st.info("Please run: python train_models.py to train and save models first")
        return None, None, None

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

@st.cache_data
def load_dataset(file1, file2):
    """Load and combine datasets"""
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    combined = pd.concat([df1, df2], ignore_index=True).drop_duplicates()
    return combined

def preprocess_data(df):
    """Preprocess data"""
    df = df.drop('id', axis=1, errors='ignore')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.iloc[:, ~df.columns.duplicated()]
    df = df.dropna()
    
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis'].map({'M': 1, 'B': 0})
    return X, y

# ============================================================================
# EVALUATION METRICS
# ============================================================================

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate all evaluation metrics"""
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

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_confusion_matrix_viz(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'])
    ax.set_title(f'Confusion Matrix - {model_name}', fontweight='bold', fontsize=12)
    ax.set_ylabel('Actual Label')
    ax.set_xlabel('Predicted Label')
    return fig

def plot_roc_curve_viz(y_true, y_pred_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    return fig

def plot_pr_curve_viz(y_true, y_pred_proba, model_name):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='green', lw=2.5, label=f'PR curve (AUC = {pr_auc:.4f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve - {model_name}', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    return fig

def plot_feature_importance_viz(model, feature_names, model_name):
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importance = model.feature_importances_
    indices = np.argsort(importance)[-15:]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importance[indices], color='steelblue')
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top 15 Feature Importance - {model_name}', fontweight='bold', fontsize=12)
    return fig

def plot_prediction_distribution(y_true, y_pred_proba, model_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    benign = y_pred_proba[y_true == 0]
    malignant = y_pred_proba[y_true == 1]
    
    ax.hist(benign, bins=30, alpha=0.6, label='Benign', color='#2ECC71', edgecolor='black')
    ax.hist(malignant, bins=30, alpha=0.6, label='Malignant', color='#E74C3C', edgecolor='black')
    
    ax.set_xlabel('Predicted Probability of Malignancy')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of Predicted Probabilities - {model_name}', fontweight='bold', fontsize=12)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    return fig

# ============================================================================
# STREAMLIT APP PAGES
# ============================================================================

def page_home():
    st.markdown("# Hospital Breast Cancer Detection System")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("## About This System")
        st.write("""
        This application uses pre-trained machine learning models to detect breast cancer
        from clinical features extracted from biopsy samples.
        
        **Available Models:**
        - Manual Logistic Regression (from scratch)
        - Logistic Regression (scikit-learn)
        - Random Forest Classifier
        - XGBoost (Gradient Boosting)
        """)
    
    with col2:
        st.markdown("## System Statistics")
        st.info("""
        **Dataset:** Breast Cancer Wisconsin Diagnostic Dataset
        
        **Total Samples:** 569
        
        **Training/Test Split:** 80/20
        
        **Features:** 30 numerical attributes
        
        **Status:** Models Pre-trained and Ready
        """)

def page_dataset():
    st.markdown("## Dataset Analysis")
    
    try:
        combined_df = load_dataset('data.csv', 'breast_cancer.csv')
        X, y = preprocess_data(combined_df)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", len(combined_df))
        with col2:
            st.metric("Features", X.shape[1])
        with col3:
            st.metric("Benign Cases", sum(y == 0))
        with col4:
            st.metric("Malignant Cases", sum(y == 1))
        
        st.markdown("### Dataset Preview")
        st.dataframe(combined_df.head(10), use_container_width=True)
        
        st.markdown("### Class Distribution")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.pie([sum(y == 0), sum(y == 1)], labels=['Benign', 'Malignant'],
               autopct='%1.1f%%', colors=['#3498DB', '#E74C3C'], startangle=90)
        ax.set_title('Class Distribution', fontweight='bold')
        st.pyplot(fig)
        
        st.markdown("### Feature Statistics")
        st.dataframe(X.describe(), use_container_width=True)
        
    except FileNotFoundError:
        st.error("Please upload data.csv and breast_cancer.csv")

def page_model_test():
    st.markdown("## Model Testing & Evaluation")
    
    models, feature_names, scaler = load_models()
    
    if models is None:
        return
    
    # Load test data
    try:
        combined_df = load_dataset('data.csv', 'breast_cancer.csv')
        X, y = preprocess_data(combined_df)
        X_scaled = scaler.transform(X)
        
        st.info(f"Loaded {len(X)} samples for testing")
        
        # Select model
        selected_model_name = st.selectbox("Select Model for Testing:", list(models.keys()))
        selected_model = models[selected_model_name]
        
        # Get predictions
        y_pred = selected_model.predict(X_scaled)
        
        if hasattr(selected_model, 'predict_proba'):
            y_pred_proba = selected_model.predict_proba(X_scaled)[:, 1]
        else:
            y_pred_proba = None
        
        # Calculate metrics
        metrics = calculate_metrics(y, y_pred, y_pred_proba)
        
        # Display metrics
        st.markdown("### Performance Metrics")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.4f}")
        with col4:
            st.metric("Specificity", f"{metrics['specificity']:.4f}")
        with col5:
            st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
        
        if metrics['auc_score']:
            st.metric("AUC Score", f"{metrics['auc_score']:.4f}")
        
        # Visualizations
        st.markdown("### Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Confusion Matrix")
            fig = plot_confusion_matrix_viz(y, y_pred, selected_model_name)
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### Prediction Distribution")
            if y_pred_proba is not None:
                fig = plot_prediction_distribution(y, y_pred_proba, selected_model_name)
                st.pyplot(fig)
        
        # ROC and PR curves
        if y_pred_proba is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### ROC Curve")
                fig = plot_roc_curve_viz(y, y_pred_proba, selected_model_name)
                st.pyplot(fig)
            
            with col2:
                st.markdown("#### Precision-Recall Curve")
                fig = plot_pr_curve_viz(y, y_pred_proba, selected_model_name)
                st.pyplot(fig)
        
        # Feature importance
        if 'Random Forest' in selected_model_name or 'XGBoost' in selected_model_name:
            st.markdown("#### Feature Importance")
            fig = plot_feature_importance_viz(selected_model, feature_names, selected_model_name)
            if fig:
                st.pyplot(fig)
        
        # Classification report
        st.markdown("### Classification Report")
        st.text(classification_report(y, y_pred, target_names=['Benign', 'Malignant']))
        
    except FileNotFoundError:
        st.error("Please ensure data.csv and breast_cancer.csv are in the project folder")

def page_single_prediction():
    st.markdown("## Single Sample Prediction")
    
    models, feature_names, scaler = load_models()
    
    if models is None:
        return
    
    st.write("Enter feature values for a single patient to get predictions from all models")
    
    # Create input fields for features
    feature_values = []
    cols = st.columns(5)
    
    for idx, feature in enumerate(feature_names):
        with cols[idx % 5]:
            value = st.number_input(f"{feature}", value=0.0, key=feature)
            feature_values.append(value)
    
    if st.button("Predict", key="predict_button"):
        # Prepare data
        sample = np.array(feature_values).reshape(1, -1)
        sample_scaled = scaler.transform(sample)
        
        st.markdown("### Predictions from All Models")
        
        predictions_data = []
        
        for model_name, model in models.items():
            prediction = model.predict(sample_scaled)[0]
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(sample_scaled)[0]
                confidence = max(proba) * 100
            else:
                confidence = None
            
            predictions_data.append({
                'Model': model_name,
                'Prediction': 'Malignant' if prediction == 1 else 'Benign',
                'Confidence': f"{confidence:.2f}%" if confidence else "N/A"
            })
        
        df_predictions = pd.DataFrame(predictions_data)
        st.dataframe(df_predictions, use_container_width=True)
        
        # Determine consensus
        malignant_count = sum(1 for p in predictions_data if p['Prediction'] == 'Malignant')
        consensus = 'Malignant' if malignant_count >= 2 else 'Benign'
        
        st.markdown("### Consensus Result")
        if consensus == 'Malignant':
            st.error(f"Consensus: **{consensus}** ({malignant_count}/4 models)")
        else:
            st.success(f"Consensus: **{consensus}** ({4-malignant_count}/4 models)")

def page_model_comparison():
    st.markdown("## Model Comparison")
    
    models, feature_names, scaler = load_models()
    
    if models is None:
        return
    
    try:
        combined_df = load_dataset('data.csv', 'breast_cancer.csv')
        X, y = preprocess_data(combined_df)
        X_scaled = scaler.transform(X)
        
        st.markdown("### Performance Metrics Comparison")
        
        all_metrics = {}
        
        for model_name, model in models.items():
            y_pred = model.predict(X_scaled)
            y_pred_proba = model.predict_proba(X_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
            metrics = calculate_metrics(y, y_pred, y_pred_proba)
            all_metrics[model_name] = metrics
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Model': list(all_metrics.keys()),
            'Accuracy': [all_metrics[m]['accuracy'] for m in all_metrics.keys()],
            'Precision': [all_metrics[m]['precision'] for m in all_metrics.keys()],
            'Recall': [all_metrics[m]['recall'] for m in all_metrics.keys()],
            'Specificity': [all_metrics[m]['specificity'] for m in all_metrics.keys()],
            'F1-Score': [all_metrics[m]['f1_score'] for m in all_metrics.keys()],
        })
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualization
        st.markdown("### Metrics Comparison Chart")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        metrics_list = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score']
        colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']
        
        for idx, metric in enumerate(metrics_list):
            values = [all_metrics[m].get(metric, 0) for m in all_metrics.keys()]
            
            ax = axes[idx]
            bars = ax.bar(range(len(all_metrics)), values, color=colors[:len(all_metrics)], alpha=0.8)
            ax.set_xticks(range(len(all_metrics)))
            ax.set_xticklabels(list(all_metrics.keys()), rotation=45, ha='right')
            ax.set_ylabel(metric.capitalize())
            ax.set_title(f'{metric.capitalize()} Comparison', fontweight='bold')
            ax.set_ylim([0, 1.05])
            ax.grid(axis='y', alpha=0.3)
        
        fig.delaxes(axes[-1])
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("### Clinical Recommendations")
        st.info("""
        - **Highest Accuracy Model:** Provides overall best correctness
        - **Highest Recall Model:** Minimizes missed malignant cases (critical for screening)
        - **Highest Specificity Model:** Reduces unnecessary biopsies
        - **Ensemble Approach:** Using multiple models reduces individual model bias
        """)
        
    except FileNotFoundError:
        st.error("Please ensure data.csv and breast_cancer.csv are in the project folder")

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio(
            "Select Page:",
            ["Home", "Dataset Analysis", "Model Testing", "Single Prediction", "Model Comparison"]
        )
    
    if page == "Home":
        page_home()
    elif page == "Dataset Analysis":
        page_dataset()
    elif page == "Model Testing":
        page_model_test()
    elif page == "Single Prediction":
        page_single_prediction()
    elif page == "Model Comparison":
        page_model_comparison()

if __name__ == "__main__":
    main()