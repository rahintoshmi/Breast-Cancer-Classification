import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, classification_report, roc_curve, auc,
                             precision_recall_curve, log_loss, roc_auc_score)
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
    .main {
        padding: 0rem 0rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA LOADING AND PREPROCESSING FUNCTIONS
# ============================================================================

@st.cache_data
def load_and_combine_data(file1, file2):
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    combined = pd.concat([df1, df2], ignore_index=True).drop_duplicates()
    return combined

@st.cache_data
def preprocess_data(df):
    df = df.drop('id', axis=1, errors='ignore')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.iloc[:, ~df.columns.duplicated()]
    df = df.dropna()
    
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis'].map({'M': 1, 'B': 0})
    return X, y

# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def forward_backward_propagation(w, b, x_train, y_train):
    m = x_train.shape[0]
    z = np.dot(x_train, w) + b
    y_head = sigmoid(z)
    
    y_train_2d = y_train.values.reshape(-1, 1) if hasattr(y_train, 'values') else y_train.reshape(-1, 1)
    
    cost = (-1/m) * np.sum(y_train_2d * np.log(y_head + 1e-8) + (1 - y_train_2d) * np.log(1 - y_head + 1e-8))
    
    derivative_weight = (1/m) * np.dot(x_train.T, (y_head - y_train_2d))
    derivative_bias = (1/m) * np.sum(y_head - y_train_2d)
    
    return cost, {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}

def update_parameters(w, b, x_train, y_train, learning_rate, num_iterations):
    costs = []
    
    for i in range(num_iterations):
        cost, gradients = forward_backward_propagation(w, b, x_train, y_train)
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        
        if i % 100 == 0:
            costs.append(cost)
    
    return {"weight": w, "bias": b}, costs

def predict_logistic(w, b, x_test):
    m = x_test.shape[0]
    y_prediction = np.zeros((m, 1))
    z = sigmoid(np.dot(x_test, w) + b)
    
    for i in range(m):
        y_prediction[i, 0] = 1 if z[i, 0] > 0.5 else 0
    
    return y_prediction.flatten()

@st.cache_resource
def train_all_models(X_train, X_test, y_train, y_test):
    models = {}
    
    # Manual Logistic Regression
    w = np.random.randn(X_train.shape[1], 1) * 0.01
    b = 0.0
    params, costs = update_parameters(w, b, X_train, y_train, 0.01, 2000)
    manual_pred = predict_logistic(params["weight"], params["bias"], X_test)
    
    models['Manual LR'] = {
        'predictions': manual_pred,
        'probabilities': None,
        'costs': costs
    }
    
    # Sklearn Models
    sklearn_models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    for name, model in sklearn_models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        
        models[name] = {
            'model': model,
            'predictions': pred,
            'probabilities': proba
        }
    
    return models

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    auc_score = roc_auc_score(y_true, y_pred_proba) if y_pred_proba is not None else None
    log_loss_val = log_loss(y_true, y_pred_proba) if y_pred_proba is not None else None
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1_score': f1,
        'auc_score': auc_score,
        'log_loss': log_loss_val,
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
    ax.set_title(f'Confusion Matrix - {model_name}', fontweight='bold')
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
    ax.set_title(f'ROC Curve - {model_name}', fontweight='bold')
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
    ax.set_title(f'Precision-Recall Curve - {model_name}', fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    return fig

def plot_feature_importance_viz(model, feature_names, model_name):
    importance = model.feature_importances_
    indices = np.argsort(importance)[-15:]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importance[indices])
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top 15 Feature Importance - {model_name}', fontweight='bold')
    return fig

def plot_cost_descent(costs, model_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(len(costs)), costs, linewidth=2, color='#2E86AB', marker='o')
    ax.set_xlabel('Iteration (x100)')
    ax.set_ylabel('Cost (Binary Cross-Entropy Loss)')
    ax.set_title(f'Cost Function Descent - {model_name}', fontweight='bold')
    ax.grid(alpha=0.3)
    return fig

def plot_prediction_distribution(y_true, y_pred_proba, model_name):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    benign = y_pred_proba[y_true == 0]
    malignant = y_pred_proba[y_true == 1]
    
    ax.hist(benign, bins=30, alpha=0.6, label='Benign', color='#2ECC71', edgecolor='black')
    ax.hist(malignant, bins=30, alpha=0.6, label='Malignant', color='#E74C3C', edgecolor='black')
    
    ax.set_xlabel('Predicted Probability of Malignancy')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Distribution of Predicted Probabilities - {model_name}', fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    return fig

def plot_metrics_comparison(all_results):
    models = list(all_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score']
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    colors = ['#3498DB', '#E74C3C', '#2ECC71', '#F39C12']
    
    for idx, metric in enumerate(metrics):
        values = [all_results[m].get(metric, 0) for m in models]
        
        ax = axes[idx]
        bars = ax.bar(models, values, color=colors[:len(models)], alpha=0.8, edgecolor='black')
        
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} Comparison', fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.03, f'{v:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    fig.delaxes(axes[-1])
    plt.tight_layout()
    return fig

# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================

def main():
    # Header
    st.markdown("# 🏥 Breast Cancer Detection using Machine Learning")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.markdown("## Navigation")
        page = st.radio(
            "Select Page:",
            ["Home", "Dataset Analysis", "Model Training", "Model Evaluation", "Predictions", "Comparison"]
        )
    
    # Home Page
    if page == "Home":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("## About This Project")
            st.write("""
            This application demonstrates the use of machine learning to detect breast cancer
            using clinical features extracted from biopsy samples.
            
            **Models Used:**
            - Manual Logistic Regression (from scratch)
            - Logistic Regression (scikit-learn)
            - Random Forest Classifier
            - XGBoost (Gradient Boosting)
            
            **Key Features:**
            - 30 numerical features from cell nuclei
            - 569 patient records
            - Binary classification (Benign/Malignant)
            """)
        
        with col2:
            st.markdown("## Project Statistics")
            st.info("""
            **Dataset:** Breast Cancer Wisconsin Diagnostic Dataset
            
            **Total Samples:** 569
            
            **Training/Test Split:** 80/20
            
            **Features:** 30 numerical attributes
            
            **Target Classes:** Benign, Malignant
            """)
    
    # Dataset Analysis
    elif page == "Dataset Analysis":
        st.markdown("## Dataset Analysis")
        
        try:
            combined_df = load_and_combine_data('data.csv', 'breast_cancer.csv')
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
            st.dataframe(combined_df.head(10))
            
            st.markdown("### Class Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.pie([sum(y == 0), sum(y == 1)], labels=['Benign', 'Malignant'],
                   autopct='%1.1f%%', colors=['#3498DB', '#E74C3C'])
            ax.set_title('Class Distribution')
            st.pyplot(fig)
            
            st.markdown("### Feature Statistics")
            st.dataframe(X.describe())
            
        except FileNotFoundError:
            st.error("⚠️ Please upload data.csv and breast_cancer.csv")
    
    # Model Training
    elif page == "Model Training":
        st.markdown("## Model Training")
        
        try:
            combined_df = load_and_combine_data('data.csv', 'breast_cancer.csv')
            X, y = preprocess_data(combined_df)
            
            # Scaling
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            st.info(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
            
            # Train models
            with st.spinner("Training models... This may take a minute..."):
                models = train_all_models(X_train, X_test, y_train, y_test)
            
            st.success("✓ All models trained successfully!")
            
            # Store in session state for other pages
            st.session_state.models = models
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.X_train = X_train
            st.session_state.y_train = y_train
            st.session_state.feature_names = X.columns.tolist()
            
            st.markdown("### Models Trained:")
            for model_name in models.keys():
                st.write(f"✓ {model_name}")
        
        except FileNotFoundError:
            st.error("⚠️ Please upload data.csv and breast_cancer.csv")
    
    # Model Evaluation
    elif page == "Model Evaluation":
        st.markdown("## Model Evaluation")
        
        if 'models' not in st.session_state:
            st.warning("Please train models first on the 'Model Training' page")
            return
        
        models = st.session_state.models
        y_test = st.session_state.y_test
        
        # Calculate all metrics
        all_results = {}
        for model_name, model_data in models.items():
            metrics = calculate_metrics(y_test, model_data['predictions'], model_data['probabilities'])
            all_results[model_name] = metrics
        
        st.session_state.all_results = all_results
        
        # Display metrics table
        st.markdown("### Performance Metrics Summary")
        
        metrics_df = pd.DataFrame({
            'Model': list(all_results.keys()),
            'Accuracy': [all_results[m]['accuracy'] for m in all_results.keys()],
            'Precision': [all_results[m]['precision'] for m in all_results.keys()],
            'Recall': [all_results[m]['recall'] for m in all_results.keys()],
            'Specificity': [all_results[m]['specificity'] for m in all_results.keys()],
            'F1-Score': [all_results[m]['f1_score'] for m in all_results.keys()],
        })
        
        st.dataframe(metrics_df, use_container_width=True)
        
        # Individual model details
        st.markdown("### Detailed Metrics by Model")
        
        selected_model = st.selectbox("Select Model:", list(models.keys()))
        
        col1, col2, col3 = st.columns(3)
        
        metrics = all_results[selected_model]
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
            st.metric("Precision", f"{metrics['precision']:.4f}")
        with col2:
            st.metric("Recall", f"{metrics['recall']:.4f}")
            st.metric("Specificity", f"{metrics['specificity']:.4f}")
        with col3:
            st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
            if metrics['auc_score']:
                st.metric("AUC Score", f"{metrics['auc_score']:.4f}")
        
        st.markdown("#### Classification Report")
        st.text(classification_report(y_test, models[selected_model]['predictions'],
                                      target_names=['Benign', 'Malignant']))
    
    # Predictions & Visualizations
    elif page == "Predictions":
        st.markdown("## Visualizations & Predictions")
        
        if 'models' not in st.session_state:
            st.warning("Please train models first on the 'Model Training' page")
            return
        
        models = st.session_state.models
        y_test = st.session_state.y_test
        
        selected_model = st.selectbox("Select Model for Visualization:", list(models.keys()))
        
        # Confusion Matrix
        st.markdown("### Confusion Matrix")
        fig = plot_confusion_matrix_viz(y_test, models[selected_model]['predictions'], selected_model)
        st.pyplot(fig)
        
        # Cost descent for Manual LR
        if selected_model == 'Manual LR':
            st.markdown("### Cost Function Descent")
            fig = plot_cost_descent(models[selected_model]['costs'], selected_model)
            st.pyplot(fig)
        
        # ROC and PR curves
        if models[selected_model]['probabilities'] is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### ROC Curve")
                fig = plot_roc_curve_viz(y_test, models[selected_model]['probabilities'], selected_model)
                st.pyplot(fig)
            
            with col2:
                st.markdown("### Precision-Recall Curve")
                fig = plot_pr_curve_viz(y_test, models[selected_model]['probabilities'], selected_model)
                st.pyplot(fig)
            
            st.markdown("### Prediction Probability Distribution")
            fig = plot_prediction_distribution(y_test, models[selected_model]['probabilities'], selected_model)
            st.pyplot(fig)
        
        # Feature importance
        if selected_model != 'Manual LR' and 'model' in models[selected_model]:
            model = models[selected_model]['model']
            if hasattr(model, 'feature_importances_'):
                st.markdown("### Feature Importance")
                fig = plot_feature_importance_viz(model, st.session_state.feature_names, selected_model)
                st.pyplot(fig)
    
    # Model Comparison
    elif page == "Comparison":
        st.markdown("## Model Comparison")
        
        if 'all_results' not in st.session_state:
            st.warning("Please evaluate models first on the 'Model Evaluation' page")
            return
        
        all_results = st.session_state.all_results
        
        st.markdown("### Comprehensive Metrics Comparison")
        fig = plot_metrics_comparison(all_results)
        st.pyplot(fig)
        
        st.markdown("### Key Findings")
        
        best_model = max(all_results.items(), key=lambda x: x[1]['f1_score'])
        best_recall = max(all_results.items(), key=lambda x: x[1]['recall'])
        best_specificity = max(all_results.items(), key=lambda x: x[1]['specificity'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info(f"**Best F1-Score:** {best_model[0]}\n\n{best_model[1]['f1_score']:.4f}")
        
        with col2:
            st.success(f"**Best Recall (Sensitivity):** {best_recall[0]}\n\n{best_recall[1]['recall']:.4f}")
        
        with col3:
            st.warning(f"**Best Specificity:** {best_specificity[0]}\n\n{best_specificity[1]['specificity']:.4f}")
        
        st.markdown("### Clinical Recommendations")
        st.info("""
        - **High Recall is Critical:** Minimizes false negatives (missed malignant cases)
        - **High Specificity:** Reduces unnecessary biopsies (false positives)
        - **Balanced Model:** Optimal for clinical utility
        """)

if __name__ == "__main__":
    main()