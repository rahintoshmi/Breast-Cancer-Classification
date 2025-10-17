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
    page_title="Breast Cancer Detection - Clinical Tool",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD MODELS
# ============================================================================

@st.cache_resource
def load_models():
    """Load all pre-trained models"""
    models = {}
    
    try:
        models['Logistic Regression'] = joblib.load('models/logistic_regression.pkl')
        models['Random Forest'] = joblib.load('models/random_forest.pkl')
        models['XGBoost'] = joblib.load('models/xgboost.pkl')
        
        feature_names = joblib.load('models/feature_names.pkl')
        scaler = joblib.load('models/scaler.pkl')
        
        return models, feature_names, scaler, True
    except FileNotFoundError as e:
        return None, None, None, False

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate evaluation metrics"""
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
    """Preprocess new patient data"""
    # Ensure correct feature order
    data = data[feature_names]
    
    # Scale data
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
                yticklabels=['Benign', 'Malignant'],
                cbar_kws={'label': 'Count'})
    ax.set_title('Confusion Matrix', fontweight='bold', fontsize=14)
    ax.set_ylabel('Actual Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    return fig

def plot_roc_curve(y_true, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('ROC Curve', fontweight='bold', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    return fig, roc_auc

def plot_pr_curve(y_true, y_pred_proba):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='green', lw=2.5, label=f'PR (AUC = {pr_auc:.4f})')
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('Precision-Recall Curve', fontweight='bold', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    return fig

# ============================================================================
# PAGE 1: HOME
# ============================================================================

def page_home():
    st.markdown("# Hospital Breast Cancer Detection System")
    st.markdown("A clinical decision support tool using machine learning")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("## About")
        st.write("""
        This is a **real-world testing application** for breast cancer detection.
        
        It uses pre-trained machine learning models to:
        - Predict cancer type from 30 numerical features
        - Provide confidence scores
        - Compare multiple models
        - Help clinical decision-making
        
        **Three Models Available:**
        1. Logistic Regression
        2. Random Forest
        3. XGBoost
        """)
    
    with col2:
        st.markdown("## Quick Start")
        st.info("""
        **Choose your testing method:**
        
        1. **Single Patient** - Enter one patient's data
        
        2. **Batch Testing** - Upload CSV with multiple patients
        
        3. **Model Comparison** - Compare all models on new data
        
        4. **Ground Truth** - Test with actual labels (validation)
        """)

# ============================================================================
# PAGE 2: SINGLE PATIENT PREDICTION
# ============================================================================

def page_single_prediction():
    st.markdown("## Single Patient Prediction")
    st.markdown("Enter patient data to get predictions")
    st.markdown("---")
    
    models, feature_names, scaler, loaded = load_models()
    
    if not loaded:
        st.error("Models not loaded. Run: python scripts/train_models.py")
        return
    
    # Create two columns for data input
    col1, col2 = st.columns(2)
    
    patient_data = {}
    
    # Display input fields for all 30 features
    for idx, feature in enumerate(feature_names):
        if idx % 2 == 0:
            with col1:
                patient_data[feature] = st.number_input(
                    feature, 
                    value=0.0, 
                    step=0.1,
                    key=f"single_{feature}"
                )
        else:
            with col2:
                patient_data[feature] = st.number_input(
                    feature, 
                    value=0.0, 
                    step=0.1,
                    key=f"single_{feature}"
                )
    
    # Prediction button
    if st.button("Get Prediction", key="single_predict"):
        st.markdown("---")
        
        # Prepare data
        df_patient = pd.DataFrame([patient_data])
        patient_scaled = preprocess_new_data(df_patient, feature_names, scaler)
        
        # Get predictions from all models
        predictions = {}
        st.markdown("### Results from All Models")
        
        for model_name, model in models.items():
            pred = model.predict(patient_scaled)[0]
            proba = model.predict_proba(patient_scaled)[0]
            confidence = max(proba) * 100
            
            predictions[model_name] = {
                'prediction': pred,
                'confidence': confidence,
                'benign_prob': proba[0] * 100,
                'malignant_prob': proba[1] * 100
            }
        
        # Display results
        results_df = pd.DataFrame({
            'Model': list(predictions.keys()),
            'Prediction': ['Malignant' if predictions[m]['prediction'] == 1 else 'Benign' 
                          for m in predictions.keys()],
            'Confidence': [f"{predictions[m]['confidence']:.2f}%" for m in predictions.keys()],
            'Benign %': [f"{predictions[m]['benign_prob']:.2f}%" for m in predictions.keys()],
            'Malignant %': [f"{predictions[m]['malignant_prob']:.2f}%" for m in predictions.keys()]
        })
        
        st.dataframe(results_df, use_container_width=True)
        
        # Consensus
        malignant_votes = sum(1 for m in predictions.values() if m['prediction'] == 1)
        consensus = "Malignant" if malignant_votes >= 2 else "Benign"
        
        st.markdown("### Final Recommendation")
        if consensus == "Malignant":
            st.error(f"🔴 **{consensus}** - {malignant_votes}/3 models indicate malignancy")
        else:
            st.success(f"🟢 **{consensus}** - {3-malignant_votes}/3 models indicate benign")

# ============================================================================
# PAGE 3: BATCH TESTING (UPLOAD CSV)
# ============================================================================

def page_batch_testing():
    st.markdown("## Batch Testing - Upload Patient Data")
    st.markdown("Upload a CSV file with patient features for bulk predictions")
    st.markdown("---")
    
    models, feature_names, scaler, loaded = load_models()
    
    if not loaded:
        st.error("Models not loaded.")
        return
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            st.markdown("### Data Preview")
            st.write(f"Loaded {len(df)} patients with {df.shape[1]} columns")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Check if labels are present
            has_labels = 'diagnosis' in df.columns
            
            if has_labels:
                st.info("This data has actual diagnosis labels. Metrics will be calculated.")
            
            # Preprocess
            try:
                df_processed = df[feature_names].copy()
                df_scaled = scaler.transform(df_processed)
            except KeyError as e:
                st.error(f"Missing features in CSV: {e}")
                return
            
            # Get predictions
            st.markdown("### Predictions")
            
            results = []
            
            for model_name, model in models.items():
                preds = model.predict(df_scaled)
                probas = model.predict_proba(df_scaled)[:, 1]
                
                results.append({
                    'Model': model_name,
                    'Predictions': preds,
                    'Probabilities': probas
                })
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'Patient_ID': range(1, len(df) + 1),
                'LR_Prediction': [pred_map[results[0]['Predictions'][i]] for i in range(len(df))],
                'RF_Prediction': [pred_map[results[1]['Predictions'][i]] for i in range(len(df))],
                'XGB_Prediction': [pred_map[results[2]['Predictions'][i]] for i in range(len(df))],
                'LR_Confidence': results[0]['Probabilities'],
                'RF_Confidence': results[1]['Probabilities'],
                'XGB_Confidence': results[2]['Probabilities']
            })
            
            pred_map = {0: 'Benign', 1: 'Malignant'}
            results_df = pd.DataFrame({
                'Patient_ID': range(1, len(df) + 1),
                'LR_Pred': [pred_map[results[0]['Predictions'][i]] for i in range(len(df))],
                'RF_Pred': [pred_map[results[1]['Predictions'][i]] for i in range(len(df))],
                'XGB_Pred': [pred_map[results[2]['Predictions'][i]] for i in range(len(df))],
                'Avg_Confidence': np.mean([results[0]['Probabilities'],
                                          results[1]['Probabilities'],
                                          results[2]['Probabilities']], axis=0)
            })
            
            st.dataframe(results_df, use_container_width=True)
            
            # If labels are present, calculate metrics
            if has_labels:
                y_true = df['diagnosis'].map({'M': 1, 'B': 0})
                
                st.markdown("### Model Performance on Uploaded Data")
                
                metrics_summary = []
                for idx, result in enumerate(results):
                    model_name = result['Model']
                    preds = result['Predictions']
                    probas = result['Probabilities']
                    
                    metrics = calculate_metrics(y_true, preds, probas)
                    metrics_summary.append({
                        'Model': model_name,
                        'Accuracy': metrics['accuracy'],
                        'Precision': metrics['precision'],
                        'Recall': metrics['recall'],
                        'F1-Score': metrics['f1_score']
                    })
                
                metrics_df = pd.DataFrame(metrics_summary)
                st.dataframe(metrics_df, use_container_width=True)
                
                # Visualize confusion matrices
                st.markdown("### Confusion Matrices")
                cols = st.columns(3)
                
                for idx, result in enumerate(results):
                    with cols[idx]:
                        fig = plot_confusion_matrix(y_true, result['Predictions'])
                        st.pyplot(fig)
                
                # ROC curves
                st.markdown("### ROC Curves")
                cols = st.columns(3)
                
                for idx, result in enumerate(results):
                    with cols[idx]:
                        fig, _ = plot_roc_curve(y_true, result['Probabilities'])
                        st.pyplot(fig)
            
            # Download results
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
        
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ============================================================================
# PAGE 4: MODEL COMPARISON
# ============================================================================

def page_model_comparison():
    st.markdown("## Compare Models on Your Data")
    st.markdown("---")
    
    models, feature_names, scaler, loaded = load_models()
    
    if not loaded:
        st.error("Models not loaded.")
        return
    
    # File upload
    uploaded_file = st.file_uploader("Upload test data (with diagnosis column)", type=['csv'], key="compare")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Check for diagnosis column
            if 'diagnosis' not in df.columns:
                st.error("CSV must contain 'diagnosis' column (M or B)")
                return
            
            # Preprocess
            df_features = df[feature_names].copy()
            df_scaled = scaler.transform(df_features)
            
            y_true = df['diagnosis'].map({'M': 1, 'B': 0})
            
            # Get predictions from all models
            all_metrics = {}
            
            for model_name, model in models.items():
                y_pred = model.predict(df_scaled)
                y_proba = model.predict_proba(df_scaled)[:, 1]
                
                metrics = calculate_metrics(y_true, y_pred, y_proba)
                all_metrics[model_name] = metrics
            
            # Display comparison table
            st.markdown("### Performance Metrics Comparison")
            
            comparison_df = pd.DataFrame({
                'Model': list(all_metrics.keys()),
                'Accuracy': [all_metrics[m]['accuracy'] for m in all_metrics.keys()],
                'Precision': [all_metrics[m]['precision'] for m in all_metrics.keys()],
                'Recall': [all_metrics[m]['recall'] for m in all_metrics.keys()],
                'Specificity': [all_metrics[m]['specificity'] for m in all_metrics.keys()],
                'F1-Score': [all_metrics[m]['f1_score'] for m in all_metrics.keys()],
                'AUC': [all_metrics[m]['auc_score'] for m in all_metrics.keys()]
            })
            
            st.dataframe(comparison_df, use_container_width=True)
            
            # Comparison charts
            st.markdown("### Performance Charts")
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score', 'auc_score']
            colors = ['#3498DB', '#E74C3C', '#2ECC71']
            
            for idx, metric in enumerate(metrics_to_plot):
                values = [all_metrics[m][metric] for m in all_metrics.keys()]
                
                ax = axes[idx]
                bars = ax.bar(range(len(all_metrics)), values, color=colors, alpha=0.8, edgecolor='black')
                ax.set_xticks(range(len(all_metrics)))
                ax.set_xticklabels(list(all_metrics.keys()), rotation=45, ha='right')
                ax.set_ylabel(metric.replace('_', ' ').title())
                ax.set_title(f'{metric.replace("_", " ").title()} Comparison', fontweight='bold')
                ax.set_ylim([0, 1.05])
                ax.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Best model
            best_model = max(all_metrics.items(), key=lambda x: x[1]['f1_score'])
            st.markdown("### Best Model")
            st.success(f"**{best_model[0]}** has the highest F1-Score: {best_model[1]['f1_score']:.4f}")
        
        except Exception as e:
            st.error(f"Error: {e}")

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    with st.sidebar:
        st.markdown("## Breast Cancer Detection")
        st.markdown("---")
        
        page = st.radio(
            "Select Testing Mode:",
            ["Home", "Single Patient", "Batch Testing", "Model Comparison"]
        )
    
    if page == "Home":
        page_home()
    elif page == "Single Patient":
        page_single_prediction()
    elif page == "Batch Testing":
        page_batch_testing()
    elif page == "Model Comparison":
        page_model_comparison()

if __name__ == "__main__":
    main()