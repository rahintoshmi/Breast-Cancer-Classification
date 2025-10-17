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
    """Load all saved models"""
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

# Load models
models, scaler, feature_names, models_exist = load_models()

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
    """Preprocess new data"""
    data = data[feature_names]
    data_scaled = scaler.transform(data)
    return data_scaled

# ============================================================================
# VISUALIZATIONS
# ============================================================================

def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Benign', 'Malignant'],
                yticklabels=['Benign', 'Malignant'],
                cbar_kws={'label': 'Count'})
    ax.set_title(f'Confusion Matrix - {model_name}', fontweight='bold', fontsize=12)
    ax.set_ylabel('Actual Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)
    return fig

def plot_roc_curve(y_true, y_pred_proba, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'ROC Curve - {model_name}', fontweight='bold', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    return fig

def plot_pr_curve(y_true, y_pred_proba, model_name):
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='green', lw=2.5, label=f'PR (AUC = {pr_auc:.4f})')
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title(f'Precision-Recall Curve - {model_name}', fontweight='bold', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    return fig

if not models_exist:
    st.error("ERROR: Models not found!")
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

# ============================================================================
# PAGE 0: LOAD SAMPLE DATA
# ============================================================================

if mode == "Load Sample Data":
    st.markdown("## Load Sample Data from Existing Dataset")
    st.markdown("Select and test predictions on actual patient samples from the dataset")
    st.markdown("---")
    
    try:
        # Load combined dataset
        df1 = pd.read_csv('data/data.csv')
        df2 = pd.read_csv('data/breast_cancer.csv')
        combined_df = pd.concat([df1, df2], ignore_index=True).drop_duplicates()
        
        # Clean data
        combined_df = combined_df.drop('id', axis=1, errors='ignore')
        combined_df = combined_df.loc[:, ~combined_df.columns.str.contains('^Unnamed')]
        combined_df = combined_df.iloc[:, ~combined_df.columns.duplicated()]
        combined_df = combined_df.dropna()
        
        st.markdown("### Dataset Information")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(combined_df))
        with col2:
            benign_count = sum(combined_df['diagnosis'] == 'B')
            st.metric("Benign Samples", benign_count)
        with col3:
            malignant_count = sum(combined_df['diagnosis'] == 'M')
            st.metric("Malignant Samples", malignant_count)
        
        # Select sample
        st.markdown("### Select a Sample Patient")
        
        col1, col2 = st.columns(2)
        
        with col1:
            sample_index = st.slider(
                "Select patient index",
                min_value=0,
                max_value=len(combined_df) - 1,
                value=0
            )
        
        with col2:
            sample_type = st.radio(
                "Filter by type:",
                ["All", "Benign Only", "Malignant Only"],
                horizontal=True
            )
            
            if sample_type == "Benign Only":
                filtered_indices = combined_df[combined_df['diagnosis'] == 'B'].index.tolist()
                sample_index = st.slider(
                    "Select benign patient",
                    min_value=0,
                    max_value=len(filtered_indices) - 1,
                    value=0,
                    key="benign_slider"
                )
                sample_index = filtered_indices[sample_index]
            
            elif sample_type == "Malignant Only":
                filtered_indices = combined_df[combined_df['diagnosis'] == 'M'].index.tolist()
                sample_index = st.slider(
                    "Select malignant patient",
                    min_value=0,
                    max_value=len(filtered_indices) - 1,
                    value=0,
                    key="malignant_slider"
                )
                sample_index = filtered_indices[sample_index]
        
        # Get sample
        sample = combined_df.iloc[sample_index]
        actual_diagnosis = sample['diagnosis']
        
        st.markdown("### Selected Patient Data")
        
        # Display patient features
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Actual Diagnosis:** " + ("🔴 Malignant" if actual_diagnosis == 'M' else "🟢 Benign"))
            st.markdown(f"**Patient Index:** {sample_index}")
        
        with col2:
            st.markdown(f"**Feature Values:** {len(feature_names)} features")
        
        # Show features table
        st.dataframe(pd.DataFrame({
            'Feature': feature_names,
            'Value': [sample[f] for f in feature_names]
        }), use_container_width=True)
        
        # Make predictions
        if st.button("Get Predictions for This Patient"):
            st.markdown("---")
            st.markdown("## Prediction Results")
            
            # Prepare data
            sample_df = pd.DataFrame([sample[feature_names]])
            sample_scaled = preprocess_new_data(sample_df, feature_names, scaler)
            
            # Get predictions
            predictions = {}
            results_data = []
            
            for model_name, model in models.items():
                pred = model.predict(sample_scaled)[0]
                proba = model.predict_proba(sample_scaled)[0]
                confidence = max(proba) * 100
                
                predictions[model_name] = pred
                
                results_data.append({
                    'Model': model_name,
                    'Prediction': 'Malignant' if pred == 1 else 'Benign',
                    'Confidence': f"{confidence:.2f}%",
                    'Benign %': f"{proba[0]*100:.2f}%",
                    'Malignant %': f"{proba[1]*100:.2f}%"
                })
            
            df_results = pd.DataFrame(results_data)
            st.dataframe(df_results, use_container_width=True)
            
            # Consensus
            malignant_votes = sum(1 for p in predictions.values() if p == 1)
            consensus = "MALIGNANT" if malignant_votes >= 2 else "BENIGN"
            
            st.markdown("---")
            st.markdown("## Comparison with Actual Diagnosis")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**Actual:** {'🔴 Malignant' if actual_diagnosis == 'M' else '🟢 Benign'}")
            
            with col2:
                st.markdown(f"**Predicted:** {'🔴 {}'.format(consensus) if consensus == 'MALIGNANT' else '🟢 {}'.format(consensus)}")
            
            with col3:
                is_correct = (consensus == 'MALIGNANT' and actual_diagnosis == 'M') or (consensus == 'BENIGN' and actual_diagnosis == 'B')
                if is_correct:
                    st.markdown("**Result:** ✅ CORRECT")
                else:
                    st.markdown("**Result:** ❌ INCORRECT")
    
    except FileNotFoundError:
        st.error("Data files not found in data/ folder")
        st.info("Make sure you have data/data.csv and data/breast_cancer.csv")

# ============================================================================
# MAIN APP
# ============================================================================

elif not models_exist:
    st.error("ERROR: Models not found!")
    st.warning("Make sure you have these files in the 'models/' folder:")
    st.code("""
    models/
    ├── logistic_regression.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    ├── scaler.pkl
    └── feature_names.pkl
    """)

st.markdown("# Hospital Breast Cancer Detection System")
st.markdown("Clinical decision support using machine learning")
st.markdown("---")

# Mode selection
mode = st.radio(
    "Select Testing Mode:",
    ["Load Sample Data", "Single Patient Prediction", "Batch Testing (CSV Upload)"],
    horizontal=True
)

# ============================================================================
# PAGE 1: SINGLE PATIENT PREDICTION
# ============================================================================

if mode == "Single Patient Prediction":
    st.markdown("## Enter Patient Information")
    st.markdown("Input the 30 clinical features for the patient")
    
    col1, col2 = st.columns(2)
    patient_data = {}
    
    for idx, feature in enumerate(feature_names):
        if idx % 2 == 0:
            with col1:
                patient_data[feature] = st.number_input(
                    feature, 
                    value=0.0, 
                    step=0.01,
                    key=f"patient_{feature}"
                )
        else:
            with col2:
                patient_data[feature] = st.number_input(
                    feature, 
                    value=0.0, 
                    step=0.01,
                    key=f"patient_{feature}"
                )
    
    if st.button("Get Prediction", key="predict_btn"):
        st.markdown("---")
        st.markdown("## Prediction Results")
        
        # Prepare data
        df_patient = pd.DataFrame([patient_data])
        patient_scaled = preprocess_new_data(df_patient, feature_names, scaler)
        
        # Get predictions from all models
        predictions = {}
        results_data = []
        
        for model_name, model in models.items():
            pred = model.predict(patient_scaled)[0]
            proba = model.predict_proba(patient_scaled)[0]
            confidence = max(proba) * 100
            
            predictions[model_name] = pred
            
            results_data.append({
                'Model': model_name,
                'Prediction': 'Malignant' if pred == 1 else 'Benign',
                'Confidence': f"{confidence:.2f}%",
                'Benign Probability': f"{proba[0]*100:.2f}%",
                'Malignant Probability': f"{proba[1]*100:.2f}%"
            })
        
        # Display results table
        df_results = pd.DataFrame(results_data)
        st.dataframe(df_results, use_container_width=True)
        
        # Calculate consensus
        malignant_votes = sum(1 for p in predictions.values() if p == 1)
        consensus = "MALIGNANT" if malignant_votes >= 2 else "BENIGN"
        
        st.markdown("---")
        st.markdown("## Final Recommendation")
        
        if consensus == "MALIGNANT":
            st.error(f"🔴 **{consensus}** - {malignant_votes}/3 models indicate malignancy")
            st.warning("Further clinical evaluation recommended")
        else:
            st.success(f"🟢 **{consensus}** - {3-malignant_votes}/3 models indicate benign")
            st.info("Low risk of malignancy")

# ============================================================================
# PAGE 2: BATCH TESTING
# ============================================================================

elif mode == "Batch Testing (CSV Upload)":
    st.markdown("## Batch Testing - Upload CSV File")
    st.markdown("Upload a CSV with patient data. If it has 'diagnosis' column, metrics will be calculated.")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'], key="batch_upload")
    
    if uploaded_file is not None:
        try:
            # Load data
            df = pd.read_csv(uploaded_file)
            
            st.markdown("### Data Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Patients", len(df))
            with col2:
                st.metric("Features", len(df.columns))
            with col3:
                st.metric("Has Diagnosis", "Yes" if 'diagnosis' in df.columns else "No")
            
            st.markdown("### Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Check if has diagnosis column
            has_diagnosis = 'diagnosis' in df.columns
            
            # Preprocess data
            try:
                df_features = df[feature_names].copy()
                df_scaled = scaler.transform(df_features)
            except KeyError as e:
                st.error(f"Missing features in CSV: {e}")
                st.stop()
            
            # Get predictions
            st.markdown("---")
            st.markdown("### Predictions")
            
            pred_map = {0: 'Benign', 1: 'Malignant'}
            predictions_list = []
            
            for model_name, model in models.items():
                preds = model.predict(df_scaled)
                probas = model.predict_proba(df_scaled)[:, 1]
                
                predictions_list.append({
                    'model_name': model_name,
                    'predictions': preds,
                    'probabilities': probas
                })
            
            # Create results dataframe
            results_df = pd.DataFrame({
                'Patient_ID': range(1, len(df) + 1),
                'LR_Prediction': [pred_map[predictions_list[0]['predictions'][i]] for i in range(len(df))],
                'RF_Prediction': [pred_map[predictions_list[1]['predictions'][i]] for i in range(len(df))],
                'XGB_Prediction': [pred_map[predictions_list[2]['predictions'][i]] for i in range(len(df))],
                'Average_Confidence': np.mean([predictions_list[0]['probabilities'],
                                               predictions_list[1]['probabilities'],
                                               predictions_list[2]['probabilities']], axis=0)
            })
            
            st.dataframe(results_df, use_container_width=True)
            
            # Download predictions
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )
            
            # If has diagnosis, calculate metrics
            if has_diagnosis:
                st.markdown("---")
                st.markdown("### Model Performance Metrics")
                
                y_true = df['diagnosis'].map({'M': 1, 'B': 0})
                
                metrics_data = []
                
                for pred_data in predictions_list:
                    model_name = pred_data['model_name']
                    preds = pred_data['predictions']
                    probas = pred_data['probabilities']
                    
                    metrics = calculate_metrics(y_true, preds, probas)
                    
                    metrics_data.append({
                        'Model': model_name,
                        'Accuracy': f"{metrics['accuracy']:.4f}",
                        'Precision': f"{metrics['precision']:.4f}",
                        'Recall': f"{metrics['recall']:.4f}",
                        'Specificity': f"{metrics['specificity']:.4f}",
                        'F1-Score': f"{metrics['f1_score']:.4f}",
                        'AUC': f"{metrics['auc_score']:.4f}" if metrics['auc_score'] else "N/A"
                    })
                
                metrics_df = pd.DataFrame(metrics_data)
                st.dataframe(metrics_df, use_container_width=True)
                
                # Visualizations
                st.markdown("---")
                st.markdown("### Confusion Matrices")
                
                cols = st.columns(3)
                for idx, pred_data in enumerate(predictions_list):
                    with cols[idx]:
                        fig = plot_confusion_matrix(y_true, pred_data['predictions'], pred_data['model_name'])
                        st.pyplot(fig)
                
                st.markdown("### ROC Curves")
                
                cols = st.columns(3)
                for idx, pred_data in enumerate(predictions_list):
                    with cols[idx]:
                        fig = plot_roc_curve(y_true, pred_data['probabilities'], pred_data['model_name'])
                        st.pyplot(fig)
                
                st.markdown("### Precision-Recall Curves")
                
                cols = st.columns(3)
                for idx, pred_data in enumerate(predictions_list):
                    with cols[idx]:
                        fig = plot_pr_curve(y_true, pred_data['probabilities'], pred_data['model_name'])
                        st.pyplot(fig)
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("Make sure CSV has the correct format and all required features")

# ============================================================================
# FOOTER
# ================================

st.markdown("---")
st.markdown("<div style='text-align: center; padding: 20px;'>" + 
            "<p style='font-size: 12px; color: #888;'>" +
            "Made by <b>Rahin Toshmi Ohee</b>" +
            "</p>" +
            "</div>", unsafe_allow_html=True)