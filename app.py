# ======================================================================================
# --------------------------------- IMPORTS & SETUP ------------------------------------
# ======================================================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve, roc_auc_score
)
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(page_title="Breast Cancer Detection", page_icon="🎗️", layout="wide")


# ======================================================================================
# ------------------------------- MODEL & DATA LOADING ---------------------------------
# ======================================================================================

@st.cache_resource
def load_models_and_data():
    """
    Loads all necessary model files, scaler, and feature names.
    Returns models, scaler, feature_names, and a status flag.
    """
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

# Load the models into the app
models, scaler, feature_names, models_exist = load_models_and_data()

# ======================================================================================
# -------------------------------- HELPER FUNCTIONS ------------------------------------
# ======================================================================================

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculates a dictionary of classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    auc_score_val = roc_auc_score(y_true, y_pred_proba) if y_pred_proba is not None else None
    return {
        'accuracy': accuracy, 'precision': precision, 'recall': recall,
        'specificity': specificity, 'f1_score': f1, 'auc_score': auc_score_val,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }

def preprocess_new_data(data, features, scl):
    """Selects features and scales the data for prediction."""
    data = data[features]
    data_scaled = scl.transform(data)
    return data_scaled

# ======================================================================================
# ------------------------------ VISUALIZATION FUNCTIONS -------------------------------
# ======================================================================================

def plot_confusion_matrix(y_true, y_pred, model_name):
    """Generates a styled confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=ax,
        xticklabels=['Benign', 'Malignant'],
        yticklabels=['Benign', 'Malignant'],
        cbar_kws={'label': 'Count'}
    )
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=16, fontweight='bold')
    ax.set_ylabel('Actual Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    return fig

def plot_roc_curve(y_true, y_pred_proba, model_name):
    """Generates a styled ROC curve plot."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve - {model_name}', fontsize=16, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig

def plot_pr_curve(y_true, y_pred_proba, model_name):
    """Generates a styled Precision-Recall curve plot."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='green', lw=2.5, label=f'PR (AUC = {pr_auc:.4f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title(f'Precision-Recall Curve - {model_name}', fontsize=16, fontweight='bold')
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig

# ======================================================================================
# ------------------------------ MAIN APPLICATION UI -----------------------------------
# ======================================================================================

# --- HEADER ---
st.markdown("<h1 style='text-align: center; color: #E83E8C;'>🏥 Early Detection of Breast Cancer Using Machine Learning</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: #6c757d;'>A clinical decision support tool powered by an ensemble of predictive models.</h4>", unsafe_allow_html=True)
st.markdown("---")

# --- MODEL EXISTENCE CHECK ---
if not models_exist:
    st.error("🔴 **CRITICAL ERROR: Model files not found!**")
    st.warning("Please ensure the following files are present in the `models/` directory:")
    st.code("""
    models/
    ├── logistic_regression.pkl
    ├── random_forest.pkl
    ├── xgboost.pkl
    ├── scaler.pkl
    └── feature_names.pkl
    """)
    st.stop()

# --- MODE SELECTION ---
st.markdown("### Select an Operating Mode")
mode = st.radio(
    "Choose how you want to test the models:",
    ["📊 Load Sample Data", "👩‍⚕️ Single Patient Prediction", "📂 Batch Testing (CSV Upload)"],
    horizontal=True,
    label_visibility="collapsed"
)

st.markdown("---")

# ======================================================================================
# ----------------------------- MODE 1: LOAD SAMPLE DATA -------------------------------
# ======================================================================================

if mode == "📊 Load Sample Data":
    st.header("📊 Test with Sample Data from Dataset")
    st.markdown("Select an actual patient sample from the pre-loaded dataset to see how the models perform.")
    
    try:
        df1 = pd.read_csv('data/data.csv')
        df2 = pd.read_csv('data/breast_cancer.csv')
        combined_df = pd.concat([df1, df2], ignore_index=True).drop_duplicates()
        
        combined_df = combined_df.drop('id', axis=1, errors='ignore')
        combined_df = combined_df.loc[:, ~combined_df.columns.str.contains('^Unnamed')]
        combined_df = combined_df.iloc[:, ~combined_df.columns.duplicated()]
        combined_df = combined_df.dropna()
        
        st.subheader("Dataset Overview")
        col1, col2, col3 = st.columns(3)
        benign_count = sum(combined_df['diagnosis'] == 'B')
        malignant_count = sum(combined_df['diagnosis'] == 'M')
        col1.metric("Total Samples", len(combined_df))
        col2.metric("🟢 Benign Samples", benign_count)
        col3.metric("🔴 Malignant Samples", malignant_count)
        
        st.markdown("### Select a Patient Sample by Index")
        sample_index = st.slider("Slide to select a patient:", 0, len(combined_df) - 1, 0)
        
        sample = combined_df.iloc[sample_index]
        actual_diagnosis = sample['diagnosis']
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### Patient Details")
            diagnosis_text = "🔴 Malignant" if actual_diagnosis == 'M' else "🟢 Benign"
            st.markdown(f"**Actual Diagnosis:** {diagnosis_text}")
            st.markdown(f"**Patient Index:** `{sample_index}`")
        
        with col2:
            st.markdown("#### Clinical Features")
            st.dataframe(pd.DataFrame({
                'Feature': feature_names,
                'Value': [sample[f] for f in feature_names]
            }), use_container_width=True, height=250)
        
        if st.button("🔬 Predict for This Patient", use_container_width=True, type="primary"):
            st.markdown("---")
            st.subheader("📈 Prediction Results")
            
            sample_df = pd.DataFrame([sample[feature_names]])
            sample_scaled = preprocess_new_data(sample_df, feature_names, scaler)
            
            predictions, results_data = {}, []
            for model_name, model in models.items():
                pred = model.predict(sample_scaled)[0]
                proba = model.predict_proba(sample_scaled)[0]
                confidence = max(proba) * 100
                predictions[model_name] = pred
                results_data.append({
                    'Model': model_name,
                    'Prediction': 'Malignant' if pred == 1 else 'Benign',
                    'Confidence': f"{confidence:.2f}%",
                    'Benign Prob.': f"{proba[0]*100:.2f}%",
                    'Malignant Prob.': f"{proba[1]*100:.2f}%"
                })
            
            st.table(pd.DataFrame(results_data).set_index('Model'))
            
            malignant_votes = sum(1 for p in predictions.values() if p == 1)
            consensus = "MALIGNANT" if malignant_votes >= 2 else "BENIGN"
            
            st.markdown("### 🏁 Final Assessment")
            is_correct = (consensus == 'MALIGNANT' and actual_diagnosis == 'M') or \
                         (consensus == 'BENIGN' and actual_diagnosis == 'B')
            
            if is_correct:
                st.success(f"**Consensus Prediction:** {consensus} | **Actual Diagnosis:** {diagnosis_text} -> ✅ **CORRECT**")
            else:
                st.error(f"**Consensus Prediction:** {consensus} | **Actual Diagnosis:** {diagnosis_text} -> ❌ **INCORRECT**")

    except FileNotFoundError:
        st.error("Data files not found in `data/` folder. Please check your data directory.")

# ======================================================================================
# ------------------------- MODE 2: SINGLE PATIENT PREDICTION --------------------------
# ======================================================================================

elif mode == "👩‍⚕️ Single Patient Prediction":
    st.header("👩‍⚕️ Predict for a Single Patient")
    st.markdown("Enter the 30 clinical feature values below to get a prediction.")
    
    with st.expander("Enter Patient Data Here", expanded=True):
        col1, col2 = st.columns(2)
        patient_data = {}
        
        for idx, feature in enumerate(feature_names):
            target_col = col1 if idx % 2 == 0 else col2
            with target_col:
                patient_data[feature] = st.number_input(feature, value=0.00, step=0.01, format="%.4f")
    
    if st.button("🔬 Get Prediction", use_container_width=True, type="primary"):
        st.markdown("---")
        st.subheader("📈 Prediction Results")
        
        df_patient = pd.DataFrame([patient_data])
        patient_scaled = preprocess_new_data(df_patient, feature_names, scaler)
        
        predictions, results_data = {}, []
        for model_name, model in models.items():
            pred = model.predict(patient_scaled)[0]
            proba = model.predict_proba(patient_scaled)[0]
            confidence = max(proba) * 100
            predictions[model_name] = pred
            results_data.append({
                'Model': model_name,
                'Prediction': 'Malignant' if pred == 1 else 'Benign',
                'Confidence': f"{confidence:.2f}%",
                'Benign Prob.': f"{proba[0]*100:.2f}%",
                'Malignant Prob.': f"{proba[1]*100:.2f}%"
            })
        
        st.table(pd.DataFrame(results_data).set_index('Model'))
        
        malignant_votes = sum(1 for p in predictions.values() if p == 1)
        consensus = "MALIGNANT" if malignant_votes >= 2 else "BENIGN"
        
        st.markdown("### 🏁 Final Recommendation")
        if consensus == "MALIGNANT":
            st.error(f"**FINAL DIAGNOSIS: MALIGNANT** ({malignant_votes}/3 models agree)")
        else:
            st.success(f"**FINAL DIAGNOSIS: BENIGN** ({3-malignant_votes}/3 models agree)")

# ======================================================================================
# -------------------------- MODE 3: BATCH TESTING (CSV) -------------------------------
# ======================================================================================

elif mode == "📂 Batch Testing (CSV Upload)":
    st.header("📂 Batch Testing with CSV File")
    st.markdown("Upload a CSV file containing patient data. If a `diagnosis` column is present, the app will calculate and display performance metrics.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.subheader("Data Summary & Preview")
            has_diagnosis = 'diagnosis' in df.columns
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Patients", len(df))
            col2.metric("Total Features", len(df.columns))
            col3.metric("Diagnosis Column", "✅ Found" if has_diagnosis else "❌ Not Found")
            st.dataframe(df.head(10), use_container_width=True)
            
            try:
                df_features = df[feature_names].copy()
                df_scaled = scaler.transform(df_features)
            except KeyError as e:
                st.error(f"**Feature Mismatch Error:** The uploaded CSV is missing a required feature column: `{e}`. Please check the file.")
                st.stop()
            
            st.markdown("---")
            st.subheader("📊 Batch Prediction Results")
            
            pred_map = {0: 'Benign', 1: 'Malignant'}
            predictions_list = []
            
            for model_name, model in models.items():
                preds = model.predict(df_scaled)
                probas = model.predict_proba(df_scaled)[:, 1]
                predictions_list.append({
                    'model_name': model_name, 'predictions': preds, 'probabilities': probas
                })
            
            results_df = pd.DataFrame({
                'Patient_ID': df.index,
                'LR_Prediction': [pred_map[p] for p in predictions_list[0]['predictions']],
                'RF_Prediction': [pred_map[p] for p in predictions_list[1]['predictions']],
                'XGB_Prediction': [pred_map[p] for p in predictions_list[2]['predictions']],
                'Avg_Malignant_Prob': np.mean([p['probabilities'] for p in predictions_list], axis=0)
            })
            
            st.dataframe(results_df, use_container_width=True)
            
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Predictions as CSV", csv, "batch_predictions.csv",
                "text/csv", key='download-csv', use_container_width=True
            )
            
            if has_diagnosis:
                st.markdown("---")
                st.subheader("📈 Model Performance Metrics")
                
                y_true = df['diagnosis'].map({'M': 1, 'B': 0})
                metrics_data = []
                
                for pred_data in predictions_list:
                    metrics = calculate_metrics(y_true, pred_data['predictions'], pred_data['probabilities'])
                    metrics_data.append({
                        'Model': pred_data['model_name'],
                        'Accuracy': f"{metrics['accuracy']:.4f}",
                        'Precision': f"{metrics['precision']:.4f}",
                        'Recall': f"{metrics['recall']:.4f}",
                        'F1-Score': f"{metrics['f1_score']:.4f}",
                        'AUC': f"{metrics['auc_score']:.4f}" if metrics['auc_score'] else "N/A"
                    })
                
                st.table(pd.DataFrame(metrics_data).set_index('Model'))
                
                st.markdown("### Performance Visualizations")
                tab1, tab2, tab3 = st.tabs(["Confusion Matrices", "ROC Curves", "Precision-Recall Curves"])
                
                with tab1:
                    cols = st.columns(3)
                    for i, p_data in enumerate(predictions_list):
                        with cols[i]:
                            fig = plot_confusion_matrix(y_true, p_data['predictions'], p_data['model_name'])
                            st.pyplot(fig)
                with tab2:
                    cols = st.columns(3)
                    for i, p_data in enumerate(predictions_list):
                        with cols[i]:
                            fig = plot_roc_curve(y_true, p_data['probabilities'], p_data['model_name'])
                            st.pyplot(fig)
                with tab3:
                    cols = st.columns(3)
                    for i, p_data in enumerate(predictions_list):
                        with cols[i]:
                            fig = plot_pr_curve(y_true, p_data['probabilities'], p_data['model_name'])
                            st.pyplot(fig)

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

# ======================================================================================
# -------------------------------------- FOOTER ----------------------------------------
# ======================================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <p style='font-size: 14px; color: #888;'>
        ✨ Made by <b><i>Rahin Toshmi Ohee</i></b> ✨
    </p>
</div>
""", unsafe_allow_html=True)