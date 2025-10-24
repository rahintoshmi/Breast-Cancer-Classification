<div align="center">
  <h1 align="center">🎗️ Breast Cancer Detection using Machine Learning 🎗️</h1>
  <p align="center">
    An end-to-end pipeline for training, evaluating, and deploying multiple machine learning models for early breast cancer detection, complete with an interactive Streamlit web application.
  </p>
</div>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python" alt="Python Version">
  <img src="https://img.shields.io/badge/Scikit--Learn-1.x-orange?style=for-the-badge&logo=scikit-learn" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Streamlit-1.x-red?style=for-the-badge&logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Pandas-2.x-green?style=for-the-badge&logo=pandas" alt="Pandas">
</p>

---
## 🚀 Live Demo

Try the project live by clicking the link below:

[🌐 View Live Demo](https://breast-cancer-classification-rahin.streamlit.app/)

---

## 📸 Model & App Pages Preview

<table align="center">
  <tr>
    <td align="center">
      <h4>Load Sample Data</h4>
      <img src="https://i.postimg.cc/cHZtdLDK/model.png" alt="Load Sample Data" style="max-width:300px; width:100%;">
    </td>
    <td align="center">
      <h4>Single Patient Prediction</h4>
      <img src="https://i.imghippo.com/files/vY8572Liw.png" alt="Single Patient Prediction" style="max-width:300px; width:100%;">
    </td>
    <td align="center">
      <h4>Batch Testing (CSV Upload)</h4>
      <img src="https://i.postimg.cc/3RKNPbRT/model3.png" alt="Batch Testing" style="max-width:300px; width:100%;">
    </td>
  </tr>
</table>

---
---

## 📋 Table of Contents

- [About the Project](#-about-the-project)
- [🚀 Live Demo](#-live-demo)
- [📸 Model & App Pages Preview](#-model-&-app-pages-previw)
- [✨ Key Features](#-key-features)
- [📂 Project Structure](#-project-structure)
- [🛠️ Technologies Used](#️-technologies-used)
- [🚀 Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [📈 Model Performance Summary](#-model-performance-summary)
- [💡 Key Findings](#-key-findings)
- [🤝 Contributing](#-contributing)
- [📝 License](#-license)
- [📬 Contact](#-contact)

---

## 📖 About the Project

This repository hosts a full pipeline for early breast cancer detection using machine learning. It includes:
- Data cleaning and preprocessing.
- Multiple classification model implementations.
- A Streamlit web app for interactive predictions.

A unique highlight is a **Logistic Regression model built entirely from scratch** using only NumPy, evaluated alongside `Scikit-learn` ,`Random Forest` and `XGBoost` models. The focus is on recall and specificity, crucial metrics in healthcare diagnostics.

---

## ✨ Key Features

- **End-to-End ML Pipeline:** From raw data ingestion to trained models and app deployment.
- **Model Variety:**
  - Logistic Regression (from scratch)
  - Scikit-learn Logistic Regression
  - Random Forest Classifier
  - XGBoost Classifier
- **Performance Insights:** Metrics: Accuracy, Precision, Recall, F1-Score, Specificity, AUC.
- **Interactive Visualization:** Confusion matrices, ROC curves, and feature importance plots.
- **Deployment Ready:** Includes model persistence with `joblib` for real-time use.
- **User-friendly UI:** Streamlit-based app supports:
  1. Sample data testing
  2. Single patient input
  3. Batch testing via CSV upload

---

## 📂 Project Structure

The repository follows a clean modular layout for scalability:
```
├── 📂 data/
│   ├── data.csv
│   └── breast-cancer.csv
│
├── 📂 models/
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── assets/
│   ├── page1.png
│   ├── page2.png
│   └── page3.png
├── 🚀 app.py                      # Streamlit web application entry point
├── 📄 requirements.txt            # Project dependencies
└── 📄 README.md                   # You are here!

```

---

## 🛠️ Technologies Used

**Languages & Libraries:**
- Python 3.9+
- Pandas
- NumPy  

**Machine Learning Frameworks:**
- Scikit-learn
- XGBoost  

**Visualization Tools:**
- Matplotlib
- Seaborn  

**Web Framework:**
- Streamlit
- Joblib (for model persistence)

---

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip package manager

### Installation

1. **Clone this Repository**

---

## 📈 Model Performance Summary

| Model                        | Accuracy | Precision | Recall | F1-Score | AUC    |
|------------------------------|:--------:|:---------:|:------:|:--------:|:------:|
| Manual Logistic Regression   | **0.9825** | **1.0000** | **0.9524** | **0.9756** | N/A    |
| Random Forest                | 0.9737   | 1.0000    | 0.9286 | 0.9630   | 0.9929 |
| Logistic Regression (SKL)    | 0.9649   | 0.9750    | 0.9286 | 0.9512   | 0.9960 |
| XGBoost                      | 0.9649   | 1.0000    | 0.9048 | 0.9500   | 0.9947 |

---

## 💡 Key Findings

- **Top Performer:** Manual Logistic Regression delivers the best trade-off between recall and precision.
- **Significance:** High Recall ensures malignant cases aren’t missed, while High Specificity reduces false positives.

---

## 🤝 Contributing

Want to improve or extend the project? Contributions are welcome!

1. Fork the repository  
2. Create a feature branch  
3. Commit changes
4. Push changes  
5. Submit a pull request  

---

## 📬 Contact

**Rahin Toshmi Ohee**  
Linkdin: [@Rahin Toshmi Ohee](https://www.linkedin.com/in/rahintoshmiohee) 
GitHub: [@rahintoshmi](https://github.com/rahintoshmi)  
Feel free to share feedback or questions!
