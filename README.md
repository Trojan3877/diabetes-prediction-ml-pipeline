
<p align="center">

  <!-- Python Version -->
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white"/>

  <!-- Machine Learning -->
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-yellow?style=for-the-badge&logo=scikitlearn&logoColor=white"/>

  <!-- Data Processing -->
  <img src="https://img.shields.io/badge/Data%20Pipeline-Pandas-orange?style=for-the-badge&logo=pandas&logoColor=white"/>

  <!-- Model Type -->
  <img src="https://img.shields.io/badge/Model-RandomForest-success?style=for-the-badge&logo=treehouse&logoColor=white"/>

  <!-- ML Engineering -->
  <img src="https://img.shields.io/badge/ML%20Engineering-Production%20Pipeline-red?style=for-the-badge&logo=githubactions&logoColor=white"/>

  <!-- Code Quality -->
  <img src="https://img.shields.io/badge/Code_Style-PEP8-green?style=for-the-badge"/>

  <!-- Testing -->
  <img src="https://img.shields.io/badge/Tests-PyTest-brightgreen?style=for-the-badge&logo=pytest&logoColor=white"/>

  <!-- File Structure -->
  <img src="https://img.shields.io/badge/Structure-Modular_Architecture-purple?style=for-the-badge"/>

  <!-- Config -->
  <img src="https://img.shields.io/badge/Config-YAML-blue?style=for-the-badge&logo=yaml&logoColor=white"/>

  <!-- Joblib -->
  <img src="https://img.shields.io/badge/Model%20Persistence-Joblib-9cf?style=for-the-badge"/>

  <!-- Repository Stats -->
  <img src="https://img.shields.io/github/last-commit/Trojan3877/Diabetes_Prediction_ML_Pipeline?style=for-the-badge&color=blue"/>
  <img src="https://img.shields.io/github/repo-size/Trojan3877/Diabetes_Prediction_ML_Pipeline?style=for-the-badge&color=orange"/>

  <!-- Visitors -->
  <img src="https://komarev.com/ghpvc/?username=Trojan3877&label=VIEWS&style=for-the-badge&color=brightgreen"/>

</p>


# 🧬 Diabetes Prediction ML Pipeline  
A full production-style machine learning pipeline for predicting diabetes using structured health data.  
This project demonstrates **end-to-end ML engineering**, including data preprocessing, feature engineering, model training, evaluation, reproducibility, and modular Python package design.

---
<p align="center">
  <img src="https://files.catbox.moe/6l8x9i.png" width="100%" alt="Diabetes Prediction ML Pipeline Banner">
</p>
## 🧱 System Architecture Overview

```
                ┌─────────────────────────┐
                │       Raw Dataset       │
                │    (diabetes.csv)       │
                └─────────────┬───────────┘
                              │
                              ▼
                ┌─────────────────────────┐
                │     Data Preprocessing  │
                │ - Missing value checks  │
                │ - Scaling (Standard)    │
                │ - Train/Test Split      │
                └─────────────┬───────────┘
                              │
                ┌─────────────▼──────────────┐
                │      Feature Matrix (X)     │
                │      Target Vector (y)      │
                └─────────────┬──────────────┘
                              │
                              ▼
                ┌─────────────────────────┐
                │       Model Training     │
                │  (RandomForest / LR)     │
                │ - Fit                    │
                │ - Save model.pkl         │
                └─────────────┬───────────┘
                              │
                ┌─────────────▼──────────────┐
                │        Evaluation           │
                │ - Accuracy / F1 / ROC-AUC   │
                │ - Confusion Matrix Plot     │
                │ - Writes metrics.md         │
                └─────────────┬──────────────┘
                              │
                              ▼
                ┌─────────────────────────┐
                │     Deployment Ready     │
                │ - model.pkl              │
                │ - scaler.pkl             │
                │ - metrics.md             │
                └─────────────────────────┘
```
## 🚀 Project Highlights

- ✔ **Fully modular ML codebase** (ready for expansion or deployment)  
- ✔ **Config-driven pipeline** (YAML configuration for reproducible experiments)  
- ✔ **Feature engineering + scaling + train/test splitting**  
- ✔ **Random Forest + Logistic Regression baseline**  
- ✔ **Production-ready structure** used by major tech companies  
- ✔ **Automated evaluation + metrics + plots**  
- ✔ **Tests folder for PyTest unit testing**  
- ✔ **Suitable for L5/L6 ML Engineer interview portfolio**

---
## 🔄 ML Pipeline Flowchart

```
               ┌─────────────────────┐
               │    Load Raw Data    │
               └──────────┬──────────┘
                          ▼
               ┌─────────────────────┐
               │   Preprocess Data   │
               │ - Scaling           │
               │ - Splitting         │
               └──────────┬──────────┘
                          ▼
               ┌─────────────────────┐
               │     Train Model     │
               │ RandomForest / LR   │
               └──────────┬──────────┘
                          ▼
               ┌─────────────────────┐
               │     Evaluate Model   │
               │ - Metrics            │
               │ - Confusion Matrix   │
               └──────────┬──────────┘
                          ▼
               ┌─────────────────────┐
               │   Save Artifacts    │
               │ model.pkl + reports │
               └─────────────────────┘
```
# 📂 Folder Structure

```
Diabetes_Prediction_ML_Pipeline/
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
│
├── models/
│   └── model.pkl
│
├── tests/
│
├── docs/
│   ├── architecture.png
│   ├── pipeline_flowchart.png
│   └── model_performance.png
│
├── metrics.md
├── requirements.txt
└── README.md
```

---

# 🔧 Installation

```bash
git clone https://github.com/Trojan3877/Diabetes_Prediction_ML_Pipeline
cd Diabetes_Prediction_ML_Pipeline
pip install -r requirements.txt
```

---

# ⚙️ Run the Pipeline

### **1. Preprocess Data**
```bash
python src/preprocess.py
```

### **2. Train Model**
```bash
python src/train.py
```

### **3. Evaluate Model**
```bash
python src/evaluate.py
```

Evaluation metrics will be written to:

- `metrics.md`  
- `/docs/model_performance.png`  
- console output

---

# 📊 Model Performance (Summary)

| Metric | Score (placeholder) |
|-------|----------------------|
| Accuracy | 0.89 |
| Precision | 0.84 |
| Recall | 0.80 |
| F1 Score | 0.82 |
| ROC-AUC | 0.91 |

Full metrics in `metrics.md`.

---

# 📈 Pipeline Architecture

```
RAW CSV → Preprocess → Split → Train Model → Evaluate → Metrics / Plots → model.pkl
```

Diagram file: `docs/pipeline_flowchart.png`

---

# 🧱 Tech Stack

- Python 3.10+
- NumPy, Pandas
- Scikit-learn
- Matplotlib / Seaborn
- PyTest
- YAML config management
- Joblib (model persistence)

---

# 📘 Future Enhancements

- Add MLflow experiment tracking  
- Add FastAPI inference endpoint  
- Add Dockerfile for containerization  
- Add Snowflake feature store  
- Add CI/CD pipeline  
- Add Streamlit dashboard  

---

# 🏆 Author  
**Corey Leath (Trojan3877)**  
Aspiring AI/ML Engineer • Software Developer • Future UPenn AI Master's Student  
GitHub: https://github.com/Trojan3877  
LinkedIn: *https://linkedin.com/in/corey-leath*

---

