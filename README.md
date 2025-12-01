# 🧬 Diabetes Prediction ML Pipeline  
A full production-style machine learning pipeline for predicting diabetes using structured health data.  
This project demonstrates **end-to-end ML engineering**, including data preprocessing, feature engineering, model training, evaluation, reproducibility, and modular Python package design.

---
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
LinkedIn: *Add your link here*

---

