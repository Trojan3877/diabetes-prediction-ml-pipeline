![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Pipeline-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Classification-f7931e?logo=scikitlearn)
![Healthcare](https://img.shields.io/badge/Use%20Case-Diabetes%20Prediction-red)
![Data Science](https://img.shields.io/badge/Data%20Science-End--to--End-purple)
![Status](https://img.shields.io/badge/Status-Portfolio%20Ready-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)
![Last Commit](https://img.shields.io/github/last-commit/Trojan3877/diabetes-prediction-ml-pipeline)
![Repo Size](https://img.shields.io/github/repo-size/Trojan3877/diabetes-prediction-ml-pipeline)
![Stars](https://img.shields.io/github/stars/Trojan3877/diabetes-prediction-ml-pipeline?style=social)


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

Design Questions & Reflections
Q: What problem does this project aim to solve?
A: This project aims to build a structured machine learning pipeline for predicting diabetes risk, moving beyond prototype notebooks to something closer to a real-world workflow. The goal was to explore how different stages — data cleaning, feature engineering, model training, and evaluation — interact when organized as a coherent system.
Q: Why did I choose this pipeline architecture instead of a quick notebook?
A: I chose a modular pipeline because ML systems in practice aren’t one big script — they have distinct stages that need to be repeatable, testable, and reusable. This structure made it easier for me to isolate issues, evaluate performance at each stage, and think about how data changes propagate through the model.
Q: What were the main trade-offs I made?
A: The trade-off was between rapid experimentation and long-term clarity. A single notebook might have let me iterate faster early on, but it wouldn’t have been maintainable or easy to reason about. By structuring the pipeline, I gained clarity and reproducibility, even if it slowed the early phase of development.
Q: What didn’t work as expected?
A: At first, feature scaling and preprocessing decisions caused inconsistent performance across validation splits, which was frustrating. Rather than tuning hyperparameters blindly, I stepped back and improved the data standardization logic, which made results more stable and interpretable.
Q: What did I learn from building this project?
A: I learned that ML performance is often more about data and pipeline quality than model choice. Investing time in cleaning and validating data improved overall results more than switching to more complex models. I also learned how important clear evaluation metrics are for trusting model behavior.
Q: If I had more time or resources, what would I improve next?
A: I would add better cross-validation and automated testing for each pipeline stage, and experiment with calibration techniques to better align model confidence with real risk levels. I’d also explore ways to visualize decision boundaries and how feature importance shifts across different cohorts.

# 🏆 Author  
**Corey Leath (Trojan3877)**  
Aspiring AI/ML Engineer • Software Developer • Future UPenn AI Master's Student  
GitHub: https://github.com/Trojan3877  
LinkedIn: *https://linkedin.com/in/corey-leath*

---

