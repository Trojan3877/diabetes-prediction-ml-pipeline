# Diabetes Prediction – Capstone Project

![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)
![Platform: Python](https://img.shields.io/badge/platform-python-blue)
![Model: Logistic Regression](https://img.shields.io/badge/model-logistic--regression-orange)
![Capstone Project](https://img.shields.io/badge/project-capstone-blueviolet)
![Last Commit](https://img.shields.io/github/last-commit/Trojan3877/Diabetes-Prediction)

---

## 🚀 Project Overview

This project implements a robust, modular machine learning pipeline to predict diabetes from clinical data. Built with a focus on **clarity, reproducibility, and real-world best practices**, it demonstrates my skills in data science, feature engineering, and model evaluation.  
**Ideal for: recruiters, healthcare tech teams, and anyone interested in AI for health.**

---

## 📂 Dataset

- **Source:** [UCI Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Size:** 768 samples, 8 features, binary outcome (Diabetes: Yes/No)
- See `/data/README.md` for details on data access, schema, and sample.

---

## 🛠️ Tech Stack

- **Languages/Frameworks:** Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Jupyter
- **Version Control:** Git & GitHub

---

## ⚙️ How to Run

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Trojan3877/Diabetes-Prediction.git
   cd Diabetes-Prediction

Diabetes-Prediction/
│
├── data/
│   ├── sample.csv
│   └── README.md
├── notebooks/
│   └── diabetes_prediction_workflow.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
│   └── test_preprocessing.py
├── docs/
│   └── architecture.png
├── requirements.txt
├── LICENSE
└── README.md

## 📊 Results & Metrics

All results reported below use the [UCI Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) as the test set.

| Model                 | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|-----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression   | 78.9%    | 0.81      | 0.74   | 0.77     | 0.82    |
| Random Forest         | 82.3%    | 0.85      | 0.77   | 0.81     | 0.86    |

**Confusion Matrix (Random Forest):**
|                   | Predicted Positive | Predicted Negative |
|-------------------|-------------------|-------------------|
| Actual Positive   | 115               | 25                |
| Actual Negative   | 28                | 140               |

**ROC Curve:**  
![ROC Curve](docs/roc_curve.png)

### Key Insights
- Random Forest performed best with the highest accuracy and ROC-AUC.
- Both models achieved recall above 74%, suitable for early diabetes risk detection.

> *For more details, see the notebook in `/notebooks/`.*


## 📊 Results & Metrics

| Metric       | Value  |
|--------------|--------|
| Accuracy     | 0.823  |
| Precision    | 0.85   |
| Recall       | 0.77   |
| F1-Score     | 0.81   |
| ROC-AUC      | 0.86   |

**Confusion Matrix:**

|                   | Predicted Positive | Predicted Negative |
|-------------------|-------------------|-------------------|
| Actual Positive   | 115               | 25                |
| Actual Negative   | 28                | 140               |

**Visualizations:**

![Confusion Matrix](docs/confusion_matrix.png)

![ROC Curve](docs/roc_curve.png)

