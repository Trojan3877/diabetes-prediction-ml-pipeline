# Data Folder

This directory contains sample data used for the Diabetes Prediction ML Pipeline.

## Sample CSV

- **File:** `sample.csv`
- **Source:** A small excerpt from the UCI Pima Indians Diabetes Database (available on Kaggle).
- **Columns:**
  1. `Pregnancies`         – Number of times pregnant
  2. `Glucose`             – Plasma glucose concentration (mg/dL)
  3. `BloodPressure`       – Diastolic blood pressure (mm Hg)
  4. `SkinThickness`       – Triceps skin fold thickness (mm)
  5. `Insulin`             – 2-Hour serum insulin (µU/mL)
  6. `BMI`                 – Body mass index (weight in kg/(height in m)^2)
  7. `DiabetesPedigreeFunction` – Diabetes pedigree function
  8. `Age`                 – Age (years)
  9. `Outcome`             – Class variable (0 = no diabetes, 1 = diabetes)

## Usage

- This `sample.csv` (10–20 rows) is provided so you can test preprocessing and model-training steps locally.
- To run on the full dataset, download from:
  https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database

Place the full CSV at `data/diabetes.csv` (or update `config/config.yaml`) to rerun the full pipeline.
