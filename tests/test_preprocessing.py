diabetes-prediction-ml-pipeline/tests/test_preprocessing.py

import pandas as pd
import pytest
from src.preprocessing import clean_missing_values

def test_clean_missing_values_replaces_zero():
    # Create a small DataFrame with zeros in key columns
    df = pd.DataFrame({
        "Glucose": [0, 120, 85],
        "BloodPressure": [72, 0, 80],
        "SkinThickness": [0, 30, 20],
        "Insulin": [0, 100, 150],
        "BMI": [0.0, 25.0, 30.0],
        "Age": [25, 35, 45],
        "DiabetesPedigreeFunction": [0.5, 1.2, 0.8],
        "Pregnancies": [1, 2, 3],
        "Outcome": [0, 1, 1]
    })

    df_cleaned = clean_missing_values(df)

    # After cleaning, none of these columns should contain zeros
    assert (df_cleaned["Glucose"] == 0).sum() == 0
    assert (df_cleaned["BloodPressure"] == 0).sum() == 0
    assert (df_cleaned["SkinThickness"] == 0).sum() == 0
    assert (df_cleaned["Insulin"] == 0).sum() == 0
    assert (df_cleaned["BMI"] == 0.0).sum() == 0

    # Check that other columns remain unchanged
    assert (df_cleaned["Age"] == df["Age"]).all()
    assert (df_cleaned["Outcome"] == df["Outcome"]).all()

pytest
