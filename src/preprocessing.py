"""
Preprocessing utilities for the Diabetes Prediction ML Pipeline.

Physiologically, a value of 0 for Glucose, BloodPressure, SkinThickness,
Insulin, or BMI is impossible, so those zeros are treated as missing data
and replaced with each column's median.
"""

import pandas as pd

# Columns where 0 is physiologically impossible and should be treated as NaN
ZERO_IMPUTE_COLS = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]


def clean_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace impossible zero values with column medians.

    Parameters
    ----------
    df : pd.DataFrame
        Raw diabetes dataset that may contain 0 as a sentinel for missing data.

    Returns
    -------
    pd.DataFrame
        A copy of the input DataFrame with zeros replaced by column medians
        for the physiologically sensitive columns.
    """
    df = df.copy()
    for col in ZERO_IMPUTE_COLS:
        if col in df.columns:
            # Replace 0 with NaN, then fill with the non-zero median
            df[col] = df[col].replace(0, float("nan"))
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    return df
