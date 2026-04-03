"""
Core data-loading and preprocessing helpers used by the training and
evaluation pipelines.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.preprocessing import clean_missing_values


def load_data(path: str):
    """Load and clean the diabetes CSV dataset.

    Parameters
    ----------
    path : str
        Path to the CSV file.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, pd.DataFrame]
        Feature matrix X, target vector y, and the cleaned full DataFrame.
    """
    df = pd.read_csv(path)

    # Replace physiologically impossible zero values before dropping rows
    df = clean_missing_values(df)

    # Remove any remaining NaN rows
    df = df.dropna()

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    return X, y, df


def preprocess_data(X):
    """Fit a StandardScaler and transform X.

    Returns
    -------
    tuple[np.ndarray, StandardScaler]
        Scaled feature array and the fitted scaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def train_test_split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    """Wrapper around sklearn's train_test_split with project defaults."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
