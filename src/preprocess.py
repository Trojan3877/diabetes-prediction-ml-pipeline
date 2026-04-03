"""
Preprocessing entry-point: load raw data, apply cleaning, and scale features.
"""

from src.data_preprocessing import load_data, preprocess_data


def preprocess(data_path: str = "data/sample.csv"):
    """Load, clean, and scale the dataset.

    Parameters
    ----------
    data_path : str
        Path to the raw CSV file.

    Returns
    -------
    tuple[np.ndarray, pd.Series, StandardScaler]
        Scaled feature array, target vector, and fitted scaler.
    """
    X, y, _ = load_data(data_path)
    X_scaled, scaler = preprocess_data(X)
    return X_scaled, y, scaler


if __name__ == "__main__":
    X_scaled, y, scaler = preprocess()
    print(f"Preprocessed {X_scaled.shape[0]} samples with {X_scaled.shape[1]} features.")
