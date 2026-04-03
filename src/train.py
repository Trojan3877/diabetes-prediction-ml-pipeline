"""
Training pipeline: trains Logistic Regression and Random Forest classifiers
on the diabetes dataset and persists them to disk.
"""

import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_preprocessing import load_data, preprocess_data, train_test_split_data


def train_model(
    data_path: str = "data/sample.csv",
    output_dir: str = "models",
) -> dict:
    """Train classifiers and save models to *output_dir*.

    Parameters
    ----------
    data_path : str
        Path to the CSV dataset.
    output_dir : str
        Directory where trained model files (.pkl) are saved.

    Returns
    -------
    dict
        Nested dictionary mapping model name → evaluation metrics dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    X, y, _ = load_data(data_path)
    X_scaled, scaler = preprocess_data(X)
    X_train, X_test, y_train, y_test = train_test_split_data(X_scaled, y)

    models = {
        "logistic_regression": LogisticRegression(max_iter=500),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results[name] = {
            "accuracy": accuracy_score(y_test, preds),
            # zero_division=0 avoids warnings on tiny test splits
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
        }

        joblib.dump(model, os.path.join(output_dir, f"{name}.pkl"))

    joblib.dump(scaler, os.path.join(output_dir, "scaler.pkl"))
    return results


def main():
    results = train_model()
    for name, metrics in results.items():
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
