"""
Evaluation pipeline: loads a trained model and reports classification metrics.
"""

import os
import joblib
from sklearn.metrics import classification_report

from src.data_preprocessing import load_data, preprocess_data
from src.train import train_model


def evaluate(
    model_name: str = "random_forest",
    data_path: str = "data/sample.csv",
    models_dir: str = "models",
) -> dict:
    """Evaluate a saved model against *data_path*.

    If the requested model has not been trained yet, ``train_model`` is called
    automatically so that the function is always runnable in isolation.

    Parameters
    ----------
    model_name : str
        Base name of the model file (without the ``.pkl`` suffix).
    data_path : str
        Path to the evaluation CSV dataset.
    models_dir : str
        Directory that contains the saved ``.pkl`` files.

    Returns
    -------
    dict
        classification_report output as a nested dictionary.
    """
    model_path = os.path.join(models_dir, f"{model_name}.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    # Auto-train if model artifacts are not present
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        train_model(data_path=data_path, output_dir=models_dir)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    X, y, _ = load_data(data_path)
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)
    report = classification_report(y, preds, output_dict=True, zero_division=0)
    return report


def main():
    report = evaluate()
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            print(f"\n{label}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        else:
            print(f"\n{label}: {metrics:.4f}")


if __name__ == "__main__":
    main()
