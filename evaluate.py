import yaml
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import seaborn as sns


def evaluate():
    """Evaluate model and write results to metrics.md + save plot."""

    config = yaml.safe_load(open("config/config.yaml"))

    model = joblib.load(config["output"]["model_path"])
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    predictions = model.predict(X_test)

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions),
        "ROC AUC": roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
    }

    # Save metrics to file
    with open(config["output"]["metrics_path"], "w") as f:
        f.write("# 📊 Model Evaluation Metrics\n\n")
        for k, v in metrics.items():
            f.write(f"- **{k}:** {v:.4f}\n")

    # Plot confusion matrix
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.savefig(config["output"]["performance_plot"])
    plt.close()

    print("✅ Evaluation complete. Metrics saved & confusion matrix plotted.")


if __name__ == "__main__":
    evaluate()