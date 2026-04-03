"""Root-level evaluation entry point (delegates to src.evaluate)."""

from src.evaluate import evaluate


def evaluate_model(model_name: str = "random_forest", data_path: str = "data/sample.csv") -> dict:
    """Evaluate a trained model. Delegates to src.evaluate.evaluate."""
    return evaluate(model_name=model_name, data_path=data_path)


if __name__ == "__main__":
    report = evaluate_model()
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            print(f"\n{label}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value:.4f}")
        else:
            print(f"\n{label}: {metrics:.4f}")