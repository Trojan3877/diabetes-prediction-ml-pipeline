"""Root-level training entry point (delegates to src.train)."""

from src.train import train_model


def train_models(data_path: str = "data/sample.csv") -> dict:
    """Train all models. Delegates to src.train.train_model."""
    return train_model(data_path=data_path)


if __name__ == "__main__":
    results = train_models()
    for name, metrics in results.items():
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")