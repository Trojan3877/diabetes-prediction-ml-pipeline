import yaml
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from pathlib import Path


def train_model():
    """Train ML model using processed dataset & save trained artifact."""

    config = yaml.safe_load(open("config/config.yaml"))

    processed_path = config["dataset"]["processed_path"]
    model_path = config["output"]["model_path"]

    df = pd.read_csv(processed_path)

    X_train = df.drop("Outcome", axis=1)
    y_train = df["Outcome"]

    # Choose model
    if config["model"]["type"] == "random_forest":
        model = RandomForestClassifier(
            n_estimators=config["model"]["n_estimators"],
            max_depth=config["model"]["max_depth"],
            random_state=config["model"]["random_state"],
        )
    else:
        model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    # Save model
    Path("models").mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    print(f"✅ Model trained and saved to {model_path}")


if __name__ == "__main__":
    train_model()