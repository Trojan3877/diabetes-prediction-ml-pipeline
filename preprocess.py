import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from data_loader import load_raw_data, save_processed_data
from pathlib import Path
import joblib


def preprocess():
    """Load, preprocess, scale, and split the dataset using config.yaml."""

    # Load configuration
    config = yaml.safe_load(open("config/config.yaml"))

    raw_path = config["dataset"]["raw_path"]
    processed_path = config["dataset"]["processed_path"]
    test_size = config["preprocessing"]["test_size"]
    random_state = config["preprocessing"]["random_state"]

    df = load_raw_data(raw_path)

    # Separate features/target
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save processed data
    processed_df = pd.DataFrame(X_train_scaled)
    processed_df["Outcome"] = y_train.reset_index(drop=True)

    save_processed_data(processed_df, processed_path)

    # Save test sets
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(X_test_scaled).to_csv("data/processed/X_test.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    # Save scaler
    joblib.dump(scaler, "models/scaler.pkl")

    print("✅ Data preprocessing complete.")


if __name__ == "__main__":
    preprocess()