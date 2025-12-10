import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path: str):
    df = pd.read_csv(path)

    # Basic cleaning
    df = df.dropna()

    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    return X, y, df

def preprocess_data(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def train_test_split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)