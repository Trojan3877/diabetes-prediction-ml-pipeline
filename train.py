import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .data_preprocessing import load_data, preprocess_data, train_test_split_data

def train_models(data_path="data/diabetes.csv"):
    X, y, df = load_data(data_path)

    X_scaled, scaler = preprocess_data(X)
    X_train, X_test, y_train, y_test = train_test_split_data(X_scaled, y)

    models = {
        "logistic_regression": LogisticRegression(max_iter=500),
        "random_forest": RandomForestClassifier(n_estimators=200)
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        results[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds)
        }

        joblib.dump(model, f"models/{name}.pkl")

    joblib.dump(scaler, "models/scaler.pkl")

    return results

if __name__ == "__main__":
    print(train_models())