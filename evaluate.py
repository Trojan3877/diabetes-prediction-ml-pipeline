import joblib
from sklearn.metrics import classification_report
from .data_preprocessing import load_data, preprocess_data

def evaluate_model(model_path, data_path="data/diabetes.csv"):
    model = joblib.load(model_path)
    scaler = joblib.load("models/scaler.pkl")

    X, y, df = load_data(data_path)
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)

    report = classification_report(y, preds, output_dict=True)
    return report