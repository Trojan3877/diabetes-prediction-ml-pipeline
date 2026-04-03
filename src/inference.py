"""
LLM-powered inference helpers.

Uses GPT-4.1 (via the OpenAI API) to generate human-readable explanations
for model predictions and evaluation metrics, optionally augmented with
context retrieved from the RAG vector store.

Requires the OPENAI_API_KEY environment variable to be set.
"""

import json
import os

import joblib
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_model(model_name: str = "random_forest", models_dir: str = "models"):
    """Load a trained model and its associated scaler from disk.

    Parameters
    ----------
    model_name : str
        Base name of the model file (without the ``.pkl`` suffix).
    models_dir : str
        Directory that contains the ``.pkl`` files.

    Returns
    -------
    tuple[Any, StandardScaler]
        The loaded model and its fitted scaler.

    Raises
    ------
    FileNotFoundError
        If the model file does not exist at the expected path.
    """
    model_path = os.path.join(models_dir, f"{model_name}.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler


def explain_prediction_with_llm(
    features: dict,
    prediction: int,
    probability: float,
    rag_context: str = "",
) -> str:
    """Use GPT-4.1 to explain a single model prediction.

    Parameters
    ----------
    features : dict
        Patient feature values used for the prediction.
    prediction : int
        Binary model output (1 = high risk, 0 = low risk).
    probability : float
        Confidence score returned by the model.
    rag_context : str, optional
        Additional context retrieved from the RAG vector store.

    Returns
    -------
    str
        A human-readable explanation of the prediction.
    """
    prompt = f"""
You are a medical AI assistant with expertise in diabetes risk analysis.

INPUT DATA:
- Patient features: {json.dumps(features, indent=2)}
- Model binary prediction (1 = high risk, 0 = low risk): {prediction}
- Model confidence: {round(probability, 4)}

RAG CONTEXT (Medical research summary and dataset insights):
{rag_context}

TASKS:
1. Explain the prediction in clear, human-friendly language.
2. Identify which features contributed the most to this prediction.
3. Provide a short clinical interpretation grounded in the RAG research.
4. Give practical next-step recommendations (NOT medical advice — just general wellness insights).
5. Summarize the reliability and limitations of this prediction.

Your tone should be professional, concise, and accessible.
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
    )

    # Use attribute access (.content) instead of dict-style access
    return response.choices[0].message.content


def explain_model_metrics(metrics: dict, rag_context: str = "") -> str:
    """Use GPT-4.1 to interpret model evaluation metrics.

    Parameters
    ----------
    metrics : dict
        Evaluation metrics dictionary (e.g. from classification_report).
    rag_context : str, optional
        Additional context retrieved from the RAG vector store.

    Returns
    -------
    str
        A structured professional interpretation of the metrics.
    """
    prompt = f"""
You are an AI model auditor specializing in medical ML systems.

MODEL EVALUATION METRICS:
{json.dumps(metrics, indent=2)}

RAG CONTEXT (clinical evidence + dataset patterns):
{rag_context}

TASK:
Provide a detailed explanation of:
- Strengths of the model
- Weaknesses or biases
- How well this model may generalise to different populations
- Common failure modes in diabetes prediction models
- What improvements could meaningfully increase accuracy

Write this in a structured, professional way.
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content
