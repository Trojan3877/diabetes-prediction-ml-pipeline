"""
RAG (Retrieval-Augmented Generation) pipeline for diabetes prediction.

Builds a ChromaDB vector store from dataset statistics and GPT-4.1-generated
medical research summaries, then retrieves relevant context to augment
LLM-based explanations of model predictions.

Requires OPENAI_API_KEY to be set in the environment.
"""

import json
import os

import chromadb
import pandas as pd
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHROMA_DIR = "vectorstore/chroma_db"

# Use the modern persistent client API (chromadb >= 0.4.0).
# The legacy duckdb+parquet settings are no longer supported.
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)


def load_or_create_collection(name: str = "diabetes_rag"):
    """Load a ChromaDB collection or create one if it does not exist."""
    try:
        return chroma_client.get_collection(name=name)
    except Exception:
        return chroma_client.create_collection(name=name)


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate OpenAI embeddings using text-embedding-3-small."""
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small",
    )
    return [item.embedding for item in response.data]


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 120) -> list[str]:
    """Split *text* into overlapping chunks for embedding.

    Parameters
    ----------
    text : str
        Full text to split.
    chunk_size : int
        Maximum character length of each chunk.
    overlap : int
        Number of characters that consecutive chunks share.

    Returns
    -------
    list[str]
        List of text chunks.
    """
    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap

    return chunks


def prepare_medical_research_context() -> str:
    """Generate a medical research summary using GPT-4.1.

    This is intended to run once during vector-store creation; the summary is
    then chunked and embedded so it can be retrieved at inference time.
    """
    prompt = """
You are a biomedical research summarization assistant.
Give a concise, evidence-based summary of:

1. Key biomarkers associated with Type 2 diabetes risk
2. Clinical risk factors (BMI, glucose, age, insulin resistance, pregnancy-related factors)
3. Early intervention strategies supported by research
4. Sources should be from credible medical literature (you may paraphrase)

Write in short paragraphs, clear and factual.
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )

    # Use attribute access (.content) — dict-style access is not supported
    # by the current openai SDK.
    return response.choices[0].message.content


def build_vectorstore(
    data_path: str = "data/diabetes.csv",
    collection_name: str = "diabetes_rag",
) -> bool:
    """Build the full RAG index.

    Combines dataset-level statistics, per-feature summaries, and a
    GPT-4.1-generated medical research summary into a ChromaDB collection.

    Parameters
    ----------
    data_path : str
        Path to the CSV dataset used to compute statistics.
    collection_name : str
        Name of the ChromaDB collection to populate.

    Returns
    -------
    bool
        ``True`` on success.
    """
    df = pd.read_csv(data_path)
    col = load_or_create_collection(collection_name)

    # 1. Add dataset-level chunks
    dataset_summary = df.describe().to_string()
    dataset_chunks = chunk_text(dataset_summary)
    dataset_embeddings = generate_embeddings(dataset_chunks)

    for i, chunk in enumerate(dataset_chunks):
        col.add(
            documents=[chunk],
            embeddings=[dataset_embeddings[i]],
            ids=[f"dataset_chunk_{i}"],
        )

    # 2. Add feature-level summaries
    feature_chunks: list[str] = []
    for feature in df.columns:
        if feature != "Outcome":
            stats = df[feature].describe().to_dict()
            summary_text = (
                f"Feature: {feature}\nStatistics: {json.dumps(stats, indent=2)}"
            )
            feature_chunks.extend(chunk_text(summary_text))

    feature_embeddings = generate_embeddings(feature_chunks)
    for i, chunk in enumerate(feature_chunks):
        col.add(
            documents=[chunk],
            embeddings=[feature_embeddings[i]],
            ids=[f"feature_chunk_{i}"],
        )

    # 3. Add GPT-4.1 medical research summary
    medical_summary = prepare_medical_research_context()
    medical_chunks = chunk_text(medical_summary)
    medical_embeddings = generate_embeddings(medical_chunks)

    for i, chunk in enumerate(medical_chunks):
        col.add(
            documents=[chunk],
            embeddings=[medical_embeddings[i]],
            ids=[f"medical_chunk_{i}"],
        )

    # PersistentClient persists automatically; no explicit .persist() call needed.
    return True


def retrieve_context(
    query: str,
    collection_name: str = "diabetes_rag",
    n_results: int = 4,
) -> str:
    """Retrieve the most relevant RAG context for *query*.

    Parameters
    ----------
    query : str
        Natural-language query to embed and search for.
    collection_name : str
        ChromaDB collection to query.
    n_results : int
        Number of chunks to retrieve.

    Returns
    -------
    str
        Retrieved document chunks joined by double newlines.
    """
    col = load_or_create_collection(collection_name)
    embedding = generate_embeddings([query])[0]

    results = col.query(
        query_embeddings=[embedding],
        n_results=n_results,
    )

    documents = results.get("documents", [[]])[0]
    return "\n\n".join(documents)
