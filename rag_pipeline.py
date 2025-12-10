import os
import json
import pandas as pd
import chromadb
from chromadb.config import Settings
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

CHROMA_DIR = "vectorstore/chroma_db"

# Initialize Chroma client
chroma_client = chromadb.Client(
    Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory=CHROMA_DIR
    )
)


def load_or_create_collection(name="diabetes_rag"):
    """Load a ChromaDB collection or create one if it does not exist."""
    try:
        return chroma_client.get_collection(name=name)
    except:
        return chroma_client.create_collection(name=name)


def generate_embeddings(texts):
    """Generate OpenAI embeddings using text-embedding-3-small."""
    response = client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]


def chunk_text(text, chunk_size=600, overlap=120):
    """Split text into overlapping chunks for embedding."""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks


def prepare_medical_research_context():
    """
    Creates high-quality medical summaries using GPT-4.1.
    This only needs to run ONE time during vector store creation.
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
        temperature=0.2
    )

    return response.choices[0].message["content"]


def build_vectorstore(data_path="data/diabetes.csv", collection_name="diabetes_rag"):
    """
    This builds the full RAG index combining:
    - dataset patterns
    - GPT-4.1 clinical research summaries
    - feature statistics
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
            ids=[f"dataset_chunk_{i}"]
        )

    # 2. Add feature-level summaries
    feature_summaries = []
    for feature in df.columns:
        if feature != "Outcome":
            stats = df[feature].describe().to_dict()
            summary_text = f"Feature: {feature}\nStatistics: {json.dumps(stats, indent=2)}"
            feature_summaries.append(summary_text)

    feature_chunks = []
    for fs in feature_summaries:
        feature_chunks.extend(chunk_text(fs))

    feature_embeddings = generate_embeddings(feature_chunks)

    for i, chunk in enumerate(feature_chunks):
        col.add(
            documents=[chunk],
            embeddings=[feature_embeddings[i]],
            ids=[f"feature_chunk_{i}"]
        )

    # 3. Add GPT-4.1 medical research summary
    medical_summary = prepare_medical_research_context()
    medical_chunks = chunk_text(medical_summary)
    medical_embeddings = generate_embeddings(medical_chunks)

    for i, chunk in enumerate(medical_chunks):
        col.add(
            documents=[chunk],
            embeddings=[medical_embeddings[i]],
            ids=[f"medical_chunk_{i}"]
        )

    chroma_client.persist()
    return True


def retrieve_context(query, collection_name="diabetes_rag", n_results=4):
    """Retrieve RAG context for an LLM explanation."""
    col = load_or_create_collection(collection_name)
    embedding = generate_embeddings([query])[0]

    results = col.query(
        query_embeddings=[embedding],
        n_results=n_results
    )

    documents = results.get("documents", [[]])[0]
    return "\n\n".join(documents)