import os
import pandas as pd
import joblib
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


def load_clothes_data(path: str) -> list[dict]:
    """
    Reads clothes.csv.
    Returns a list of rows, each as a dict (one product).
    """
    data = pd.read_csv(path)
    return data.to_dict(orient='records') 

def load_faq_data(path: str) -> list[dict]:
    """
    Reads faq.joblib.
    Returns a list of FAQs with {"question": ..., "answer": ..., "type": ...}.
    """
    data = joblib.load(path)
    return data

def build_product_text(row: dict) -> str:
    """
    Takes one product row.
    Creates a descriptive string for embedding (combine gender, colour, category, productDisplayName, etc.).
    """
    return (
        f"{row.get('gender', '')} {row.get('baseColour', '')} "
        f"{row.get('articleType', '')}, {row.get('usage', '')} "
        f"{row.get('season', '')} {int(row.get('year', 0))} - "
        f"{row.get('productDisplayName', '')}"
    ).strip() 

def generate_embeddings(texts: list[str]) -> np.ndarray:
    """
    Calls embedding model (sentence-transformers or similar).
    Returns embeddings for a batch of texts.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2") 
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings.astype("float32")

def prepare_documents(data: list[dict], _type: str) -> list[dict]:
    """
    Takes either clothes or FAQ entries.
    Builds a unified format with id, text, metadata (no vector yet).
    """
    documents = []
    if _type.lower() == "clothes":
        for row in data:
            documents.append({
                "id": row["id"],
                "text": build_product_text(row),
                "metadata": row
            })

    elif _type.lower() == "faq":
        for i, entry in enumerate(data):
            documents.append({
                "id": f"faq_{i}",
                "text": entry["question"],
                "metadata": {"answer": entry["answer"], "type": entry.get("type", "")} 
            })
    return documents

def store_in_vector_db(documents: list[dict], index, mapping: dict):
    """
    Stores embeddings in FAISS.
    Maintains a parallel mapping (dict: index → metadata & text).
    """
    texts = [doc["text"] for doc in documents]
    vectors = generate_embeddings(texts)

    start_idx = index.ntotal
    index.add(vectors)

    for i, doc in enumerate(documents):
        mapping[start_idx + i] = {
            "id": doc["id"],
            "text": doc["text"],
            "metadata": doc["metadata"]
        }


def save_index(index, path: str):
    """
    Saves the vector DB index to disk.
    """
    faiss.write_index(index, path)


def load_index(path: str):
    """
    Loads the vector DB index from disk.
    """
    return faiss.read_index(path)


def validate_clothes_row(row: dict) -> bool:
    """
    Ensures required fields are present and valid in a clothes row.
    """
    required = ["id", "gender", "articleType", "baseColour", "year", "productDisplayName"]
    return all(field in row and row[field] not in [None, ""] for field in required)

def validate_faq_entry(entry: dict) -> bool:
    """
    Ensures required fields are present and valid in an FAQ entry.
    """
    required = ["question", "answer"]
    return all(field in entry and entry[field] not in [None, ""] for field in required)


def main():
    # Load data
    clothes = load_clothes_data("data/clothes.csv")
    faqs = load_faq_data("data/faq.joblib")

    # Validate
    clothes = [row for row in clothes if validate_clothes_row(row)]
    faqs = [entry for entry in faqs if validate_faq_entry(entry)]

    # Prepare docs
    clothes_docs = prepare_documents(clothes, "clothes")
    faq_docs = prepare_documents(faqs, "faq")

    all_docs = clothes_docs + faq_docs

    # Init FAISS index (cosine similarity using inner product)
    if not all_docs:
        print("No documents to index.")
        return

    vector_dim = generate_embeddings([all_docs[0]["text"]]).shape[1]
    index = faiss.IndexFlatIP(vector_dim)

    # Metadata mapping
    mapping = {}

    # Store
    store_in_vector_db(all_docs, index, mapping)

    # Save index + mapping
    os.makedirs("vector_store", exist_ok=True)
    save_index(index, "vector_store/index.faiss")
    joblib.dump(mapping, "vector_store/mapping.joblib")

    print(f"Ingestion complete. Indexed {index.ntotal} documents successfully.\n")


if __name__ == "__main__":
    main()
