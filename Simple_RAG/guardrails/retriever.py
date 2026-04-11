# retriever.py
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DB_PATH = "vectordb/"
_vectordb = None

def _get_vectordb():
    global _vectordb
    if _vectordb is None:
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(
                f"Vector DB not found at '{DB_PATH}'. Run ingestion.py first."
            )
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        _vectordb = FAISS.load_local(
            DB_PATH, embeddings, allow_dangerous_deserialization=True
        )
    return _vectordb

def retrieve(query: str, k: int = 10, score_threshold: float = 1.5) -> str:
    db = _get_vectordb()
    results = db.similarity_search_with_score(query, k=k)
    filtered = [doc for doc, dist in results if dist < score_threshold]
    if not filtered:
        filtered = [doc for doc, _ in results[:3]]
    return "\n\n".join(doc.page_content for doc in filtered)

def get_sample_context() -> str:
    """Returns a small sample from the KB so guardrails can infer the domain."""
    db = _get_vectordb()
    results = db.similarity_search("overview introduction summary", k=3)
    return "\n\n".join(doc.page_content for doc in results)