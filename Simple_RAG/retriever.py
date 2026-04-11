# retriever.py
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DB_PATH = "vectordb/"
_vectordb = None  # module-level cache

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

def retrieve(query: str, k: int =10 , score_threshold: float = 1.5) -> str:
    """Return top-k chunks above score_threshold, joined by double newline."""
    db = _get_vectordb()

    # similarity_search_with_score returns (doc, distance) pairs
    results = db.similarity_search_with_score(query, k=k)

    # FAISS returns L2 distance — lower is better; convert to similarity
    # distance 0 = identical, so we keep chunks with distance < threshold
    filtered = [doc for doc, dist in results if dist < score_threshold]

    if not filtered:
        # Fall back to top-3 regardless of score so we always return something
        filtered = [doc for doc, _ in results[:3]]

    return "\n\n".join(doc.page_content for doc in filtered)