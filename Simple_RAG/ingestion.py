import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DATA_PATH = "data/"
DB_PATH = "vectordb/"

def load_documents():
    docs = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            docs.extend(loader.load())
    return docs

# ingestion.py  (only chunk_documents changes)
def chunk_documents(docs):
    # Strip excessive whitespace that confuses embeddings
    for doc in docs:
        doc.page_content = " ".join(doc.page_content.split())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,   # increased overlap catches split answers
    )
    return splitter.split_documents(docs)

def create_vectorstore(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(DB_PATH)

    print("✅ Vector DB created and saved!")

if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    create_vectorstore(chunks)