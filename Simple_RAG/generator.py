# generator.py
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from retriever import retrieve

load_dotenv()

SYSTEM_PROMPT = """You are a helpful AI assistant.
- Answer ONLY from the given context.
- If the answer is not in the context, say exactly: I don't know.
- Be concise but NEVER truncate lists — always include every item found in context."""

MODEL_NAME = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")

llm = ChatGroq(model=MODEL_NAME, temperature=0.2, max_tokens=512)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
])

chain = prompt | llm

def generate_answer(query: str) -> str:
    context = retrieve(query)          # loads vectordb from cache
    response = chain.invoke({"context": context, "question": query})
    return response.content
