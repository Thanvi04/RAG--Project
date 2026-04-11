import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from retriever import retrieve, get_sample_context
from guardrails.guard import is_allowed

load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
llm = ChatGroq(model=MODEL_NAME, temperature=0, max_tokens=512)

# --- Query rewriter ---
rewrite_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a query rewriter.
- If the current question is vague or refers to a previous question (uses words like 'it', 'this', 'more', 'explain', 'elaborate' etc.), rewrite it into a clear standalone question using the previous question as context.
- If the current question is already clear and standalone, return it exactly as is.
- Return ONLY the rewritten question. No explanation."""),
    ("human", "Previous question: {last_query}\nCurrent question: {current_query}")
])

rewrite_chain = rewrite_prompt | llm

# --- Answer prompt ---
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant.
- Answer ONLY from the given context.
- If the answer is not in the context, say exactly: I don't know.
- Be concise but NEVER truncate lists — always include every item found in context.
- When asked for varieties, types, diseases, or pests, list ALL of them."""),
    ("human", "Context:\n{context}\n\nQuestion:\n{question}")
])

answer_chain = answer_prompt | llm

# Load domain sample once at startup for guardrails
_domain_sample = None

def get_domain_sample() -> str:
    global _domain_sample
    if _domain_sample is None:
        _domain_sample = get_sample_context()
    return _domain_sample


def generate_answer(query: str, last_query: str = "") -> str:
    # Step 1: Guardrail — LLM checks relevance against actual KB content
    domain_sample = get_domain_sample()
    allowed, block_message = is_allowed(query, domain_sample)
    if not allowed:
        return block_message

    # Step 2: Rewrite vague query using last question
    if last_query:
        rewritten = rewrite_chain.invoke({
            "last_query": last_query,
            "current_query": query
        })
        effective_query = rewritten.content.strip()
    else:
        effective_query = query

    # Step 3: Retrieve relevant chunks
    context = retrieve(effective_query)

    if not context.strip():
        return "I don't know."

    # Step 4: Generate answer
    response = answer_chain.invoke({
        "context": context,
        "question": effective_query
    })
    return response.content