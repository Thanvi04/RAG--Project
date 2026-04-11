from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
llm = ChatGroq(model=MODEL_NAME, temperature=0, max_tokens=16)

guardrail_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a strict query classifier for a domain-specific chatbot.
You will be given a knowledge base sample and a user query.

Decide if the query is relevant to the knowledge base content.

Reply with ONLY one word:
- ALLOWED
- BLOCKED

No explanation. No extra text. Just one word."""),
    ("human", """Knowledge base sample:
{context}

User query: {query}""")
])

guardrail_chain = guardrail_prompt | llm

# Generate block message separately with strict instructions
block_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a strict chatbot assistant.
Tell the user in one sentence that you cannot help with their question.
Mention what topics you can help with based on the knowledge base.
Do NOT answer the user's question under any circumstances.
Do NOT provide any information related to their query."""),
    ("human", """Knowledge base sample:
{context}

User query: {query}""")
])

block_chain = block_prompt | llm

def is_allowed(query: str, context_sample: str) -> tuple[bool, str]:
    result = guardrail_chain.invoke({
        "context": context_sample,
        "query": query
    })
    decision = result.content.strip().upper()

    if "ALLOWED" in decision:
        return True, ""

    block_response = block_chain.invoke({
        "context": context_sample,
        "query": query
    })
    return False, block_response.content.strip()