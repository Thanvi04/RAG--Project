
# RAG-Based Arecanut Information Chatbot

## Overview

This project is a Retrieval-Augmented Generation (RAG) chatbot that answers user questions using information extracted from an arecanut PDF document. It combines semantic search with a Large Language Model (LLM) to generate context-based responses.

The application uses Streamlit to provide a chat interface, FAISS to retrieve relevant document chunks, Hugging Face embeddings to convert text into numerical vectors, and Groq to generate answers using an LLM.

The chatbot is designed to answer questions based on the available document context. If the requested information is not available, it is instructed to respond with "I don't know."

## Objectives

- Build a question-answering chatbot using RAG.
- Extract and process information from PDF documents.
- Convert document chunks into numerical embeddings.
- Retrieve relevant information using vector similarity search.
- Generate context-based answers using an LLM.
- Provide an interactive chatbot interface using Streamlit.

## Technologies Used

- **Programming Language:** Python
- **Frontend:** Streamlit
- **LLM Integration:** Groq API
- **Language Model:** Llama 4 Scout (configurable)
- **Embeddings:** Hugging Face `all-MiniLM-L6-v2`
- **Vector Database:** FAISS
- **Framework:** LangChain
- **PDF Processing:** PyPDF
- **Environment Configuration:** python-dotenv

## How It Works

1. **Document Loading:** The PDF document is loaded using PyPDFLoader.
2. **Text Chunking:** The extracted text is divided into smaller chunks using RecursiveCharacterTextSplitter, with a chunk size of 1000 characters and an overlap of 150 characters.
3. **Embedding Generation:** Hugging Face's `all-MiniLM-L6-v2` model converts the text chunks into numerical vectors.
4. **Vector Storage:** FAISS stores the embeddings and supports efficient similarity search.
5. **User Query:** The user enters a question through the Streamlit chat interface.
6. **Information Retrieval:** The query is compared with stored document embeddings to retrieve relevant chunks.
7. **Answer Generation:** The retrieved context and user question are passed to the Groq-hosted LLM through LangChain.
8. **Response Display:** The generated answer is displayed in the chat interface.

## Architecture

```text
PDF Document
     |
     v
Document Loading
     |
     v
Text Chunking
     |
     v
Hugging Face Embeddings
     |
     v
FAISS Vector Database
     |
     v
User Query
     |
     v
Similarity Search
     |
     v
Relevant Context
     |
     v
Groq LLM
     |
     v
Generated Answer
     |
     v
Streamlit Chat Interface
```

## Key Features

- PDF-based question answering.
- Semantic retrieval using vector embeddings.
- FAISS-based similarity search.
- Context-grounded answer generation.
- Interactive chat interface using Streamlit.
- Chat history maintained during the session.
- Configurable Groq model through environment variables.
- Retrieval fallback that returns the top three results when no chunks meet the configured distance threshold.

## Project Structure

```text
Simple_RAG/
|
|-- data/
|   |-- arecanut.pdf
|
|-- vectordb/
|   |-- index.faiss
|   |-- index.pkl
|
|-- main.py
|-- ingestion.py
|-- retriever.py
|-- generator.py
|-- requirements.txt
|-- .gitignore
|
|-- guardrails/
    |-- guard.py
    |-- retriever.py
    |-- generator.py
```

## Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python version supported by the dependencies.
- A Groq API key.
- Internet access for downloading dependencies and the embedding model.

### Step 1: Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd RAG--Project-main/Simple_RAG
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with your actual repository URL. Adjust the directory name if your cloned repository has a different name.

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
venv\Scripts\Activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure the Groq API Key

Create a file named `.env` in the `Simple_RAG` directory and add:

```env
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
```

Replace `your_groq_api_key` with your actual API key. Keep `.env` private and never commit API keys to GitHub.

### Step 5: Create the Vector Database

Run the ingestion script from the `Simple_RAG` directory:

```bash
python ingestion.py
```

This script loads PDF files from the `data/` directory, splits their contents into chunks, generates embeddings, and saves the FAISS vector database in `vectordb/`.

Run ingestion again whenever you change the source documents.

### Step 6: Launch the Chatbot

```bash
streamlit run main.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser to interact with the chatbot.

## Guardrails and Query Handling

The project also contains an additional implementation under the `guardrails/` directory.

This implementation includes:

- **Query Rewriting:** Uses the previous question to clarify follow-up questions.
- **Relevance Checking:** Uses an LLM to determine whether a query is relevant to the knowledge base.
- **Restricted Responses:** Generates a refusal message for queries classified as outside the knowledge base.
- **Context-Based Answers:** Instructs the model to answer using retrieved context and respond with "I don't know" when the answer is unavailable.

The standard `main.py` imports the root-level `generator.py`. The guardrails implementation is separate and is not automatically activated by running `streamlit run main.py`.

## Applications

- Agricultural information retrieval.
- Question answering over PDF documents.
- Knowledge-base chatbots.
- Document search and summarization.
- Domain-specific conversational assistants.

## Future Enhancements

- Support multiple PDF documents and knowledge sources.
- Display source references alongside answers.
- Improve retrieval using reranking techniques.
- Add persistent chat history.
- Build a more robust evaluation pipeline.
- Integrate additional document formats.

## Limitations

- Responses depend on the quality and completeness of the source documents.
- The chatbot may not answer questions that are not covered by the knowledge base.
- Answer quality depends on retrieval accuracy and LLM behavior.
- A valid Groq API key and access to the configured model are required.
- The vector database must be generated or supplied before retrieval can work.

## Author

**Thanvi Rai N**

GitHub: [Thanvi04](https://github.com/Thanvi04)
