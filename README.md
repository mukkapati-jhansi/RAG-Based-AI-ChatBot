# RAG-Based AI Chatbot

A retrieval-augmented generation (RAG) chatbot built in Python using **LangGraph, Pinecone, Gemini embeddings, and Gemini LLM**.

The chatbot answers questions strictly using information retrieved from the provided **Agentic AI eBook**. It also returns the retrieved context and retrieval similarity score for transparency.

---

## 1. Project Overview

This project implements a document-grounded AI chatbot using a RAG architecture.

The system:

1. Loads the Agentic AI eBook PDF.
2. Extracts its text.
3. Splits the text into smaller chunks.
4. Generates vector embeddings for each chunk.
5. Stores the embeddings in Pinecone.
6. Retrieves the most relevant chunks for a user question.
7. Uses LangGraph to orchestrate the retrieval and generation workflow.
8. Uses Gemini to generate an answer based only on the retrieved context.
9. Returns the answer, retrieved context, page numbers, and retrieval similarity score.

If the retrieved information is not sufficiently relevant, the chatbot does not generate an answer from general knowledge. Instead, it reports that the information could not be found in the provided knowledge base.

---

## 2. Architecture

```text
                    ┌─────────────────────┐
                    │   Agentic-AI.pdf    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Text Extraction    │
                    │      PyMuPDF         │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Text Chunking     │
                    │ Recursive Splitter  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Gemini Embeddings   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Pinecone       │
                    │    Vector Store     │
                    └──────────┬──────────┘
                               │
                               │ User Question
                               ▼
                    ┌─────────────────────┐
                    │      LangGraph      │
                    │                     │
                    │  Retrieve → Generate│
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Gemini LLM       │
                    │  Grounded Response  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │      /chat          │
                    └─────────────────────┘
```

---

## 3. Tech Stack

* **Python 3.12**
* **LangGraph** — RAG workflow orchestration
* **LangChain** — LLM and embedding integration
* **Gemini** — text embeddings and answer generation
* **Pinecone** — vector database
* **PyMuPDF** — PDF text extraction
* **FastAPI** — REST API
* **Uvicorn** — API server
* **Pydantic** — request/response validation
* **python-dotenv** — environment variable management

---

## 4. Project Structure

```text
rag-based-ai-chatbot/
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── graph.py
│   ├── llm.py
│   ├── main.py
│   ├── prompts.py
│   ├── retriever.py
│   └── state.py
│
├── data/
│   └── Agentic-AI.pdf
│
├── ingestion/
│   ├── __init__.py
│   └── ingest.py
│
├── .env
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── test_rag.py
```

---

## 5. Environment Variables

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=rag-based-ai-chatbot
```

Never commit the real `.env` file to GitHub.

A `.env.example` file is included as a template.

---

## 6. Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd rag-based-ai-chatbot
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Configure the environment variables in `.env`.

---

## 7. PDF Ingestion

Place the provided PDF at:

```text
data/Agentic-AI.pdf
```

Run:

```bash
python ingestion/ingest.py
```

The ingestion pipeline:

* Extracts text from the PDF.
* Splits the text into chunks.
* Generates Gemini embeddings.
* Creates/connects to the Pinecone index.
* Uploads the vectors and metadata to Pinecone.

The current document was processed into **112 text chunks across 59 pages**.

---

## 8. Run the API

Start the FastAPI server:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 9. API Endpoints

### Health Check

```http
GET /
```

Example response:

```json
{
  "message": "RAG-Based AI Chatbot API is running."
}
```

### Health Status

```http
GET /health
```

Example:

```json
{
  "status": "healthy"
}
```

### Chat

```http
POST /chat
```

Request:

```json
{
  "question": "What is Agentic AI?"
}
```

Example response:

```json
{
  "answer": "Agentic AI refers to systems capa
```
