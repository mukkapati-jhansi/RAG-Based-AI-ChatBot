from fastapi import FastAPI
from pydantic import BaseModel

from app.graph import rag_graph


app = FastAPI(
    title="RAG-Based AI Chatbot",
    description="A LangGraph + Pinecone + Gemini RAG chatbot grounded in the Agentic AI eBook.",
    version="1.0.0",
)


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str
    retrieval_score: float
    retrieved_context: list


@app.get("/")
def root():
    return {
        "message": "RAG-Based AI Chatbot API is running."
    }


@app.get("/health")
def health():
    return {
        "status": "healthy"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    result = rag_graph.invoke(
        {
            "question": request.question
        }
    )

    return {
        "answer": result["answer"],
        "retrieval_score": result["confidence"],
        "retrieved_context": result["context"],
    }