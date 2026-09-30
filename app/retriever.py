from typing import List, Dict

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone

from app.config import (
    GOOGLE_API_KEY,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)


# Initialize embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

# Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)


def retrieve_documents(
    question: str,
    top_k: int = 5,
) -> tuple[List[Dict], float]:

    # Convert the question into an embedding
    query_vector = embeddings.embed_query(question)

    # Search Pinecone
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
    )

    contexts = []

    scores = []

    for match in results.matches:

        score = float(match.score)
        scores.append(score)

        metadata = match.metadata or {}

        contexts.append(
            {
                "text": metadata.get("text", ""),
                "source": metadata.get("source", "Agentic-AI.pdf"),
                "page": metadata.get("page"),
                "score": round(score, 4),
            }
        )

    # Average similarity score
    confidence = (
        sum(scores) / len(scores)
        if scores
        else 0.0
    )

    return contexts, round(confidence, 4)