import os

from dotenv import load_dotenv


load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing.")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing.")

if not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME is missing.")