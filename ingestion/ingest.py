import os
import time
import fitz  # PyMuPDF

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pinecone import Pinecone, ServerlessSpec


# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing from .env")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is missing from .env")

if not INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME is missing from .env")


# ---------------------------------------------------------
# 1. Extract text from PDF
# ---------------------------------------------------------

PDF_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "data",
    "Agentic-AI.pdf",
)

print(f"Loading PDF: {PDF_PATH}")

pdf = fitz.open(PDF_PATH)

documents = []

for page_number, page in enumerate(pdf):
    text = page.get_text("text").strip()

    if text:
        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": "Agentic-AI.pdf",
                    "page": page_number + 1,
                },
            )
        )

pdf.close()

print(f"Extracted text from {len(documents)} pages.")


# ---------------------------------------------------------
# 2. Split text into chunks
# ---------------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150,
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")


# ---------------------------------------------------------
# 3. Create Google embedding model
# ---------------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=GOOGLE_API_KEY,
)

print("Embedding model initialized.")


# ---------------------------------------------------------
# 4. Connect to Pinecone
# ---------------------------------------------------------

pc = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = [index.name for index in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    print(f"Creating Pinecone index: {INDEX_NAME}")

    pc.create_index(
        name=INDEX_NAME,
        dimension=3072,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1",
        ),
    )

    # Give Pinecone some time to initialize
    time.sleep(10)

else:
    print(f"Pinecone index already exists: {INDEX_NAME}")


index = pc.Index(INDEX_NAME)

print("Connected to Pinecone.")


# ---------------------------------------------------------
# 5. Generate embeddings and upload vectors
# ---------------------------------------------------------

BATCH_SIZE = 50

# First 100 chunks were already uploaded successfully.
# Resume from chunk 101.
for start in range(100, len(chunks), BATCH_SIZE):

    batch = chunks[start:start + BATCH_SIZE]

    texts = [doc.page_content for doc in batch]

    print(
        f"Embedding chunks "
        f"{start + 1}-{start + len(batch)} "
        f"of {len(chunks)}..."
    )

    vectors = embeddings.embed_documents(texts)

    records = []

    for i, (doc, vector) in enumerate(zip(batch, vectors)):

        vector_id = f"chunk-{start + i}"

        records.append(
            {
                "id": vector_id,
                "values": vector,
                "metadata": {
                    "text": doc.page_content,
                    "source": doc.metadata["source"],
                    "page": doc.metadata["page"],
                },
            }
        )

    index.upsert(vectors=records)

    print(f"Uploaded batch ending at chunk {start + len(batch)}.")


print("\n======================================")
print("PDF INGESTION COMPLETED SUCCESSFULLY")
print("======================================")
print(f"Total pages : {len(documents)}")
print(f"Total chunks: {len(chunks)}")
print(f"Index       : {INDEX_NAME}")