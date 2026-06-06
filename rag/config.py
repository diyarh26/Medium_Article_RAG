import os
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))
OVERLAP_RATIO = float(os.getenv("OVERLAP_RATIO", "0.15"))
TOP_K = int(os.getenv("TOP_K", "8"))

CHAT_MODEL = os.getenv("CHAT_MODEL", "4UHRUIN-gpt-5-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "4UHRUIN-text-embedding-3-small")
EMBEDDING_DIMENSION = 1536

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "medium-rag")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "medium-test-100")

DRY_RUN = os.getenv("DRY_RUN", "true").lower() == "true"


def get_stats() -> dict:
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K,
    }