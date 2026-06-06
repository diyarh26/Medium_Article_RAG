from pinecone import Pinecone

from rag.config import PINECONE_API_KEY, PINECONE_INDEX


_pc = None
_index = None


def get_pinecone_index():
    global _pc, _index

    if not PINECONE_API_KEY:
        raise ValueError("Missing PINECONE_API_KEY in .env")

    if _pc is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)

    if _index is None:
        _index = _pc.Index(PINECONE_INDEX)

    return _index