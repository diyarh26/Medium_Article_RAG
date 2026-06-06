from rag.config import CHUNK_SIZE, OVERLAP_RATIO
from rag.chunking import chunk_text


sample_text = """
This is a sample Medium article about education, habits, productivity, and learning.
The goal is to test whether our chunking function works correctly before using the real dataset.
We want to split a long article into smaller pieces so each piece can later be embedded and stored
inside Pinecone. This test does not call OpenAI, does not call Pinecone, and does not spend money.
""" * 80


chunks = chunk_text(
    text=sample_text,
    chunk_size=CHUNK_SIZE,
    overlap_ratio=OVERLAP_RATIO,
)

print("Chunk size:", CHUNK_SIZE)
print("Overlap ratio:", OVERLAP_RATIO)
print("Number of chunks:", len(chunks))
print()
print("First chunk preview:")
print(chunks[0][:500])
print()
print("Last chunk preview:")
print(chunks[-1][:500])