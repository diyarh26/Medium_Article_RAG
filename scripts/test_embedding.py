from rag.config import EMBEDDING_MODEL
from rag.embeddings import embed_texts


def main():
    print("Testing embedding model...")
    print("Embedding model:", EMBEDDING_MODEL)

    texts = ["This is a tiny test for the Medium RAG assistant."]
    vectors = embed_texts(texts)

    vector = vectors[0]

    print("Embedding successful.")
    print("Number of vectors:", len(vectors))
    print("Vector dimension:", len(vector))
    print("First 5 values:", vector[:5])


if __name__ == "__main__":
    main()