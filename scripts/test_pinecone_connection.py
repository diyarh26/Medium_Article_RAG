from rag.config import PINECONE_INDEX
from rag.pinecone_client import get_pinecone_index


def main():
    print("Testing Pinecone connection...")
    print("Index name:", PINECONE_INDEX)

    index = get_pinecone_index()
    stats = index.describe_index_stats()

    print("Connection successful.")
    print(stats)


if __name__ == "__main__":
    main()