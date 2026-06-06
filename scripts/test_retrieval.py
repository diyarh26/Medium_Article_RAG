from rag.config import TOP_K, PINECONE_NAMESPACE
from rag.retrieval import retrieve_context


def main():
    question = "Find an article about coronavirus and the brain."

    print("Testing retrieval...")
    print("Question:", question)
    print("Namespace:", PINECONE_NAMESPACE)
    print("Top K:", TOP_K)
    print("=" * 80)

    contexts = retrieve_context(question)

    print("Number of retrieved chunks:", len(contexts))
    print("=" * 80)

    for i, item in enumerate(contexts, start=1):
        print(f"Result {i}")
        print("Score:", item["score"])
        print("Article ID:", item["article_id"])
        print("Title:", item["title"])
        print("Authors:", item["authors"])
        print("Chunk preview:")
        print(item["chunk"][:500])
        print("-" * 80)


if __name__ == "__main__":
    main()