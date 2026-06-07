from rag.config import TOP_K, PINECONE_NAMESPACE
from rag.embeddings import embed_texts
from rag.pinecone_client import get_pinecone_index
from rag.query_expansion import build_rule_based_query, unique_queries


def match_to_context(match) -> dict:
    metadata = match.metadata or {}

    return {
        "article_id": str(metadata.get("article_id", "")),
        "title": metadata.get("title", ""),
        "authors": metadata.get("authors", ""),
        "chunk": metadata.get("chunk", ""),
        "score": float(match.score),
    }


def build_retrieval_queries(question: str) -> list[str]:
    rule_based_query = build_rule_based_query(question)

    return unique_queries([question, rule_based_query])


def retrieve_context_for_queries(queries: list[str], top_k: int = TOP_K) -> list[dict]:
    queries = unique_queries(queries)

    if not queries:
        return []

    query_vectors = embed_texts(queries)

    index = get_pinecone_index()

    # Retrieve more than top_k because some results may be chunks from the same article.
    raw_top_k = min(top_k * 4, 30)
    results_by_query = []

    for query_vector in query_vectors:
        result = index.query(
            vector=query_vector,
            top_k=raw_top_k,
            include_metadata=True,
            namespace=PINECONE_NAMESPACE,
        )
        results_by_query.append([match_to_context(match) for match in result.matches])

    contexts = []
    seen_article_ids = set()

    for rank in range(raw_top_k):
        for query_results in results_by_query:
            if rank >= len(query_results):
                continue

            context = query_results[rank]
            article_id = context["article_id"]

            if not article_id or article_id in seen_article_ids:
                continue

            seen_article_ids.add(article_id)
            contexts.append(context)

            if len(contexts) >= top_k:
                return contexts

    return contexts


def retrieve_context(question: str, top_k: int = TOP_K) -> list[dict]:
    queries = build_retrieval_queries(question)
    return retrieve_context_for_queries(queries, top_k=top_k)
