import re


STOP_WORDS = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "article",
    "articles",
    "because",
    "before",
    "being",
    "between",
    "could",
    "exactly",
    "find",
    "from",
    "give",
    "have",
    "into",
    "list",
    "only",
    "provide",
    "return",
    "should",
    "that",
    "their",
    "there",
    "these",
    "this",
    "those",
    "three",
    "title",
    "titles",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "your",
}


def build_rule_based_query(question: str, max_terms: int = 24) -> str:
    terms = re.findall(r"[A-Za-z0-9][A-Za-z0-9+#._-]*", question)
    selected_terms = []
    seen_terms = set()

    for term in terms:
        normalized_term = term.lower().strip("._-")

        if len(normalized_term) < 3:
            continue

        if normalized_term in STOP_WORDS:
            continue

        if normalized_term in seen_terms:
            continue

        seen_terms.add(normalized_term)
        selected_terms.append(term)

        if len(selected_terms) >= max_terms:
            break

    return " ".join(selected_terms)


def unique_queries(queries: list[str]) -> list[str]:
    unique = []
    seen = set()

    for query in queries:
        clean_query = " ".join(str(query).split())
        query_key = clean_query.lower()

        if not clean_query or query_key in seen:
            continue

        seen.add(query_key)
        unique.append(clean_query)

    return unique
