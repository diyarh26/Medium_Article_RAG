from openai import OpenAI

from rag.config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    EMBEDDING_MODEL,
    CHAT_MODEL,
)


_client = None


def get_client() -> OpenAI:
    global _client

    if not OPENAI_API_KEY:
        raise ValueError("Missing OPENAI_API_KEY in .env")

    if _client is None:
        kwargs = {"api_key": OPENAI_API_KEY}

        if OPENAI_BASE_URL:
            kwargs["base_url"] = OPENAI_BASE_URL

        _client = OpenAI(**kwargs)

    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    client = get_client()

    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=texts,
    )

    return [item.embedding for item in response.data]


def generate_answer(system_prompt: str, user_prompt: str) -> str:
    client = get_client()

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content


def generate_search_query(question: str) -> str:
    client = get_client()

    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite the user question as one short search query for "
                    "retrieving relevant Medium article passages. Do not answer "
                    "the question. Do not invent article titles, authors, or "
                    "specific facts. Prefer keywords and concepts from the user "
                    "question. Return only the search query text."
                ),
            },
            {"role": "user", "content": question},
        ],
    )

    search_query = response.choices[0].message.content or ""
    search_query = search_query.strip().strip("\"'")
    search_query = " ".join(search_query.split())

    return search_query[:300]
