from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.config import DRY_RUN, get_stats
from rag.prompts import SYSTEM_PROMPT, build_user_prompt
from rag.retrieval import retrieve_context
from rag.embeddings import generate_answer


app = FastAPI(title="Medium Article RAG Assistant")


class PromptRequest(BaseModel):
    question: str


def public_context(contexts: list[dict]) -> list[dict]:
    return [
        {
            "article_id": item.get("article_id", ""),
            "title": item.get("title", ""),
            "chunk": item.get("chunk", ""),
            "score": item.get("score", 0.0),
        }
        for item in contexts
    ]


@app.get("/")
def home():
    return {
        "message": "Medium Article RAG Assistant is running.",
        "stats": "/api/stats",
        "prompt": "/api/prompt",
        "docs": "/docs",
    }


@app.get("/api/stats")
def stats():
    return get_stats()


@app.post("/api/prompt")
def prompt(request: PromptRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if DRY_RUN:
        fake_context = [
            {
                "article_id": "dry-run",
                "title": "Dry Run Article",
                "chunk": "This is fake context. DRY_RUN=true means no embedding, Pinecone, or GPT call was made.",
                "score": 0.0,
            }
        ]

        user_prompt = build_user_prompt(question, fake_context)

        return {
            "response": "DRY_RUN is enabled. This is a fake response, so no budget was used.",
            "context": public_context(fake_context),
            "Augmented_prompt": {
                "System": SYSTEM_PROMPT,
                "User": user_prompt,
            },
        }

    contexts = retrieve_context(question)
    user_prompt = build_user_prompt(question, contexts)

    if not contexts:
        response = "I don't know based on the provided Medium articles data."
    else:
        response = generate_answer(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

    return {
        "response": response,
        "context": public_context(contexts),
        "Augmented_prompt": {
            "System": SYSTEM_PROMPT,
            "User": user_prompt,
        },
    }
