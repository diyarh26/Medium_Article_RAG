from rag.prompts import SYSTEM_PROMPT, build_user_prompt
from rag.retrieval import retrieve_context
from rag.embeddings import generate_answer


def main():
    question = "Find an article about coronavirus and the brain. Provide the title and author."

    print("Testing full RAG answer...")
    print("Question:", question)
    print("=" * 80)

    contexts = retrieve_context(question)

    print("Retrieved contexts:", len(contexts))
    for i, item in enumerate(contexts, start=1):
        print(f"{i}. {item['title']} | {item['authors']} | score={item['score']}")

    print("=" * 80)

    user_prompt = build_user_prompt(question, contexts)

    answer = generate_answer(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )

    print("MODEL ANSWER:")
    print(answer)


if __name__ == "__main__":
    main()