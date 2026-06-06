import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from rag.config import CHUNK_SIZE, OVERLAP_RATIO, PINECONE_NAMESPACE
from rag.chunking import chunk_text, clean_text
from rag.embeddings import embed_texts
from rag.pinecone_client import get_pinecone_index


CSV_PATH = Path("data/medium-english-50mb.csv")


def safe_metadata_value(value) -> str:
    return clean_text(value)[:5000]


def flush_batch(ids, texts, metadatas, namespace, dry_run):
    if not ids:
        return

    if dry_run:
        print(f"[DRY RUN] Would embed and upload {len(ids)} chunks.")
        return

    vectors = embed_texts(texts)

    pinecone_vectors = []
    for vector_id, vector, metadata in zip(ids, vectors, metadatas):
        pinecone_vectors.append(
            {
                "id": vector_id,
                "values": vector,
                "metadata": metadata,
            }
        )

    index = get_pinecone_index()
    index.upsert(vectors=pinecone_vectors, namespace=namespace)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--namespace", type=str, default=PINECONE_NAMESPACE)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV file not found at {CSV_PATH}. "
            "Put medium-english-50mb.csv inside the data folder."
        )

    print("Reading CSV...")
    df = pd.read_csv(CSV_PATH)

    if args.limit > 0:
        df = df.head(args.limit)

    print()
    print("Ingestion started")
    print("Articles selected:", len(df))
    print("Chunk size:", CHUNK_SIZE)
    print("Overlap ratio:", OVERLAP_RATIO)
    print("Namespace:", args.namespace)
    print("Batch size:", args.batch_size)
    print("Dry run:", args.dry_run)
    print("=" * 80)

    index = get_pinecone_index()

    if args.reset and not args.dry_run:
        print(f"Deleting all vectors in namespace: {args.namespace}")
        index.delete(delete_all=True, namespace=args.namespace)

    ids = []
    texts_to_embed = []
    metadatas = []
    total_chunks = 0

    for article_id, row in tqdm(df.iterrows(), total=len(df)):
        title = safe_metadata_value(row.get("title", ""))
        authors = safe_metadata_value(row.get("authors", ""))
        url = safe_metadata_value(row.get("url", ""))
        timestamp = safe_metadata_value(row.get("timestamp", ""))
        tags = safe_metadata_value(row.get("tags", ""))
        text = clean_text(row.get("text", ""))

        chunks = chunk_text(text, CHUNK_SIZE, OVERLAP_RATIO)

        for chunk_index, chunk in enumerate(chunks):
            vector_id = f"article-{article_id}-chunk-{chunk_index}"

            text_for_embedding = f"""
Title: {title}
Authors: {authors}
Tags: {tags}

Article passage:
{chunk}
""".strip()

            metadata = {
                "article_id": str(article_id),
                "title": title,
                "authors": authors,
                "url": url,
                "timestamp": timestamp,
                "tags": tags,
                "chunk_index": chunk_index,
                "chunk": chunk,
            }

            ids.append(vector_id)
            texts_to_embed.append(text_for_embedding)
            metadatas.append(metadata)
            total_chunks += 1

            if len(ids) >= args.batch_size:
                flush_batch(ids, texts_to_embed, metadatas, args.namespace, args.dry_run)
                ids.clear()
                texts_to_embed.clear()
                metadatas.clear()

    flush_batch(ids, texts_to_embed, metadatas, args.namespace, args.dry_run)

    print()
    print("=" * 80)
    print("Ingestion complete.")
    print("Total articles processed:", len(df))
    print("Total chunks:", total_chunks)

    if args.dry_run:
        print("No embeddings were created.")
        print("No Pinecone upload happened.")
        print("Budget used: $0")
    else:
        print("Embeddings created and uploaded to Pinecone.")


if __name__ == "__main__":
    main()