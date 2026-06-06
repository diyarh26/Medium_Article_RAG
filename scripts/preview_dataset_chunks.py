from pathlib import Path

import pandas as pd

from rag.config import CHUNK_SIZE, OVERLAP_RATIO
from rag.chunking import chunk_text, clean_text


CSV_PATH = Path("data/medium-english-50mb.csv")


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(
            f"CSV file not found at: {CSV_PATH}\n"
            "Put medium-english-50mb.csv inside the data folder."
        )

    print("Reading CSV...")
    df = pd.read_csv(CSV_PATH)

    print()
    print("CSV loaded successfully.")
    print("Number of rows:", len(df))
    print("Columns:", list(df.columns))
    print()

    required_columns = ["title", "text", "url", "authors", "timestamp", "tags"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    print("Previewing first 3 articles")
    print("Chunk size:", CHUNK_SIZE)
    print("Overlap ratio:", OVERLAP_RATIO)
    print("=" * 80)

    for article_id, row in df.head(3).iterrows():
        title = clean_text(row["title"])
        authors = clean_text(row["authors"])
        text = clean_text(row["text"])

        chunks = chunk_text(text, CHUNK_SIZE, OVERLAP_RATIO)

        print()
        print(f"Article ID: {article_id}")
        print(f"Title: {title}")
        print(f"Authors: {authors}")
        print(f"Text length in words: {len(text.split())}")
        print(f"Number of chunks: {len(chunks)}")

        if chunks:
            print()
            print("First chunk preview:")
            print(chunks[0][:700])

        print("=" * 80)


if __name__ == "__main__":
    main()