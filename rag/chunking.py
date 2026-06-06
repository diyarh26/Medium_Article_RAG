def clean_text(value) -> str:
    """
    Convert any value into clean text.
    Handles None, NaN-like values, and weird null characters.
    """
    if value is None:
        return ""

    text = str(value)
    text = text.replace("\x00", " ")
    text = " ".join(text.split())

    return text.strip()


def chunk_text(text: str, chunk_size: int, overlap_ratio: float) -> list[str]:
    """
    Split text into overlapping chunks.

    For now, we use words as an approximate token count.
    This is simple, safe, and enough for the homework.
    """
    text = clean_text(text)
    words = text.split()

    if not words:
        return []

    overlap = int(chunk_size * overlap_ratio)
    step = max(chunk_size - overlap, 1)

    chunks = []

    for start in range(0, len(words), step):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words).strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

    return chunks