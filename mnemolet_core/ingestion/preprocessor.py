def chunk_text(text: str, max_length: int = 500) -> list[str]:
    """
    Split large text into smaller chunks (by words).

    Args:
        text: input text
        max_length: chunk size in words
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_length):
        chunk = " ".join(words[i : i + max_length])
        chunks.append(chunk)
    return chunks
