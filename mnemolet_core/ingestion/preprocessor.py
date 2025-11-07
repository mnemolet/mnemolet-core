from .loader import stream_files


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


def process_directory(dir):
    """
    Combine file streaming and chunking.
    """
    for data in stream_files(dir):
        for chunk in chunk_text(data["content"], max_length=500):
            yield {
                "path": data["path"],
                "chunk": chunk,
                "hash": data["hash"],
            }
