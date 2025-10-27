from pathlib import Path


def extract_txt(file: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Yield text chunks from a file.
    """
    with open(file, "r", encoding="utf-8") as f:
        while chunk := f.read(chunk_size):
            yield chunk
