from pathlib import Path
from typing import Iterator


class Extractor:
    """
    Base extractor class.
    All extractors must define supported extensions and implement extract().
    """

    extensions: set[str] = set()

    def __init__(self, chunk_size: int = 1024 * 1024):
        self.chunk_size = chunk_size

    def extract(self, file: Path) -> Iterator[str]:
        """
        Yield text chunks from file.
        """
        raise NotImplementedError("Subclasses must implement extract()")
