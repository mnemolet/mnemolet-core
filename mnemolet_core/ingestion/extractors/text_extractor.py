from pathlib import Path
from typing import Iterator

from .base import Extractor


class TextExtractor(Extractor):
    extensions = {".txt"}

    def extract(self, file: Path) -> Iterator[str]:
        """
        Yield text chunks from a file.
        """
        with open(file, "r", encoding="utf-8") as f:
            while chunk := f.read(self.chunk_size):
                yield chunk
