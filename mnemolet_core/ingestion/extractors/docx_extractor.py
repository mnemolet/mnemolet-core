from pathlib import Path
from typing import Iterator

from ..loaders.docx_loader import extract_docx
from .base import Extractor


class DocxExtractor(Extractor):
    extensions = {".docx"}

    def extract(self, file: Path) -> Iterator[str]:
        yield from extract_docx(file, self.chunk_size)
