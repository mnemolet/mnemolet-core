from pathlib import Path
from typing import Iterator

from ..loaders.pdf_loader import extract_pdf
from .base import Extractor


class PDFExtractor(Extractor):
    extensions = {".pdf"}

    def extract(self, file: Path) -> Iterator[str]:
        yield from extract_pdf(file, self.chunk_size)
