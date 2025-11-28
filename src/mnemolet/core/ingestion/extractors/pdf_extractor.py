from pathlib import Path
from typing import Iterator

from mnemolet.core.ingestion.extractors.base import Extractor
from mnemolet.core.ingestion.loaders.pdf_loader import extract_pdf


class PDFExtractor(Extractor):
    extensions = {".pdf"}

    def extract(self, file: Path) -> Iterator[str]:
        yield from extract_pdf(file, self.chunk_size)
