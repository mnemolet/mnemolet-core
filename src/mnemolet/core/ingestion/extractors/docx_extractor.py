from pathlib import Path
from typing import Iterator

from mnemolet.core.ingestion.extractors.base import Extractor
from mnemolet.core.ingestion.loaders.docx_loader import extract_docx


class DocxExtractor(Extractor):
    extensions = {".docx"}

    def extract(self, file: Path) -> Iterator[str]:
        yield from extract_docx(file, self.chunk_size)
