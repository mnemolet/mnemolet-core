from pathlib import Path
from typing import Iterator

from mnemolet_core.core.ingestion.extractors.base import Extractor
from mnemolet_core.core.ingestion.loaders.docx_loader import extract_docx


class DocxExtractor(Extractor):
    extensions = {".docx"}

    def extract(self, file: Path) -> Iterator[str]:
        yield from extract_docx(file, self.chunk_size)
