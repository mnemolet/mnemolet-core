from pathlib import Path
from typing import Iterator

from mnemolet_core.core.ingestion.loaders.odt_loader import extract_odt
from mnemolet_core.core.ingestion.extractors.base import Extractor


class OdtExtractor(Extractor):
    extensions = {".odt"}

    def extract(self, file: Path) -> Iterator[str]:
        yield from extract_odt(file, self.chunk_size)
