from pathlib import Path
from typing import Iterator

from ..loaders.odt_loader import extract_odt
from .base import Extractor


class OdtExtractor(Extractor):
    extensions = {".odt"}

    def extract(self, file: Path) -> Iterator[str]:
        yield from extract_odt(file, self.chunk_size)
