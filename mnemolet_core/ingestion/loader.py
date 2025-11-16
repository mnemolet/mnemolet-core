import logging
from collections.abc import Iterator
from pathlib import Path

from .extractors.registry import get_extractor
from .utils import hash_file

logger = logging.getLogger(__name__)


def stream_files(dir: Path) -> Iterator[dict[str, str, str]]:
    """
    Yield for supported file types.
    """
    for file in dir.rglob("*"):
        extractor = get_extractor(file)
        logger.debug(f" -> extractor: {extractor}")
        if not extractor:
            continue

        file_hash = hash_file(file)

        try:
            for content_part in extractor.extract(file):
                logger.warning(f"[LOADER] Received part: len={len(content_part)}")
                data = {
                    "path": str(file.resolve()),
                    "content": content_part,
                    "hash": file_hash,
                }
            yield data
        except Exception as e:
            print(f"Skipping {file}: {e}")
