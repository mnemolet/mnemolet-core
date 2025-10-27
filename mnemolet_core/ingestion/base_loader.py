from pathlib import Path
from .utils import hash_file
from collections.abc import Iterator
from .loader_registry import get_extractor

def stream_files(dir: Path) -> Iterator[dict[str, str, str]]:
    """
    Yield for supported file types.
    """
    for file in dir.rglob("*"):
        file_hash = hash_file(file)
        extractor = get_extractor(file)
        if not extractor:
            continue
        try:
            for content_part in extractor(file):
                data = {
                    "path": str(file.resolve()),
                    "content": content_part,
                    "hash": file_hash,
                }
            yield data
        except Exception as e:
            print(f"Skipping {file}: {e}")

