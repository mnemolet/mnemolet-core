from pathlib import Path

from .base import Extractor

EXTRACTOR_REGISTRY = {
    ext: cls() for cls in Extractor.__subclasses__() for ext in cls.extensions
}


def get_extractor(file: Path) -> Extractor | None:
    """
    Return registered extractor for the given file extension.
    """
    return EXTRACTOR_REGISTRY.get(file.suffix.lower())
