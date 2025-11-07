import logging
from pathlib import Path

from .base import Extractor

logger = logging.getLogger(__name__)

EXTRACTOR_REGISTRY = {
    ext: cls() for cls in Extractor.__subclasses__() for ext in cls.extensions
}

logger.debug("EXTRACTOR_REGISTRY: ", sorted(EXTRACTOR_REGISTRY.keys()))


def get_extractor(file: Path) -> Extractor | None:
    """
    Return registered extractor for the given file extension.
    """
    return EXTRACTOR_REGISTRY.get(file.suffix.lower())
