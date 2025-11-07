import importlib
import logging
import pkgutil
from pathlib import Path

from . import __name__ as pkg_name
from . import __path__
from .base import Extractor

# dynamically import all modules in extractors (except base and registry)
for _, modname, _ in pkgutil.iter_modules(__path__):
    if modname not in {"base", "registry"}:
        importlib.import_module(f"{pkg_name}.{modname}")

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
