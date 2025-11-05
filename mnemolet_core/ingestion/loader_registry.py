from pathlib import Path

from .loaders.pdf_loader import extract_pdf

# from collections.abc import Optional, Callable
from .loaders.txt_loader import extract_txt

EXTRACTORS: dict[str, [[Path], str]] = {
    ".txt": extract_txt,
    ".pdf": extract_pdf,
}


def get_extractor(file: Path) -> [[[Path], str]]:
    """
    Returns an extractor fn based on file extension.
    """
    return EXTRACTORS.get(file.suffix.lower())
