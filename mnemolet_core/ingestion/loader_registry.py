from pathlib import Path

# from collections.abc import Optional, Callable
from .loaders.txt_loader import extract_txt
from .loaders.pdf_loader import extract_pdf

EXTRACTORS: dict[str, [[Path], str]] = {
    ".txt": extract_txt,
}


def get_extractor(file: Path) -> [[[Path], str]]:
    """
    Returns an extractor fn based on file extension.
    """
    return EXTRACTORS.get(file.suffix.lower())
