from pathlib import Path
from .txt_loader import load_txt_files
from .pdf_loader import load_pdf_files


def load_all_files(dir: Path) -> list[dict]:
    """
    Load all supported file types from a given directory.

    Returns:
        list[dict]
    """
    return load_txt_files(dir) + load_pdf_files(dir)
