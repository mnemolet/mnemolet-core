from pathlib import Path
from pypdf import PdfReader
from .base_loader import load_files


def extract_pdf(file: Path) -> str:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def load_pdf_files(dir: Path):
    return load_files(dir, "*.pdf", extract_pdf)
