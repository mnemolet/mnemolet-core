from pathlib import Path
from pypdf import PdfReader


def extract_pdf(file: Path) -> str:
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text
