from pathlib import Path
from pypdf import PdfReader
from .utils import hash_file


def load_pdf_files(dir: Path) -> list[dict]:
    """
    Load and extract text from all PDF files from a given directory.

    Args:
        directory (Path): dir containing PDF files.

    Returns:
        list[dict]: list of dicts with extracted text
        {
            "path": str,
            "content": str,
        }
    """
    pdf_data = []
    for p in Path(dir).rglob("*.pdf"):
        try:
            reader = PdfReader(p)
            text = ""
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            file_hash = hash_file(p)
            data = {
                "path": str(p.resolve()),
                "content": text.strip(),
                "hash": file_hash,
            }
            pdf_data.append(data)
            print(f"Loaded PDF: {p.name} ({len(reader.pages)} pages)")
        except Exception as e:
            print(f"Error reading {p.name}: {e}")
    return pdf_data
