from pathlib import Path
from .utils import hash_file


def load_files(dir: Path, pattern: str, extract_content: [[Path], str]) -> list[dict]:
    """
    Recursively load files from a given directory using extract_content.

    Args:
        dir: path to directory
        pattern: glob pattern, eg: *.txt, *.pdf, etc
        extract_content: fn that takes path and returns text content

    Returns:
        list[dict]: containing:
        {
            "path": str,
            "content": str,
            "hash": str,
        }
    """
    files_data = []
    for p in dir.rglob(pattern):
        try:
            content = extract_content(p).strip()
            if not content:
                continue
            file_hash = hash_file(p)
            data = {
                "path": str(p.resolve()),
                "content": content,
                "hash": file_hash,
            }
            files_data.append(data)
            print(f"Loaded {pattern}: {p.name}")
        except Exception as e:
            print(f"Error reading {p}: {e}")
    return files_data
