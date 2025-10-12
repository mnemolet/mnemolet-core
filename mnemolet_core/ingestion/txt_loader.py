from pathlib import Path
from .utils import hash_file


def load_txt_files(dir: str) -> list[dict]:
    """
    Recursively load .txt files from a given directory

    Returns:
        List[str]: containing:
        {
            "path": str,
            "content": str,
        }
    """
    texts = []
    for p in Path(dir).rglob("*.txt"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                content = f.read().strip()
            # skip empty files
            if not content:
                continue
            file_hash = hash_file(p)
            texts.append(
                {
                    "path": str(p.resolve()),
                    "content": content,
                    "hash": file_hash,
                }
            )
        except Exception as e:
            print(f"Skipping {p}: {e}")
    return texts
