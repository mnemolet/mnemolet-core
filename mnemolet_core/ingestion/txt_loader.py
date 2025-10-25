from pathlib import Path
from .base_loader import load_files


def extract_txt(file: Path) -> str:
    with open(file, "r", encoding="utf-8") as f:
        return f.read()


def load_txt_files(dir: Path):
    return load_files(dir, "*.txt", extract_txt)
