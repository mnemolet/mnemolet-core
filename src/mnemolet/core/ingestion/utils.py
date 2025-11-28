import hashlib
from pathlib import Path


def hash_file(path: Path) -> str:
    """
    Return SHA256 hash of file content
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        # reads 8KB (8192), stops when get to b"" (empty byte)
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()
