import sqlite3
from pathlib import Path
from datetime import datetime, UTC
from typing import Optional, Tuple, List


DB_PATH = Path.home() / ".mnemolet" / "tracker.db"


CREATE_TABLE_FILES = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE,
    hash TEXT,
    ingested_at TEXT,
    indexed INTEGER DEFAULT 0
);
"""

CREATE_TABLE_EMBEDDINGS_METADATA = """
CREATE TABLE IF NOT EXISTS embeddings_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    embedding_file TEXT NOT NULL,
    num_chunks INTEGER NOT NULL,
    embedding_dim INTEGER NOT NULL,
    model_name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def get_connection() -> sqlite3.Connection:
    """
    Returns a SQLite connection.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Init SQLite db if it does not exist (only once).
    """
    with get_connection() as conn:
        conn.execute(CREATE_TABLE_FILES)
        conn.execute(CREATE_TABLE_EMBEDDINGS_METADATA)


def add_file(path: str, file_hash: str):
    """
    Insert a new file if if does not exist.
    """
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO files (path, hash, ingested_at)
            VALUES (?, ?, ?)
        """,
            (path, file_hash, datetime.now(UTC).isoformat()),
        )


def file_exists(file_hash: str) -> bool:
    """
    Check if file with this hash is already in db.
    """
    with get_connection() as conn:
        curr = conn.execute("SELECT 1 FROM files WHERE hash = ?", (file_hash,))
        return curr.fetchone() is not None


def mark_indexed(file_hash: str):
    """
    Mark file as indexed is Qdrant.
    """
    with get_connection() as conn:
        conn.execute("UPDATE files SET indexed = 1 WHERE hash = ?", (file_hash,))


def list_files(indexed: Optional[bool] = None) -> List[dict]:
    """
    List all tracked files, can be optionally filtered by indexed status.
    """
    query = "SELECT path, hash, ingested_at, indexed FROM files"
    params = ()
    if indexed is not None:
        query += " WHERE indexed = ?"
        params = (1 if indexed else 0,)
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def save_embeddings_metadata(
    embedding_file: str, num_chunks: int, embedding_dim: int, model_name: str
):
    """
    Insert new embedding metadata.
    """
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO embeddings_metadata (embedding_file, num_chunks,
            embedding_dim, model_name)
            VALUES (?, ?, ?, ?)
            """,
            (embedding_file, num_chunks, embedding_dim, model_name),
        )


def load_embeddings_metadata() -> Optional[Tuple[str, int, int, str]]:
    """
    Load the latest embedding metadata.
    """
    with get_connection() as conn:
        cursor = conn.execute(
            """
        SELECT embedding_file, num_chunks, embedding_dim, model_name
        FROM embeddings_metadata
        ORDER BY created_at DESC
        LIMIT 1
        """
        )
        row = cursor.fetchone()
        return row
