import sqlite3
from pathlib import Path
from datetime import datetime, UTC
from typing import Optional


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


def get_connection() -> sqlite3.Connection:
    """
    Returns a SQLite connection.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db():
    """
    Init SQLite db if it does not exist (only once).
    """
    with get_connection() as conn:
        conn.execute(CREATE_TABLE_FILES)
        # conn.execute(CREATE_TABLE_EMBEDDINGS_METADATA)


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


def list_files(indexed: Optional[bool] = None) -> list[dict]:
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
