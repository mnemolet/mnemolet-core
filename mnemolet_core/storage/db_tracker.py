import sqlite3
from pathlib import Path
from datetime import datetime, UTC


DB_PATH = Path.home() / ".mnemolet" / "tracker.db"


def init_db():
    """
    Init SQLite db if it does not exist.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE,
                hash TEXT,
                ingested_at TEXT,
                indexed INTEGER DEFAULT 0
            )
        """)
        conn.commit()


def add_file(path: str, file_hash: str):
    """
    Insert a new file if if does not exist.
    """
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO files (path, hash, ingested_at)
            VALUES (?, ?, ?)
        """,
            (path, file_hash, datetime.now(UTC).isoformat()),
        )
        conn.commit()


def file_exists(file_hash: str) -> bool:
    """
    Check if file with this hash is already in db.
    """
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        curr = conn.execute("SELECT 1 FROM files WHERE hash = ?", (file_hash,))
        return curr.fetchone() is not None


def mark_indexed(file_hash: str):
    """
    Mark file as indexed is Qdrant.
    """
    init_db()
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("UPDATE files SET indexed = 1 WHERE hash = ?", (file_hash,))
        conn.commit()


def list_files(indexed: bool | None = None) -> list[dict]:
    """
    List all tracked files, can be optionally filtered by indexed status.
    """
    init_db()
    query = "SELECT path, hash, ingested_at, indexed FROM files"
    params = ()
    if indexed is not None:
        query += " WHERE indexed = ?"
        params = (1 if indexed else 0,)
    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]
