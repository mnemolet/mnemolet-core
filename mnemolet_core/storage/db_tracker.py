import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional

from mnemolet_core.config import DB_PATH

logger = logging.getLogger(__name__)

CREATE_TABLE_FILES = """
CREATE TABLE IF NOT EXISTS files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT UNIQUE,
    hash TEXT,
    ingested_at TEXT,
    indexed INTEGER DEFAULT 0
);
"""


class DBTracker:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        self._create_tables()

    def _get_connection(self) -> sqlite3.Connection:
        """
        Returns a SQLite connection.
        """
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _create_tables(self):
        """
        Init SQLite db if it does not exist (only once).
        """
        with self._get_connection() as conn:
            logger.info("[DBTracker] Create Table Files")
            conn.execute(CREATE_TABLE_FILES)

    def add_file(self, path: str, file_hash: str):
        """
        Insert a new file if if does not exist.
        """
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO files (path, hash, ingested_at)
                VALUES (?, ?, ?)
            """,
                (path, file_hash, datetime.now(UTC).isoformat()),
            )

    def file_exists(self, file_hash: str) -> bool:
        """
        Check if file with this hash is already in db.
        """
        with self._get_connection() as conn:
            curr = conn.execute("SELECT 1 FROM files WHERE hash = ?", (file_hash,))
            return curr.fetchone() is not None

    def mark_indexed(self, file_hash: str):
        """
        Mark file as indexed is Qdrant.
        """
        with self._get_connection() as conn:
            conn.execute("UPDATE files SET indexed = 1 WHERE hash = ?", (file_hash,))

    def list_files(self, indexed: Optional[bool] = None) -> list[dict]:
        """
        List all tracked files, can be optionally filtered by indexed status.
        """
        query = "SELECT path, hash, ingested_at, indexed FROM files"
        params = ()
        if indexed is not None:
            query += " WHERE indexed = ?"
            params = (1 if indexed else 0,)
        with self._get_connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(row) for row in rows]
