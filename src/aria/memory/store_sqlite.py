"""SQLite-backed durable store for semantic facts and reflections.

Holds the structured, long-lived state that does not belong in the vector index:
the user-profile **facts** (key → value) and a log of generated **reflections** (for
the UI). Uses ``check_same_thread=False`` so a single connection can be shared across
Streamlit reruns; ``:memory:`` is used for non-persistent (test) configurations.
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

from ..exceptions import MemoryStoreError

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    source     TEXT,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS reflections (
    id         TEXT PRIMARY KEY,
    text       TEXT NOT NULL,
    created_at REAL NOT NULL
);
"""


class SqliteMemoryStore:
    """Durable key/value fact store plus a reflections log."""

    def __init__(self, path: str | Path | None = None) -> None:
        target = ":memory:" if path is None else str(path)
        try:
            self._conn = sqlite3.connect(target, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(_SCHEMA)
            self._conn.commit()
        except sqlite3.Error as exc:  # pragma: no cover - defensive
            raise MemoryStoreError(f"failed to open sqlite store at {target}: {exc}") from exc

    # -- facts ---------------------------------------------------------------
    def upsert_fact(self, key: str, value: str, source: str = "") -> None:
        self._conn.execute(
            "INSERT INTO facts(key, value, source, updated_at) VALUES(?,?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
            "source=excluded.source, updated_at=excluded.updated_at",
            (key, value, source, time.time()),
        )
        self._conn.commit()

    def get_fact(self, key: str) -> str | None:
        row = self._conn.execute("SELECT value FROM facts WHERE key=?", (key,)).fetchone()
        return row["value"] if row else None

    def all_facts(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT key, value FROM facts ORDER BY updated_at").fetchall()
        return {r["key"]: r["value"] for r in rows}

    def delete_fact(self, key: str) -> None:
        self._conn.execute("DELETE FROM facts WHERE key=?", (key,))
        self._conn.commit()

    def clear_facts(self) -> None:
        self._conn.execute("DELETE FROM facts")
        self._conn.commit()

    # -- reflections ---------------------------------------------------------
    def add_reflection(self, id: str, text: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO reflections(id, text, created_at) VALUES(?,?,?)",
            (id, text, time.time()),
        )
        self._conn.commit()

    def recent_reflections(self, limit: int = 20) -> list[str]:
        rows = self._conn.execute(
            "SELECT text FROM reflections ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        return [r["text"] for r in rows]

    def close(self) -> None:
        self._conn.close()


__all__ = ["SqliteMemoryStore"]
