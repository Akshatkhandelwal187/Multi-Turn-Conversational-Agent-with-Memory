"""Checkpointer construction.

Returns a durable SQLite checkpointer for the app (so conversations survive restarts
and can be resumed by ``thread_id``) and an in-memory one for tests. Critically, the
SQLite saver is built from a *directly-constructed* connection with
``check_same_thread=False`` — **not** ``SqliteSaver.from_conn_string``, which is a
context manager that closes the connection on exit and would silently drop memory
between Streamlit reruns. The connection's internal lock makes single-process sharing
safe.
"""

from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING, Any

from langgraph.checkpoint.memory import InMemorySaver

if TYPE_CHECKING:
    from ..config import Settings


def make_checkpointer(settings: Settings) -> Any:
    """Build a checkpointer: durable SQLite when persisting, else in-memory."""
    if not settings.persist:
        return InMemorySaver()

    from langgraph.checkpoint.sqlite import SqliteSaver

    settings.ensure_dirs()
    conn = sqlite3.connect(str(settings.checkpoint_path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return saver


__all__ = ["make_checkpointer"]
