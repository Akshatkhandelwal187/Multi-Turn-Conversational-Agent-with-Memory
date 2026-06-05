"""Named, resumable conversations.

Each conversation maps a human-friendly name to a LangGraph ``thread_id`` (the key the
checkpointer uses). This registry persists that mapping (JSON under the data dir) so the
sidebar can list past conversations and resume any of them — the durable checkpointer
restores the full history and the shared long-term memory carries across all of them.
Pure logic with no Streamlit dependency, so it is unit-tested directly.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path


class ConversationRegistry:
    """Tracks conversations as ``thread_id -> {name, created_at, last_used}``."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self._data: dict[str, dict] = {}
        if self.path is not None and self.path.exists():
            self._load()

    def create(self, name: str | None = None) -> str:
        thread_id = uuid.uuid4().hex
        now = time.time()
        self._data[thread_id] = {
            "name": name or f"Conversation {len(self._data) + 1}",
            "created_at": now,
            "last_used": now,
        }
        self._save()
        return thread_id

    def touch(self, thread_id: str) -> None:
        if thread_id in self._data:
            self._data[thread_id]["last_used"] = time.time()
            self._save()

    def rename(self, thread_id: str, name: str) -> None:
        if thread_id in self._data:
            self._data[thread_id]["name"] = name
            self._save()

    def delete(self, thread_id: str) -> None:
        if thread_id in self._data:
            del self._data[thread_id]
            self._save()

    def name_of(self, thread_id: str) -> str:
        entry = self._data.get(thread_id)
        return entry["name"] if entry else thread_id[:8]

    def list(self) -> list[dict]:
        """Conversations as ``{id, name, created_at, last_used}``, most-recent first."""
        items = [{"id": tid, **meta} for tid, meta in self._data.items()]
        items.sort(key=lambda x: x["last_used"], reverse=True)
        return items

    def __len__(self) -> int:
        return len(self._data)

    def _save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data), encoding="utf-8")

    def _load(self) -> None:
        if self.path is None:
            return
        try:
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self._data = {}


__all__ = ["ConversationRegistry"]
