"""The unit of memory: :class:`MemoryRecord`.

A record carries the metadata the Generative-Agents retrieval model needs —
``importance`` (how salient), ``timestamp`` (when created) and ``last_access`` /
``access_count`` (how recently/often retrieved) — alongside the text and its
embedding namespace ``kind`` (``episodic`` | ``reflection`` | ``summary`` | ``fact``).
Records round-trip through the vector store's metadata dict.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


def new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class MemoryRecord:
    """A single memory."""

    text: str
    kind: str = "episodic"
    id: str = field(default_factory=new_id)
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5  # normalised to [0, 1]
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    extra: dict = field(default_factory=dict)

    def touch(self, now: float | None = None) -> None:
        """Mark the memory as just retrieved (updates recency + access count)."""
        self.last_access = time.time() if now is None else now
        self.access_count += 1

    def to_metadata(self) -> dict:
        """Serialise to a flat, JSON-safe dict for the vector store."""
        return {
            "text": self.text,
            "kind": self.kind,
            "timestamp": self.timestamp,
            "importance": self.importance,
            "last_access": self.last_access,
            "access_count": self.access_count,
            **self.extra,
        }

    @classmethod
    def from_metadata(cls, id: str, meta: dict) -> MemoryRecord:
        known = {"text", "kind", "timestamp", "importance", "last_access", "access_count"}
        return cls(
            id=id,
            text=str(meta.get("text", "")),
            kind=str(meta.get("kind", "episodic")),
            timestamp=float(meta.get("timestamp", 0.0)),
            importance=float(meta.get("importance", 0.5)),
            last_access=float(meta.get("last_access", meta.get("timestamp", 0.0))),
            access_count=int(meta.get("access_count", 0)),
            extra={k: v for k, v in meta.items() if k not in known},
        )


__all__ = ["MemoryRecord", "new_id"]
