"""Episodic (long-term) memory with Generative-Agents retrieval.

Every salient utterance is embedded and stored as a :class:`MemoryRecord`. Retrieval
follows Park et al. (2023): each candidate is scored by a weighted combination of

* **relevance** — cosine similarity to the query embedding,
* **recency** — exponential decay since the memory was last accessed, and
* **importance** — the memory's stored poignancy,

with each component min-max normalised across candidates before weighting. Retrieved
memories are "touched" (recency + access-count updated) so attended memories stay
salient — a small but faithful piece of the cognitive model.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

from ..embeddings.base import Embedder
from ..vectorstore.base import VectorStore
from .schema import MemoryRecord

if TYPE_CHECKING:
    from ..config import Settings


def _min_max_normalise(values: list[float]) -> list[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi - lo < 1e-9:
        return [0.5 for _ in values]  # neutral when there is no spread
    return [(v - lo) / (hi - lo) for v in values]


class EpisodicMemory:
    """Vector-indexed long-term memory with importance/recency/relevance retrieval."""

    def __init__(self, embedder: Embedder, store: VectorStore, settings: Settings) -> None:
        self.embedder = embedder
        self.store = store
        self.settings = settings

    def add(
        self,
        text: str,
        *,
        importance: float = 0.5,
        kind: str = "episodic",
        timestamp: float | None = None,
        extra: dict | None = None,
    ) -> MemoryRecord:
        record = MemoryRecord(
            text=text,
            kind=kind,
            importance=importance,
            timestamp=timestamp if timestamp is not None else time.time(),
            extra=extra or {},
        )
        record.last_access = record.timestamp
        self.store.add(record.id, self.embedder.embed_one(text), record.to_metadata())
        return record

    def _recency(self, last_access: float, now: float) -> float:
        hours = max(0.0, (now - last_access) / 3600.0)
        half_life = self.settings.recency_half_life_hours
        return math.pow(0.5, hours / half_life)

    def retrieve(
        self, query: str, k: int | None = None, *, now: float | None = None, touch: bool = True
    ) -> list[MemoryRecord]:
        """Return the top-k memories for ``query`` by the combined GA score."""
        k = self.settings.episodic_top_k if k is None else k
        if k <= 0 or len(self.store) == 0:
            return []
        now = time.time() if now is None else now

        # Pull every memory with its cosine relevance, then re-rank.
        hits = self.store.search(self.embedder.embed_one(query), k=len(self.store))
        if not hits:
            return []

        relevance = [h.score for h in hits]
        recency = [self._recency(float(h.metadata.get("last_access", 0.0)), now) for h in hits]
        importance = [float(h.metadata.get("importance", 0.5)) for h in hits]

        rel_n = _min_max_normalise(relevance)
        rec_n = _min_max_normalise(recency)
        imp_n = _min_max_normalise(importance)

        s = self.settings
        scored = [
            (
                s.relevance_weight * rel_n[i]
                + s.recency_weight * rec_n[i]
                + s.importance_weight * imp_n[i],
                hits[i],
            )
            for i in range(len(hits))
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = scored[:k]

        records = [MemoryRecord.from_metadata(h.id, h.metadata) for _, h in top]
        if touch:
            for record in records:
                record.touch(now)
                self.store.update_metadata(
                    record.id, last_access=record.last_access, access_count=record.access_count
                )
        return records

    def search(self, query: str, k: int = 5) -> list[MemoryRecord]:
        """Read-only retrieval (no recency update) — used by the search_memory tool."""
        return self.retrieve(query, k=k, touch=False)

    def recent(self, n: int = 15) -> list[MemoryRecord]:
        """The ``n`` most recently created memories (input for reflection)."""
        records = [
            MemoryRecord.from_metadata(mid, meta)
            for mid in self.store.all_ids()
            if (meta := self.store.get(mid)) is not None
        ]
        records.sort(key=lambda r: r.timestamp, reverse=True)
        return records[:n]

    def __len__(self) -> int:
        return len(self.store)


__all__ = ["EpisodicMemory"]
