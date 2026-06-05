"""Vector-store interface and the search-hit type.

The store keeps an id → (vector, metadata) mapping and answers nearest-neighbour
queries by cosine similarity. Metadata is an arbitrary JSON-serialisable dict (the
episodic memory stores ``text``, ``timestamp``, ``importance``, ``last_access`` and
``access_count`` here, and re-ranks raw cosine hits by its own scoring function).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass
class VectorHit:
    """A single search result."""

    id: str
    score: float  # cosine similarity in [-1, 1]
    metadata: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        return str(self.metadata.get("text", ""))


@runtime_checkable
class VectorStore(Protocol):
    """Protocol implemented by the numpy and faiss backends."""

    dim: int

    def add(self, id: str, embedding: np.ndarray, metadata: dict | None = None) -> None: ...

    def search(
        self, query: np.ndarray, k: int = 5, filter: dict | None = None
    ) -> list[VectorHit]: ...

    def get(self, id: str) -> dict | None: ...

    def update_metadata(self, id: str, **changes: object) -> None: ...

    def delete(self, ids: list[str]) -> None: ...

    def persist(self) -> None: ...

    def __len__(self) -> int: ...
