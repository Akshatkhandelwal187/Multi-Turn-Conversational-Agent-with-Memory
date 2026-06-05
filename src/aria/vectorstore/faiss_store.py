"""Optional FAISS-backed vector store (approximate/exact NN at scale).

Reuses :class:`NumpyVectorStore` for storage, metadata, and persistence, and only
swaps the *search* path to a FAISS ``IndexFlatIP`` (inner product over normalised
vectors == cosine). The index is rebuilt lazily after writes. Enable with
``ARIA_VECTOR_BACKEND=faiss`` and ``pip install -e '.[faiss]'``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..exceptions import MemoryStoreError
from .base import VectorHit
from .numpy_store import NumpyVectorStore, _normalize


class FaissVectorStore(NumpyVectorStore):
    """A vector store whose nearest-neighbour search uses FAISS."""

    def __init__(self, dim: int, path: str | Path | None = None) -> None:
        try:
            import faiss  # noqa: F401
        except ImportError as exc:  # pragma: no cover - only without the extra
            raise MemoryStoreError(
                "faiss is not installed. Install the faiss extra (pip install -e "
                "'.[faiss]') or set ARIA_VECTOR_BACKEND=numpy."
            ) from exc
        self._index: Any = None
        self._dirty = True
        super().__init__(dim=dim, path=path)

    def add(self, id: str, embedding: np.ndarray, metadata: dict | None = None) -> None:
        super().add(id, embedding, metadata)
        self._dirty = True

    def delete(self, ids: list[str]) -> None:
        super().delete(ids)
        self._dirty = True

    def load(self) -> None:
        super().load()
        self._dirty = True

    def _ensure_index(self) -> None:
        if not self._dirty and self._index is not None:
            return
        import faiss

        index = faiss.IndexFlatIP(self.dim)
        if len(self._ids) > 0:
            index.add(np.ascontiguousarray(self._matrix, dtype=np.float32))
        self._index = index
        self._dirty = False

    def search(self, query: np.ndarray, k: int = 5, filter: dict | None = None) -> list[VectorHit]:
        if len(self._ids) == 0 or k <= 0:
            return []
        self._ensure_index()
        q = np.ascontiguousarray(_normalize(query)[None, :], dtype=np.float32)
        # Over-fetch when filtering so post-filtering can still return k hits.
        fetch = len(self._ids) if filter else min(len(self._ids), k)
        scores, idxs = self._index.search(q, fetch)
        hits: list[VectorHit] = []
        for score, row in zip(scores[0], idxs[0], strict=False):
            if row < 0:
                continue
            mid = self._ids[int(row)]
            meta = self._meta[mid]
            if filter and not all(meta.get(fk) == fv for fk, fv in filter.items()):
                continue
            hits.append(VectorHit(id=mid, score=float(score), metadata=dict(meta)))
            if len(hits) >= k:
                break
        return hits


__all__ = ["FaissVectorStore"]
