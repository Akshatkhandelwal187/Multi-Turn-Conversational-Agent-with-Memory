"""A dependency-light vector store: cosine similarity over a numpy matrix.

This is the default backend. It needs nothing beyond numpy, persists to a small set
of files on disk, and is more than adequate for demo-/research-scale corpora (a few
thousand memories). For larger corpora, swap in the optional faiss backend via
``ARIA_VECTOR_BACKEND=faiss`` — the interface is identical.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..exceptions import MemoryStoreError
from .base import VectorHit


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0.0 else vec


class NumpyVectorStore:
    """In-memory cosine vector store with optional on-disk persistence."""

    def __init__(self, dim: int, path: str | Path | None = None) -> None:
        self.dim = dim
        self.path = Path(path) if path else None
        self._ids: list[str] = []
        self._row_of: dict[str, int] = {}
        self._matrix: np.ndarray = np.zeros((0, dim), dtype=np.float32)
        self._meta: dict[str, dict] = {}
        if self.path is not None and (self.path / "store.json").exists():
            self.load()

    # -- writes ---------------------------------------------------------------
    def add(self, id: str, embedding: np.ndarray, metadata: dict | None = None) -> None:
        vec = _normalize(embedding)
        if vec.shape[0] != self.dim:
            raise MemoryStoreError(f"embedding dim {vec.shape[0]} != store dim {self.dim}")
        meta = dict(metadata or {})
        if id in self._row_of:
            self._matrix[self._row_of[id]] = vec
            self._meta[id] = meta
            return
        self._row_of[id] = len(self._ids)
        self._ids.append(id)
        if self._matrix.shape[0] > 0:
            self._matrix = np.vstack([self._matrix, vec[None, :]])
        else:
            self._matrix = vec[None, :]
        self._meta[id] = meta

    def update_metadata(self, id: str, **changes: object) -> None:
        if id not in self._meta:
            raise MemoryStoreError(f"unknown id: {id}")
        self._meta[id].update(changes)

    def delete(self, ids: list[str]) -> None:
        keep = [i for i in self._ids if i not in set(ids)]
        self._rebuild(keep)

    def _rebuild(self, keep_ids: list[str]) -> None:
        if not keep_ids:
            self._ids, self._row_of = [], {}
            self._matrix = np.zeros((0, self.dim), dtype=np.float32)
            self._meta = {k: v for k, v in self._meta.items() if k in keep_ids}
            return
        rows = [self._row_of[i] for i in keep_ids]
        self._matrix = self._matrix[rows]
        self._ids = list(keep_ids)
        self._row_of = {i: r for r, i in enumerate(self._ids)}
        self._meta = {i: self._meta[i] for i in self._ids}

    # -- reads ----------------------------------------------------------------
    def get(self, id: str) -> dict | None:
        meta = self._meta.get(id)
        return dict(meta) if meta is not None else None

    def get_embedding(self, id: str) -> np.ndarray | None:
        row = self._row_of.get(id)
        return self._matrix[row].copy() if row is not None else None

    def search(self, query: np.ndarray, k: int = 5, filter: dict | None = None) -> list[VectorHit]:
        if len(self._ids) == 0 or k <= 0:
            return []
        q = _normalize(query)
        sims = self._matrix @ q  # rows are pre-normalised → dot == cosine
        order = np.argsort(-sims)
        hits: list[VectorHit] = []
        for row in order:
            mid = self._ids[int(row)]
            meta = self._meta[mid]
            if filter and not all(meta.get(fk) == fv for fk, fv in filter.items()):
                continue
            hits.append(VectorHit(id=mid, score=float(sims[int(row)]), metadata=dict(meta)))
            if len(hits) >= k:
                break
        return hits

    def all_ids(self) -> list[str]:
        return list(self._ids)

    def __len__(self) -> int:
        return len(self._ids)

    # -- persistence ----------------------------------------------------------
    def persist(self) -> None:
        if self.path is None:
            return
        self.path.mkdir(parents=True, exist_ok=True)
        np.save(self.path / "matrix.npy", self._matrix)
        payload = {"dim": self.dim, "ids": self._ids, "meta": self._meta}
        (self.path / "store.json").write_text(json.dumps(payload), encoding="utf-8")

    def load(self) -> None:
        if self.path is None:
            return
        try:
            payload = json.loads((self.path / "store.json").read_text(encoding="utf-8"))
            matrix = np.load(self.path / "matrix.npy")
        except (OSError, ValueError) as exc:
            raise MemoryStoreError(f"failed to load vector store at {self.path}: {exc}") from exc
        self.dim = int(payload["dim"])
        self._ids = list(payload["ids"])
        self._meta = dict(payload["meta"])
        self._row_of = {i: r for r, i in enumerate(self._ids)}
        self._matrix = matrix.astype(np.float32).reshape(len(self._ids), self.dim)


__all__ = ["NumpyVectorStore"]
