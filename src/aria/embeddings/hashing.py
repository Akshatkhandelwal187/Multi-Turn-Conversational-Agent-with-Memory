"""Deterministic hashing embedder (the "feature hashing" / hashing-trick).

Maps text to a fixed-dimensional vector by hashing word tokens into buckets with a
signed accumulator, then L2-normalising. It is:

* **deterministic** across processes and platforms (uses :mod:`hashlib`, not the
  salted built-in ``hash``), so tests and the evaluation harness are reproducible;
* **offline & dependency-light** (no torch / no model download); and
* **semantically meaningful enough** for retrieval tests — texts sharing tokens get
  higher cosine similarity than unrelated texts.

It is the default embedder so the whole project installs and runs with no ML stack.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _bucket_and_sign(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.md5(token.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big")
    bucket = value % dim
    sign = 1.0 if (value >> 8) & 1 else -1.0
    return bucket, sign


class HashingEmbedder:
    """A deterministic, offline embedder via the signed hashing trick."""

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in _TOKEN_RE.findall(text.lower()):
            bucket, sign = _bucket_and_sign(token, self.dim)
            vec[bucket] += sign
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec /= norm
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self.embed_one(t) for t in texts]).astype(np.float32)


__all__ = ["HashingEmbedder"]
