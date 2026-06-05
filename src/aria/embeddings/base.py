"""The embedder interface.

An :class:`Embedder` maps text to dense vectors. Two implementations exist: a
deterministic, dependency-light :class:`~aria.embeddings.hashing.HashingEmbedder`
(the default — used everywhere in tests/CI so no model download or network is ever
required) and a real :class:`~aria.embeddings.sentence_transformer.SentenceTransformerEmbedder`
for runtime semantic quality. Vectors are returned L2-normalised so downstream
cosine similarity reduces to a dot product.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    """Protocol for text embedders."""

    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch of texts into an ``(len(texts), dim)`` float32 array."""
        ...

    def embed_one(self, text: str) -> np.ndarray:
        """Embed a single text into a ``(dim,)`` float32 vector."""
        ...
