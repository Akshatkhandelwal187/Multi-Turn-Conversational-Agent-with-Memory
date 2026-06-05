"""Real semantic embeddings via sentence-transformers (optional, runtime only).

Lazily imports :mod:`sentence_transformers` (which pulls torch) so the dependency is
only required when this embedder is actually used. Enable it with
``ARIA_EMBEDDER=sentence_transformer`` after installing the ``embeddings`` extra::

    pip install -e .[embeddings]
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..constants import DEFAULT_EMBEDDING_MODEL
from ..exceptions import EmbeddingError


class SentenceTransformerEmbedder:
    """Embedder backed by a sentence-transformers model."""

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise EmbeddingError(
                "sentence-transformers is not installed. Install the embeddings extra: "
                "pip install -e '.[embeddings]', or set ARIA_EMBEDDER=hashing."
            ) from exc

        self.model_name = model_name
        self._model: Any = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        try:
            vectors = self._model.encode(
                texts, normalize_embeddings=True, convert_to_numpy=True
            )
        except Exception as exc:  # pragma: no cover - runtime/model errors
            raise EmbeddingError(f"sentence-transformers encode failed: {exc}") from exc
        return np.asarray(vectors, dtype=np.float32)

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


__all__ = ["SentenceTransformerEmbedder"]
