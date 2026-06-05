"""Pluggable vector stores (numpy default, optional faiss)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import VectorHit, VectorStore
from .numpy_store import NumpyVectorStore

if TYPE_CHECKING:
    from ..config import Settings


def build_vector_store(dim: int, namespace: str, settings: Settings | None = None) -> VectorStore:
    """Construct a vector store for a named collection (e.g. ``"episodic"``, ``"docs"``).

    Uses on-disk persistence under ``<data_dir>/vectors/<namespace>`` when
    persistence is enabled, and the backend selected by ``ARIA_VECTOR_BACKEND``
    (falling back to numpy if faiss is unavailable).
    """
    from ..config import get_settings
    from ..logging import get_logger

    settings = settings or get_settings()
    path = settings.vectorstore_dir / namespace if settings.persist else None

    if settings.vector_backend == "faiss":
        try:
            from .faiss_store import FaissVectorStore

            return FaissVectorStore(dim=dim, path=path)
        except Exception as exc:  # pragma: no cover - depends on optional dep
            get_logger(__name__).warning("faiss_unavailable_fallback_numpy", error=str(exc))
    return NumpyVectorStore(dim=dim, path=path)


__all__ = ["NumpyVectorStore", "VectorHit", "VectorStore", "build_vector_store"]
