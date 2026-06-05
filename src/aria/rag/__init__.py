"""Retrieval-augmented generation over uploaded documents."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .chunker import chunk_text
from .index import DocumentIndex
from .loaders import load_bytes, load_path

if TYPE_CHECKING:
    from ..config import Settings


def build_document_index(
    settings: Settings | None = None, embedder: Any | None = None
) -> DocumentIndex:
    """Build a :class:`DocumentIndex` using the configured embedder + a ``docs`` collection."""
    from ..config import get_settings
    from ..embeddings import build_embedder
    from ..vectorstore import build_vector_store

    settings = settings or get_settings()
    embedder = embedder or build_embedder(settings)
    store = build_vector_store(dim=embedder.dim, namespace="docs", settings=settings)
    return DocumentIndex(embedder=embedder, store=store, settings=settings)


__all__ = ["DocumentIndex", "build_document_index", "chunk_text", "load_bytes", "load_path"]
