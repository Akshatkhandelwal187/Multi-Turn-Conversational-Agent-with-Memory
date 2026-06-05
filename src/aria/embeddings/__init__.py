"""Pluggable text embedders."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import Embedder
from .hashing import HashingEmbedder

if TYPE_CHECKING:
    from ..config import Settings


def build_embedder(settings: Settings | None = None) -> Embedder:
    """Construct the embedder selected by configuration.

    Defaults to the offline :class:`HashingEmbedder`; returns a
    :class:`SentenceTransformerEmbedder` when ``ARIA_EMBEDDER=sentence_transformer``.
    """
    from ..config import get_settings

    settings = settings or get_settings()
    if settings.embedder == "sentence_transformer":
        from .sentence_transformer import SentenceTransformerEmbedder

        return SentenceTransformerEmbedder(settings.st_model_name)
    return HashingEmbedder(dim=settings.hashing_dim)


__all__ = ["Embedder", "HashingEmbedder", "build_embedder"]
