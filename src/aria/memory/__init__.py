"""Aria's layered, cognitively-inspired memory system.

Tiers: :class:`WorkingMemory` (recent window), :class:`EpisodicMemory` (vector-indexed
events with Generative-Agents retrieval), :class:`SemanticMemory` (structured profile
facts), :class:`Summarizer` (consolidation) and :class:`Reflection` (insight synthesis),
all coordinated by :class:`MemoryManager`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .consolidation import Summarizer
from .episodic import EpisodicMemory
from .importance import ImportanceScorer
from .manager import AssembledContext, MemoryManager, WriteResult
from .reflection import Reflection
from .schema import MemoryRecord
from .semantic import SemanticMemory
from .store_sqlite import SqliteMemoryStore
from .working import WorkingMemory

if TYPE_CHECKING:
    from ..config import Settings


def build_memory_manager(
    settings: Settings | None = None,
    model: Any | None = None,
    embedder: Any | None = None,
) -> MemoryManager:
    """Wire up a fully-configured :class:`MemoryManager` from settings.

    Args:
        settings: Configuration (defaults to :func:`aria.config.get_settings`).
        model: Optional chat model enabling LLM-backed importance/extraction/
            summary/reflection. With ``None`` everything uses deterministic fallbacks.
        embedder: Optional embedder override (defaults to the configured one).
    """
    from ..config import get_settings
    from ..embeddings import build_embedder
    from ..rag.index import DocumentIndex
    from ..vectorstore import build_vector_store

    settings = settings or get_settings()
    settings.ensure_dirs()
    embedder = embedder or build_embedder(settings)

    store = build_vector_store(dim=embedder.dim, namespace="episodic", settings=settings)
    episodic = EpisodicMemory(embedder=embedder, store=store, settings=settings)
    sqlite_store = SqliteMemoryStore(settings.sqlite_path if settings.persist else None)
    semantic = SemanticMemory(store=sqlite_store, model=model)

    doc_store = build_vector_store(dim=embedder.dim, namespace="docs", settings=settings)
    document_index = DocumentIndex(embedder=embedder, store=doc_store, settings=settings)

    manager = MemoryManager(
        embedder=embedder,
        episodic=episodic,
        semantic=semantic,
        working=WorkingMemory(settings.working_window_messages, settings.working_window_tokens),
        summarizer=Summarizer(settings, model),
        reflector=Reflection(settings, model),
        importance=ImportanceScorer(model),
        settings=settings,
        persona=settings.persona,
        document_index=document_index,
    )
    manager.use_llm_memory = model is not None
    return manager


__all__ = [
    "AssembledContext",
    "EpisodicMemory",
    "ImportanceScorer",
    "MemoryManager",
    "MemoryRecord",
    "Reflection",
    "SemanticMemory",
    "SqliteMemoryStore",
    "Summarizer",
    "WorkingMemory",
    "WriteResult",
    "build_memory_manager",
]
