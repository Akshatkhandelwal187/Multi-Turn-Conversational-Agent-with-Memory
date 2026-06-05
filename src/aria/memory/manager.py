"""The MemoryManager — the single seam the agent (and UI) talk to.

It composes the four memory tiers and exposes two primary operations used by the
graph: :meth:`assemble` (read path — build the context block from persona + profile +
retrieved episodic memories + rolling summary + working window) and :meth:`write`
(write path — persist the exchange, extract facts, score importance). It also drives
the periodic :meth:`maybe_reflect` and :meth:`maybe_summarize` triggers and backs the
``search_memory`` tool via :meth:`search`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..embeddings.base import Embedder
from .consolidation import Summarizer
from .episodic import EpisodicMemory
from .importance import ImportanceScorer
from .reflection import Reflection
from .schema import MemoryRecord
from .semantic import SemanticMemory
from .working import WorkingMemory

if TYPE_CHECKING:
    from ..config import Settings


@dataclass
class AssembledContext:
    """The result of the read path — ready to send to the model."""

    system_text: str
    working_messages: list[Any]
    retrieved: list[MemoryRecord] = field(default_factory=list)
    profile: dict[str, str] = field(default_factory=dict)


@dataclass
class WriteResult:
    """The result of the write path."""

    importance: float
    facts: dict[str, str] = field(default_factory=dict)
    memory_id: str | None = None


class MemoryManager:
    """Orchestrates working, episodic, semantic, consolidation and reflection memory."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        episodic: EpisodicMemory,
        semantic: SemanticMemory,
        working: WorkingMemory,
        summarizer: Summarizer,
        reflector: Reflection,
        importance: ImportanceScorer,
        settings: Settings,
        persona: str,
        document_index: Any = None,
    ) -> None:
        self.embedder = embedder
        self.episodic = episodic
        self.semantic = semantic
        self.working = working
        self.summarizer = summarizer
        self.reflector = reflector
        self.importance = importance
        self.settings = settings
        self.persona = persona
        self.document_index = document_index  # RAG over uploaded docs (optional)
        self.use_llm_memory = False  # graph may flip this on when a real model is present

    # -- read path -----------------------------------------------------------
    def assemble(self, messages: list[Any], query: str, summary: str = "") -> AssembledContext:
        profile_block = self.semantic.profile_text()
        retrieved = self.episodic.retrieve(query) if query else []
        working_messages = self.working.select(messages)
        system_text = self._build_system_text(profile_block, retrieved, summary)
        return AssembledContext(
            system_text=system_text,
            working_messages=working_messages,
            retrieved=retrieved,
            profile=self.semantic.facts(),
        )

    def _build_system_text(
        self, profile_block: str, retrieved: list[MemoryRecord], summary: str
    ) -> str:
        parts = [self.persona]
        if profile_block:
            parts.append(profile_block)
        if summary:
            parts.append("Summary of earlier conversation:\n" + summary)
        if retrieved:
            lines = "\n".join(f"- {r.text}" for r in retrieved)
            parts.append(
                "Relevant things you remember (may help answer the user):\n" + lines
            )
        return "\n\n".join(parts)

    # -- write path ----------------------------------------------------------
    def write(self, user_text: str, ai_text: str = "") -> WriteResult:
        importance = self.importance.score(user_text, use_llm=self.use_llm_memory)
        record = self.episodic.add(user_text, importance=importance, extra={"role": "user"})
        if ai_text:
            ai_importance = min(0.5, self.importance.score(ai_text))
            self.episodic.add(ai_text, importance=ai_importance, extra={"role": "assistant"})
        facts = self.semantic.extract_and_update(user_text, use_llm=self.use_llm_memory)
        return WriteResult(importance=importance, facts=facts, memory_id=record.id)

    # -- triggers ------------------------------------------------------------
    def maybe_reflect(self, turn_count: int) -> list[str]:
        if not self.reflector.should_reflect(turn_count):
            return []
        recents = self.episodic.recent(self.settings.reflection_top_memories)
        insights = self.reflector.reflect(recents)
        for insight in insights:
            record = self.episodic.add(insight, importance=0.85, kind="reflection")
            self.semantic.store.add_reflection(record.id, insight)
        return insights

    def maybe_summarize(
        self, messages: list[Any], summary: str
    ) -> tuple[str, list[Any], bool]:
        if not self.summarizer.needs_summary(messages, summary):
            return summary, messages, False
        new_summary, kept = self.summarizer.summarize(messages, summary)
        did = len(kept) < len(messages)
        return new_summary, kept, did

    # -- tool / UI access ----------------------------------------------------
    def search(self, query: str, k: int = 5) -> list[MemoryRecord]:
        return self.episodic.search(query, k=k)

    def reflections(self, limit: int = 20) -> list[str]:
        return self.semantic.store.recent_reflections(limit)

    def persist(self) -> None:
        self.episodic.store.persist()
        if self.document_index is not None:
            self.document_index.persist()


__all__ = ["AssembledContext", "MemoryManager", "WriteResult"]
