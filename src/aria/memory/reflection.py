"""Reflection: synthesising higher-level insights from recent memories.

Following Generative Agents, every ``reflection_every_k_turns`` the agent steps back
and asks the model to infer a few abstract insights about the user from their recent
memories (e.g. "The user is an ML practitioner focused on recommender systems"). These
insights are stored as high-importance episodic memories, so they surface readily in
future retrieval — letting the agent reason about the user at a level no single message
contains. Reflection requires the LLM; with no model it is a no-op.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..utils.jsonparse import extract_first_json
from ..utils.messages import message_text

if TYPE_CHECKING:
    from .schema import MemoryRecord

_REFLECT_PROMPT = (
    "Below are recent memories about the user. Infer up to {n} high-level insights "
    "about their interests, goals, expertise, or communication style. Each insight "
    "should be a general statement not stated verbatim in any single memory. "
    'Return a JSON array of short strings.\n\nMemories:\n{memories}\n\nInsights (JSON):'
)


class Reflection:
    """Generates and parses reflective insights."""

    def __init__(self, settings: Any, model: Any | None = None) -> None:
        self.settings = settings
        self.model = model

    def should_reflect(self, turn_count: int) -> bool:
        every_k = self.settings.reflection_every_k_turns
        return every_k > 0 and turn_count > 0 and turn_count % every_k == 0

    def reflect(self, memories: list[MemoryRecord], max_insights: int = 3) -> list[str]:
        if self.model is None or not memories:
            return []
        rendered = "\n".join(f"- {m.text}" for m in memories)
        try:
            reply = self.model.invoke(
                _REFLECT_PROMPT.format(n=max_insights, memories=rendered)
            )
            content = message_text(reply)
        except Exception:  # pragma: no cover - defensive
            return []
        return self._parse(content, max_insights)

    @staticmethod
    def _parse(content: str, max_insights: int) -> list[str]:
        data = extract_first_json(content)
        insights: list[str] = []
        if isinstance(data, list):
            insights = [str(x).strip() for x in data if str(x).strip()]
        else:
            for line in content.splitlines():
                cleaned = line.strip().lstrip("-*0123456789. ").strip()
                if cleaned and len(cleaned) > 5:
                    insights.append(cleaned)
        return insights[:max_insights]


__all__ = ["Reflection"]
