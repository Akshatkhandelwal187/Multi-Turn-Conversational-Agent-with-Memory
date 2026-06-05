"""Token counting and per-turn usage tracking.

Uses :mod:`tiktoken` when available for an accurate count, and falls back to a
character-based heuristic otherwise — so token accounting never requires network or
a specific model. :class:`UsageTracker` aggregates per-turn usage for the live
metrics panel and the evaluation harness.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Any

_AVG_CHARS_PER_TOKEN = 4.0


@lru_cache(maxsize=1)
def _encoder() -> Any | None:
    try:
        import tiktoken

        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - tiktoken missing or offline
        return None


def count_tokens(text: str) -> int:
    """Count tokens in ``text`` (tiktoken if available, else a heuristic)."""
    if not text:
        return 0
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, round(len(text) / _AVG_CHARS_PER_TOKEN))


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # multimodal content blocks
        parts = [p.get("text", "") if isinstance(p, dict) else str(p) for p in content]
        return " ".join(parts)
    return str(content)


def count_message_tokens(messages: Iterable[Any]) -> int:
    """Approximate the prompt token count of a list of chat messages."""
    total = 0
    for msg in messages:
        content = getattr(msg, "content", msg)
        total += count_tokens(_content_to_text(content)) + 4  # per-message overhead
    return total


@dataclass
class TurnUsage:
    """Resource usage for a single conversation turn."""

    tokens_in: int = 0
    tokens_out: int = 0
    tool_calls: int = 0
    retrieved_memories: int = 0
    latency_ms: float = 0.0
    reflected: bool = False
    summarized: bool = False

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class UsageTracker:
    """Accumulates :class:`TurnUsage` records across a session."""

    turns: list[TurnUsage] = field(default_factory=list)

    def record(self, usage: TurnUsage) -> None:
        self.turns.append(usage)

    def last(self) -> TurnUsage | None:
        return self.turns[-1] if self.turns else None

    def totals(self) -> dict:
        return {
            "turns": len(self.turns),
            "tokens_in": sum(t.tokens_in for t in self.turns),
            "tokens_out": sum(t.tokens_out for t in self.turns),
            "total_tokens": sum(t.total_tokens for t in self.turns),
            "tool_calls": sum(t.tool_calls for t in self.turns),
            "avg_latency_ms": (
                sum(t.latency_ms for t in self.turns) / len(self.turns) if self.turns else 0.0
            ),
        }


__all__ = ["TurnUsage", "UsageTracker", "count_message_tokens", "count_tokens"]
