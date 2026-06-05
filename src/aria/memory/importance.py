"""Memory importance scoring.

Generative Agents rate each memory's *poignancy* (1-10) with the LLM. We support
that (``score(..., use_llm=True)``) but default to a fast, deterministic heuristic so
that ordinary memory writes cost nothing and tests/CI are reproducible. The heuristic
rewards durable, personal, declarative statements ("my name is…", "I'm building…")
over small talk and questions. Scores are normalised to ``[0, 1]``.
"""

from __future__ import annotations

import re
from typing import Any

from ..utils.jsonparse import extract_first_json

_SIGNALS: tuple[re.Pattern, ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bmy name is\b",
        r"\bi am\b|\bi'm\b",
        r"\bi (?:like|love|prefer|hate|enjoy|want|need)\b",
        r"\bfavou?rite\b",
        r"\bi (?:work|working|build|building|study|studying|live|living)\b",
        r"\bmy (?:goal|job|project|deadline|birthday|address|email|phone)\b",
        r"\bremember\b",
        r"\bi have\b",
        r"\d",  # contains a number (dates, quantities)
    )
)
_TRIVIAL = re.compile(r"^\s*(ok(ay)?|thanks?|thank you|cool|nice|sure|yes|no|hi|hello)\b", re.I)

_LLM_PROMPT = (
    "On a scale of 1 to 10, rate how important the following statement is to remember "
    "for understanding the user long-term (1 = trivial small talk, 10 = a core, "
    'durable fact about them). Reply with JSON: {{"importance": <int>}}.\n\n'
    "Statement: {text}"
)


def heuristic_importance(text: str) -> float:
    """A deterministic importance estimate in ``[0.05, 0.95]``."""
    text = text.strip()
    if not text:
        return 0.05
    if _TRIVIAL.match(text) and len(text) < 25:
        return 0.1
    hits = sum(1 for pat in _SIGNALS if pat.search(text))
    score = 0.3 + 0.13 * hits
    if text.endswith("?"):  # questions are usually less durable than statements
        score -= 0.1
    if len(text) > 140:
        score += 0.05
    return max(0.05, min(0.95, score))


class ImportanceScorer:
    """Scores memory importance heuristically or with the LLM."""

    def __init__(self, model: Any | None = None) -> None:
        self.model = model

    def score(self, text: str, use_llm: bool = False) -> float:
        if use_llm and self.model is not None:
            value = self._llm_score(text)
            if value is not None:
                return value
        return heuristic_importance(text)

    def _llm_score(self, text: str) -> float | None:
        if self.model is None:
            return None
        try:
            reply = self.model.invoke(_LLM_PROMPT.format(text=text))
            content = getattr(reply, "content", str(reply))
            data = extract_first_json(content)
            if isinstance(data, dict) and "importance" in data:
                return max(0.05, min(1.0, float(data["importance"]) / 10.0))
            match = re.search(r"\b(10|[1-9])\b", str(content))
            if match:
                return max(0.05, min(1.0, int(match.group(1)) / 10.0))
        except Exception:  # pragma: no cover - defensive against model failures
            return None
        return None


__all__ = ["ImportanceScorer", "heuristic_importance"]
