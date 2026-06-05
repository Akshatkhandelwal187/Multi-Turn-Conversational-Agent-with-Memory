"""Semantic memory: a structured, durable profile of facts about the user.

Distinct from episodic memory (raw events), semantic memory holds *consolidated*
knowledge — the user's name, preferences, projects, goals — as ``key → value`` pairs
in SQLite. Facts are extracted from each user message either by the LLM (a strict
JSON object) or, as a deterministic fallback, by a small set of regex patterns, so
the profile is populated even offline / when the model returns prose.
"""

from __future__ import annotations

import re
from typing import Any

from ..utils.jsonparse import extract_json_object
from .store_sqlite import SqliteMemoryStore

_EXTRACT_PROMPT = (
    "Extract durable, stable facts about the USER from their message as a flat JSON "
    "object of snake_case_key to value (strings). Only include lasting personal facts "
    "(name, preferences, projects, profession, goals). Omit questions and transient "
    "context. If there are none, return {{}}.\n\nMessage: {text}\n\nJSON:"
)

# Deterministic fallback patterns: (regex, key-template using group names/indexes).
_PATTERNS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\bmy name is ([A-Z][a-zA-Z'-]+)", re.I), "name"),
    (re.compile(r"\bi'?m called ([A-Z][a-zA-Z'-]+)", re.I), "name"),
    (re.compile(r"\bmy favou?rite (\w+) is ([\w\s+#.\-]+?)(?:[.!?,]|$)", re.I), "favorite_{0}"),
    (
        re.compile(r"\bi(?:'m| am)? (?:building|working on|developing) (?:a |an )?"
                   r"([\w\s+#.\-]+?)(?:[.!?,]|$)", re.I),
        "project",
    ),
    (re.compile(r"\bi(?:'m| am) (?:a|an) ([\w\s\-]+?)(?:[.!?,]|$)", re.I), "role"),
    (re.compile(r"\bi (?:like|love|prefer|enjoy) ([\w\s+#.\-]+?)(?:[.!?,]|$)", re.I), "likes"),
)


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" .!?,")


def heuristic_extract(text: str) -> dict[str, str]:
    facts: dict[str, str] = {}
    for pattern, key_tmpl in _PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        groups = match.groups()
        if "{0}" in key_tmpl:
            key = key_tmpl.format(_clean(groups[0]).lower().replace(" ", "_"))
            value = _clean(groups[1])
        else:
            key = key_tmpl
            value = _clean(groups[-1])
        if value:
            facts[key] = value
    return facts


class SemanticMemory:
    """User-profile facts with LLM + heuristic extraction."""

    def __init__(self, store: SqliteMemoryStore, model: Any | None = None) -> None:
        self.store = store
        self.model = model

    def extract_and_update(self, user_text: str, use_llm: bool = False) -> dict[str, str]:
        facts = self._llm_extract(user_text) if (use_llm and self.model) else {}
        if not facts:
            facts = heuristic_extract(user_text)
        for key, value in facts.items():
            self.store.upsert_fact(key, value, source="user")
        return facts

    def _llm_extract(self, text: str) -> dict[str, str]:
        if self.model is None:
            return {}
        try:
            reply = self.model.invoke(_EXTRACT_PROMPT.format(text=text))
            data = extract_json_object(getattr(reply, "content", str(reply)))
        except Exception:  # pragma: no cover - defensive
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            str(k).strip().lower().replace(" ", "_"): _clean(str(v))
            for k, v in data.items()
            if v not in (None, "", [])
        }

    def facts(self) -> dict[str, str]:
        return self.store.all_facts()

    def profile_text(self) -> str:
        facts = self.store.all_facts()
        if not facts:
            return ""
        lines = [f"- {key.replace('_', ' ')}: {value}" for key, value in facts.items()]
        return "Known facts about the user:\n" + "\n".join(lines)


__all__ = ["SemanticMemory", "heuristic_extract"]
