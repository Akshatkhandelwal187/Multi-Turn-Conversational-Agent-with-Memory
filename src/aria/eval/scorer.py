"""Scoring for the recall benchmark.

The default scorer is deterministic (normalised substring match + token-F1), so the
headline results are reproducible with no network. An optional LLM-as-judge is provided
for grading free-form answers from a real model, gated behind a Hugging Face token.
"""

from __future__ import annotations

import re
from typing import Any

from ..utils.jsonparse import extract_json_object

_WORD = re.compile(r"[a-z0-9]+")


def normalise(text: str) -> str:
    return " ".join(_WORD.findall(text.lower()))


def contains_needle(text: str, needle: str) -> bool:
    """True when ``needle`` appears (normalised) within ``text``."""
    return normalise(needle) in normalise(text)


def token_f1(answer: str, expected: str) -> float:
    a, e = set(_WORD.findall(answer.lower())), set(_WORD.findall(expected.lower()))
    if not a or not e:
        return 0.0
    overlap = len(a & e)
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(a), overlap / len(e)
    return 2 * precision * recall / (precision + recall)


class LLMJudge:
    """Optional LLM-as-judge: grades whether an answer conveys the expected fact."""

    _PROMPT = (
        "You are grading a memory test. Question: {question}\nExpected fact: {expected}\n"
        "Answer: {answer}\nDoes the answer correctly convey the expected fact? "
        'Reply with JSON: {{"correct": true|false}}.'
    )

    def __init__(self, model: Any) -> None:
        self.model = model

    def score(self, question: str, answer: str, expected: str) -> float:
        reply = self.model.invoke(
            self._PROMPT.format(question=question, expected=expected, answer=answer)
        )
        data = extract_json_object(getattr(reply, "content", str(reply)))
        if isinstance(data, dict) and "correct" in data:
            return 1.0 if bool(data["correct"]) else 0.0
        return 1.0 if contains_needle(answer, expected) else 0.0


__all__ = ["LLMJudge", "contains_needle", "normalise", "token_f1"]
