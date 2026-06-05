"""Tolerant JSON extraction from noisy LLM output.

7B-class models often wrap JSON in prose or code fences. :func:`extract_first_json`
scans for the first balanced ``{...}`` / ``[...]`` block (correctly skipping braces
inside strings) and parses it, returning ``None`` on failure. This is the shared
foundation for the ReAct action parser, fact extraction, and importance scoring,
all of which must degrade gracefully rather than crash on malformed model output.
"""

from __future__ import annotations

import json
from typing import Any

_OPENERS = "{["
_PAIRS = {"}": "{", "]": "["}


def extract_first_json(text: str) -> Any | None:
    """Return the first balanced JSON value found in ``text``, or ``None``."""
    if not text:
        return None
    starts = [text.find(c) for c in _OPENERS if text.find(c) != -1]
    if not starts:
        return None
    start = min(starts)

    stack: list[str] = []
    in_str = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in _OPENERS:
            stack.append(ch)
        elif ch in _PAIRS:
            if not stack or stack[-1] != _PAIRS[ch]:
                return None
            stack.pop()
            if not stack:
                try:
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def extract_json_object(text: str) -> dict | None:
    """Like :func:`extract_first_json` but only returns ``dict`` results."""
    value = extract_first_json(text)
    return value if isinstance(value, dict) else None


__all__ = ["extract_first_json", "extract_json_object"]
