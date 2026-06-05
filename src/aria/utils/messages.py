"""Helpers for rendering LangChain messages to plain text."""

from __future__ import annotations

from typing import Any

_ROLE_BY_TYPE = {"human": "User", "ai": "Aria", "system": "System", "tool": "Tool"}


def message_text(msg: Any) -> str:
    content = getattr(msg, "content", msg)
    if isinstance(content, list):
        return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def message_role(msg: Any) -> str:
    return _ROLE_BY_TYPE.get(getattr(msg, "type", ""), "User")


def render_transcript(messages: list[Any]) -> str:
    """Render messages as a ``Role: text`` transcript (skips empty content)."""
    lines = []
    for msg in messages:
        text = message_text(msg).strip()
        if text:
            lines.append(f"{message_role(msg)}: {text}")
    return "\n".join(lines)


__all__ = ["message_role", "message_text", "render_transcript"]
