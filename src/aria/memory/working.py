"""Working (short-term) memory: the recent verbatim window.

Selects the tail of the conversation that fits within both a message-count and a
token budget, always keeping at least the most recent message. This is the
fast-access store that, together with the rolling summary, bounds how much raw
history is replayed to the model each turn.
"""

from __future__ import annotations

from typing import Any

from ..observability.tokens import count_message_tokens


class WorkingMemory:
    """A token- and count-bounded sliding window over recent messages."""

    def __init__(self, max_messages: int = 12, max_tokens: int = 1500) -> None:
        self.max_messages = max_messages
        self.max_tokens = max_tokens

    def select(self, messages: list[Any]) -> list[Any]:
        """Return the most recent messages that fit the window budget (in order)."""
        selected: list[Any] = []
        tokens = 0
        for msg in reversed(messages):
            msg_tokens = count_message_tokens([msg])
            over_count = len(selected) >= self.max_messages
            over_tokens = tokens + msg_tokens > self.max_tokens
            if selected and (over_count or over_tokens):
                break
            selected.append(msg)
            tokens += msg_tokens
        selected.reverse()
        return selected


__all__ = ["WorkingMemory"]
