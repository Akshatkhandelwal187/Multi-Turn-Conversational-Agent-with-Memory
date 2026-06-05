"""Memory consolidation: MemGPT-style rolling summarisation.

When the live transcript plus the running summary exceed a token budget, the oldest
turns are folded into the summary (which preserves durable facts and decisions) and
dropped from the working window. This bounds the context sent to the model on every
turn regardless of how long the conversation grows — the key to "infinite" chats on a
fixed context window. Falls back to a deterministic extractive summary with no model.
"""

from __future__ import annotations

from typing import Any

from ..observability.tokens import count_message_tokens, count_tokens
from ..utils.messages import message_text, render_transcript

_SUMMARY_PROMPT = (
    "You maintain a running summary of a conversation. Update the summary so it "
    "preserves every durable fact, preference, decision, and open thread about the "
    "user. Keep it concise (a few sentences).\n\n"
    "Previous summary:\n{previous}\n\n"
    "New conversation to fold in:\n{transcript}\n\nUpdated summary:"
)


class Summarizer:
    """Decides when to summarise and produces the new running summary."""

    def __init__(self, settings: Any, model: Any | None = None) -> None:
        self.settings = settings
        self.model = model

    def needs_summary(self, messages: list[Any], summary: str) -> bool:
        budget = self.settings.summary_token_budget
        return count_message_tokens(messages) + count_tokens(summary) > budget

    def summarize(self, messages: list[Any], previous: str) -> tuple[str, list[Any]]:
        """Return ``(new_summary, kept_messages)``.

        The most recent ``summary_keep_last_messages`` messages stay in the window;
        everything older is folded into the summary.
        """
        keep = self.settings.summary_keep_last_messages
        if len(messages) <= keep:
            return previous, messages
        to_fold = messages[:-keep]
        kept = messages[-keep:]
        transcript = render_transcript(to_fold)
        if not transcript:
            return previous, kept
        new_summary = self._summarise_text(previous, transcript)
        return new_summary, kept

    def _summarise_text(self, previous: str, transcript: str) -> str:
        if self.model is not None:
            try:
                reply = self.model.invoke(
                    _SUMMARY_PROMPT.format(previous=previous or "(none)", transcript=transcript)
                )
                text = message_text(reply).strip()
                if text:
                    return text
            except Exception:  # pragma: no cover - fall back on model failure
                pass
        return self._extractive_fallback(previous, transcript)

    @staticmethod
    def _extractive_fallback(previous: str, transcript: str) -> str:
        """Deterministic fallback: keep the previous summary + the user's statements."""
        user_lines = [
            line.split(": ", 1)[1] for line in transcript.splitlines() if line.startswith("User: ")
        ]
        merged = [s for s in [previous.strip(), *user_lines] if s]
        summary = " ".join(merged)
        # Keep the fallback bounded.
        return summary[:1000]


__all__ = ["Summarizer"]
