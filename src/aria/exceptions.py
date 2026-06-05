"""Typed exception hierarchy for Aria.

A small, explicit hierarchy lets the UI and tests distinguish *what* failed
(model call, embedding, persistence, a tool) from generic runtime errors, and lets
us attach retry/degradation behaviour to the right boundary.
"""

from __future__ import annotations


class AriaError(Exception):
    """Base class for every error raised by the Aria package."""


class ConfigError(AriaError):
    """Invalid or missing configuration (e.g. a required credential)."""


class ModelError(AriaError):
    """The language model call failed (network, auth, provider, or decoding)."""


class EmbeddingError(AriaError):
    """Computing or loading embeddings failed."""


class MemoryStoreError(AriaError):
    """A memory/persistence backend (SQLite or the vector store) failed."""


class ToolError(AriaError):
    """A tool raised while executing inside the ReAct loop."""


__all__ = [
    "AriaError",
    "ConfigError",
    "EmbeddingError",
    "MemoryStoreError",
    "ModelError",
    "ToolError",
]
