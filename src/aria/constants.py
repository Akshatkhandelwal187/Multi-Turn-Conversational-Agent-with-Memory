"""Project-wide constants.

These values are part of the public, backward-compatible surface of the package
(re-exported from :mod:`aria` and the legacy ``agent`` shim). ``SYSTEM_PERSONA`` and
``DEFAULT_MODEL`` are preserved verbatim from the original project so existing tests
that assert on them continue to pass byte-for-byte.
"""

from __future__ import annotations

# Default open-source instruct model. Qwen2.5-7B-Instruct is openly accessible
# (no gating) and served by Hugging Face Inference Providers. Override with the
# ``ARIA_HF_MODEL`` / ``HF_MODEL`` environment variable.
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# The embedding model used when the (optional) sentence-transformers backend is
# enabled. Small, fast, and produces 384-dimensional vectors.
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Fixed system persona. Prepended to every model call (never stored in state) so
# the personality stays constant and is not duplicated in the message history.
SYSTEM_PERSONA = (
    "You are Aria, a warm, concise, and helpful assistant. "
    "You remember everything the user has said earlier in this conversation and "
    "use that context to answer follow-up questions accurately. "
    "When the user refers to something they mentioned before, recall it precisely. "
    "Keep answers clear and to the point, and admit when you don't know something."
)

__all__ = ["DEFAULT_EMBEDDING_MODEL", "DEFAULT_MODEL", "SYSTEM_PERSONA"]
