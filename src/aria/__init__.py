"""Aria — a cognitively-inspired multi-turn conversational agent with memory.

This package upgrades the original minimal LangGraph chatbot into a layered memory
system (working / episodic / semantic memory, consolidation, and reflection), a
tool-using ReAct agent, document RAG, and an empirical evaluation harness — while
preserving the original public API for backward compatibility.

Backward-compatible exports (used by the legacy ``agent`` shim and the original
tests): :data:`SYSTEM_PERSONA`, :data:`DEFAULT_MODEL`, :func:`build_model`,
:func:`build_agent`.
"""

from __future__ import annotations

from .constants import DEFAULT_MODEL, SYSTEM_PERSONA
from .graph.builder import build_agent, build_cognitive_agent
from .models.factory import build_model

__version__ = "1.0.0"

__all__ = [
    "DEFAULT_MODEL",
    "SYSTEM_PERSONA",
    "__version__",
    "build_agent",
    "build_cognitive_agent",
    "build_model",
]
