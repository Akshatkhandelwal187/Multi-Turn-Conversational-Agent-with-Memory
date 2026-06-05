"""Backward-compatibility shim.

The implementation moved into the installable :mod:`aria` package. This module
preserves the original import surface — ``build_agent``, ``build_model``,
``SYSTEM_PERSONA`` and ``DEFAULT_MODEL`` — so existing code, notebooks, and tests
keep working unchanged::

    from agent import build_agent, SYSTEM_PERSONA

New code should import from :mod:`aria` directly and use
:func:`aria.build_cognitive_agent` for the advanced memory pipeline.
"""

from __future__ import annotations

from aria import (  # noqa: F401
    DEFAULT_MODEL,
    SYSTEM_PERSONA,
    build_agent,
    build_cognitive_agent,
    build_model,
)

__all__ = [
    "DEFAULT_MODEL",
    "SYSTEM_PERSONA",
    "build_agent",
    "build_cognitive_agent",
    "build_model",
]
