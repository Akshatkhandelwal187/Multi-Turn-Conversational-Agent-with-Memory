"""A pluggable web-search tool — disabled by default.

Shipped as a clearly-bounded stub: with no provider configured it returns a helpful
message rather than failing. This keeps the agent self-contained/offline while leaving
a clean extension point (wire a real search provider in :func:`make_web_search_tool`).
"""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.tools import tool


def make_web_search_tool(provider: Callable[[str], str] | None = None):
    """Build a ``web_search`` tool. If ``provider`` is ``None`` it is a safe no-op."""

    @tool
    def web_search(query: str) -> str:
        """Search the web for current information. Use only for facts not in memory."""
        if provider is None:
            return "Web search is not configured in this deployment."
        return provider(query)

    return web_search


__all__ = ["make_web_search_tool"]
