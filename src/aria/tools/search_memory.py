"""A tool that lets the agent query its *own* long-term memory.

This is on-theme for a memory agent: rather than relying only on what the retrieval
node surfaced, the model can deliberately search episodic memory mid-reasoning (e.g.
"what did the user say about their project?"). The tool closes over the live
:class:`~aria.memory.MemoryManager`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.tools import tool

if TYPE_CHECKING:
    from ..memory.manager import MemoryManager


def make_search_memory_tool(manager: MemoryManager):
    """Build a ``search_memory`` tool bound to ``manager``."""

    @tool
    def search_memory(query: str) -> str:
        """Search your long-term memory for things the user told you earlier.

        Use this to recall facts, preferences, or past statements when the user refers
        to something from earlier or from a previous conversation.
        """
        results = manager.search(query, k=5)
        if not results:
            return "No relevant memories found."
        return "\n".join(f"- {record.text}" for record in results)

    return search_memory


__all__ = ["make_search_memory_tool"]
