"""Assemble the enabled tool set from configuration.

Returns LangChain tool objects for the names listed in ``settings.enabled_tools``.
``search_memory`` is bound to the live memory manager, and ``retrieve_documents`` is
included only when a document index has been attached (RAG phase).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .calculator import calculator
from .datetime_tool import current_datetime
from .search_memory import make_search_memory_tool
from .web_search import make_web_search_tool

if TYPE_CHECKING:
    from ..config import Settings
    from ..memory.manager import MemoryManager


def build_tools(manager: MemoryManager, model: Any, settings: Settings) -> list[Any]:
    """Build the list of enabled tools."""
    factories: dict[str, Any] = {
        "calculator": lambda: calculator,
        "current_datetime": lambda: current_datetime,
        "search_memory": lambda: make_search_memory_tool(manager),
        "web_search": lambda: make_web_search_tool(None),
    }

    doc_index = getattr(manager, "document_index", None)
    if doc_index is not None:
        from .retrieve_documents import make_retrieve_documents_tool

        factories["retrieve_documents"] = lambda: make_retrieve_documents_tool(doc_index, settings)

    tools: list[Any] = []
    for name in settings.enabled_tools:
        factory = factories.get(name)
        if factory is not None:
            tools.append(factory())
    return tools


__all__ = ["build_tools"]
