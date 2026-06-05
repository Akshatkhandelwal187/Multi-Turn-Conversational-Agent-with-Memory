"""Tools available to the ReAct agent."""

from __future__ import annotations

from typing import Any

from .calculator import calculator, safe_eval
from .datetime_tool import current_datetime
from .registry import build_tools
from .search_memory import make_search_memory_tool
from .web_search import make_web_search_tool


def build_tool_node(manager: Any, model: Any, settings: Any) -> Any:
    """Convenience: a LangGraph ToolNode over the enabled tools."""
    from langgraph.prebuilt import ToolNode

    return ToolNode(build_tools(manager, model, settings))


__all__ = [
    "build_tool_node",
    "build_tools",
    "calculator",
    "current_datetime",
    "make_search_memory_tool",
    "make_web_search_tool",
    "safe_eval",
]
