"""Assembles and compiles the cognitive agent graph.

Wiring::

    START → retrieve_memory → agent → write_memory ─┬─(reflect?)→ reflect ─┐
                                                     ├─(summarize?)────────┤
                                                     └─────────────────────┴→ summarize → END

The agent node is the plain model caller here; the tools phase swaps in the ReAct
loop and routes ``agent → tools → agent`` before ``write_memory``. :func:`build_cognitive_agent`
returns an :class:`AriaAgent` wrapper that proxies the LangGraph runnable while also
exposing the live :class:`~aria.memory.MemoryManager` for the UI and tools.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langgraph.graph import END, START, StateGraph

from .checkpointer import make_checkpointer
from .nodes import CognitiveNodes
from .state import AriaState

if TYPE_CHECKING:
    from ..config import Settings
    from ..memory.manager import MemoryManager


class AriaAgent:
    """A thin wrapper over the compiled graph that also carries its memory manager."""

    def __init__(self, graph: Any, manager: MemoryManager, settings: Settings, model: Any):
        self.graph = graph
        self.manager = manager
        self.settings = settings
        self.model = model

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        return self.graph.invoke(*args, **kwargs)

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        return self.graph.stream(*args, **kwargs)

    def get_state(self, *args: Any, **kwargs: Any) -> Any:
        return self.graph.get_state(*args, **kwargs)

    def get_state_history(self, *args: Any, **kwargs: Any) -> Any:
        return self.graph.get_state_history(*args, **kwargs)

    def update_state(self, *args: Any, **kwargs: Any) -> Any:
        return self.graph.update_state(*args, **kwargs)


def _assemble_graph(nodes: CognitiveNodes, tool_node: Any | None = None) -> StateGraph:
    """Build the StateGraph. When ``tool_node`` is provided, insert the ReAct loop."""
    graph = StateGraph(AriaState)
    graph.add_node("retrieve_memory", nodes.retrieve_memory)
    graph.add_node("agent", nodes.agent)
    graph.add_node("write_memory", nodes.write_memory)
    graph.add_node("reflect", nodes.reflect)
    graph.add_node("summarize", nodes.summarize)

    graph.add_edge(START, "retrieve_memory")
    graph.add_edge("retrieve_memory", "agent")

    if tool_node is not None:
        from langgraph.prebuilt import tools_condition

        graph.add_node("tools", tool_node)
        graph.add_conditional_edges(
            "agent", tools_condition, {"tools": "tools", "__end__": "write_memory"}
        )
        graph.add_edge("tools", "agent")
    else:
        graph.add_edge("agent", "write_memory")

    graph.add_conditional_edges(
        "write_memory",
        nodes.after_write,
        {"reflect": "reflect", "summarize": "summarize", "__end__": END},
    )
    graph.add_conditional_edges(
        "reflect", nodes.after_reflect, {"summarize": "summarize", "__end__": END}
    )
    graph.add_edge("summarize", END)
    return graph


def build_cognitive_agent(
    model: Any | None = None,
    settings: Settings | None = None,
    *,
    memory_manager: MemoryManager | None = None,
    checkpointer: Any | None = None,
    enable_tools: bool | None = None,
) -> AriaAgent:
    """Build the advanced cognitive-memory agent.

    Args:
        model: Chat model (defaults to the configured Hugging Face model). Inject a
            fake for offline tests.
        settings: Configuration (defaults to :func:`aria.config.get_settings`).
        memory_manager: Override the memory manager (tests inject one with no aux model).
        checkpointer: Override the checkpointer (defaults to durable SQLite / in-memory).
        enable_tools: Whether to attach the ReAct tool loop (defaults to settings).
    """
    from ..config import get_settings
    from ..memory import build_memory_manager
    from ..models.factory import build_model

    settings = settings or get_settings()
    settings.ensure_dirs()
    if model is None:
        model = build_model(settings)
    manager = memory_manager or build_memory_manager(settings=settings, model=model)
    if checkpointer is None:
        checkpointer = make_checkpointer(settings)

    tool_node = None
    react_agent = None
    want_tools = settings.enable_tools if enable_tools is None else enable_tools
    if want_tools:
        from langgraph.prebuilt import ToolNode

        from ..tools import build_tools
        from .react import ReActAgent

        tools = build_tools(manager, model, settings)
        if tools:
            react_agent = ReActAgent(model, tools, settings)
            tool_node = ToolNode(tools)

    nodes = CognitiveNodes(model=model, manager=manager, settings=settings, react_agent=react_agent)
    graph = _assemble_graph(nodes, tool_node=tool_node).compile(checkpointer=checkpointer)
    return AriaAgent(graph=graph, manager=manager, settings=settings, model=model)


__all__ = ["AriaAgent", "build_cognitive_agent"]
