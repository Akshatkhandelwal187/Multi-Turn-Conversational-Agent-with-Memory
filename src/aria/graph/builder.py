"""Graph builders.

:func:`build_agent` is the original, intentionally minimal agent — a single model
node over ``MessagesState`` with full-history replay. It is preserved verbatim as the
backward-compatible public contract (the original tests assert its exact message
shapes). The advanced, cognitively-inspired pipeline is :func:`build_cognitive_agent`
(see :mod:`aria.graph.cognitive`), added in a later phase and lazily imported so that
``import aria`` stays light.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, MessagesState, StateGraph

from ..constants import SYSTEM_PERSONA

if TYPE_CHECKING:
    from ..config import Settings


def build_agent(model: BaseChatModel | None = None, persona: str = SYSTEM_PERSONA) -> Any:
    """Build and compile the *legacy* LangGraph agent with conversation memory.

    A single ``model`` node prepends the persona to the full running history and
    re-invokes the model every turn (full-history replay). State is the built-in
    ``MessagesState`` and an in-memory checkpointer keeps each thread's messages.

    Args:
        model: Chat model to use. Defaults to the Hugging Face model. Inject a fake
            to run offline.
        persona: System persona prepended to every model invocation.

    Returns:
        A compiled LangGraph application. Invoke it with a ``thread_id`` so the
        checkpointer keeps memory for that conversation.
    """
    if model is None:
        from ..models.factory import build_model

        model = build_model()

    def call_model(state: MessagesState) -> dict:
        messages = [SystemMessage(content=persona)] + state["messages"]
        response = model.invoke(messages)
        return {"messages": response}

    graph = StateGraph(MessagesState)
    graph.add_node("model", call_model)
    graph.add_edge(START, "model")

    return graph.compile(checkpointer=InMemorySaver())


def build_cognitive_agent(
    model: BaseChatModel | None = None,
    settings: Settings | None = None,
    **kwargs: Any,
) -> Any:
    """Build the advanced cognitive-memory agent (lazy import to keep startup light)."""
    from .cognitive import build_cognitive_agent as _build

    return _build(model=model, settings=settings, **kwargs)


__all__ = ["build_agent", "build_cognitive_agent"]
