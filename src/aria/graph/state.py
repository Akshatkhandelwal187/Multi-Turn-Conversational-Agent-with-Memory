"""The graph state schema.

Beyond the running ``messages`` (reduced by ``add_messages``, which also lets nodes
delete messages via ``RemoveMessage`` during consolidation), the state threads the
rolling ``summary``, the ``retrieved_memories`` and ``profile`` surfaced this turn (for
the UI), a per-turn ``usage`` dict, the ``turn_count`` that drives reflection cadence,
and the assembled ``system_context`` handed from the retrieval node to the agent node.

``usage`` has no reducer: nodes read the current value and return an updated copy, so
counters accumulate correctly across the (possibly multi-step) ReAct loop within a
turn, while ``retrieve_memory`` resets it at the start of each turn.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AriaState(TypedDict, total=False):
    """State for the cognitive agent graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    summary: str
    retrieved_memories: list[dict]
    profile: dict
    usage: dict
    turn_count: int
    system_context: str


__all__ = ["AriaState"]
