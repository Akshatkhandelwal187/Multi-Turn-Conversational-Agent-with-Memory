"""The graph state schema.

Beyond the running ``messages`` (reduced by ``add_messages``, which also lets nodes
delete messages via ``RemoveMessage`` during consolidation), the state threads the
rolling ``summary``, the ``retrieved_memories`` and ``profile`` surfaced this turn (for
the UI), a per-turn ``usage`` dict (merged across nodes), the ``turn_count`` that drives
reflection cadence, and the assembled ``system_context`` handed from the retrieval node
to the agent node.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


def merge_usage(left: dict | None, right: dict | None) -> dict:
    """Reducer that shallow-merges per-node usage contributions within a turn."""
    return {**(left or {}), **(right or {})}


class AriaState(TypedDict, total=False):
    """State for the cognitive agent graph."""

    messages: Annotated[list[BaseMessage], add_messages]
    summary: str
    retrieved_memories: list[dict]
    profile: dict
    usage: Annotated[dict, merge_usage]
    turn_count: int
    system_context: str


__all__ = ["AriaState", "merge_usage"]
