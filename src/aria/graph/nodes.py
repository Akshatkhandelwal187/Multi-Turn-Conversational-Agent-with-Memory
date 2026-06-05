"""The cognitive graph's node functions.

Flow: ``retrieve_memory`` (assemble context) → ``agent`` (call the model, possibly
looping through ``tools``) → ``write_memory`` (persist the exchange) → conditional
``reflect`` / ``summarize``. The agent node delegates to a :class:`ReActAgent` when
tools are enabled, otherwise calls the model directly. All long-term side effects go
through the injected :class:`~aria.memory.MemoryManager`, keeping the nodes thin and
testable. ``usage`` is accumulated by reading the current value and returning a copy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)

from ..observability.metrics import Timer
from ..observability.tokens import count_message_tokens, count_tokens
from ..utils.messages import message_text
from .state import AriaState

if TYPE_CHECKING:
    from ..config import Settings
    from ..memory.manager import MemoryManager
    from .react import ReActAgent


def last_text_of(messages: list[Any], message_cls: type) -> str:
    for msg in reversed(messages):
        if isinstance(msg, message_cls):
            text = message_text(msg).strip()
            if text:
                return text
    return ""


def _tool_messages_this_turn(messages: list[Any]) -> int:
    """Count ToolMessages appended since the most recent user message."""
    count = 0
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            break
        if isinstance(msg, ToolMessage):
            count += 1
    return count


class CognitiveNodes:
    """Bundles the node callables with their injected dependencies."""

    def __init__(
        self,
        model: Any,
        manager: MemoryManager,
        settings: Settings,
        react_agent: ReActAgent | None = None,
    ) -> None:
        self.model = model
        self.manager = manager
        self.settings = settings
        self.react_agent = react_agent

    # -- retrieve ------------------------------------------------------------
    def retrieve_memory(self, state: AriaState) -> dict:
        messages = state.get("messages", [])
        turn_count = state.get("turn_count", 0) + 1
        query = last_text_of(messages, HumanMessage)
        ctx = self.manager.assemble(messages, query, summary=state.get("summary", ""))
        return {
            "system_context": ctx.system_text,
            "retrieved_memories": [{"id": r.id, **r.to_metadata()} for r in ctx.retrieved],
            "profile": ctx.profile,
            "turn_count": turn_count,
            # Fresh per-turn usage baseline (accumulated by later nodes).
            "usage": {
                "tokens_in": 0,
                "tokens_out": 0,
                "latency_ms": 0.0,
                "tool_calls": 0,
                "retrieved_memories": len(ctx.retrieved),
                "reflected": False,
                "summarized": False,
            },
        }

    # -- agent ---------------------------------------------------------------
    def agent(self, state: AriaState) -> dict:
        messages = state.get("messages", [])
        system_context = state.get("system_context") or self.manager.persona
        working = self.manager.working.select(messages)
        if self.react_agent is not None:
            system_context = f"{system_context}\n\n{self.react_agent.tool_instructions}"
        model_messages = [SystemMessage(content=system_context), *working]

        with Timer() as timer:
            if self.react_agent is not None:
                force_final = _tool_messages_this_turn(messages) >= self.settings.max_tool_iters
                response = self.react_agent.act(model_messages, force_final=force_final)
            else:
                response = self.model.invoke(model_messages)

        tool_calls = getattr(response, "tool_calls", None) or []
        usage = dict(state.get("usage", {}))
        usage["tokens_in"] = usage.get("tokens_in", 0) + count_message_tokens(model_messages)
        usage["tokens_out"] = usage.get("tokens_out", 0) + count_tokens(message_text(response))
        usage["latency_ms"] = round(usage.get("latency_ms", 0.0) + timer.elapsed_ms, 2)
        usage["tool_calls"] = usage.get("tool_calls", 0) + len(tool_calls)
        return {"messages": response, "usage": usage}

    # -- write ---------------------------------------------------------------
    def write_memory(self, state: AriaState) -> dict:
        messages = state.get("messages", [])
        user_text = last_text_of(messages, HumanMessage)
        ai_text = last_text_of(messages, AIMessage)
        if user_text:
            self.manager.write(user_text, ai_text)
        if self.settings.persist:
            self.manager.persist()
        return {}

    # -- reflect (runs only when routed here) --------------------------------
    def reflect(self, state: AriaState) -> dict:
        insights = self.manager.maybe_reflect(state.get("turn_count", 0))
        usage = dict(state.get("usage", {}))
        usage["reflected"] = bool(insights)
        return {"usage": usage}

    # -- summarize (runs only when routed here) ------------------------------
    def summarize(self, state: AriaState) -> dict:
        messages = state.get("messages", [])
        new_summary, kept, did = self.manager.maybe_summarize(messages, state.get("summary", ""))
        usage = dict(state.get("usage", {}))
        usage["summarized"] = did
        updates: dict = {"usage": usage}
        if did:
            updates["summary"] = new_summary
            folded = messages[: len(messages) - len(kept)]
            removals = []
            for msg in folded:
                mid = getattr(msg, "id", None)
                if mid:
                    removals.append(RemoveMessage(id=mid))
            if removals:
                updates["messages"] = removals
        return updates

    # -- conditional routing -------------------------------------------------
    def after_write(self, state: AriaState) -> str:
        if self.manager.reflector.should_reflect(state.get("turn_count", 0)):
            return "reflect"
        if self.manager.summarizer.needs_summary(
            state.get("messages", []), state.get("summary", "")
        ):
            return "summarize"
        return "__end__"

    def after_reflect(self, state: AriaState) -> str:
        if self.manager.summarizer.needs_summary(
            state.get("messages", []), state.get("summary", "")
        ):
            return "summarize"
        return "__end__"


__all__ = ["CognitiveNodes", "last_text_of"]
