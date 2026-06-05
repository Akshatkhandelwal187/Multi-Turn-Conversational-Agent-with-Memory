"""The cognitive graph's node functions.

Flow: ``retrieve_memory`` (assemble context) → ``agent`` (call the model) →
``write_memory`` (persist the exchange) → conditional ``reflect`` / ``summarize``.
The agent node here is the plain (non-tool) caller; the ReAct tool loop replaces it in
the tools phase. All long-term side effects go through the injected
:class:`~aria.memory.MemoryManager`, keeping the nodes thin and testable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage

from ..observability.metrics import Timer
from ..observability.tokens import count_message_tokens, count_tokens
from ..utils.messages import message_text
from .state import AriaState

if TYPE_CHECKING:
    from ..config import Settings
    from ..memory.manager import MemoryManager


def last_text_of(messages: list[Any], message_cls: type) -> str:
    for msg in reversed(messages):
        if isinstance(msg, message_cls):
            text = message_text(msg).strip()
            if text:
                return text
    return ""


class CognitiveNodes:
    """Bundles the node callables with their injected dependencies."""

    def __init__(self, model: Any, manager: MemoryManager, settings: Settings) -> None:
        self.model = model
        self.manager = manager
        self.settings = settings

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
            # Reset the per-turn usage flags so merged usage reflects this turn only.
            "usage": {
                "reflected": False,
                "summarized": False,
                "tool_calls": 0,
                "retrieved_memories": len(ctx.retrieved),
            },
        }

    # -- agent (plain; replaced by the ReAct loop in the tools phase) --------
    def agent(self, state: AriaState) -> dict:
        system_context = state.get("system_context") or self.manager.persona
        working = self.manager.working.select(state.get("messages", []))
        model_messages = [SystemMessage(content=system_context), *working]
        with Timer() as timer:
            response = self.model.invoke(model_messages)
        usage = {
            "tokens_in": count_message_tokens(model_messages),
            "tokens_out": count_tokens(message_text(response)),
            "latency_ms": round(timer.elapsed_ms, 2),
        }
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
        return {"usage": {"reflected": bool(insights)}}

    # -- summarize (runs only when routed here) ------------------------------
    def summarize(self, state: AriaState) -> dict:
        messages = state.get("messages", [])
        new_summary, kept, did = self.manager.maybe_summarize(messages, state.get("summary", ""))
        updates: dict = {"usage": {"summarized": did}}
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
