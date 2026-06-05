"""ReAct tool-calling with a robust structured-JSON fallback.

7B-class open models served via different inference providers are unreliable at native
tool-calling, so this agent uses two paths:

1. **Native** — when ``prefer_native_tool_calls`` and ``model.bind_tools`` succeed, the
   model emits ``AIMessage.tool_calls`` directly.
2. **Structured fallback (default-safe, the CI-tested path)** — the model is prompted to
   emit ``Action: {"tool": ..., "args": {...}}`` / ``Final: <answer>``; a tolerant
   brace-matching parser extracts the action and synthesises ``tool_calls`` so the same
   LangGraph ``ToolNode`` executes it. Unparseable output degrades to a final answer.

Either way the node returns an ``AIMessage``; the graph's ``tools_condition`` + ``ToolNode``
drive the loop, with a ``max_tool_iters`` guard in the agent node forcing a final answer.
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.messages import AIMessage

from ..utils.jsonparse import extract_json_object
from ..utils.messages import message_text


def _tool_arg_names(tool: Any) -> list[str]:
    try:
        return list(getattr(tool, "args", {}) or {})
    except Exception:  # pragma: no cover - defensive
        return []


def _first_line(text: str) -> str:
    text = (text or "").strip()
    return text.splitlines()[0] if text else ""


def strip_markers(text: str) -> str:
    """Remove a leading ``Final:`` marker (and surrounding whitespace)."""
    if "Final:" in text:
        text = text.split("Final:", 1)[1]
    return text.strip()


class ReActAgent:
    """A single ReAct step: decide to call a tool or give a final answer."""

    def __init__(self, model: Any, tools: list[Any], settings: Any) -> None:
        self.settings = settings
        self.tools = list(tools)
        self.tools_by_name = {t.name: t for t in self.tools}
        self.raw_model = model
        self.native_model: Any = None
        self.use_native = bool(getattr(settings, "prefer_native_tool_calls", True))
        if self.use_native:
            try:
                self.native_model = model.bind_tools(self.tools)
            except (NotImplementedError, AttributeError):
                self.use_native = False
                self.native_model = None
        self.tool_instructions = self._render_instructions()

    def _render_instructions(self) -> str:
        if not self.tools:
            return ""
        lines = ["You have access to tools. Available tools:"]
        for tool in self.tools:
            args = ", ".join(_tool_arg_names(tool))
            lines.append(f"- {tool.name}({args}): {_first_line(tool.description)}")
        lines.append(
            "To use a tool, reply with EXACTLY one JSON object after 'Action:', e.g.\n"
            'Action: {"tool": "<tool_name>", "args": {<arguments>}}\n'
            "When you can answer, reply with:\n"
            "Final: <your answer to the user>\n"
            "Never include both an Action and a Final in the same reply."
        )
        return "\n".join(lines)

    def act(self, model_messages: list[Any], force_final: bool = False) -> AIMessage:
        if force_final or not self.tools:
            return self._final(model_messages)
        if self.use_native and self.native_model is not None:
            response = self.native_model.invoke(model_messages)
            if isinstance(response, AIMessage):
                return response
            return AIMessage(content=message_text(response))
        return self._structured(model_messages)

    def _structured(self, model_messages: list[Any]) -> AIMessage:
        response = self.raw_model.invoke(model_messages)
        text = message_text(response)
        action = self._parse_action(text)
        if action is not None:
            tool_call = {
                "name": action["tool"],
                "args": action.get("args") or {},
                "id": uuid.uuid4().hex,
                "type": "tool_call",
            }
            return AIMessage(content="", tool_calls=[tool_call])
        return AIMessage(content=strip_markers(text) or text)

    def _final(self, model_messages: list[Any]) -> AIMessage:
        response = self.raw_model.invoke(model_messages)
        return AIMessage(content=strip_markers(message_text(response)))

    def _parse_action(self, text: str) -> dict | None:
        marker = text.find("Action:")
        if marker != -1:
            obj = extract_json_object(text[marker + len("Action:") :])
        elif "Final:" in text:
            return None
        else:
            obj = extract_json_object(text)
        if isinstance(obj, dict) and obj.get("tool") in self.tools_by_name:
            return obj
        return None


__all__ = ["ReActAgent", "strip_markers"]
