"""Deterministic fake chat models for offline tests, eval, and CI.

These implement just enough of the LangChain chat-model surface (``invoke``,
``stream``, ``bind_tools``, ``with_structured_output``) for the graph, the ReAct
loop, and the evaluation harness to run with no network and fully reproducible
behaviour. This is the same dependency-injection pattern the original project used.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage

Response = str | BaseMessage | dict | Callable[[list[BaseMessage]], Any]


def _to_ai_message(value: Any) -> AIMessage:
    """Coerce a scripted response into an :class:`AIMessage`."""
    if isinstance(value, AIMessage):
        return value
    if isinstance(value, BaseMessage):  # pragma: no cover - defensive
        return AIMessage(content=value.content)
    if isinstance(value, dict):
        content = value.get("content", "")
        tool_calls = value.get("tool_calls")
        if tool_calls:
            return AIMessage(content=content, tool_calls=tool_calls)
        return AIMessage(content=content)
    return AIMessage(content=str(value))


class RecordingModel:
    """Records every ``messages`` list it is asked to answer; returns a fixed reply.

    Used to assert *what the graph passed to the model* (memory replay, persona
    injection). Mirrors the fake from the original test-suite.
    """

    def __init__(self, reply: str = "ack") -> None:
        self.reply = reply
        self.calls: list[list[BaseMessage]] = []

    def invoke(self, messages: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        self.calls.append(list(messages))
        return AIMessage(content=self.reply)

    def bind_tools(self, tools: Any, **kwargs: Any) -> RecordingModel:
        return self


class FlakyModel:
    """Raises on the first ``fail_times`` calls, then returns a fixed reply.

    Used to test the retry/backoff wrapper deterministically.
    """

    def __init__(self, fail_times: int = 2, reply: str = "ok") -> None:
        self.fail_times = fail_times
        self.reply = reply
        self.calls = 0

    def invoke(self, messages: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("transient failure")
        return AIMessage(content=self.reply)

    def bind_tools(self, tools: Any, **kwargs: Any) -> FlakyModel:
        return self


class ScriptedModel:
    """A chat model that returns pre-scripted responses in order.

    Args:
        responses: A list of responses consumed one per ``invoke`` call. Each item
            may be a ``str``, an :class:`AIMessage`, a ``dict`` with ``content`` and
            optional ``tool_calls`` (to simulate native tool-calling), or a callable
            ``fn(messages) -> response``. When exhausted, the last item repeats.
        handler: An optional callable ``fn(messages) -> response`` evaluated on every
            call (takes precedence over ``responses``); ideal for input-dependent
            behaviour such as echoing remembered facts in eval scenarios.
    """

    def __init__(
        self,
        responses: list[Response] | None = None,
        handler: Callable[[list[BaseMessage]], Any] | None = None,
    ) -> None:
        self._responses = list(responses or [])
        self._handler = handler
        self._i = 0
        self.calls: list[list[BaseMessage]] = []
        self.bound_tools: Any = None

    # -- LangChain-ish surface -------------------------------------------------
    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedModel:
        self.bound_tools = tools
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> ScriptedModel:
        return self

    def _next(self, messages: list[BaseMessage]) -> AIMessage:
        if self._handler is not None:
            out: Any = self._handler(messages)
        elif self._i < len(self._responses):
            out = self._responses[self._i]
            self._i += 1
        elif self._responses:
            out = self._responses[-1]
        else:
            out = "ack"
        if callable(out) and not isinstance(out, BaseMessage):
            out = out(messages)
        return _to_ai_message(out)

    def invoke(self, messages: Any, config: Any = None, **kwargs: Any) -> AIMessage:
        msgs = list(messages) if isinstance(messages, (list, tuple)) else [messages]
        self.calls.append(msgs)
        return self._next(msgs)

    def stream(self, messages: Any, config: Any = None, **kwargs: Any) -> Iterator[AIMessage]:
        # Emit the full message as a single chunk — enough for streaming tests.
        yield self.invoke(messages, config, **kwargs)


__all__ = ["FlakyModel", "RecordingModel", "Response", "ScriptedModel"]
