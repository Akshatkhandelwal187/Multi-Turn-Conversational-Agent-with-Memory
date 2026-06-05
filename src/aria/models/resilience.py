"""Resilience wrapper for chat models.

Wraps any LangChain-style chat model so ``invoke`` retries transient failures with
exponential backoff + jitter (via :mod:`tenacity`) and surfaces a typed
:class:`~aria.exceptions.ModelError` on final failure — important because the Hugging
Face Inference API can be flaky / rate-limited. ``bind_tools`` returns a wrapped model
so the ReAct loop is covered too; streaming is proxied as-is.
"""

from __future__ import annotations

from typing import Any

from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from ..exceptions import ModelError


class RetryingChatModel:
    """Proxy that adds retry/backoff and typed errors to a chat model's ``invoke``."""

    def __init__(self, inner: Any, *, max_retries: int = 3, base_delay: float = 0.5) -> None:
        self.inner = inner
        self.max_retries = max_retries
        self.base_delay = base_delay

    def _retrying(self) -> Retrying:
        return Retrying(
            stop=stop_after_attempt(self.max_retries + 1),
            wait=wait_exponential(multiplier=self.base_delay, max=8) + wait_random(0, 0.3),
            retry=retry_if_exception_type(Exception),
            reraise=True,
        )

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        try:
            for attempt in self._retrying():
                with attempt:
                    return self.inner.invoke(*args, **kwargs)
        except Exception as exc:
            raise ModelError(str(exc)) from exc
        raise ModelError("model invocation failed")  # pragma: no cover - unreachable

    def stream(self, *args: Any, **kwargs: Any) -> Any:
        return self.inner.stream(*args, **kwargs)

    def bind_tools(self, *args: Any, **kwargs: Any) -> RetryingChatModel:
        return RetryingChatModel(
            self.inner.bind_tools(*args, **kwargs),
            max_retries=self.max_retries,
            base_delay=self.base_delay,
        )

    def with_structured_output(self, *args: Any, **kwargs: Any) -> Any:
        return self.inner.with_structured_output(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Delegate any other attribute access to the wrapped model.
        return getattr(self.inner, name)


__all__ = ["RetryingChatModel"]
