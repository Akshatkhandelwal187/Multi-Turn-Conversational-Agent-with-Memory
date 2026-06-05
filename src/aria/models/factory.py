"""Hugging Face chat-model factory.

Builds a ``ChatHuggingFace`` from configuration. The heavy ``langchain-huggingface``
import is performed lazily inside the function so importing :mod:`aria` (and running
the offline tests, which inject fakes) never requires it or a network connection.
"""

from __future__ import annotations

from typing import Any

from ..config import Settings, get_settings
from .resilience import RetryingChatModel


def build_model(settings: Settings | None = None) -> Any:
    """Create the Hugging Face-backed chat model from configuration.

    Reads the model id, token, provider, temperature and ``max_new_tokens`` from
    :class:`~aria.config.Settings`. Returns a ``ChatHuggingFace`` (standard LangChain
    message interface, ``bind_tools`` / streaming) wrapped in
    :class:`~aria.models.resilience.RetryingChatModel` for backoff + typed errors.
    """
    settings = settings or get_settings()

    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    # `repo_id` is the runtime-correct argument; the type stub over-requires `model`.
    llm = HuggingFaceEndpoint(  # type: ignore[call-arg]
        repo_id=settings.hf_model,
        task="text-generation",
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        provider=settings.hf_provider,
        huggingfacehub_api_token=settings.hf_token,
    )
    return RetryingChatModel(
        ChatHuggingFace(llm=llm),
        max_retries=settings.request_max_retries,
        base_delay=settings.request_retry_base_delay,
    )


__all__ = ["build_model"]
