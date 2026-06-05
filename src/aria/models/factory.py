"""Hugging Face chat-model factory.

Builds a ``ChatHuggingFace`` from configuration. The heavy ``langchain-huggingface``
import is performed lazily inside the function so importing :mod:`aria` (and running
the offline tests, which inject fakes) never requires it or a network connection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..config import Settings, get_settings

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


def build_model(settings: Settings | None = None) -> BaseChatModel:
    """Create the Hugging Face-backed chat model from configuration.

    Reads the model id, token, provider, temperature and ``max_new_tokens`` from
    :class:`~aria.config.Settings`. The returned model is a ``ChatHuggingFace`` that
    speaks the standard LangChain message interface (and supports ``bind_tools`` /
    streaming).
    """
    settings = settings or get_settings()

    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

    llm = HuggingFaceEndpoint(
        repo_id=settings.hf_model,
        task="text-generation",
        max_new_tokens=settings.max_new_tokens,
        temperature=settings.temperature,
        provider=settings.hf_provider,
        huggingfacehub_api_token=settings.hf_token,
    )
    return ChatHuggingFace(llm=llm)


__all__ = ["build_model"]
