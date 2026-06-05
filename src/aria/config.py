"""Typed, environment-driven configuration via :mod:`pydantic-settings`.

Every tunable knob in the system lives here as a validated field with a sensible
default, so the agent is fully configurable from the environment / a ``.env`` file
without touching code. Read settings through :func:`get_settings` (cached).

Environment variables use the ``ARIA_`` prefix, e.g. ``ARIA_HF_MODEL``,
``ARIA_EPISODIC_TOP_K``. The Hugging Face token is read from the conventional
``HUGGINGFACEHUB_API_TOKEN`` (also accepted as ``ARIA_HF_TOKEN``).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from .constants import DEFAULT_EMBEDDING_MODEL, DEFAULT_MODEL, SYSTEM_PERSONA

EmbedderName = Literal["hashing", "sentence_transformer"]
VectorBackend = Literal["numpy", "faiss"]


class Settings(BaseSettings):
    """All runtime configuration for Aria."""

    model_config = SettingsConfigDict(
        env_prefix="ARIA_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- language model (Hugging Face Inference API) ---
    hf_model: str = Field(default=DEFAULT_MODEL, description="Chat model repo id.")
    hf_token: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ARIA_HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN"),
        description="Hugging Face Inference API token.",
    )
    hf_provider: str = Field(default="auto", description="HF inference provider routing.")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_new_tokens: int = Field(default=512, gt=0)
    persona: str = Field(default=SYSTEM_PERSONA)

    # --- resilience ---
    request_max_retries: int = Field(default=3, ge=0)
    request_retry_base_delay: float = Field(default=0.5, gt=0)

    # --- embeddings / vector store ---
    embedder: EmbedderName = Field(
        default="hashing",
        description="Default embedder. 'hashing' is deterministic/offline (tests + "
        "no-torch installs); 'sentence_transformer' gives real semantics at runtime.",
    )
    st_model_name: str = Field(default=DEFAULT_EMBEDDING_MODEL)
    hashing_dim: int = Field(default=256, gt=0, description="Dim of the hashing embedder.")
    vector_backend: VectorBackend = Field(default="numpy")

    # --- working memory ---
    working_window_messages: int = Field(default=12, gt=0)
    working_window_tokens: int = Field(default=1500, gt=0)

    # --- episodic memory retrieval (Generative-Agents scoring weights) ---
    episodic_top_k: int = Field(default=4, ge=0)
    relevance_weight: float = Field(default=1.0, ge=0.0)
    recency_weight: float = Field(default=1.0, ge=0.0)
    importance_weight: float = Field(default=1.0, ge=0.0)
    recency_half_life_hours: float = Field(
        default=24.0, gt=0, description="Exponential recency-decay half life."
    )

    # --- consolidation (summarisation) ---
    summary_token_budget: int = Field(
        default=2000, gt=0, description="Trigger summarisation above this token budget."
    )
    summary_keep_last_messages: int = Field(default=6, gt=0)

    # --- reflection ---
    reflection_every_k_turns: int = Field(default=4, ge=0, description="0 disables reflection.")
    reflection_top_memories: int = Field(default=15, gt=0)

    # --- ReAct / tools ---
    enable_tools: bool = Field(default=True)
    max_tool_iters: int = Field(default=4, gt=0)
    prefer_native_tool_calls: bool = Field(
        default=True, description="Try native bind_tools before the structured-JSON fallback."
    )
    enabled_tools: list[str] = Field(
        default_factory=lambda: ["calculator", "current_datetime", "search_memory"]
    )

    # --- document RAG ---
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)
    rag_top_k: int = Field(default=4, ge=0)

    # --- persistence paths ---
    data_dir: Path = Field(default=Path("data"))
    persist: bool = Field(default=True, description="Use durable SQLite + on-disk vector store.")

    # --- logging ---
    log_level: str = Field(default="INFO")
    log_json: bool = Field(default=False)

    @property
    def sqlite_path(self) -> Path:
        return self.data_dir / "aria.sqlite3"

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "checkpoints.sqlite3"

    @property
    def vectorstore_dir(self) -> Path:
        return self.data_dir / "vectors"

    def ensure_dirs(self) -> None:
        """Create the data directories if persistence is enabled."""
        if self.persist:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.vectorstore_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached :class:`Settings` instance."""
    return Settings()


__all__ = ["EmbedderName", "Settings", "VectorBackend", "get_settings"]
