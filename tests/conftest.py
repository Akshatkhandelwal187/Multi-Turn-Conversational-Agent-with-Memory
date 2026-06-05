"""Shared pytest fixtures and offline defaults.

Guarantees the whole suite runs with **no network and no torch**: the deterministic
hashing embedder and an ephemeral (non-persistent) configuration are the defaults,
and language models are injected fakes.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make both the src package and the repo root importable when pytest is invoked
# from anywhere (the repo root carries the legacy ``agent`` shim).
ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Force offline, reproducible defaults before any Settings instance is built.
os.environ.setdefault("ARIA_EMBEDDER", "hashing")
os.environ.setdefault("ARIA_PERSIST", "false")
os.environ.setdefault("ARIA_HASHING_DIM", "128")


@pytest.fixture
def settings(tmp_path):
    """A fresh, isolated, non-persistent Settings pointing at a temp data dir."""
    from aria.config import Settings

    return Settings(persist=False, data_dir=tmp_path, embedder="hashing", hashing_dim=128)


@pytest.fixture
def persistent_settings(tmp_path):
    """Settings with durable SQLite + on-disk vector store under a temp dir."""
    from aria.config import Settings

    s = Settings(persist=True, data_dir=tmp_path, embedder="hashing", hashing_dim=128)
    s.ensure_dirs()
    return s


@pytest.fixture
def embedder():
    """The deterministic, offline hashing embedder."""
    from aria.embeddings.hashing import HashingEmbedder

    return HashingEmbedder(dim=128)


@pytest.fixture
def recording_model():
    from aria.models.fakes import RecordingModel

    return RecordingModel()


@pytest.fixture
def scripted_model():
    """Factory fixture: ``scripted_model([...])`` or ``scripted_model(handler=fn)``."""
    from aria.models.fakes import ScriptedModel

    def _make(responses=None, handler=None):
        return ScriptedModel(responses=responses, handler=handler)

    return _make
