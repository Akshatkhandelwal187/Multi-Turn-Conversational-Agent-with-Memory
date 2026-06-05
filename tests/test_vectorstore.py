"""Tests for the numpy vector store (and faiss when available)."""

from __future__ import annotations

import numpy as np
import pytest

from aria.embeddings.hashing import HashingEmbedder
from aria.vectorstore.numpy_store import NumpyVectorStore


@pytest.fixture
def populated():
    emb = HashingEmbedder(dim=256)
    store = NumpyVectorStore(dim=256)
    docs = {
        "m1": "python is great for data science",
        "m2": "I love hiking in the mountains",
        "m3": "machine learning with python and numpy",
    }
    for mid, text in docs.items():
        store.add(mid, emb.embed_one(text), {"text": text, "kind": "note"})
    return emb, store


def test_search_ranks_by_similarity(populated):
    emb, store = populated
    hits = store.search(emb.embed_one("python machine learning"), k=3)
    assert hits[0].id in {"m1", "m3"}
    assert hits[0].score >= hits[-1].score
    assert "python" in hits[0].text


def test_get_update_delete(populated):
    _, store = populated
    assert store.get("m1")["kind"] == "note"
    store.update_metadata("m1", importance=0.9)
    assert store.get("m1")["importance"] == 0.9
    store.delete(["m1"])
    assert store.get("m1") is None
    assert len(store) == 2


def test_filter(populated):
    emb, store = populated
    store.add("m4", emb.embed_one("a reflection"), {"text": "a reflection", "kind": "reflection"})
    hits = store.search(emb.embed_one("reflection"), k=5, filter={"kind": "reflection"})
    assert [h.id for h in hits] == ["m4"]


def test_persistence_round_trip(tmp_path):
    emb = HashingEmbedder(dim=128)
    path = tmp_path / "vs"
    store = NumpyVectorStore(dim=128, path=path)
    store.add("x", emb.embed_one("durable memory"), {"text": "durable memory"})
    store.persist()

    reopened = NumpyVectorStore(dim=128, path=path)
    assert len(reopened) == 1
    hit = reopened.search(emb.embed_one("durable memory"), k=1)[0]
    assert hit.id == "x"
    assert hit.text == "durable memory"


def test_update_existing_id_does_not_grow(populated):
    emb, store = populated
    before = len(store)
    store.add("m1", emb.embed_one("python is still great"), {"text": "python is still great"})
    assert len(store) == before
    assert np.isclose(np.linalg.norm(store.get_embedding("m1")), 1.0, atol=1e-5)
