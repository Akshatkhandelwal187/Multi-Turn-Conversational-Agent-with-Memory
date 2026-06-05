"""Tests for episodic memory and Generative-Agents retrieval scoring."""

from __future__ import annotations

import time

from aria.config import Settings
from aria.embeddings.hashing import HashingEmbedder
from aria.memory.episodic import EpisodicMemory
from aria.vectorstore.numpy_store import NumpyVectorStore


def _episodic(**weight_overrides) -> EpisodicMemory:
    settings = Settings(persist=False, embedder="hashing", hashing_dim=256, **weight_overrides)
    emb = HashingEmbedder(dim=256)
    return EpisodicMemory(emb, NumpyVectorStore(dim=256), settings)


def test_relevance_retrieval():
    ep = _episodic(relevance_weight=1.0, recency_weight=0.0, importance_weight=0.0)
    ep.add("my favorite color is teal")
    ep.add("i had a sandwich for lunch")
    ep.add("python is my programming language")
    hits = ep.retrieve("what is my favorite color", k=1)
    assert hits and "teal" in hits[0].text


def test_importance_component_orders_results():
    ep = _episodic(relevance_weight=0.0, recency_weight=0.0, importance_weight=1.0)
    ep.add("alpha note", importance=0.9)
    ep.add("beta note", importance=0.1)
    hits = ep.retrieve("unrelated query", k=2)
    assert hits[0].text == "alpha note"


def test_recency_component_orders_results():
    ep = _episodic(relevance_weight=0.0, recency_weight=1.0, importance_weight=0.0)
    old = time.time() - 7 * 24 * 3600
    ep.add("an old memory", timestamp=old)
    ep.add("a fresh memory")  # timestamp defaults to now
    hits = ep.retrieve("anything", k=2)
    assert hits[0].text == "a fresh memory"


def test_retrieval_touches_memories():
    ep = _episodic()
    rec = ep.add("touch me")
    assert ep.store.get(rec.id)["access_count"] == 0
    ep.retrieve("touch me", k=1, touch=True)
    assert ep.store.get(rec.id)["access_count"] == 1


def test_search_is_read_only():
    ep = _episodic()
    rec = ep.add("do not touch via search")
    ep.search("do not touch via search", k=1)
    assert ep.store.get(rec.id)["access_count"] == 0


def test_recent_orders_by_timestamp():
    ep = _episodic()
    ep.add("first", timestamp=time.time() - 100)
    ep.add("second")
    recents = ep.recent(2)
    assert [r.text for r in recents] == ["second", "first"]
