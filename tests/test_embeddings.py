"""Tests for the deterministic hashing embedder."""

from __future__ import annotations

import numpy as np

from aria.embeddings.hashing import HashingEmbedder


def test_dimension_and_shape():
    emb = HashingEmbedder(dim=64)
    vecs = emb.embed(["hello world", "foo bar baz"])
    assert vecs.shape == (2, 64)
    assert vecs.dtype == np.float32
    assert emb.embed_one("hello").shape == (64,)


def test_empty_inputs():
    emb = HashingEmbedder(dim=32)
    assert emb.embed([]).shape == (0, 32)


def test_determinism():
    emb = HashingEmbedder(dim=128)
    a = emb.embed_one("My favorite language is Python")
    b = emb.embed_one("My favorite language is Python")
    assert np.array_equal(a, b)


def test_unit_norm_for_nonempty():
    emb = HashingEmbedder(dim=128)
    v = emb.embed_one("some non empty text")
    assert abs(float(np.linalg.norm(v)) - 1.0) < 1e-5


def test_shared_tokens_are_more_similar():
    emb = HashingEmbedder(dim=512)
    query = emb.embed_one("python recommender library")
    related = emb.embed_one("I am building a python recommender system")
    unrelated = emb.embed_one("the weather tomorrow looks rainy")
    assert float(query @ related) > float(query @ unrelated)
