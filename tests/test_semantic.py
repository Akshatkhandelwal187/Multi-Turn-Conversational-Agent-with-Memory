"""Tests for semantic memory: fact extraction and the profile."""

from __future__ import annotations

from aria.memory.semantic import SemanticMemory, heuristic_extract
from aria.memory.store_sqlite import SqliteMemoryStore


def test_heuristic_extracts_name_and_favorite():
    facts = heuristic_extract("My name is Sam and my favorite language is Python.")
    assert facts.get("name") == "Sam"
    assert facts.get("favorite_language") == "Python"


def test_heuristic_extracts_project():
    facts = heuristic_extract("I'm building a recommender system this semester.")
    assert "recommender system" in facts.get("project", "")


def test_update_and_profile_text():
    sm = SemanticMemory(SqliteMemoryStore(None))
    sm.extract_and_update("My name is Sam and my favorite language is Python")
    assert sm.facts().get("name") == "Sam"
    profile = sm.profile_text()
    assert "name: Sam" in profile
    assert "favorite language: Python" in profile


def test_empty_profile_is_blank():
    sm = SemanticMemory(SqliteMemoryStore(None))
    assert sm.profile_text() == ""


def test_llm_extraction(scripted_model):
    model = scripted_model(['Here you go: {"goal": "learn RAG", "role": "student"}'])
    sm = SemanticMemory(SqliteMemoryStore(None), model=model)
    facts = sm.extract_and_update("Some message", use_llm=True)
    assert facts.get("goal") == "learn RAG"
    assert facts.get("role") == "student"
