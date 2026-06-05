"""Tests for the MemoryManager orchestration seam."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from aria.config import Settings
from aria.memory import build_memory_manager


def _manager() -> object:
    settings = Settings(persist=False, embedder="hashing", hashing_dim=256)
    return build_memory_manager(settings=settings, model=None)


def test_write_then_assemble_recalls_fact():
    mgr = _manager()
    mgr.write("My favorite color is teal", "Got it, teal!")

    messages = [HumanMessage(content="What is my favorite color?")]
    ctx = mgr.assemble(messages, query="What is my favorite color?")
    # The fact surfaces via semantic profile and/or episodic retrieval.
    assert "teal" in ctx.system_text.lower()
    assert ctx.profile.get("favorite_color") == "teal"
    assert ctx.working_messages[-1].content == "What is my favorite color?"


def test_assemble_includes_persona_and_summary():
    mgr = _manager()
    ctx = mgr.assemble([HumanMessage(content="hi")], query="hi", summary="We discussed RAG.")
    assert "Aria" in ctx.system_text
    assert "We discussed RAG." in ctx.system_text


def test_fresh_manager_retrieves_nothing():
    mgr = _manager()
    ctx = mgr.assemble([HumanMessage(content="hello")], query="hello")
    assert ctx.retrieved == []


def test_search_finds_written_memory():
    mgr = _manager()
    mgr.write("I am allergic to peanuts", "Noted.")
    results = mgr.search("peanut allergy", k=3)
    assert any("peanut" in r.text.lower() for r in results)


def test_maybe_summarize_triggers_over_budget():
    settings = Settings(
        persist=False, embedder="hashing", hashing_dim=256,
        summary_token_budget=5, summary_keep_last_messages=2,
    )
    mgr = build_memory_manager(settings=settings, model=None)
    long_history = [HumanMessage(content="a fairly long message about my life " * 3)] * 6
    _summary, kept, did = mgr.maybe_summarize(long_history, "")
    assert did is True
    assert len(kept) == 2


def test_reflection_stored_as_memory(scripted_model):
    settings = Settings(persist=False, embedder="hashing", hashing_dim=256, reflection_every_k_turns=1)
    model = scripted_model(['["The user cares about privacy"]'])
    mgr = build_memory_manager(settings=settings, model=model)
    mgr.write("Please never share my data", "Understood.")
    insights = mgr.maybe_reflect(turn_count=1)
    assert insights == ["The user cares about privacy"]
    assert any("privacy" in r.text.lower() for r in mgr.search("privacy", k=5))
    assert "The user cares about privacy" in mgr.reflections()
