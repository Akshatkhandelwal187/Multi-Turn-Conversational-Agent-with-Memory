"""Tests for MemGPT-style consolidation/summarisation."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from aria.config import Settings
from aria.memory.consolidation import Summarizer


def _msgs(n: int) -> list:
    out: list = []
    for i in range(n):
        out.append(HumanMessage(content=f"fact number {i} about me"))
        out.append(AIMessage(content=f"acknowledged {i}"))
    return out


def test_needs_summary_respects_budget():
    settings = Settings(persist=False, summary_token_budget=10)
    summarizer = Summarizer(settings)
    assert summarizer.needs_summary(_msgs(5), "") is True
    assert summarizer.needs_summary([HumanMessage(content="hi")], "") is False


def test_summarize_with_model_keeps_tail(scripted_model):
    settings = Settings(persist=False, summary_keep_last_messages=2)
    summarizer = Summarizer(settings, model=scripted_model(["SUMMARY: durable facts"]))
    new_summary, kept = summarizer.summarize(_msgs(4), previous="")
    assert new_summary == "SUMMARY: durable facts"
    assert len(kept) == 2
    assert kept[-1].content == "acknowledged 3"


def test_extractive_fallback_without_model():
    settings = Settings(persist=False, summary_keep_last_messages=2)
    summarizer = Summarizer(settings, model=None)
    new_summary, kept = summarizer.summarize(_msgs(3), previous="earlier stuff")
    assert "earlier stuff" in new_summary
    assert "fact number 0 about me" in new_summary
    assert len(kept) == 2
