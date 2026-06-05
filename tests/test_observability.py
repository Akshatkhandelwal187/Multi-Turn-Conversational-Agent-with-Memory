"""Tests for token counting, usage tracking, and timing."""

from __future__ import annotations

import time

from langchain_core.messages import AIMessage, HumanMessage

from aria.observability import (
    Timer,
    TurnUsage,
    UsageTracker,
    count_message_tokens,
    count_tokens,
)


def test_count_tokens_nonzero():
    assert count_tokens("hello there friend") > 0
    assert count_tokens("") == 0


def test_count_message_tokens():
    msgs = [HumanMessage(content="What is my favorite color?"), AIMessage(content="Teal.")]
    assert count_message_tokens(msgs) > count_tokens("Teal.")


def test_usage_tracker_totals():
    tracker = UsageTracker()
    tracker.record(TurnUsage(tokens_in=10, tokens_out=5, tool_calls=1, latency_ms=20.0))
    tracker.record(TurnUsage(tokens_in=20, tokens_out=10, tool_calls=0, latency_ms=40.0))
    totals = tracker.totals()
    assert totals["turns"] == 2
    assert totals["total_tokens"] == 45
    assert totals["tool_calls"] == 1
    assert totals["avg_latency_ms"] == 30.0
    assert tracker.last().tokens_in == 20


def test_timer_measures_elapsed():
    with Timer() as t:
        time.sleep(0.005)
    assert t.elapsed_ms >= 4.0
