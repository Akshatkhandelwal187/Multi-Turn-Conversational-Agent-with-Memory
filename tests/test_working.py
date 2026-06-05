"""Tests for the working-memory sliding window."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

from aria.memory.working import WorkingMemory


def _conversation(n: int) -> list:
    msgs: list = []
    for i in range(n):
        msgs.append(HumanMessage(content=f"user message {i}"))
        msgs.append(AIMessage(content=f"assistant reply {i}"))
    return msgs


def test_message_count_bound():
    wm = WorkingMemory(max_messages=3, max_tokens=100_000)
    selected = wm.select(_conversation(5))  # 10 messages
    assert len(selected) == 3
    assert selected[-1].content == "assistant reply 4"  # keeps the most recent


def test_always_keeps_at_least_one():
    wm = WorkingMemory(max_messages=10, max_tokens=1)  # absurdly small token budget
    selected = wm.select(_conversation(3))
    assert len(selected) == 1
    assert selected[-1].content == "assistant reply 2"


def test_order_is_preserved():
    wm = WorkingMemory(max_messages=4, max_tokens=100_000)
    selected = wm.select(_conversation(3))  # 6 messages -> keep last 4, in order
    assert [m.content for m in selected] == [
        "user message 1",
        "assistant reply 1",
        "user message 2",
        "assistant reply 2",
    ]
