"""Offline tests proving the agent keeps multi-turn memory and a system persona.

These tests inject a fake chat model (no Hugging Face token or network needed), so
they run anywhere — including CI. The fake records exactly which messages the graph
passes to the model on each turn, which is how we assert that earlier turns are
replayed back into the model (i.e. that memory works).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Make the project root importable when pytest is run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import SYSTEM_PERSONA, build_agent  # noqa: E402


class RecordingModel:
    """Minimal stand-in for a chat model.

    Records every ``messages`` list it is asked to answer and returns a fixed
    ``AIMessage``. It implements just ``.invoke``, which is all the graph node uses.
    """

    def __init__(self) -> None:
        self.calls: list[list] = []

    def invoke(self, messages, config=None, **kwargs):
        self.calls.append(list(messages))
        return AIMessage(content="ack")


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def test_system_persona_is_injected():
    model = RecordingModel()
    agent = build_agent(model=model)

    agent.invoke({"messages": [HumanMessage(content="Hello")]}, _config("t1"))

    first_call = model.calls[0]
    assert isinstance(first_call[0], SystemMessage)
    assert first_call[0].content == SYSTEM_PERSONA


def test_memory_persists_across_turns():
    model = RecordingModel()
    agent = build_agent(model=model)
    cfg = _config("conversation-1")

    # Turn 1
    agent.invoke({"messages": [HumanMessage(content="My name is Sam")]}, cfg)
    # Turn 2 — a follow-up that only makes sense with memory.
    agent.invoke({"messages": [HumanMessage(content="What did I just ask you?")]}, cfg)

    second_call = model.calls[1]
    contents = [m.content for m in second_call]

    # Turn 2 must replay the persona, the full prior turn, and the new question.
    assert SYSTEM_PERSONA in contents
    assert "My name is Sam" in contents          # the user's earlier message
    assert "ack" in contents                      # the assistant's earlier reply
    assert "What did I just ask you?" in contents  # the new question

    # Sanity on ordering/types: persona first, new human question last.
    assert isinstance(second_call[0], SystemMessage)
    assert isinstance(second_call[-1], HumanMessage)
    assert second_call[-1].content == "What did I just ask you?"
    assert len(second_call) == 4


def test_threads_are_isolated():
    model = RecordingModel()
    agent = build_agent(model=model)

    agent.invoke({"messages": [HumanMessage(content="Remember: blue")]}, _config("A"))
    # A brand-new thread should start with no memory of thread A.
    agent.invoke({"messages": [HumanMessage(content="Hi there")]}, _config("B"))

    second_call = model.calls[1]
    contents = [m.content for m in second_call]
    assert "Remember: blue" not in contents
    # Only the system persona + the single new human message.
    assert len(second_call) == 2
    assert isinstance(second_call[0], SystemMessage)
    assert second_call[-1].content == "Hi there"


@pytest.mark.skipif(
    not os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
    reason="requires HUGGINGFACEHUB_API_TOKEN to call the Hugging Face Inference API",
)
def test_live_hf_round_trip():
    """End-to-end recall against the real model (runs only when a token is set)."""
    agent = build_agent()  # real Hugging Face model
    cfg = _config("live-test")

    agent.invoke(
        {"messages": [HumanMessage(content="My favorite color is teal.")]}, cfg
    )
    result = agent.invoke(
        {"messages": [HumanMessage(content="What is my favorite color?")]}, cfg
    )

    assert "teal" in result["messages"][-1].content.lower()
