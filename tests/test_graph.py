"""Tests for the cognitive agent graph (memory-aware, durable, offline)."""

from __future__ import annotations

from langchain_core.messages import HumanMessage

from aria.config import Settings
from aria.graph.cognitive import build_cognitive_agent
from aria.models.fakes import ScriptedModel
from aria.utils.messages import message_text


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _echo_handler(messages):
    """Agent echoes its system context (so retrieved memories surface in the reply);
    also answers the reflection/summary sub-prompts used by memory ops."""
    text = " ".join(message_text(m) for m in messages)
    if "Insights (JSON)" in text:
        return '["The user likes teal"]'
    if "Updated summary" in text:
        return "Rolling summary of the conversation."
    system = message_text(messages[0]) if messages else ""
    return f"I recall: {system}"


def _stable_settings(**overrides) -> Settings:
    base = {
        "persist": False,
        "embedder": "hashing",
        "hashing_dim": 256,
        "reflection_every_k_turns": 0,
        "summary_token_budget": 100_000,
        "enable_tools": False,
    }
    base.update(overrides)
    return Settings(**base)


def test_two_turn_recall():
    agent = build_cognitive_agent(
        model=ScriptedModel(handler=_echo_handler),
        settings=_stable_settings(),
        enable_tools=False,
    )
    cfg = _config("t1")
    agent.invoke({"messages": [HumanMessage(content="My favorite color is teal.")]}, cfg)
    result = agent.invoke(
        {"messages": [HumanMessage(content="What is my favorite color?")]}, cfg
    )
    assert "teal" in result["messages"][-1].content.lower()


def test_message_history_is_thread_isolated():
    agent = build_cognitive_agent(
        model=ScriptedModel(["ok"]), settings=_stable_settings(), enable_tools=False
    )
    agent.invoke({"messages": [HumanMessage(content="thread A secret")]}, _config("A"))
    agent.invoke({"messages": [HumanMessage(content="hello from B")]}, _config("B"))
    state_b = agent.get_state(_config("B"))
    contents = " ".join(message_text(m) for m in state_b.values["messages"])
    assert "thread A secret" not in contents
    assert "hello from B" in contents


def test_usage_is_recorded():
    agent = build_cognitive_agent(
        model=ScriptedModel(["hi"]), settings=_stable_settings(), enable_tools=False
    )
    result = agent.invoke({"messages": [HumanMessage(content="hello")]}, _config("u"))
    usage = result["usage"]
    assert usage["tokens_in"] > 0
    assert usage["retrieved_memories"] == 0  # nothing stored yet on the first turn
    assert usage["reflected"] is False


def test_durable_resumability(tmp_path):
    settings = _stable_settings(persist=True, data_dir=tmp_path)
    cfg = _config("resume")

    agent1 = build_cognitive_agent(
        model=ScriptedModel(["Noted."]), settings=settings, enable_tools=False
    )
    agent1.invoke({"messages": [HumanMessage(content="My favorite color is teal.")]}, cfg)

    # A brand-new agent on the same data dir must restore history + long-term memory.
    agent2 = build_cognitive_agent(
        model=ScriptedModel(["Noted."]), settings=settings, enable_tools=False
    )
    restored = agent2.get_state(cfg).values["messages"]
    assert any("teal" in message_text(m).lower() for m in restored)
    assert any("teal" in r.text.lower() for r in agent2.manager.search("favorite color", k=5))


def test_reflection_trigger_stores_insight():
    agent = build_cognitive_agent(
        model=ScriptedModel(handler=_echo_handler),
        settings=_stable_settings(reflection_every_k_turns=1),
        enable_tools=False,
    )
    result = agent.invoke(
        {"messages": [HumanMessage(content="I really like teal everything.")]}, _config("r")
    )
    assert result["usage"]["reflected"] is True
    assert "The user likes teal" in agent.manager.reflections()


def test_summary_trigger_folds_history():
    agent = build_cognitive_agent(
        model=ScriptedModel(handler=_echo_handler),
        settings=_stable_settings(summary_token_budget=5, summary_keep_last_messages=1),
        enable_tools=False,
    )
    cfg = _config("s")
    agent.invoke({"messages": [HumanMessage(content="A long first message about me.")]}, cfg)
    result = agent.invoke({"messages": [HumanMessage(content="And a second message.")]}, cfg)
    assert result["usage"]["summarized"] is True
    assert result.get("summary")
