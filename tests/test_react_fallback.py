"""Tests for the ReAct tool-calling agent (native + structured fallback paths)."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from aria.config import Settings
from aria.graph.cognitive import build_cognitive_agent
from aria.graph.react import ReActAgent
from aria.models.fakes import ScriptedModel
from aria.tools.calculator import calculator

_MSGS = [SystemMessage(content="sys"), HumanMessage(content="what is 2+2?")]


def _settings(**kw):
    return Settings(persist=False, **kw)


def test_structured_fallback_parses_action_then_final():
    model = ScriptedModel(
        ['Action: {"tool": "calculator", "args": {"expression": "2+2"}}', "Final: It is 4."]
    )
    agent = ReActAgent(model, [calculator], _settings(prefer_native_tool_calls=False))

    step1 = agent.act(_MSGS)
    assert step1.tool_calls
    assert step1.tool_calls[0]["name"] == "calculator"
    assert step1.tool_calls[0]["args"] == {"expression": "2+2"}

    step2 = agent.act(_MSGS)
    assert not step2.tool_calls
    assert "4" in step2.content


def test_native_tool_calls_path():
    model = ScriptedModel(
        [
            {
                "content": "",
                "tool_calls": [
                    {
                        "name": "calculator",
                        "args": {"expression": "2+2"},
                        "id": "1",
                        "type": "tool_call",
                    }
                ],
            }
        ]
    )
    agent = ReActAgent(model, [calculator], _settings(prefer_native_tool_calls=True))
    assert agent.use_native is True  # ScriptedModel.bind_tools succeeded
    step = agent.act(_MSGS)
    assert step.tool_calls and step.tool_calls[0]["name"] == "calculator"


def test_unparseable_output_degrades_to_final():
    agent = ReActAgent(
        ScriptedModel(["I think it is four."]),
        [calculator],
        _settings(prefer_native_tool_calls=False),
    )
    step = agent.act(_MSGS)
    assert not step.tool_calls
    assert "four" in step.content


def test_force_final_skips_tools():
    agent = ReActAgent(
        ScriptedModel(['Action: {"tool": "calculator", "args": {"expression": "2+2"}}']),
        [calculator],
        _settings(prefer_native_tool_calls=False),
    )
    step = agent.act(_MSGS, force_final=True)
    assert not step.tool_calls


def test_graph_executes_a_tool_end_to_end():
    settings = Settings(
        persist=False,
        embedder="hashing",
        hashing_dim=256,
        reflection_every_k_turns=0,
        summary_token_budget=100_000,
        enable_tools=True,
        enabled_tools=["calculator"],
        prefer_native_tool_calls=False,
        max_tool_iters=4,
    )
    model = ScriptedModel(
        ['Action: {"tool": "calculator", "args": {"expression": "21*2"}}', "Final: It is 42."]
    )
    agent = build_cognitive_agent(model=model, settings=settings)
    result = agent.invoke(
        {"messages": [HumanMessage(content="what is 21 times 2?")]},
        {"configurable": {"thread_id": "calc"}},
    )
    assert "42" in result["messages"][-1].content
    assert result["usage"]["tool_calls"] >= 1
