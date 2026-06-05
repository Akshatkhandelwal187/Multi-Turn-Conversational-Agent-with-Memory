"""Tests for the agent's tools."""

from __future__ import annotations

import pytest

from aria.config import Settings
from aria.exceptions import ToolError
from aria.memory import build_memory_manager
from aria.tools.calculator import calculator, safe_eval
from aria.tools.datetime_tool import current_datetime
from aria.tools.search_memory import make_search_memory_tool


def test_calculator_arithmetic():
    assert safe_eval("2 + 3 * 4") == 14
    assert safe_eval("(10 - 4) / 2") == 3.0
    assert calculator.invoke({"expression": "2 ** 5"}) == "32"


def test_calculator_rejects_unsafe_input():
    with pytest.raises(ToolError):
        safe_eval("__import__('os').system('ls')")
    with pytest.raises(ToolError):
        safe_eval("2 ** 99999")  # exponent guard


def test_current_datetime_is_iso():
    out = current_datetime.invoke({})
    assert "T" in out and out.count(":") >= 2


def test_search_memory_tool_finds_memories():
    mgr = build_memory_manager(
        settings=Settings(persist=False, embedder="hashing", hashing_dim=256), model=None
    )
    mgr.write("I am allergic to peanuts", "Noted.")
    tool = make_search_memory_tool(mgr)
    out = tool.invoke({"query": "peanut allergy"})
    assert "peanut" in out.lower()


def test_search_memory_tool_handles_empty():
    mgr = build_memory_manager(
        settings=Settings(persist=False, embedder="hashing", hashing_dim=256), model=None
    )
    tool = make_search_memory_tool(mgr)
    assert "No relevant memories" in tool.invoke({"query": "anything"})
