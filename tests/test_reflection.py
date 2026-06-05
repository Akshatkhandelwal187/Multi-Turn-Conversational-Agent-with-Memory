"""Tests for reflection (insight synthesis)."""

from __future__ import annotations

from aria.config import Settings
from aria.memory.reflection import Reflection
from aria.memory.schema import MemoryRecord


def test_should_reflect_cadence():
    settings = Settings(persist=False, reflection_every_k_turns=2)
    reflection = Reflection(settings)
    assert reflection.should_reflect(2) is True
    assert reflection.should_reflect(4) is True
    assert reflection.should_reflect(3) is False
    assert reflection.should_reflect(0) is False


def test_reflection_disabled_when_k_zero():
    settings = Settings(persist=False, reflection_every_k_turns=0)
    assert Reflection(settings).should_reflect(4) is False


def test_reflect_parses_json_array(scripted_model):
    settings = Settings(persist=False)
    model = scripted_model(['["The user is an ML practitioner", "They value concise answers"]'])
    reflection = Reflection(settings, model=model)
    memories = [MemoryRecord(text="I work on recommender systems")]
    insights = reflection.reflect(memories)
    assert insights == ["The user is an ML practitioner", "They value concise answers"]


def test_reflect_is_noop_without_model():
    settings = Settings(persist=False)
    assert Reflection(settings).reflect([MemoryRecord(text="x")]) == []
