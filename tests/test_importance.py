"""Tests for memory importance scoring (heuristic + LLM)."""

from __future__ import annotations

from aria.memory.importance import ImportanceScorer, heuristic_importance


def test_personal_statements_outrank_small_talk():
    high = heuristic_importance("My name is Sam and my favorite language is Python")
    low = heuristic_importance("ok thanks")
    assert 0.0 <= low < high <= 1.0


def test_scores_are_bounded():
    for text in ["", "?", "I am building a recommender system with a deadline on the 5th"]:
        assert 0.0 <= heuristic_importance(text) <= 1.0


def test_llm_scoring(scripted_model):
    model = scripted_model(['{"importance": 9}'])
    scorer = ImportanceScorer(model=model)
    assert abs(scorer.score("I was born in 1999", use_llm=True) - 0.9) < 1e-6


def test_llm_falls_back_to_heuristic_on_garbage(scripted_model):
    model = scripted_model(["I cannot rate this, sorry."])
    scorer = ImportanceScorer(model=model)
    score = scorer.score("My favorite color is teal", use_llm=True)
    assert score == heuristic_importance("My favorite color is teal")
