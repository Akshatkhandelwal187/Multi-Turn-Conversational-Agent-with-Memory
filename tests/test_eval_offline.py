"""Offline tests for the evaluation harness and ablation study."""

from __future__ import annotations

from pathlib import Path

from aria.config import Settings
from aria.eval.ablation import run_ablation
from aria.eval.benchmark import generate_benchmark
from aria.eval.report import render_csv, render_markdown, write_report
from aria.eval.runner import main as eval_main
from aria.eval.scorer import contains_needle, token_f1


def _settings() -> Settings:
    return Settings(
        persist=False, embedder="hashing", hashing_dim=256,
        working_window_messages=6, episodic_top_k=5,
    )


def test_benchmark_is_deterministic():
    a = generate_benchmark(5, seed=7)
    b = generate_benchmark(5, seed=7)
    assert [t.text for s in a for t in s.turns] == [t.text for s in b for t in s.turns]
    assert all(s.probes for s in a)


def test_scorer_helpers():
    assert contains_needle("Your favorite color is Teal!", "teal")
    assert not contains_needle("I don't know", "teal")
    assert token_f1("the capital is paris", "paris") > 0


def test_ablation_cognitive_beats_window_and_is_cheaper_than_buffer():
    result = run_ablation(generate_benchmark(8, seed=0), _settings())
    by_name = {m.name: m for m in result.metrics}

    assert by_name["no_memory"].recall == 0.0
    # Cognitive recalls long-distance facts the bounded window misses...
    assert by_name["cognitive"].recall >= by_name["sliding_window"].recall
    assert by_name["cognitive"].recall >= 0.8
    # ...at far lower token cost than replaying the whole transcript.
    assert by_name["cognitive"].avg_context_tokens < by_name["buffer"].avg_context_tokens


def test_report_rendering_and_writing(tmp_path):
    result = run_ablation(generate_benchmark(3, seed=1), _settings())
    md = render_markdown(result)
    assert "Memory Ablation Report" in md and "cognitive" in md
    assert render_csv(result).startswith("strategy,recall")

    paths = write_report(result, tmp_path, charts=False)
    assert Path(paths["markdown"]).exists()
    assert Path(paths["csv"]).exists()


def test_runner_main_writes_report(tmp_path):
    code = eval_main(["--scenarios", "3", "--seed", "0", "--out", str(tmp_path), "--no-charts"])
    assert code == 0
    assert (tmp_path / "report.md").exists()
    assert (tmp_path / "results.csv").exists()
