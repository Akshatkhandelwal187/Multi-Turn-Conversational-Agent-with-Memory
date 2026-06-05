"""Evaluation harness: a reproducible multi-turn memory-recall benchmark and an
ablation study comparing memory strategies.
"""

from __future__ import annotations

from .ablation import AblationResult, StrategyMetrics, run_ablation
from .benchmark import Scenario, Turn, generate_benchmark
from .report import render_markdown, write_report
from .scorer import LLMJudge, contains_needle, token_f1

__all__ = [
    "AblationResult",
    "LLMJudge",
    "Scenario",
    "StrategyMetrics",
    "Turn",
    "contains_needle",
    "generate_benchmark",
    "render_markdown",
    "run_ablation",
    "token_f1",
    "write_report",
]
