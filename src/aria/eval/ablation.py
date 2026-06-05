"""Ablation study comparing memory strategies on the recall benchmark.

Each strategy assembles the *context* it would feed the model at a probe turn; we then
measure (a) **recall** — did the planted needle make it into that context, (b) the
**token cost** of the context, and (c) the **assembly latency**. This isolates the
memory mechanism itself (no model needed), demonstrating empirically that cognitive
memory recalls long-distance facts like full-history replay but at a fraction of the
token cost, and far better than a bounded sliding window.

Strategies:
  * ``no_memory``      — only the current question (lower bound).
  * ``sliding_window`` — persona + the last N messages.
  * ``buffer``         — persona + the entire transcript (the classic approach).
  * ``cognitive``      — Aria's memory: profile + retrieved episodic memories + window.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from time import perf_counter
from typing import TYPE_CHECKING

from langchain_core.messages import AIMessage, HumanMessage

from ..constants import SYSTEM_PERSONA
from ..observability.tokens import count_tokens
from ..utils.messages import render_transcript
from .benchmark import Scenario
from .scorer import contains_needle

if TYPE_CHECKING:
    from ..config import Settings


class MemoryStrategy:
    """Base strategy: keeps a running message list and builds a probe-time context."""

    name = "base"

    def __init__(self) -> None:
        self.messages: list = []

    def reset(self) -> None:
        self.messages = []

    def add_user(self, text: str) -> None:
        self.messages.append(HumanMessage(content=text))

    def add_assistant(self, text: str) -> None:
        self.messages.append(AIMessage(content=text))

    def build_context(self, query: str) -> str:  # pragma: no cover - overridden
        raise NotImplementedError


class NoMemoryStrategy(MemoryStrategy):
    name = "no_memory"

    def build_context(self, query: str) -> str:
        return f"{SYSTEM_PERSONA}\n{query}"


class BufferStrategy(MemoryStrategy):
    name = "buffer"

    def build_context(self, query: str) -> str:
        return f"{SYSTEM_PERSONA}\n{render_transcript(self.messages)}\n{query}"


class SlidingWindowStrategy(MemoryStrategy):
    name = "sliding_window"

    def __init__(self, window: int = 6) -> None:
        super().__init__()
        self.window = window

    def build_context(self, query: str) -> str:
        recent = render_transcript(self.messages[-self.window :])
        return f"{SYSTEM_PERSONA}\n{recent}\n{query}"


class CognitiveStrategy(MemoryStrategy):
    name = "cognitive"

    def __init__(self, settings: Settings) -> None:
        super().__init__()
        self.settings = settings
        self._pending_user: str | None = None
        self.reset()

    def reset(self) -> None:
        from ..memory import build_memory_manager

        self.messages = []
        self._pending_user = None
        self.manager = build_memory_manager(settings=self.settings, model=None)

    def add_user(self, text: str) -> None:
        super().add_user(text)
        self._pending_user = text

    def add_assistant(self, text: str) -> None:
        super().add_assistant(text)
        if self._pending_user is not None:
            self.manager.write(self._pending_user, text)
            self._pending_user = None

    def build_context(self, query: str) -> str:
        ctx = self.manager.assemble(self.messages, query)
        return ctx.system_text + "\n" + render_transcript(ctx.working_messages)


@dataclass
class StrategyMetrics:
    name: str
    recall: float
    avg_context_tokens: float
    avg_latency_ms: float
    probes: int


@dataclass
class AblationResult:
    metrics: list[StrategyMetrics] = field(default_factory=list)
    n_scenarios: int = 0
    n_probes: int = 0

    def best_recall(self) -> StrategyMetrics:
        return max(self.metrics, key=lambda m: m.recall)


def default_strategies(settings: Settings) -> list[MemoryStrategy]:
    return [
        NoMemoryStrategy(),
        SlidingWindowStrategy(window=6),
        BufferStrategy(),
        CognitiveStrategy(settings),
    ]


def run_ablation(
    scenarios: list[Scenario],
    settings: Settings,
    strategies: list[MemoryStrategy] | None = None,
) -> AblationResult:
    """Run every strategy over every scenario and aggregate per-strategy metrics."""
    strategies = strategies or default_strategies(settings)
    result = AblationResult(n_scenarios=len(scenarios))

    for strategy in strategies:
        recalls: list[float] = []
        tokens: list[float] = []
        latencies: list[float] = []
        for scenario in scenarios:
            strategy.reset()
            for turn in scenario.turns:
                if turn.role == "user":
                    if turn.kind == "probe" and turn.needle is not None:
                        start = perf_counter()
                        context = strategy.build_context(turn.text)
                        latencies.append((perf_counter() - start) * 1000.0)
                        recalls.append(1.0 if contains_needle(context, turn.needle) else 0.0)
                        tokens.append(count_tokens(context))
                    strategy.add_user(turn.text)
                else:
                    strategy.add_assistant(turn.text)
        result.metrics.append(
            StrategyMetrics(
                name=strategy.name,
                recall=mean(recalls) if recalls else 0.0,
                avg_context_tokens=mean(tokens) if tokens else 0.0,
                avg_latency_ms=mean(latencies) if latencies else 0.0,
                probes=len(recalls),
            )
        )
    result.n_probes = result.metrics[0].probes if result.metrics else 0
    return result


__all__ = [
    "AblationResult",
    "BufferStrategy",
    "CognitiveStrategy",
    "MemoryStrategy",
    "NoMemoryStrategy",
    "SlidingWindowStrategy",
    "StrategyMetrics",
    "default_strategies",
    "run_ablation",
]
