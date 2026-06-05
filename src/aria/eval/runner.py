"""CLI entry point for the evaluation harness.

Run the offline, reproducible ablation::

    python -m aria.eval.runner            # or: aria-eval

Writes a Markdown + CSV (+ PNG) report and prints a summary table. The whole default
path is deterministic and needs no network. ``--live`` additionally grades a real model
end-to-end (requires HUGGINGFACEHUB_API_TOKEN).
"""

from __future__ import annotations

import argparse
import sys

from ..config import Settings
from ..logging import get_logger
from .ablation import run_ablation
from .benchmark import generate_benchmark
from .report import render_markdown, write_report

log = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aria memory evaluation / ablation harness.")
    parser.add_argument("--scenarios", type=int, default=8, help="number of scenarios")
    parser.add_argument("--facts", type=int, default=3, help="facts planted per scenario")
    parser.add_argument("--seed", type=int, default=0, help="random seed (reproducibility)")
    parser.add_argument("--out", default="eval_reports", help="output directory")
    parser.add_argument("--no-charts", action="store_true", help="skip matplotlib charts")
    parser.add_argument("--live", action="store_true", help="also grade a real HF model")
    return parser


def _eval_settings() -> Settings:
    # Offline, reproducible; bound the cognitive window so its token savings are visible.
    return Settings(
        persist=False,
        embedder="hashing",
        hashing_dim=256,
        working_window_messages=6,
        episodic_top_k=5,
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    settings = _eval_settings()

    scenarios = generate_benchmark(args.scenarios, args.facts, args.seed)
    result = run_ablation(scenarios, settings)
    paths = write_report(result, args.out, charts=not args.no_charts)

    print(render_markdown(result))
    print(f"\nReport written to: {paths.get('markdown')}")
    if "chart" in paths:
        print(f"Chart written to:  {paths['chart']}")

    if args.live:
        _run_live(scenarios, settings)
    return 0


def _run_live(scenarios: list, settings: Settings) -> None:  # pragma: no cover - needs token
    """Grade the real cognitive agent end-to-end (deterministic substring scorer)."""
    import os

    if not os.environ.get("HUGGINGFACEHUB_API_TOKEN"):
        print("\n[--live] Skipped: HUGGINGFACEHUB_API_TOKEN not set.")
        return

    from langchain_core.messages import HumanMessage

    from ..graph.cognitive import build_cognitive_agent
    from ..utils.messages import message_text
    from .scorer import contains_needle

    correct = total = 0
    for index, scenario in enumerate(scenarios):
        agent = build_cognitive_agent(settings=settings)  # fresh memory per scenario
        thread = {"configurable": {"thread_id": f"live-{index}"}}
        for turn in scenario.turns:
            if turn.role != "user":
                continue
            result = agent.invoke({"messages": [HumanMessage(content=turn.text)]}, thread)
            if turn.kind == "probe" and turn.needle:
                answer = message_text(result["messages"][-1])
                total += 1
                correct += int(contains_needle(answer, turn.needle))
    if total:
        pct = correct / total
        print(f"\n[--live] End-to-end recall on real model: {correct}/{total} = {pct:.0%}")


if __name__ == "__main__":
    sys.exit(main())
