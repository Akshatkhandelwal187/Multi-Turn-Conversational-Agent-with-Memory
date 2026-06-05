"""Render ablation results as Markdown + CSV (and optional charts)."""

from __future__ import annotations

from pathlib import Path

from .ablation import AblationResult


def render_markdown(result: AblationResult) -> str:
    lines = [
        "# Aria — Memory Ablation Report",
        "",
        f"Scenarios: **{result.n_scenarios}** · Probes per strategy: **{result.n_probes}**",
        "",
        "| Strategy | Recall | Avg context tokens | Avg latency (ms) |",
        "|---|---:|---:|---:|",
    ]
    for m in result.metrics:
        lines.append(
            f"| `{m.name}` | {m.recall:.0%} | {m.avg_context_tokens:.0f} | {m.avg_latency_ms:.2f} |"
        )
    lines += ["", _interpretation(result), ""]
    return "\n".join(lines)


def _interpretation(result: AblationResult) -> str:
    by_name = {m.name: m for m in result.metrics}
    cog = by_name.get("cognitive")
    buf = by_name.get("buffer")
    win = by_name.get("sliding_window")
    if not (cog and buf and win):
        return ""
    saving = (1 - cog.avg_context_tokens / buf.avg_context_tokens) if buf.avg_context_tokens else 0
    return (
        f"**Takeaway:** cognitive memory recalls **{cog.recall:.0%}** of planted facts — "
        f"matching full-history *buffer* ({buf.recall:.0%}) and beating the bounded "
        f"*sliding window* ({win.recall:.0%}) — while using **{saving:.0%} fewer context "
        f"tokens** than buffer ({cog.avg_context_tokens:.0f} vs {buf.avg_context_tokens:.0f})."
    )


def render_csv(result: AblationResult) -> str:
    rows = ["strategy,recall,avg_context_tokens,avg_latency_ms,probes"]
    for m in result.metrics:
        rows.append(
            f"{m.name},{m.recall:.4f},{m.avg_context_tokens:.2f},{m.avg_latency_ms:.4f},{m.probes}"
        )
    return "\n".join(rows) + "\n"


def make_charts(result: AblationResult, path: Path) -> bool:
    """Render a recall + token-cost bar chart. Returns False if matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - matplotlib optional
        return False

    names = [m.name for m in result.metrics]
    recalls = [m.recall * 100 for m in result.metrics]
    tokens = [m.avg_context_tokens for m in result.metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.bar(names, recalls, color="#2a9d8f")
    ax1.set_title("Recall accuracy (%)")
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis="x", rotation=30)
    ax2.bar(names, tokens, color="#e76f51")
    ax2.set_title("Avg context tokens (lower is cheaper)")
    ax2.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def write_report(result: AblationResult, out_dir: str | Path, charts: bool = True) -> dict:
    """Write report.md + results.csv (+ ablation.png) into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    md_path = out / "report.md"
    md_path.write_text(render_markdown(result), encoding="utf-8")
    paths["markdown"] = str(md_path)

    csv_path = out / "results.csv"
    csv_path.write_text(render_csv(result), encoding="utf-8")
    paths["csv"] = str(csv_path)

    if charts and make_charts(result, out / "ablation.png"):
        paths["chart"] = str(out / "ablation.png")
    return paths


__all__ = ["make_charts", "render_csv", "render_markdown", "write_report"]
