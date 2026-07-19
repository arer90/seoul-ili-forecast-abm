"""fig_aria_multi_llm.py — ARIA multi-LLM grounding comparison (Fig 15).

Reproducible generator for the multi-LLM ARIA evaluation figure (replaces the old
slide-mockup Fig 15). Reads the real measured evaluation JSON and draws Claude CLI
vs five local Ollama models across the ARIA grounding metrics.

Honest framing:
  - All metrics are grounded against REAL thesis outputs (ABM forward-validation +
    ABM real-wave fit). There is no "ungrounded" comparison run; the grounding
    contract itself IS the baseline — every model is scored on how well it recalls
    and stays faithful to the same gold facts. The figure caption states this.
  - fact_recall and faithfulness are in [0, 1] (higher = better). The
    mean-sub-question count is a COUNT (not in [0,1]) and is drawn on its own
    separate axis so the two scales are never mixed.

Data source (read-only, measured — no LLM call, no DB, no retraining):
    simulation/results/aria_grounding_multi_llm.json
      comparison_table[].{numeric_fact_recall, numeric_faithfulness,
                          selfask_subq_recall, selfask_faithfulness,
                          selfask_mean_n_subq}

Output:
    simulation/results/figures/aria_multi_llm_comparison.png

Discipline: matplotlib Agg, English labels (DejaVu Sans), deterministic, sqlite=0
(reads a JSON only), honest skip if the JSON is absent (no fabricated data).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless render
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parents[2]
DATA_JSON = PROJECT_ROOT / "simulation" / "results" / "aria_grounding_multi_llm.json"
FIG_DIR = PROJECT_ROOT / "simulation" / "results" / "figures"
OUT_PNG = FIG_DIR / "aria_multi_llm_comparison.png"

CLAUDE_COLOR = "#d62728"
OLLAMA_COLOR = "#1f77b4"

# (json key, panel title, scale annotation) for the four [0,1] metrics.
SCORE_PANELS = [
    ("numeric_fact_recall", "Numeric grounding: fact recall",
     "fraction of gold facts recalled, [0,1]"),
    ("numeric_faithfulness", "Numeric grounding: faithfulness",
     "1 - hallucinated-claim rate, [0,1]"),
    ("selfask_subq_recall", "Self-Ask: sub-question fact recall",
     "fraction of gold facts recalled via decomposition, [0,1]"),
    ("selfask_faithfulness", "Self-Ask: faithfulness",
     "1 - hallucinated-claim rate, [0,1]"),
]


def _setup_font() -> str:
    """Force the default English font for paper figures."""
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans"


def _load() -> dict | None:
    """Load aria_grounding_multi_llm.json (read-only). None if absent."""
    if not DATA_JSON.exists():
        return None
    with DATA_JSON.open(encoding="utf-8") as fh:
        return json.load(fh)


def build_figure() -> Path | None:
    """Draw the multi-LLM ARIA grounding comparison. Honest skip if no data.

    Returns:
        OUT_PNG path, or None if the JSON is absent / empty.

    Side effects: writes OUT_PNG; reads DATA_JSON (read-only). No DB / LLM call.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _setup_font()
    data = _load()
    if not data or not data.get("comparison_table"):
        print(f"[skip] {DATA_JSON} absent/empty — no fabricated figure drawn.")
        return None

    table = data["comparison_table"]
    labels = [r["label"] for r in table]
    tiers = [r.get("tier", "ollama") for r in table]
    colors = [CLAUDE_COLOR if t == "cli" else OLLAMA_COLOR for t in tiers]
    x = np.arange(len(labels))

    # 2x2 score panels + a 5th panel (count, separate axis) -> 2 rows x 3 cols.
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.6))
    axes = axes.ravel()

    for ax, (key, title, scale) in zip(axes[:4], SCORE_PANELS):
        vals = [float(r.get(key, np.nan)) for r in table]
        ax.bar(x, vals, color=colors, edgecolor="#333333", linewidth=0.6)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel("score", fontsize=10)
        ax.set_title(f"{title}\n({scale})", fontsize=11)
        ax.grid(axis="y", alpha=0.25)

    # 5th panel: mean sub-questions = a COUNT (own axis, NOT [0,1]).
    ax_cnt = axes[4]
    cnts = [float(r.get("selfask_mean_n_subq", np.nan)) for r in table]
    ax_cnt.bar(x, cnts, color=colors, edgecolor="#333333", linewidth=0.6)
    for xi, v in zip(x, cnts):
        ax_cnt.text(xi, v + 0.05, f"{v:.1f}", ha="center", fontsize=9)
    ax_cnt.set_xticks(x)
    ax_cnt.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax_cnt.set_ylabel("count (sub-questions)", fontsize=10)
    ax_cnt.set_title(
        "Self-Ask: mean sub-questions per query\n(COUNT, not in [0,1] - separate axis)",
        fontsize=11,
    )
    ax_cnt.grid(axis="y", alpha=0.25)

    # 6th cell: legend + metric definitions text block.
    ax_txt = axes[5]
    ax_txt.axis("off")
    handles = [
        mpatches.Patch(color=CLAUDE_COLOR, label="Claude CLI (cloud)"),
        mpatches.Patch(color=OLLAMA_COLOR, label="Ollama (local)"),
    ]
    ax_txt.legend(handles=handles, loc="upper center", fontsize=11, frameon=True,
                  title="Backend", title_fontsize=11)
    defn = (
        "Metric definitions (all grounded against REAL thesis outputs; higher = better):\n"
        "  - fact recall [0,1]: fraction of the gold numeric facts the answer\n"
        "    correctly recalls.\n"
        "  - faithfulness [0,1]: 1 - (hallucinated / unsupported claims rate);\n"
        "    1.0 = no claim contradicts or invents beyond the gold facts.\n"
        "  - Self-Ask: the question is decomposed into sub-questions, each\n"
        "    grounded independently before synthesis.\n"
        "  - mean sub-questions: a COUNT (own axis), not a quality score.\n\n"
        "Baseline note: the grounding contract is itself the baseline - every\n"
        "model is scored on the SAME gold facts, so the gap to Claude reflects\n"
        "grounded-vs-weakly-grounded behavior, not a different task.\n"
        "Live LLMs, single pass each (non-deterministic); values as measured."
    )
    ax_txt.text(
        0.0, 0.80, defn, ha="left", va="top", fontsize=9.2, color="#333333",
        transform=ax_txt.transAxes, linespacing=1.35,
    )

    fig.suptitle(
        "ARIA grounding: Claude CLI vs local Ollama models\n"
        "(real ABM thesis outputs; live LLM single pass; higher = better)",
        fontsize=15, fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {OUT_PNG}")
    for r in table:
        print(
            f"  {r['label']:14s} num_recall={r['numeric_fact_recall']:.3f} "
            f"num_faith={r['numeric_faithfulness']:.3f} "
            f"sa_recall={r['selfask_subq_recall']:.3f} "
            f"sa_faith={r['selfask_faithfulness']:.3f} "
            f"mean_subq={r['selfask_mean_n_subq']:.1f}"
        )
    return OUT_PNG


def main() -> int:
    """Entry point: regenerate the multi-LLM ARIA figure. 0 on success."""
    return 0 if build_figure() is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
