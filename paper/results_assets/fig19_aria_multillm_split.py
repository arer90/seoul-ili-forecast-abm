"""fig19_aria_multillm_split.py — Figure 19 split + enlarged (multi-LLM ARIA grounding).

PURPOSE
    The thesis Figure 19 ("Multi-LLM ARIA grounding comparison") crammed five
    per-model bar panels plus a dense definition block into one small 2x3 image,
    so the model-by-model bars were not distinguishable at print size. This
    regenerator SPLITS the same real-data SSOT into two enlarged, per-model
    figures:

      Figure 26.1  Numeric grounding (direct prompt): per-model fact recall and
                   faithfulness side by side.
      Figure 26.2  Self-Ask decomposition: per-model sub-question fact recall,
                   faithfulness, and mean sub-question count (count on its own
                   axis, never mixed with the [0,1] scores).

    *** LLM COVERAGE — HONEST NOTE (data-driven, never hardcoded) ***
    This split draws EXACTLY the backends present in the measured SSOT JSON;
    no model is invented. As of 2026-06-29 the data contains:
        Claude CLI (cloud) + Codex/OpenAI-GPT CLI (cloud)
        + 5 local Ollama models
        (Qwen2.5-3B, Phi3.5-3.8B, Mistral-7B, Llama3.2-1B, Gemma3-1B).
    Codex was added via ``simulation.scripts.aria_add_cli_backends`` using the
    SAME test set + scoring. Gemini was probed but its free daily quota was
    exhausted (TerminalQuotaError), so it is recorded under
    ``added_backends_skipped`` and NOT drawn (future work — fabrication forbidden).
    The coverage phrase / legend / colors are all derived from the table at draw
    time, so adding Gemini later (when quota recovers) needs no figure edit.

DATA SSOT (read-only, measured — no LLM call, no DB):
    simulation/results/aria_grounding_multi_llm.json  (comparison_table[])

This reuses the original source figure's loader path
(``simulation/scripts/fig_aria_multi_llm.py`` reads the same JSON); only the
LAYOUT changes (split + enlarge, per-model). No values are altered.

Output (PNG, white bg, dpi=150):
    paper/results_assets/fig19_1_aria_numeric_grounding.png
    paper/results_assets/fig19_2_aria_selfask.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA_JSON = PROJECT_ROOT / "simulation" / "results" / "aria_grounding_multi_llm.json"
OUT_DIR = _THIS.parent
OUT_1 = OUT_DIR / "fig19_1_aria_numeric_grounding.png"
OUT_2 = OUT_DIR / "fig19_2_aria_selfask.png"

CLAUDE_COLOR = "#d62728"   # cloud CLI — Claude
CODEX_COLOR = "#2ca02c"    # cloud CLI — Codex (OpenAI GPT)
GEMINI_COLOR = "#9467bd"   # cloud CLI — Gemini (only if measured)
OLLAMA_COLOR = "#1f77b4"   # local Ollama


def _color_for(row: dict) -> str:
    """Per-row bar color: distinguish each cloud CLI backend; Ollama shares blue.

    The figure originally had a single CLI backend (Claude). Codex/Gemini are now
    distinct cloud CLI backends, so they get their own colors (not all-red) to
    keep the legend honest and the bars readable (B5).
    """
    if row.get("tier") == "cli":
        bid = row.get("backend_id", "")
        if "codex" in bid:
            return CODEX_COLOR
        if "gemini" in bid:
            return GEMINI_COLOR
        return CLAUDE_COLOR
    return OLLAMA_COLOR


def _coverage_phrase(table) -> str:
    """Honest 'X cloud CLI vs N local Ollama' phrase from the measured table.

    Lists each cloud-CLI backend that is actually present (Claude / Codex /
    Gemini) and the count of Ollama models; never names a backend not in data.
    """
    cli = [r for r in table if r.get("tier") == "cli"]
    n_ollama = sum(1 for r in table if r.get("tier") == "ollama")
    names = []
    for r in cli:
        bid = r.get("backend_id", "")
        if "claude" in bid:
            names.append("Claude")
        elif "codex" in bid:
            names.append("Codex (OpenAI GPT)")
        elif "gemini" in bid:
            names.append("Gemini")
        else:
            names.append(r.get("label", bid))
    cli_part = " + ".join(names) + " CLI" if names else "CLI"
    return f"{cli_part} vs {n_ollama} local Ollama models"


def _font() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def _load() -> dict:
    with DATA_JSON.open(encoding="utf-8") as fh:
        return json.load(fh)


def _bar_panel(ax, x, labels, vals, colors, title, ylabel, ymax, fmt) -> None:
    ax.bar(x, vals, color=colors, edgecolor="#333333", linewidth=0.8)
    for xi, v in zip(x, vals):
        if np.isfinite(v):
            ax.text(xi, v + ymax * 0.018, fmt.format(v), ha="center", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=12)
    ax.set_ylim(0, ymax)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.tick_params(axis="y", labelsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)


def _legend(fig, table) -> None:
    """Legend with one entry per cloud-CLI backend actually present + Ollama.

    Only draws a Codex / Gemini handle when that backend exists in the measured
    table (no fabricated legend entry for an unmeasured backend).
    """
    ids = {r.get("backend_id", "") for r in table if r.get("tier") == "cli"}
    handles = [mpatches.Patch(color=CLAUDE_COLOR, label="Claude CLI (cloud)")]
    if any("codex" in b for b in ids):
        handles.append(mpatches.Patch(color=CODEX_COLOR, label="Codex / OpenAI GPT (cloud CLI)"))
    if any("gemini" in b for b in ids):
        handles.append(mpatches.Patch(color=GEMINI_COLOR, label="Gemini (cloud CLI)"))
    handles.append(mpatches.Patch(color=OLLAMA_COLOR, label="Ollama (local)"))
    # Lower the legend just under the (now longer, two-line multi-backend)
    # suptitle so the two never collide at the top-right corner.
    fig.legend(handles=handles, loc="upper right", fontsize=11, frameon=True,
               title="Backend", title_fontsize=12, bbox_to_anchor=(0.998, 0.905))


def make_numeric(table) -> None:
    """Figure 26.1 — numeric grounding (fact recall + faithfulness), per-model."""
    labels = [r["label"] for r in table]
    colors = [_color_for(r) for r in table]
    x = np.arange(len(labels))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 7.0))
    _bar_panel(ax1, x, labels,
               [float(r["numeric_fact_recall"]) for r in table], colors,
               "Numeric grounding: fact recall\n(fraction of gold numeric facts recalled, [0,1])",
               "score", 1.08, "{:.2f}")
    _bar_panel(ax2, x, labels,
               [float(r["numeric_faithfulness"]) for r in table], colors,
               "Numeric grounding: faithfulness\n(1 - hallucinated-claim rate, [0,1])",
               "score", 1.08, "{:.2f}")
    _legend(fig, table)
    fig.suptitle(
        f"Figure 26.1  ARIA numeric grounding by model: {_coverage_phrase(table)}\n"
        "(real ABM thesis outputs; direct prompt; live LLM single pass; higher = better)",
        fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT_1, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {OUT_1}  (n_models={len(labels)})")


def make_selfask(table) -> None:
    """Figure 26.2 — Self-Ask (recall + faithfulness + mean sub-question count)."""
    labels = [r["label"] for r in table]
    colors = [_color_for(r) for r in table]
    x = np.arange(len(labels))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20.5, 7.0))
    _bar_panel(ax1, x, labels,
               [float(r["selfask_subq_recall"]) for r in table], colors,
               "Self-Ask: sub-question fact recall\n(gold facts recalled via decomposition, [0,1])",
               "score", 1.08, "{:.2f}")
    _bar_panel(ax2, x, labels,
               [float(r["selfask_faithfulness"]) for r in table], colors,
               "Self-Ask: faithfulness\n(1 - hallucinated-claim rate, [0,1])",
               "score", 1.08, "{:.2f}")
    cnts = [float(r["selfask_mean_n_subq"]) for r in table]
    _bar_panel(ax3, x, labels, cnts, colors,
               "Self-Ask: mean sub-questions per query\n(COUNT, not in [0,1] - separate axis)",
               "count (sub-questions)", max(cnts) * 1.25, "{:.1f}")
    _legend(fig, table)
    fig.suptitle(
        f"Figure 26.2  ARIA Self-Ask grounding by model: {_coverage_phrase(table)}\n"
        "(real ABM thesis outputs; question decomposed into sub-questions, each grounded; live LLM single pass)",
        fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(OUT_2, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {OUT_2}  (n_models={len(labels)})")


def main() -> int:
    _font()
    data = _load()
    table = data["comparison_table"]
    print(f"[coverage] backends in data ({len(table)}): "
          + ", ".join(r["label"] for r in table))
    has_gpt = any("gpt" in r["label"].lower() or "chatgpt" in r["label"].lower()
                  for r in table)
    has_gem = any("gemini" in r["label"].lower() for r in table)
    print(f"[coverage] ChatGPT present: {has_gpt} | Gemini present: {has_gem}")
    make_numeric(table)
    make_selfask(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
