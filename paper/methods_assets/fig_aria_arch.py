"""Methods figure: ARIA architecture (retrieval-grounded, auditable LLM advisory layer).

Renders the ARIA stack as a top-to-bottom flow:

    Epidemiologist query
      -> ARIA LLM advisory layer        (multi-provider orchestrator; never diagnoses)
      -> MCP epi server                 (11 retrieval-grounded tools over epi_real_seoul.db)
      -> Grounding / provenance         (numeric grounding, semantic-consistency check,
                                         Hermes SHA-256 hash-chain audit ledger)
      -> Self-Ask (SubQ) + CoVe verify  (atomic numeric sub-questions; independent grounding gate)
      -> Human-in-the-loop output       (grounded, hedged, audit-trailed)

The figure deliberately separates the *retrieval-grounded* spine (every numeric claim must trace to
an MCP tool result) from the *auditable* spine (every result carries a provenance envelope and every
turn is hash-chained), because those are the two methodological guarantees ARIA contributes over a
bare chat LLM.

SSOT (no retraining, no DB reads — static schematic of the documented implementation):
  simulation/server/mcp_epi.py
    - TOOL_SPECS: 11 MCP tools (epi.query_db, epi.forecast, epi.model_compare, epi.shap_features,
      epi.rt_estimate, epi.lead_time_analysis, epi.outbreak_detect, epi.validity_check,
      epi.literature_rag, epi.scenario_run, epi.international_compare)        (L159-531)
    - read-only SQL guard: validate_read_only + DuckDB READ_ONLY attach       (L910-936, sql_guard)
    - _attach_provenance: {server_version, db_vintage_ts, artifact_sha256, ...} + freshness  (L727-763)
    - _audit_call: append-only mcp_audit.jsonl (ts, tool, args_hash, status)  (L699-724)
  simulation/llm_compare/runner.py
    - _append_audit / verify_audit_chain: Hermes SHA-256 hash chain (prev_hash linkage)  (L134-188)
    - DEFAULT_SYSTEM_PROMPT: "never diagnose; interpret outputs for an epidemiologist"    (L46-54)
  simulation/llm_compare/aria_grounding.py
    - numeric_grounding (fact_recall / n_spurious)                            (L154-181)
    - semantic_consistency (reversed-comparison contradiction check)          (L74-151)
    - self_ask_decompose / self_ask_answer (corpus-free numeric SubQ; Press et al. 2022)  (L377-424)
  simulation/llm_compare/aria_multiagent.py
    - Retriever/Grounder -> Analyst -> Verifier/Critic CoVe gate (Dhuliawala et al. 2023)  (L34-46, L406-484)
  simulation/llm_compare/backends.py
    - discover_backends: Anthropic / Google / OpenAI / Ollama providers       (discover_backends)

Style matches paper/methods_assets/fig_selection_hierarchy.py (DejaVu Sans, savefig.dpi=160,
TEAL/NAVY/amber palette).

Constraints honoured: REAL documented implementation only (no fabricated numbers / capabilities);
seed=42 (no stochastic step — schematic); B5-width friendly aspect; NO uv sync.

Run:
    .venv/bin/python paper/methods_assets/fig_aria_arch.py
Output:
    paper/methods_assets/fig_aria_arch.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "savefig.dpi": 160,
    "figure.dpi": 130,
})

# ── palette (matches thesis figure SSOT) ─────────────────────────────────────
TEAL = "#0f766e"     # MCP tools / grounded spine
NAVY = "#1e3a5f"     # advisory layer / refit
AMBER = "#b45309"    # audit / provenance (the tamper-evident spine)
PLUM = "#6d28d9"     # verification (Self-Ask + CoVe)
GREY = "#6b7280"
INK = "#1f2937"
USER = "#334155"     # human boxes
BAND_GROUND = "#d1fae5"   # retrieval-grounded region tint
BAND_AUDIT = "#fde7c9"    # auditable region tint
EDGE = "#0b3d3a"

np.random.seed(42)  # determinism contract (no stochastic op; documented)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "methods_assets" / "fig_aria_arch.png"

# ── canvas ───────────────────────────────────────────────────────────────────
fig_w, fig_h = 13.4, 9.2   # B5-width friendly
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def box(x, y, w, h, title, lines, fill, txt="white", *, fs_title=11.0, fs_line=8.4,
        edge=EDGE, two_line_title=None, lw=1.7):
    """Rounded box with a bold title row and stacked caption lines (top-anchored)."""
    bb = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.4,rounding_size=1.1",
        linewidth=lw, edgecolor=edge, facecolor=fill, zorder=3,
    )
    ax.add_patch(bb)
    n_title = (title.count("\n") + 1)
    title_y = y + h - 2.0 - (1.4 if n_title > 1 else 0.0)
    ax.text(x + w / 2, title_y, title, ha="center", va="center",
            fontsize=fs_title, fontweight="bold", color=txt, zorder=4, linespacing=1.02)
    sub = "#d8f3ef" if txt == "white" else GREY
    body = "white" if txt == "white" else INK
    p_y0 = title_y - (2.7 if n_title == 1 else 2.4)
    for j, ln in enumerate(lines):
        ax.text(x + w / 2, p_y0 - j * 2.35, ln, ha="center", va="center",
                fontsize=fs_line, color=body if j == 0 else sub, zorder=4, linespacing=1.0)


def varrow(x, y0, y1, color=GREY, lw=2.0, label=None, lx_off=0.0, ls="-"):
    """Downward vertical arrow (y0 -> y1, y0 > y1)."""
    arr = FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=16,
                          linewidth=lw, color=color, zorder=2, linestyle=ls)
    ax.add_patch(arr)
    if label:
        ax.text(x + 1.0 + lx_off, (y0 + y1) / 2, label, ha="left", va="center",
                fontsize=7.8, color=color, style="italic", zorder=5)


def uarrow(x, y0, y1, color=USER, lw=2.0):
    """Upward vertical arrow (y0 -> y1, y0 < y1)."""
    arr = FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=16,
                          linewidth=lw, color=color, zorder=2)
    ax.add_patch(arr)


# ── title ────────────────────────────────────────────────────────────────────
ax.text(50, 97.4, "ARIA — retrieval-grounded, auditable LLM advisory layer",
        ha="center", va="center", fontsize=14.0, fontweight="bold", color=INK)
ax.text(50, 94.2,
        "every numeric claim traces to an MCP tool result; every turn is hash-chain audited "
        "and verified before a human sees it",
        ha="center", va="center", fontsize=9.2, color=GREY, style="italic")

# ── side-region bands ─────────────────────────────────────────────────────────
# Retrieval-grounded spine (advisory layer + MCP tools) on the left/centre;
# Auditable spine (provenance + audit) annotated on the right.
ax.axvspan(0.04, 0.62, ymin=0.16, ymax=0.86, color=BAND_GROUND, alpha=0.0)  # placeholder; bands drawn as boxes below

# Region labels (rotated)
ax.text(2.0, 56, "RETRIEVAL-GROUNDED", ha="center", va="center", rotation=90,
        fontsize=9.0, fontweight="bold", color=TEAL)
ax.text(98.0, 47, "AUDITABLE", ha="center", va="center", rotation=90,
        fontsize=9.0, fontweight="bold", color=AMBER)

cx = 50.0   # main spine x-centre
W = 56.0    # wide-box width
X = cx - W / 2

# ── (1) Epidemiologist query ─────────────────────────────────────────────────
box(X + 6, 86.5, W - 12, 6.0,
    "Epidemiologist query",
    ['"Gangnam alert next week?"   "S3 fatigue vs S2 trade-off?"'],
    USER, fs_title=10.4, fs_line=8.2, edge="#1f2937", lw=1.6)

varrow(cx, 86.4, 80.6)

# ── (2) ARIA LLM advisory layer ──────────────────────────────────────────────
box(X, 70.0, W, 10.4,
    "ARIA LLM advisory layer  —  interchangeable backend",
    ["orchestrator interprets forecasts/simulations, never diagnoses;  grounding (below) is the arbiter",
     "external: Anthropic · Google · OpenAI · Ollama   +   own family: from-scratch LM · LoRA · ensemble",
     "own family = deliberate weak→strong comparison anchors (grounding, not model size, carries skill)"],
    NAVY, fs_title=10.6, fs_line=7.9)

varrow(cx, 69.9, 63.8, color=TEAL, label="tool calls (read-only)", lx_off=0.0)

# ── (3) MCP epi server — 11 retrieval-grounded tools ─────────────────────────
box(X, 44.0, W, 19.4,
    "MCP epi server  —  11 retrieval-grounded tools",
    [""],
    TEAL, fs_title=11.5)
# subtitle line under the title
ax.text(cx, 60.0, "every tool reads epi_real_seoul.db read-only (SQL guard + DuckDB READ_ONLY attach)",
        ha="center", va="center", fontsize=8.3, color="#d8f3ef", zorder=4)

# tool chips in a 3-column grid
tools = [
    "query_db", "forecast", "model_compare",
    "shap_features", "rt_estimate", "lead_time_analysis",
    "outbreak_detect", "validity_check", "literature_rag",
    "scenario_run", "international_compare",
]
n_cols = 3
chip_w, chip_h = 16.2, 2.6
gx0 = cx - (n_cols * chip_w + (n_cols - 1) * 1.6) / 2
gy0 = 56.6
for idx, t in enumerate(tools):
    r, c = divmod(idx, n_cols)
    cxx = gx0 + c * (chip_w + 1.6)
    cyy = gy0 - r * (chip_h + 0.9)
    chip = FancyBboxPatch((cxx, cyy - chip_h), chip_w, chip_h,
                          boxstyle="round,pad=0.2,rounding_size=0.7",
                          linewidth=1.0, edgecolor="#0b3d3a",
                          facecolor="#0b524c", zorder=4)
    ax.add_patch(chip)
    ax.text(cxx + chip_w / 2, cyy - chip_h / 2, f"epi.{t}",
            ha="center", va="center", fontsize=7.6, color="white",
            fontweight="bold", zorder=5)

varrow(cx, 43.9, 37.8, color=TEAL)

# ── (4) Grounding / provenance ───────────────────────────────────────────────
box(X, 24.0, W, 13.6,
    "Grounding  ·  provenance",
    ["numeric grounding  (fact_recall, n_spurious)   ·   semantic-consistency check",
     "provenance envelope per result  {server_version, db_vintage, artifact_sha256, freshness}",
     "Hermes SHA-256 hash-chain audit ledger  (prev_hash linkage, tamper-evident)"],
    "#0e7490", fs_title=11.2, fs_line=8.2, edge="#0b3d3a")

varrow(cx, 23.9, 18.3, color=PLUM)

# ── (5) Self-Ask (SubQ) + CoVe verification ──────────────────────────────────
box(X, 8.0, W, 10.0,
    "Self-Ask (SubQ)  +  CoVe verification",
    ["atomic numeric sub-questions answered from the tool numbers  (corpus-free)",
     "independent CoVe gate: Retriever -> Analyst -> Verifier; reject ungrounded numbers"],
    PLUM, fs_title=11.2, fs_line=8.2, edge="#3b1d70")

varrow(cx, 7.9, 3.6, color=USER, label=None)

# ── (6) Human-in-the-loop output ─────────────────────────────────────────────
box(X + 6, -3.0, W - 12, 6.2,
    "Human-in-the-loop output",
    ["grounded · hedged · audit-trailed   —   epidemiologist decides"],
    "#0369a1", fs_title=10.6, fs_line=8.2, edge="#0c4a6e", lw=1.6)

# ── right-side AUDITABLE bracket: provenance + ledger flow back to the audit log ──
audit_x = X + W + 3.0
# a slim audit-log box to the right of the grounding/verification stages
ab = FancyBboxPatch((audit_x, 9.0), 9.2, 28.0,
                    boxstyle="round,pad=0.4,rounding_size=1.0",
                    linewidth=1.6, edgecolor="#7c3a00",
                    facecolor="#fef3e3", zorder=3)
ax.add_patch(ab)
ax.text(audit_x + 4.6, 34.0, "append-only\naudit ledger", ha="center", va="center",
        fontsize=8.8, fontweight="bold", color=AMBER, zorder=4, linespacing=1.05)
ax.text(audit_x + 4.6, 27.5, "mcp_audit.jsonl\n+ Hermes\nhash chain",
        ha="center", va="center", fontsize=7.6, color="#7c3a00", zorder=4, linespacing=1.1)
ax.text(audit_x + 4.6, 19.5, "verify_audit_\nchain()\n(tamper-evident)",
        ha="center", va="center", fontsize=7.4, color="#7c3a00", style="italic",
        zorder=4, linespacing=1.1)
# arrows: MCP + grounding stages feed the ledger (auditable spine)
for ystart in (30.8, 24.8):
    a = FancyArrowPatch((X + W, ystart), (audit_x, ystart - 1.2),
                        arrowstyle="-|>", mutation_scale=12, linewidth=1.4,
                        color=AMBER, zorder=2, linestyle=(0, (4, 3)))
    ax.add_patch(a)

# ── left-side GROUNDED bracket annotation: tools constrain the advisory layer ──
ground_x = X - 3.0 - 9.2
gb = FancyBboxPatch((ground_x, 46.0), 9.2, 26.0,
                    boxstyle="round,pad=0.4,rounding_size=1.0",
                    linewidth=1.6, edgecolor="#0b3d3a",
                    facecolor="#e6fbf5", zorder=3)
ax.add_patch(gb)
ax.text(ground_x + 4.6, 67.5, "grounding\nconstraint", ha="center", va="center",
        fontsize=8.8, fontweight="bold", color=TEAL, zorder=4, linespacing=1.05)
ax.text(ground_x + 4.6, 60.0, "answers may\nonly cite\ntool results",
        ha="center", va="center", fontsize=7.6, color="#0b3d3a", zorder=4, linespacing=1.15)
ax.text(ground_x + 4.6, 51.0, "no free-form\nnumbers",
        ha="center", va="center", fontsize=7.4, color="#0b3d3a", style="italic",
        zorder=4, linespacing=1.1)
# arrow: MCP tools -> advisory layer constraint
a = FancyArrowPatch((X, 53.7), (ground_x + 9.2, 59.0),
                    arrowstyle="-|>", mutation_scale=12, linewidth=1.4,
                    color=TEAL, zorder=2, linestyle=(0, (4, 3)))
ax.add_patch(a)
a = FancyArrowPatch((ground_x + 9.2, 65.5), (X, 73.0),
                    arrowstyle="-|>", mutation_scale=12, linewidth=1.4,
                    color=TEAL, zorder=2, linestyle=(0, (4, 3)))
ax.add_patch(a)

# ── footer (SSOT provenance, matches sibling figures) ─────────────────────────
fig.text(0.012, 0.014,
         "SSOT: simulation/server/mcp_epi.py (11 MCP tools, provenance, audit) · "
         "simulation/llm_compare/{backends.py (multi-provider + AriaTorchLMBackend own model), "
         "runner.py (Hermes hash chain), aria_grounding.py (numeric/semantic + Self-Ask), "
         "aria_multiagent.py (CoVe)} · scripts/train_aria_modern_lm.py — documented implementation only.",
         ha="left", va="bottom", fontsize=6.6, color="#94a3b8")

fig.savefig(OUT, bbox_inches="tight", dpi=160, facecolor="white")
plt.close(fig)
print(f"wrote {OUT}")
