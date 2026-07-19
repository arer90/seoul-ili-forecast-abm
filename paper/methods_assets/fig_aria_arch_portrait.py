"""Figure 9 (docx) — ARIA architecture, PORTRAIT layout for B5 page.

The ARIA LLM advisory layer is an INTERCHANGEABLE backend spanning external providers
(Anthropic/Google/OpenAI/Ollama) AND the project's OWN comparison-model family (from-scratch
decoder-only LM, LoRA/optimizer-tuned variants, grounded ensemble) — deliberate weak->strong
benchmark anchors that isolate grounding's contribution (grounding, not model size, carries
skill). Every backend is swappable; the grounding gate is the arbiter of correctness.

SSOT (documented implementation only; static schematic):
  simulation/llm_compare/backends.py — LLMBackend + AriaTorchLMBackend (own model, flag-gated)
  simulation/scripts/{train_aria_modern_lm, build_aria_lora_scidata, aria_torch_optimize}.py
  simulation/server/mcp_epi.py (11 tools, provenance, audit);
  simulation/llm_compare/{runner.py Hermes, aria_grounding.py, aria_multiagent.py CoVe}

Run:    .venv/bin/python paper/methods_assets/fig_aria_arch_portrait.py
Output: paper/methods_assets/fig_aria_arch_portrait.png
"""
from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 170, "figure.dpi": 130})

SCF = "STIX Two Math"   # math-script header (matches Figure 27)
_SC = {"A": "\U0001D49C", "B": "ℬ", "C": "\U0001D49E", "D": "\U0001D49F", "E": "ℰ",
       "F": "ℱ", "G": "\U0001D4A2", "H": "ℋ", "I": "ℐ", "J": "\U0001D4A5",
       "K": "\U0001D4A6", "L": "ℒ", "M": "ℳ", "N": "\U0001D4A9", "O": "\U0001D4AA",
       "P": "\U0001D4AB", "Q": "\U0001D4AC", "R": "ℛ", "S": "\U0001D4AE", "T": "\U0001D4AF",
       "U": "\U0001D4B0", "V": "\U0001D4B1", "W": "\U0001D4B2", "X": "\U0001D4B3", "Y": "\U0001D4B4",
       "Z": "\U0001D4B5"}
def sc(s):
    return "".join(_SC.get(ch, ch) for ch in s)

NAVY = "#1e3a5f"; TEAL = "#0f766e"; GREEN = "#2f8f7d"; PURPLE = "#6d28d9"
GREY = "#6b7280"; INK = "#1f2937"; AMBER = "#b45309"
QUERY = "#d5dae6"; CHIP = "#e6fbf5"; OUT = "#123a5e"

OUTP = Path(__file__).resolve().parents[2] / "paper" / "methods_assets" / "fig_aria_arch_portrait.png"

fig, ax = plt.subplots(figsize=(9.4, 12.4))
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

CX = 51.0; W = 84.0; X = CX - W / 2   # near full width


def box(x, y, w, h, title, lines, fill, txt="white", fs_t=14, fs_l=9.4, edge=None, tcol=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5,rounding_size=1.3",
                                linewidth=2.0, edgecolor=edge or fill, facecolor=fill, zorder=3))
    tc = tcol or ("white" if txt == "white" else INK)
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="center",
            fontsize=fs_t, fontweight="bold", color=tc, zorder=4)
    body = "#e6eef7" if txt == "white" else INK
    for j, ln in enumerate(lines):
        ax.text(x + w / 2, y + h - 5.6 - j * 2.55, ln, ha="center", va="center",
                fontsize=fs_l, color=body, zorder=4)


def down(x, y0, y1, color=GREY, label=None):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=17,
                                 linewidth=2.2, color=color, zorder=2))
    if label:
        ax.text(x + 1.3, (y0 + y1) / 2, label, ha="left", va="center",
                fontsize=8.2, color=color, style="italic", zorder=5)


ax.text(CX, 98.6, sc("ARIA — retrieval-grounded, auditable LLM advisory layer"),
        ha="center", va="center", fontsize=16, color=INK, fontfamily=SCF)

# side rotated labels
ax.text(4.0, 58, "RETRIEVAL-GROUNDED", ha="center", va="center", rotation=90,
        fontsize=10, fontweight="bold", color=TEAL)
ax.text(98.5, 40, "AUDITABLE", ha="center", va="center", rotation=90,
        fontsize=10, fontweight="bold", color=AMBER)

# 1 query
box(X + 12, 89.6, W - 24, 6.8, "Epidemiologist query",
    ['e.g. "Gangnam ILI alert next week?"'], QUERY, txt="ink", fs_t=12.5, fs_l=10,
    edge=NAVY, tcol=NAVY)
down(CX, 89.5, 86.4)

# 2 ARIA LLM advisory layer — Layer 1 (backend detail → Appendix J)
box(X, 71.0, W, 15.0, "Layer 1 · ARIA LLM advisory  (orchestrator)",
    ["interprets forecasts and simulations; never answers a number ungrounded",
     "runs on an interchangeable LLM backend  (multi-provider + own model — Appendix J)",
     "the grounding gate below is the arbiter of correctness, not the language model"],
    NAVY, fs_t=13.5, fs_l=9.8)
down(CX, 70.9, 64.0, color=TEAL, label="tool calls (read-only)")

# 3 MCP tools (3 cols x 4 rows)
box(X, 36.0, W, 27.6, "Layer 2 · MCP epi server — 11 retrieval-grounded tools", [], TEAL, fs_t=13.0)
ax.text(CX, 58.6, "every tool reads epi_real_seoul.db read-only  (SQL guard + DuckDB READ_ONLY)",
        ha="center", va="center", fontsize=8.8, color="#d8f3ef", zorder=4)
tools = ["query_db", "forecast", "model_compare", "rt_estimate", "outbreak_detect",
         "lead_time_analysis", "validity_check", "literature_rag", "scenario_run",
         "international_compare", "shap_features"]
ncol = 3; cw, ch = 24.5, 3.1; gapx, gapy = 1.6, 1.2
gx = CX - (ncol * cw + (ncol - 1) * gapx) / 2
gy = 55.4
for i, t in enumerate(tools):
    r, c = divmod(i, ncol)
    xx = gx + c * (cw + gapx); yy = gy - r * (ch + gapy)
    ax.add_patch(FancyBboxPatch((xx, yy - ch), cw, ch, boxstyle="round,pad=0.15,rounding_size=0.6",
                                linewidth=1.0, edgecolor="#0b3d3a", facecolor=CHIP, zorder=4))
    ax.text(xx + cw / 2, yy - ch / 2, f"epi.{t}", ha="center", va="center",
            fontsize=7.9, color=INK, family="monospace", zorder=5)
down(CX, 35.9, 32.9, color=GREEN)

# 4 grounding (Layer 3a)
box(X, 23.5, W, 9.2, "Layer 3 · grounding & provenance",
    ["numeric grounding · semantic-consistency check",
     "provenance envelope per tool result  (data-vintage · config-hash)"],
    GREEN, fs_t=13, fs_l=9.4)
down(CX, 23.4, 20.4, color=PURPLE)

# 5 self-ask + cove (Layer 3b — verification)
box(X, 10.0, W, 9.2, "Layer 3 · Self-Ask (SubQ) + Chain-of-Verification",
    ["decompose numeric sub-questions · independent verify · reject ungrounded",
     "critic separate from generator  (judge ≠ generator)"],
    PURPLE, fs_t=13, fs_l=9.4)
down(CX, 9.9, 6.9, color=NAVY)

# 6 governance + HITL (Layer 4)
box(X + 3, 0.6, W - 6, 6.8, "Layer 4 · governance & audit  (Hermes hash-chain)",
    ["tamper-evident audit ledger · human sign-off → the epidemiologist decides"],
    OUT, fs_t=12, fs_l=9.0)

fig.savefig(OUTP, bbox_inches="tight", dpi=170, facecolor="white")
plt.close(fig)
print(f"wrote {OUTP}")
