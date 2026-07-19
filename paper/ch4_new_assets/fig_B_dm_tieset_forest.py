"""Figure B (§4.2) — DM tie-set forest plot: FusedEpi vs 6 competitors.

Horizontal CI-whisker (forest) plot of the moving-block-bootstrap per-week
loss-differential ΔL = AE_FusedEpi − AE_competitor (negative => FusedEpi lower
loss = better) with 95% bootstrap CIs. Tie-set members (CI includes zero) are
visually distinct from decisive pairs (CI excludes zero). Zero reference line
marks the no-difference boundary. This is the figure companion to docx Table 6.

REAL DATA ONLY. Source = the persisted bootstrap output (already computed
leak-free from the DB-backed per-week test predictions by
``simulation.scripts.sci_delta_wis_bootstrap``):
    simulation/results/sci_supplement/delta_wis_bootstrap.json
No retraining, no model load, no fabrication. seed=42 baked into the source run
(reps=5000, block=4 weeks, n_test_weeks=68).

Style matches thesis fig_*.py: matplotlib Agg, dpi=150, derived-from-source.

Run:
    .venv/bin/python paper/ch4_new_assets/fig_B_dm_tieset_forest.py

Side effects: writes one PNG next to this script.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive, deterministic
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)  # determinism per project policy (no stochastic ops here)

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "simulation" / "results" / "sci_supplement" / "delta_wis_bootstrap.json"
_OUT = Path(__file__).resolve().parent / "fig_B_dm_tieset_forest.png"

# Thesis palette (consistent with fig_horizon_decay.py).
_C_TIE = "#1f3b8c"       # tie-set (CI includes zero) — navy
_C_DECISIVE = "#8c1f3b"  # decisive (CI excludes zero) — maroon
_C_ZERO = "#222222"


def _load(src: Path) -> dict:
    """Load the persisted bootstrap result.

    Args:
        src: path to delta_wis_bootstrap.json.

    Returns:
        Parsed dict with keys 'pairs', 'reps', 'block_weeks', 'n_test_weeks',
        'seed', 'champion'.

    Raises:
        SystemExit: if the source file is missing (do NOT invent values).
    """
    if not src.exists():
        raise SystemExit(
            f"MISSING DATA FILE: {src}\n"
            "Run `.venv/bin/python -m simulation.scripts.sci_delta_wis_bootstrap` first."
        )
    return json.loads(src.read_text(encoding="utf-8"))


def render(data: dict, out_path: Path) -> list[dict]:
    """Draw the forest plot and save it.

    Args:
        data: parsed bootstrap result (see _load).
        out_path: PNG destination.

    Returns:
        The list of pair dicts in plotted order (for SSOT cross-check).

    Side effects: writes out_path.
    """
    pairs = list(data["pairs"])
    # Sort by mean ΔL ascending (most-improved/most-negative at top after invert).
    pairs.sort(key=lambda p: p["mean_delta_ae"])

    labels = [p["competitor"] for p in pairs]
    means = np.array([p["mean_delta_ae"] for p in pairs])
    lo = np.array([p["ci95_lo"] for p in pairs])
    hi = np.array([p["ci95_hi"] for p in pairs])
    is_tie = [p["verdict"] == "tie" for p in pairs]

    y = np.arange(len(pairs))[::-1]  # top row = first (most negative)

    fig, ax = plt.subplots(figsize=(9.5, 5.4))

    ax.axvline(0.0, color=_C_ZERO, lw=1.2, ls="--", zorder=1,
               label="no difference (ΔL = 0)")

    for yi, m, l, h, tie in zip(y, means, lo, hi, is_tie):
        col = _C_TIE if tie else _C_DECISIVE
        ax.errorbar(
            m, yi,
            xerr=[[m - l], [h - m]],
            fmt="o", color=col, ecolor=col, elinewidth=1.8,
            capsize=4, ms=8, zorder=3,
        )
        # numeric label: mean [lo, hi]
        ax.annotate(
            f"{m:+.2f}  [{l:+.2f}, {h:+.2f}]",
            (h, yi), textcoords="offset points", xytext=(8, 0),
            va="center", ha="left", fontsize=8, color=col,
        )

    ax.set_yticks(y)
    ax.set_yticklabels([f"vs {lab}" for lab in labels], fontsize=10)
    ax.set_ylim(-0.6, len(pairs) - 0.4)
    ax.set_xlabel(
        "per-week loss differential  ΔL = AE(FusedEpi) − AE(competitor)\n"
        "(negative ⇒ FusedEpi lower loss = better)",
        fontsize=10.5,
    )

    # Legend proxies for tie vs decisive.
    tie_proxy = plt.Line2D([0], [0], marker="o", color=_C_TIE, lw=0, ms=8,
                           label="tie-set (95% CI includes 0)")
    dec_proxy = plt.Line2D([0], [0], marker="o", color=_C_DECISIVE, lw=0, ms=8,
                           label="decisive (95% CI excludes 0)")
    zero_proxy = plt.Line2D([0], [0], color=_C_ZERO, lw=1.2, ls="--",
                            label="no difference (ΔL = 0)")
    ax.legend(handles=[tie_proxy, dec_proxy, zero_proxy], loc="lower right",
              fontsize=8.5, framealpha=0.92)

    ax.set_title(
        "Statistical-equivalence forest plot — champion FusedEpi vs competitors\n"
        f"moving-block bootstrap (n={data['n_test_weeks']} test weeks, "
        f"block={data['block_weeks']}w, reps={data['reps']}, seed={data['seed']})",
        fontsize=11.5,
    )
    # Pad x so right-side labels fit.
    xmin = float(min(lo.min(), 0.0))
    xmax = float(max(hi.max(), 0.0))
    span = xmax - xmin
    ax.set_xlim(xmin - 0.10 * span, xmax + 0.55 * span)
    ax.grid(axis="x", color="#dddddd", lw=0.6, zorder=0)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return pairs


def main() -> None:
    data = _load(_SRC)
    pairs = render(data, _OUT)
    print(f"[fig_B] champion={data['champion']} reps={data['reps']} "
          f"block={data['block_weeks']}w n_weeks={data['n_test_weeks']} seed={data['seed']}")
    for p in pairs:
        print(f"  vs {p['competitor']:<18} dL={p['mean_delta_ae']:+.4f} "
              f"[{p['ci95_lo']:+.4f}, {p['ci95_hi']:+.4f}]  "
              f"p={p['boot_p_two_sided']:.4f}  {p['verdict']}")
    print(f"  -> {_OUT}")


if __name__ == "__main__":
    main()
