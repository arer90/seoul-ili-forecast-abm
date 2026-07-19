"""Figure E (§4.x) — ABM 26-origin behaviour-gap forest, peak-adjacent coded.

Per-origin forward-window R2 gap = (behaviour-ON R2) - (behaviour-OFF R2) for the
champion-anchored behavioural ABM, across 26 leak-free weekly forecast origins of
the 2025-26 Seoul ILI season (each in-sample <= cutoff, forward = real post-cutoff
truth, horizon min(16, available)). Strip/forest by origin, ordered by cutoff date,
with markers colour-coded peak-adjacent vs off-peak (peak-adjacent = forward-window
max ILI >= 50% of the season peak = the rise/peak phase; off-peak = decay/flat tail).
A moving-block bootstrap (block=4 weeks) 95% CI band for the MEAN gap is shaded, and
the mean / median / positive-count are annotated. This is the figure companion to the
ABM robustness analysis and visualises that the behaviour gap is modest and its CI
includes 0 overall (under-powered, n_eff~2.6), while at the epidemic peak / on the
non-overlapping subset the gap is consistently positive (DIRECTIONAL, not a
confirmatory significance claim).

REAL DATA ONLY. Sources (read-only, no retraining, no model load, no fabrication):
    simulation/results/abm_multiorigin_forward/result.json            (26 per-origin gaps)
    simulation/results/abm_multiorigin_forward/behavior_robustness.json
        (persisted moving-block bootstrap CI [seed=43, block=4, n_boot=10000]
         + peak-adjacent flags from the season-peak 50% threshold)

The two underlying DB-backed inputs (per-origin ABM arrays) were produced upstream
by `simulation.cli` run_abm_multiorigin_forward, which itself reads the epi DB via
`from simulation.database import read_only_connect`. This figure consumes only the
persisted REAL JSON outputs; it performs NO raw sqlite3.connect.

Style matches thesis fig_*.py: matplotlib Agg, dpi=150, seed=42, derived-from-source.

Run:
    .venv/bin/python paper/ch4_new_assets/fig_E_abm_behaviour_gap.py

Side effects: writes one PNG next to this script.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive, deterministic
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)  # determinism per project policy

_ROOT = Path(__file__).resolve().parents[2]
_ABM_DIR = _ROOT / "simulation" / "results" / "abm_multiorigin_forward"
_RESULT = _ABM_DIR / "result.json"
_ROBUST = _ABM_DIR / "behavior_robustness.json"
_OUT = Path(__file__).resolve().parent / "fig_E_abm_behaviour_gap.png"

_C_PEAK = "#0f766e"      # peak-adjacent (rise/peak) — teal
_C_OFF = "#b45309"       # off-peak (decay/flat tail) — amber
_C_CI = "#1e3a5f"        # CI band / mean line — navy
_C_ZERO = "#888888"


def _load() -> dict:
    """Load the persisted ABM multi-origin REAL outputs.

    Returns:
        dict with keys: cutoffs (list[str]), gaps (np.ndarray[26]),
        peak_flags (np.ndarray[bool, 26]), boot_lo, boot_hi (mean-gap CI),
        plus the persisted summary stats (mean/median/n_positive/sign_p).

    Raises:
        SystemExit: if a source file is missing (do NOT invent values).
    """
    for p in (_RESULT, _ROBUST):
        if not p.exists():
            raise SystemExit(
                f"MISSING DATA FILE: {p}\n"
                "Run `.venv/bin/python -m simulation run-abm-multiorigin-forward` first."
            )
    res = json.loads(_RESULT.read_text(encoding="utf-8"))
    rob = json.loads(_ROBUST.read_text(encoding="utf-8"))

    per = res["per_origin"]
    cutoffs = [o["cutoff"] for o in per]
    gaps = np.array([o["behavior_gap"] for o in per], dtype=float)

    pa = rob["peak_adjacent"]
    flags = np.array(pa["peak_adjacent_flags"], dtype=bool)
    if len(flags) != len(gaps):
        raise SystemExit(
            f"FLAG/GAP length mismatch: {len(flags)} flags vs {len(gaps)} gaps"
        )

    mbb = rob["moving_block_bootstrap"]
    boot_lo, boot_hi = mbb["mean_gap_ci95"]

    dist = res["distribution"]["behavior_gap_all_origins"]
    sign = res["distribution"]["behavior_gap_sign_test_all"]

    return {
        "cutoffs": cutoffs,
        "gaps": gaps,
        "peak_flags": flags,
        "boot_lo": float(boot_lo),
        "boot_hi": float(boot_hi),
        "mean": float(dist["mean"]),
        "median": float(dist["median"]),
        "n": int(dist["n"]),
        "n_positive": int(sign["n_positive"]),
        "sign_p": float(sign["p_value"]),
        "boot_block": int(mbb["block"]),
        "boot_n": int(mbb["n_boot"]),
        "boot_seed": int(mbb["seed"]),
        "pa_mean": float(pa["peak_adjacent"]["mean_gap"]),
        "pa_n": int(pa["peak_adjacent"]["n"]),
        "pa_pos": int(pa["peak_adjacent"]["n_positive"]),
        "off_mean": float(pa["off_peak"]["mean_gap"]),
        "off_n": int(pa["off_peak"]["n"]),
        "off_pos": int(pa["off_peak"]["n_positive"]),
    }


def render(d: dict, out_path: Path) -> None:
    """Draw the per-origin behaviour-gap forest + bootstrap CI band and save.

    Args:
        d: output of _load.
        out_path: PNG destination.

    Side effects: writes out_path.
    """
    gaps = d["gaps"]
    flags = d["peak_flags"]
    cutoffs = d["cutoffs"]
    n = len(gaps)
    y = np.arange(n)[::-1]  # origin 0 (earliest cutoff) at top

    fig, ax = plt.subplots(figsize=(9.5, 9.0))

    # Bootstrap 95% CI band for the MEAN gap (vertical span across all origins).
    ax.axvspan(d["boot_lo"], d["boot_hi"], color=_C_CI, alpha=0.10, zorder=0,
               label=f"mean-gap 95% CI (moving-block, block={d['boot_block']}w)")
    ax.axvline(d["mean"], color=_C_CI, lw=1.8, ls="-", zorder=2,
               label=f"mean gap = {d['mean']:+.3f}")
    ax.axvline(d["median"], color=_C_CI, lw=1.2, ls="--", zorder=2,
               label=f"median gap = {d['median']:+.3f}")
    ax.axvline(0.0, color=_C_ZERO, lw=1.4, ls=":", zorder=1)

    # Per-origin stems + markers, colour-coded by regime.
    for yi, g, pk in zip(y, gaps, flags):
        c = _C_PEAK if pk else _C_OFF
        ax.plot([0, g], [yi, yi], color=c, lw=1.2, alpha=0.55, zorder=3)
        ax.scatter(g, yi, s=46, color=c, edgecolor="white", linewidth=0.7,
                   zorder=4)

    # Legend proxies for the two regimes.
    ax.scatter([], [], s=46, color=_C_PEAK, edgecolor="white", linewidth=0.7,
               label=f"peak-adjacent (rise/peak)  n={d['pa_n']}, "
                     f"{d['pa_pos']}/{d['pa_n']} positive")
    ax.scatter([], [], s=46, color=_C_OFF, edgecolor="white", linewidth=0.7,
               label=f"off-peak (decay/tail)  n={d['off_n']}, "
                     f"{d['off_pos']}/{d['off_n']} positive")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{i:02d}  {c}" for i, c in enumerate(cutoffs)],
                       fontsize=8)
    ax.set_ylim(-0.8, n - 0.2)
    ax.set_xlabel("behaviour-gap = forward-R2(behaviour ON) - "
                  "forward-R2(behaviour OFF)", fontsize=10.5)
    ax.set_ylabel("forecast origin (cutoff week, earliest -> latest)",
                  fontsize=10.5)
    ax.set_title(
        "Behavioural ABM 26-origin forward behaviour gap "
        "(2025-26 Seoul season, leak-free)\n"
        "positive => behaviour mechanism improves the forward fit",
        fontsize=11.5,
    )

    # Annotation box: pooled summary.
    txt = (
        f"n = {d['n']} origins   positive = {d['n_positive']}/{d['n']} "
        f"(sign test p = {d['sign_p']:.3f})\n"
        f"mean = {d['mean']:+.3f}   median = {d['median']:+.3f}\n"
        f"95% CI = [{d['boot_lo']:+.3f}, {d['boot_hi']:+.3f}]  (CI includes 0)\n"
        f"peak-adjacent mean = {d['pa_mean']:+.3f}  vs  "
        f"off-peak mean = {d['off_mean']:+.3f}"
    )
    ax.text(0.015, 0.015, txt, transform=ax.transAxes, va="bottom", ha="left",
            fontsize=9.0,
            bbox=dict(boxstyle="round", fc="#f0fdfa", ec=_C_CI, alpha=0.95))

    ax.legend(loc="upper right", fontsize=8.4, framealpha=0.94)
    ax.grid(axis="x", color="#dddddd", lw=0.6, zorder=0)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    d = _load()
    render(d, _OUT)
    print(f"[fig_E] n={d['n']} origins  mean={d['mean']:+.4f} "
          f"median={d['median']:+.4f}  positive={d['n_positive']}/{d['n']}")
    print(f"  mean-gap 95% CI = [{d['boot_lo']:+.4f}, {d['boot_hi']:+.4f}] "
          f"(block={d['boot_block']}w, n_boot={d['boot_n']}, seed={d['boot_seed']})")
    print(f"  peak-adjacent mean={d['pa_mean']:+.4f} ({d['pa_pos']}/{d['pa_n']})  "
          f"off-peak mean={d['off_mean']:+.4f} ({d['off_pos']}/{d['off_n']})")
    print(f"  -> {_OUT}")


if __name__ == "__main__":
    main()
