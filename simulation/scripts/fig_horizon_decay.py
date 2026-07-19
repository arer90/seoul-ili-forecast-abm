"""Operational multi-horizon forecast-skill decay figure (Seoul ILI champion).

Renders the expanding-window, multi-horizon operational forecasting result from
``run_expanding_multihorizon`` as a skill-decay curve: point skill (R^2, left
axis) and 95% prediction-interval coverage (right axis) versus forecast horizon,
with the operational-horizon cutoff and the three reporting tiers shaded.

The figure is the honest companion to the thesis "real-time forecasting" claim:
it shows where the champion is operationally trustworthy (1-4 weeks: R^2 > 0.3
AND coverage near nominal) and where it reverts to climatology (>= 8 weeks: R^2
< 0 and coverage collapses). Nothing is hard-coded -- every number is recomputed
from the per-origin (pred, actual, pi_lo, pi_hi) records in ``result.json``.

Run:
    .venv/bin/python -m simulation.scripts.fig_horizon_decay

Performance: O(origins * horizons), < 1s, < 50MB.
Side effects: writes one PNG to simulation/results/figures/.
Caller responsibility: run_expanding_multihorizon must have produced result.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive, deterministic
import matplotlib.pyplot as plt
import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_RESULT = _ROOT / "simulation" / "results" / "expanding_multihorizon" / "result.json"
FIG_DIR = _ROOT / "simulation" / "results" / "figures"

# Operational-horizon cutoff (weeks): point skill (R^2) AND interval coverage
# both hold through here; beyond it the forecast is climatology-equivalent.
_OPERATIONAL_CUTOFF_WEEKS = 4


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination; NaN when variance is degenerate.

    Args:
        y_true: realized values, shape (n,).
        y_pred: forecasts aligned to y_true, shape (n,).

    Returns:
        R^2 in (-inf, 1], or NaN if n < 2 or Var(y_true) == 0.
    """
    if y_true.size < 2:
        return float("nan")
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    return 1.0 - ss_res / ss_tot


def aggregate_by_horizon(result: dict) -> list[dict]:
    """Recompute per-horizon skill metrics from the raw per-origin records.

    Args:
        result: parsed result.json with an "origins" list, each holding a
            "horizons" list of {h, pred, actual, pi_lo, pi_hi} records.

    Returns:
        One dict per horizon (ascending h) with keys h, r2, mae, coverage95,
        pi_width, n -- every value derived, never read from a stored summary.
    """
    by_h: dict[int, dict[str, list]] = {}
    for origin in result.get("origins", []):
        for rec in origin.get("horizons", []):
            actual = rec.get("actual")
            if actual is None:
                continue
            h = int(rec["h"])
            slot = by_h.setdefault(h, {"pred": [], "act": [], "cov": [], "width": []})
            slot["pred"].append(float(rec["pred"]))
            slot["act"].append(float(actual))
            lo, hi = rec.get("pi_lo"), rec.get("pi_hi")
            if lo is not None and hi is not None:
                slot["cov"].append(1.0 if (lo <= actual <= hi) else 0.0)
                slot["width"].append(float(hi) - float(lo))

    rows: list[dict] = []
    for h in sorted(by_h):
        s = by_h[h]
        yt, yp = np.asarray(s["act"]), np.asarray(s["pred"])
        rows.append(
            {
                "h": h,
                "r2": _r2(yt, yp),
                "mae": float(np.mean(np.abs(yt - yp))),
                "coverage95": float(np.mean(s["cov"])) if s["cov"] else float("nan"),
                "pi_width": float(np.mean(s["width"])) if s["width"] else float("nan"),
                "n": int(yt.size),
            }
        )
    return rows


def render(rows: list[dict], out_path: Path) -> None:
    """Draw the dual-axis skill-decay figure and save it.

    Args:
        rows: output of aggregate_by_horizon (ascending h).
        out_path: PNG destination.

    Side effects: writes out_path; creates parent dir.
    """
    hs = [r["h"] for r in rows]
    r2 = [r["r2"] for r in rows]
    cov = [r["coverage95"] for r in rows]
    x = np.arange(len(hs))  # categorical spacing (1,2,3,4,8,12,28 unevenly spaced)

    fig, ax1 = plt.subplots(figsize=(9.5, 5.4))

    # Three reporting tiers (by horizon index, derived from the cutoff).
    op_idx = [i for i, h in enumerate(hs) if h <= _OPERATIONAL_CUTOFF_WEEKS]
    primary = [i for i in op_idx if hs[i] <= 2]
    useful = [i for i in op_idx if hs[i] > 2]
    beyond = [i for i, h in enumerate(hs) if h > _OPERATIONAL_CUTOFF_WEEKS]
    if primary:
        ax1.axvspan(-0.5, max(primary) + 0.5, color="#2ca02c", alpha=0.08, zorder=0)
    if useful:
        ax1.axvspan(min(useful) - 0.5, max(useful) + 0.5, color="#ff7f0e", alpha=0.08, zorder=0)
    if beyond:
        ax1.axvspan(min(beyond) - 0.5, len(hs) - 0.5, color="#d62728", alpha=0.07, zorder=0)

    ax1.axhline(0.0, color="#444444", lw=0.8, ls=":", zorder=1)
    (l1,) = ax1.plot(x, r2, "o-", color="#1f3b8c", lw=2.2, ms=7, label="point skill R²", zorder=3)
    ax1.set_ylabel("point skill  R²", color="#1f3b8c", fontsize=11)
    ax1.tick_params(axis="y", labelcolor="#1f3b8c")
    ax1.set_ylim(-0.8, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"+{h}w" for h in hs])
    ax1.set_xlabel("forecast horizon (weeks ahead)", fontsize=11)
    ax1.set_xlim(-0.5, len(hs) - 0.5)

    ax2 = ax1.twinx()
    (l2,) = ax2.plot(x, cov, "s--", color="#8c1f3b", lw=1.8, ms=6, label="95% PI coverage", zorder=3)
    ax2.axhline(0.95, color="#8c1f3b", lw=0.8, ls=":", alpha=0.6)
    ax2.set_ylabel("95% PI coverage", color="#8c1f3b", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#8c1f3b")
    ax2.set_ylim(0.5, 1.02)

    # Operational-horizon cutoff line (last horizon <= cutoff).
    if op_idx:
        cut_x = max(op_idx) + 0.5
        ax1.axvline(cut_x, color="#222222", lw=1.4, ls="--", zorder=2)
        ax1.text(cut_x - 0.05, 0.92, "operational limit\n(≤4w: R²>0.3 & cov≈nom)",
                 ha="right", va="top", fontsize=8.5, color="#222222")

    # R^2 value labels.
    for xi, v in zip(x, r2):
        ax1.annotate(f"{v:.2f}", (xi, v), textcoords="offset points", xytext=(0, 9),
                     ha="center", fontsize=8, color="#1f3b8c")

    ax1.set_title(
        "Operational multi-horizon forecast-skill decay — Seoul ILI\n"
        "champion FusedEpi, expanding-window rolling origins (n="
        f"{rows[0]['n']} at +1w), recursive multi-step",
        fontsize=11.5,
    )
    ax1.legend(handles=[l1, l2], loc="upper right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Load result.json, recompute metrics, render the figure, print a summary."""
    if not _RESULT.exists():
        raise SystemExit(f"missing {_RESULT} — run run_expanding_multihorizon first")
    result = json.loads(_RESULT.read_text(encoding="utf-8"))
    rows = aggregate_by_horizon(result)
    out = FIG_DIR / "horizon_decay.png"
    render(rows, out)
    print(f"[fig_horizon_decay] {len(rows)} horizons, {len(result.get('origins', []))} origins")
    for r in rows:
        print(f"  +{r['h']:>2}w  R2={r['r2']:+.3f}  MAE={r['mae']:6.2f}  "
              f"cov95={r['coverage95']:.3f}  PIw={r['pi_width']:6.1f}  n={r['n']}")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
