"""fig_uq_conformal.py — Methods figure: champion (FusedEpi) uncertainty
quantification via adaptive conformal prediction intervals.

Shows, for ONE contiguous forecast segment (the n=68 leak-free rolling 1-step
test slab of the deployed champion FusedEpi), the point forecast, the observed
ILI rate overlaid, and the 50% / 80% / 95% prediction-interval bands produced by
adaptive (online Conformal-PID) recalibration. A side panel makes the
recalibration effect explicit: raw split-conformal 95% PI under-covers
(empirical 0.735 at nominal 0.95) while adaptive recalibration restores
coverage to 0.897 — on the SAME test slab.

★ Real data only — no fabrication. Bands are reproduced with the exact SSOT
functions used by simulation/scripts/adaptive_pi_eval.py:
  - point forecast + observed:   simulation/results/csv/predictions_FusedEpi.csv
  - leak-free residual seed:      simulation/results/per_model_optimal/FusedEpi.json
                                  (val_metrics.insample_residuals)
  - coverage numbers (caption):   simulation/results/csv/adaptive_pi_metrics.csv
  - half-widths:  hub_metrics.k11_pi_widths_from_residuals (Lei 2018 split-conformal)
  - bands:        adaptive_conformal.adaptive_conformal_bounds (Angelopoulos 2024
                  Conformal-PID / Gibbs-Candes 2021 ACI)
leak-free: each step's interval uses only past observations (operational rolling
1-step), identical to the deployed setting.

Output (B5 width, 300 dpi):
    paper/methods_assets/fig_uq_conformal.png

Discipline: matplotlib Agg, English labels only (DejaVu Sans), deterministic,
sqlite=0 (reads CSV + JSON only), honest crash if inputs are absent (no
fabricated data).

Usage: .venv/bin/python paper/methods_assets/fig_uq_conformal.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless render
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parents[2]
# Make `simulation` importable (SSOT conformal functions live there).
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PRED_CSV = PROJECT_ROOT / "simulation" / "results" / "csv" / "predictions_FusedEpi.csv"
OPT_JSON = PROJECT_ROOT / "simulation" / "results" / "per_model_optimal" / "FusedEpi.json"
METRICS_CSV = PROJECT_ROOT / "simulation" / "results" / "csv" / "adaptive_pi_metrics.csv"
OUT_PNG = _THIS.parent / "fig_uq_conformal.png"

CHAMPION = "FusedEpi"
# B5 page text width ~ 5.0 in usable; full-width two-panel figure.
B5_WIDTH_IN = 6.7
FIG_HEIGHT_IN = 3.7

# Colors (color-blind-safe Okabe-Ito-ish blues/orange).
C_OBS = "#222222"
C_PRED = "#0072B2"
C_50 = "#0072B2"
C_80 = "#56B4E9"
C_95 = "#9ECAE9"
C_STATIC = "#B0B0B0"
C_ADAPT = "#0072B2"


def _setup_font() -> None:
    """Force English paper font (DejaVu Sans), ASCII minus."""
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def _load_forecast() -> tuple[np.ndarray, np.ndarray]:
    """Champion point forecast + observed over the leak-free test slab.

    Returns:
        (y_true, y_pred) each (n,) float arrays in chronological order.

    Raises:
        FileNotFoundError / ValueError if the prediction CSV is missing/empty.
    """
    if not PRED_CSV.exists():
        raise FileNotFoundError(f"missing real predictions: {PRED_CSV}")
    df = pd.read_csv(PRED_CSV)
    t = df[df["split"] == "test"].sort_values("idx")
    if len(t) < 10:
        raise ValueError(f"too few test rows in {PRED_CSV}: {len(t)}")
    return (t["y_true"].to_numpy(dtype=np.float64),
            t["y_pred"].to_numpy(dtype=np.float64))


def _load_residuals() -> np.ndarray:
    """Leak-free in-sample residual seed for conformal calibration.

    Returns:
        (m,) finite residual array (m >= 2).

    Raises:
        ValueError if residuals are absent (no fabricated calibration).
    """
    d = json.loads(OPT_JSON.read_text())
    r = (d.get("val_metrics", {}) or {}).get("insample_residuals")
    if r is None:
        raise ValueError(f"no insample_residuals in {OPT_JSON} (cannot calibrate)")
    a = np.asarray(r, dtype=np.float64)
    a = a[np.isfinite(a)]
    if len(a) < 2:
        raise ValueError(f"insufficient residuals in {OPT_JSON}: {len(a)}")
    return a


def _load_coverage_row() -> dict:
    """Champion's measured static vs adaptive coverage row (caption numbers)."""
    df = pd.read_csv(METRICS_CSV)
    row = df[df["model"] == CHAMPION]
    if row.empty:
        raise ValueError(f"{CHAMPION} not in {METRICS_CSV}")
    return row.iloc[0].to_dict()


def build() -> Path:
    """Render the UQ conformal figure from measured champion data.

    Returns:
        Path to the written PNG.

    Side effects: writes OUT_PNG (300 dpi). Reads 3 result files (read-only).
    """
    _setup_font()
    from simulation.analytics.hub_metrics import (
        FLUSIGHT_ALPHAS, k11_pi_widths_from_residuals,
    )
    from simulation.analytics.adaptive_conformal import adaptive_conformal_bounds

    y_true, y_pred = _load_forecast()
    res = _load_residuals()
    cov = _load_coverage_row()
    n = len(y_true)
    weeks = np.arange(1, n + 1)

    # SSOT bands — same call as adaptive_pi_eval.evaluate().
    k11 = k11_pi_widths_from_residuals(np.abs(res), FLUSIGHT_ALPHAS)
    bounds = adaptive_conformal_bounds(y_pred, k11, res, y_true, FLUSIGHT_ALPHAS)

    # FluSight α: PI(1-α). 95% -> α=0.05, 80% -> α=0.20, 50% -> α=0.50.
    a95, a80, a50 = 0.05, 0.20, 0.50
    lo95, hi95 = bounds[a95]
    lo80, hi80 = bounds[a80]
    lo50, hi50 = bounds[a50]

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(B5_WIDTH_IN, FIG_HEIGHT_IN),
        gridspec_kw={"width_ratios": [3.05, 1.0], "wspace": 0.40},
    )

    # ── Left: forecast + adaptive PI bands + observed overlay ──────────────
    axL.fill_between(weeks, lo95, hi95, color=C_95, alpha=0.55,
                     linewidth=0, label="95% PI (adaptive)")
    axL.fill_between(weeks, lo80, hi80, color=C_80, alpha=0.50,
                     linewidth=0, label="80% PI")
    axL.fill_between(weeks, lo50, hi50, color=C_50, alpha=0.32,
                     linewidth=0, label="50% PI")
    axL.plot(weeks, y_pred, color=C_PRED, lw=1.6, label="FusedEpi forecast")
    axL.plot(weeks, y_true, color=C_OBS, lw=0.0, marker="o", ms=3.0,
             mfc=C_OBS, mec="white", mew=0.4, label="Observed ILI")

    # Mark observations that fall outside the 95% band (miscoverage).
    out_mask = (y_true < lo95) | (y_true > hi95)
    if out_mask.any():
        axL.scatter(weeks[out_mask], y_true[out_mask], s=46, facecolors="none",
                    edgecolors="#D55E00", linewidths=1.3, zorder=6,
                    label="Outside 95% PI")

    axL.set_xlabel("Test week (rolling 1-step origin)", fontsize=9)
    axL.set_ylabel("ILI rate (per 1,000 outpatient visits)", fontsize=9)
    axL.set_title(
        f"(a) {CHAMPION} forecast with adaptive conformal intervals",
        fontsize=9.5, loc="left", pad=6,
    )
    axL.set_xlim(0.5, n + 0.5)
    axL.set_ylim(bottom=0)
    axL.tick_params(labelsize=8)
    axL.grid(True, alpha=0.18, lw=0.5)
    for s in ("top", "right"):
        axL.spines[s].set_visible(False)
    axL.legend(fontsize=6.7, loc="upper right", frameon=True, framealpha=0.9,
               handlelength=1.4, borderpad=0.4, labelspacing=0.32)

    # ── Right: recalibration effect on empirical coverage ──────────────────
    nominal = np.array([0.50, 0.80, 0.95])
    static = np.array([float(cov["static_pi50"]), float(cov["static_pi80"]),
                       float(cov["static_pi95"])])
    adapt = np.array([float(cov["adapt_pi50"]), float(cov["adapt_pi80"]),
                      float(cov["adapt_pi95"])])

    axR.plot([0, 1], [0, 1], color="#999999", ls=":", lw=1.0, zorder=1,
             label="Ideal")
    axR.plot(nominal, static, color=C_STATIC, marker="s", ms=5, lw=1.4,
             label="Raw split-conformal", zorder=3)
    axR.plot(nominal, adapt, color=C_ADAPT, marker="o", ms=5, lw=1.7,
             label="Adaptive (Conformal-PID)", zorder=4)
    for xn, ys, ya in zip(nominal, static, adapt):
        axR.annotate("", xy=(xn, ya), xytext=(xn, ys),
                     arrowprops=dict(arrowstyle="->", color="#D55E00",
                                     lw=1.0, alpha=0.8), zorder=2)

    axR.set_xlim(0.40, 1.02)
    axR.set_ylim(0.0, 1.02)
    axR.set_xticks(nominal)
    axR.set_xticklabels(["50%", "80%", "95%"], fontsize=8)
    axR.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axR.tick_params(labelsize=8)
    axR.set_xlabel("Nominal coverage", fontsize=9)
    axR.set_ylabel("Empirical coverage", fontsize=9)
    axR.set_title("(b) Calibration", fontsize=9.5, loc="left", pad=6)
    axR.grid(True, alpha=0.18, lw=0.5)
    for s in ("top", "right"):
        axR.spines[s].set_visible(False)
    axR.legend(fontsize=6.5, loc="lower right", frameon=True, framealpha=0.9,
               handlelength=1.4, borderpad=0.4, labelspacing=0.3)

    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.135)
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Console summary (also used to confirm caption numbers).
    emp95 = float(np.mean((y_true >= lo95) & (y_true <= hi95)))
    emp80 = float(np.mean((y_true >= lo80) & (y_true <= hi80)))
    emp50 = float(np.mean((y_true >= lo50) & (y_true <= hi50)))
    print(f"  wrote {OUT_PNG}")
    print(f"  n_test={n}  (rolling 1-step leak-free)")
    print(f"  adaptive empirical coverage on this slab: "
          f"50%={emp50:.3f}  80%={emp80:.3f}  95%={emp95:.3f}")
    print(f"  CSV (caption SSOT): static95={cov['static_pi95']} "
          f"-> adapt95={cov['adapt_pi95']}; "
          f"static80={cov['static_pi80']} -> adapt80={cov['adapt_pi80']}; "
          f"static50={cov['static_pi50']} -> adapt50={cov['adapt_pi50']}; "
          f"adapt_WIS={cov['adapt_wis']}")
    return OUT_PNG


if __name__ == "__main__":
    build()
