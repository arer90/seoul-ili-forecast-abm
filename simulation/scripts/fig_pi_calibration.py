"""fig_pi_calibration.py — champion PI calibration: raw split-conformal vs adaptive.

Replaces the misleading "all models near-perfect" reading of the old R12
``calibration_curve.png``. The honest paper narrative is: the champion FusedEpi
95% prediction interval UNDER-COVERS under the raw leak-free split-conformal band
(empirical coverage ~0.735 at nominal 0.95), and adaptive (online Conformal-PID)
recalibration restores coverage to ~0.90. This figure draws BOTH curves for the
champion so the figure matches the text exactly.

Data source (read-only, measured — no retraining, no DB writes):
    simulation/results/csv/adaptive_pi_metrics.csv
      columns: model, static_pi95, adapt_pi95, static_pi80, adapt_pi80,
               static_pi50, adapt_pi50, adapt_wis, n_test
    (static_* = raw split-conformal coverage on the n=68 leak-free test slab;
     adapt_*  = adaptive online-conformal coverage on the same slab.)

Output:
    simulation/results/figures/pi_calibration_champion.png

Discipline: matplotlib Agg, English labels only (DejaVu Sans), deterministic,
sqlite=0 (reads a CSV only), honest skip if the CSV is absent (no fabricated data).
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless render (no display dependency)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parents[2]
METRICS_CSV = PROJECT_ROOT / "simulation" / "results" / "csv" / "adaptive_pi_metrics.csv"
FIG_DIR = PROJECT_ROOT / "simulation" / "results" / "figures"
OUT_PNG = FIG_DIR / "pi_calibration_champion.png"

# Champion preference (FusedEpi is the deployed champion; NegBinGLM = interpretable count model).
CHAMPION = "FusedEpi"
NOMINAL = np.array([0.50, 0.80, 0.95])


def _setup_font() -> str:
    """Force the default English font for paper figures.

    Returns:
        The applied font family name ("DejaVu Sans").

    Side effects: sets plt.rcParams["font.family"], ["axes.unicode_minus"].
    """
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans"


def _load_metrics() -> list[dict[str, str]]:
    """Load adaptive_pi_metrics.csv as a list of row dicts (read-only).

    Returns:
        List of CSV row dicts. Empty list if the file is absent.
    """
    if not METRICS_CSV.exists():
        return []
    with METRICS_CSV.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _row_for(rows: list[dict], model: str) -> dict | None:
    """Return the metrics row for ``model`` (exact match), else None."""
    for r in rows:
        if r.get("model") == model:
            return r
    return None


def _coverages(row: dict, prefix: str) -> np.ndarray:
    """Extract the [pi50, pi80, pi95] coverage triple for a row given a prefix.

    Args:
        row: a CSV row dict.
        prefix: "static" or "adapt".

    Returns:
        (3,) float array of empirical coverages (NaN where missing).
    """
    out = []
    for lvl in ("50", "80", "95"):
        v = row.get(f"{prefix}_pi{lvl}")
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.asarray(out, dtype=float)


def build_figure() -> Path | None:
    """Draw the champion raw-vs-adaptive PI calibration curve. Honest skip if no data.

    Returns:
        The output PNG path, or None if the metrics CSV is absent / champion missing.

    Side effects: writes OUT_PNG; reads METRICS_CSV (read-only). No DB, no retraining.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _setup_font()
    rows = _load_metrics()
    if not rows:
        print(f"[skip] {METRICS_CSV} absent — no fabricated calibration drawn.")
        return None

    champ_row = _row_for(rows, CHAMPION)
    if champ_row is None:
        print(f"[skip] champion '{CHAMPION}' not in {METRICS_CSV.name}.")
        return None

    raw = _coverages(champ_row, "static")
    adapt = _coverages(champ_row, "adapt")
    n_test = champ_row.get("n_test", "?")

    fig, ax = plt.subplots(figsize=(7.2, 6.4))

    # Perfect-calibration diagonal.
    ax.plot([0, 1], [0, 1], "k--", lw=1.1, alpha=0.6, label="Perfect calibration")

    # Faint context: every other model's raw split-conformal curve (gray).
    other_label_used = False
    for r in rows:
        if r.get("model") == CHAMPION:
            continue
        emp = _coverages(r, "static")
        if np.any(np.isfinite(emp)):
            ax.plot(
                NOMINAL, emp, "-", color="#bdbdbd", lw=0.8, alpha=0.55, zorder=1,
                label=("Other models (raw split-conformal)"
                       if not other_label_used else None),
            )
            other_label_used = True

    # Champion: raw split-conformal (under-covers).
    ax.plot(
        NOMINAL, raw, "o-", color="#d7301f", lw=2.2, markersize=7, zorder=4,
        label=f"{CHAMPION} raw split-conformal (under-covers)",
    )
    # Champion: adaptive (online conformal) recalibrated (restored).
    ax.plot(
        NOMINAL, adapt, "s-", color="#1a9850", lw=2.2, markersize=7, zorder=5,
        label=f"{CHAMPION} adaptive conformal (recalibrated)",
    )

    # Annotate the headline 95% numbers (raw 0.735 -> adaptive ~0.90).
    ax.annotate(
        f"raw 95% PI coverage = {raw[2]:.3f}\n(under-coverage)",
        xy=(0.95, raw[2]), xytext=(0.55, raw[2] - 0.16),
        fontsize=9.5, color="#a50f15", ha="left", va="top",
        arrowprops=dict(arrowstyle="->", color="#a50f15", lw=1.0),
    )
    ax.annotate(
        f"adaptive 95% PI coverage = {adapt[2]:.3f}\n(restored to ~nominal)",
        xy=(0.95, adapt[2]), xytext=(0.36, 0.62),
        fontsize=9.5, color="#1a6b34", ha="left", va="center",
        arrowprops=dict(arrowstyle="->", color="#1a6b34", lw=1.0),
    )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xticks([0.0, 0.5, 0.8, 0.95, 1.0])
    ax.set_xlabel("Nominal coverage", fontsize=11)
    ax.set_ylabel("Empirical coverage (n=%s test weeks)" % n_test, fontsize=11)
    ax.set_title(
        f"Champion ({CHAMPION}) PI calibration: raw vs adaptive conformal",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.92)

    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {OUT_PNG}")
    print(
        f"[champion {CHAMPION}] raw 95%={raw[2]:.3f} adapt 95%={adapt[2]:.3f} | "
        f"raw 80%={raw[1]:.3f} adapt 80%={adapt[1]:.3f} | "
        f"raw 50%={raw[0]:.3f} adapt 50%={adapt[0]:.3f}"
    )
    return OUT_PNG


def main() -> int:
    """Entry point: generate the champion PI calibration figure. 0 on success."""
    out = build_figure()
    return 0 if out is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
