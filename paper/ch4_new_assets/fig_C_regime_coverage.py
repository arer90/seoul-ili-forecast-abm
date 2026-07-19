"""Figure C (§4.2) — regime-stratified 95% PI coverage: raw vs adaptive.

Grouped bar chart of the champion's empirical 95% prediction-interval coverage,
stratified by regime (incidence tertile: low/medium/high; wave phase:
pre-peak/peak/tail) for raw split-conformal vs adaptive (Conformal-PID) PIs.
A nominal-0.95 reference line marks the calibration target. This is the figure
companion to docx Table 7 and shows that raw under-coverage is concentrated in
the high-incidence / peak regime, where adaptive conformal restores coverage.

REAL DATA ONLY. Source = the persisted regime-calibration table (already
computed leak-free from the DB-backed per-week test predictions by
``simulation.scripts.sci_regime_calibration``):
    simulation/results/csv/regime_calibration_FusedEpi.csv
No retraining, no model load, no fabrication.

Style matches thesis fig_*.py: matplotlib Agg, dpi=150, derived-from-source.

Run:
    .venv/bin/python paper/ch4_new_assets/fig_C_regime_coverage.py

Side effects: writes one PNG next to this script.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive, deterministic
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)  # determinism per project policy (no stochastic ops here)

_ROOT = Path(__file__).resolve().parents[2]
_CHAMPION = "FusedEpi"
_SRC = _ROOT / "simulation" / "results" / "csv" / f"regime_calibration_{_CHAMPION}.csv"
_OUT = Path(__file__).resolve().parent / "fig_C_regime_coverage.png"

# Display order + human labels for the plotted regimes (skip 'overall' summary row,
# drawn separately as a final reference group).
_ORDER = [
    ("incidence", "low", "Incidence\nlow"),
    ("incidence", "medium", "Incidence\nmedium"),
    ("incidence", "high", "Incidence\nhigh"),
    ("phase", "pre-peak", "Phase\npre-peak"),
    ("phase", "peak", "Phase\npeak"),
    ("phase", "tail", "Phase\ntail"),
    ("overall", "all", "Overall"),
]

_C_RAW = "#8c1f3b"       # raw split-conformal — maroon
_C_ADAPT = "#1f3b8c"     # adaptive conformal — navy
_C_NOMINAL = "#2ca02c"


def _load(src: Path) -> dict[tuple[str, str], dict]:
    """Load the regime-calibration table keyed by (regime_type, regime).

    Args:
        src: path to regime_calibration_<champion>.csv.

    Returns:
        Map (regime_type, regime) -> {n, cov95_static, cov95_adaptive}.

    Raises:
        SystemExit: if the source file is missing (do NOT invent values).
    """
    if not src.exists():
        raise SystemExit(
            f"MISSING DATA FILE: {src}\n"
            "Run `.venv/bin/python -m simulation.scripts.sci_regime_calibration` first."
        )
    out: dict[tuple[str, str], dict] = {}
    with src.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[(row["regime_type"], row["regime"])] = {
                "n": int(row["n"]),
                "static": float(row["cov95_static"]),
                "adaptive": float(row["cov95_adaptive"]),
            }
    return out


def render(table: dict[tuple[str, str], dict], out_path: Path) -> list[dict]:
    """Draw the grouped raw-vs-adaptive coverage bars and save.

    Args:
        table: output of _load.
        out_path: PNG destination.

    Returns:
        Plotted rows (for SSOT cross-check).

    Side effects: writes out_path.
    """
    rows = []
    for rtype, rkey, label in _ORDER:
        rec = table.get((rtype, rkey))
        if rec is None:
            continue
        rows.append({"label": label, "key": f"{rtype}:{rkey}", **rec})

    labels = [r["label"] for r in rows]
    raw = np.array([r["static"] for r in rows])
    adapt = np.array([r["adaptive"] for r in rows])
    ns = [r["n"] for r in rows]

    x = np.arange(len(rows))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10.5, 5.6))

    b1 = ax.bar(x - w / 2, raw, w, color=_C_RAW, label="raw split-conformal", zorder=3)
    b2 = ax.bar(x + w / 2, adapt, w, color=_C_ADAPT, label="adaptive (Conformal-PID)", zorder=3)

    ax.axhline(0.95, color=_C_NOMINAL, lw=1.6, ls="--", zorder=2,
               label="nominal 0.95")

    # Value labels.
    for rect, v in list(zip(b1, raw)) + list(zip(b2, adapt)):
        ax.annotate(f"{v:.2f}", (rect.get_x() + rect.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 3),
                    ha="center", va="bottom", fontsize=7.8,
                    color="#333333")

    # n per regime, inside the plot just above the axis (avoids colliding with
    # the two-line x-tick labels below the axis).
    for xi, n in zip(x, ns):
        ax.annotate(f"n={n}", (xi, 0.012), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=7.5, color="#888888")

    # Separator before the 'Overall' reference group.
    if any(r["key"] == "overall:all" for r in rows):
        sep = next(i for i, r in enumerate(rows) if r["key"] == "overall:all")
        ax.axvline(sep - 0.5, color="#bbbbbb", lw=1.0, ls=":", zorder=1)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("empirical 95% PI coverage", fontsize=11)
    ax.set_ylim(0.0, 1.08)
    ax.set_title(
        f"Regime-stratified 95% PI coverage — champion {_CHAMPION}\n"
        "raw split-conformal vs adaptive conformal "
        "(under-coverage concentrated in high-incidence / peak regime)",
        fontsize=11.5,
    )
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3,
              fontsize=9.5, framealpha=0.95, frameon=True)
    ax.grid(axis="y", color="#dddddd", lw=0.6, zorder=0)

    fig.subplots_adjust(bottom=0.16)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return rows


def main() -> None:
    table = _load(_SRC)
    rows = render(table, _OUT)
    print(f"[fig_C] champion={_CHAMPION}")
    for r in rows:
        print(f"  {r['key']:<18} n={r['n']:>3}  "
              f"raw={r['static']:.3f}  adaptive={r['adaptive']:.3f}")
    print(f"  -> {_OUT}")


if __name__ == "__main__":
    main()
