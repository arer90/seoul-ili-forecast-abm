"""SCI-standard RELATIVE-WIS leaderboard for the thesis Results.

Field-standard comparison artifact (FluSight/forecast-hub convention): each
model's WIS is normalized by a reference baseline WIS on a CONSISTENT slab, so
relative-WIS_i = WIS_i / WIS_baseline. < 1.0 = skillful (beats the baseline).

This is a READ-ONLY reporting script. It performs NO retraining and NO model
evaluation; it only consumes two frozen SSOT result files:
  - simulation/results/wis_ssot.csv               (per-model WIS SSOT, oof/test slabs)
  - simulation/results/per_model_eval/per_model_metrics.csv  (R10 129-metric table,
    carries the headline pairwise relative-WIS, e.g. FusedEpi=0.4427)

Two relative-WIS DEFINITIONS coexist and are NOT interchangeable:
  (A) vs-baseline (this script): WIS_i / WIS_FluSight-Baseline on the oof slab.
      Simple ratio against ONE fixed reference forecaster.
  (B) pairwise-geometric (thesis headline, Sherratt et al. 2023 / FluSight):
      geometric mean of pairwise WIS ratios across ALL models, then normalized
      to the baseline. Carried in per_model_metrics.relative_wis_pairwise.
Both are reported side-by-side so the two columns are never conflated.

Slab choice: oof_wis (leak-free out-of-fold) is preferred because it is finite
for ~41 models (vs 26 for hold-out test_wis), giving the widest consistent
leaderboard. The reference baseline WIS is read from the SAME slab.

Usage:
    .venv/bin/python -m simulation.scripts.sci_relative_wis_leaderboard

Performance: O(n) over ~48 model rows; <1s, <50MB.
Side effects: writes CSV to simulation/results/sci_supplement/ and a PNG to
    simulation/results/figures/. No DB / no model state touched.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- SSOT paths (single source of truth; do not duplicate elsewhere) ---------
ROOT = Path(__file__).resolve().parents[2]
WIS_SSOT = ROOT / "simulation/results/wis_ssot.csv"
METRICS = ROOT / "simulation/results/per_model_eval/per_model_metrics.csv"
OUT_CSV = ROOT / "simulation/results/sci_supplement/sci_relative_wis_leaderboard.csv"
OUT_FIG = ROOT / "simulation/results/figures/sci_relative_wis_leaderboard.png"

SLAB_COL = "oof_wis"  # leak-free out-of-fold; finite for ~41 models
BASELINE = "FluSight-Baseline"  # field-standard reference forecaster

# The single champion (G-339 leak-free), plus the count model highlighted alongside it because it
# is epidemiologically interpretable — not because it shares the title. There is one champion.
CHAMPION = "FusedEpi"
INTERPRETABLE = "NegBinGLM"

# Published seasonal-influenza skill band for relative-WIS of skillful
# FluSight/forecast-hub models (Cramer 2022; Sherratt 2023): ~0.6-0.9.
EXTERNAL_BAND = "0.6-0.9"


def _to_float(s: str) -> float:
    s = (s or "").strip()
    if s == "" or s.lower() == "nan":
        return float("nan")
    try:
        return float(s)
    except ValueError:
        return float("nan")


def load_wis() -> dict[str, float]:
    """Read per-model WIS on the chosen slab from the WIS SSOT.

    Returns:
        {model_name: oof_wis} including only finite values.
    """
    out: dict[str, float] = {}
    with WIS_SSOT.open() as f:
        for row in csv.DictReader(f):
            v = _to_float(row[SLAB_COL])
            if v == v:  # not NaN
                out[row["model"]] = v
    return out


def load_pairwise() -> dict[str, float]:
    """Read the thesis headline pairwise relative-WIS from the R10 metrics table.

    Returns:
        {model_name: relative_wis_pairwise} (geometric pairwise definition).
    """
    out: dict[str, float] = {}
    with METRICS.open() as f:
        for row in csv.DictReader(f):
            out[row["model"]] = _to_float(row.get("relative_wis_pairwise", ""))
    return out


def build_leaderboard() -> list[dict]:
    """Compute relative-WIS_i = WIS_i / WIS_baseline on a consistent slab.

    Returns:
        Rows sorted ascending by relative_wis_vs_baseline (most skillful first).

    Raises:
        KeyError: if the baseline model has no finite WIS on the slab.
    """
    wis = load_wis()
    pair = load_pairwise()
    if BASELINE not in wis:
        raise KeyError(f"Baseline {BASELINE!r} has no finite {SLAB_COL}")
    base_wis = wis[BASELINE]

    rows = []
    for model, w in wis.items():
        rows.append(
            {
                "rank": 0,
                "model": model,
                "slab": SLAB_COL,
                "wis": round(w, 4),
                "baseline_wis": round(base_wis, 4),
                "relative_wis_vs_baseline": round(w / base_wis, 4),
                "relative_wis_pairwise": (
                    round(pair[model], 4) if pair.get(model, float("nan")) == pair.get(model, float("nan")) else ""
                ),
                "skillful": "yes" if w < base_wis else "no",
                "is_baseline": "yes" if model == BASELINE else "",
                "role": (
                    "champion"
                    if model == CHAMPION
                    else "interpretable"
                    if model == INTERPRETABLE
                    else ""
                ),
            }
        )
    rows.sort(key=lambda r: r["relative_wis_vs_baseline"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
    return rows


def write_csv(rows: list[dict]) -> None:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "model",
        "slab",
        "wis",
        "baseline_wis",
        "relative_wis_vs_baseline",
        "relative_wis_pairwise",
        "skillful",
        "is_baseline",
        "role",
    ]
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def make_figure(rows: list[dict]) -> None:
    """Ranked horizontal bar of relative-WIS with the rWIS=1 skill line."""
    OUT_FIG.parent.mkdir(parents=True, exist_ok=True)
    # Most skillful at top.
    rr = list(reversed(rows))
    labels = [r["model"] for r in rr]
    vals = [r["relative_wis_vs_baseline"] for r in rr]

    colors = []
    for r in rr:
        if r["model"] == CHAMPION:
            colors.append("#1b5e20")  # dark green champion
        elif r["model"] == INTERPRETABLE:
            colors.append("#388e3c")  # green: interpretable count model
        elif r["model"] == BASELINE:
            colors.append("#9e9e9e")  # grey baseline
        elif r["relative_wis_vs_baseline"] < 1.0:
            colors.append("#64b5f6")  # skillful blue
        else:
            colors.append("#ef9a9a")  # unskillful red

    fig, ax = plt.subplots(figsize=(8, max(8, 0.26 * len(rr))))
    ax.barh(range(len(rr)), vals, color=colors, edgecolor="white", linewidth=0.4)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1.2,
               label="rWIS = 1 (baseline skill)")
    ax.set_yticks(range(len(rr)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(f"Relative WIS  (WIS / {BASELINE} WIS,  {SLAB_COL} slab)")
    ax.set_title(
        f"SCI relative-WIS leaderboard (n={len(rr)} models, vs {BASELINE} baseline)\n"
        f"<1 = skillful;  champion {CHAMPION} & interpretable {INTERPRETABLE} highlighted"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.margins(y=0.005)
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    plt.close(fig)


def main() -> None:
    rows = build_leaderboard()
    write_csv(rows)
    make_figure(rows)
    base = next(r for r in rows if r["model"] == BASELINE)
    champ = next(r for r in rows if r["model"] == CHAMPION)
    print(f"baseline={BASELINE}  baseline_wis={base['baseline_wis']}  slab={SLAB_COL}")
    print(f"n_models={len(rows)}  champion={CHAMPION} rWIS={champ['relative_wis_vs_baseline']} "
          f"pairwise={champ['relative_wis_pairwise']}")
    print(f"CSV -> {OUT_CSV}")
    print(f"FIG -> {OUT_FIG}")
    print("\nTop 10 (rank | model | rWIS_vs_baseline | rWIS_pairwise):")
    for r in rows[:10]:
        print(f"  {r['rank']:2} | {r['model']:20} | {r['relative_wis_vs_baseline']:.4f} | "
              f"{r['relative_wis_pairwise']}")


if __name__ == "__main__":
    main()
