"""
phase11_loso_full.py — Full 7-season Leave-One-Season-Out CV (TRIPOD-AI 5j)
===========================================================================

Reproduces docs/METRIC_FAIRNESS_LOSO.md §2 numbers (cross-season external validity).

Distinct from `r3_6_loso_runner.py` (which holds out 2022-23 only for FE-cache experiments).
This script does FULL N-fold LOSO across 2019-2025 with persistence baseline,
producing era-stratified MAE/R²/F1 to detect COVID OOD scenarios.

Usage:
    .venv/bin/python -m simulation.scripts.phase11_loso_full \\
        [--db simulation/data/db/epi_real_seoul.db] \\
        [--out simulation/results/phase11_loso/]

Output:
    simulation/results/phase11_loso/
      ├── loso_per_season.json
      ├── era_stratified.json
      └── report.md

Added 2026-05-26 to address Round 3 audit G1 (Fairness/LOSO code missing).

Reference: TRIPOD-AI 2024 §5j external validity.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np

from simulation.database import safe_connect

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = ROOT / "simulation" / "data" / "db" / "epi_real_seoul.db"
DEFAULT_OUT = ROOT / "simulation" / "results" / "phase11_loso"

SEED = 42

# Era context (epidemiological framing for stratification)
SEASON_CONTEXT = {
    2019: ("normal", "pre-COVID baseline"),
    2020: ("covid", "COVID year-1 (lockdowns + school closures)"),
    2021: ("covid", "COVID year-2 (vaccines, mask mandates)"),
    2022: ("covid", "COVID year-3 (Omicron, partial reopening)"),
    2023: ("normal", "post-COVID rebound"),
    2024: ("normal", "normal season"),
    2025: ("normal", "current season (partial)"),
}


def run_loso_full(
    db_path: Path = DEFAULT_DB,
    out_dir: Path = DEFAULT_OUT,
    seasons: list = None,
) -> dict:
    """Run full N-fold LOSO across all available seasons.

    Args:
        db_path: KDCA DB
        out_dir: results dir
        seasons: list of int seasons; default 2019-2025

    Returns:
        dict with per_season + era_stratified summaries
    """
    if seasons is None:
        seasons = sorted(SEASON_CONTEXT.keys())

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # R4 audit fix: use default_rng (consistent with phase11)
    rng = np.random.default_rng(SEED)
    con = safe_connect(str(db_path))

    # Aggregate weekly ILI rate across all ages
    rows = con.execute(
        """
        SELECT season_start, week_seq, AVG(ili_rate) AS mean_rate
        FROM sentinel_influenza
        WHERE ili_rate IS NOT NULL AND ili_rate >= 0
        GROUP BY season_start, week_seq
        ORDER BY season_start, week_seq
        """
    ).fetchall()
    if not rows:
        raise RuntimeError("sentinel_influenza table empty")

    by_season = {}
    for s, w, r in rows:
        by_season.setdefault(s, []).append((w, r))
    for s in by_season:
        by_season[s].sort(key=lambda x: x[0])

    per_season = []
    for held_out in seasons:
        if held_out not in by_season:
            continue
        train_y = []
        for s in seasons:
            if s == held_out:
                continue
            train_y.extend([r for _, r in by_season[s]])
        test_y = np.array([r for _, r in by_season[held_out]], dtype=np.float64)
        if len(test_y) < 10:
            continue
        train_y = np.array(train_y, dtype=np.float64)

        # Persistence baseline within held-out season
        pred = np.roll(test_y, 1)
        pred[0] = train_y[-1]
        err = pred - test_y

        # σ from train-only
        train_pred = np.roll(train_y, 1)[1:]
        train_res = train_y[1:] - train_pred
        sigma = float(np.std(train_res))

        # Point metrics
        sse = float(np.sum(err ** 2))
        sst = float(np.sum((test_y - test_y.mean()) ** 2))
        r2 = 1.0 - sse / sst if sst > 0 else float("nan")
        mae = float(np.mean(np.abs(err)))

        # Threshold = 70%ile of train (consistent with fairness script)
        t_thr = float(np.percentile(train_y, 70))
        ev_true = (test_y > t_thr).astype(int)
        ev_pred = (pred > t_thr).astype(int)
        tp = int(((ev_true == 1) & (ev_pred == 1)).sum())
        fn = int(((ev_true == 1) & (ev_pred == 0)).sum())
        fp = int(((ev_true == 0) & (ev_pred == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else None

        era, context = SEASON_CONTEXT.get(held_out, ("unknown", ""))
        per_season.append({
            "season": held_out,
            "context": context,
            "era": era,
            "n_train": len(train_y),
            "n_test": len(test_y),
            "sigma": round(sigma, 4),
            "threshold": round(t_thr, 4),
            "mae": round(mae, 4),
            "r2": round(r2, 4) if np.isfinite(r2) else None,
            "f1": round(f1, 4) if f1 is not None else None,
        })

    # Era stratification
    normal = [r for r in per_season if r["era"] == "normal" and 2019 <= r["season"] <= 2024]
    covid = [r for r in per_season if r["era"] == "covid"]

    era = {}
    if normal:
        era["normal_mean_mae"] = round(float(np.mean([r["mae"] for r in normal])), 4)
        era["normal_seasons"] = [r["season"] for r in normal]
    if covid:
        era["covid_mean_mae"] = round(float(np.mean([r["mae"] for r in covid])), 4)
        era["covid_seasons"] = [r["season"] for r in covid]
    if normal and covid and era["normal_mean_mae"] > 0:
        era["covid_vs_normal_ratio"] = round(era["covid_mean_mae"] / era["normal_mean_mae"], 4)
        era["ood_detected"] = bool(
            era["covid_vs_normal_ratio"] < 0.5 or era["covid_vs_normal_ratio"] > 2.0
        )

    out_data = {
        "seed": SEED,
        "per_season": per_season,
        "era_stratified": era,
    }

    # Save
    (out_dir / "loso_per_season.json").write_text(
        json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md_lines = [
        "# Full 7-Season LOSO Report (TRIPOD-AI 5j)",
        "",
        f"> seed={SEED}",
        "",
        "## Per-season metrics",
        "",
        "| Held-out | Context | n_train | n_test | σ | MAE | R² | F1 |",
        "|----------|---------|---------|--------|------|------|------|------|",
    ]
    for r in per_season:
        md_lines.append(
            f"| {r['season']} | {r['context']} | {r['n_train']} | {r['n_test']} | "
            f"{r['sigma']} | {r['mae']} | {r['r2']} | {r['f1']} |"
        )
    md_lines += ["", "## Era-stratified summary", ""]
    if era:
        if "normal_mean_mae" in era:
            md_lines.append(
                f"- **Normal era** ({', '.join(map(str, era['normal_seasons']))}): "
                f"mean MAE = {era['normal_mean_mae']}"
            )
        if "covid_mean_mae" in era:
            md_lines.append(
                f"- **COVID era** ({', '.join(map(str, era['covid_seasons']))}): "
                f"mean MAE = {era['covid_mean_mae']}"
            )
        if "covid_vs_normal_ratio" in era:
            md_lines.append(f"- COVID/normal ratio = **{era['covid_vs_normal_ratio']}×**")
            md_lines.append(
                f"- OOD detected: **{'YES ✓' if era['ood_detected'] else 'no'}**"
            )
            md_lines.append(
                f"- **Caveat**: 2020 R² often negative (flat data, "
                f"variance ~0 from NPI suppression). R² alone misleading; pair w/ MAE + F1."
            )
    (out_dir / "report.md").write_text("\n".join(md_lines), encoding="utf-8")

    log.info(f"[loso_full] Saved to {out_dir}")
    return out_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seasons", nargs="*", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    result = run_loso_full(
        db_path=Path(args.db),
        out_dir=Path(args.out),
        seasons=args.seasons,
    )

    print("\n" + "=" * 70)
    print("Full LOSO Summary")
    print("=" * 70)
    for r in result["per_season"]:
        print(f"  {r['season']}  [{r['era']:<6}]  "
              f"MAE={r['mae']:<7.2f}  R²={r['r2']}  F1={r['f1']}")
    era = result["era_stratified"]
    if era:
        print(f"\nNormal era MAE = {era.get('normal_mean_mae')}")
        print(f"COVID era MAE  = {era.get('covid_mean_mae')}")
        if "covid_vs_normal_ratio" in era:
            print(f"OOD ratio      = {era['covid_vs_normal_ratio']}× "
                  f"({'detected' if era['ood_detected'] else 'no OOD'})")


if __name__ == "__main__":
    sys.exit(main() or 0)
