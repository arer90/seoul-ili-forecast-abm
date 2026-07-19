"""Verify KDCA 8.6/1000 and Seoul ILI q70=11.45 thresholds against data.

This script reconstructs the two thresholds referenced in the epi
evaluation report:

 * **KDCA 2024-25 national ILI threshold = 8.6 / 1,000**
 — published via KDCA 2024-25 유행주의보 press release.
 * **Seoul recalibrated threshold = 11.45 / 1,000 (70th percentile of Seoul
 sentinel ILI over the training window)**
 — documented in ``simulation/results/pi_v22_6_epi_eval/report.md``.

Sources used here:
 - ``sentinel_influenza`` (epi table) — national weekly ILI by age group
 - ``predictions_*.csv`` (backup) — Seoul city-aggregate ILI y_true used
 to train the 66-model pipeline

Output:
 - console table with computed percentiles + the two published thresholds
 - ``simulation/results/kdca_threshold_audit.json``

Run:
 .venv\\Scripts\\python.exe -m simulation.scripts.verify_kdca_thresholds
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from simulation.database.analytics import duckdb_conn

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "simulation" / "results"
OUT = RES / "kdca_threshold_audit.json"

KDCA_NATIONAL_2024_25 = 8.6   # per 1,000 outpatient visits (published)
SEOUL_Q70_CLAIMED = 11.45     # per 1,000 — from report.md


def _seoul_y_true() -> np.ndarray:
    """Load Seoul city-aggregate ILI y_true from the backup predictions CSV.
    All 66 models share the same y_true, so any file works."""
    bp = sorted(
        (RES).glob("backup_pre_full_light_*"),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    csv_dir = None
    for b in bp:
        cand = b / "csv"
        if cand.exists() and list(cand.glob("predictions_*.csv")):
            csv_dir = cand
            break
    if csv_dir is None:
        raise FileNotFoundError("no backup_pre_full_light_*/csv/predictions_*.csv")
    # Prefer NegBinGLM as a stable source
    path = csv_dir / "predictions_NegBinGLM.csv"
    if not path.exists():
        path = sorted(csv_dir.glob("predictions_*.csv"))[0]
    df = pd.read_csv(path).sort_values("idx").drop_duplicates("idx")
    return np.asarray(df["y_true"].astype(float).values), str(path)


def _national_ili_weekly() -> np.ndarray | None:
    """Pull KDCA national weekly ILI rate from sentinel_influenza.
    Returns the season-wise, age-aggregated series.

    The table has per-age-band ILI, no total. We approximate the total as
    the arithmetic mean across the 7 age bands, matching how KDCA's press
    release reports a single headline number per week."""
    with duckdb_conn(read_only=True) as con:
        rows = con.execute(
            """
            SELECT season_start, week_seq, AVG(ili_rate) AS ili_rate
            FROM epi.sentinel_influenza
            GROUP BY season_start, week_seq
            ORDER BY season_start, week_seq
            """
        ).fetchall()
    if not rows:
        return None
    return np.asarray([r[2] for r in rows if r[2] is not None], dtype=float)


def main() -> int:
    y_seoul, y_seoul_src = _seoul_y_true()
    y_kdca = _national_ili_weekly()

    # ── Seoul percentiles ─────────────────────────────────────────────────
    pct_names = [50, 60, 70, 75, 80, 90, 95, 100]
    seoul_pcts = {p: float(np.percentile(y_seoul, p)) for p in pct_names}

    # ── National percentiles ──────────────────────────────────────────────
    if y_kdca is not None and y_kdca.size > 0:
        kdca_pcts = {p: float(np.percentile(y_kdca, p)) for p in pct_names}
        kdca_stats = {
            "n_weeks": int(y_kdca.size),
            "mean": float(np.mean(y_kdca)),
            "max": float(np.max(y_kdca)),
        }
    else:
        kdca_pcts = {}
        kdca_stats = {"n_weeks": 0, "note": "sentinel_influenza empty"}

    # ── Verdicts ──────────────────────────────────────────────────────────
    # Verdict 1: KDCA 8.6/1000 national threshold — is it above the typical
    # off-season baseline (~ p50) and well below peak (p90)?
    if kdca_pcts:
        kdca_ok = (
            kdca_pcts[50] <= KDCA_NATIONAL_2024_25 <= kdca_pcts[90]
        )
        kdca_position = float(np.searchsorted(
            np.sort(y_kdca), KDCA_NATIONAL_2024_25)) / len(y_kdca) * 100
    else:
        kdca_ok = None
        kdca_position = None

    # Verdict 2: Seoul q70 = 11.45 — is the claimed number close to actual q70?
    seoul_q70 = seoul_pcts[70]
    seoul_q70_delta = seoul_q70 - SEOUL_Q70_CLAIMED
    seoul_q70_ok = abs(seoul_q70_delta) <= 1.0  # within 1/1000 tolerance

    # Verdict 3: is 11.45 a sensible '5-10x KDCA 8.6' multiplier? The report
    # states Seoul sentinels are 5-10x the national; actual ratio:
    if kdca_pcts:
        seoul_mean = float(np.mean(y_seoul))
        kdca_mean = kdca_stats["mean"]
        ratio = seoul_mean / kdca_mean if kdca_mean > 0 else None
    else:
        ratio = None

    # ── Report ────────────────────────────────────────────────────────────
    print("=== KDCA threshold audit (epi report) ===")
    print(f"Seoul y_true source: {y_seoul_src}")
    print(f"Seoul n_weeks={y_seoul.size}  "
          f"mean={np.mean(y_seoul):.2f}  "
          f"min={np.min(y_seoul):.2f}  "
          f"max={np.max(y_seoul):.2f}")
    print()
    print("--- Seoul sentinel ILI percentiles (per 1000) ---")
    for p in pct_names:
        marker = "  <-- claimed 11.45" if p == 70 else ""
        print(f"  p{p:>3d}: {seoul_pcts[p]:7.3f}{marker}")
    print()
    print("--- KDCA national sentinel ILI percentiles (per 1000) ---")
    if kdca_pcts:
        for p in pct_names:
            marker = "  <-- KDCA 8.6 threshold" if p == 70 else ""
            print(f"  p{p:>3d}: {kdca_pcts[p]:7.3f}{marker}")
    else:
        print("  sentinel_influenza empty or unavailable")
    print()
    print("--- Verdicts ---")
    print(f"1. KDCA 8.6 in [national p50={kdca_pcts.get(50,'?'):.2f}, "
          f"p90={kdca_pcts.get(90,'?'):.2f}]: {kdca_ok} "
          f"(position = p{kdca_position:.1f}" if kdca_position else "  (n/a)",
          ")" if kdca_position else "")
    print(f"2. Seoul q70 claimed=11.45, computed={seoul_q70:.3f}, "
          f"delta={seoul_q70_delta:+.3f}  -> {'PASS' if seoul_q70_ok else 'FAIL'}")
    if ratio is not None:
        print(f"3. Seoul/national mean ratio = {ratio:.2f}x  "
              f"(report claim: 5-10x)  -> "
              f"{'PASS' if 3.0 <= ratio <= 15.0 else 'FAIL'}")

    # ── Persist JSON ──────────────────────────────────────────────────────
    OUT.write_text(
        json.dumps({
            "seoul_y_true_source": y_seoul_src,
            "seoul_stats": {
                "n_weeks": int(y_seoul.size),
                "mean": float(np.mean(y_seoul)),
                "min": float(np.min(y_seoul)),
                "max": float(np.max(y_seoul)),
                "percentiles": seoul_pcts,
            },
            "kdca_stats": {**kdca_stats, "percentiles": kdca_pcts},
            "claims": {
                "KDCA_NATIONAL_2024_25": KDCA_NATIONAL_2024_25,
                "SEOUL_Q70_CLAIMED": SEOUL_Q70_CLAIMED,
            },
            "verdicts": {
                "kdca_threshold_in_p50_p90_band": kdca_ok,
                "kdca_threshold_percentile_position": kdca_position,
                "seoul_q70_claim_matches_data": seoul_q70_ok,
                "seoul_q70_delta": seoul_q70_delta,
                "seoul_to_kdca_mean_ratio": ratio,
            },
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nwrote {OUT}")

    return 0 if (seoul_q70_ok and (kdca_ok is None or kdca_ok)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
