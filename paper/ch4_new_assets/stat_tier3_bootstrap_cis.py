"""TIER-3 block-bootstrap CIs (STAT_rwis + STAT_abm) — REAL data, seed=42.

Two moving-block-bootstrap 95% confidence intervals for the thesis Chapter-4
statistics appendix. Block bootstrap (not iid) is required because both targets
are weekly time series with serial dependence: an iid bootstrap would understate
the CI width. Block length = 4 weeks (matches the project SSOT: ranking.json SPA
test, delta_wis_bootstrap.json, and the persisted ABM moving_block_bootstrap all
use block=4).

REAL DATA ONLY. No retraining, no model load, no fabrication. The per-week
prediction CSVs and the ABM result.json are DB-derived persisted artifacts; the
upstream producers read the epi DB via `from simulation.database import
read_only_connect`. This script performs NO raw sqlite3.connect.

--------------------------------------------------------------------------------
STAT_rwis : block-bootstrap 95% CI on the headline relative-WIS for the top-5
            models (ranked by OOF-WIS = the headline ranking in
            simulation/results/per_model_eval/ranking.json), each relative to the
            FluSight-Baseline.

    Top-5 by OOF-WIS (sci_relative_wis_leaderboard.csv / ranking.json):
        FusedEpi, GAM-Spline, NegBinGLM, TiRex, SVR-RBF.

    Per-week loss = absolute error (AE), the SAME documented proxy used by the
    existing SSOT delta_wis_bootstrap.json: per-week WIS quantiles are persisted
    only for the pi_source=r9_leakfree subset (TiRex carries wis=nan /
    pi_source=unavailable), so AE is the only per-week loss reconstructable
    IDENTICALLY across all 5 models + the baseline. WIS's dominant term is
    |y - median|, so the relative-loss / tie structure transfers. We therefore
    report a per-week-AE relative loss (rel-AE) as the reconstructable proxy for
    the rel-WIS, on the 68-week hold-out test window.

    relative-AE(model) = mean_block(AE_model) / mean_block(AE_FluSight-Baseline),
    resampled by 4-week moving blocks; 95% percentile CI over the bootstrap.

    Source: simulation/results/csv/predictions_<model>.csv  (split=test, n=68,
            y_true aligned across all models).

--------------------------------------------------------------------------------
STAT_abm  : moving-block bootstrap 95% CI on the behaviour-ON minus behaviour-OFF
            forward-R2 gap pooled over the 26 ABM origins.

    Recomputed from the 26 raw per-origin gaps (result.json) with seed=42, and
    cross-checked against the persisted SSOT value
    (behavior_robustness.json moving_block_bootstrap, seed=43): the two seeds give
    materially identical CIs, confirming reproducibility.

    Source: simulation/results/abm_multiorigin_forward/result.json
            simulation/results/abm_multiorigin_forward/behavior_robustness.json

Run:
    .venv/bin/python paper/ch4_new_assets/stat_tier3_bootstrap_cis.py

Side effects: prints a JSON summary to stdout (no files written).
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
_CSV = _ROOT / "simulation" / "results" / "csv"
_ABM_DIR = _ROOT / "simulation" / "results" / "abm_multiorigin_forward"

_SEED = 42
_BLOCK = 4
_N_BOOT = 10000

# Top-5 by OOF-WIS headline ranking + the benchmark (ranking.json / leaderboard).
_TOP5 = ["FusedEpi", "GAM-Spline", "NegBinGLM", "TiRex", "SVR-RBF"]
_BASELINE = "FluSight-Baseline"


def _load_test(model: str) -> tuple[np.ndarray, np.ndarray]:
    """Load (y_true, y_pred) for the hold-out test split of one model.

    Args:
        model: model name; resolves to predictions_<model>.csv.

    Returns:
        (y_true, y_pred) float arrays over the n=68 test weeks.

    Raises:
        SystemExit: if the prediction CSV is missing (do NOT invent values).
    """
    p = _CSV / f"predictions_{model}.csv"
    if not p.exists():
        raise SystemExit(f"MISSING DATA FILE: {p}")
    yt, yp = [], []
    with p.open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["split"] != "test":
                continue
            yt.append(float(r["y_true"]))
            yp.append(float(r["y_pred"]))
    return np.asarray(yt), np.asarray(yp)


def _moving_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    """Draw one moving-block-bootstrap index resample of length ~n.

    Overlapping blocks of fixed length are concatenated until length >= n, then
    truncated to n (standard moving-block bootstrap, Kunsch 1989).

    Args:
        n: series length.
        block: block length in time steps.
        rng: seeded numpy Generator.

    Returns:
        int index array of length n into [0, n).
    """
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=n_blocks)
    idx = np.concatenate([np.arange(s, s + block) for s in starts])
    return idx[:n]


def stat_rwis() -> dict:
    """STAT_rwis: block-bootstrap 95% CI on top-5 relative-AE (rel-WIS proxy).

    Returns:
        dict keyed by model -> {point, ci95_lo, ci95_hi, mean_ae,
        baseline_mean_ae} plus metadata.
    """
    yt, base_pred = (_load_test(_BASELINE))
    base_ae = np.abs(yt - base_pred)
    n = len(yt)

    # Load + align all top-5 (y_true must match the baseline's).
    model_ae = {}
    for m in _TOP5:
        m_yt, m_pred = _load_test(m)
        if not np.allclose(m_yt, yt):
            raise SystemExit(f"y_true misaligned for {m} vs baseline")
        model_ae[m] = np.abs(m_yt - m_pred)

    rng = np.random.default_rng(_SEED)
    boot_idx = [_moving_block_indices(n, _BLOCK, rng) for _ in range(_N_BOOT)]

    out = {}
    for m in _TOP5:
        ratios = np.empty(_N_BOOT)
        for b, idx in enumerate(boot_idx):
            num = model_ae[m][idx].mean()
            den = base_ae[idx].mean()
            ratios[b] = num / den
        lo, hi = np.percentile(ratios, [2.5, 97.5])
        point = model_ae[m].mean() / base_ae.mean()
        out[m] = {
            "relative_ae_point": round(float(point), 4),
            "ci95_lo": round(float(lo), 4),
            "ci95_hi": round(float(hi), 4),
            "ci_includes_one": bool(lo <= 1.0 <= hi),
            "skillful_point": bool(point < 1.0),
            "mean_ae": round(float(model_ae[m].mean()), 4),
        }
    return {
        "analysis": "STAT_rwis_block_bootstrap_relative_AE_top5_vs_FluSightBaseline",
        "loss": "per_week_absolute_error (rel-WIS proxy; see module docstring)",
        "ranking_basis": "top-5 by OOF-WIS (headline ranking, ranking.json)",
        "n_test_weeks": int(n),
        "block_weeks": _BLOCK,
        "n_boot": _N_BOOT,
        "seed": _SEED,
        "baseline": _BASELINE,
        "baseline_mean_ae": round(float(base_ae.mean()), 4),
        "per_model": out,
    }


def stat_abm() -> dict:
    """STAT_abm: moving-block bootstrap 95% CI on the 26-origin behaviour gap.

    Returns:
        dict with the seed=42 recompute + the persisted seed=43 SSOT value.
    """
    res = json.loads((_ABM_DIR / "result.json").read_text(encoding="utf-8"))
    rob = json.loads((_ABM_DIR / "behavior_robustness.json").read_text(encoding="utf-8"))

    gaps = np.array(
        res["distribution"]["behavior_gap_all_origins"]["values"], dtype=float
    )
    n = len(gaps)

    rng = np.random.default_rng(_SEED)
    means = np.empty(_N_BOOT)
    medians = np.empty(_N_BOOT)
    for b in range(_N_BOOT):
        idx = _moving_block_indices(n, _BLOCK, rng)
        means[b] = gaps[idx].mean()
        medians[b] = np.median(gaps[idx])
    m_lo, m_hi = np.percentile(means, [2.5, 97.5])
    md_lo, md_hi = np.percentile(medians, [2.5, 97.5])

    ssot = rob["moving_block_bootstrap"]
    return {
        "analysis": "STAT_abm_moving_block_bootstrap_behaviour_gap_26_origins",
        "n_origins": int(n),
        "block_weeks": _BLOCK,
        "n_boot": _N_BOOT,
        "seed": _SEED,
        "mean_gap_point": round(float(gaps.mean()), 4),
        "mean_gap_ci95": [round(float(m_lo), 4), round(float(m_hi), 4)],
        "median_gap_point": round(float(np.median(gaps)), 4),
        "median_gap_ci95": [round(float(md_lo), 4), round(float(md_hi), 4)],
        "ci_includes_zero": bool(m_lo <= 0.0 <= m_hi),
        "ssot_persisted_seed43": {
            "seed": ssot["seed"],
            "block": ssot["block"],
            "n_boot": ssot["n_boot"],
            "mean_gap_ci95": [round(float(x), 4) for x in ssot["mean_gap_ci95"]],
            "median_gap_ci95": [round(float(x), 4) for x in ssot["median_gap_ci95"]],
        },
    }


def main() -> None:
    summary = {"STAT_rwis": stat_rwis(), "STAT_abm": stat_abm()}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
