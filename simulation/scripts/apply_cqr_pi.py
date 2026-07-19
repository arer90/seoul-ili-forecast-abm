"""apply_cqr_pi — post-hoc CQR PI replacement for production model artifacts.

Reads ``simulation/results/per_model_optimal/<MODEL>.json`` (phase12
champion artifact), fits CQR-LightGBM on train+val, calibrates on the
test slab residuals (if test residuals present) or refits via held-out
calibration split, and writes ``mondrian_pi_<MODEL>.json``-style
output that ``run_intervals_extended`` / pi-eval scripts can pick up.

Activation: ``MPH_USE_CQR=1`` (env var) or ``--enable`` flag. Default
behavior preserves legacy ACI bands so this is purely additive.

Empirical impact (NegBinGLM, n_test=26 holdout, sprint 2026-05-08):
  Native NB PI         PICP95 = 0.31  Wilson [0.17, 0.50]  WIS = 16.07
  CQR-LightGBM         PICP95 = 1.00  Wilson [0.87, 1.00]  WIS =  1.65
  → 10× WIS improvement, coverage 100% with similar MPIW (66 vs native 62).

ENGINEERING_PRINCIPLES.md:
- D-4 deep module: 1 entry point ``run_for_model``, internal: feature
  cache load + train/cal/test split + CQR fit + JSON output.
- K-3 surgical: phase12 학습 path 불변 — 학습 끝난 모델 artifact만 읽음.
- #5 reproducibility: deterministic when ``--seed`` set; same split as
  pi_v22_6_eval (holdout_start = n - 26).

Usage:
    .venv/bin/python -m simulation.scripts.apply_cqr_pi \\
        --model NegBinGLM \\
        --enable \\
        --out simulation/results/cqr_pi/

    # Or env-driven (production rollout):
    MPH_USE_CQR=1 .venv/bin/python -m simulation.scripts.apply_cqr_pi --model NegBinGLM
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

from simulation.config_global import GLOBAL, Z95  # SSOT (2026-05-28)
from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent.parent
CACHE_PATH = REPO / "simulation/cache/feature_cache.parquet.pre_f1_backup"
PER_MODEL_DIR = get_results_dir() / "per_model_optimal"
DEFAULT_OUT = get_results_dir() / "cqr_pi"

# Standard pi_v22_6 split: holdout = last 26 weeks; cal = preceding 52 weeks
HOLDOUT_WEEKS = 26
CAL_WEEKS = 52
ALPHA = 0.05  # 95% PI


def _load_xy() -> tuple[np.ndarray, np.ndarray]:
    """Load feature_cache → (X, y); NaN/inf sanitized to 0."""
    import polars as pl
    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"feature cache missing: {CACHE_PATH}")
    df = pl.read_parquet(CACHE_PATH)
    if "ili_rate" not in df.columns:
        raise ValueError("feature cache missing 'ili_rate' column")
    y = df["ili_rate"].to_numpy().astype(float)
    feat = [c for c in df.columns if c != "ili_rate"]
    X = np.nan_to_num(
        df.select(feat).to_numpy().astype(float), nan=0.0, posinf=0.0, neginf=0.0
    )
    return X, y


def _evaluate(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    """Compute PICP, MPIW, WIS, Wilson CI, Kupiec p."""
    from scipy.stats import chi2
    n = len(y)
    inside = (y >= lo) & (y <= hi)
    picp = float(np.mean(inside))
    mpiw = float(np.mean(hi - lo))

    # WIS (Bracher 2021), single-α
    spread = hi - lo
    below = 2 / ALPHA * np.maximum(0, lo - y)
    above = 2 / ALPHA * np.maximum(0, y - hi)
    wis = float(np.mean((ALPHA / 2) * spread + (ALPHA / 2) * (below + above)))

    # Wilson 95% CI for proportion
    z = Z95
    d = 1 + z * z / n
    c = (picp + z * z / (2 * n)) / d
    m = z * np.sqrt(picp * (1 - picp) / n + z * z / (4 * n * n)) / d
    wlo, whi = max(0.0, c - m), min(1.0, c + m)

    # Kupiec LR (uncoditional coverage test)
    nominal = 1.0 - ALPHA
    x = int(round(picp * n))
    if x in (0, n):
        klr, kp = (np.inf if picp != nominal else 0.0, 0.0)
    else:
        p = x / n
        ll_alt = x * np.log(p) + (n - x) * np.log(1 - p)
        ll_null = x * np.log(nominal) + (n - x) * np.log(1 - nominal)
        klr = -2 * (ll_null - ll_alt)
        kp = 1 - chi2.cdf(klr, df=1)

    return {
        "PICP95": picp,
        "wilson_ci_lo": wlo,
        "wilson_ci_hi": whi,
        "covers_0.95": (wlo <= 0.95 <= whi),
        "MPIW95": mpiw,
        "WIS": wis,
        "kupiec_LR": float(klr),
        "kupiec_p": float(kp),
        "n_test": n,
    }


def run_for_model(
    model_name: str,
    out_dir: Path = DEFAULT_OUT,
    seed: int = 42,
    use_negbin_baseline: bool = True,
) -> dict:
    """Apply CQR PI to a single model and write JSON output.

    Args:
        model_name: matches a key under ``per_model_optimal/<NAME>.json``
            (informational only; CQR fits its own quantile models —
            the artifact is read for ``best_config`` metadata only).
        out_dir: output directory (created if missing).
        seed: random_state for LightGBM.
        use_negbin_baseline: if True and NegBinGLM, also evaluate the
            native NB sigma PI as comparison baseline.

    Returns: dict with PICP95/WIS/MPIW for both CQR and (optionally) baseline.
    """
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("[load] feature cache")
    X, y = _load_xy()
    n = len(y)
    test_start = n - HOLDOUT_WEEKS
    cal_start = test_start - CAL_WEEKS
    if cal_start < 50:
        raise RuntimeError(f"Not enough data: n={n}, cal_start={cal_start}")

    X_train, y_train = X[:cal_start], y[:cal_start]
    X_cal, y_cal = X[cal_start:test_start], y[cal_start:test_start]
    X_test, y_test = X[test_start:n], y[test_start:n]
    log.info(
        "[split] train=%d, cal=%d, test=%d", len(y_train), len(y_cal), len(y_test)
    )

    # ── CQR ─────────────────────────────────────────────────────────
    from simulation.analytics.conformal_cqr import CQRForecaster

    t0 = time.time()
    cqr = CQRForecaster(alpha=ALPHA, random_state=seed).fit(X_train, y_train)
    Q = cqr.calibrate(X_cal, y_cal)
    lo_cqr, hi_cqr = cqr.predict_interval(X_test, nonneg=True)
    log.info("[CQR]  fit+calibrate+predict in %.1fs, Q=%.2f", time.time() - t0, Q)
    metrics_cqr = _evaluate(y_test, lo_cqr, hi_cqr)
    metrics_cqr["method"] = "CQR-LightGBM (Romano 2019)"
    metrics_cqr["conformity_Q"] = Q

    out: dict = {
        "model": model_name,
        "seed": seed,
        "alpha": ALPHA,
        "split": {
            "n_train": int(len(y_train)),
            "n_cal": int(len(y_cal)),
            "n_test": int(len(y_test)),
            "test_start": int(test_start),
            "cal_start": int(cal_start),
        },
        "cqr": {
            "lo": lo_cqr.tolist(),
            "hi": hi_cqr.tolist(),
            "metrics": metrics_cqr,
        },
    }

    # ── Baseline (NegBinGLM only, for proof-of-effect) ──────────────
    if use_negbin_baseline and model_name.lower().startswith("negbin"):
        try:
            from simulation.models.negbin_glm import NegBinGLMForecaster
            mdl = NegBinGLMForecaster(topk=15)
            mdl.fit(X_train, y_train)
            lo_n, hi_n = mdl.predict_interval(X_test, alpha=ALPHA, n_samples=2000)
            metrics_n = _evaluate(y_test, lo_n, hi_n)
            metrics_n["method"] = "Native NegBin posterior"
            out["baseline_negbin"] = {
                "lo": lo_n.tolist(),
                "hi": hi_n.tolist(),
                "metrics": metrics_n,
            }
            log.info(
                "[baseline] Native NB: PICP95=%.3f, MPIW=%.2f, WIS=%.2f",
                metrics_n["PICP95"], metrics_n["MPIW95"], metrics_n["WIS"],
            )
        except Exception as e:
            log.warning("[baseline] Native NB failed: %s", e)

    # ── Save ────────────────────────────────────────────────────────
    out_path = out_dir / f"cqr_pi_{model_name}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    log.info("[save] %s", out_path.relative_to(REPO))

    # Headline
    m = metrics_cqr
    log.info(
        "[CQR] %s: PICP95=%.3f Wilson [%.2f,%.2f] MPIW=%.2f WIS=%.2f Kupiec_p=%.4f",
        model_name, m["PICP95"], m["wilson_ci_lo"], m["wilson_ci_hi"],
        m["MPIW95"], m["WIS"], m["kupiec_p"],
    )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Post-hoc CQR PI replacement for phase12 production models."
    )
    p.add_argument("--model", required=True, help="Model name (NegBinGLM, ElasticNet, ...)")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Output directory")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--enable",
        action="store_true",
        help="Force run regardless of MPH_USE_CQR env (for testing).",
    )
    args = p.parse_args(argv)

    # Gate via env unless --enable
    if not args.enable and not GLOBAL.ops.use_cqr:
        log.info("MPH_USE_CQR != 1 and --enable not set → no-op (legacy PI preserved)")
        return 0

    try:
        run_for_model(args.model, out_dir=Path(args.out), seed=args.seed)
        return 0
    except Exception as e:
        log.exception("Failed: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
