"""Post-E P0#2 fix — extend split-conformal PI to all 66 models.

Before: 5/66 models had phase6_conformal WIS, 61/66 fell back to Gaussian
approximation (wis ≈ 0.81 × crps_gaussian, i.e. WIS ranking degenerated
to CRPS ranking).

After: every model gets a proper split-conformal 5-quantile band using
val (41 wks) as the calibration set and test (69 wks) as the evaluation
set. This is the S0-1-compliant disjoint split (val ≠ test).

Outputs:
  - simulation/results/post_E/conformal_index.json — per-model quantiles
  - simulation/results/post_E/pi_samples_wide.csv  — 5-quantile wide table
      rewritten with ``source='post_E_conformal_all'`` for all 66 models.

Run:
    .venv/Scripts/python.exe -m simulation.scripts.post_E_extend_conformal
"""
from __future__ import annotations

import glob
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("post_E.conformal_all")

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "simulation" / "results" / "csv"
OUT_DIR = ROOT / "simulation" / "results" / "post_E"
OUT_INDEX = OUT_DIR / "conformal_index.json"
OUT_PI = OUT_DIR / "pi_samples_wide.csv"

# Week start reference date (reused from the original pi_samples_wide)
WEEK_START_REF = "2024-12-15"


def _conformal_quantile(abs_resid: np.ndarray, alpha: float) -> float:
    """Split-conformal quantile, formula: q_alpha = sorted(|r|)[⌈(n+1)(1-α)⌉ - 1].

    This is the finite-sample correction (Vovk et al., matches phase6).
    """
    n = len(abs_resid)
    if n < 2:
        return float("nan")
    k = int(np.ceil((n + 1) * (1.0 - alpha))) - 1
    k = min(max(k, 0), n - 1)
    return float(np.sort(abs_resid)[k])


def build_bands() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    week_start = pd.date_range(WEEK_START_REF, periods=69, freq="W-SUN")
    csvs = sorted(glob.glob(str(CSV / "predictions_*.csv")))
    log.info("found %d predictions CSVs", len(csvs))

    index: dict[str, dict] = {}
    pi_rows: list[dict] = []
    for fp in csvs:
        name = os.path.basename(fp).replace("predictions_", "").replace(".csv", "")
        df = pd.read_csv(fp)
        v = df[df["split"] == "val"].sort_values("idx")
        t = df[df["split"] == "test"].sort_values("idx")
        if len(v) < 5 or len(t) < 5:
            log.warning("skip %s (val=%d, test=%d)", name, len(v), len(t))
            continue
        y_val_true = v["y_true"].to_numpy(dtype=float)
        y_val_pred = v["y_pred"].to_numpy(dtype=float)
        y_te_true = t["y_true"].to_numpy(dtype=float)
        y_te_pred = t["y_pred"].to_numpy(dtype=float)

        abs_r = np.abs(y_val_true - y_val_pred)
        q_alpha05 = _conformal_quantile(abs_r, alpha=0.05)   # 95% band  → q025/q975
        q_alpha10 = _conformal_quantile(abs_r, alpha=0.10)   # 90% band  → q050/q950
        if not (np.isfinite(q_alpha05) and np.isfinite(q_alpha10)):
            log.warning("skip %s (conformal quantile NaN)", name)
            continue

        q025 = np.clip(y_te_pred - q_alpha05, 0.0, None)
        q975 = y_te_pred + q_alpha05
        q050 = np.clip(y_te_pred - q_alpha10, 0.0, None)
        q950 = y_te_pred + q_alpha10
        q500 = y_te_pred

        for i, ws in enumerate(week_start[:len(y_te_true)]):
            pi_rows.append({
                "week_start": ws.date().isoformat(),
                "model": name,
                "y_true": float(y_te_true[i]),
                "q025": float(q025[i]),
                "q500": float(q500[i]),
                "q975": float(q975[i]),
                "q050": float(q050[i]),
                "q950": float(q950[i]),
                "source": "post_E_conformal_all",
            })

        # Empirical coverage on val (calibration)
        cov95_val = float(np.mean(abs_r <= q_alpha05))
        cov90_val = float(np.mean(abs_r <= q_alpha10))
        # Empirical coverage on test (out-of-cal → target 95% / 90%)
        abs_te = np.abs(y_te_true - y_te_pred)
        cov95_te = float(np.mean(abs_te <= q_alpha05))
        cov90_te = float(np.mean(abs_te <= q_alpha10))
        index[name] = {
            "n_cal": int(len(abs_r)),
            "n_test": int(len(abs_te)),
            "q_alpha05": q_alpha05,
            "q_alpha10": q_alpha10,
            "cov_95_val": cov95_val,
            "cov_90_val": cov90_val,
            "cov_95_test": cov95_te,
            "cov_90_test": cov90_te,
            "source": "post_E_conformal_all",
        }

    pi = pd.DataFrame(pi_rows)
    pi.to_csv(OUT_PI, index=False)
    log.info("wrote %s (%d rows × %d models)", OUT_PI.name, len(pi), pi["model"].nunique())
    OUT_INDEX.write_text(json.dumps({
        "generated_at": pd.Timestamp.now().isoformat(),
        "schema_version": "0.1",
        "method": "split_conformal (val_cal → test_eval)",
        "alphas": [0.05, 0.10],
        "models": index,
    }, indent=2), encoding="utf-8")
    log.info("wrote %s (%d models)", OUT_INDEX.name, len(index))


if __name__ == "__main__":
    build_bands()
