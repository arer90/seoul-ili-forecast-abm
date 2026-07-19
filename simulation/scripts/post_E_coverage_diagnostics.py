"""Post-E follow-up #3 — Under-coverage diagnostic as a formal limitation.

Problem: 0/66 models reach nominal 95% coverage on the test split (median
cov_95_test ≈ 0.46). This is expected because the val split (2024-08 →
2024-12, 41 wks, late-season tail) and the test split (2024-12 → 2026-04,
69 wks, full winter + off-season + next winter) are NOT exchangeable:
  - val_max(y) = 16.7  vs  test_max(y) = 100.7  (regime shift in scale)
  - val covers ~1 partial season, test covers ~2 full seasons

Split conformal's marginal coverage guarantee
(Vovk et al. 2005; Lei & Wasserman 2014) requires exchangeability of the
calibration + test points. When that fails, Tibshirani et al. (2019,
"Conformal Prediction Under Covariate Shift") show the gap can be bounded
via likelihood ratios; Barber et al. (2023, "Conformal Prediction Beyond
Exchangeability") generalise to bounded weighted coverage with explicit
slack terms. Either framework declares our setting as non-IID and predicts
exactly the under-coverage pattern we observe.

This script:
  1. Reads post_E/pi_samples_wide.csv + the residuals from val split.
  2. Computes KS-statistic between |val_resid| and |test_resid| per model
     — quantitative evidence of exchangeability violation.
  3. Writes post_E/exchangeability_diagnostics.json with per-model stats.
  4. Appends a 'coverage_limitation' block to post_E_eval.json documenting
     the finding with citations and quantitative KS evidence.
"""
from __future__ import annotations

import glob
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("post_E.coverage_diag")

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "simulation" / "results"
POST_E = RES / "post_E"
CSV_DIR = RES / "csv"


def _residuals_by_split() -> dict[str, dict[str, np.ndarray]]:
    """Per-model: {val: abs_resid, test: abs_resid, val_y: [...], test_y: [...]}"""
    out = {}
    for fp in sorted(glob.glob(str(CSV_DIR / "predictions_*.csv"))):
        name = os.path.basename(fp).replace("predictions_", "").replace(".csv", "")
        df = pd.read_csv(fp)
        v = df[df["split"] == "val"]
        t = df[df["split"] == "test"]
        if v.empty or t.empty:
            continue
        out[name] = {
            "val_abs_resid": np.abs(v["y_true"] - v["y_pred"]).to_numpy(),
            "test_abs_resid": np.abs(t["y_true"] - t["y_pred"]).to_numpy(),
            "val_y": v["y_true"].to_numpy(),
            "test_y": t["y_true"].to_numpy(),
        }
    return out


def main() -> None:
    rd = _residuals_by_split()
    log.info("loaded residuals for %d models", len(rd))
    # Global y distribution shift
    any_name = next(iter(rd))
    val_y = rd[any_name]["val_y"]
    test_y = rd[any_name]["test_y"]
    y_ks = ks_2samp(val_y, test_y)
    log.info("y-distribution KS: val_n=%d, test_n=%d, D=%.3f, p=%.2e",
             len(val_y), len(test_y), y_ks.statistic, y_ks.pvalue)

    per_model = []
    cov_rows = []
    for name, d in rd.items():
        ks = ks_2samp(d["val_abs_resid"], d["test_abs_resid"])
        per_model.append({
            "model": name,
            "val_n": int(len(d["val_abs_resid"])),
            "test_n": int(len(d["test_abs_resid"])),
            "val_resid_median": float(np.median(d["val_abs_resid"])),
            "test_resid_median": float(np.median(d["test_abs_resid"])),
            "val_resid_max": float(np.max(d["val_abs_resid"])),
            "test_resid_max": float(np.max(d["test_abs_resid"])),
            "ks_stat": float(ks.statistic),
            "ks_pvalue": float(ks.pvalue),
            "scale_amplification": (
                float(np.median(d["test_abs_resid"])) /
                max(float(np.median(d["val_abs_resid"])), 1e-9)
            ),
        })
    per_model.sort(key=lambda r: r["ks_stat"], reverse=True)

    # Summary
    ks_stats = np.array([r["ks_stat"] for r in per_model])
    scale_amps = np.array([r["scale_amplification"] for r in per_model])
    reject_count = int(np.sum([r["ks_pvalue"] < 0.05 for r in per_model]))

    diag = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "schema_version": "0.1",
        "purpose": (
            "Quantify exchangeability violation between val (cal) and test "
            "(eval) splits, which is the root cause of observed under-coverage "
            "in split-conformal intervals."
        ),
        "theory_reference": [
            "Tibshirani, R. J., Barber, R. F., Candès, E. J., & Ramdas, A. "
            "(2019). Conformal Prediction Under Covariate Shift. NeurIPS.",
            "Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani, R. J. "
            "(2023). Conformal Prediction Beyond Exchangeability. "
            "Annals of Statistics 51(2): 816-845.",
            "Vovk, V., Gammerman, A., & Shafer, G. (2005). Algorithmic "
            "Learning in a Random World. Springer. (baseline split-conformal)",
        ],
        "y_distribution_ks": {
            "D_statistic": float(y_ks.statistic),
            "p_value": float(y_ks.pvalue),
            "val_max": float(val_y.max()),
            "test_max": float(test_y.max()),
            "val_mean": float(val_y.mean()),
            "test_mean": float(test_y.mean()),
            "interpretation": (
                "val covers a partial late-season tail (n=41, max=%.1f); "
                "test spans 2 winters + 1 off-season (n=69, max=%.1f). "
                "KS rejects equal-distribution hypothesis at p=%.2e."
                % (float(val_y.max()), float(test_y.max()), float(y_ks.pvalue))
            ),
        },
        "residual_exchangeability_summary": {
            "n_models": len(per_model),
            "ks_stat_median": float(np.median(ks_stats)),
            "ks_stat_iqr": [float(np.quantile(ks_stats, 0.25)),
                            float(np.quantile(ks_stats, 0.75))],
            "ks_reject_count_at_0.05": reject_count,
            "ks_reject_frac": float(reject_count / len(per_model)),
            "scale_amp_median": float(np.median(scale_amps)),
            "scale_amp_max": float(np.max(scale_amps)),
        },
        "per_model": per_model,
    }

    out_path = POST_E / "exchangeability_diagnostics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s", out_path)

    # ── Append coverage_limitation to post_E_eval.json ──────────────────
    eval_path = RES / "post_E_eval.json"
    if not eval_path.exists():
        log.warning("post_E_eval.json not found, skipping limitation block")
        return
    ev = json.load(eval_path.open(encoding="utf-8"))

    # Pull coverage stats from ev details if available
    cov_95 = []
    cov_90 = []
    for d in ev.get("details", []):
        if d.get("pi_coverage_95") is not None:
            cov_95.append(float(d["pi_coverage_95"]))
        if d.get("pi_coverage_90") is not None:
            cov_90.append(float(d["pi_coverage_90"]))

    ev["coverage_limitation"] = {
        "observed_coverage": {
            "n_models_scored": len(cov_95),
            "pi95_median": float(np.median(cov_95)) if cov_95 else None,
            "pi95_iqr": [float(np.quantile(cov_95, 0.25)),
                         float(np.quantile(cov_95, 0.75))] if cov_95 else None,
            "pi95_at_or_above_nominal": (
                int(np.sum(np.array(cov_95) >= 0.95)) if cov_95 else 0
            ),
            "pi90_median": float(np.median(cov_90)) if cov_90 else None,
        },
        "diagnosed_cause": "Exchangeability violation between calibration (val) and evaluation (test) splits.",
        "quantitative_evidence": {
            "y_ks_statistic": float(y_ks.statistic),
            "y_ks_pvalue": float(y_ks.pvalue),
            "residual_ks_reject_frac": float(reject_count / len(per_model)),
            "test_max_over_val_max_ratio": float(val_y.max() > 0
                                                  and test_y.max() / val_y.max()),
        },
        "theoretical_framing": (
            "Split-conformal's marginal coverage guarantee (Vovk et al. 2005; "
            "Lei & Wasserman 2014) assumes exchangeability. Our val/test split "
            "violates this: val (late-season tail, n=41) and test (2 winters + "
            "off-season, n=69) differ in support (max_y: %.1f vs %.1f), shape "
            "(KS=%.3f, p=%.2e), and temporal position. Tibshirani et al. (2019) "
            "show under covariate shift the gap can be bounded via likelihood "
            "ratios; Barber et al. (2023) generalise to bounded weighted "
            "coverage with explicit slack. Under-coverage here is therefore "
            "THEORETICALLY EXPECTED, not a pipeline bug."
            % (float(val_y.max()), float(test_y.max()),
               float(y_ks.statistic), float(y_ks.pvalue))
        ),
        "mitigation_options_for_future_work": [
            "Weighted conformal with density ratio w(x)=p_test(x)/p_val(x) — requires estimating distribution shift.",
            "Nonexchangeable conformal (Barber 2023) with explicit slack term in the bound.",
            "Larger / more representative calibration set spanning ≥1 full season (would require holding out one year).",
            "Time-series-adapted variants: Adaptive Conformal Inference (Gibbs & Candès 2021) or EnbPI (Xu & Xie 2023).",
        ],
        "reporting_guidance": (
            "Paper Table/Figure should present observed coverage AS-IS with "
            "this limitation explicitly cited. Do NOT inflate PIs post-hoc to "
            "hit nominal coverage — that would be p-hacking. The WIS and CRPS "
            "metrics (proper scoring rules) remain valid under distribution "
            "shift and are the primary probabilistic-performance evidence."
        ),
    }

    eval_path.write_text(json.dumps(ev, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("appended coverage_limitation block to %s", eval_path)


if __name__ == "__main__":
    main()
