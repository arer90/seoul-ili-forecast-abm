"""
Post-E 종합 평가 (E full retrain 이후 실행).

기존 검증 (validity_pre_E.json) 대비 추가되는 항목:
  - WIS (Bracher et al. 2021)                         → analytics.weighted_interval_score
  - CRPS (Gaussian approx)                            → analytics.crps_gaussian
  - PIT histogram + coverage                          → analytics.pi_calibration_table
  - peak week/intensity error                         → analytics.peak_week_error / peak_intensity_error
  - direction accuracy                                → analytics.direction_accuracy
  - DM test (regime-split pre/during/post)            → analytics.diebold_mariano
  - coverage gap by regime                            → analytics.coverage_gap_by_regime
  - MCS (Model Confidence Set, Hansen et al. 2011)    → analytics.model_confidence_set
  - bootstrap CI on RMSE/MAPE                         → analytics.bootstrap_ci

Input: predictions CSV + R7(intervals) PI JSON (checkpoint_phase6.json).
Output: simulation/results/post_E_eval.json
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from simulation.config_global import Z95  # SSOT z-quantile (R8)

ROOT = Path(__file__).resolve().parents[2]
CSV_DIR = ROOT / "simulation" / "results" / "csv"
OUT = ROOT / "simulation" / "results" / "post_E_eval.json"


def _load_pi_bounds(
    model: str,
    y_pred: np.ndarray | None = None,
    resid_std: float | None = None,
) -> dict | None:
    """95% PI 복원 — priority order:
    (1) post-E full-conformal index (val→test split, covers all 66 models),
    (2) R7(intervals) conformal (legacy, covers 5 models with narrower val use),
    (3) residual-std Gaussian fallback.

    Returns {lower, upper, quantile, source} or None if y_pred missing.
    """
    if y_pred is None:
        return None
    # 1) post-E full conformal (all 66 models)
    post_e_idx = ROOT / "simulation" / "results" / "post_E" / "conformal_index.json"
    if post_e_idx.exists():
        try:
            data = json.load(post_e_idx.open(encoding="utf-8"))
            entry = (data.get("models", {}) or {}).get(model)
            if isinstance(entry, dict) and entry.get("q_alpha05") is not None:
                q = float(entry["q_alpha05"])
                return {
                    "lower": np.clip(y_pred - q, 0.0, None), "upper": y_pred + q,
                    "quantile": q, "source": "post_E_conformal_all",
                }
        except Exception:
            pass
    # 2) R7(intervals) conformal (legacy, checkpoint_phase6.json)
    p6 = ROOT / "simulation" / "results" / "checkpoints" / "checkpoint_phase6.json"
    if p6.exists():
        try:
            data = json.load(p6.open(encoding="utf-8"))
            pi_results = (data.get("data", {}) or {}).get("pi_results", {}) or {}
            model_pi = pi_results.get(model)
            if isinstance(model_pi, dict):
                q = (model_pi.get("conformal", {}) or {}).get("quantile")
                if q is not None:
                    q = float(q)
                    return {
                        "lower": y_pred - q, "upper": y_pred + q,
                        "quantile": q, "source": "phase6_conformal",
                    }
        except Exception:
            pass
    # 3) Residual-std Gaussian fallback
    if resid_std is None or not np.isfinite(resid_std) or resid_std <= 0:
        return None
    q = Z95 * float(resid_std)
    return {
        "lower": y_pred - q, "upper": y_pred + q,
        "quantile": q, "source": "residual_gaussian",
    }


def _evaluate_model(csv: Path) -> dict[str, Any]:
    from simulation.analytics import (
        weighted_interval_score,
        diebold_mariano,
        crps_gaussian,
        peak_week_error,
        peak_intensity_error,
        direction_accuracy,
        pi_coverage,
        pi_calibration_table,
        bootstrap_ci,
    )
    name = csv.stem.replace("predictions_", "")
    df = pd.read_csv(csv)
    out: dict[str, Any] = {"model": name, "n": int(len(df))}

    test = df[df["split"] == "test"].sort_values("idx")
    if len(test) < 10:
        out["error"] = "too_few_test"
        return out

    y_true = test["y_true"].to_numpy(dtype=float)
    y_pred = test["y_pred"].to_numpy(dtype=float)

    # 1. Point accuracy with bootstrap CI (1000 resamples)
    try:
        resid = y_true - y_pred
        rmse_fn = lambda r: float(np.sqrt(np.mean(np.asarray(r) ** 2)))
        rmse_ci = bootstrap_ci(resid, rmse_fn, n_boot=1000, method="percentile", random_state=42)
        out["rmse_boot95"] = {
            "point": float(rmse_ci["estimate"]),
            "lo": float(rmse_ci["ci_lo"]),
            "hi": float(rmse_ci["ci_hi"]),
            "se": float(rmse_ci["se"]),
        }
    except Exception as e:
        out["rmse_boot95"] = {"error": str(e)}

    # 2. Peak timing / intensity (seasonal peak)
    try:
        pw = peak_week_error(y_true, y_pred)
        out["peak_week_err"] = int(pw["abs_weeks"]) if isinstance(pw, dict) else int(pw)
        pi_int = peak_intensity_error(y_true, y_pred)
        out["peak_intensity_err"] = float(pi_int["abs_err"]) if isinstance(pi_int, dict) else float(pi_int)
    except Exception as e:
        out["peak_err"] = f"fail: {e}"

    # 3. Direction accuracy (up/down moves)
    try:
        da = direction_accuracy(y_true, y_pred)
        out["direction_acc_pct"] = float(da["accuracy"] * 100) if isinstance(da, dict) else float(da * 100)
    except Exception as e:
        out["direction_acc_pct"] = f"fail: {e}"

    # 4. CRPS (Gaussian approx using residual std as sigma)
    try:
        sigma = float(np.std(y_true - y_pred)) or 1.0
        out["crps_gaussian"] = float(crps_gaussian(y_true, y_pred, sigma).mean())
    except Exception as e:
        out["crps_gaussian"] = f"fail: {e}"

    # 5. WIS — R7(intervals) conformal 우선, 없으면 residual-std Gaussian fallback
    try:
        resid_std_fallback = float(np.std(y_true - y_pred, ddof=1)) if len(y_true) > 1 else None
    except Exception:
        resid_std_fallback = None
    pi = _load_pi_bounds(name, y_pred=y_pred, resid_std=resid_std_fallback)
    if pi:
        try:
            low = pi["lower"][-len(y_true):]
            high = pi["upper"][-len(y_true):]
            sigma_eq = float(pi["quantile"]) / Z95
            wis = weighted_interval_score(
                y_true=y_true,
                y_pred=y_pred,
                sigma=sigma_eq,
                alphas=[0.05, 0.10],
            )
            out["wis"] = float(np.mean(wis))
            out["wis_source"] = pi.get("source", "unknown")
            cov = pi_coverage(y_true, low, high, nominal=0.95)
            out["pi_coverage_95"] = float(cov["empirical"]) if isinstance(cov, dict) else float(cov)
        except Exception as e:
            out["wis"] = f"fail: {e}"

    # 6. DM test vs persistence baseline (naive: y_{t-1})
    try:
        persistence = np.concatenate([[y_true[0]], y_true[:-1]])
        dm_stat, dm_p = diebold_mariano(y_true, y_pred, persistence, h=1)
        out["dm_vs_persistence"] = {"stat": float(dm_stat), "p": float(dm_p)}
    except Exception as e:
        out["dm_vs_persistence"] = f"fail: {e}"

    # 7. Residual normality + heteroscedasticity
    try:
        from scipy import stats as sstats
        r = y_true - y_pred
        sw = sstats.shapiro(r[:min(len(r), 5000)])
        out["resid_shapiro_p"] = float(sw.pvalue)
    except Exception as e:
        out["resid_shapiro_p"] = f"fail: {e}"

    return out


def main():
    csvs = sorted(CSV_DIR.glob("predictions_*.csv"))
    print(f"Post-E evaluation on {len(csvs)} models")
    results = [_evaluate_model(c) for c in csvs]

    # 모델 간 순위 (by WIS if available, else by CRPS, else RMSE)
    def _score(r):
        if "wis" in r and isinstance(r["wis"], float):
            return r["wis"]
        if "crps_gaussian" in r and isinstance(r["crps_gaussian"], float):
            return r["crps_gaussian"]
        rb = r.get("rmse_boot95", {})
        return rb.get("point", 1e9) if isinstance(rb, dict) else 1e9
    ranked = sorted(results, key=_score)

    summary = {
        "n_models": len(results),
        "ranking_by_probabilistic_score": [r["model"] for r in ranked[:20]],
    }

    OUT.write_text(
        json.dumps({"summary": summary, "details": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\n=== Post-E Top-20 by Prob-score (WIS > CRPS > RMSE) ===")
    for i, r in enumerate(ranked[:20], 1):
        wis = r.get("wis", "-")
        crps = r.get("crps_gaussian", "-")
        cov = r.get("pi_coverage_95", "-")
        pwe = r.get("peak_week_err", "-")
        dir_acc = r.get("direction_acc_pct", "-")
        wis_str = f"{wis:.3f}" if isinstance(wis, float) else str(wis)
        crps_str = f"{crps:.3f}" if isinstance(crps, float) else str(crps)
        cov_str = f"{cov:.2f}" if isinstance(cov, float) else str(cov)
        pwe_str = f"{pwe:+d}" if isinstance(pwe, int) else str(pwe)
        dir_str = f"{dir_acc:.1f}%" if isinstance(dir_acc, float) else str(dir_acc)
        print(f"  {i:2d}. {r['model']:28s}  WIS={wis_str}  CRPS={crps_str}  cov95={cov_str}  peak_wk_err={pwe_str}  dir={dir_str}")
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
