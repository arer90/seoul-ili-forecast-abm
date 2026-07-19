"""
역학/의학 타당성 종합 검사 (existing predictions CSV 기반).

Scenario E 에 본격 진입하기 전의 사전 검증:
 - magnitude range (ILI rate [0, 100])
 - peak timing (Dec-Feb Seoul flu season)
 - seasonality (lag-52 autocorrelation)
 - coverage (simple residual bound)
 - Rt proxy (서로 인접한 week 의 smoothed ratio)
 - regime-split R²/MAPE (pre-COVID/during/post)

입력: simulation/results/csv/predictions_<model>.csv  (split, idx, y_true, y_pred)
출력: simulation/results/validity_pre_E.json  +  summary stdout
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV_DIR = ROOT / "simulation" / "results" / "csv"
DEFAULT_OUT = ROOT / "simulation" / "results" / "validity_pre_E.json"


def _lag_autocorr(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag + 3:
        return float("nan")
    a = x[:-lag]
    b = x[lag:]
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _rt_proxy(y: np.ndarray, window: int = 5) -> np.ndarray:
    """생 단순 Rt proxy: 이동평균의 주간 비율 (influenza 주기 4-7일)."""
    if len(y) < window + 1:
        return np.array([])
    kernel = np.ones(window) / window
    sm = np.convolve(y, kernel, mode="valid")
    sm = np.where(sm < 1e-3, 1e-3, sm)
    rt = sm[1:] / sm[:-1]
    return rt


def _peak_weeks(y: np.ndarray, k: int = 3) -> list[int]:
    """상위 k 주 (오름차순 week-index)."""
    if len(y) == 0:
        return []
    idx = np.argsort(y)[-k:]
    return sorted(int(i) for i in idx)


def _regime_metrics(df: pd.DataFrame) -> dict:
    """regime split 은 idx 기반 proxy — phase1 주차를 모르므로
    test split 의 앞/중/뒤 3분할로 대체 (~2020, 2021-22, 2023-)."""
    out = {}
    test = df[df["split"] == "test"].sort_values("idx")
    if len(test) < 9:
        return {"skipped": True, "reason": "too few test weeks"}
    n = len(test)
    t1 = test.iloc[: n // 3]
    t2 = test.iloc[n // 3 : 2 * n // 3]
    t3 = test.iloc[2 * n // 3 :]
    for tag, chunk in [("regime_a", t1), ("regime_b", t2), ("regime_c", t3)]:
        yt = chunk["y_true"].to_numpy()
        yp = chunk["y_pred"].to_numpy()
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        mape = float(np.mean(np.abs((yt - yp) / np.clip(yt, 1e-3, None))) * 100)
        out[tag] = {"n": int(len(chunk)), "rmse": rmse, "mape": mape}
    return out


def _check_model(csv: Path) -> dict:
    name = csv.stem.replace("predictions_", "")
    try:
        df = pd.read_csv(csv)
    except Exception as e:
        return {"model": name, "error": f"read_fail: {e}"}

    if "y_true" not in df.columns or "y_pred" not in df.columns:
        return {"model": name, "error": "no y_true/y_pred columns"}

    r: dict = {"model": name, "n": int(len(df))}
    y_true = df["y_true"].to_numpy(dtype=float)
    y_pred = df["y_pred"].to_numpy(dtype=float)

    # 1. Range sanity (ILI rate [0, 100])
    r["pred_min"] = float(np.min(y_pred))
    r["pred_max"] = float(np.max(y_pred))
    r["pred_mean"] = float(np.mean(y_pred))
    r["nan_count"] = int(np.isnan(y_pred).sum())
    r["neg_count"] = int(np.sum(y_pred < 0))
    r["oob_high_count"] = int(np.sum(y_pred > 100))  # ILI rate 100% 이상은 비현실

    # 2. Peak timing alignment
    r["true_top3_weeks"] = _peak_weeks(y_true, 3)
    r["pred_top3_weeks"] = _peak_weeks(y_pred, 3)
    overlap = len(set(r["true_top3_weeks"]) & set(r["pred_top3_weeks"]))
    r["peak_overlap_top3"] = int(overlap)

    # 3. Seasonality (lag-52 autocorr of predictions)
    r["autocorr_lag52_pred"] = _lag_autocorr(y_pred, 52)
    r["autocorr_lag52_true"] = _lag_autocorr(y_true, 52)

    # 4. Rt proxy range
    rt = _rt_proxy(y_pred, window=5)
    if rt.size > 0:
        r["rt_min"] = float(np.min(rt))
        r["rt_max"] = float(np.max(rt))
        r["rt_out_of_bounds_frac"] = float(
            np.mean((rt < 0.3) | (rt > 8.0))
        )

    # 5. Magnitude ratio (peak true vs pred)
    if np.max(y_true) > 0:
        r["peak_magnitude_ratio_pred_over_true"] = float(
            np.max(y_pred) / max(np.max(y_true), 1e-6)
        )

    # 6. Global metrics
    resid = y_true - y_pred
    r["rmse"] = float(np.sqrt(np.mean(resid ** 2)))
    r["mae"] = float(np.mean(np.abs(resid)))
    r["mape"] = float(np.mean(np.abs(resid / np.clip(y_true, 1e-3, None))) * 100)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r["r2"] = float(1 - ss_res / max(ss_tot, 1e-9))

    # 7. Residual normality (simple)
    if len(resid) > 10:
        r["resid_mean"] = float(np.mean(resid))
        r["resid_std"] = float(np.std(resid))
        r["resid_bias_ratio"] = float(abs(np.mean(resid)) / max(np.std(resid), 1e-6))

    # 8. Regime split
    r["regimes"] = _regime_metrics(df)

    # 9. Simple pass/fail verdict
    fails = []
    if r["neg_count"] > 0:
        fails.append("negative_predictions")
    if r["nan_count"] > 0:
        fails.append("nan_predictions")
    if r.get("pred_max", 0) > 200:
        fails.append("explosive_pred_>200")
    if r.get("peak_magnitude_ratio_pred_over_true", 1.0) > 3.0:
        fails.append("peak_overshoot_>3x")
    if r.get("peak_magnitude_ratio_pred_over_true", 1.0) < 0.3:
        fails.append("peak_undershoot_<0.3x")
    if r.get("rt_out_of_bounds_frac", 0) > 0.2:
        fails.append("rt_out_of_bounds_>20%")
    if not math.isnan(r.get("autocorr_lag52_pred", float("nan"))) and r["autocorr_lag52_pred"] < 0.0:
        fails.append("no_seasonality")
    if r.get("r2", 1.0) < 0.0:
        fails.append("r2_negative")
    if r.get("mape", 0) > 80.0:
        fails.append("mape_>80pct")
    r["fails"] = fails
    r["status"] = "PASS" if not fails else ("WARN" if len(fails) <= 2 else "FAIL")
    return r


def _find_csv_dir(requested: Path) -> Path:
    """If requested has predictions_*.csv use it; else fall back to latest backup."""
    if list(requested.glob("predictions_*.csv")):
        return requested
    print(f"[WARN] no predictions_*.csv in {requested} — searching backups")
    backups = sorted(
        (ROOT / "simulation" / "results").glob("backup_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for bp in backups:
        cand = bp / "csv"
        if cand.exists() and list(cand.glob("predictions_*.csv")):
            print(f"[WARN] using backup {cand}")
            return cand
        for nested in bp.glob("*/csv"):
            if list(nested.glob("predictions_*.csv")):
                print(f"[WARN] using backup {nested}")
                return nested
    raise FileNotFoundError(
        f"no predictions_*.csv in {requested} or any backup_* subtree")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    csv_dir = _find_csv_dir(args.csv_dir)
    global OUT
    OUT = args.out
    csvs = sorted(csv_dir.glob("predictions_*.csv"))
    print(f"Checking {len(csvs)} prediction CSVs from {csv_dir}")
    results = [_check_model(c) for c in csvs]

    # summary
    by_status = {"PASS": [], "WARN": [], "FAIL": [], "ERROR": []}
    for r in results:
        s = "ERROR" if "error" in r else r.get("status", "ERROR")
        by_status[s].append(r["model"])

    summary = {
        "total": len(results),
        "PASS": len(by_status["PASS"]),
        "WARN": len(by_status["WARN"]),
        "FAIL": len(by_status["FAIL"]),
        "ERROR": len(by_status["ERROR"]),
        "pass_list": by_status["PASS"],
        "warn_list": by_status["WARN"],
        "fail_list": by_status["FAIL"],
        "error_list": by_status["ERROR"],
    }

    with OUT.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, indent=2, ensure_ascii=False)

    print(f"\n=== Validity Gate Summary ({OUT.name}) ===")
    for k in ("total", "PASS", "WARN", "FAIL", "ERROR"):
        print(f"  {k}: {summary[k]}")
    print("\nPASS:", ", ".join(summary["pass_list"]))
    print("\nWARN:", ", ".join(summary["warn_list"]))
    print("\nFAIL:", ", ".join(summary["fail_list"]))
    if summary["error_list"]:
        print("\nERROR:", ", ".join(summary["error_list"]))

    # top-5 by R² across all
    ok = [r for r in results if "r2" in r and not math.isnan(r.get("r2", float("nan")))]
    top = sorted(ok, key=lambda x: -x.get("r2", -1))[:10]
    print("\n=== Top-10 by R² ===")
    for r in top:
        peak_ratio = r.get("peak_magnitude_ratio_pred_over_true", float("nan"))
        print(
            f"  {r['model']:28s}  "
            f"R²={r['r2']:+.3f}  MAPE={r['mape']:6.2f}%  "
            f"peak_ratio={peak_ratio:.2f}  status={r.get('status', '?')}"
        )


if __name__ == "__main__":
    main()
