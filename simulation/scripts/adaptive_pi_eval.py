"""adaptive_pi_eval.py — 전 모델 adaptive conformal PI post-hoc 평가 (G-365b, 2026-06-26).

R10 generic split-conformal PI 가 전 모델 과소피복(중위 0.67) — in-sample 잔차가 out-of-sample
정점 오차 과소추정. adaptive conformal(Conformal-PID)을 *전 모델*에 적용하면 정점 분포이동서
구간 자동확장 → 커버리지 회복(FusedEpi 0.926 처럼). R10 resume 가 test_preds 유실로 skip 되므로,
저장된 예측 CSV + 잔차로 **재실행 없이** post-hoc 적용한다(MPH_ADAPTIVE_CONFORMAL 배선 G-365 와 동일 로직).

입력: simulation/results/csv/predictions_<model>.csv(test 슬랩) + per_model_optimal/<model>.json
  (val_metrics.insample_residuals = leak-free 보정 잔차). 출력: csv/adaptive_pi_metrics.csv —
  모델별 static vs adaptive PI95/80/50 커버리지 + adaptive WIS. leak-free(rolling 과거 obs만).

Usage: .venv/bin/python -m simulation.scripts.adaptive_pi_eval
Returns: csv 경로(print). Side effects: csv 작성. (모델 로드 없음 = 가벼움.)
"""
from __future__ import annotations

import csv
import glob
import json
import os
import warnings

import numpy as np

warnings.filterwarnings("ignore")


def _residuals(name: str):
    """per_model_optimal/<name>.json 의 leak-free in-sample 잔차 (없으면 None)."""
    f = f"simulation/results/per_model_optimal/{name}.json"
    if not os.path.exists(f):
        return None
    try:
        d = json.load(open(f))
        r = (d.get("val_metrics", {}) or {}).get("insample_residuals")
        if r is None:
            return None
        a = np.asarray(r, dtype=np.float64)
        a = a[np.isfinite(a)]
        return a if len(a) >= 2 else None
    except Exception:
        return None


def evaluate() -> list[dict]:
    """전 예측 CSV × 잔차 → static vs adaptive conformal 커버리지. Returns rows."""
    import pandas as pd
    from simulation.analytics.hub_metrics import (
        FLUSIGHT_ALPHAS, k11_pi_widths_from_residuals,
    )
    from simulation.analytics.adaptive_conformal import (
        adaptive_conformal_bounds, wis_from_bounds,
    )
    pairs = {"pi95": 0.05, "pi80": 0.20, "pi50": 0.50}
    rows: list[dict] = []
    for pf in sorted(glob.glob("simulation/results/csv/predictions_*.csv")):
        name = os.path.basename(pf)[len("predictions_"):-4]
        try:
            df = pd.read_csv(pf)
            t = df[df["split"] == "test"]
            if len(t) < 10:
                continue
            y = t["y_true"].values.astype(np.float64)
            pred = t["y_pred"].values.astype(np.float64)
            res = _residuals(name)
            if res is None:
                continue
            k11 = k11_pi_widths_from_residuals(np.abs(res), FLUSIGHT_ALPHAS)
            b = adaptive_conformal_bounds(pred, k11, res, y, FLUSIGHT_ALPHAS)
            row = {"model": name, "n_test": len(y)}
            for tag, a in pairs.items():
                q = k11.get(a)
                if q is not None and np.isfinite(q):
                    row[f"static_{tag}"] = round(float(np.mean((y >= pred - q) & (y <= pred + q))), 3)
                else:
                    row[f"static_{tag}"] = float("nan")
                if a in b:
                    lo, hi = b[a]
                    row[f"adapt_{tag}"] = round(float(np.mean((y >= lo) & (y <= hi))), 3)
                else:
                    row[f"adapt_{tag}"] = float("nan")
            row["adapt_wis"] = round(float(np.mean(wis_from_bounds(y, b, FLUSIGHT_ALPHAS, median=pred))), 3)
            rows.append(row)
        except Exception as e:
            print(f"  {name}: skip ({type(e).__name__}: {str(e)[:50]})", flush=True)
    return rows


def main() -> int:
    rows = evaluate()
    out = "simulation/results/csv/adaptive_pi_metrics.csv"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cols = ["model", "static_pi95", "adapt_pi95", "static_pi80", "adapt_pi80",
            "static_pi50", "adapt_pi50", "adapt_wis", "n_test"]
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})
    # 요약
    if rows:
        sa = np.array([r["static_pi95"] for r in rows if np.isfinite(r.get("static_pi95", np.nan))])
        ad = np.array([r["adapt_pi95"] for r in rows if np.isfinite(r.get("adapt_pi95", np.nan))])
        print(f"\n  PI95 coverage 중위: static {np.median(sa):.3f} → adaptive {np.median(ad):.3f}", flush=True)
        print(f"  adaptive ≥0.90 모델: {int((ad >= 0.90).sum())}/{len(ad)} (static {int((sa >= 0.90).sum())}/{len(sa)})", flush=True)
    print(f"  {out} ({len(rows)} models)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
