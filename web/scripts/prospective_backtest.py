#!/usr/bin/env python3
"""TRUE prospective backtest — train to a cutoff, forecast FORWARD, validate vs REAL.

Answers the user's correct objection: the existing backtest "test" is an in-sample
holdout (the model saw that span) and the SEIR hindcast used 2019 (OOD past).  The
honest test is: cut off training BEFORE the 2025-26 winter, refit the champion, forecast
the winter forward, and score against the REAL 2025-26 ILI (which exists in the DB up to
~May 2026).

NegBinGLM (V6 RidgeCV+log1p) refit only — single model, fast, no 53-model pipeline.
1-step rolling (each forecast week uses the real prior-week lag, known at forecast time).

Read-only on the feature cache; prints the prospective accuracy.  Run:
    .venv/bin/python web/scripts/prospective_backtest.py [CUTOFF_YYYY-MM-DD]
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web" / "scripts"))

from build_production_forecast import _load_feature_matrix, _extract_basic_features  # noqa: E402
from simulation.models.epi_models import NegBinGLMForecaster  # noqa: E402


def _as_date(w) -> date:
    if isinstance(w, datetime):
        return w.date()
    if isinstance(w, date):
        return w
    return datetime.fromisoformat(str(w)[:10]).date()


def main() -> None:
    cutoff = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2025, 10, 15)

    X_all, y_all, feature_cols, week_starts = _load_feature_matrix()
    X_basic, cols, _ = _extract_basic_features(X_all, feature_cols)
    ws = [_as_date(w) for w in week_starts]

    train = [i for i in range(len(ws)) if ws[i] <= cutoff]
    test = [i for i in range(len(ws)) if ws[i] > cutoff]
    if len(train) < 30 or not test:
        print(f"insufficient split: train={len(train)} test={len(test)} (cutoff {cutoff})")
        return

    Xtr, ytr = X_basic[train], y_all[train]
    Xte, yte = X_basic[test], y_all[test]
    test_dates = [ws[i] for i in test]

    model = NegBinGLMForecaster(topk=20)
    model.fit(Xtr, ytr)
    pred_raw = np.asarray(model.predict(Xte), dtype=float).ravel()

    # Production gate (mirrors real_eval/_gate_forecast): clamp to ≤3× train max AND a
    # per-step |Δ| ceiling (q99.5 of historical week-over-week change) so a single
    # extrapolation spike can't blow up the deployed forecast.  This is what the web shows.
    train_max = float(np.nanmax(ytr))
    dcap = float(np.quantile(np.abs(np.diff(ytr[np.isfinite(ytr)])), 0.995))
    pred = np.clip(np.nan_to_num(pred_raw, nan=0.0, posinf=3 * train_max, neginf=0.0),
                   0.0, 3.0 * train_max)
    last_real = ytr[-1]
    for i in range(len(pred)):
        anchor = yte[i - 1] if i > 0 else last_real     # 1-step: prior real is known
        pred[i] = float(np.clip(pred[i], anchor - dcap, anchor + dcap))

    n = len(yte)
    err = pred - yte
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    ss = float(np.sum((yte - yte.mean()) ** 2))
    r2 = 1.0 - float(np.sum(err ** 2)) / ss if ss > 0 else float("nan")

    print(f"=== TRUE PROSPECTIVE backtest — NegBinGLM (cutoff {cutoff}) ===")
    print(f"  train n={len(train)} (≤{cutoff})  ·  forecast n={n} ({test_dates[0]} → {test_dates[-1]})")
    print(f"  out-of-sample: 학습이 forecast 구간을 못 봄 (진짜 전향적)\n")
    print(f"  MAE {mae:.2f} · RMSE {rmse:.2f} · R² {r2:.3f} · 편향(pred−actual) {bias:+.2f}")

    # peak stratum
    thr = float(np.quantile(yte, 0.75))
    hi = [i for i in range(n) if yte[i] >= thr]
    if hi:
        pm = float(np.mean([abs(err[i]) for i in hi]))
        pb = float(np.mean([err[i] for i in hi]))
        print(f"  피크(상위25%, ILI≥{thr:.0f}, n={len(hi)}): MAE {pm:.2f} · 편향 {pb:+.2f}")

    pk_a = int(np.argmax(yte))
    pk_p = int(np.argmax(pred))
    print(f"\n  실측 피크: {yte[pk_a]:.1f}/1k @ {test_dates[pk_a]}")
    print(f"  예측 피크: {pred[pk_p]:.1f}/1k @ {test_dates[pk_p]}  (magnitude {pred[pk_a] / yte[pk_a] * 100:.0f}% of true peak week)")

    # trajectory (every ~3rd week)
    print("\n  궤적 (실측 → 예측):")
    for i in range(0, n, max(1, n // 12)):
        flag = " ←피크" if i == pk_a else ""
        print(f"    {test_dates[i]}: 실측 {yte[i]:6.1f}  예측 {pred[i]:6.1f}  (err {err[i]:+6.1f}){flag}")


if __name__ == "__main__":
    main()
