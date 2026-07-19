#!/usr/bin/env python3
"""WEB production champion → 2025-12 기준 → 2026 recursive roll-forward → vs REAL.

웹의 실제 production forecast(build_production_forecast.py)와 **동일한 메커니즘**을 그대로
재사용한다:
  ① 전체(train+val+test) in-sample refit — hold-out 없음, 배포 그대로 (_refit_negbin_glm)
  ② synthetic-future-row 1-step + gate (climatology, oracle 누설 없음) (_build_future_row,
     _gate_forecast)

다른 점은 단 하나 — 기준점(anchor)을 2025-12 로 truncate 해서, 거기서부터 2026-01 …
forward 로 **recursive 1-step roll-forward**(각 주 예측이 다음 주의 lag1 이 됨) 하고, 매 주
예측을 REAL ILI 와 대조한다. (실측이 2026-05 까지뿐이라 그 이후는 '미래'.)

→ 사용자 요구 그대로: "train,val,test 다 학습 → 2025-12 기준 → 2026-01 부터 real 예측."
   이전 prospective_horizon_decay 는 direct multi-horizon(수평선별 별도 모델)이라 production
   메커니즘이 아니었음. 이 스크립트가 진짜 웹-production 을 Dec-2025 기준으로 돌린 것.

Read-only. Run: .venv/bin/python web/scripts/production_rollforward_2026.py [ANCHOR] [WEEKS]
"""
from __future__ import annotations

import datetime
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web" / "scripts"))
logging.disable(logging.INFO)  # silence per-step production INFO logs

from build_production_forecast import (  # noqa: E402
    _load_feature_matrix,
    _extract_basic_features,
    _refit_negbin_glm,
    _build_future_row,
    _gate_forecast,
)

# 사용자가 나열한 수평선 (주 단위) — 실측=주간이라 +1d~+6d 는 +1주로 대표
MARKERS = {1: "+1주", 2: "+2주", 3: "+3주", 4: "+1개월", 13: "+3개월",
           26: "+6개월", 39: "+9개월", 52: "+12개월"}


def _d(w) -> datetime.date:
    if hasattr(w, "date"):
        return w.date()
    if isinstance(w, datetime.date):
        return w
    return datetime.date.fromisoformat(str(w)[:10])


def main() -> None:
    anchor = (datetime.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1
              else datetime.date(2025, 12, 31))
    weeks = int(sys.argv[2]) if len(sys.argv) > 2 else 52

    X_all, y_all, fcols, ws = _load_feature_matrix()
    _Xb, bcols, bidx = _extract_basic_features(X_all, fcols)
    dates = [_d(w) for w in ws]
    real_by_date = {dates[i]: float(y_all[i]) for i in range(len(dates))}
    last_real = dates[-1]

    def find_real(d: datetime.date):
        for off in (0, -1, 1, -2, 2, -3, 3):
            if (d + datetime.timedelta(days=off)) in real_by_date:
                return real_by_date[d + datetime.timedelta(days=off)]
        return None

    # truncate at anchor: train on ALL rows ≤ anchor (no hold-out — like production)
    keep = [i for i in range(len(dates)) if dates[i] <= anchor]
    if len(keep) < 30:
        print(f"insufficient training ≤ {anchor}")
        return
    a = keep[-1]
    roll_X = X_all[: a + 1].astype(float).copy()
    roll_y = y_all[: a + 1].astype(float).copy()
    roll_ws = [dates[i] for i in keep]
    y_train = roll_y.copy()  # fixed history for gate caps

    model = _refit_negbin_glm(roll_X[:, bidx], roll_y)

    print(f"=== WEB production (전체 refit) → 기준 {dates[a]} → 2026 recursive roll → REAL ===")
    print(f"  학습: train+val+test 전체 ≤기준 = {a + 1}주 (hold-out 없음, 배포 그대로)")
    print(f"  메커니즘: build_production_forecast 그대로 (refit + 1-step synthetic row + gate),")
    print(f"           기준점만 {dates[a]} → 매주 예측을 다음 lag 로 먹여 forward roll")
    print(f"  실측 ILI …{last_real} 까지 → 그 이후 목표주는 '미래(실측 대기)'\n")
    print(f"  {'예측주':<13}{'수평선':>7}{'예측':>9}{'실측':>9}{'오차':>9}  상태")
    print(f"  {'-' * 56}")

    traj, val_err = [], []
    for step in range(1, weeks + 1):
        frow, fdate, _fy, _fw = _build_future_row(roll_X, roll_y, fcols, bcols, bidx, roll_ws)
        raw = model.predict(frow)
        gate = _gate_forecast(raw, y_train, fallback=float(roll_y[-1]), k=3.0)
        pred = float(gate["pred"][0])

        real = find_real(fdate) if fdate <= last_real else None
        mark = MARKERS.get(step, "")
        if real is not None:
            err = pred - real
            val_err.append(abs(err))
            flag = "✓ 실측" + ("  ◀" + mark if mark else "")
            print(f"  {str(fdate):<13}{mark:>7}{pred:>9.1f}{real:>9.1f}{err:>+9.1f}  {flag}")
        elif mark:  # only print future rows at the user's markers (keep compact)
            print(f"  {str(fdate):<13}{mark:>7}{pred:>9.1f}{'—':>9}{'—':>9}  미래(실측대기)")
        traj.append(dict(week=str(fdate), step=step, marker=mark or None,
                         predicted=round(pred, 1),
                         actual=round(real, 1) if real is not None else None,
                         error=round(pred - real, 1) if real is not None else None))

        # extend rolling state (prediction → next lag1)
        new_row = roll_X[-1].copy()
        for j, idx in enumerate(bidx):
            new_row[idx] = frow[0, j]
        roll_X = np.vstack([roll_X, new_row])
        roll_y = np.append(roll_y, pred)
        roll_ws = roll_ws + [fdate]

    if val_err:
        n_val = len(val_err)
        print(f"\n  검증(실측 존재) {n_val}주 — 기준+1주 … {[t['week'] for t in traj if t['actual'] is not None][-1]}:")
        print(f"    MAE {np.mean(val_err):.2f}/1k")
        print(f"  미래(실측 미존재): 기준+{n_val + 1}주 이후 (실측 {last_real}까지뿐)")

    out = {
        "source": "production-rollforward",
        "mechanism": "build_production_forecast (full refit + 1-step synthetic row + gate), "
                     "anchored at Dec-2025, recursive roll-forward",
        "anchor": str(dates[a]),
        "n_train": a + 1,
        "real_until": str(last_real),
        "trajectory": traj,
    }
    (ROOT / "web" / "public" / "aggregates" / "production-rollforward-2026.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  → wrote web/public/aggregates/production-rollforward-2026.json")


if __name__ == "__main__":
    main()
