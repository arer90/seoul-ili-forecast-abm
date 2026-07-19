#!/usr/bin/env python3
"""PRODUCTION forecast → 미래로 발사 → 실측 대조 (수평선별 +1주 … +12개월).

사용자 요구 그대로:
  "2025-12 까지 학습한 production 모델을, 2026 1~12월 미래로 예측 → 실측과 비교."
  train/test 분할(여러 origin·held-out)이 아니라, **단일 production origin 1개**에서
  미래로 쏜 forecast 궤적을 REAL ILI 와 수평선별로 대조한다.

각 수평선 h(주)의 예측 = origin 시점 feature 로 'h주 뒤 ILI' 를 direct 예측
(champion NegBinGLM V6 를 (X[i] → y[i+h]) 로 적합, 학습 페어는 origin·target 둘 다
origin 이전 = 미래 누설 0). 프로덕션 gate(≤3×train_max) 적용 = 배포 forecast 기준.

정직성:
  - 실측 ILI = 주간(KDCA 표본감시) → +1d~+6d 일별 ground truth 없음, +1주가 최소 단위.
  - 실측은 ~2026-05 까지만 존재 → origin+h 가 그 뒤면 '미래(실측 대기)' 로 표기.
    +12개월을 실측 검증하려면 origin 이 2025-05 이전이어야 함(2025-05 origin 으로도 실행 가능).

Read-only. Run: .venv/bin/python web/scripts/prospective_horizon_decay.py [ORIGIN_YYYY-MM-DD]
"""
from __future__ import annotations

import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web" / "scripts"))

from build_production_forecast import _load_feature_matrix, _extract_basic_features  # noqa: E402
from simulation.models.epi_models import NegBinGLMForecaster  # noqa: E402

HORIZONS = [
    ("+1주 (+1~7일)", 1),
    ("+2주 (+14일)", 2),
    ("+3주 (+21일)", 3),
    ("+1개월", 4),
    ("+3개월", 13),
    ("+6개월", 26),
    ("+9개월", 39),
    ("+12개월", 52),
]


def _as_date(w) -> date:
    if isinstance(w, datetime):
        return w.date()
    if isinstance(w, date):
        return w
    return datetime.fromisoformat(str(w)[:10]).date()


def main() -> None:
    origin = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2025, 12, 15)

    X_all, y_all, feature_cols, week_starts = _load_feature_matrix()
    X, _cols, _ = _extract_basic_features(X_all, feature_cols)
    y_all = np.asarray(y_all, dtype=float)
    dates = [_as_date(w) for w in week_starts]
    n = len(dates)
    didx = {d: i for i, d in enumerate(dates)}

    def find(d: date):
        for off in (0, -1, 1, -2, 2, -3, 3, -4, 4):
            j = didx.get(d + timedelta(days=off))
            if j is not None:
                return j
        return None

    origin_i = max((i for i in range(n) if dates[i] <= origin), default=-1)
    if origin_i < 30:
        print(f"insufficient training before {origin} (got {origin_i + 1})")
        return
    last_real = dates[-1]

    print(f"=== PRODUCTION forecast → 미래 → 실측 대조 (NegBinGLM, 학습 ≤ {origin}) ===")
    print(f"  origin = {dates[origin_i]} (그 주 실측 ILI {y_all[origin_i]:.1f}/1k)  ·  학습표본 {origin_i + 1}주")
    print(f"  실측 ILI 존재구간 …{last_real}  → 그 이후 목표일은 '미래(실측 대기)'\n")
    print(f"  {'수평선':<14}{'목표일':<13}{'예측':>8}{'실측':>8}{'오차':>8}   상태")
    print(f"  {'-' * 62}")

    rows = []
    for label, h in HORIZONS:
        tdate = dates[origin_i] + timedelta(weeks=h)
        # direct model for this lead: (X[i] → y[i+h]) pairs, both ≤ origin
        Xtr, ytr = [], []
        for i in range(origin_i + 1):
            j = find(dates[i] + timedelta(weeks=h))
            if j is not None and j <= origin_i:
                Xtr.append(X[i]); ytr.append(y_all[j])
        if len(Xtr) < 30:
            print(f"  {label:<14}{str(tdate):<13}{'—':>8}{'—':>8}{'—':>8}   학습부족({len(Xtr)})")
            continue
        Xtr_a, ytr_a = np.asarray(Xtr), np.asarray(ytr)
        model = NegBinGLMForecaster(topk=20)
        model.fit(Xtr_a, ytr_a)
        tmax = float(np.nanmax(ytr_a))
        fc = float(np.clip(np.nan_to_num(model.predict(X[origin_i : origin_i + 1])[0],
                                          nan=0.0, posinf=3 * tmax, neginf=0.0), 0.0, 3.0 * tmax))

        j = find(tdate)
        real = float(y_all[j]) if (j is not None and j > origin_i and dates[j] <= last_real) else None
        if real is not None:
            err = fc - real
            print(f"  {label:<14}{str(tdate):<13}{fc:>8.1f}{real:>8.1f}{err:>+8.1f}   ✓ 실측")
            rows.append(dict(label=label, weeks=h, target=str(tdate),
                             predicted=round(fc, 1), actual=round(real, 1),
                             error=round(err, 1)))
        else:
            print(f"  {label:<14}{str(tdate):<13}{fc:>8.1f}{'—':>8}{'—':>8}   미래(실측 대기)")
            rows.append(dict(label=label, weeks=h, target=str(tdate),
                             predicted=round(fc, 1), actual=None, error=None))

    val = [r for r in rows if r["actual"] is not None]
    if val:
        mae = float(np.mean([abs(r["error"]) for r in val]))
        bias = float(np.mean([r["error"] for r in val]))
        print(f"\n  검증된 수평선 {len(val)}개 (origin+1주 … origin+{val[-1]['weeks']}주, "
              f"~{dates[origin_i] + timedelta(weeks=val[-1]['weeks'])}):")
        print(f"    MAE {mae:.2f}/1k · 평균편향(pred−real) {bias:+.2f}")
        fut = [r for r in rows if r["actual"] is None]
        if fut:
            print(f"  미검증(미래) {len(fut)}개: {', '.join(r['label'] for r in fut)} "
                  f"— 실측이 {last_real}까지만 존재")

    out = {
        "source": "production-horizon-forecast",
        "origin": str(dates[origin_i]),
        "origin_ili": round(float(y_all[origin_i]), 1),
        "n_train": origin_i + 1,
        "real_until": str(last_real),
        "note": "단일 production origin → 미래 direct multi-horizon forecast vs REAL. "
                "실측=주간이라 +1d~+6d 는 +1주로 대표. 실측 2026-05까지만 → 그 이후는 미래.",
        "horizons": rows,
    }
    (ROOT / "web" / "public" / "aggregates" / "production-horizon-forecast.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  → wrote web/public/aggregates/production-horizon-forecast.json")


if __name__ == "__main__":
    main()
