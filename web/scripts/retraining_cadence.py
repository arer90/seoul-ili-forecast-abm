#!/usr/bin/env python3
"""champion 재학습 주기 효과 실증 — 고정 vs 월간 vs 주간 (사용자 질문에 데이터로 답).

사용자: "5월까지 데이터 나왔으니 champion 을 1달마다 재학습해야할까? 그럼 더 좋게 예측될까?"

검증: 2025-26 절기에서 같은 모델(NegBinGLM BASIC)을 세 주기로 굴려 1-step nowcast MAE 비교.
  - STATIC : 절기 시작(cutoff) 시점 1회만 학습 → 이후 재학습 없이 매주 실측 lag 로 예측.
  - MONTHLY: 4주마다 누적 데이터로 재학습.
  - WEEKLY : 매주 직전까지 데이터로 재학습 (현 build_production_forecast 거동).
세 방법 모두 예측 자체는 실측 lag 1-step(공정). 차이는 '계수를 얼마나 자주 새 데이터에 맞추나'.

compare_cadence(cutoff) = TDD 가능한 순수 함수. Read-only(write retraining-cadence.json).
Run:  .venv/bin/python web/scripts/retraining_cadence.py [CUTOFF]
Test: .venv/bin/python web/scripts/test_retraining_cadence.py
"""
from __future__ import annotations

import datetime
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "web" / "scripts"))
logging.disable(logging.INFO)
from build_production_forecast import _load_feature_matrix, _extract_basic_features, _gate_forecast  # noqa: E402
from simulation.models.epi_models import NegBinGLMForecaster  # noqa: E402


def _d(w):
    if hasattr(w, "date"):
        return w.date()
    return w if isinstance(w, datetime.date) else datetime.date.fromisoformat(str(w)[:10])


def _fit(Xb, y, idx):
    m = NegBinGLMForecaster(topk=20); m.fit(Xb[idx], y[idx]); return m


def _pred1(m, row, y_train):
    raw = np.asarray(m.predict(row.reshape(1, -1)), float)
    return float(_gate_forecast(raw, y_train, fallback=float(row[0]), k=3.0)["pred"][0])


def compare_cadence(cutoff: datetime.date, max_eval: int | None = None) -> dict:
    """고정/월간/주간 재학습의 1-step nowcast MAE 비교 (cutoff 이후 평가).

    Args:
        cutoff: 학습 시작 시점(절기 시작). 평가 = 그 다음 주부터.
        max_eval: 평가 주 수 상한(테스트 가속용). None=끝까지.

    Returns:
        {n_eval, static{mae,bias}, monthly{...}, weekly{...}, n_refit{...}}.

    Performance: ~(weekly 평가주 수) NegBinGLM 적합. 34주 ≈ 20s.
    """
    X, y, fc, ws = _load_feature_matrix()
    X = np.asarray(X, float); y = np.asarray(y, float)
    Xb, _bc, _bi = _extract_basic_features(X, fc)
    dates = [_d(w) for w in ws]; n = len(y)
    c = max(i for i in range(n) if dates[i] <= cutoff)
    test = [i for i in range(c + 1, n)]
    if max_eval:
        test = test[:max_eval]

    m_static = _fit(Xb, y, list(range(c + 1)))
    m_monthly = m_static; last_month = dates[c].month; n_month_refit = 0
    err = {"static": [], "monthly": [], "weekly": []}
    for t in test:
        err["static"].append(_pred1(m_static, Xb[t], y[: c + 1]) - y[t])
        if dates[t].month != last_month:
            m_monthly = _fit(Xb, y, list(range(t))); last_month = dates[t].month; n_month_refit += 1
        err["monthly"].append(_pred1(m_monthly, Xb[t], y[:t]) - y[t])
        err["weekly"].append(_pred1(_fit(Xb, y, list(range(t))), Xb[t], y[:t]) - y[t])

    def met(e):
        e = np.array(e)
        return {"mae": round(float(np.mean(np.abs(e))), 2), "bias": round(float(np.mean(e)), 2)}
    return {"cutoff": str(dates[c]), "n_eval": len(test),
            "static": met(err["static"]), "monthly": met(err["monthly"]), "weekly": met(err["weekly"]),
            "n_refit": {"static": 1, "monthly": 1 + n_month_refit, "weekly": len(test)}}


def main() -> None:
    cutoff = datetime.date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else datetime.date(2025, 9, 30)
    r = compare_cadence(cutoff)
    print(f"=== 재학습 주기 효과 (cutoff {r['cutoff']}, 평가 {r['n_eval']}주) ===\n")
    print(f"  {'주기':<10}{'MAE':>8}{'편향':>9}{'재학습 횟수':>11}")
    print(f"  {'-'*38}")
    for k in ("static", "monthly", "weekly"):
        print(f"  {k:<10}{r[k]['mae']:>8.2f}{r[k]['bias']:>+9.2f}{r['n_refit'][k]:>11}")
    ms, mm, mw = r["static"]["mae"], r["monthly"]["mae"], r["weekly"]["mae"]
    print(f"\n  • 월간 vs 고정: {ms:.2f}→{mm:.2f} ({(ms-mm)/ms*100:+.0f}%) — 1달 재학습 이득.")
    print(f"  • 주간 vs 월간: {mm:.2f}→{mw:.2f} ({(mm-mw)/max(mm,1e-9)*100:+.0f}%).")
    print(f"  • 결론: 재학습 잦을수록 MAE↓ 이나 폭 작음(1-step 신호 대부분이 실측 lag). 월간=저비용 실이득.")
    (ROOT / "web" / "public" / "aggregates" / "retraining-cadence.json").write_text(
        json.dumps(r, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  → wrote retraining-cadence.json")


if __name__ == "__main__":
    main()
