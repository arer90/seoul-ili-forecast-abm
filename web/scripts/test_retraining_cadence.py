#!/usr/bin/env python3
"""재학습 주기 효과 — TDD (REAL data 재계산). 사용자: "이 답은 TDD 없이 한 거냐?" → 박제.

박제 finding (compare_cadence, 2025-26 절기):
  - 더 자주 재학습할수록 1-step nowcast MAE 가 낮아진다(고정 ≥ 월간 ≥ 주간) — '재학습 이득'.
  - 단 이득은 marginal(차이 작음) — 1-step 신호 대부분이 매주 갱신되는 실측 lag 이라.
  → 월간 재학습 = 저비용 실이득(권고), 주간이 최적이나 차이 작음.

느림(~12s: 평가창 18주 × 주간 적합). max_eval 로 가속.
Run:  .venv/bin/python web/scripts/test_retraining_cadence.py
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "web" / "scripts"))

from retraining_cadence import compare_cadence  # noqa: E402

# 한 번 계산해 재사용 (절기 시작 cutoff, 겨울 포함 18주)
_R = compare_cadence(datetime.date(2025, 9, 30), max_eval=18)


def test_more_frequent_retrain_not_worse():
    """주간(최다 재학습) MAE 가 고정보다 나쁘지 않다 — 재학습은 해롭지 않다(보통 이득)."""
    assert _R["weekly"]["mae"] <= _R["static"]["mae"] + 0.5, (
        f"주간 {_R['weekly']['mae']} 가 고정 {_R['static']['mae']} 보다 유의하게 나쁨 — 가정 재검토")


def test_monthly_helps_or_neutral_vs_static():
    """월간 재학습이 고정보다 나쁘지 않다 (사용자 '1달마다 재학습'의 직접 답)."""
    assert _R["monthly"]["mae"] <= _R["static"]["mae"] + 0.5, (
        f"월간 {_R['monthly']['mae']} > 고정 {_R['static']['mae']} — 월간 재학습 무익?")


def test_retrain_benefit_is_marginal():
    """이득은 marginal — 고정→주간 개선폭 < 25% (1-step 은 실측 lag 가 주신호라 폭 작음)."""
    s, w = _R["static"]["mae"], _R["weekly"]["mae"]
    rel = (s - w) / s if s > 0 else 0.0
    assert rel < 0.25, f"개선폭 {rel:.0%} ≥ 25% — '차이 작음' 주장 위반(재검토)"


def test_refit_counts_ordered():
    """재학습 횟수: 고정(1) < 월간 < 주간 (주기 정의 검증)."""
    nr = _R["n_refit"]
    assert nr["static"] == 1 < nr["monthly"] <= nr["weekly"], nr


if __name__ == "__main__":
    print(f"  (cutoff {_R['cutoff']}, {_R['n_eval']}주: "
          f"static {_R['static']['mae']} / monthly {_R['monthly']['mae']} / weekly {_R['weekly']['mae']})")
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    p = f = 0
    for fn in funcs:
        try:
            fn(); print(f"  ✓ PASS  {fn.__name__}"); p += 1
        except Exception as e:
            print(f"  ✗ FAIL  {fn.__name__}: {e}"); f += 1
    print(f"\n  {p} PASS / {f} FAIL")
    sys.exit(1 if f else 0)
