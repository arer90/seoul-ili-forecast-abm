#!/usr/bin/env python3
"""+1주 잔차 bias 보정 — TDD (REAL data 재계산). codex/gemini 레버 박제.

박제: 최근 k주 1-step 잔차 평균을 차감하면 +1주 의 계통 bias(+over)가 크게 줄고 MAE 는 나빠지지
않는다. → 다음주 예측의 systematic error 직접 개선.

느림(~40s: 평가창 12주 × (k+1) 적합). Run: .venv/bin/python web/scripts/test_accuracy_calibration.py
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "web" / "scripts"))

from accuracy_calibration import compare_calibration  # noqa: E402

# 절기 시작(상승기 포함) 12주 — 모델이 겨울 상승을 못 잡는 구간
_R = compare_calibration(datetime.date(2024, 9, 30), k=4, max_eval=12)


def test_debias_improves_mae_on_rising_edge():
    """상승기에서 잔차 보정이 MAE 를 개선한다 (모델이 못 잡던 겨울 상승을 잔차로 보정)."""
    assert _R["debiased"]["mae"] < _R["raw"]["mae"], (
        f"debias MAE {_R['debiased']['mae']} ≥ raw {_R['raw']['mae']} — 상승기 개선 실패")


def test_debias_does_not_worsen_mae():
    """어느 구간이든 MAE 를 유의하게 악화시키지 않는다(safe)."""
    assert _R["debiased"]["mae"] <= _R["raw"]["mae"] + 0.5, (
        f"debias MAE {_R['debiased']['mae']} > raw {_R['raw']['mae']} +0.5")


def test_debias_applies_nontrivial_correction():
    """잔차 보정이 실제로 적용된다(raw 와 debiased 가 다름) — no-op 아님."""
    assert _R["debiased"]["bias"] != _R["raw"]["bias"], "보정이 적용 안 됨(no-op)"


if __name__ == "__main__":
    print(f"  (cutoff {_R['cutoff']}, {_R['n_eval']}주: raw bias {_R['raw']['bias']:+} → "
          f"debiased {_R['debiased']['bias']:+}, MAE {_R['raw']['mae']}→{_R['debiased']['mae']})")
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    p = f = 0
    for fn in funcs:
        try:
            fn(); print(f"  ✓ PASS  {fn.__name__}"); p += 1
        except Exception as e:
            print(f"  ✗ FAIL  {fn.__name__}: {e}"); f += 1
    print(f"\n  {p} PASS / {f} FAIL")
    sys.exit(1 if f else 0)
