#!/usr/bin/env python3
"""PI 보정법 — TDD 증명 (누설없는 롤링쌍 pi-pairs.json 위에서). 사용자: "TDD로 증명해봐."

증명한 사실 (실측):
  - relative(예측수준 비례) conformal 은 coverage 를 개선하지 못한다 — 오히려 나빠짐(가설 기각).
  - 현행 additive 는 이미 ~0.90 (목표 0.95 근접).
  - per-regime(저/고 분리)은 같은 coverage 에 더 좁은 폭 → 정밀도 win.
정직: 비싼 롤링쌍은 pi_calibration_compare.py 가 생성(pi-pairs.json). 이 테스트는 그 위에서
순수 band 함수로 빠르게 증명(결정적).

Run:  .venv/bin/python web/scripts/test_pi_calibration.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGG = ROOT / "web" / "public" / "aggregates"
sys.path.insert(0, str(ROOT / "web" / "scripts"))

from pi_calibration_compare import evaluate  # noqa: E402


def _pairs():
    f = AGG / "pi-pairs.json"
    assert f.is_file(), "pi-pairs.json 없음 — pi_calibration_compare.py 먼저 실행"
    return [(p, a) for p, a in json.loads(f.read_text(encoding="utf-8"))["pairs"]]


_R = evaluate(_pairs())


def test_relative_does_not_improve_coverage():
    """relative conformal 이 additive 보다 coverage 를 개선하지 못한다(가설 기각, 실측)."""
    assert _R["relative"]["coverage"] <= _R["additive(현행)"]["coverage"] + 0.01, (
        f"relative {_R['relative']['coverage']} 가 additive {_R['additive(현행)']['coverage']} 보다 "
        "유의 개선 — 가설 기각이 틀렸으니 재검토")


def test_additive_coverage_near_target():
    """현행 additive 는 이미 목표 0.95 근처(0.85~0.97) — 84% 위기 아님."""
    c = _R["additive(현행)"]["coverage"]
    assert 0.85 <= c <= 0.97, f"additive coverage {c} 가 기대범위 밖"


def test_per_regime_tighter_at_equal_coverage():
    """per-regime 은 additive 와 같은(±) coverage 에 더 좁은 폭 = 정밀도 win."""
    a, r = _R["additive(현행)"], _R["per-regime"]
    assert r["coverage"] >= a["coverage"] - 0.02, f"per-regime coverage {r['coverage']} < additive"
    assert r["mean_width"] < a["mean_width"], (
        f"per-regime 폭 {r['mean_width']} ≥ additive {a['mean_width']} — 정밀도 win 없음")


def test_production_conformal_is_regime_aware():
    """production _conformal_half_width 가 배선됨: 저regime q < 전체 < 고regime q (정밀/적절폭)."""
    sys.path.insert(0, str(ROOT / "web" / "scripts"))
    from build_production_forecast import _conformal_half_width
    q_lo = _conformal_half_width(0.05, pred_level=6.0)
    q_glob = _conformal_half_width(0.05)
    q_hi = _conformal_half_width(0.05, pred_level=60.0)
    assert q_lo < q_glob < q_hi, f"regime q 정렬 위반: 저 {q_lo:.1f} / 전체 {q_glob:.1f} / 고 {q_hi:.1f}"


def test_leakage_free_calibration():
    """보정이 각 origin 의 과거 잔차만 쓴다(미래 정보 0) — evaluate() cal=pairs[i-w:i] 구조 박제."""
    import inspect
    from pi_calibration_compare import evaluate as ev
    src = inspect.getsource(ev)
    assert "pairs[i - cal_w:i]" in src, "보정창이 과거쌍만 쓰는 구조가 아님(누설 위험)"


if __name__ == "__main__":
    print("  " + " · ".join(f"{k}:{v['coverage']}/{v['mean_width']}" for k, v in _R.items()))
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    p = f = 0
    for fn in funcs:
        try:
            fn(); print(f"  ✓ PASS  {fn.__name__}"); p += 1
        except Exception as e:
            print(f"  ✗ FAIL  {fn.__name__}: {e}"); f += 1
    print(f"\n  {p} PASS / {f} FAIL")
    sys.exit(1 if f else 0)
