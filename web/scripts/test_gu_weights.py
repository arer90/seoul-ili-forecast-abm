#!/usr/bin/env python3
"""자치구 분배 — 통계검정 게이트 계단식 TDD (REAL DB). 사용자: TDD 평가로 확인, 수치 틀리면 미사용.

박제(실측): endemic 구별 패턴은 순열검정으로 노이즈와 구별 안 됨(p≈0.07, 비유의) → 게이트 미달 →
도시값 균등(구별 분배 안 함)으로 정직하게 강하. 가짜 per-gu 정밀 회피.

느림(~3s: 순열검정 2000). Run: .venv/bin/python web/scripts/test_gu_weights.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "web" / "scripts"))

from build_gu_weights import resolve_gu_tier, ENDEMIC_DISEASES, GATE_P  # noqa: E402

_R = resolve_gu_tier()
_T2 = next(t for t in _R["ladder"] if t["tier"] == 2)


def test_evaluation_quantified():
    """2차 endemic 단계에 검정수치(LOO·순열 p·연도안정성·유의여부)가 박제됨(사용자: 수치화)."""
    ev = _T2["eval"]
    for k in ("mean_loo", "p_value", "null95", "temporal_stability", "significant"):
        assert k in ev, f"검정지표 {k} 누락(수치화 안 됨)"


def test_endemic_not_significant():
    """실측: endemic 구별 신호가 순열검정 비유의(p≥{GATE_P}) — 노이즈와 구별 안 됨."""
    assert _T2["eval"]["significant"] is False, (
        f"endemic 이 유의(p={_T2['eval']['p_value']})로 나옴 — 데이터/검정 재확인")
    assert _T2["eval"]["p_value"] >= GATE_P


def test_falls_to_uniform_when_not_significant():
    """비유의 → 도시값 균등(구별 분배 안 함) 선택 — 가짜 정밀 회피(사용자: 안 맞으면 미사용)."""
    assert _R["selected_source"] == "uniform_city" and _R["selected_tier"] == 3


def test_uniform_means_equal_weights():
    """균등 선택이면 모든 자치구 가중=1.0 (분배 차등 없음)."""
    v = list(_R["weights"].values())
    assert all(abs(x - 1.0) < 1e-6 for x in v), "균등인데 가중이 차등됨"


def test_covid_excluded():
    assert not any("코로나" in d or "COVID" in d.upper() for d in ENDEMIC_DISEASES), ENDEMIC_DISEASES


def test_production_uses_uniform():
    from build_production_forecast import _load_abm_weights, _gu_source_summary
    w = _load_abm_weights()
    assert abs(w.get("동대문구", 0) - w.get("강남구", 0)) < 1e-6, "production 이 균등(차등없음) 미반영"
    assert _gu_source_summary().get("selected_tier") == 3


if __name__ == "__main__":
    ev = _T2["eval"]
    print(f"  endemic 검정: mean-LOO={ev['mean_loo']} p={ev['p_value']} (null95={ev['null95']}) "
          f"안정성={ev['temporal_stability']} → {'유의' if ev['significant'] else '비유의'}")
    print(f"  선택: {_R['selected_tier']}차 {_R['selected_source']} · {_R['confidence']}")
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    p = f = 0
    for fn in funcs:
        try:
            fn(); print(f"  ✓ PASS  {fn.__name__}"); p += 1
        except Exception as e:
            print(f"  ✗ FAIL  {fn.__name__}: {e}"); f += 1
    print(f"\n  {p} PASS / {f} FAIL")
    sys.exit(1 if f else 0)
