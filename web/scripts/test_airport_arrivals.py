#!/usr/bin/env python3
"""국제선 입국 '인원수'(KOSIS) 유입압 신호 — TDD.

정정 박제(사용자 지적 + 조사): flight-status(운항편, 머릿수 없음) 폐기 → KOSIS 외래객 입국(인원, 명).
실측: 정상시 입국 인원은 계절 ILI 와 COVID 통제 후 r≈0(정반대 계절성). 따라서:
  - 계절 forecaster feature 아님.
  - 팬데믹 유입 context 로만 — 유입압 z(탈계절 anomaly)가 높아도 **단독으로는 WATCH 안 켬**
    (관광 회복으로 고분산 → FP 방지). novel/KDCA 가 이미 켠 경보에 seeding 근거만 부가.

Run:  .venv/bin/python web/scripts/test_airport_arrivals.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "web" / "scripts"))

from build_airport_arrivals import _importation_z, _load_cached  # noqa: E402
from build_external_risk import read_arrivals  # noqa: E402
from resolve_mode import resolve_mode_raw  # noqa: E402


def _synth(base=1_000_000, years=range(2015, 2025)):
    """같은 달 = 계절성분 + 연도별 자연변동(std>0) 합성 입국 시리즈."""
    s = {}
    for i, y in enumerate(years):
        for m in range(1, 13):
            seasonal = 1.0 + 0.3 * (1 if m in (7, 8, 10) else 0)  # 여름·가을 정점
            yoy = 1.0 + 0.04 * ((i % 5) - 2)                       # 연도별 ±8% 변동 → 같은달 std>0
            s[y * 100 + m] = int(base * seasonal * yoy)
    return s


def test_importation_z_detects_spike():
    """같은-달 과거 대비 큰 양의 anomaly → z 높음(유입 급증 감지)."""
    s = _synth()
    s[202507] = int(1_000_000 * 1.3 * 3.0)  # 7월 평소의 3배
    z = _importation_z(s, 202507)
    assert z["z"] > 2.5, f"급증인데 z 낮음: {z}"
    assert z["n_history"] >= 3


def test_importation_z_normal_is_low():
    """평소 수준(yoy≈평균 연도) → z≈0 (안정)."""
    s = _synth()
    z = _importation_z(s, 202207)  # 2022 = yoy 평균 수준 → anomaly 작음
    assert abs(z["z"]) < 1.0, f"평소인데 z 큼: {z}"


def test_importation_z_insufficient_history():
    """같은-달 과거 표본 <3 → z=0(판단보류, 거짓신호 방지)."""
    s = {201507: 1_000_000, 201607: 1_100_000, 201707: 3_000_000}  # 7월 과거 2개 + asof
    z = _importation_z(s, 201707)
    assert z["z"] == 0.0 and z["n_history"] == 2


def test_arrivals_not_standalone_watch_trigger():
    """유입압이 높아도 신종신호 없으면 단독으로 WATCH 안 켬 — 관광 FP 방지(핵심 정직성)."""
    er = {"kdca_alert_level": 0, "summary": {"respiratory_novel_confirmed": False,
          "news_spike": False, "kdca_stale": False, "arrivals_pressure_high": True}}
    r = resolve_mode_raw(er, [5.0, 5.1, 4.9, 5.2])
    assert r["raw_ord"] == 0, f"입국 유입압 단독으로 WATCH 발화(FP): {r}"
    assert r["signals"]["arrivals_pressure"] is True  # 기록은 됨(투명성)


def test_arrivals_amplifies_existing_watch():
    """신종신호로 WATCH 켜진 상태 + 유입압 높음 → seeding 근거 부가(트리거는 안 바꿈)."""
    er = {"kdca_alert_level": 0, "summary": {"respiratory_novel_confirmed": True,
          "news_spike": False, "kdca_stale": False, "arrivals_pressure_high": True}}
    r = resolve_mode_raw(er, [5.0, 5.1, 4.9, 5.2])
    assert r["raw_ord"] == 1 and "유입압" in r["reason"], r["reason"]


def test_read_arrivals_gate_threshold():
    """read_arrivals: pressure_high = z≥2.5 (보수적 컷 — 관광 회복 z≈1.6 은 미발화)."""
    a = read_arrivals()  # 실제 arrivals-monthly.json
    assert "z" in a and "pressure_high" in a
    assert a["pressure_high"] == (a["z"] >= 2.5)


def test_cached_series_is_persons_scale():
    """cached 시리즈 = 월 ~수십만~수백만 명(인원 규모) — 운항편수(수백)가 아님."""
    s = _load_cached()
    if not s:
        return  # csv 없으면 skip(다른 환경)
    vals = list(s.values())
    assert len(s) >= 100, f"개월 수 부족: {len(s)}"
    assert max(vals) > 500_000, f"입국 인원 규모가 아님(운항편수 의심): max={max(vals)}"


if __name__ == "__main__":
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    p = f = 0
    for fn in funcs:
        try:
            fn(); print(f"  ✓ PASS  {fn.__name__}"); p += 1
        except Exception as e:
            print(f"  ✗ FAIL  {fn.__name__}: {e}"); f += 1
    print(f"\n  {p} PASS / {f} FAIL")
    sys.exit(1 if f else 0)
