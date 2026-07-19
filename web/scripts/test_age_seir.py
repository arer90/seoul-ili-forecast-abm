#!/usr/bin/env python3
"""age별 SEIR 빌더(build_age_seir) — TDD (순수 집계 + 산출 스키마 + age gradient).

박제: full WAIFW age-구조화 SEIR 를 web 에 배선 — gu×age 인구(decade 밴드) 집계 정확성 + 산출
age-seir-forecast.json 이 학령기 高·노년 低 gradient(검증 ρ≈0.68) 를 표면화.
Run:  .venv/bin/python web/scripts/test_age_seir.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "web" / "scripts"))

from build_age_seir import pop_matrix_from_rows, BAND_LABELS, N_BANDS  # noqa: E402
AGG = ROOT / "web" / "public" / "aggregates"


def test_decade_band_mapping():
    """age_group → decade 밴드(0-9→0, 10-19→1, 65-69→6) 정확 집계."""
    rows = [("강남구", "0-9", 100), ("강남구", "10-14", 50), ("강남구", "15-19", 30),
            ("강남구", "65-69", 20), ("서초구", "20-24", 200)]
    M = pop_matrix_from_rows(rows, ["강남구", "서초구"])
    assert M.shape == (2, 7)
    assert M[0, 0] == 100        # 0-9
    assert M[0, 1] == 80         # 10-14 + 15-19 (둘 다 밴드1)
    assert M[0, 6] == 20         # 65-69 → 60+
    assert M[1, 2] == 200        # 20-24 → 밴드2


def test_unmapped_and_missing_ignored():
    """미매핑 라벨/미지정 gu/None 인구는 무시(crash X)."""
    rows = [("강남구", "전체", 999), ("없는구", "0-9", 100), ("강남구", "0-9", None)]
    M = pop_matrix_from_rows(rows, ["강남구"])
    assert M.sum() == 0


def test_band_labels_are_decades():
    assert len(BAND_LABELS) == N_BANDS == 7
    assert BAND_LABELS[0] == "0-9" and BAND_LABELS[-1] == "60+"


def test_age_seir_json_gradient():
    """산출 age-seir-forecast.json: 7밴드, 학령기(10-19) 최고·노년(60+) 최저 공격률."""
    p = AGG / "age-seir-forecast.json"
    if not p.is_file():
        print("  (age-seir-forecast.json 없음 — build_age_seir.py 먼저 실행)"); return
    d = json.loads(p.read_text(encoding="utf-8"))
    ar = d["attack_rate_pct"]
    assert len(d["bands"]) == 7 and len(ar) == 7
    assert ar.index(max(ar)) <= 1, "공격률 peak 가 학령기(0-9/10-19)가 아님"
    assert ar.index(min(ar)) == 6, "공격률 trough 가 60+ 가 아님"
    assert d["validation"]["vs_real_sentinel_age_ili_spearman"] >= 0.5


def test_age_seir_population_realistic():
    """산출 band_population 합이 Seoul 규모(~1000만)."""
    p = AGG / "age-seir-forecast.json"
    if not p.is_file():
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    tot = sum(d["band_population"])
    assert 8e6 < tot < 1.1e7, f"인구 규모 비현실적: {tot:,}"


def test_web_wiring_age_card():
    """app.jsx 가 age-seir-forecast 를 로드하고 Sim 탭에 age 카드를 렌더(배선 박제)."""
    src = (ROOT / "web_prototype" / "app.jsx").read_text(encoding="utf-8")
    abs_ = (ROOT / "web" / "public" / "abs" / "app.jsx").read_text(encoding="utf-8")
    assert "age-seir-forecast.json" in src and "setAgeSeir" in src, "App 이 age-seir 로드 안 함"
    assert "ageSeir={ageSeir}" in src, "SimTab 에 ageSeir 전달 안 됨"
    assert "age별 SEIR 예측 (WAIFW" in src, "Sim 탭 age 카드 미렌더"
    # ARIA 챗 grounding: 연령 위험을 WAIFW 모델로 인용(환각 금지)
    assert "const ageBlock" in src and "연령별 위험(WAIFW" in src, "ARIA simContext 에 age 블록 미배선"
    assert "${ageBlock}" in src, "ageBlock 이 simContext 에 주입 안 됨"
    # 팬데믹 기계론 카드: age 라인
    assert "setAgeSeirData" in src and "연령별 위험" in src, "PANDEMIC 카드에 age 라인 미배선"
    # hermes.ts(Vercel 경로) age 규칙
    herm = (ROOT / "web" / "lib" / "hermes.ts").read_text(encoding="utf-8")
    assert "연령별 위험" in herm and "age-seir-forecast" in herm, "hermes 에 age 규칙 미배선"
    assert src == abs_, "dual-file 미동기화 — sync_app.mjs 실행 필요"


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
