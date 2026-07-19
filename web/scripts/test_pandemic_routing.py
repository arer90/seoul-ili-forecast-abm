#!/usr/bin/env python3
"""PANDEMIC 차트 라우팅 — 데이터 계약 + 웹 실스왑 TDD (ⓑ ② 검증).

박제(사용자: 평상시 외엔 외부신호 선행 + 기계론 SEIR/ABM; ML 외삽 금지):
  데이터층: routing_manifest(mode_ord) 가 모드별 권위 forecast 를 올바르게 라우팅.
    - PANDEMIC(≥2) → primary=seir-forecast-360(기계론), alert=pandemic, ML=fallback.
    - WATCH(1)     → primary=ili-forecast(ML 유지) + secondary=seir(병렬), alert=watch.
    - SEASONAL(0)  → primary=ili-forecast, secondary=None, alert=none.
  웹층(실스왑, 텍스트배지만 X): nowcast 섹션이 alert_level 에 따라 기계론 SEIR 요약 카드를
    실제로 표면화(seir-forecast-360 로드)하고 PANDEMIC 시 ML 다중수평선 표를 강등(opacity/참고용).

Run:  .venv/bin/python web/scripts/test_pandemic_routing.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "web" / "scripts"))

from resolve_mode import routing_manifest  # noqa: E402

SRC = ROOT / "web_prototype" / "app.jsx"
ABS = ROOT / "web" / "public" / "abs" / "app.jsx"


def _txt(p):
    return p.read_text(encoding="utf-8")


# ── 데이터 계약 ──────────────────────────────────────────────────────────
def test_pandemic_routes_to_mechanistic():
    """PANDEMIC(≥2) → 권위=기계론 SEIR, ML 은 fallback(외삽 불가)."""
    rm = routing_manifest(2)
    assert rm["primary"] == "seir-forecast-360.json", rm
    assert rm["secondary"] == "abm-scenarios.json", rm
    assert rm["fallback"] == "ili-forecast.json", rm
    assert rm["alert_level"] == "pandemic" and rm["alert"], rm


def test_watch_keeps_ml_plus_parallel_seir():
    """WATCH(1) → ML nowcast 유지 + 기계론 병렬(secondary)."""
    rm = routing_manifest(1)
    assert rm["primary"] == "ili-forecast.json", rm
    assert rm["secondary"] == "seir-forecast-360.json", rm
    assert rm["alert_level"] == "watch" and rm["alert"], rm


def test_seasonal_is_ml_only_no_alert():
    """SEASONAL(0) → ML 단독, 경보 없음."""
    rm = routing_manifest(0)
    assert rm["primary"] == "ili-forecast.json", rm
    assert rm["secondary"] is None and rm["alert_level"] == "none" and rm["alert"] == "", rm


def test_alert_level_monotone():
    """모드 상승 = 경보 강도 단조(none→watch→pandemic)."""
    order = {"none": 0, "watch": 1, "pandemic": 2}
    levels = [order[routing_manifest(o)["alert_level"]] for o in (0, 1, 2)]
    assert levels == [0, 1, 2], levels


# ── 웹 실스왑 (텍스트 배지만이 아니라 차트 소스 스왑) ─────────────────────
def test_web_loads_mechanistic_for_routing():
    """nowcast drawer 가 seir-forecast-360(기계론 권위)을 로드함."""
    t = _txt(SRC)
    assert "setSeir360Data" in t and "seir-forecast-360.json" in t, \
        "drawer 가 기계론 권위 forecast(seir-forecast-360)를 로드 안 함 — 스왑 불가"


def test_web_surfaces_mechanistic_card_on_alert():
    """alert_level(pandemic/watch) 시 기계론 SEIR 요약 카드를 실제 표면화(텍스트 배지 외)."""
    t = _txt(SRC)
    assert "권위 예측 — 기계론 SEIR" in t, "기계론 권위 카드 미표면화"
    assert "seir360Data?.summary" in t, "기계론 카드가 seir360 summary 를 안 읽음"
    # 카드가 alert_level 게이트(none 이면 표시 안 함)
    assert re.search(r"lvl === 'none'.*return null", t, re.S), "기계론 카드가 평시에도 떠버림(게이트 없음)"


def test_web_demotes_ml_under_pandemic():
    """PANDEMIC 시 ML 다중수평선 표를 강등(참고용 라벨 + opacity)."""
    t = _txt(SRC)
    assert "참고용 — PANDEMIC" in t, "PANDEMIC 에서 ML 표 강등 라벨 없음(ML 을 권위로 오인 위험)"
    assert re.search(r"alert_level === 'pandemic' \? 0\.55", t), "PANDEMIC 에서 ML 표 opacity 강등 없음"


def test_aria_prompt_is_mode_aware():
    """ARIA 챗 프롬프트가 운영모드를 인지 — PANDEMIC 시 기계론 SEIR 권위 지시(감사 HIGH 수정).
    이전: modeData 가 UI 배지 전용 → ARIA 는 mode-blind, 늘 ML nowcast 로만 답함."""
    t = _txt(SRC)
    assert "const modeBlock" in t, "ARIA 프롬프트에 운영모드 블록(modeBlock) 없음"
    assert "예측 권위 = 기계론 SEIR" in t, "PANDEMIC/WATCH 시 기계론 권위 지시 누락"
    # modeBlock 이 실제로 simContext(=sysContent)에 주입됨
    assert "${modeBlock}" in t, "modeBlock 이 simContext 에 주입 안 됨 — ARIA 가 여전히 mode-blind"


def test_hermes_grounding_has_mode_routing():
    """Vercel 경로(hermes.ts) SYSTEM_GROUNDING 도 운영모드 라우팅 규칙 보유."""
    h = (ROOT / "web" / "lib" / "hermes.ts").read_text(encoding="utf-8")
    assert "운영 모드 라우팅" in h, "hermes SYSTEM_GROUNDING 에 모드 라우팅 규칙 없음"
    assert "seir-forecast-360" in h and "PANDEMIC" in h, "기계론 권위/PANDEMIC 지시 누락"


def test_dual_file_synced():
    assert _txt(SRC) == _txt(ABS), "dual-file 미동기화 — sync_app.mjs 실행 필요"


if __name__ == "__main__":
    print("  routing_manifest: SEASONAL=%s WATCH=%s PANDEMIC=%s" % (
        routing_manifest(0)["primary"], routing_manifest(1)["primary"], routing_manifest(2)["primary"]))
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    p = f = 0
    for fn in funcs:
        try:
            fn(); print(f"  ✓ PASS  {fn.__name__}"); p += 1
        except Exception as e:
            print(f"  ✗ FAIL  {fn.__name__}: {e}"); f += 1
    print(f"\n  {p} PASS / {f} FAIL")
    sys.exit(1 if f else 0)
