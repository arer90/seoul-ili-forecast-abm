#!/usr/bin/env python3
"""Sim 탭 신설 — 실측/시뮬 분리 2단계 invariant TDD (소스 검증). codex/gemini 설계 박제.

박제(사용자: map=실시간/과거 실측, 새 탭=시뮬레이션·agent 변화량 — 엉킴 분리):
  - 시나리오 선택(setScenario)은 SimTab 에만, MapTab 엔 없음(지도 실측 전용).
  - SimTab 이 baseline 대비 Δ(변화량) 표 + 시뮬 런처(ABM/What-if/히트맵애니) 보유.
  - 3-탭(map/sim/chat) BottomTabBar + PANDEMIC 펄스(mode prop).
  - App 이 pane-sim 에 SimTab 렌더(modeData 주입).
  - ARIA grounding: 선택 gu ILI 는 baseline 실측(iliFor(id,day)), 시나리오는 가설로 분리.
  - dual-file(web_prototype + abs) 동기화 + 3-탭 CSS.

소스-레벨 invariant(거대 app.jsx 단위추출 불가). 1단계(map baseline) = test_map_data_invariant.py.
Run:  .venv/bin/python web/scripts/test_sim_tab.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "web_prototype" / "app.jsx"
ABS = ROOT / "web" / "public" / "abs" / "app.jsx"
CSS = ROOT / "web_prototype" / "styles.css"


def _txt(p):
    return p.read_text(encoding="utf-8")


def _component(t, name):
    """function <name>(...) 본문을 다음 top-level `function ` 까지 추출(근사)."""
    m = re.search(rf"\nfunction {name}\(", t)
    assert m, f"{name} 컴포넌트 정의를 못 찾음"
    start = m.start()
    nxt = re.search(r"\nfunction [A-Z]", t[start + 1:])
    return t[start: start + 1 + (nxt.start() if nxt else len(t))]


def test_simtab_component_exists():
    """SimTab 컴포넌트가 정의됨(시뮬 전용 탭)."""
    assert re.search(r"\nfunction SimTab\(", _txt(SRC)), "SimTab 컴포넌트 미정의"


def test_scenario_selector_only_in_simtab():
    """시나리오 선택(setScenario)은 SimTab 에만 — MapTab 엔 없음(지도=실측 전용)."""
    t = _txt(SRC)
    maptab = _component(t, "MapTab")
    simtab = _component(t, "SimTab")
    assert "setScenario(key)" in simtab, "SimTab 에 시나리오 선택 버튼이 없음"
    assert "setScenario" not in maptab, "MapTab 에 아직 setScenario(시나리오 선택)가 남아있음 — 분리 미완"


def test_map_has_no_scenario_bar():
    """MapTab 본문에 시나리오 바(map-scenario-bar)가 없음 — SimTab 으로 이전됨."""
    t = _txt(SRC)
    maptab = _component(t, "MapTab")
    assert "map-scenario-bar" not in maptab, "MapTab 에 시나리오 바 잔존 — 지도가 아직 실측 전용 아님"
    simtab = _component(t, "SimTab")
    assert "map-scenario-bar" in simtab, "SimTab 에 시나리오 바가 없음"


def test_simtab_has_delta_and_launchers():
    """SimTab 이 변화량(Δ) 표 + 시뮬 런처(ABM/What-if/히트맵애니)를 보유."""
    simtab = _component(_txt(SRC), "SimTab")
    assert "baselineCity" in simtab, "SimTab 에 baseline 대비 Δ 계산(baselineCity)이 없음"
    assert "iliFor(g.id, day)" in simtab, "Δ baseline 이 실측 iliFor(g.id, day)(시나리오 무관)이 아님"
    for launcher in ("onOpenABM", "onOpenWhatIf", "onSimGif"):
        assert launcher in simtab, f"SimTab 에 시뮬 런처 {launcher} 누락"


def test_bottombar_three_tabs_with_pulse():
    """BottomTabBar = 3-탭(map/sim/chat) + 평시 외 mode 펄스."""
    bar = _component(_txt(SRC), "BottomTabBar")
    assert "id:'sim'" in bar or 'id:"sim"' in bar, "하단 탭에 sim 탭이 없음"
    assert "id:'map'" in bar and "id:'chat'" in bar, "map/chat 탭 누락"
    assert "mode" in bar and "SEASONAL" in bar, "BottomTabBar 가 mode(평시 외 펄스)를 안 씀"


def test_app_renders_pane_sim():
    """App 이 pane-sim 에 SimTab(modeData 주입)을 렌더."""
    t = _txt(SRC)
    assert "pane pane-sim" in t, "App 에 pane-sim 이 없음"
    assert re.search(r"<SimTab\b", t), "App 이 SimTab 을 렌더하지 않음"
    assert re.search(r"<BottomTabBar[^>]*mode=", t), "BottomTabBar 에 mode prop 전달 안 됨"


def test_aria_grounding_baseline_not_scenario():
    """ARIA 선택-gu context = baseline(iliFor(id,day)), 시나리오 배수 미적용 + 정직 라벨(추정, 관측 아님)."""
    t = _txt(SRC)
    assert "iliFor(selectedGu.id,day,scenario)" not in t, \
        "ARIA guLine 이 아직 scenario 배수된 ILI 를 '관측'으로 사용 — 실측/가설 미분리"
    assert "iliFor(selectedGu.id,day).toFixed(1)} [추정 per-gu" in t, \
        "ARIA guLine 이 baseline + 정직 라벨(추정 per-gu)이 아님"
    # per-gu 를 '실측/관측'으로 단정 금지(감사 MEDIUM 정직성): 합성 disaggregation 임을 명시
    assert "[실측/baseline]" not in t, "per-gu 합성값을 '실측'으로 라벨(거짓 관측 주장)"


def test_mode_loaded_at_app_level():
    """mode-state.json 이 App 레벨에서 로드됨(setModeData)."""
    t = _txt(SRC)
    assert "setModeData" in t and "/aggregates/mode-state.json" in t, "App 레벨 mode 로드 누락"


def test_css_three_tab_grid():
    """CSS 3-탭 그리드 + 인디케이터 3위치 + 데스크톱 pane-sim."""
    c = _txt(CSS)
    assert "grid-template-columns: repeat(3, 1fr)" in c, "tabbar 3-컬럼 그리드 아님"
    assert 'data-tab="sim"' in c and "translateX(200%)" in c, "탭 인디케이터 3위치(sim/chat) 미설정"
    assert ".pane-sim" in c, "데스크톱 pane-sim 규칙 누락"


def test_css_desktop_no_dead_state():
    """데스크톱 dead-state 방지(적대리뷰 HIGH): chat 탭서 왼쪽 컬럼이 비지 않음.
    blanket `.pane-map:not(.is-active)` 숨김 금지 — map 은 sim 탭일 때만 숨김(data-tab=sim)."""
    c = _txt(CSS)
    assert ".pane-map:not(.is-active)" not in c, \
        "blanket pane-map 숨김 잔존 — chat 탭서 데스크톱 왼쪽 컬럼 dead-state(적대리뷰 HIGH)"
    assert '.app[data-tab="sim"] .pane-map' in c, \
        "map 은 sim 탭일 때만 숨겨야 함(chat 탭선 유지) — 규칙 누락"


def test_map_agent_layer_baseline_decoupled():
    """지도(에이전트 점 포함)는 scenario='baseline' 고정 — 시뮬탭 시나리오가 지도를 안 바꿈."""
    t = _txt(SRC)
    assert 'scenario="baseline" customAgents' in t, \
        "LeafletSeoulMap 에 scenario='baseline' 고정 안 됨 — 에이전트 점 scenario leak"
    # MapTab 시그니처(첫 줄 destructure)에 scenario 파라미터 없음(지도 실측 전용)
    sig = re.search(r"function MapTab\(\{([^}]*)\}\)", t).group(1)
    params = [p.strip() for p in sig.split(",")]
    assert "scenario" not in params, f"MapTab 시그니처에 scenario 잔존(지도 실측 전용 위반): {params[:6]}"


def test_dual_file_synced():
    """web_prototype 과 web/public/abs 가 동기화(배포본 일치)."""
    assert _txt(SRC) == _txt(ABS), "dual-file 미동기화 — sync_app.mjs 실행 필요"


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
