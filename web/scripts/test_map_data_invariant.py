#!/usr/bin/env python3
"""지도 데이터 = 실측 분리 invariant — TDD (소스 검증). codex/gemini 권고 1단계 박제.

박제(사용자: map=실시간/과거 실데이터, 시뮬/시나리오 분리):
  - 지도 rows(map 중심 데이터)는 iliFor 를 **scenario 없이** 호출(baseline 실측 고정).
  - 누적(cumulative)은 sc.mult 이중곱 제거(*1100*sc.mult → *1100).
  - sim-GIF(명시적 시뮬)는 scenario 유지 — 실측/시뮬 경계 분리.
  - 두 파일(web_prototype + web/public/abs) 동기화.

소스-레벨 invariant(iliFor 가 거대 app.jsx 내부라 단위추출 대신; 본격 추출은 2단계 Sim 탭).
Run:  .venv/bin/python web/scripts/test_map_data_invariant.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "web_prototype" / "app.jsx"
ABS = ROOT / "web" / "public" / "abs" / "app.jsx"


def _txt(p):
    return p.read_text(encoding="utf-8")


def _rows_block(t):
    """rows useMemo 블록 추출."""
    m = re.search(r"const rows=useMemo\(\(\)=>GUS\.map\(g=>\{.*?\}\), \[[^\]]*\]\);", t, re.S)
    assert m, "rows useMemo 블록을 못 찾음"
    return m.group(0)


def test_map_rows_baseline_no_scenario():
    """지도 rows 의 iliFor 호출에 scenario 인자가 없다(baseline 실측 고정)."""
    blk = _rows_block(_txt(SRC))
    assert "iliFor(g.id,day)" in blk, "rows 가 baseline iliFor(g.id,day) 를 안 씀"
    assert "iliFor(g.id,day,scenario)" not in blk, "rows 가 아직 scenario 를 map ILI 에 적용함"


def test_rows_deps_no_scenario():
    """rows useMemo deps 에 scenario 없음(시나리오가 지도 데이터를 재계산 안 함)."""
    blk = _rows_block(_txt(SRC))
    deps = re.search(r"\}\), \[([^\]]*)\]\);", blk).group(1)
    assert "scenario" not in deps, f"rows deps 에 scenario 잔존: [{deps}]"


def test_cumulative_double_mult_removed():
    """누적 이중곱 버그 수정: *1100*sc.mult → *1100 (scenario 무관)."""
    t = _txt(SRC)
    assert "*1100*sc.mult" not in t, "누적 이중곱(*1100*sc.mult) 잔존 — 버그 미수정"
    assert re.search(r"rows\.reduce\(\(s,r\)=>s\+r\.ili,0\)\*1100\)", t), "누적이 *1100 실측 고정이 아님"


def test_sim_gif_keeps_scenario():
    """sim-GIF(명시적 시뮬)는 scenario 유지 — 실측/시뮬 분리(시뮬은 시나리오 적용)."""
    t = _txt(SRC)
    assert "iliFor(id, d, scenario)" in t or "iliFor(id,d,scenario)" in t, \
        "sim-GIF 가 scenario 를 안 씀 — 분리가 아니라 시뮬도 죽임"


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
