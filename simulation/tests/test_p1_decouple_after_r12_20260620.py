"""TDD: P1(real_forecaster)을 R12 뒤(production-track 시작)로 이동 + R12↔real_eval 디커플.

사용자 지시(2026-06-20): production(P1)은 모든 research(R)가 끝난 뒤 실행돼야 하고,
comprehensive eval(R12)은 챔피언/family를 R9(per_model_optimize)·R10(per_model_eval)에서
소비해야 한다 — P1의 real-slab이 아니라. (real_eval=default-HP orphan; R9=SSOT)

Reproduction: 이전엔 phases 순서가 R10→P1→R11→R12 라서 P1이 research 중간에 끼었고,
R12가 all_results["real_eval"]를 소비(_push_metrics real slab + coverage row)했다.
"""
from pathlib import Path

from simulation.pipeline import phases

ROOT = Path(__file__).resolve().parents[1] / "pipeline"


def test_p1_after_all_research():
    r_idx = [phases.order(f"R{i}") for i in range(1, 13)]
    assert phases.order("P1") > max(r_idx), "P1 must run after every R phase (R1..R12)"
    assert phases.order("P1") < phases.order("P2"), "P1 starts the production track"


def test_phase_order_research_block_then_p1():
    labels = [p[0] for p in phases.PHASES]
    p1_pos = labels.index("P1")
    assert labels[p1_pos - 1] == "R12", "P1 must come immediately after R12"
    research_before = [l for l in labels[:p1_pos] if l.startswith("R")]
    assert research_before == [f"R{i}" for i in range(1, 13)], "R1..R12 all precede P1"


def test_runner_p1_block_after_r12_block():
    src = (ROOT / "runner.py").read_text(encoding="utf-8")
    assert src.count('should_run("P1"') == 1, "exactly one live P1 dispatch gate"
    r12 = src.index('should_run("R12"')
    p1 = src.index('should_run("P1"')
    assert p1 > r12, "runner: P1 dispatch block must come AFTER the R12 block"


def test_r12_does_not_consume_real_eval():
    src = (ROOT / "comprehensive_eval.py").read_text(encoding="utf-8")
    assert '_push_metrics("real_eval"' not in src, \
        "R12 must not push a real_eval slab (decoupled from P1)"
    assert '("12", "real_eval")' not in src, \
        "R12 coverage table must not list real_eval as a research-track row"
