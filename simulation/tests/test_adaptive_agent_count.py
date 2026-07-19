"""동적 agent 수 선택 TDD — 수렴 기준 자동 N (사용자 제안 2026-06-05).

estimate_fn(n)->(point,cv) 를 합성으로 주어 selector 로직만 검증 (heavy ABM 분리).
macOS: run PER-FILE.
"""
import numpy as np
import pytest

from simulation.abm.adaptive_agent_count import select_n_agents_adaptive


def test_converges_before_max():
    """추정치가 asymptote 로 수렴 + cv 감소 → 최대 N 전에 채택."""
    def est(n):
        point = 100.0 * (1.0 - np.exp(-n / 2000.0))   # asymptote 100
        cv = 0.1 * (2000.0 / n) ** 0.5                 # cv ↓ as n ↑
        return point, cv
    cand = [1000, 2000, 4000, 8000, 16000, 32000]
    r = select_n_agents_adaptive(est, cand, tol=0.02, cv_tol=0.05, patience=1)
    assert r["converged"] is True
    assert 2000 <= r["n_optimal"] < 32000             # 동적으로 적정 N 선택
    assert len(r["trace"]) <= len(cand)


def test_nonconvergent_falls_back_to_max():
    """잡음 큰(cv 높은) 추정치 → 수렴 실패 → 최대 N fallback (안전)."""
    def est(n):
        return float(50 + (n % 7)), 0.5               # cv=0.5 항상 > cv_tol
    r = select_n_agents_adaptive(est, [1000, 2000, 4000], tol=0.001, cv_tol=0.01)
    assert r["converged"] is False
    assert r["n_optimal"] == 4000                     # fallback = max candidate


def test_cv_gate_blocks_early_stop():
    """추정치 안정해도 cv 크면 채택 안 함 (stochastic 잡음 가드)."""
    def est(n):
        return 100.0, 0.20                            # point 일정하지만 cv=0.2 > tol
    r = select_n_agents_adaptive(est, [1000, 2000, 4000], tol=0.02, cv_tol=0.05)
    assert r["converged"] is False


def test_validation():
    with pytest.raises(ValueError):
        select_n_agents_adaptive(lambda n: (1.0, 0.0), [])
    with pytest.raises(ValueError):
        select_n_agents_adaptive(lambda n: (1.0, 0.0), [4000, 2000])   # 비오름차순


def test_trace_records_rel_change():
    seq = iter([(10.0, 0.01), (10.05, 0.01), (10.06, 0.01)])
    r = select_n_agents_adaptive(lambda n: next(seq), [1000, 2000, 4000],
                                 tol=0.02, cv_tol=0.05, patience=1)
    assert r["trace"][0]["rel_change"] == float("inf")   # 첫 점은 비교대상 없음
    assert r["trace"][1]["rel_change"] < 0.02            # 10→10.05 = 0.5%
    assert r["converged"] is True and r["n_optimal"] == 2000
