"""행동-질병 결합 SEIR-V-D 평가 TDD (논문 RQ-A/H-A).

H-A: 적응형 행동(ON)이 정적(OFF) 대비 공격률·peak 를 낮추고, 채택률이 유병률에
내생적으로 반응한다. macOS: run PER-FILE.
"""
import numpy as np
import pytest

from simulation.abm.behavioural import BehaviouralParams
from simulation.abm.behavior_disease_eval import (
    build_demo_metapop, run_adaptive_vs_static, observation_curve,
)
from simulation.abm.behavioural import run_coupled_abm


def _run(days=120, R0=1.9):
    mp = build_demo_metapop(G=3, days=days, R0=R0, seed=0)
    return mp, run_adaptive_vs_static(mp, BehaviouralParams())


def test_demo_metapop_row_stochastic():
    mp = build_demo_metapop(G=4)
    assert mp.populations.shape == (4,)
    assert np.allclose(mp.mobility.sum(axis=1), 1.0)        # 행 확률
    assert np.all(np.diag(mp.mobility) > 0)                 # 자기루프 양수


def test_structure_and_active():
    _, r = _run()
    assert {"adaptive", "static", "behavior", "comparison"} <= set(r)
    assert r["comparison"]["behaviour_active"] is True
    for k in ("attack_rate", "peak_prevalence", "peak_day"):
        assert k in r["adaptive"] and k in r["static"]


def test_adaptive_suppresses_epidemic():
    """H-A 핵심: 적응형이 정적 대비 공격률을 낮춘다 (접촉감소 → 전파 억제)."""
    _, r = _run()
    assert r["adaptive"]["attack_rate"] <= r["static"]["attack_rate"] + 1e-9
    assert r["comparison"]["rel_attack_rate_reduction"] > 0.0   # 측정 가능한 억제
    assert r["comparison"]["delta_peak_prevalence"] <= 1e-9     # peak 도 ≤


def test_behavior_endogenous_prevalence_elastic():
    """채택률이 유병률에 내생 반응(상관>0) = policy on/off 아님 (adaptive 정당화)."""
    _, r = _run()
    assert r["behavior"]["peak_adoption"] > 0.0
    assert r["behavior"]["min_beta_scale"] < 1.0               # 실제 접촉감소 발생
    assert r["behavior"]["prevalence_response_corr"] > 0.0     # prevalence-elastic


def test_static_arm_matches_behaviour_off():
    """static arm 이 진짜 behaviour-off (β 고정) 인지 — 독립 behaviour-off run 과 AR 일치."""
    mp, r = _run()
    off = run_coupled_abm(mp, BehaviouralParams(alpha=0.0, kappa=0.0, tau=float("inf")))
    N = float(mp.populations.sum())
    ar_off = float(off.seir.incidence.sum()) / N
    assert abs(ar_off - r["static"]["attack_rate"]) < 1e-9


def test_observation_fit_block():
    """y_obs 주면 관측모형 NegBin loglik/MAE fit 비교가 계산된다."""
    mp = build_demo_metapop(G=3, days=120, R0=1.9, seed=0)
    res_static = run_coupled_abm(mp, BehaviouralParams(alpha=0.0, kappa=0.0, tau=float("inf")))
    y = observation_curve(res_static, __import__("simulation.abm.observation_model",
                          fromlist=["ObservationParams"]).ObservationParams())
    r = run_adaptive_vs_static(mp, BehaviouralParams(), y_obs_ili=y)
    f = r["observation_fit"]
    assert f["n_weeks"] > 0
    assert np.isfinite(f["loglik_adaptive"]) or f["loglik_adaptive"] == float("-inf")
    assert f["mae_static"] >= 0.0 and f["mae_adaptive"] >= 0.0
