"""simulation.abm.behavior_disease_eval

행동-질병 결합(behavior-disease coupling)을 **SEIR-V-D 로 평가**한다 (논문 RQ-A/H-A).

제목의 핵심 주장 "adaptive behavioral responses → transmission" 을 입증하려면
**적응형 행동 ON vs OFF** 비교가 1차 결과여야 한다. 본 모듈은 동일 metapop 에서
  - adaptive : 위험인식 dR/dt=α(I/N)−λR·R + 피로 → 순응 → 접촉감소 → β_i(t) 결합
  - static   : behaviour-off (α=0,κ=0,τ=∞) — β_i(t)=β0 고정
두 arm 을 `run_coupled_abm` 으로 돌려 **감염동학·행동변화·관측(ILI)·통계** metric 을 산출한다
(SYSTEM §6 5-layer metrics 의 실제 구현 — "어떻게 했는가"의 답).

핵심 가설 H-A: 적응형은 정적 대비 **공격률(AR)·peak↓** (agents reduce contact when prevalence
rises) 이고, 채택률 B_t 가 **유병률에 내생적으로 반응**(policy on/off 아님)한다.

Gray-box 계약
-------------
- `run_adaptive_vs_static(metapop, behaviour, y_obs_ili=None)` → dict (순수; ABM 2회 실행).
  y_obs_ili 주면 관측모형으로 ILI fit(NegBin loglik) adaptive vs static 비교.
- `build_demo_metapop(...)` : TDD/데모용 작은 합성 metapop (행 확률 mobility).
- 부작용 없음. Performance: O(2 · ABM run).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from simulation.abm.behavioural import BehaviouralParams, run_coupled_abm
from simulation.abm.observation_model import (
    ObservationParams, ili_mean, negbin_loglik,
)
from simulation.sim.parameters import DiseaseParams, MetapopParams


def build_demo_metapop(G: int = 3, pop: float = 100_000.0, days: int = 200,
                       R0: float = 1.8, seed: int = 0,
                       initial_infected_gu0: float = 20.0) -> MetapopParams:
    """TDD/데모용 G-구 metapop. mobility = 행 확률(대각 0.85 + 균등 분산)."""
    populations = np.full(G, float(pop))
    mob = np.full((G, G), 0.15 / max(G - 1, 1))
    np.fill_diagonal(mob, 0.85)
    mob = mob / mob.sum(axis=1, keepdims=True)   # 행 확률 보장
    init = np.zeros(G); init[0] = float(initial_infected_gu0)
    return MetapopParams(
        disease=DiseaseParams(R0=R0),
        populations=populations,
        mobility=mob,
        district_names=[f"gu_{i}" for i in range(G)],
        initial_infected=init,
        days=int(days),
        seed=int(seed),
    )


def _static_of(behaviour: BehaviouralParams) -> BehaviouralParams:
    """behaviour-off 쌍 (α=0,κ=0,τ=∞ → β_i(t)=β0 고정)."""
    return replace(behaviour, alpha=0.0, kappa=0.0, tau=float("inf"))


def epidemic_metrics(res, N: float) -> dict:
    """감염동학 층: AR, peak incidence·timing, final size, 사망."""
    inc = res.seir.incidence                       # (T, G) 신규감염
    city_inc = inc.sum(axis=1)                      # (T,)
    cum = float(inc.sum())
    cityI = res.city_I()                            # (T+1,) 감염자 prevalence
    ipk = int(np.argmax(cityI))
    return {
        "attack_rate": cum / float(N),
        "cumulative_infections": cum,
        "peak_prevalence": float(np.max(cityI)),
        "peak_day": ipk,
        "peak_incidence": float(np.max(city_inc)) if city_inc.size else 0.0,
        "final_deaths": float(res.seir.city_total("D")[-1]),
    }


def behavior_metrics(res) -> dict:
    """행동변화 층: 채택률 B_t, 접촉감소, 피로, **유병률 내생반응**(상관)."""
    B = res.mean_compliance()                       # (T+1,) 채택률
    beta_scale = res.mean_beta_scale()              # (T+1,) β_i/β0 (접촉감소 반영)
    cityI = res.city_I()
    N = float(res.seir.params.populations.sum())
    prev = cityI / N
    # 내생성: 채택률이 (지연) 유병률에 반응? B_t vs prev_{t-1} 상관 (>0 면 prevalence-elastic)
    if B.size > 3 and np.std(B[1:]) > 1e-9 and np.std(prev[:-1]) > 1e-9:
        resp = float(np.corrcoef(B[1:], prev[:-1])[0, 1])
    else:
        resp = 0.0
    return {
        "peak_adoption": float(np.max(B)),
        "mean_adoption": float(np.mean(B)),
        "min_beta_scale": float(np.min(beta_scale)),   # 최대 접촉감소 시점
        "peak_fatigue": float(np.max(res.fatigue.mean(axis=1))),
        "prevalence_response_corr": resp,              # >0 = 내생적 적응 (policy on/off 아님)
    }


def observation_curve(res, obs: ObservationParams) -> np.ndarray:
    """관측 층: 잠재 신규감염 → 기대 ILI μ_t = ρ·symptomatic_frac·incidence."""
    city_inc = res.seir.incidence.sum(axis=1)
    sym = obs.symptomatic_frac * city_inc
    return ili_mean(sym, obs)


def run_adaptive_vs_static(metapop: MetapopParams, behaviour: BehaviouralParams,
                           y_obs_ili=None, obs: ObservationParams | None = None) -> dict:
    """adaptive vs static(behaviour-off) 비교 — RQ-A/H-A 1차 결과.

    Args:
        metapop: 공통 metapop.
        behaviour: adaptive 파라미터 (α>0). static 은 자동 behaviour-off.
        y_obs_ili: (선택) 실제 ILI (T,) — 주면 관측모형 NegBin loglik fit 비교.
        obs: 관측모형 파라미터 (기본 ObservationParams()).

    Returns:
        {adaptive, static, behavior, comparison[, observation_fit]}.
    """
    obs = obs or ObservationParams()
    N = float(np.asarray(metapop.populations, float).sum())

    res_a = run_coupled_abm(metapop, behaviour)
    res_s = run_coupled_abm(metapop, _static_of(behaviour))

    em_a, em_s = epidemic_metrics(res_a, N), epidemic_metrics(res_s, N)
    bm = behavior_metrics(res_a)

    out = {
        "adaptive": em_a,
        "static": em_s,
        "behavior": bm,
        "comparison": {
            # H-A: 적응이 AR·peak 를 낮추는가 (음수 = 적응이 유행 억제)
            "delta_attack_rate": em_a["attack_rate"] - em_s["attack_rate"],
            "rel_attack_rate_reduction": (
                (em_s["attack_rate"] - em_a["attack_rate"]) / em_s["attack_rate"]
                if em_s["attack_rate"] > 0 else 0.0),
            "delta_peak_prevalence": em_a["peak_prevalence"] - em_s["peak_prevalence"],
            "peak_delay_days": em_a["peak_day"] - em_s["peak_day"],
            "behaviour_active": not behaviour.is_behaviour_off(),
        },
    }

    if y_obs_ili is not None:
        y = np.asarray(y_obs_ili, dtype=np.float64)
        mu_a = observation_curve(res_a, obs)
        mu_s = observation_curve(res_s, obs)
        n = min(len(y), len(mu_a), len(mu_s))
        phi = obs.nb_dispersion
        out["observation_fit"] = {
            "n_weeks": int(n),
            "loglik_adaptive": negbin_loglik(y[:n], mu_a[:n], phi),
            "loglik_static": negbin_loglik(y[:n], mu_s[:n], phi),
            "mae_adaptive": float(np.mean(np.abs(mu_a[:n] - y[:n]))),
            "mae_static": float(np.mean(np.abs(mu_s[:n] - y[:n]))),
        }
    return out
