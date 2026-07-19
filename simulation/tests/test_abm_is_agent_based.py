"""DECISIVE test — 개체기반(ABM)인가 확률적 구획모델인가 (외부평가 compass, 2026-06-08).

compass 보고서의 결정적 판별 기준:
  "agent_kernel.py가 **길이-N 개체 상태 배열**(개체별 속성/접촉)이면 → ABM VERIFIED.
   구별당 S,E,I,R,V,D 스칼라/벡터 카운트에 이항추출이면 → 확률적 구획모델."

이 test는 `run_agent_world`가 전자임을 외부에서(반환값으로) 증명한다. Grimm et al. 2006
IBM 정의(개별 상태 + 이질 속성 + 명시적 접촉구조)의 3요소를 각각 assert한다.
"""
import numpy as np

from simulation.abm.agent_kernel import run_agent_world


def _run(N, days=30, seed=42, theta_sd=0.2):
    return run_agent_world(N, T_days=days, beta=0.35, sigma=0.2, gamma=0.1,
                           delta=0.001, nu=0.0, global_seed=seed, theta_sd=theta_sd)


def test_length_N_individual_state_array_not_compartment_counts():
    """① 길이-N 개체 상태 배열 (구획 카운트가 아님) — compass 판별의 핵심."""
    N = 2000
    ag = _run(N)["agents"]
    assert ag["state"].shape == (N,)              # 개체 1명당 1 disease-state entry
    assert ag["state"].dtype == np.int8           # per-agent compartment label
    assert len(np.unique(ag["state"])) >= 2       # 개체들이 여러 상태에 분포(개별 추적)


def test_per_agent_heterogeneous_attributes():
    """② 개체별 이질 속성 (각 개체가 자기 age/occupation/behavioral params) — Grimm 이질성."""
    N = 2000
    ag = _run(N, theta_sd=0.25)["agents"]
    for k in ("age_band", "home_gu", "work_gu", "alpha", "kappa", "tau", "theta"):
        assert ag[k].shape == (N,), f"{k}는 per-agent 배열이어야 함"
    assert ag["theta"].std() > 0                  # 개체마다 다른 threshold(이질성 실재)
    assert len(np.unique(ag["age_band"])) >= 2    # 연령 이질


def test_explicit_spatial_structure_per_agent_districts():
    """③ 명시적 공간구조 — 개체별 home_gu/work_gu(25 자치구)에 분산. 통근 coupling(home≠work)은
    synthetic population 또는 mixing_matrix로 활성(half-day work/home phase)."""
    ag = _run(3000)["agents"]
    assert ag["home_gu"].shape == (3000,) and ag["work_gu"].shape == (3000,)
    assert set(np.unique(ag["home_gu"])).issubset(set(range(25)))   # 25 자치구
    assert len(np.unique(ag["home_gu"])) >= 2      # 개체들이 여러 구에 분산(공간 이질)


def test_stochastic_reproducible_but_not_deterministic_ode():
    """④ report1 test: 같은 seed=bit-identical, 다른 seed=peak 분산>0 (확률 ABM, ODE 아님)."""
    def peak(seed):
        return float(np.asarray(_run(3000, days=60, seed=seed)["I"]).max())
    assert peak(42) == peak(42)                    # 결정성(같은 seed)
    peaks = [peak(s) for s in range(6)]
    assert np.std(peaks) > 0                        # 확률성(다른 seed) → ODE 아님


def test_per_agent_behavioral_state_individually_tracked():
    """⑤ 개체별 행동상태(theta/fatigue/compliance)가 개별 추적 — compartment-mean이 아니라
    개체마다 다른 값을 가짐(이질성은 집계서 평균화될 수 있으나 상태는 개별 저장)."""
    ag = _run(3000, theta_sd=0.3)["agents"]
    for k in ("theta", "fatigue", "compliance"):
        assert ag[k].shape == (3000,), f"{k}는 per-agent 배열"
    assert ag["theta"].min() != ag["theta"].max()   # 개체마다 다른 threshold(개별 저장)
    assert ag["theta"].std() > 0                      # 이질성 실재
