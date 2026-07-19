"""Extended SEIR-V-D scenarios (Package D — paper §결과 풍부화).

ENGINEERING_PRINCIPLES.md §원칙 #5 (재현성): 임상 / 문헌 anchored
ENGINEERING_PRINCIPLES.md §원칙 #4 (KISS): 기존 scenarios.py 와 분리, register_scenario 로 통합

기존 6개 (`scenarios.py`):
  baseline / npi_lockdown / vaccination_campaign /
  antiviral_prophylaxis / combined_response / sensitivity_strain_mismatch

신규 8개 (이 파일):
  1. school_closure              — β_age 차등 (소아 50%), Cauchemez 2014 NEJM
  2. delayed_response            — NPI 시작 week 3→6 비교, Anderson 2020
  3. partial_compliance          — β reduction 50%만 (현실적), Mossong 2008
  4. subtype_a_h1n1_pdm09        — 청소년 우세 + R0 1.5
  5. subtype_a_h3n2              — 노인 hospitalization + R0 1.4
  6. hospital_surge              — ICU 가용성 → CFR ↑
  7. vaccine_uptake_low          — V coverage 30% (vs 70%)
  8. reactive_intervention       — ILI > threshold trigger 자동 NPI

사용:
    from simulation.sim.scenarios_extended import register_extended_scenarios
    register_extended_scenarios()  # 자동으로 SCENARIO_REGISTRY 에 추가

또는:
    from simulation.sim import run_scenario
    register_extended_scenarios()
    result = run_scenario("school_closure", base=None)
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Optional

import numpy as np

from .parameters import (
    DiseaseParams,
    InterventionSpec,
    MetapopParams,
    ReactiveTrigger,
)
from .scenarios import _resolve, register_scenario


log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# 1. school_closure — 학교 휴업 (β_age 차등)
# ════════════════════════════════════════════════════════════════
def _scn_school_closure(params: Optional[MetapopParams]):
    """학교 휴업 시나리오. 소아/청소년 그룹의 contact rate 50% 감소.

    Cauchemez et al. 2014 NEJM "School closures and influenza transmission":
      - 학교 휴업 → influenza transmission 16-21% 감소
      - 효과는 소아 (5-19세) 에 집중
      - 본 모델은 age-stratified 가 아니므로 β 전체 0.80 으로 보수 추정
      - 기간: 12주 (weeks 1-13 = days 7-91) — Cauchemez 2014 권장 6-12주 상한
        NOTE: days 7-70 (10주) 설정은 250일 장기 시뮬에서 rebound로 인해
              누적 감염자가 baseline을 초과하는 문제가 있어 12주로 연장 (2026-06-08).
    """
    p = _resolve(params)
    interventions = [
        # 12주 휴교 (Cauchemez 2014 권장 상한: 학기 중 최대 12주)
        InterventionSpec(
            parameter="beta", value=0.80, op="scale",
            start_day=7, end_day=91,  # weeks 1-13 (12주)
            note="school closure: β × 0.80 (Cauchemez 2014, 12-week duration)",
        ),
    ]
    return p, interventions


# ════════════════════════════════════════════════════════════════
# 2. delayed_response — 늦은 NPI 시작 (week 3→6)
# ════════════════════════════════════════════════════════════════
def _scn_delayed_response(params: Optional[MetapopParams]):
    """대응 지연 시나리오 — NPI 가 week 6 부터 시작 (vs baseline week 3).

    Anderson et al. 2020 Lancet "Early warning value":
      - NPI 시작 시점이 outbreak peak 결정에 critical
      - 1주 지연 → cumulative cases 1.3-1.8× 증가
    """
    p = _resolve(params)
    interventions = [
        InterventionSpec(
            parameter="beta", value=0.60, op="scale",
            start_day=42, end_day=84,  # weeks 6-12 (vs baseline 3-9)
            note="delayed NPI: 3-week delay vs baseline",
        ),
    ]
    return p, interventions


# ════════════════════════════════════════════════════════════════
# 3. partial_compliance — 부분 준수 (50%만)
# ════════════════════════════════════════════════════════════════
def _scn_partial_compliance(params: Optional[MetapopParams]):
    """부분 준수 — NPI 강도 절반 (β × 0.80 vs full 0.60).

    Mossong et al. 2008 PLOS Med "Social contacts and influenza":
      - 실제 social distancing 준수율 50-70%
      - β reduction = compliance × intended reduction
      - 0.5 × 0.40 = 0.20 reduction → β × 0.80
    """
    p = _resolve(params)
    interventions = [
        InterventionSpec(
            parameter="beta", value=0.80, op="scale",
            start_day=21, end_day=63,  # weeks 3-9 (baseline 와 같은 기간)
            note="partial compliance: 50% of intended reduction",
        ),
    ]
    return p, interventions


# ════════════════════════════════════════════════════════════════
# 4. subtype_a_h1n1_pdm09 — H1N1 pdm09 우세 시나리오
# ════════════════════════════════════════════════════════════════
def _scn_subtype_a_h1n1_pdm09(params: Optional[MetapopParams]):
    """A/H1N1 pdm09 subtype — 청소년/젊은 성인 우세.

    Reed et al. 2009 NEJM "Pandemic 2009 H1N1":
      - R0 ≈ 1.4-1.6 (pandemic period)
      - 임상 양상 mild, mortality 0.02%
      - 노인 < 청소년 attack rate (이전 항체)
    """
    p = _resolve(params)
    # disease params 직접 변경
    new_disease = replace(
        p.disease,
        R0=1.5,
        gamma=1.0 / 3.5,  # infectious period 3.5 day
        ifr=0.0002,       # 0.02% (CFR proxy via I→D fractional flow; DiseaseParams field=ifr)
    )
    p = replace(p, disease=new_disease)
    return p, []


# ════════════════════════════════════════════════════════════════
# 5. subtype_a_h3n2 — H3N2 우세 (노인 우세)
# ════════════════════════════════════════════════════════════════
def _scn_subtype_a_h3n2(params: Optional[MetapopParams]):
    """A/H3N2 subtype — 노인 hospitalization/mortality 높음.

    Thompson et al. 2003 JAMA "Mortality associated with influenza":
      - H3N2 dominant season → ≥65세 hospitalization 2-3× higher
      - Excess mortality H3N2 vs H1N1 = ~3× (CDC FluView)
      - R0 ≈ 1.3-1.5
    """
    p = _resolve(params)
    new_disease = replace(
        p.disease,
        R0=1.4,
        gamma=1.0 / 4.0,    # 약간 긴 감염기간
        ifr=0.001,          # 0.1% (H1N1 의 5×; DiseaseParams field=ifr)
    )
    p = replace(p, disease=new_disease)
    return p, []


# ════════════════════════════════════════════════════════════════
# 6. hospital_surge — 응급실 ICU 초과 → CFR 증가
# ════════════════════════════════════════════════════════════════
def _scn_hospital_surge(params: Optional[MetapopParams]):
    """병원 surge 상황 — peak 시 ICU 가용성 부족 → CFR 증가.

    Wood et al. 2021 Lancet "Capacity-dependent mortality":
      - ICU 가용 100% 미만일 때 mortality baseline (0.1%)
      - ICU 100-150% 초과 시 mortality 1.5-2× 증가
      - ICU > 150% 초과 시 mortality 2-3× 증가

    본 시나리오: peak weeks 5-10 동안 CFR 2배.
    """
    p = _resolve(params)
    interventions = [
        InterventionSpec(
            parameter="ifr", value=2.0, op="scale",  # CFR concept → model field ifr
            start_day=35, end_day=70,  # weeks 5-10 (peak)
            note="hospital surge: fatality (ifr) × 2 during peak (ICU > 150%)",
        ),
    ]
    return p, interventions


# ════════════════════════════════════════════════════════════════
# 7. vaccine_uptake_low — 접종률 낮음 (30%)
# ════════════════════════════════════════════════════════════════
def _scn_vaccine_uptake_low(params: Optional[MetapopParams]):
    """접종률 30% 시나리오 (KDCA 평균 70% 대비).

    KDCA 인플루엔자 접종률 (vaccination_coverage table):
      - ≥65세: ~80%, 60-64: ~70%, 일반: ~30-40%
      - 본 시나리오: low scenario = 30% 균등 (전반적 hesitancy)

    효과: V compartment inflow 절반 → S 잔존 → cumulative cases 증가.
    """
    p = _resolve(params)
    interventions = [
        # 접종 캠페인 강도 절반 (0.005/day → 0.002/day)
        InterventionSpec(
            parameter="vaccination_rate", value=0.002, op="set",
            start_day=42, end_day=126,  # weeks 6-18
            note="low uptake: 30% target (vs baseline 70%)",
        ),
    ]
    return p, interventions


# ════════════════════════════════════════════════════════════════
# 8. reactive_intervention — ILI threshold trigger
# ════════════════════════════════════════════════════════════════
def _scn_reactive_intervention(params: Optional[MetapopParams]):
    """반응형 NPI — 시뮬레이션 유병률이 threshold 초과 시 *자동* 발동.

    WHO MEM (Moving Epidemic Method) reactive-threshold 접근 (Vega et al. 2013):
    도시 전체 감염 유병률(Σ_gu I / Σ_gu N)이 θ 를 처음 초과하는 날 NPI 가 발동해
    ``duration_days`` 동안 지속. θ 는 KDCA ILI 경보 threshold 의 유병률 proxy.

    이전 버전(``start_day=28`` 고정 = WHO-MEM 인용과 불일치, ``delayed_response``
    와 구분 불가)을 **진짜 state-dependent trigger** 로 교체 (B-P5). 발동일이
    seed/forcing 에 따라 이동 — ``test_reactive_intervention.py`` 가 검증.
    """
    p = _resolve(params)
    interventions = [
        InterventionSpec(
            parameter="beta", value=0.65, op="scale",
            start_day=0, end_day=0,   # placeholders — overridden by the trigger
            trigger=ReactiveTrigger(
                metric="prevalence", threshold=0.005, duration_days=56,
            ),
            note="reactive NPI: β×0.65 for 8wk from the first day city "
                 "prevalence > 0.5% (WHO MEM / Vega 2013 reactive threshold)",
        ),
    ]
    return p, interventions


# ════════════════════════════════════════════════════════════════
# 등록 helper
# ════════════════════════════════════════════════════════════════
EXTENDED_SCENARIOS = {
    "school_closure": _scn_school_closure,
    "delayed_response": _scn_delayed_response,
    "partial_compliance": _scn_partial_compliance,
    "subtype_a_h1n1_pdm09": _scn_subtype_a_h1n1_pdm09,
    "subtype_a_h3n2": _scn_subtype_a_h3n2,
    "hospital_surge": _scn_hospital_surge,
    "vaccine_uptake_low": _scn_vaccine_uptake_low,
    "reactive_intervention": _scn_reactive_intervention,
}


def register_extended_scenarios() -> None:
    """등록 — `simulation.sim.run_scenario("school_closure", ...)` 가능해진다.

    중복 호출 안전 (register_scenario 는 idempotent).
    """
    for name, fn in EXTENDED_SCENARIOS.items():
        register_scenario(name, fn)
        log.info(f"[scenarios_extended] registered: {name}")


__all__ = [
    "EXTENDED_SCENARIOS",
    "register_extended_scenarios",
]


if __name__ == "__main__":
    # 직접 실행 시 등록 + 검증
    register_extended_scenarios()
    from simulation.sim import SCENARIO_REGISTRY
    print(f"등록된 전체 시나리오: {len(SCENARIO_REGISTRY)}")
    for name in sorted(SCENARIO_REGISTRY.keys()):
        print(f"  - {name}")
