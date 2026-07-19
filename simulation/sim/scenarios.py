"""
simulation.sim.scenarios
========================
Named scenario registry for the metapop SEIR-V-D simulator.

A "scenario" is a function ``(MetapopParams | None) -> (MetapopParams, list[InterventionSpec])``.
Scenarios can either take an explicit ``MetapopParams`` (e.g. pre-loaded
from the DB via ``simulation.sim.io``) or fall back to a synthetic
25-district example. In both cases the scenario returns the **resolved**
params + the intervention list to be applied at run time.

This module intentionally holds **no simulation logic** — it exists so
callers can ``run_scenario("vax_campaign")`` without knowing the
intervention-construction details.

Built-in scenarios (Stage 5 baseline set)
-----------------------------------------------
- ``baseline`` — no interventions, default flu biology
- ``npi_lockdown`` — 40 % β reduction, weeks 3–9
- ``vaccination_campaign`` — 0.5 %/day S→V draw, weeks 6–18
- ``antiviral_prophylaxis`` — γ doubled, weeks 4–10 (models early treatment)
- ``combined_response`` — NPI + vax + antiviral, staggered
- ``sensitivity_strain_mismatch`` — VE set to 0.20 (poor match year)

New scenarios register via ``register_scenario(name, fn)``.
"""
from __future__ import annotations

import logging
from dataclasses import replace
from typing import Callable, Optional

import numpy as np

from .parameters import (
    DEFAULT_FLU_PARAMS,
    DiseaseParams,
    InterventionSpec,
    MetapopParams,
)


log = logging.getLogger(__name__)

ScenarioBuilder = Callable[
    [Optional[MetapopParams]],
    tuple[MetapopParams, list[InterventionSpec]],
]

SCENARIO_REGISTRY: dict[str, ScenarioBuilder] = {}


def register_scenario(name: str, builder: ScenarioBuilder) -> None:
    """Register (or overwrite) a scenario builder. Idempotent; logs on overwrite."""
    if name in SCENARIO_REGISTRY:
        log.info("scenario %r overwritten", name)
    SCENARIO_REGISTRY[name] = builder


# ══════════════════════════════════════════════════════════════════════
# Default synthetic 25-district example
# ══════════════════════════════════════════════════════════════════════
def _default_params() -> MetapopParams:
    """25 equal-sized districts with a uniform-mixing mobility matrix
    (10 % commute-out split evenly among the 24 others, 90 % stay home).
    Seeds a single infectious case in district 0 to start the epidemic."""
    G = 25
    pops = np.full(G, 400_000.0)  # rough Seoul avg — 10M / 25
    # Row-stochastic M: diagonal = 0.9, off-diag = 0.1 / 24
    M = np.full((G, G), 0.1 / (G - 1))
    np.fill_diagonal(M, 0.9)

    infected0 = np.zeros(G)
    infected0[0] = 10.0

    return MetapopParams(
        disease=DEFAULT_FLU_PARAMS,
        populations=pops,
        mobility=M,
        district_names=[f"gu_{i:02d}" for i in range(G)],
        initial_infected=infected0,
        days=200,
        dt=0.25,
        seed=42,
    )


def _resolve(params: Optional[MetapopParams]) -> MetapopParams:
    """Return a *fresh copy* of params, or the default synthetic set.

    Uses ``dataclasses.replace`` so scenarios can freely mutate the
    returned object without side-effects on the caller's instance.
    """
    if params is None:
        return _default_params()
    # Shallow copy via replace() — keeps arrays as-is (they're not mutated
    # by the intervention layer; interventions operate on disease params).
    return replace(params)


# ══════════════════════════════════════════════════════════════════════
# Sprint 2026-05-06 Phase B.1 — propagation origin auto-resolve
# 사용자 critique: "전파 위치의 시작점이 인구밀집도 / 지하철이 높은 데서
# 하는게 좋지 않아? 그래서 실시간 업데이트를 반영한건데"
# ══════════════════════════════════════════════════════════════════════
def resolve_origin_gu(
    strategy: str = "pop_density_top1",
    *,
    db_path: str = "simulation/data/db/epi_real_seoul.db",
    fallback: str = "강남구",
) -> str:
    """전파 시작점 (origin gu) 결정 로직.

    Args:
        strategy:
            - "default" / "fixed": fallback gu (default 강남구, legacy)
            - "pop_density_top1": 실시간 인구밀집도 highest gu
              (`rt_population_detail.congestion_max` 최근 1일 평균 max)
            - "subway_hub": 지하철 ride_cnt highest gu (commuter morning rush)
            - "school_outbreak": 학교 휴교 활성 학교 위치 (school_closure_seoul)
        db_path: SQLite DB path (default 표준 위치)
        fallback: strategy 매칭 실패 시 default gu

    Returns:
        origin gu name (한글, e.g. "강남구")
    """
    if strategy in ("default", "fixed"):
        return fallback
    try:
        # safe_connect (G-116/G-117 enforced) — read-only access via
        # SELECT 만 사용, 단 module 의 quick_check + WAL handling 통과.
        from simulation.database import safe_connect
        con = safe_connect(db_path, timeout=5.0)
        try:
            if strategy == "pop_density_top1":
                row = con.execute(
                    """
                    SELECT area_nm, AVG(ppltn_max) AS pop_avg
                    FROM rt_population_detail
                    WHERE collected_at > datetime('now', '-1 day')
                    GROUP BY area_nm
                    ORDER BY pop_avg DESC
                    LIMIT 1
                    """
                ).fetchone()
                # area_nm → gu_nm mapping via poi_metadata (e.g. "강남역" → "강남구")
                if row:
                    poi = con.execute(
                        "SELECT area_nm FROM poi_metadata WHERE area_nm = ?",
                        (row[0],),
                    ).fetchone()
                    # poi_metadata 의 area_nm 은 명소 이름 — gu 매칭 위해
                    # daily_population_hotspot 통한 lookup
                    gu_row = con.execute(
                        """
                        SELECT gu_nm FROM daily_population_hotspot
                        WHERE area_nm = ?
                        ORDER BY stdr_de DESC LIMIT 1
                        """,
                        (row[0],),
                    ).fetchone()
                    return gu_row[0] if gu_row else fallback
            elif strategy == "subway_hub":
                row = con.execute(
                    """
                    SELECT station_nm, SUM(ride_cnt) AS rides
                    FROM monthly_subway_hourly
                    WHERE use_ym = (SELECT MAX(use_ym) FROM monthly_subway_hourly)
                      AND hour BETWEEN 7 AND 9
                    GROUP BY station_nm
                    ORDER BY rides DESC
                    LIMIT 1
                    """
                ).fetchone()
                if row:
                    # subway station → gu mapping (간단화: KCDC poi_metadata 사용)
                    return fallback  # TODO: subway_station_to_gu mapping table 별 sprint
            elif strategy == "school_outbreak":
                row = con.execute(
                    """
                    SELECT s.gu_name, COUNT(*) AS closures
                    FROM school_closure_seoul c
                    JOIN school_info_seoul s ON c.school_name = s.school_name
                    WHERE c.is_closure = 1
                      AND c.date > date('now', '-7 day')
                    GROUP BY s.gu_name
                    ORDER BY closures DESC
                    LIMIT 1
                    """
                ).fetchone()
                if row:
                    return row[0]
        finally:
            con.close()
    except Exception as e:
        log.warning("resolve_origin_gu(%r) fail: %s; fallback=%s", strategy, e, fallback)
    return fallback


def initial_infected_array(
    district_names: list[str],
    origin_gu: str,
    *,
    n_infected: float = 10.0,
) -> np.ndarray:
    """origin gu 의 index 에 n_infected 개 감염, 나머지는 0.

    district_names: ["강남구", "강동구", ...] (Seoul 25-gu order)
    origin_gu: 시작 gu (resolve_origin_gu() 결과)
    """
    G = len(district_names)
    inf = np.zeros(G)
    if origin_gu in district_names:
        inf[district_names.index(origin_gu)] = n_infected
    else:
        # fallback: gu 이름 mismatch 시 첫 gu (district 0)
        inf[0] = n_infected
        log.warning("origin_gu %r not in district_names; seeded gu_0 (%s)",
                    origin_gu, district_names[0] if district_names else "?")
    return inf


# ══════════════════════════════════════════════════════════════════════
# Built-in scenarios
# ══════════════════════════════════════════════════════════════════════
def _scn_baseline(params):
    return _resolve(params), []


def _scn_npi_lockdown(params):
    p = _resolve(params)
    interventions = [
        # weeks 3-9 ≈ days 21-63: β reduced to 60 % of baseline
        InterventionSpec(
            parameter="beta", value=0.60, op="scale",
            start_day=21, end_day=63,
            note="citywide NPI: distancing + masks",
        ),
    ]
    return p, interventions


def _scn_vaccination_campaign(params):
    p = _resolve(params)
    interventions = [
        # weeks 6-18 ≈ days 42-126: 0.5 % of S → V per day
        InterventionSpec(
            parameter="vaccination_rate", value=0.005, op="set",
            start_day=42, end_day=126,
            note="seasonal influenza vaccination campaign",
        ),
    ]
    return p, interventions


def _scn_antiviral_prophylaxis(params):
    p = _resolve(params)
    # β is the derived property R0·γ, so scaling γ alone leaves R_eff=R0
    # unchanged (transmission rate up proportionally) → cumulative cases
    # actually INCREASE. Oseltamixir has two clinical effects: viral-shedding
    # reduction (→ R0 down) and duration reduction (→ γ up). Model both.
    interventions = [
        # weeks 4-10 ≈ days 28-70
        InterventionSpec(
            parameter="R0", value=0.50, op="scale",
            start_day=28, end_day=70,
            note="oseltamivir: shedding reduction ~50 %",
        ),
        InterventionSpec(
            parameter="gamma", value=1.4, op="scale",
            start_day=28, end_day=70,
            note="oseltamivir: infectious period 3.5 d → 2.5 d",
        ),
    ]
    return p, interventions


def _scn_combined_response(params):
    p = _resolve(params)
    interventions = [
        InterventionSpec("beta", 0.70, 14, 56, op="scale",
                         note="moderate NPI"),
        InterventionSpec("vaccination_rate", 0.004, 35, 140, op="set",
                         note="staggered vax rollout"),
        InterventionSpec("gamma", 1.5, 28, 70, op="scale",
                         note="antiviral stockpile deployment"),
    ]
    return p, interventions


def _scn_strain_mismatch(params):
    p = _resolve(params)
    # Poor vaccine match — override VE for the whole run
    interventions = [
        InterventionSpec("VE", 0.20, 0, 10_000, op="set",
                         note="strain mismatch — VE ≈ 20 %"),
    ]
    return p, interventions


# Pre-populate the registry
for _name, _fn in [
    ("baseline", _scn_baseline),
    ("npi_lockdown", _scn_npi_lockdown),
    ("vaccination_campaign", _scn_vaccination_campaign),
    ("antiviral_prophylaxis", _scn_antiviral_prophylaxis),
    ("combined_response", _scn_combined_response),
    ("sensitivity_strain_mismatch", _scn_strain_mismatch),
]:
    register_scenario(_name, _fn)
