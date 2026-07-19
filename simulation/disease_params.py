"""
simulation/disease_params.py ( -- 전면 재설계)
===================================================
DB에서 질환 자동 탐지 + 데이터 기반 파라미터 추정 + 문헌값 보강

[ 문제점]
 - 7개 질환만 하드코딩
 - DB의 42개 활성 질환 무시
 - 파라미터 수정 불가

[ 개선]
 - DB disease_catalog 67개 + weekly_disease 42개 활성 질환 자동 탐지
 - 데이터에서 계절성, 추세, Re 자동 추정
 - 문헌값은 보강용으로만 사용 (override 가능)
 - 사용자가 임의 파라미터 입력 가능
 - 프롬프트에서 "R0=5, 잠복기=10일" 형식 파싱
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import sqlite3
import logging
from pathlib import Path

log = logging.getLogger(__name__)

import tempfile as _tempfile

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
# Fallback DB locations. Use tempfile.gettempdir() so this works on
# Windows (%TEMP%), macOS (/var/folders/...), and Linux (/tmp) alike.
_DB_CANDIDATES = [
    DATA_DIR / "db" / "epi_real_seoul.db",
    Path(_tempfile.gettempdir()) / "epi_sim" / "epi_real_seoul.db",
]


@dataclass
class DiseaseParams:
    """단일 질환의 역학적 파라미터."""

    name: str
    name_en: str = ""

    # ── 기본 역학 파라미터 ──
    R0_mean: float = 2.0
    R0_range: tuple[float, float] = (1.5, 3.0)
    latent_period: float = 5.0        # 일
    infectious_period: float = 7.0     # 일
    cfr: float = 0.001
    hospitalization_rate: float = 0.05
    asymptomatic_ratio: float = 0.20

    # ── 계절성 (데이터에서 자동 추정) ──
    seasonal_amplitude: float = 0.0
    peak_week: int = 0

    # ── 백신 ──
    vaccine_efficacy: float = 0.0
    baseline_vaccination_rate: float = 0.0

    # ── 데이터 기반 ──
    total_cases: int = 0
    data_weeks: int = 0
    year_range: tuple[int, int] = (2020, 2025)
    disease_group: str = ""
    transmission_route: str = "unknown"
    has_weekly_data: bool = False
    has_district_data: bool = False

    # ── 연령 ──
    age_susceptibility: dict = field(default_factory=dict)

    @property
    def sigma(self) -> float:
        return 1.0 / self.latent_period if self.latent_period > 0 else 0.0

    @property
    def gamma(self) -> float:
        return 1.0 / self.infectious_period if self.infectious_period > 0 else 0.0

    @property
    def beta(self) -> float:
        return self.R0_mean * self.gamma

    def Re(self, susceptible_fraction: float, interventions: dict = None) -> float:
        Re = self.R0_mean * susceptible_fraction
        if interventions:
            total_reduction = min(sum(interventions.values()), 0.95)
            Re *= (1 - total_reduction)
        return max(Re, 0.0)

    def seasonal_beta(self, day_of_year: int) -> float:
        if self.seasonal_amplitude == 0:
            return self.beta
        peak_day = (self.peak_week - 1) * 7 if self.peak_week > 0 else 0
        return self.beta * (
            1 + self.seasonal_amplitude * np.cos(
                2 * np.pi * (day_of_year - peak_day) / 365
            )
        )

    def override(self, **kwargs) -> 'DiseaseParams':
        """사용자 파라미터 오버라이드. 원본 복사 후 변경."""
        import copy
        new = copy.deepcopy(self)
        for k, v in kwargs.items():
            if hasattr(new, k):
                setattr(new, k, v)
        if "R0_mean" in kwargs and "R0_range" not in kwargs:
            r0 = kwargs["R0_mean"]
            new.R0_range = (r0 * 0.8, r0 * 1.2)
        return new


# ═══════════════════════════════════════════════════════════════════════════
# 문헌 기반 역학 파라미터 (보강용, DB에 없는 값)
# ═══════════════════════════════════════════════════════════════════════════

LITERATURE_PARAMS: dict[str, dict] = {
    "수두": dict(R0_mean=11.0, R0_range=(10, 12), latent_period=14, infectious_period=7,
                cfr=0.00003, vaccine_efficacy=0.85, baseline_vaccination_rate=0.97,
                transmission_route="airborne", name_en="Varicella"),
    "유행성이하선염": dict(R0_mean=5.5, R0_range=(4, 7), latent_period=17, infectious_period=9,
                    cfr=0.000002, vaccine_efficacy=0.88, baseline_vaccination_rate=0.97,
                    transmission_route="droplet", name_en="Mumps"),
    "성홍열": dict(R0_mean=6.0, R0_range=(5, 8), latent_period=3, infectious_period=5,
                cfr=0.00001, transmission_route="droplet", name_en="Scarlet Fever"),
    "카바페넴내성장내세균목(CRE) 감염증": dict(R0_mean=1.8, R0_range=(1.2, 2.5),
        latent_period=5, infectious_period=14, cfr=0.30, transmission_route="contact",
        name_en="CRE Infection"),
    "백일해": dict(R0_mean=13.0, R0_range=(12, 17), latent_period=9, infectious_period=21,
                cfr=0.0003, vaccine_efficacy=0.80, baseline_vaccination_rate=0.95,
                transmission_route="droplet", name_en="Pertussis"),
    "A형간염": dict(R0_mean=2.5, R0_range=(1.5, 3.5), latent_period=28, infectious_period=14,
                cfr=0.003, vaccine_efficacy=0.95, transmission_route="fecal-oral",
                name_en="Hepatitis A"),
    "C형간염": dict(R0_mean=2.0, R0_range=(1.5, 3), latent_period=42, infectious_period=180,
                cfr=0.01, transmission_route="blood-borne", name_en="Hepatitis C"),
    "쯔쯔가무시증": dict(R0_mean=0.0, latent_period=10, infectious_period=14,
                    cfr=0.005, transmission_route="vector", name_en="Scrub Typhus"),
    "홍역": dict(R0_mean=15.0, R0_range=(12, 18), latent_period=10, infectious_period=8,
              cfr=0.002, vaccine_efficacy=0.97, transmission_route="airborne", name_en="Measles"),
    "말라리아": dict(R0_mean=0.0, latent_period=12, infectious_period=14,
                  cfr=0.01, transmission_route="vector", name_en="Malaria"),
    "레지오넬라증": dict(R0_mean=0.0, latent_period=6, infectious_period=14,
                    cfr=0.10, transmission_route="environmental", name_en="Legionellosis"),
    "E형간염": dict(R0_mean=1.8, R0_range=(1.2, 2.5), latent_period=40, infectious_period=21,
                cfr=0.01, transmission_route="fecal-oral", name_en="Hepatitis E"),
    "매독": dict(R0_mean=3.0, R0_range=(2, 5), latent_period=21, infectious_period=60,
              cfr=0.005, transmission_route="sexual", name_en="Syphilis"),
    "폐렴구균 감염증": dict(R0_mean=2.5, R0_range=(1.5, 4), latent_period=3, infectious_period=10,
                      cfr=0.05, vaccine_efficacy=0.70, transmission_route="droplet",
                      name_en="Pneumococcal Disease"),
    "B형간염": dict(R0_mean=2.0, R0_range=(1.5, 3), latent_period=75, infectious_period=180,
                cfr=0.01, vaccine_efficacy=0.95, transmission_route="blood-borne",
                name_en="Hepatitis B"),
    "신증후군출혈열": dict(R0_mean=0.0, latent_period=14, infectious_period=10,
                    cfr=0.05, transmission_route="vector", name_en="HFRS"),
    "장출혈성대장균감염증": dict(R0_mean=3.0, R0_range=(2, 5), latent_period=4, infectious_period=7,
                        cfr=0.01, transmission_route="fecal-oral", name_en="EHEC"),
    "중증열성혈소판감소증후군(SFTS)": dict(R0_mean=0.0, latent_period=7, infectious_period=14,
        cfr=0.15, transmission_route="vector", name_en="SFTS"),
    "뎅기열": dict(R0_mean=0.0, latent_period=5, infectious_period=7,
                cfr=0.01, transmission_route="vector", name_en="Dengue"),
    "엠폭스": dict(R0_mean=2.0, R0_range=(1.5, 3), latent_period=9, infectious_period=21,
                cfr=0.03, vaccine_efficacy=0.85, transmission_route="contact", name_en="Mpox"),
    # G-128: MetapopSEIR 에서 get_disease_params("인플루엔자") 호출 시 필요
    "인플루엔자": dict(R0_mean=1.3, R0_range=(1.1, 1.5), latent_period=2, infectious_period=5,
                   cfr=0.001, vaccine_efficacy=0.50, baseline_vaccination_rate=0.35,
                   seasonal_amplitude=0.3, peak_week=5,
                   transmission_route="droplet", name_en="Influenza"),
}

# 기본 전파 경로별 기본값
ROUTE_DEFAULTS = {
    "airborne": dict(R0_mean=8.0, latent_period=10, infectious_period=7),
    "droplet": dict(R0_mean=4.0, latent_period=5, infectious_period=7),
    "contact": dict(R0_mean=2.0, latent_period=7, infectious_period=14),
    "fecal-oral": dict(R0_mean=2.5, latent_period=14, infectious_period=10),
    "vector": dict(R0_mean=0.0, latent_period=10, infectious_period=14),
    "blood-borne": dict(R0_mean=1.5, latent_period=30, infectious_period=90),
    "sexual": dict(R0_mean=2.5, latent_period=21, infectious_period=30),
    "environmental": dict(R0_mean=0.0, latent_period=7, infectious_period=14),
    "unknown": dict(R0_mean=2.0, latent_period=7, infectious_period=10),
}


# ═══════════════════════════════════════════════════════════════════════════
# DB 기반 자동 파라미터 추정
# ═══════════════════════════════════════════════════════════════════════════

def _find_db() -> Optional[Path]:
    for p in _DB_CANDIDATES:
        if p.exists():
            return p
    return None


def _estimate_seasonality(conn: sqlite3.Connection, disease_nm: str) -> tuple[float, int]:
    """주간 발생 데이터에서 계절 진폭과 피크 주 추정."""
    cur = conn.execute("""
        SELECT week_no, AVG(cases) as avg_cases
        FROM weekly_disease
        WHERE disease_nm = ? AND source_type = 'weekly_national'
          AND week_no IS NOT NULL AND cases > 0
        GROUP BY week_no
        ORDER BY week_no
    """, (disease_nm,))
    rows = cur.fetchall()
    if len(rows) < 20:
        return 0.0, 0

    weeks = [r[0] for r in rows]
    cases = np.array([r[1] for r in rows])

    if cases.mean() == 0:
        return 0.0, 0

    # 계절 진폭 = (max - min) / (max + min)
    amplitude = (cases.max() - cases.min()) / (cases.max() + cases.min() + 1e-8)
    peak_week = int(weeks[np.argmax(cases)])

    return round(float(amplitude), 3), peak_week


def _estimate_cfr(conn: sqlite3.Connection, disease_nm: str) -> float:
    """사망 데이터에서 CFR 추정."""
    cur = conn.execute("""
        SELECT SUM(deaths) FROM disease_death WHERE disease_nm = ?
    """, (disease_nm,))
    deaths_row = cur.fetchone()
    deaths = deaths_row[0] if deaths_row and deaths_row[0] else 0

    cur = conn.execute("""
        SELECT SUM(cases) FROM weekly_disease WHERE disease_nm = ? AND cases > 0
    """, (disease_nm,))
    cases_row = cur.fetchone()
    cases = cases_row[0] if cases_row and cases_row[0] else 0

    if cases > 0 and deaths > 0:
        return round(deaths / cases, 6)
    return 0.001  # 기본값


def discover_all_diseases() -> dict[str, DiseaseParams]:
    """
    DB에서 모든 질환을 자동 탐지하고 파라미터 생성.

    Returns
    -------
    dict[질환명, DiseaseParams]
    """
    db_path = _find_db()
    if not db_path:
        log.warning("[DiseaseParams] DB 없음 → 문헌값만 사용")
        return {name: DiseaseParams(name=name, **params)
                for name, params in LITERATURE_PARAMS.items()}

    # : safe_connect 로 일원화 (quick_check + WAL + tuning)
    from simulation.database import safe_connect
    conn = safe_connect(str(db_path))
    registry = {}

    # 주간 데이터가 있는 질환
    cur = conn.execute("""
        SELECT disease_nm, COUNT(*) as weeks, SUM(cases) as total,
               MIN(year) as y_min, MAX(year) as y_max
        FROM weekly_disease
        WHERE source_type = 'weekly_national' AND cases > 0
        GROUP BY disease_nm
        ORDER BY total DESC
    """)

    for row in cur.fetchall():
        disease_nm, weeks, total, y_min, y_max = row

        # 자치구 데이터 여부
        cur2 = conn.execute(
            "SELECT COUNT(*) FROM seoul_disease_district WHERE disease_nm = ?",
            (disease_nm,)
        )
        has_district = cur2.fetchone()[0] > 0

        # 계절성 추정
        amplitude, peak_week = _estimate_seasonality(conn, disease_nm)

        # CFR 추정
        cfr = _estimate_cfr(conn, disease_nm)

        # 문헌값 보강
        lit = LITERATURE_PARAMS.get(disease_nm, {})

        # 질환 그룹 조회
        cur3 = conn.execute(
            "SELECT disease_group FROM disease_catalog WHERE disease_nm = ?",
            (disease_nm,)
        )
        group_row = cur3.fetchone()
        disease_group = group_row[0] if group_row else ""

        # 전파 경로 기반 기본값
        route = lit.get("transmission_route", "unknown")
        defaults = ROUTE_DEFAULTS.get(route, ROUTE_DEFAULTS["unknown"])

        params = DiseaseParams(
            name=disease_nm,
            name_en=lit.get("name_en", disease_nm),
            R0_mean=lit.get("R0_mean", defaults["R0_mean"]),
            R0_range=lit.get("R0_range", (defaults["R0_mean"] * 0.8, defaults["R0_mean"] * 1.2)),
            latent_period=lit.get("latent_period", defaults["latent_period"]),
            infectious_period=lit.get("infectious_period", defaults["infectious_period"]),
            cfr=cfr if cfr > 0 else lit.get("cfr", 0.001),
            vaccine_efficacy=lit.get("vaccine_efficacy", 0.0),
            baseline_vaccination_rate=lit.get("baseline_vaccination_rate", 0.0),
            seasonal_amplitude=amplitude,
            peak_week=peak_week,
            total_cases=int(total),
            data_weeks=int(weeks),
            year_range=(int(y_min), int(y_max)),
            disease_group=disease_group,
            transmission_route=route,
            has_weekly_data=True,
            has_district_data=has_district,
        )

        registry[disease_nm] = params

    conn.close()

    # G-128: LITERATURE_PARAMS 에만 있고 DB weekly_disease 에 없는 질환
    # (예: "인플루엔자" 는 HIRA/ILI 로만 추적, weekly_disease 에는 없음)
    # → 문헌값으로만 DiseaseParams 생성해 Metapop-SEIR 호출 가능하게 함
    for lit_name, lit_params in LITERATURE_PARAMS.items():
        if lit_name in registry:
            continue
        route = lit_params.get("transmission_route", "unknown")
        defaults = ROUTE_DEFAULTS.get(route, ROUTE_DEFAULTS["unknown"])
        registry[lit_name] = DiseaseParams(
            name=lit_name,
            name_en=lit_params.get("name_en", lit_name),
            R0_mean=lit_params.get("R0_mean", defaults["R0_mean"]),
            R0_range=lit_params.get("R0_range",
                (lit_params.get("R0_mean", defaults["R0_mean"]) * 0.8,
                 lit_params.get("R0_mean", defaults["R0_mean"]) * 1.2)),
            latent_period=lit_params.get("latent_period", defaults["latent_period"]),
            infectious_period=lit_params.get("infectious_period", defaults["infectious_period"]),
            cfr=lit_params.get("cfr", 0.001),
            vaccine_efficacy=lit_params.get("vaccine_efficacy", 0.0),
            baseline_vaccination_rate=lit_params.get("baseline_vaccination_rate", 0.0),
            seasonal_amplitude=lit_params.get("seasonal_amplitude", 0.0),
            peak_week=lit_params.get("peak_week", 0),
            transmission_route=route,
            has_weekly_data=False,
            has_district_data=False,
        )

    log.info(f"[DiseaseParams] {len(registry)}개 질환 자동 탐지 완료 "
             f"(문헌-only 병합 후)")
    return registry


# ── 글로벌 레지스트리 (lazy load) ────────────────────────────────────────
_REGISTRY: Optional[dict[str, DiseaseParams]] = None


def get_registry() -> dict[str, DiseaseParams]:
    """전체 질환 레지스트리 (최초 호출 시 DB 탐색)."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = discover_all_diseases()
    return _REGISTRY


def get_disease_params(disease_nm: str) -> DiseaseParams:
    """질환명으로 파라미터 조회."""
    reg = get_registry()
    if disease_nm in reg:
        return reg[disease_nm]
    raise ValueError(f"미등록 질환: {disease_nm}. "
                     f"DB에 {len(reg)}개 질환 등록. "
                     f"create_custom_disease()로 임의 질환 생성 가능.")


def create_custom_disease(
    name: str,
    R0: float = 2.0,
    latent_period: float = 7.0,
    infectious_period: float = 10.0,
    **kwargs,
) -> DiseaseParams:
    """
    사용자 임의 질환 생성.

    예:
      create_custom_disease("신종바이러스X", R0=6.5, latent_period=5, infectious_period=10)
    """
    return DiseaseParams(
        name=name,
        name_en=kwargs.get("name_en", name),
        R0_mean=R0,
        R0_range=kwargs.get("R0_range", (R0 * 0.8, R0 * 1.2)),
        latent_period=latent_period,
        infectious_period=infectious_period,
        cfr=kwargs.get("cfr", 0.001),
        vaccine_efficacy=kwargs.get("vaccine_efficacy", 0.0),
        seasonal_amplitude=kwargs.get("seasonal_amplitude", 0.0),
        transmission_route=kwargs.get("transmission_route", "unknown"),
    )


def list_diseases(min_cases: int = 0, has_data: bool = True) -> list[str]:
    """등록 질환 목록 (필터링 가능)."""
    reg = get_registry()
    result = []
    for name, p in reg.items():
        if has_data and not p.has_weekly_data:
            continue
        if p.total_cases >= min_cases:
            result.append(name)
    return result


def list_diseases_summary() -> str:
    """질환 요약 (프롬프트용 텍스트)."""
    reg = get_registry()
    lines = [f"총 {len(reg)}개 질환 등록:\n"]
    for name, p in sorted(reg.items(), key=lambda x: -x[1].total_cases):
        lines.append(
            f"  {name}: {p.total_cases:,}건, {p.data_weeks}주, "
            f"R0={p.R0_mean}, 계절성={p.seasonal_amplitude:.2f}, "
            f"피크={p.peak_week}주"
        )
    return "\n".join(lines)
