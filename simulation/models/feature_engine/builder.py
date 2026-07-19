"""
Builder and orchestration functions for the feature engineering pipeline.

- build_enriched_features: Main pipeline orchestrator
- select_features: Feature selection via mutual information
- _categorize_features: Feature categorization into groups
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import polars as pl
from sklearn.feature_selection import mutual_info_regression

from simulation.config_global import GLOBAL  # SSOT (2026-05-28)

from .loaders import (
    _load_sentinel_ili, _load_weather, _load_vaccination,
    _load_sentinel_ari, _load_weekly_disease,
    _load_daily_population_district, _load_daily_subway, _load_daily_bus,
    _load_daily_population_hotspot, _load_school_info, _load_hospitals,
    _load_employment_workplace, _load_employment_monthly,
    _load_hira_gender_age, _load_daily_population_gu_hourly,
    _load_weather_forecast,
    # 미연결 테이블 로더 (2026-04-11b)
    _load_monthly_subway_hourly, _load_monthly_bus_hourly,
    _load_sentinel_sari, _load_sentinel_hfmd, _load_sentinel_enterovirus,
    _load_hira_inpat_opat, _load_hira_region_seoul,
    _load_childhood_vaccination, _load_daily_population_dong_agg,
    _load_employment_residence,
    # 확장 실시간 테이블 로더 (2026-04-11)
    # 2026-04-17: _load_rt_bike_status 제거 (따릉이 피처 제외 지시)
    _load_rt_road_traffic, _load_rt_subway_crowd,
    _load_rt_population_detail,
    _load_rt_population_forecast,
    _load_rt_spatial_aggregation,
    _load_rt_temporal_patterns,
    # 환경 피처 (2026-04-20: 사용자 요청 — weather/pollution 미연결 로더 연결)
    _load_rt_air_quality, _load_rt_sdot_env,
    # 추가 데이터소스 (2026-04-11d)
    _load_google_search_trends, _load_school_closure,
    # Phase D (2026-05-06): G1=A3 + B.4=mixc implementation
    _load_flunet_positivity, _load_korean_holiday,
)
from .transforms import (
    _add_lag_features, _add_rolling_features, _add_diff_features,
    _add_log_features, _add_quantile_encoding, _add_binary_encoding,
    _add_multi_resolution_seasonal, _add_wavelet_features,
    _add_interaction_features, _add_epidemic_phase_features,
    _add_multi_resolution_agg
)

log = logging.getLogger(__name__)


def _categorize_features(cols: list[str]) -> dict[str, list[str]]:
    """피처를 그룹별로 분류."""
    groups = {
        "lag": [], "rolling": [], "diff": [],
        "age_group": [], "weather": [], "population": [],
        "vaccination": [], "pathogen": [], "disease": [],
        "air_quality": [], "iot_env": [], "transport": [], "transport_hourly": [],
        "hourly_population": [], "dong_population": [],
        "emergency": [], "school": [], "employment": [],
        "hospital_infra": [], "hotspot": [], "hira_utilization": [],
        "sentinel_other": [], "childhood_vax": [],
        "rt_estimation": [], "rt_road": [], "rt_subway": [],
        "rt_popdetail": [], "rt_forecast": [], "rt_spatial": [], "rt_temporal": [],
        "google_trends": [], "school_closure": [],
        "multi_resolution": [],
        "log": [], "quantile": [],
        "binary": [], "seasonal": [], "wavelet": [],
        "interaction": [], "epidemic": [], "other": [],
    }

    for c in cols:
        if c.startswith("mr_"):
            groups["multi_resolution"].append(c)
        elif c.startswith("ari_"):
            groups["pathogen"].append(c)
        elif c.startswith("wd_"):
            groups["disease"].append(c)
        elif c.startswith("pm_"):
            groups["air_quality"].append(c)
        elif c.startswith("subway_") or c.startswith("bus_"):
            groups["transport"].append(c)
        elif c.startswith("sub_rush") or c.startswith("sub_night") or c.startswith("sub_hourly") or c.startswith("bus_rush") or c.startswith("bus_hourly"):
            groups["transport_hourly"].append(c)
        elif c.startswith("dong_"):
            groups["dong_population"].append(c)
        elif c in ("sari_count", "hfmd_rate", "enterovirus_count"):
            groups["sentinel_other"].append(c)
        elif c.startswith("child_vax"):
            groups["childhood_vax"].append(c)
        elif c.startswith("hpop_") or c.startswith("wp_"):
            groups["hourly_population"].append(c)
        elif c.startswith("er_"):
            groups["emergency"].append(c)
        elif c.startswith("sch_"):
            groups["school"].append(c)
        elif c.startswith("emp_"):
            groups["employment"].append(c)
        elif c.startswith("hosp_"):
            groups["hospital_infra"].append(c)
        elif c.startswith("hs_"):
            groups["hotspot"].append(c)
        elif c.startswith("hira_"):
            groups["hira_utilization"].append(c)
        elif c.startswith("gt_"):
            groups["google_trends"].append(c)
        elif c.startswith("sch_closure"):
            groups["school_closure"].append(c)
        elif c.startswith("rt_road_"):
            groups["rt_road"].append(c)
        elif c.startswith("rt_sub_"):
            groups["rt_subway"].append(c)
        elif c.startswith("rt_popdet_"):
            groups["rt_popdetail"].append(c)
        # 2026-04-17: rt_bike_ 분류 제거 (따릉이 피처 제외)
        elif c.startswith("rt_fcst_"):
            groups["rt_forecast"].append(c)
        elif c.startswith("rt_spatial_"):
            groups["rt_spatial"].append(c)
        elif c.startswith("rt_temp_"):
            groups["rt_temporal"].append(c)
        elif c.startswith("rt_pm") or c.startswith("rt_o3") or c.startswith("rt_no2") \
                or c.startswith("rt_co_") or c.startswith("rt_air_"):
            groups["air_quality"].append(c)
        elif c.startswith("rt_sdot_"):
            groups["iot_env"].append(c)
        elif c.startswith("rt_"):
            groups["rt_estimation"].append(c)
        elif c.startswith("wf_"):
            groups["weather"].append(c)
        elif "_lag" in c:
            groups["lag"].append(c)
        elif "rmean" in c or "rstd" in c or "rmin" in c or "rmax" in c:
            groups["rolling"].append(c)
        elif "_diff" in c:
            groups["diff"].append(c)
        elif c.startswith("ili_age_"):
            groups["age_group"].append(c)
        elif c in ("temp_avg", "temp_min", "humidity", "wind_speed", "temp_std") or "temp_" in c or c.startswith("humid"):
            groups["weather"].append(c)
        elif c.startswith("pop_"):
            groups["population"].append(c)
        elif "vax" in c:
            groups["vaccination"].append(c)
        elif "_log1p" in c:
            groups["log"].append(c)
        elif "_qbin" in c or "_qnorm" in c:
            groups["quantile"].append(c)
        elif "_bit" in c:
            groups["binary"].append(c)
        elif "sin_" in c or "cos_" in c:
            groups["seasonal"].append(c)
        elif "_wavelet" in c:
            groups["wavelet"].append(c)
        elif c in ("cold_ili", "humid_ili", "inflow_ili",
                    "subway_ili", "bus_ili", "peak_ratio_ili", "er_burden_ili",
                    "sch_session_ili", "emp_contact_ili", "wp_inflow_ili",
                    "hs_congestion_ili",
                    "rt_subcrowd_ili", "rt_roadcong_ili",
                    "rt_nonresnt_ili", "rt_highrisk_ili"):
            groups["interaction"].append(c)
        elif c in ("above_threshold", "consec_rise", "season_cum_ili"):
            groups["epidemic"].append(c)
        else:
            groups["other"].append(c)

    return {k: v for k, v in groups.items() if v}


def load_optuna_features(model_type: str, results_dir: str = "results") -> Optional[list[str]]:
    """Optuna 피처 선택 결과 파일에서 선택된 피처 목록 로드.

    Args:
        model_type: 모델 타입 (lightgbm, xgboost, elasticnet, dnn 등)
        results_dir: 결과 파일 디렉토리

    Returns:
        선택된 피처 이름 리스트 또는 None (파일 없을 시)
    """
    import json
    from pathlib import Path

    # Cross-platform Optuna result search. Priority order:
    #   1. Caller-supplied `results_dir`
    #   2. $MPH_OUTPUT_ROOT/results/ (env override — e.g. external disk)
    #   3. <repo>/simulation/results/ (project-local default)
    #   4. ./results/ (legacy relative path)
    from simulation.utils.paths import get_optuna_dir
    search_paths = [
        Path(results_dir) / f"optuna_feat_sel_{model_type}.json",
        get_optuna_dir() / f"optuna_feat_sel_{model_type}.json",
        Path("simulation/results") / f"optuna_feat_sel_{model_type}.json",
        Path("results") / f"optuna_feat_sel_{model_type}.json",
    ]

    for path in search_paths:
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                features = data.get("selected_features", [])
                composite = data.get("composite_features", [])
                strategy = data.get("strategy", "unknown")
                if features:
                    log.info(f"  Optuna 피처 로드: {path.name} → {len(features)}개 "
                             f"(+{len(composite)}개 복합변수, 전략={strategy})")
                    return features + composite
            except Exception as e:
                log.warning(f"  Optuna 결과 파일 파싱 실패: {path} — {e}")
    return None


# Optuna 모델명 → 파일명 매핑 (1:1 개별 매핑 우선, fallback은 대표 타입)
# individual 모드 실행 시: 각 모델별 optuna_feat_sel_{key}.json 로드
# representative 모드 실행 시: 대표 4개(lightgbm/xgboost/elasticnet/dnn)만 존재
_OPTUNA_MODEL_MAP_INDIVIDUAL = {
    # Tree
    "LightGBM": "lightgbm",
    "XGBoost": "xgboost",
    "RandomForest": "randomforest",
    "GradientBoosting": "gradientboosting",
    # Linear
    "ElasticNet": "elasticnet",
    "SVR-RBF": "svr_rbf",
    "SVR-Linear": "svr_linear",
    "KRR": "krr",
    # DL
    "DNN": "dnn", "DNN-Optuna": "dnn", "DNN-Conformal": "dnn",
    # Modern TS → DNN 피처 공유 (구조 유사)
    "TCN": "dnn", "TCN-Optuna": "dnn",
    "N-BEATS": "dnn", "N-HiTS": "dnn",
    "TFT": "dnn", "PatchTST": "dnn",
    "iTransformer": "dnn", "TiDE": "dnn",
    "Mamba": "dnn", "TimesNet": "dnn",
    # Epi/Bayesian
    "GP-RBF-Periodic": "gp_rbf_periodic",
    "BayesianRidge": "bayesianridge",
    "NegBinGLM": "negbinglm",
    "BayesianMCMC": "bayesianmcmc",
    "PoissonAutoreg": "poissonautoreg",
    "GAM": "gam",
    # Graph / Tabular
    "GE-DNN": "ge_dnn",
    "TabularDNN": "tabular_dnn",
}

# 기본 매핑: individual 우선 시도
_OPTUNA_MODEL_MAP = _OPTUNA_MODEL_MAP_INDIVIDUAL


def select_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    max_features: int = 40,
    method: str = "mutual_info",
    model_name: Optional[str] = None,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """피처 선택 -- Optuna 결과 우선, 없으면 MI 기반 상위 K개.

    Args:
        model_name: 모델 이름 (Optuna 결과 매칭용, optional)
    """

    # 1) Optuna 결과가 있으면 우선 사용
    if model_name:
        optuna_type = _OPTUNA_MODEL_MAP.get(model_name)
        if optuna_type:
            optuna_features = load_optuna_features(optuna_type)
            if optuna_features:
                # Optuna 선택 피처 중 현재 피처셋에 있는 것만 사용
                valid_features = [f for f in optuna_features if f in feature_names]
                if len(valid_features) >= 5:  # 최소 5개 이상이어야 유효
                    sel_idx = [feature_names.index(f) for f in valid_features]
                    sel_idx = sorted(sel_idx)
                    selected_names = [feature_names[i] for i in sel_idx]
                    # MI 스코어는 참고용으로 계산
                    try:
                        mi = mutual_info_regression(X[:, sel_idx], y, random_state=42, n_neighbors=5)
                    except Exception:
                        mi = np.ones(len(sel_idx))
                    log.info(f"  피처 선택 (Optuna): {X.shape[1]}개 → {len(sel_idx)}개 "
                             f"(model={model_name}, optuna_type={optuna_type})")
                    return X[:, sel_idx], selected_names, mi
                else:
                    log.warning(f"  Optuna 피처 유효 수 부족 ({len(valid_features)}개) → MI fallback")

    # 2) MI 기반 폴백
    if X.shape[1] <= max_features:
        return X, feature_names, np.ones(X.shape[1])

    if method == "mutual_info":
        mi = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
    elif method == "correlation":
        mi = np.array([abs(np.corrcoef(X[:, i], y)[0, 1]) if np.std(X[:, i]) > 0 else 0
                        for i in range(X.shape[1])])
    else:
        raise ValueError(f"Unknown method: {method}")

    top_idx = np.argsort(mi)[::-1][:max_features]
    top_idx = np.sort(top_idx)

    selected_names = [feature_names[i] for i in top_idx]
    scores = mi[top_idx]

    log.info(f"  피처 선택 (MI): {X.shape[1]}개 → {len(top_idx)}개 (method={method})")
    sorted_pairs = sorted(zip(selected_names, scores), key=lambda x: -x[1])
    for name, score in sorted_pairs[:10]:
        log.info(f"    {name:30s} MI={score:.4f}")

    return X[:, top_idx], selected_names, scores


def build_enriched_features(
    db_path: str = "data/db/epi_real_seoul.db",
    include_weather: bool = True,
    include_vaccination: bool = True,
    include_sentinel_ari: bool = True,
    include_weekly_disease: bool = True,
    include_population_district: bool = True,
    include_subway: bool = True,
    include_bus: bool = True,
    include_hotspot: bool = True,
    include_school: bool = True,
    include_hospitals: bool = True,
    include_employment: bool = True,
    include_hira: bool = True,
    include_hourly_population: bool = True,
    include_weather_forecast: bool = True,
    # rename: 실제 로더는 monthly_subway_hourly / monthly_bus_hourly.
    include_monthly_subway_hourly: bool = True,
    include_monthly_bus_hourly: bool = True,
    include_sentinel_sari: bool = True,  # re-collected (52 rows, limited 2019-2020)
    include_sentinel_hfmd: bool = True,
    include_sentinel_enterovirus: bool = True,
    include_hira_inpat: bool = True,
    include_hira_region: bool = True,
    include_childhood_vax: bool = True,
    include_dong_population: bool = True,
    include_emp_residence: bool = True,
    include_rt_road: bool = True,          # re-collected via citydata API
    include_rt_subway_crowd: bool = True,  # re-collected via citydata API
    include_rt_pop_detail: bool = True,    # re-collected via citydata API
    # 2026-04-17: include_rt_bike 제거 (따릉이 피처 제외 지시)
    include_rt_pop_forecast: bool = True,
    include_rt_spatial: bool = True,       # spatial aggregation across 115 POIs
    include_rt_temporal: bool = True,      # temporal/hourly patterns from RT data
    # 2026-04-20 — air pollution (PM10/PM2.5/O3/NO2/CO) + IoT env (temp/humid/noise/wind)
    include_rt_air_quality: bool = True,
    include_rt_sdot_env: bool = True,
    include_google_trends: bool = True,
    include_school_closure: bool = True,
    # Phase D (sprint 2026-05-06):
    #   D.3 = G1=A3 (FluNet positivity + subtype share + influenza-only proxy)
    #   D.2 = Korean holiday clinic closure (lag1)
    #   D.5 = NPI intervention dummy (covid_npi_period + npi_level, B.4 B layer)
    #   D.4 = lag52 + EWM + Fourier seasonality
    include_flunet_positivity: bool = True,
    include_korean_holiday: bool = True,
    # 사용자 통찰 (2026-05-06): NPI dummy = catch-all confounder 위험 (Wagner 2002
    # §3.1 critique). 기존 mechanism-aware features (rt_subway_crowd, bus,
    # school_closure, holiday) 가 NPI 효과 자동 capture → dummy 회피.
    # default False; sensitivity analysis 시만 명시 enable.
    include_npi_dummy: bool = False,
    include_seasonal_extra: bool = True,
    train_ratio: float = 0.8,
    log_transform: bool = True,
    quantile_encode: bool = True,
    binary_encode: bool = True,
    wavelet_features: bool = True,
    multi_resolution: bool = True,
    interaction_features: bool = True,
    epidemic_phase: bool = True,
    multi_resolution_agg: bool = True,
    advanced_features: Optional[bool] = None,   # 2026-04-28: env-gated
) -> tuple[pl.DataFrame, dict]:
    """
    전체 피처 엔지니어링 파이프라인.

    Returns:
        (feat_df, meta_info)
        - feat_df: 모든 피처 + 타겟("ili_rate") 포함 DataFrame (polars)
        - meta_info: 피처 수, 행 수, 피처 그룹 등
    """
    log.info("=== 고급 피처 엔지니어링 시작 ===")

    # ── A. ILI rate 로드 ──
    df = _load_sentinel_ili(db_path)
    log.info(f"  ILI data: {len(df)}행, {df['season_start'].n_unique()}시즌")

    # ── B. 기본 외부 데이터 결합 ──
    if include_weather:
        try:
            weather = _load_weather(db_path)
            df = df.with_columns([
                (pl.col("cal_date") - pl.duration(days=pl.col("cal_date").dt.weekday()))
                .cast(pl.Datetime("us")).alias("week_start")
            ])
            df = df.join(weather, on="week_start", how="left")

            for c in ["temp_avg", "temp_min", "humidity", "wind_speed", "temp_std"]:
                if c in df.columns:
                    df = df.with_columns([
                        pl.col(c).interpolate().forward_fill().backward_fill().alias(c)
                    ])
            log.info(f"  기상 데이터 결합 완료")
        except Exception as e:
            log.warning(f"  기상 데이터 결합 실패: {e}")

    if include_vaccination:
        try:
            vax = _load_vaccination(db_path)
            df = df.join(vax, left_on="season_start", right_on="year", how="left")
            df = df.with_columns([
                pl.col("vax_coverage").forward_fill().backward_fill().alias("vax_coverage")
            ])
            log.info(f"  접종률 데이터 결합 완료")
        except Exception as e:
            log.warning(f"  접종률 데이터 결합 실패: {e}")

    # Ensure week_start column exists for subsequent joins
    if "week_start" not in df.columns:
        df = df.with_columns([
            (pl.col("cal_date") - pl.duration(days=pl.col("cal_date").dt.weekday()))
            .cast(pl.Datetime("us")).alias("week_start")
        ])

    # ── B-ext. 추가 데이터 소스 ──

    if include_sentinel_ari:
        try:
            ari = _load_sentinel_ari(db_path)
            if not ari.is_empty():
                df = df.join(ari, on="week_start", how="left")
                log.info(f"  급성호흡기감염병 데이터 결합 완료")
        except Exception as e:
            log.warning(f"  급성호흡기감염병 데이터 결합 실패: {e}")

    if include_weekly_disease:
        try:
            dis = _load_weekly_disease(db_path)
            if not dis.is_empty():
                df = df.join(dis, on="week_start", how="left")
                log.info(f"  주간 감염병 데이터 결합 완료")
        except Exception as e:
            log.warning(f"  주간 감염병 데이터 결합 실패: {e}")

    if include_population_district:
        try:
            pop = _load_daily_population_district(db_path)
            if not pop.is_empty():
                df = df.join(pop, on="week_start", how="left")
                for c in ["pop_total_avg", "pop_daytime_avg", "pop_inflow_avg"]:
                    if c in df.columns:
                        df = df.with_columns([
                            pl.col(c).interpolate().forward_fill().backward_fill().alias(c)
                        ])
                log.info(f"  생활인구(자치구) 데이터 결합 완료")
        except Exception as e:
            log.warning(f"  생활인구(자치구) 데이터 결합 실패: {e}")

    if include_subway:
        try:
            subway = _load_daily_subway(db_path)
            if not subway.is_empty():
                df = df.join(subway, on="week_start", how="left")
                for c in ["subway_ride_avg", "subway_alight_avg", "subway_total_avg"]:
                    if c in df.columns:
                        df = df.with_columns([
                            pl.col(c).interpolate().forward_fill().backward_fill().alias(c)
                        ])
                log.info(f"  지하철 데이터 결합 완료")
        except Exception as e:
            log.warning(f"  지하철 데이터 결합 실패: {e}")

    if include_bus:
        try:
            bus = _load_daily_bus(db_path)
            if not bus.is_empty():
                df = df.join(bus, on="week_start", how="left")
                for c in ["bus_ride_avg", "bus_alight_avg", "bus_total_avg"]:
                    if c in df.columns:
                        df = df.with_columns([
                            pl.col(c).interpolate().forward_fill().backward_fill().alias(c)
                        ])
                log.info(f"  버스 데이터 결합 완료")
        except Exception as e:
            log.warning(f"  버스 데이터 결합 실패: {e}")

    if include_hotspot:
        try:
            hotspot = _load_daily_population_hotspot(db_path)
            if not hotspot.is_empty():
                df = df.join(hotspot, on="week_start", how="left")
                for c in ["hotspot_congestion_max", "hotspot_congestion_avg", "hotspot_ppltn_peak", "hotspot_ppltn_avg"]:
                    if c in df.columns:
                        df = df.with_columns([
                            pl.col(c).interpolate().forward_fill().backward_fill().alias(c)
                        ])
                log.info(f"  핫스팟 인구 데이터 결합 완료")
        except Exception as e:
            log.warning(f"  핫스팟 인구 데이터 결합 실패: {e}")

    if include_school:
        try:
            school = _load_school_info(db_path)
            if not school.is_empty():
                school = school.drop("week_start")
                for col in school.columns:
                    df = df.with_columns([
                        pl.lit(school[col].item(0)).alias(col)
                    ])
                log.info(f"  학교 정보 데이터 결합 완료")
        except Exception as e:
            log.warning(f"  학교 정보 데이터 결합 실패: {e}")

    if include_hospitals:
        try:
            hosp = _load_hospitals(db_path)
            if not hosp.is_empty():
                hosp = hosp.drop("week_start")
                for col in hosp.columns:
                    df = df.with_columns([
                        pl.lit(hosp[col].item(0)).alias(col)
                    ])
                log.info(f"  병원 정보 데이터 결합 완료")
        except Exception as e:
            log.warning(f"  병원 정보 데이터 결합 실패: {e}")

    if include_employment:
        try:
            emp_wp = _load_employment_workplace(db_path)
            emp_mn = _load_employment_monthly(db_path)
            if not emp_wp.is_empty():
                df = df.join(emp_wp, on="week_start", how="left")
                log.info(f"  산업인구(직장) 데이터 결합 완료")
            if not emp_mn.is_empty():
                df = df.join(emp_mn, on="week_start", how="left")
                log.info(f"  산업인구(월별) 데이터 결합 완료")
            for c in df.columns:
                if c.startswith("emp_"):
                    df = df.with_columns([
                        pl.col(c).interpolate().forward_fill().backward_fill().alias(c)
                    ])
        except Exception as e:
            log.warning(f"  산업인구 데이터 결합 실패: {e}")

    if include_hira:
        try:
            hira = _load_hira_gender_age(db_path)
            if not hira.is_empty():
                df = df.join(hira, on="week_start", how="left")
                for c in ["hira_visits_avg", "hira_visits_total"]:
                    if c in df.columns:
                        df = df.with_columns([
                            pl.col(c).interpolate().forward_fill().backward_fill().alias(c)
                        ])
                log.info(f"  HIRA 외래 방문 데이터 결합 완료")
        except Exception as e:
            log.warning(f"  HIRA 외래 방문 데이터 결합 실패: {e}")

    if include_hourly_population:
        try:
            hourly = _load_daily_population_gu_hourly(db_path)
            if not hourly.is_empty():
                df = df.join(hourly, on="week_start", how="left")
                for c in ["hourly_pop_avg", "hourly_pop_peak"]:
                    if c in df.columns:
                        df = df.with_columns([
                            pl.col(c).interpolate().forward_fill().backward_fill().alias(c)
                        ])
                log.info(f"  시간대별 인구 데이터 결합 완료")
        except Exception as e:
            log.warning(f"  시간대별 인구 데이터 결합 실패: {e}")

    if include_weather_forecast:
        try:
            fcst = _load_weather_forecast(db_path)
            if not fcst.is_empty():
                df = df.join(fcst, on="week_start", how="left")
                for c in df.columns:
                    if c.startswith("fcst_"):
                        df = df.with_columns([
                            pl.col(c).interpolate().forward_fill().backward_fill().alias(c)
                        ])
                log.info(f"  기상 예보 데이터 결합 완료")
        except Exception as e:
            log.warning(f"  기상 예보 데이터 결합 실패: {e}")

    # ── B-ext2. 미연결 테이블 (2026-04-11b 전수 점검 후 추가) ──

    if include_monthly_subway_hourly:
        try:
            sub_h = _load_monthly_subway_hourly(db_path)
            if not sub_h.is_empty():
                df = df.join(sub_h, on="week_start", how="left")
                for c in ["sub_rush_ratio", "sub_night_ratio", "sub_hourly_total"]:
                    if c in df.columns:
                        df = df.with_columns([
                            pl.col(c).interpolate().forward_fill().backward_fill().alias(c)
                        ])
                log.info(f"  지하철 시간대별 밀집 데이터 결합 완료")
        except Exception as e:
            log.warning(f"  지하철 시간대별 밀집 결합 실패: {e}")

    if include_monthly_bus_hourly:
        try:
            bus_h = _load_monthly_bus_hourly(db_path)
            if not bus_h.is_empty():
                df = df.join(bus_h, on="week_start", how="left")
                for c in ["bus_rush_ratio", "bus_hourly_total"]:
                    if c in df.columns:
                        df = df.with_columns([
                            pl.col(c).interpolate().forward_fill().backward_fill().alias(c)
                        ])
                log.info(f"  버스 시간대별 밀집 데이터 결합 완료")
        except Exception as e:
            log.warning(f"  버스 시간대별 밀집 결합 실패: {e}")

    if include_sentinel_sari:
        try:
            sari = _load_sentinel_sari(db_path)
            if not sari.is_empty():
                df = df.join(sari, on="week_start", how="left")
                if "sari_count" in df.columns:
                    df = df.with_columns([
                        pl.col("sari_count").fill_null(0).alias("sari_count")
                    ])
                log.info(f"  SARI(중증호흡기) 데이터 결합 완료")
        except Exception as e:
            log.warning(f"  SARI 결합 실패: {e}")

    if include_sentinel_hfmd:
        try:
            hfmd = _load_sentinel_hfmd(db_path)
            if not hfmd.is_empty():
                df = df.join(hfmd, on="week_start", how="left")
                if "hfmd_rate" in df.columns:
                    df = df.with_columns([
                        pl.col("hfmd_rate").fill_null(0).alias("hfmd_rate")
                    ])
                log.info(f"  수족구병 데이터 결합 완료")
        except Exception as e:
            log.warning(f"  수족구병 결합 실패: {e}")

    if include_sentinel_enterovirus:
        try:
            entero = _load_sentinel_enterovirus(db_path)
            if not entero.is_empty():
                df = df.join(entero, on="week_start", how="left")
                if "enterovirus_count" in df.columns:
                    df = df.with_columns([
                        pl.col("enterovirus_count").fill_null(0).alias("enterovirus_count")
                    ])
                log.info(f"  엔테로바이러스 데이터 결합 완료")
        except Exception as e:
            log.warning(f"  엔테로바이러스 결합 실패: {e}")

    if include_hira_inpat:
        try:
            hira_io = _load_hira_inpat_opat(db_path)
            if not hira_io.is_empty():
                df = df.join(hira_io, on="week_start", how="left")
                for c in ["hira_inpat_ratio", "hira_inpat_count"]:
                    if c in df.columns:
                        df = df.with_columns([
                            pl.col(c).interpolate().forward_fill().backward_fill().alias(c)
                        ])
                log.info(f"  HIRA 입원/외래 데이터 결합 완료")
        except Exception as e:
            log.warning(f"  HIRA 입원/외래 결합 실패: {e}")

    if include_hira_region:
        try:
            hira_r = _load_hira_region_seoul(db_path)
            if not hira_r.is_empty():
                df = df.join(hira_r, on="week_start", how="left")
                for c in ["hira_seoul_patients", "hira_seoul_visits"]:
                    if c in df.columns:
                        df = df.with_columns([
                            pl.col(c).interpolate().forward_fill().backward_fill().alias(c)
                        ])
                log.info(f"  HIRA 서울 지역 데이터 결합 완료")
        except Exception as e:
            log.warning(f"  HIRA 서울 지역 결합 실패: {e}")

    if include_childhood_vax:
        try:
            cvax = _load_childhood_vaccination(db_path)
            if not cvax.is_empty():
                df = df.join(cvax, on="week_start", how="left")
                for c in ["child_vax_avg", "child_vax_min"]:
                    if c in df.columns:
                        df = df.with_columns([
                            pl.col(c).interpolate().forward_fill().backward_fill().alias(c)
                        ])
                log.info(f"  소아 접종률 데이터 결합 완료")
        except Exception as e:
            log.warning(f"  소아 접종률 결합 실패: {e}")

    if include_dong_population:
        try:
            dong_pop = _load_daily_population_dong_agg(db_path)
            if not dong_pop.is_empty():
                df = df.join(dong_pop, on="week_start", how="left")
                for c in ["dong_pop_avg", "dong_pop_std", "dong_child_ratio",
                           "dong_elderly_ratio", "dong_70p_ratio"]:
                    if c in df.columns:
                        df = df.with_columns([
                            pl.col(c).interpolate().forward_fill().backward_fill().alias(c)
                        ])
                log.info(f"  행정동 인구구조 데이터 결합 완료")
        except Exception as e:
            log.warning(f"  행정동 인구구조 결합 실패: {e}")

    if include_emp_residence:
        try:
            emp_r = _load_employment_residence(db_path)
            if not emp_r.is_empty():
                df = df.join(emp_r, on="week_start", how="left")
                if "emp_residence_avg" in df.columns:
                    df = df.with_columns([
                        pl.col("emp_residence_avg").interpolate().forward_fill().backward_fill()
                    ])
                log.info(f"  거주지 기준 고용 데이터 결합 완료")
        except Exception as e:
            log.warning(f"  거주지 기준 고용 결합 실패: {e}")

    # ── B-rt. 확장 실시간 데이터 (2026-04-11) ──

    if include_rt_road:
        try:
            rt_road = _load_rt_road_traffic(db_path)
            if not rt_road.is_empty():
                df = df.join(rt_road, on="week_start", how="left")
                for c in ["rt_road_cong_avg", "rt_road_spd_avg", "rt_road_spd_min"]:
                    if c in df.columns:
                        df = df.with_columns([
                            pl.col(c).interpolate().forward_fill().backward_fill().alias(c)
                        ])
                log.info(f"  도로소통 데이터 결합 완료")
        except Exception as e:
            log.warning(f"  도로소통 데이터 결합 실패: {e}")

    if include_rt_subway_crowd:
        try:
            rt_sub = _load_rt_subway_crowd(db_path)
            if not rt_sub.is_empty():
                df = df.join(rt_sub, on="week_start", how="left")
                for c in ["rt_sub_acml_total_avg", "rt_sub_30m_gton_avg",
                           "rt_sub_10m_gton_avg", "rt_sub_stn_cnt_avg"]:
                    if c in df.columns:
                        df = df.with_columns([
                            pl.col(c).interpolate().forward_fill().backward_fill().alias(c)
                        ])
                log.info(f"  지하철 밀집도 데이터 결합 완료")
        except Exception as e:
            log.warning(f"  지하철 밀집도 데이터 결합 실패: {e}")

    if include_rt_pop_detail:
        try:
            rt_popd = _load_rt_population_detail(db_path)
            if not rt_popd.is_empty():
                df = df.join(rt_popd, on="week_start", how="left")
                for c in ["rt_popdet_nonresnt_avg", "rt_popdet_highrisk_age",
                           "rt_popdet_active_age", "rt_popdet_ppltn_avg"]:
                    if c in df.columns:
                        df = df.with_columns([
                            pl.col(c).interpolate().forward_fill().backward_fill().alias(c)
                        ])
                log.info(f"  인구 상세(연령/거주) 데이터 결합 완료")
        except Exception as e:
            log.warning(f"  인구 상세 데이터 결합 실패: {e}")

    # 2026-04-17: rt_bike_status 결합 블록 제거 (따릉이 피처 제외 지시)

    if include_rt_pop_forecast:
        try:
            rt_fcst = _load_rt_population_forecast(db_path)
            if not rt_fcst.is_empty():
                df = df.join(rt_fcst, on="week_start", how="left")
                for c in ["rt_fcst_ppltn_max_avg", "rt_fcst_cong_avg"]:
                    if c in df.columns:
                        df = df.with_columns([
                            pl.col(c).interpolate().forward_fill().backward_fill().alias(c)
                        ])
                log.info(f"  인구 AI 예측 데이터 결합 완료")
        except Exception as e:
            log.warning(f"  인구 AI 예측 데이터 결합 실패: {e}")

    if include_rt_spatial:
        try:
            rt_spat = _load_rt_spatial_aggregation(db_path)
            if not rt_spat.is_empty():
                df = df.join(rt_spat, on="week_start", how="left")
                for c in [col for col in rt_spat.columns if col.startswith("rt_spatial_")]:
                    if c in df.columns:
                        df = df.with_columns([
                            pl.col(c).interpolate().forward_fill().backward_fill().alias(c)
                        ])
                n_spatial_cols = len([c for c in rt_spat.columns if c.startswith("rt_spatial_")])
                log.info(f"  spatial aggregation features joined ({n_spatial_cols} cols)")
        except Exception as e:
            log.warning(f"  spatial aggregation join failed: {e}")

    if include_rt_temporal:
        try:
            rt_temp = _load_rt_temporal_patterns(db_path)
            if not rt_temp.is_empty():
                df = df.join(rt_temp, on="week_start", how="left")
                for c in [col for col in rt_temp.columns if col.startswith("rt_temp_")]:
                    if c in df.columns:
                        df = df.with_columns([
                            pl.col(c).interpolate().forward_fill().backward_fill().alias(c)
                        ])
                n_temporal_cols = len([c for c in rt_temp.columns if c.startswith("rt_temp_")])
                log.info(f"  temporal pattern features joined ({n_temporal_cols} cols)")
        except Exception as e:
            log.warning(f"  temporal pattern join failed: {e}")

    # ── B-env. 환경 피처 (2026-04-20: 대기오염 + IoT 센서) ──
    if include_rt_air_quality:
        try:
            rt_air = _load_rt_air_quality(db_path)
            if not rt_air.is_empty():
                df = df.join(rt_air, on="week_start", how="left")
                for c in ["rt_pm10_avg", "rt_pm25_avg", "rt_o3_avg",
                         "rt_no2_avg", "rt_co_avg", "rt_air_obs_count"]:
                    if c in df.columns:
                        df = df.with_columns([
                            pl.col(c).interpolate().forward_fill().backward_fill().alias(c)
                        ])
                log.info(f"  대기오염(PM10/PM2.5/O3/NO2/CO) 데이터 결합 완료")
        except Exception as e:
            log.warning(f"  대기오염 데이터 결합 실패: {e}")

    if include_rt_sdot_env:
        try:
            rt_sdot = _load_rt_sdot_env(db_path)
            if not rt_sdot.is_empty():
                df = df.join(rt_sdot, on="week_start", how="left")
                for c in ["rt_sdot_temp_avg", "rt_sdot_hum_avg",
                         "rt_sdot_noise_avg", "rt_sdot_wspd_avg", "rt_sdot_obs_count"]:
                    if c in df.columns:
                        df = df.with_columns([
                            pl.col(c).interpolate().forward_fill().backward_fill().alias(c)
                        ])
                log.info(f"  S-DoT IoT 환경(기온/습도/소음/풍속) 데이터 결합 완료")
        except Exception as e:
            log.warning(f"  S-DoT IoT 환경 결합 실패: {e}")

    # ── B-ext3: Google Trends (G-105 leakage fix: lag1 적용) ──
    if include_google_trends:
        try:
            gt = _load_google_search_trends(db_path)
            if not gt.is_empty():
                df = df.join(gt, on="week_start", how="left")
                gt_cols = [c for c in df.columns if c.startswith("gt_")]
                for c in gt_cols:
                    df = df.with_columns([
                        pl.col(c).interpolate().forward_fill().backward_fill().alias(c)
                    ])
                # ★ G-105: Google Trends는 동시점 역인과(사람들이 아프면 검색) → lag1 필수
                gt_lag_cols = [c for c in df.columns if c.startswith("gt_")]
                for c in gt_lag_cols:
                    df = df.with_columns([
                        pl.col(c).shift(1).alias(f"{c}_lag1"),
                    ])
                # 동시점 Google Trends 제거 (leakage 원인)
                df = df.drop(gt_lag_cols)
                log.info(f"  Google Trends 결합 완료 ({len(gt_lag_cols)}개 → lag1 적용, 동시점 제거)")
        except Exception as e:
            log.warning(f"  Google Trends 결합 실패: {e}")

    # ── B-ext4: 학교 휴업 (G-106 leakage fix: lag1 적용) ──
    if include_school_closure:
        try:
            sc = _load_school_closure(db_path)
            if not sc.is_empty():
                df = df.join(sc, on="week_start", how="left")
                if "sch_closure_count" in df.columns:
                    df = df.with_columns([
                        pl.col("sch_closure_count").fill_null(0).alias("sch_closure_count")
                    ])
                    # ★ G-106: 학교 휴업은 ILI 급증 → 반응적 폐쇄 (역인과) → lag1 필수
                    df = df.with_columns([
                        pl.col("sch_closure_count").shift(1).alias("sch_closure_lag1"),
                    ])
                    df = df.drop("sch_closure_count")
                log.info(f"  학교 휴업 데이터 결합 완료 (lag1 적용, 동시점 제거)")
        except Exception as e:
            log.warning(f"  학교 휴업 데이터 결합 실패: {e}")

    # ── B-ext5 (Phase D, sprint 2026-05-06): G1=A3 + B.4=mixc implementation ──

    # D.3: WHO FluNet positivity + subtype shares (lag1 leakage prevention G-186)
    if include_flunet_positivity:
        try:
            flu = _load_flunet_positivity(db_path)
            if not flu.is_empty():
                # Codex review (2026-05-06 #2): sort after join — shift(1) below
                # must respect temporal order; otherwise lag1 may cross
                # season boundaries silently.
                df = df.join(flu, on="week_start", how="left").sort("week_start")
                flu_cols = ["flu_positivity", "flu_AH3_share", "flu_AH1_share",
                            "flu_BVic_share", "flu_BYam_share"]
                for c in flu_cols:
                    if c in df.columns:
                        df = df.with_columns([
                            pl.col(c).fill_null(0.0).shift(1).fill_null(0.0)
                            .alias(f"{c}_lag1"),
                        ])
                drop_existing = [c for c in flu_cols if c in df.columns]
                df = df.drop(drop_existing)
                log.info(f"  WHO FluNet positivity 결합 완료 "
                         f"({len(flu_cols)} → lag1, 동시점 제거)")
        except Exception as e:
            log.warning(f"  WHO FluNet positivity 결합 실패: {e}")

    # D.2: Korean holiday clinic closure (lag1 reporting artifact)
    if include_korean_holiday:
        try:
            hol = _load_korean_holiday()
            if not hol.is_empty():
                # Codex review (2026-05-06 #2): sort after join (same reason
                # as D.3) — temporal order required before shift(1).
                df = df.join(hol, on="week_start", how="left").sort("week_start")
                for c in ["holiday_count", "is_lunar_chuseok"]:
                    if c in df.columns:
                        df = df.with_columns([
                            pl.col(c).fill_null(0).shift(1).fill_null(0)
                            .alias(f"{c}_lag1"),
                        ])
                drop_h = [c for c in ["holiday_count", "is_lunar_chuseok"]
                           if c in df.columns]
                df = df.drop(drop_h)
                log.info("  Korean holiday 결합 완료 (lag1, 동시점 제거)")
        except Exception as e:
            log.warning(f"  Korean holiday 결합 실패: {e}")

    # D.5: NPI intervention dummy (B.4 B layer — date-based, no DB query)
    if include_npi_dummy:
        try:
            covid_start = pl.datetime(2020, 3, 1)
            covid_end = pl.datetime(2022, 12, 31)
            df = df.with_columns([
                ((pl.col("week_start") >= covid_start)
                 & (pl.col("week_start") <= covid_end))
                .cast(pl.Float64).alias("covid_npi_period"),
                # 3-level Wagner 2002 ITS regime: pre=0, intra=2, post=0
                pl.when(pl.col("week_start") < covid_start).then(0.0)
                .when(pl.col("week_start") <= covid_end).then(2.0)
                .otherwise(0.0).alias("npi_level"),
            ])
            log.info("  D.5: NPI intervention dummy 추가 (B.4 B layer)")
        except Exception as e:
            log.warning(f"  D.5 NPI dummy 추가 실패: {e}")

    # ── C. Lag / Rolling / Diff ──
    df = _add_lag_features(df, "ili_rate", [1, 2, 3, 4, 6, 8, 12])
    df = _add_rolling_features(df, "ili_rate", [4, 8, 13, 26])
    df = _add_diff_features(df, "ili_rate", [1, 2, 4])

    # D.4 (Phase D, sprint 2026-05-06): lag52 + EWM + Fourier seasonality
    if include_seasonal_extra:
        try:
            # G-186 + Codex review (2026-05-06 #2): sort by week_start before
            # any shift/EWM operation. Earlier joins (FluNet, holiday, etc.)
            # may leave rows in non-temporal order, which silently corrupts
            # `shift(N)`. Forcing sort here also benefits D.3 / D.2 lag1 paths.
            df = df.sort("week_start")
            df = df.with_columns([
                pl.col("ili_rate").shift(52).fill_null(0.0).alias("ili_rate_lag52"),
            ])
            # G-186 + Codex review (2026-05-06 #8 CRITICAL): EWM on raw
            # `ili_rate` would inject the current-week target into the feature
            # used to predict that very target → target leakage. shift(1) first
            # so EWM only uses information up to t-1.
            ts_lag1 = pl.col("ili_rate").shift(1).fill_null(0.0)
            df = df.with_columns([
                ts_lag1.ewm_mean(half_life=4).alias("ili_rate_ewm_4w"),
                ts_lag1.ewm_mean(half_life=12).alias("ili_rate_ewm_12w"),
                ts_lag1.ewm_mean(half_life=26).alias("ili_rate_ewm_26w"),
            ])
            n_rows = df.height
            df = df.with_columns([
                pl.int_range(0, n_rows, dtype=pl.Int64).cast(pl.Float64)
                .alias("__week_idx_seas"),
            ])
            import math as _math
            for k in [1, 2, 3]:
                df = df.with_columns([
                    (2.0 * _math.pi * pl.col("__week_idx_seas") * k / 52.0)
                    .sin().alias(f"fourier_sin_h{k}"),
                    (2.0 * _math.pi * pl.col("__week_idx_seas") * k / 52.0)
                    .cos().alias(f"fourier_cos_h{k}"),
                ])
            df = df.drop("__week_idx_seas")
            log.info("  D.4: lag52 + EWM (3 windows) + Fourier (3 harmonics) 추가")
        except Exception as e:
            log.warning(f"  D.4 seasonal_extra 추가 실패: {e}")

    # ── C-2. ili_age_* 연령군 피처: lag 적용 (G-089 leakage fix) ──
    age_cols_in_df = [c for c in df.columns if c.startswith("ili_age_")]
    for ac in age_cols_in_df:
        df = df.with_columns([
            pl.col(ac).shift(1).alias(f"{ac}_lag1"),
            pl.col(ac).shift(2).alias(f"{ac}_lag2"),
        ])
    # 동시점 연령군 피처 제거 (leakage 원인)
    df = df.drop(age_cols_in_df)

    # ── D. 고급 변환 ──
    if log_transform:
        log_cols = ["ili_rate_lag1", "ili_rate_lag2"]
        log_cols = [c for c in log_cols if c in df.columns]
        if log_cols:
            df = _add_log_features(df, log_cols)
            log.info(f"  Log1p 변환 적용: {len(log_cols)}개 컬럼")

    _qe_train_end = int(len(df) * train_ratio) if train_ratio > 0 else None
    if quantile_encode:
        if "ili_rate_lag1" in df.columns:
            df = _add_quantile_encoding(df, "ili_rate_lag1", n_bins=10, train_end=_qe_train_end)
        if "temp_avg" in df.columns:
            df = _add_quantile_encoding(df, "temp_avg", n_bins=8, train_end=_qe_train_end)
        log.info(f"  Quantile encoding 적용")

    if binary_encode:
        if "ili_rate_lag1" in df.columns:
            df = _add_binary_encoding(df, "ili_rate_lag1", n_bits=10)
        log.info(f"  Binary encoding 적용")

    if multi_resolution:
        df = _add_multi_resolution_seasonal(df)
        log.info(f"  Multi-resolution seasonal 피처 생성")

    if wavelet_features:
        if "ili_rate_lag1" in df.columns:
            df = _add_wavelet_features(df, "ili_rate_lag1", [4, 8, 16])
        log.info(f"  Wavelet(Ricker) 피처 생성")

    if interaction_features:
        df = _add_interaction_features(df)
        log.info(f"  Interaction 피처 생성")

    if epidemic_phase:
        df = _add_epidemic_phase_features(df, train_ratio=train_ratio)
        log.info(f"  Epidemic phase 피처 생성")

    if multi_resolution_agg:
        df = _add_multi_resolution_agg(df)

    # ── D'. Advanced derived features (2026-04-28, env MPH_ADVANCED_FEATURES=1) ──
    # 8 categories: Hilbert / EMD-lite / Takens / PermEntropy / SpecEntropy /
    # Hjorth / catch22-lite / quantum-inspired. All causal (no leakage).
    _adv_flag = (
        advanced_features
        if advanced_features is not None
        else GLOBAL.training.advanced_features
    )
    if _adv_flag and "ili_rate" in df.columns:
        try:
            from simulation.models.feature_engine.advanced_transforms import (
                add_advanced_features as _add_adv,
            )
            n_before = len(df.columns)
            # 환경변수로 enabled subset 지정 가능
            _enabled_env = GLOBAL.training.advanced_enabled
            _enabled = (
                set(_enabled_env.split(","))
                if _enabled_env
                else None     # None → all 8 categories
            )
            df = _add_adv(df, col="ili_rate", enabled=_enabled)
            n_after = len(df.columns)
            log.info(f"  Advanced features 생성: {n_after - n_before} 개 추가 "
                     f"({_enabled or 'all 8'})")
        except Exception as _adv_err:
            log.warning(f"  Advanced features 실패 (skipped): {_adv_err}")

    # ── E. Season 인코딩 ──
    if "season_start" in df.columns:
        unique_seasons = sorted(df["season_start"].unique().to_list())
        season_map = {s: i for i, s in enumerate(unique_seasons)}
        season_idx_vals = np.array([season_map.get(s, 0) for s in df["season_start"]])
        df = df.with_columns([
            pl.lit(season_idx_vals).alias("season_idx"),
            pl.lit(season_idx_vals / max(season_idx_vals.max(), 1)).alias("season_norm"),
        ])

    # ── F. 정리 ──
    # F1: week_start 는 sanitize_polars_df 의 non-float fill_null(0) 가
    # datetime 타입을 깨뜨리므로 drop_nulls 직후, sanitize 이전에 꺼낸다.
    drop_cols = ["season_start", "week_seq", "week_label", "cal_date", "year", "month", "iso_week"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    feat_df = df.drop(drop_cols)

    from simulation.pipeline.sanitize import sanitize_polars_df

    n_before = len(feat_df)
    feat_df = feat_df.drop_nulls(subset=["ili_rate_lag1"])

    dates_arr: Optional[np.ndarray] = None
    if "week_start" in feat_df.columns:
        dates_arr = feat_df["week_start"].to_numpy()
        feat_df = feat_df.drop("week_start")

    feat_df, n_dirty = sanitize_polars_df(feat_df, fill_value=0.0)
    n_after = len(feat_df)

    # 타겟과 피처 분리
    target_col = "ili_rate"
    all_feature_cols = [c for c in feat_df.columns if c != target_col]

    meta = {
        "n_rows": n_after,
        "n_features": len(all_feature_cols),
        "n_dropped": n_before - n_after,
        "feature_cols": all_feature_cols,
        "target_col": target_col,
        "feature_groups": _categorize_features(all_feature_cols),
        "dates": dates_arr,  # F1: week_start per row, None if weather not joined
    }

    # ── G. Leakage 방어 검증 (2026-04-11) ──
    leakage_flags = []
    for c in all_feature_cols:
        if c.startswith("ili_age_") and "_lag" not in c:
            leakage_flags.append(f"CRITICAL: {c} -- 동시점 타겟 구성요소 (lag 없음)")
    if leakage_flags:
        for lf in leakage_flags:
            log.error(f"  LEAKAGE: {lf}")
        raise ValueError(f"Data leakage detected: {len(leakage_flags)} contemporaneous target components")
    else:
        log.info(f"  Leakage guard: ili_age_* -- no contemporaneous features (lag only)")

    # ── G-2. Target proxy 경고 (r > 0.95) ──
    # humid_ili, subway_ili, bus_ili 등 X*lag1 교차항은 lag1의 높은 자기상관으로
    # target과 r>0.95가 되어 다중공선성 유발. leakage는 아님(lag1 정당) but 경고.
    _target_proxy_threshold = 0.95
    _target_arr = feat_df[target_col].to_numpy()
    _proxy_warnings = []
    for c in all_feature_cols:
        try:
            _feat_arr = feat_df[c].to_numpy().astype(float)
            _valid = np.isfinite(_target_arr) & np.isfinite(_feat_arr)
            if _valid.sum() > 10:
                _corr = np.abs(np.corrcoef(_target_arr[_valid], _feat_arr[_valid])[0, 1])
                if _corr > _target_proxy_threshold:
                    _proxy_warnings.append((c, _corr))
        except Exception:
            pass
    if _proxy_warnings:
        log.warning(f"  Target proxy 경고: {len(_proxy_warnings)}개 피처 |r(target)| > {_target_proxy_threshold}")
        for _pc, _pr in sorted(_proxy_warnings, key=lambda x: -x[1]):
            log.warning(f"    {_pc}: r = {_pr:.4f} (다중공선성 주의, Optuna 피처선택 권장)")
    meta["target_proxy_warnings"] = _proxy_warnings

    log.info(f"=== Feature engineering done: {n_after} rows, {len(all_feature_cols)} features ===")
    for group, cols in meta["feature_groups"].items():
        log.info(f"  {group}: {len(cols)}")

    return feat_df, meta