"""Real-time Seoul Open-API loaders (rt_*).

Sprint β Item 5 full migration (2026-05-26): real bodies moved here from
`loaders.py`. The legacy `loaders` module re-exports these for back-compat.

9 functions:
- _load_rt_population            — POI 혼잡도 + 인구 (citydata_ppltn)
- _load_rt_air_quality           — PM10/PM25/O3 (RealtimeCityAir)
- _load_rt_road_traffic          — 도로 소통/속도
- _load_rt_subway_crowd          — 지하철 밀집도
- _load_rt_population_detail     — 연령/성별/거주 비율
- _load_rt_population_forecast   — AI 인구예측 (uses _CONG_MAP_PRIMARY/ROAD direct)
- _load_rt_sdot_env              — S-DoT 환경 (온/습/소음)
- _load_rt_spatial_aggregation   — 79 POI 공간 분산/밀도/유동 features
- _load_rt_temporal_patterns     — 시간대/요일 패턴 features

Uses `_safe_congestion_score` + `_CONG_MAP_PRIMARY` + `_CONG_MAP_ROAD` from
`_common.py` (Korean encoding-safe). The lazy `numpy` import inside
`_load_rt_spatial_aggregation` is preserved per R3 (keep import-time cost low).
"""
from __future__ import annotations

import logging

import polars as pl

from ..utils import _read_sql
from ._common import (
    _CONG_MAP_PRIMARY,
    _CONG_MAP_ROAD,
    _safe_congestion_score,
)

log = logging.getLogger(__name__)


def _load_rt_population(db_path: str) -> pl.DataFrame:
    """rt_population -> 주간 집계: 혼잡도 평균, 인구 max/min 평균.

    Chrome API 검증 (2026-04-11): citydata_ppltn
      AREA_NM, AREA_CD, AREA_CONGEST_LVL, AREA_PPLTN_MIN, AREA_PPLTN_MAX,
      MALE/FEMALE_PPLTN_RATE, PPLTN_RATE_0~70, PPLTN_TIME, FCST_PPLTN
    """
    try:
        df = _read_sql("""
            SELECT collected_at, area_nm, congestion, ppltn_min, ppltn_max
            FROM rt_population
            WHERE collected_at IS NOT NULL
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        # congestion 수치화: 여유=1, 보통=2, 약간 붐빔=3, 붐빔=4
        # (removed hardcoded cong_map - using _safe_congestion_score instead)
        df = df.with_columns([
            pl.col("collected_at").str.slice(0, 10).str.to_datetime("%Y-%m-%d").alias("date"),
            pl.col("congestion").map_elements(_safe_congestion_score, return_dtype=pl.Float64).alias("cong_score"),
            pl.col("ppltn_max").cast(pl.Float64),
        ])

        # ISO week -> week_start (Monday)
        df = df.with_columns([
            (pl.col("date") - pl.duration(days=pl.col("date").dt.weekday())).alias("week_start"),
        ])

        # 주간 집계: 전체 POI 평균
        weekly = (
            df.group_by("week_start")
            .agg([
                pl.col("cong_score").mean().alias("rt_congestion_avg"),
                pl.col("ppltn_max").mean().alias("rt_ppltn_max_avg"),
                pl.len().alias("rt_pop_obs_count"),
            ])
            .with_columns(pl.col("week_start").cast(pl.Datetime("us")))
            .sort("week_start")
        )
        return weekly
    except Exception:
        return pl.DataFrame()


def _load_rt_air_quality(db_path: str) -> pl.DataFrame:
    """rt_air_quality -> 주간 집계: PM10/PM2.5/O3 평균.

    Chrome API 검증 (2026-04-11): RealtimeCityAir
      MSRSTN_NM, PM(=pm10), FPM(=pm25), OZON(=o3), NTDX(=no2),
      CBMX(=co), SPDX(=so2), CAI_GRD, CAI_IDX
    """
    try:
        df = _read_sql("""
            SELECT collected_at, source, location_nm, pm10, pm25, o3, no2, co
            FROM rt_air_quality
            WHERE collected_at IS NOT NULL
              AND (pm10 IS NOT NULL OR pm25 IS NOT NULL)
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        df = df.with_columns([
            pl.col("collected_at").str.slice(0, 10).str.to_datetime("%Y-%m-%d").alias("date"),
        ])
        df = df.with_columns([
            (pl.col("date") - pl.duration(days=pl.col("date").dt.weekday())).alias("week_start"),
        ])

        weekly = (
            df.group_by("week_start")
            .agg([
                pl.col("pm10").mean().alias("rt_pm10_avg"),
                pl.col("pm25").mean().alias("rt_pm25_avg"),
                pl.col("o3").mean().alias("rt_o3_avg"),
                pl.col("no2").mean().alias("rt_no2_avg"),
                pl.col("co").mean().alias("rt_co_avg"),
                pl.len().alias("rt_air_obs_count"),
            ])
            .with_columns(pl.col("week_start").cast(pl.Datetime("us")))
            .sort("week_start")
        )
        return weekly
    except Exception:
        return pl.DataFrame()


def _load_rt_road_traffic(db_path: str) -> pl.DataFrame:
    """rt_road_traffic → 주간 집계: 도로소통 지표.

    역학 활용: 도로 평균속도 ↓ = 차량 밀집 ↑ = 이동 밀도 proxy
      - road_traffic_idx: 원활/서행/정체 → 수치화
      - road_traffic_spd: 평균 속도 (km/h) → 역수 = 밀집도
    """
    try:
        df = _read_sql("""
            SELECT collected_at, area_nm, road_traffic_idx, road_traffic_spd
            FROM rt_road_traffic
            WHERE collected_at IS NOT NULL
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        # 소통 지표 수치화: 원활=1, 서행=2, 정체=3
        idx_map = {"원활": 1.0, "서행": 2.0, "정체": 3.0}
        df = df.with_columns([
            pl.col("collected_at").str.slice(0, 10).str.to_datetime("%Y-%m-%d").alias("date"),
            pl.col("road_traffic_idx").replace_strict(idx_map, default=None).cast(pl.Float64).alias("road_cong_score"),
            pl.col("road_traffic_spd").cast(pl.Float64),
        ])
        df = df.with_columns([
            (pl.col("date") - pl.duration(days=pl.col("date").dt.weekday())).alias("week_start"),
        ])

        weekly = (
            df.group_by("week_start")
            .agg([
                pl.col("road_cong_score").mean().alias("rt_road_cong_avg"),
                pl.col("road_traffic_spd").mean().alias("rt_road_spd_avg"),
                pl.col("road_traffic_spd").min().alias("rt_road_spd_min"),
                pl.len().alias("rt_road_obs_count"),
            ])
            .with_columns(pl.col("week_start").cast(pl.Datetime("us")))
            .sort("week_start")
        )
        return weekly
    except Exception as e:
        log.warning(f"  rt_road_traffic 로드 실패: {e}")
        return pl.DataFrame()


def _load_rt_subway_crowd(db_path: str) -> pl.DataFrame:
    """rt_subway_crowd → 주간 집계: 지하철 밀집도.

    역학 활용: 밀폐 공간 접촉 강도 (핵심 감염 전파 지표)
      - 누적 승하차 인구 → 총 접촉량
      - 30분/10분 이내 승하차 → 단기 밀집도 (피크 혼잡)
      - sub_stn_cnt → 인근 지하철역 수 (인프라 밀도)
    """
    try:
        df = _read_sql("""
            SELECT collected_at, area_nm, sub_stn_cnt,
                   acml_gton_max, acml_gtoff_max,
                   gton_30m_max, gtoff_30m_max,
                   gton_10m_max, gtoff_10m_max
            FROM rt_subway_crowd
            WHERE collected_at IS NOT NULL
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        df = df.with_columns([
            pl.col("collected_at").str.slice(0, 10).str.to_datetime("%Y-%m-%d").alias("date"),
        ])
        df = df.with_columns([
            (pl.col("date") - pl.duration(days=pl.col("date").dt.weekday())).alias("week_start"),
        ])

        # 주간 집계: 전체 POI 평균
        weekly = (
            df.group_by("week_start")
            .agg([
                pl.col("acml_gton_max").mean().alias("rt_sub_acml_gton_avg"),
                pl.col("acml_gtoff_max").mean().alias("rt_sub_acml_gtoff_avg"),
                (pl.col("acml_gton_max") + pl.col("acml_gtoff_max")).mean().alias("rt_sub_acml_total_avg"),
                pl.col("gton_30m_max").mean().alias("rt_sub_30m_gton_avg"),
                pl.col("gton_10m_max").mean().alias("rt_sub_10m_gton_avg"),
                pl.col("sub_stn_cnt").mean().alias("rt_sub_stn_cnt_avg"),
                pl.len().alias("rt_sub_obs_count"),
            ])
            .with_columns(pl.col("week_start").cast(pl.Datetime("us")))
            .sort("week_start")
        )
        return weekly
    except Exception as e:
        log.warning(f"  rt_subway_crowd 로드 실패: {e}")
        return pl.DataFrame()


def _load_rt_population_detail(db_path: str) -> pl.DataFrame:
    """rt_population_detail → 주간 집계: 연령/성별/거주비율.

    역학 활용:
      - 연령별 비율: 고령자 비율 ↑ = 감수성 인구 ↑
      - 비거주자 비율 ↑ = 외부 유입 감염 위험 ↑
      - 남녀 비율: 활동 패턴 차이 proxy
    """
    try:
        df = _read_sql("""
            SELECT collected_at, area_nm, congestion,
                   ppltn_min, ppltn_max,
                   male_rate, female_rate,
                   rate_0, rate_10, rate_20, rate_30, rate_40,
                   rate_50, rate_60, rate_70,
                   resnt_rate, non_resnt_rate
            FROM rt_population_detail
            WHERE collected_at IS NOT NULL
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        df = df.with_columns([
            pl.col("collected_at").str.slice(0, 10).str.to_datetime("%Y-%m-%d").alias("date"),
        ])
        df = df.with_columns([
            (pl.col("date") - pl.duration(days=pl.col("date").dt.weekday())).alias("week_start"),
        ])

        # 고위험 연령층(0-9 + 60+) 비율 합산
        df = df.with_columns([
            (pl.col("rate_0").fill_null(0) +
             pl.col("rate_60").fill_null(0) +
             pl.col("rate_70").fill_null(0)).alias("high_risk_age_rate"),
            # 활동인구(20-50대) 비율
            (pl.col("rate_20").fill_null(0) +
             pl.col("rate_30").fill_null(0) +
             pl.col("rate_40").fill_null(0) +
             pl.col("rate_50").fill_null(0)).alias("active_age_rate"),
        ])

        weekly = (
            df.group_by("week_start")
            .agg([
                pl.col("ppltn_max").mean().alias("rt_popdet_ppltn_avg"),
                pl.col("non_resnt_rate").mean().alias("rt_popdet_nonresnt_avg"),
                pl.col("resnt_rate").mean().alias("rt_popdet_resnt_avg"),
                pl.col("high_risk_age_rate").mean().alias("rt_popdet_highrisk_age"),
                pl.col("active_age_rate").mean().alias("rt_popdet_active_age"),
                pl.col("rate_0").mean().alias("rt_popdet_rate_0_9"),
                pl.col("rate_70").mean().alias("rt_popdet_rate_70p"),
                pl.col("male_rate").mean().alias("rt_popdet_male_rate"),
                pl.len().alias("rt_popdet_obs_count"),
            ])
            .with_columns(pl.col("week_start").cast(pl.Datetime("us")))
            .sort("week_start")
        )
        return weekly
    except Exception as e:
        log.warning(f"  rt_population_detail 로드 실패: {e}")
        return pl.DataFrame()


def _load_rt_population_forecast(db_path: str) -> pl.DataFrame:
    """rt_population_forecast → 주간 집계: AI 인구예측.

    역학 활용: 향후 밀집도 예측 → 선제적 방역 지표
      - 예측 최대 인구 = 잠재적 밀집 위험

    NOTE (R1 from Gemini plan): direct dict-merge `{**_CONG_MAP_PRIMARY,
    **_CONG_MAP_ROAD}` is intentional — Polars `replace_strict` requires a
    literal dict, so the Unicode-fallback path is bypassed here.
    """
    try:
        df = _read_sql("""
            SELECT collected_at, area_nm, fcst_congest,
                   fcst_ppltn_min, fcst_ppltn_max
            FROM rt_population_forecast
            WHERE collected_at IS NOT NULL
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        # 예측 혼잡도 수치화
        # Using encoding-safe mapper instead of hardcoded Korean dict
        df = df.with_columns([
            pl.col("collected_at").str.slice(0, 10).str.to_datetime("%Y-%m-%d").alias("date"),
            pl.col("fcst_congest").replace_strict(
                {**_CONG_MAP_PRIMARY, **_CONG_MAP_ROAD}, default=None
            ).cast(pl.Float64).alias("fcst_cong_score"),
        ])
        df = df.with_columns([
            (pl.col("date") - pl.duration(days=pl.col("date").dt.weekday())).alias("week_start"),
        ])

        weekly = (
            df.group_by("week_start")
            .agg([
                pl.col("fcst_ppltn_max").mean().alias("rt_fcst_ppltn_max_avg"),
                pl.col("fcst_cong_score").mean().alias("rt_fcst_cong_avg"),
                pl.len().alias("rt_fcst_obs_count"),
            ])
            .with_columns(pl.col("week_start").cast(pl.Datetime("us")))
            .sort("week_start")
        )
        return weekly
    except Exception as e:
        log.warning(f"  rt_population_forecast 로드 실패: {e}")
        return pl.DataFrame()


def _load_rt_sdot_env(db_path: str) -> pl.DataFrame:
    """rt_sdot_env -> 주간 집계: 온도/습도/소음/풍속 평균.

    Chrome API 검증 (2026-04-11): IotVdata017
      SN, CGG, DONG, AVG_TP(=temperature), AVG_HUM(=humidity),
      AVG_NIS(=noise), AVG_UV(=uv_index), AVG_WSPD(=wind_speed)
      ⚠️ PM10/PM25 필드 없음 (S-DoT 센서 미측정)
    """
    try:
        df = _read_sql("""
            SELECT collected_at, sensor_id, cgg, temperature, humidity,
                   noise, uv_index, wind_speed
            FROM rt_sdot_env
            WHERE collected_at IS NOT NULL
              AND (temperature IS NOT NULL OR humidity IS NOT NULL)
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        df = df.with_columns([
            pl.col("collected_at").str.slice(0, 10).str.to_datetime("%Y-%m-%d").alias("date"),
        ])
        df = df.with_columns([
            (pl.col("date") - pl.duration(days=pl.col("date").dt.weekday())).alias("week_start"),
        ])

        weekly = (
            df.group_by("week_start")
            .agg([
                pl.col("temperature").mean().alias("rt_sdot_temp_avg"),
                pl.col("humidity").mean().alias("rt_sdot_hum_avg"),
                pl.col("noise").mean().alias("rt_sdot_noise_avg"),
                pl.col("wind_speed").mean().alias("rt_sdot_wspd_avg"),
                pl.len().alias("rt_sdot_obs_count"),
            ])
            .with_columns(pl.col("week_start").cast(pl.Datetime("us")))
            .sort("week_start")
        )
        return weekly
    except Exception:
        return pl.DataFrame()


def _load_rt_spatial_aggregation(db_path: str) -> pl.DataFrame:
    """rt_population_detail + rt_road_traffic + rt_subway_crowd -> weekly spatial agg.

    79 POI spatial dispersion/density/flow features:
      - rt_spatial_cong_std: congestion std across POIs
      - rt_spatial_crowded_ratio: fraction of crowded POIs
      - rt_spatial_total_ppltn: total floating population
      - rt_spatial_ppltn_gini: population Gini coefficient
      - rt_spatial_nonresnt_std: non-resident rate std
      - rt_spatial_road_spd_std: road speed std
      - rt_spatial_road_spd_cv: road speed coefficient of variation
      - rt_spatial_sub_total_gton: total subway boarding
      - rt_spatial_poi_count: number of POIs with data
    """
    try:
        import numpy as np  # lazy — keep import-time cost low (R3)

        df_pop = _read_sql("""
            SELECT collected_at, area_nm, congestion,
                   ppltn_max, non_resnt_rate
            FROM rt_population_detail
            WHERE collected_at IS NOT NULL
        """, db_path)

        df_road = _read_sql("""
            SELECT collected_at, area_nm, road_traffic_spd
            FROM rt_road_traffic
            WHERE collected_at IS NOT NULL
        """, db_path)

        df_sub = _read_sql("""
            SELECT collected_at, area_nm, acml_gton_max
            FROM rt_subway_crowd
            WHERE collected_at IS NOT NULL
        """, db_path)

        if df_pop.is_empty():
            return pl.DataFrame()

        # Using encoding-safe mapper instead of hardcoded Korean dict
        df_pop = df_pop.with_columns([
            pl.col("congestion").map_elements(_safe_congestion_score, return_dtype=pl.Float64).alias("cong_score"),
            pl.col("ppltn_max").cast(pl.Float64),
            pl.col("non_resnt_rate").cast(pl.Float64),
        ])

        snap_pop = (
            df_pop.group_by("collected_at")
            .agg([
                pl.col("cong_score").std().alias("rt_spatial_cong_std"),
                (pl.col("cong_score") >= 3.0).mean().cast(pl.Float64).alias("rt_spatial_crowded_ratio"),
                pl.col("ppltn_max").sum().alias("rt_spatial_total_ppltn"),
                pl.col("non_resnt_rate").std().alias("rt_spatial_nonresnt_std"),
                pl.col("area_nm").n_unique().cast(pl.Float64).alias("rt_spatial_poi_count"),
                pl.col("ppltn_max").alias("_ppltn_list"),
            ])
        )

        def gini(vals):
            arr = [v for v in vals if v is not None and v > 0]
            if len(arr) < 2:
                return 0.0
            a = np.array(sorted(arr), dtype=float)
            n = len(a)
            return float((2.0 * np.sum((np.arange(1, n+1)) * a)) / (n * np.sum(a)) - (n+1)/n)

        gini_vals = [gini(row["_ppltn_list"]) for row in snap_pop.iter_rows(named=True)]
        snap_pop = snap_pop.with_columns(
            pl.Series("rt_spatial_ppltn_gini", gini_vals, dtype=pl.Float64)
        ).drop("_ppltn_list")

        if not df_road.is_empty():
            df_road = df_road.with_columns(pl.col("road_traffic_spd").cast(pl.Float64))
            snap_road = (
                df_road.group_by("collected_at")
                .agg([
                    pl.col("road_traffic_spd").std().alias("rt_spatial_road_spd_std"),
                    (pl.col("road_traffic_spd").std() /
                     pl.col("road_traffic_spd").mean()).alias("rt_spatial_road_spd_cv"),
                ])
            )
            snap_pop = snap_pop.join(snap_road, on="collected_at", how="left")
        else:
            snap_pop = snap_pop.with_columns([
                pl.lit(None).cast(pl.Float64).alias("rt_spatial_road_spd_std"),
                pl.lit(None).cast(pl.Float64).alias("rt_spatial_road_spd_cv"),
            ])

        if not df_sub.is_empty():
            df_sub = df_sub.with_columns(pl.col("acml_gton_max").cast(pl.Float64))
            snap_sub = (
                df_sub.group_by("collected_at")
                .agg(pl.col("acml_gton_max").sum().alias("rt_spatial_sub_total_gton"))
            )
            snap_pop = snap_pop.join(snap_sub, on="collected_at", how="left")
        else:
            snap_pop = snap_pop.with_columns(
                pl.lit(None).cast(pl.Float64).alias("rt_spatial_sub_total_gton")
            )

        snap_pop = snap_pop.with_columns(
            pl.col("collected_at").str.slice(0, 10).str.to_datetime("%Y-%m-%d").alias("date")
        )
        snap_pop = snap_pop.with_columns(
            (pl.col("date") - pl.duration(days=pl.col("date").dt.weekday())).alias("week_start")
        )

        feat_cols = [c for c in snap_pop.columns if c.startswith("rt_spatial_")]
        weekly = (
            snap_pop.group_by("week_start")
            .agg([pl.col(c).mean().alias(c) for c in feat_cols]
                 + [pl.len().alias("rt_spatial_obs_count")])
            .with_columns(pl.col("week_start").cast(pl.Datetime("us")))
            .sort("week_start")
        )
        return weekly

    except Exception as e:
        log.warning(f"  rt_spatial_aggregation load fail: {e}")
        return pl.DataFrame()


def _load_rt_temporal_patterns(db_path: str) -> pl.DataFrame:
    """rt_population_detail -> weekly temporal pattern features.

    Extracts hour-of-day and day-of-week patterns from collected_at timestamps:
      - rt_temp_peak_ratio: peak-hour(12-18) vs off-peak population ratio
      - rt_temp_night_ratio: night(22-06) vs daytime population ratio
      - rt_temp_ppltn_cv: intra-day population coefficient of variation
      - rt_temp_weekend_cong: weekend avg congestion score
      - rt_temp_weekday_cong: weekday avg congestion score
      - rt_temp_cong_wkend_diff: weekend - weekday congestion difference
      - rt_temp_nonresnt_morning: morning(7-10) non-resident rate avg
      - rt_temp_nonresnt_evening: evening(18-21) non-resident rate avg
      - rt_temp_commute_sig: morning - evening non-resident rate (commuting signature)
      - rt_temp_collection_density: observations per day (data richness)
    """
    try:
        df = _read_sql("""
            SELECT collected_at, area_nm, congestion,
                   ppltn_max, non_resnt_rate
            FROM rt_population_detail
            WHERE collected_at IS NOT NULL
              AND ppltn_max IS NOT NULL
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        # Using encoding-safe mapper instead of hardcoded Korean dict
        df = df.with_columns([
            pl.col("collected_at").str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False).alias("ts"),
            pl.col("congestion").map_elements(_safe_congestion_score, return_dtype=pl.Float64).alias("cong_score"),
            pl.col("ppltn_max").cast(pl.Float64),
            pl.col("non_resnt_rate").cast(pl.Float64),
        ])
        df = df.filter(pl.col("ts").is_not_null())
        df = df.with_columns([
            pl.col("ts").dt.hour().alias("hour"),
            pl.col("ts").dt.weekday().alias("dow"),  # 0=Mon, 6=Sun
            pl.col("ts").dt.date().alias("date"),
        ])
        df = df.with_columns(
            (pl.col("date") - pl.duration(days=pl.col("ts").dt.weekday())).cast(pl.Date).alias("week_date")
        )
        df = df.with_columns(
            pl.col("week_date").cast(pl.Datetime("us")).alias("week_start")
        )

        # Per-week temporal aggregation (across all POIs and all snapshots within the week)
        # Peak hours: 12-18, Night: 22-06, Morning commute: 7-10, Evening: 18-21
        df = df.with_columns([
            ((pl.col("hour") >= 12) & (pl.col("hour") < 18)).alias("is_peak"),
            ((pl.col("hour") >= 22) | (pl.col("hour") < 6)).alias("is_night"),
            ((pl.col("hour") >= 7) & (pl.col("hour") < 10)).alias("is_morning"),
            ((pl.col("hour") >= 18) & (pl.col("hour") < 21)).alias("is_evening"),
            (pl.col("dow") >= 5).alias("is_weekend"),
        ])

        weekly = (
            df.group_by("week_start")
            .agg([
                # Peak vs off-peak population ratio
                pl.col("ppltn_max").filter(pl.col("is_peak")).mean().alias("_peak_pop"),
                pl.col("ppltn_max").filter(~pl.col("is_peak")).mean().alias("_offpeak_pop"),
                # Night vs day ratio
                pl.col("ppltn_max").filter(pl.col("is_night")).mean().alias("_night_pop"),
                pl.col("ppltn_max").filter(~pl.col("is_night")).mean().alias("_day_pop"),
                # Intra-day population CV
                pl.col("ppltn_max").std().alias("_ppltn_std"),
                pl.col("ppltn_max").mean().alias("_ppltn_mean"),
                # Weekend vs weekday congestion
                pl.col("cong_score").filter(pl.col("is_weekend")).mean().alias("rt_temp_weekend_cong"),
                pl.col("cong_score").filter(~pl.col("is_weekend")).mean().alias("rt_temp_weekday_cong"),
                # Commuting signature
                pl.col("non_resnt_rate").filter(pl.col("is_morning")).mean().alias("rt_temp_nonresnt_morning"),
                pl.col("non_resnt_rate").filter(pl.col("is_evening")).mean().alias("rt_temp_nonresnt_evening"),
                # Data richness
                pl.len().alias("_total_obs"),
                pl.col("date").n_unique().cast(pl.Float64).alias("_n_days"),
            ])
        )

        # Derive ratio features
        weekly = weekly.with_columns([
            (pl.col("_peak_pop") / pl.col("_offpeak_pop").clip(lower_bound=1.0)).alias("rt_temp_peak_ratio"),
            (pl.col("_night_pop") / pl.col("_day_pop").clip(lower_bound=1.0)).alias("rt_temp_night_ratio"),
            (pl.col("_ppltn_std") / pl.col("_ppltn_mean").clip(lower_bound=1.0)).alias("rt_temp_ppltn_cv"),
            (pl.col("rt_temp_weekend_cong") - pl.col("rt_temp_weekday_cong")).alias("rt_temp_cong_wkend_diff"),
            (pl.col("rt_temp_nonresnt_morning") - pl.col("rt_temp_nonresnt_evening")).alias("rt_temp_commute_sig"),
            (pl.col("_total_obs").cast(pl.Float64) / pl.col("_n_days").clip(lower_bound=1.0)).alias("rt_temp_collection_density"),
        ])

        # Keep only final features
        keep_cols = ["week_start"] + [c for c in weekly.columns if c.startswith("rt_temp_")]
        weekly = weekly.select(keep_cols)
        weekly = weekly.with_columns(pl.col("week_start").cast(pl.Datetime("us"))).sort("week_start")
        return weekly

    except Exception as e:
        log.warning(f"  rt_temporal_patterns load fail: {e}")
        return pl.DataFrame()


__all__ = [
    "_load_rt_population",
    "_load_rt_air_quality",
    "_load_rt_road_traffic",
    "_load_rt_subway_crowd",
    "_load_rt_population_detail",
    "_load_rt_population_forecast",
    "_load_rt_sdot_env",
    "_load_rt_spatial_aggregation",
    "_load_rt_temporal_patterns",
]
