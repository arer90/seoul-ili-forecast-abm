"""Mobility loaders — daily population, subway/bus transit, demographic structure.

Sprint β Item 5 full migration (2026-05-26): real bodies moved here from
`loaders.py`. The legacy `loaders` module re-exports these for back-compat.

8 functions:
- _load_daily_population_district  — 서울 자치구 일별 생활인구
- _load_daily_subway               — 일별 지하철 승하차
- _load_daily_bus                  — 일별 버스 승하차
- _load_daily_population_hotspot   — 일별 핫스팟 혼잡도
- _load_daily_population_gu_hourly — 일별 자치구 시간대별 인구
- _load_daily_population_dong_agg  — 행정동별 인구 → 연령구조
- _load_monthly_subway_hourly      — 시간대별 지하철 이용 (rush/night ratio)
- _load_monthly_bus_hourly         — 시간대별 버스 이용 (rush ratio)
"""
from __future__ import annotations

import logging

import polars as pl

from ..utils import _read_sql

log = logging.getLogger(__name__)


def _load_daily_population_district(db_path: str) -> pl.DataFrame:
    """daily_population_district → 일별 자치구 생활인구 (주간 평균)."""
    try:
        df = _read_sql("""
            SELECT stdr_de, signgu_nm, tot_livpop, day_livpop, inflow_livpop
            FROM daily_population_district
            WHERE stdr_de IS NOT NULL
            ORDER BY stdr_de
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        # stdr_de를 날짜로 변환 (YYYYMMDD)
        df = df.with_columns([
            pl.col("stdr_de").str.to_datetime("%Y%m%d").alias("date")
        ])
        df = df.with_columns([
            (pl.col("date") - pl.duration(days=pl.col("date").dt.weekday())).alias("week_start")
        ])

        # 주간 서울시 전체 평균
        weekly = (
            df.group_by("week_start")
            .agg([
                pl.col("tot_livpop").mean().alias("pop_total_avg"),
                pl.col("day_livpop").mean().alias("pop_daytime_avg"),
                pl.col("inflow_livpop").mean().alias("pop_inflow_avg"),
            ])
            .sort("week_start")
        )

        return weekly
    except Exception as e:
        log.warning(f"  daily_population_district 로드 실패: {e}")
        return pl.DataFrame()


def _load_daily_subway(db_path: str) -> pl.DataFrame:
    """daily_subway → 일별 지하철 승하차 (주간 평균)."""
    try:
        df = _read_sql("""
            SELECT use_dt, line_num, ride_pasgr, alight_pasgr
            FROM daily_subway
            WHERE use_dt IS NOT NULL
            ORDER BY use_dt
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        # use_dt를 날짜로 변환 (YYYYMMDD)
        df = df.with_columns([
            pl.col("use_dt").str.to_datetime("%Y%m%d").alias("date")
        ])
        df = df.with_columns([
            (pl.col("date") - pl.duration(days=pl.col("date").dt.weekday())).alias("week_start")
        ])

        # 주간 평균
        weekly = (
            df.group_by("week_start")
            .agg([
                (pl.col("ride_pasgr").sum() /
                 pl.col("ride_pasgr").count()).alias("subway_ride_avg"),
                (pl.col("alight_pasgr").sum() /
                 pl.col("alight_pasgr").count()).alias("subway_alight_avg"),
                (pl.col("ride_pasgr").sum() + pl.col("alight_pasgr").sum()).alias("subway_total_avg"),
            ])
            .sort("week_start")
        )

        return weekly
    except Exception as e:
        log.warning(f"  daily_subway 로드 실패: {e}")
        return pl.DataFrame()


def _load_daily_bus(db_path: str) -> pl.DataFrame:
    """daily_bus → 일별 버스 승하차 (주간 평균)."""
    try:
        df = _read_sql("""
            SELECT use_dt, ride_cnt, alight_cnt
            FROM daily_bus
            WHERE use_dt IS NOT NULL
            ORDER BY use_dt
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        # use_dt를 날짜로 변환 (YYYYMMDD)
        df = df.with_columns([
            pl.col("use_dt").str.to_datetime("%Y%m%d").alias("date")
        ])
        df = df.with_columns([
            (pl.col("date") - pl.duration(days=pl.col("date").dt.weekday())).alias("week_start")
        ])

        # 주간 평균
        weekly = (
            df.group_by("week_start")
            .agg([
                pl.col("ride_cnt").mean().alias("bus_ride_avg"),
                pl.col("alight_cnt").mean().alias("bus_alight_avg"),
                (pl.col("ride_cnt") + pl.col("alight_cnt")).mean().alias("bus_total_avg"),
            ])
            .sort("week_start")
        )

        return weekly
    except Exception as e:
        log.warning(f"  daily_bus 로드 실패: {e}")
        return pl.DataFrame()


def _load_daily_population_hotspot(db_path: str) -> pl.DataFrame:
    """daily_population_hotspot → 일별 핫스팟 혼잡도 (주간 최대/평균)."""
    try:
        df = _read_sql("""
            SELECT stdr_de, congestion, ppltn_max
            FROM daily_population_hotspot
            WHERE stdr_de IS NOT NULL
            ORDER BY stdr_de
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        # stdr_de를 날짜로 변환 (YYYYMMDD)
        df = df.with_columns([
            pl.col("stdr_de").str.to_datetime("%Y%m%d").alias("date")
        ])
        df = df.with_columns([
            (pl.col("date") - pl.duration(days=pl.col("date").dt.weekday())).alias("week_start")
        ])

        # 주간 통계
        weekly = (
            df.group_by("week_start")
            .agg([
                pl.col("congestion").max().alias("hotspot_congestion_max"),
                pl.col("congestion").mean().alias("hotspot_congestion_avg"),
                pl.col("ppltn_max").max().alias("hotspot_ppltn_peak"),
                pl.col("ppltn_max").mean().alias("hotspot_ppltn_avg"),
            ])
            .sort("week_start")
        )

        return weekly
    except Exception as e:
        log.warning(f"  daily_population_hotspot 로드 실패: {e}")
        return pl.DataFrame()


def _load_daily_population_gu_hourly(db_path: str) -> pl.DataFrame:
    """daily_population_gu_hourly → 일별 자치구 시간대별 인구."""
    try:
        try:
            df = _read_sql("""
                SELECT stdr_de, gu_nm, tot_pop
                FROM daily_population_gu_hourly
                WHERE stdr_de IS NOT NULL
                ORDER BY stdr_de
            """, db_path)
        except Exception as inner_e:
            log.warning(f"  daily_population_gu_hourly initial query failed: {inner_e}, retrying without gu_nm")
            df = _read_sql("""
                SELECT stdr_de, tot_pop
                FROM daily_population_gu_hourly
                WHERE stdr_de IS NOT NULL
                ORDER BY stdr_de
            """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        # stdr_de를 날짜로 변환
        df = df.with_columns([
            pl.col("stdr_de").str.to_datetime("%Y%m%d").alias("date")
        ])
        df = df.with_columns([
            (pl.col("date") - pl.duration(days=pl.col("date").dt.weekday())).alias("week_start")
        ])

        # 주간 시간대 인구 평균
        weekly = (
            df.group_by("week_start")
            .agg([
                pl.col("tot_pop").mean().alias("hourly_pop_avg"),
                pl.col("tot_pop").max().alias("hourly_pop_peak"),
            ])
            .sort("week_start")
        )

        return weekly
    except Exception as e:
        log.warning(f"  daily_population_gu_hourly 로드 실패 (DB 손상): {e}")
        return pl.DataFrame()


def _load_daily_population_dong_agg(db_path: str) -> pl.DataFrame:
    """daily_population_dong → 행정동별 인구 → 주간 연령구조 지표.

    역학 활용:
      - 고령자(70+) 비율 = 감수성 인구 밀도
      - 소아(0-9) 비율 = 학교/어린이집 전파 위험
      - 동별 인구 밀도 편차 = 공간적 집중도
    """
    try:
        df = _read_sql("""
            SELECT stdr_de, tot_pop, pop_0_9, pop_10_19,
                   pop_60_69, pop_70plus
            FROM daily_population_dong
            WHERE stdr_de IS NOT NULL
            ORDER BY stdr_de
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        df = df.with_columns([
            pl.col("stdr_de").str.to_datetime("%Y%m%d").alias("date"),
        ])
        df = df.with_columns([
            (pl.col("date") - pl.duration(days=pl.col("date").dt.weekday())).alias("week_start"),
        ])

        # 동별 → 서울 전체 일별 집계 후 주간 평균
        weekly = (
            df.group_by("week_start")
            .agg([
                pl.col("tot_pop").mean().alias("dong_pop_avg"),
                pl.col("tot_pop").std().alias("dong_pop_std"),
                (pl.col("pop_0_9") / (pl.col("tot_pop") + 1)).mean().alias("dong_child_ratio"),
                ((pl.col("pop_60_69") + pl.col("pop_70plus")) / (pl.col("tot_pop") + 1)).mean().alias("dong_elderly_ratio"),
                (pl.col("pop_70plus") / (pl.col("tot_pop") + 1)).mean().alias("dong_70p_ratio"),
            ])
            .with_columns(pl.col("week_start").cast(pl.Datetime("us")))
            .sort("week_start")
        )
        return weekly
    except Exception as e:
        log.warning(f"  daily_population_dong 로드 실패: {e}")
        return pl.DataFrame()


def _load_monthly_subway_hourly(db_path: str) -> pl.DataFrame:
    """monthly_subway_hourly → 시간대별 지하철 이용 → 출퇴근 밀집 지표.

    역학 활용:
      - 출퇴근 시간대(7-9, 18-20) 승객 비율 = 밀폐 공간 접촉 강도
      - 피크/오프피크 비율 = 밀집 집중도
      - 야간(22-05) 비율 = 유흥/심야 활동 proxy
    """
    try:
        df = _read_sql("""
            SELECT use_ym, hour, ride_cnt, alight_cnt
            FROM monthly_subway_hourly
            WHERE use_ym IS NOT NULL
            ORDER BY use_ym, hour
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        # use_ym(YYYYMM) → week_start (해당 월 1일 기준)
        df = df.with_columns([
            (pl.col("use_ym").str.slice(0, 4) + "-" + pl.col("use_ym").str.slice(4, 2) + "-01")
            .str.to_date().alias("date"),
        ])
        df = df.with_columns([
            (pl.col("date") - pl.duration(days=pl.col("date").dt.weekday())).alias("week_start"),
        ])

        # 시간대 분류
        df = df.with_columns([
            pl.when((pl.col("hour") >= 7) & (pl.col("hour") <= 9))
            .then(pl.lit("rush_am"))
            .when((pl.col("hour") >= 18) & (pl.col("hour") <= 20))
            .then(pl.lit("rush_pm"))
            .when((pl.col("hour") >= 22) | (pl.col("hour") <= 5))
            .then(pl.lit("night"))
            .otherwise(pl.lit("offpeak"))
            .alias("period"),
            (pl.col("ride_cnt") + pl.col("alight_cnt")).alias("total_psgr"),
        ])

        # 월별(=주별) 시간대 집계
        period_agg = (
            df.group_by(["week_start", "period"])
            .agg(pl.col("total_psgr").sum().alias("psgr"))
        )

        # 피벗
        pivot = period_agg.pivot(
            on="period", index="week_start", values="psgr",
            aggregate_function="sum"
        ).fill_null(0)

        # 비율 계산
        all_cols = [c for c in pivot.columns if c != "week_start"]
        pivot = pivot.with_columns([
            sum(pl.col(c) for c in all_cols).alias("_total")
        ])

        result = pivot.select([
            "week_start",
        ])

        # 출퇴근 밀집 비율
        if "rush_am" in pivot.columns and "rush_pm" in pivot.columns:
            result = result.with_columns([
                ((pivot["rush_am"] + pivot["rush_pm"]) / (pivot["_total"] + 1)).alias("sub_rush_ratio"),
            ])
        if "night" in pivot.columns:
            result = result.with_columns([
                (pivot["night"] / (pivot["_total"] + 1)).alias("sub_night_ratio"),
            ])

        # 총 승객 수 (스케일 지표)
        result = result.with_columns([
            pivot["_total"].alias("sub_hourly_total"),
        ])

        result = result.with_columns(pl.col("week_start").cast(pl.Datetime("us")))
        return result.sort("week_start")
    except Exception as e:
        log.warning(f"  monthly_subway_hourly 로드 실패: {e}")
        return pl.DataFrame()


def _load_monthly_bus_hourly(db_path: str) -> pl.DataFrame:
    """monthly_bus_hourly → 시간대별 버스 이용 → 출퇴근 밀집 지표.

    역학 활용: 지하철과 유사하되 버스는 환기 조건이 다름 (창문 개방 가능)
      - 출퇴근 비율, 야간 비율, 총 이용량
    """
    try:
        df = _read_sql("""
            SELECT use_ym, hour, SUM(ride_cnt) as ride_sum, SUM(alight_cnt) as alight_sum
            FROM monthly_bus_hourly
            WHERE use_ym IS NOT NULL
            GROUP BY use_ym, hour
            ORDER BY use_ym, hour
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        df = df.with_columns([
            (pl.col("use_ym").str.slice(0, 4) + "-" + pl.col("use_ym").str.slice(4, 2) + "-01")
            .str.to_date().alias("date"),
        ])
        df = df.with_columns([
            (pl.col("date") - pl.duration(days=pl.col("date").dt.weekday())).alias("week_start"),
            (pl.col("ride_sum") + pl.col("alight_sum")).alias("total_psgr"),
            pl.when((pl.col("hour") >= 7) & (pl.col("hour") <= 9))
            .then(pl.lit("rush_am"))
            .when((pl.col("hour") >= 18) & (pl.col("hour") <= 20))
            .then(pl.lit("rush_pm"))
            .otherwise(pl.lit("other"))
            .alias("period"),
        ])

        period_agg = (
            df.group_by(["week_start", "period"])
            .agg(pl.col("total_psgr").sum().alias("psgr"))
        )

        pivot = period_agg.pivot(
            on="period", index="week_start", values="psgr",
            aggregate_function="sum"
        ).fill_null(0)

        all_cols = [c for c in pivot.columns if c != "week_start"]
        pivot = pivot.with_columns([
            sum(pl.col(c) for c in all_cols).alias("_total")
        ])

        result = pivot.select(["week_start"])
        if "rush_am" in pivot.columns and "rush_pm" in pivot.columns:
            result = result.with_columns([
                ((pivot["rush_am"] + pivot["rush_pm"]) / (pivot["_total"] + 1)).alias("bus_rush_ratio"),
            ])
        result = result.with_columns([
            pivot["_total"].alias("bus_hourly_total"),
        ])

        result = result.with_columns(pl.col("week_start").cast(pl.Datetime("us")))
        return result.sort("week_start")
    except Exception as e:
        log.warning(f"  monthly_bus_hourly 로드 실패: {e}")
        return pl.DataFrame()


__all__ = [
    "_load_daily_population_district",
    "_load_daily_subway",
    "_load_daily_bus",
    "_load_daily_population_hotspot",
    "_load_daily_population_gu_hourly",
    "_load_daily_population_dong_agg",
    "_load_monthly_subway_hourly",
    "_load_monthly_bus_hourly",
]
