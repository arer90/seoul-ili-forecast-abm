"""Weather loaders — historical (KMA stn 108) + forecast.

Sprint β Item 5 full migration (2026-05-26): real bodies moved here from
`loaders.py`. The legacy `loaders` module re-exports these for back-compat.
"""
from __future__ import annotations

import logging

import polars as pl

from ..utils import _read_sql

log = logging.getLogger(__name__)


def _load_weather(db_path: str) -> pl.DataFrame:
    """weather_historical → 주간 기상 집계."""
    df = _read_sql("""
        SELECT obs_date, ta_avg, ta_min, hm_avg, ws_avg, rn_day, ps_avg, ss_day
        FROM weather_historical
        WHERE stn_id = 108
        ORDER BY obs_date
    """, db_path)

    df = df.with_columns([
        pl.col("obs_date").str.to_datetime("%Y%m%d").alias("date"),
    ])

    df = df.with_columns([
        (pl.col("date") - pl.duration(days=pl.col("date").dt.weekday())).alias("week_start")
    ])

    weekly = (
        df.group_by("week_start")
        .agg([
            pl.col("ta_avg").mean().alias("temp_avg"),
            pl.col("ta_min").mean().alias("temp_min"),
            pl.col("hm_avg").mean().alias("humidity"),
            pl.col("ws_avg").mean().alias("wind_speed"),
            pl.col("rn_day").sum().alias("rainfall"),
            pl.col("ps_avg").mean().alias("pressure"),
            pl.col("ss_day").sum().alias("sunshine"),
            pl.col("ta_avg").std().alias("temp_std"),
        ])
        .sort("week_start")
    )

    weekly = weekly.with_columns([
        pl.when(pl.col("rainfall") < 0).then(0).otherwise(pl.col("rainfall")).alias("rainfall"),
        pl.col("temp_std").fill_null(0),
    ])

    return weekly


def _load_weather_forecast(db_path: str) -> pl.DataFrame:
    """weather_forecast → 예보 데이터 (주간 평균/최대)."""
    try:
        df = _read_sql("""
            SELECT issued_at, valid_at, variable, value
            FROM weather_forecast
            WHERE issued_at IS NOT NULL AND valid_at IS NOT NULL
            ORDER BY valid_at
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        # valid_at를 날짜로 변환
        df = df.with_columns([
            (pl.lit("") + pl.col("valid_at").str.slice(0, 4) + "-"
             + pl.col("valid_at").str.slice(4, 2) + "-"
             + pl.col("valid_at").str.slice(6, 2))
            .str.to_date().alias("date")
        ])
        df = df.with_columns([
            (pl.col("date") - pl.duration(days=pl.col("date").dt.weekday())).alias("week_start")
        ])

        # 변수별 피봇
        pivot = (
            df.group_by(["week_start", "variable"])
            .agg(pl.col("value").mean().alias("value"))
            .pivot(
                on="variable",
                index="week_start",
                values="value"
            )
            .fill_null(0)
        )

        # 컬럼명 정리
        rename_map = {}
        for col in pivot.columns:
            if col != "week_start":
                safe = col.replace(" ", "_").replace("(", "").replace(")", "").lower()
                rename_map[col] = f"fcst_{safe}"
        pivot = pivot.rename(rename_map)

        # week_start를 datetime[μs]로 캐스트
        pivot = pivot.with_columns(pl.col("week_start").cast(pl.Datetime("us")))

        return pivot
    except Exception as e:
        log.warning(f"  weather_forecast 로드 실패: {e}")
        return pl.DataFrame()


__all__ = ["_load_weather", "_load_weather_forecast"]
