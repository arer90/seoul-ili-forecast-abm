"""HIRA claims loaders — gender/age, inpatient/outpatient, Seoul region.

Sprint β Item 5 full migration (2026-05-26): real bodies moved here from
`loaders.py`. The legacy `loaders` module re-exports these for back-compat.
"""
from __future__ import annotations

import logging

import polars as pl

from ..utils import _read_sql

log = logging.getLogger(__name__)


def _load_hira_gender_age(db_path: str) -> pl.DataFrame:
    """hira_gender_age → 연도별 성별/연령별 외래 방문수."""
    try:
        df = _read_sql("""
            SELECT ref_year, sex, age_group, visit_days
            FROM hira_gender_age
            WHERE ref_year IS NOT NULL
            ORDER BY ref_year
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        # 연도별 평균 visit_days
        yearly = (
            df.group_by("ref_year")
            .agg([
                pl.col("visit_days").mean().alias("hira_visits_avg"),
                pl.col("visit_days").sum().alias("hira_visits_total"),
            ])
            .sort("ref_year")
        )

        # ref_year → 연도 1월 1일
        yearly = yearly.with_columns([
            (pl.lit("") + pl.col("ref_year").cast(pl.Utf8) + "-01-01")
            .str.to_date().alias("date")
        ])
        yearly = yearly.with_columns([
            (pl.col("date") - pl.duration(days=pl.col("date").dt.weekday())).alias("week_start")
        ])

        yearly = yearly.select(["week_start", "hira_visits_avg", "hira_visits_total"])

        # week_start를 datetime[μs]로 캐스트
        yearly = yearly.with_columns(pl.col("week_start").cast(pl.Datetime("us")))

        return yearly
    except Exception as e:
        log.warning(f"  hira_gender_age 로드 실패: {e}")
        return pl.DataFrame()


def _load_hira_inpat_opat(db_path: str) -> pl.DataFrame:
    """hira_inpat_opat → 연도별 입원/외래 비율.

    역학 활용:
      - 입원/(입원+외래) 비율 = 질환 중증도 지표
      - 입원 환자수 추이 = 의료부담 (hospital burden)
    """
    try:
        df = _read_sql("""
            SELECT ref_year, inpat_opat, SUM(patient_count) as patients,
                   SUM(visit_days) as visits
            FROM hira_inpat_opat
            WHERE ref_year IS NOT NULL
            GROUP BY ref_year, inpat_opat
            ORDER BY ref_year
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        # 입원/외래 피벗
        pivot = df.pivot(
            on="inpat_opat", index="ref_year",
            values="patients", aggregate_function="sum"
        ).fill_null(0)

        # 입원 비율 계산
        inpat_col = "입원" if "입원" in pivot.columns else None
        opat_col = "외래" if "외래" in pivot.columns else None

        if inpat_col and opat_col:
            pivot = pivot.with_columns([
                (pl.col(inpat_col) / (pl.col(inpat_col) + pl.col(opat_col) + 1)).alias("hira_inpat_ratio"),
                pl.col(inpat_col).alias("hira_inpat_count"),
            ])
        else:
            return pl.DataFrame()

        # ref_year → week_start
        pivot = pivot.with_columns([
            (pl.col("ref_year").cast(pl.Utf8) + "-07-01").str.to_date().alias("date")
        ])
        pivot = pivot.with_columns([
            (pl.col("date") - pl.duration(days=pl.col("date").dt.weekday())).alias("week_start")
        ])

        result = pivot.select([
            "week_start", "hira_inpat_ratio", "hira_inpat_count"
        ]).with_columns(pl.col("week_start").cast(pl.Datetime("us")))

        return result.sort("week_start")
    except Exception as e:
        log.warning(f"  hira_inpat_opat 로드 실패: {e}")
        return pl.DataFrame()


def _load_hira_region_seoul(db_path: str) -> pl.DataFrame:
    """hira_region → 서울 지역 연도별 환자수/진료일수.

    역학 활용: 서울 의료이용 추이 = 감염병 부담 proxy
    """
    try:
        df = _read_sql("""
            SELECT ref_year, SUM(patient_count) as patients,
                   SUM(visit_days) as visits
            FROM hira_region
            WHERE region = '서울' AND ref_year IS NOT NULL
            GROUP BY ref_year
            ORDER BY ref_year
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        df = df.with_columns([
            (pl.col("ref_year").cast(pl.Utf8) + "-07-01").str.to_date().alias("date")
        ])
        df = df.with_columns([
            (pl.col("date") - pl.duration(days=pl.col("date").dt.weekday())).alias("week_start"),
        ])

        result = df.select([
            "week_start",
            pl.col("patients").alias("hira_seoul_patients"),
            pl.col("visits").alias("hira_seoul_visits"),
        ]).with_columns(pl.col("week_start").cast(pl.Datetime("us")))

        return result.sort("week_start")
    except Exception as e:
        log.warning(f"  hira_region(서울) 로드 실패: {e}")
        return pl.DataFrame()


__all__ = [
    "_load_hira_gender_age",
    "_load_hira_inpat_opat",
    "_load_hira_region_seoul",
]
