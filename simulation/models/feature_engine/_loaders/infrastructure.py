"""Infrastructure / static-facility loaders — schools, hospitals, employment.

Sprint β Item 5 full migration (2026-05-26): real bodies moved here from
`loaders.py`. The legacy `loaders` module re-exports these for back-compat.
"""
from __future__ import annotations

import logging

import polars as pl

from ..utils import _read_sql

log = logging.getLogger(__name__)


def _load_school_info(db_path: str) -> pl.DataFrame:
    """school_info → 정적 학교 수 (서울시 전체, scalar broadcast)."""
    try:
        df = _read_sql("""
            SELECT school_kind
            FROM school_info
            WHERE school_kind IS NOT NULL
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        # 학교종별 총 개수
        school_counts = (
            df.group_by("school_kind")
            .agg(pl.len().alias("count"))
        )

        # 종별별 count를 dictionary로 변환
        counts_dict = {}
        for row in school_counts.to_dicts():
            school_kind = row["school_kind"]
            count = row["count"]
            safe = school_kind.replace(" ", "_").replace("(", "").replace(")", "").lower()
            counts_dict[f"school_{safe}"] = count

        # 전체 학교 수
        total_schools = school_counts["count"].sum()

        # DataFrame으로 변환
        result = pl.DataFrame({"week_start": [None]})
        for col_name, val in counts_dict.items():
            result = result.with_columns([pl.lit(val).alias(col_name)])
        result = result.with_columns([pl.lit(total_schools).alias("school_total")])

        return result
    except Exception as e:
        log.warning(f"  school_info 로드 실패: {e}")
        return pl.DataFrame()


def _load_hospitals(db_path: str) -> pl.DataFrame:
    """hospitals → 정적 병원 수 및 통계."""
    try:
        df = _read_sql("""
            SELECT clcd_nm, bed_cnt, dr_cnt
            FROM hospitals
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        # 병원종별 통계
        stats = (
            df.group_by("clcd_nm")
            .agg([
                pl.col("bed_cnt").sum().alias("beds"),
                pl.col("dr_cnt").sum().alias("doctors"),
            ])
        )

        # 전체 집계
        total_beds = stats["beds"].sum()
        total_doctors = stats["doctors"].sum()
        hospital_count = len(stats)

        # type별 통계 dictionary
        type_stats = {}
        for row in stats.to_dicts():
            clcd_nm = row["clcd_nm"]
            safe = clcd_nm.replace(" ", "_").replace("(", "").replace(")", "").lower()
            type_stats[f"hospital_{safe}_beds"] = row["beds"]
            type_stats[f"hospital_{safe}_doctors"] = row["doctors"]

        # DataFrame으로 변환
        result = pl.DataFrame({"week_start": [None]})
        for col_name, val in type_stats.items():
            result = result.with_columns([pl.lit(val).alias(col_name)])
        result = result.with_columns([
            pl.lit(total_beds).alias("hospital_total_beds"),
            pl.lit(total_doctors).alias("hospital_total_doctors"),
            pl.lit(hospital_count).alias("hospital_count")
        ])

        return result
    except Exception as e:
        log.warning(f"  hospitals 로드 실패: {e}")
        return pl.DataFrame()


def _load_employment_workplace(db_path: str) -> pl.DataFrame:
    """employment_workplace → 반기별 산업별 종사자."""
    try:
        df = _read_sql("""
            SELECT prd_de, c2_nm, dt
            FROM employment_workplace
            WHERE prd_de IS NOT NULL AND c2_nm IS NOT NULL
            ORDER BY prd_de
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        # prd_de = "YYYYHH" (반기 파싱)
        df = df.with_columns([
            (pl.col("prd_de").str.slice(0, 4).cast(pl.Int32)).alias("year"),
            (pl.col("prd_de").str.slice(4, 2).cast(pl.Int32)).alias("half"),
        ])

        # 반기별 월 매핑
        df = df.with_columns([
            pl.when(pl.col("half") == 1).then(4)
            .when(pl.col("half") == 2).then(10)
            .otherwise(1).alias("est_month"),
        ])

        # 연도별 추정 날짜 생성
        df = df.with_columns([
            (pl.lit("") + pl.col("year").cast(pl.Utf8) + "-" + pl.col("est_month").cast(pl.Utf8) + "-01")
            .str.to_date().alias("date")
        ])
        df = df.with_columns([
            (pl.col("date") - pl.duration(days=pl.col("date").dt.weekday())).alias("week_start")
        ])

        # 산업별 주간 집계
        pivot = (
            df.group_by(["week_start", "c2_nm"])
            .agg(pl.col("dt").mean().alias("emp_count"))
            .group_by("week_start")
            .agg(pl.col("emp_count").mean().alias("emp_workplace_avg"))
            .sort("week_start")
        )

        # 주간 선형보간
        pivot = pivot.with_columns([
            pl.col("emp_workplace_avg").interpolate().forward_fill().backward_fill()
        ])

        # week_start를 datetime[μs]로 캐스트
        pivot = pivot.with_columns(pl.col("week_start").cast(pl.Datetime("us")))

        return pivot
    except Exception as e:
        log.warning(f"  employment_workplace 로드 실패: {e}")
        return pl.DataFrame()


def _load_employment_monthly(db_path: str) -> pl.DataFrame:
    """employment_monthly → 월별 산업별 종사자."""
    try:
        df = _read_sql("""
            SELECT prd_de, itm_nm, dt
            FROM employment_monthly
            WHERE prd_de IS NOT NULL AND itm_nm IS NOT NULL
            ORDER BY prd_de
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        # prd_de = "YYYYMM" → "YYYY-MM-01"로 변환
        df = df.with_columns([
            (pl.lit("") + pl.col("prd_de").str.slice(0, 4) + "-" + pl.col("prd_de").str.slice(4, 2) + "-01")
            .str.to_date().alias("date")
        ])
        df = df.with_columns([
            (pl.col("date") - pl.duration(days=pl.col("date").dt.weekday())).alias("week_start")
        ])

        # 월별 평균
        monthly = (
            df.group_by("week_start")
            .agg(pl.col("dt").mean().alias("emp_monthly_avg"))
            .sort("week_start")
        )

        # 선형보간
        monthly = monthly.with_columns([
            pl.col("emp_monthly_avg").interpolate().forward_fill().backward_fill().alias("emp_monthly_avg")
        ])

        # week_start를 datetime[μs]로 캐스트
        monthly = monthly.with_columns(pl.col("week_start").cast(pl.Datetime("us")))

        return monthly
    except Exception as e:
        log.warning(f"  employment_monthly 로드 실패: {e}")
        return pl.DataFrame()


def _load_employment_residence(db_path: str) -> pl.DataFrame:
    """employment_residence → 거주지 기준 고용률.

    역학 활용: 거주지-직장 불일치 = 통근 이동량 proxy
      workplace_avg와 비교하면 유입/유출 패턴 파악 가능
    """
    try:
        df = _read_sql("""
            SELECT prd_de, c2_nm, dt
            FROM employment_residence
            WHERE prd_de IS NOT NULL AND c2_nm IS NOT NULL
            ORDER BY prd_de
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        df = df.with_columns([
            (pl.col("prd_de").str.slice(0, 4).cast(pl.Int32)).alias("year"),
            (pl.col("prd_de").str.slice(4, 2).cast(pl.Int32)).alias("half"),
        ])
        df = df.with_columns([
            pl.when(pl.col("half") == 1).then(4)
            .when(pl.col("half") == 2).then(10)
            .otherwise(1).alias("est_month"),
        ])
        df = df.with_columns([
            (pl.col("year").cast(pl.Utf8) + "-" + pl.col("est_month").cast(pl.Utf8) + "-01")
            .str.to_date().alias("date")
        ])
        df = df.with_columns([
            (pl.col("date") - pl.duration(days=pl.col("date").dt.weekday())).alias("week_start")
        ])

        pivot = (
            df.group_by("week_start")
            .agg(pl.col("dt").mean().alias("emp_residence_avg"))
            .sort("week_start")
        )
        pivot = pivot.with_columns([
            pl.col("emp_residence_avg").interpolate().forward_fill().backward_fill()
        ])
        pivot = pivot.with_columns(pl.col("week_start").cast(pl.Datetime("us")))
        return pivot
    except Exception as e:
        log.warning(f"  employment_residence 로드 실패: {e}")
        return pl.DataFrame()


__all__ = [
    "_load_school_info",
    "_load_hospitals",
    "_load_employment_workplace",
    "_load_employment_monthly",
    "_load_employment_residence",
]
