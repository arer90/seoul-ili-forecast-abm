"""Vaccination loaders — adult flu coverage + childhood vaccine coverage.

Sprint β Item 5 full migration (2026-05-26): real bodies moved here from
`loaders.py`. The legacy `loaders` module re-exports these for back-compat.
"""
from __future__ import annotations

import logging

import polars as pl

from ..utils import _read_sql

log = logging.getLogger(__name__)


def _load_vaccination(db_path: str) -> pl.DataFrame:
    """vaccination_coverage → 연도별 서울시 인플루엔자 접종률."""
    df = _read_sql("""
        SELECT ref_year, coverage_pct
        FROM vaccination_coverage
        WHERE vaccine_nm LIKE '%인플루엔자%'
          AND gu_nm = '서울특별시'
          AND (age_group IS NULL OR age_group = '')
        ORDER BY ref_year
    """, db_path)

    df = df.rename({"ref_year": "year", "coverage_pct": "vax_coverage"})
    return df


def _load_childhood_vaccination(db_path: str) -> pl.DataFrame:
    """childhood_vaccination_rates → 서울 연령별 접종률.

    역학 활용:
      - DTaP/MMR/수두/B형간염 접종률 = 집단면역 수준
      - 접종률 하락 시기 = 감수성 인구 증가 → 유행 위험 ↑
    """
    try:
        # 서울 지역 주요 백신 연도별 접종률
        df = _read_sql("""
            SELECT ref_year, vaccine_nm, coverage_pct
            FROM childhood_vaccination_rates
            WHERE region_nm LIKE '%서울%' AND sex = '계'
              AND ref_year IS NOT NULL
            ORDER BY ref_year
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        # 주요 백신별 연도 평균
        yearly = (
            df.group_by("ref_year")
            .agg([
                pl.col("coverage_pct").mean().alias("child_vax_avg"),
                pl.col("coverage_pct").min().alias("child_vax_min"),
            ])
            .sort("ref_year")
        )

        yearly = yearly.with_columns([
            (pl.col("ref_year").cast(pl.Utf8) + "-07-01").str.to_date().alias("date")
        ])
        yearly = yearly.with_columns([
            (pl.col("date") - pl.duration(days=pl.col("date").dt.weekday())).alias("week_start"),
        ])

        result = yearly.select([
            "week_start", "child_vax_avg", "child_vax_min"
        ]).with_columns(pl.col("week_start").cast(pl.Datetime("us")))

        return result.sort("week_start")
    except Exception as e:
        log.warning(f"  childhood_vaccination_rates 로드 실패: {e}")
        return pl.DataFrame()


__all__ = ["_load_vaccination", "_load_childhood_vaccination"]
