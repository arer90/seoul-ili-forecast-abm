"""Sentinel surveillance SQL loaders — KDCA Sentinel ILI/ARI/SARI/HFMD/Enterovirus
+ broader weekly_disease aggregate.

Sprint β Item 5 full migration (2026-05-26): real bodies moved here from
`loaders.py`. The legacy `loaders` module re-exports these for back-compat.

Functions:
- _load_sentinel_ili         — weekly ILI rate (KDCA Sentinel 200+ clinics)
                              (also returns cal_date/year/month/iso_week —
                               anchor for builder.py join chain, R8 in plan)
- _load_sentinel_ari         — weekly ARI rate (acute respiratory infections)
- _load_sentinel_sari        — SARI (severe acute respiratory infections)
- _load_sentinel_hfmd        — HFMD (hand-foot-mouth disease)
- _load_sentinel_enterovirus — enterovirus weekly counts
- _load_weekly_disease       — broader KDCA weekly disease aggregate
"""
from __future__ import annotations

import logging

import polars as pl

from ..utils import _read_sql, _season_weekseq_to_date

log = logging.getLogger(__name__)


def _load_sentinel_ili(db_path: str) -> pl.DataFrame:
    """sentinel_influenza → 주차별 전체 + 연령군별 ILI rate."""
    df_raw = _read_sql("""
        SELECT season_start, week_seq, week_label, age_group, ili_rate
        FROM sentinel_influenza
        ORDER BY season_start, week_seq, age_group
    """, db_path)

    # 전체 평균 ILI rate
    df_total = (
        df_raw.group_by(["season_start", "week_seq", "week_label"])
        .agg(ili_rate=pl.col("ili_rate").mean())
        .sort(["season_start", "week_seq"])
    )

    # 연령군별 ILI rate (피벗 → wide format)
    age_pivot = df_raw.pivot(
        on="age_group",
        index=["season_start", "week_seq"],
        values="ili_rate",
        aggregate_function="mean"
    )

    # 컬럼명 정리
    age_cols = {}
    for col in age_pivot.columns:
        if col not in ("season_start", "week_seq"):
            safe = col.replace("세", "").replace("-", "_").replace(" ", "").replace("이상", "p")
            age_cols[col] = f"ili_age_{safe}"
    age_pivot = age_pivot.rename(age_cols)

    # 합치기
    df = df_total.join(age_pivot, on=["season_start", "week_seq"], how="left")

    # 달력 날짜 생성
    dates = []
    for row in df.iter_rows(named=True):
        dt = _season_weekseq_to_date(int(row["season_start"]), int(row["week_seq"]))
        dates.append(dt)

    df = df.with_columns(pl.Series("cal_date", dates, dtype=pl.Date))
    df = df.with_columns([
        pl.col("cal_date").dt.year().alias("year"),
        pl.col("cal_date").dt.month().alias("month"),
        pl.col("cal_date").dt.week().alias("iso_week"),
    ])

    return df


def _load_sentinel_ari(db_path: str) -> pl.DataFrame:
    """sentinel_ari → 주간 급성호흡기감염병 병원체별 집계."""
    try:
        df = _read_sql("""
            SELECT year, week_no, pathogen_group, pathogen_nm, count
            FROM sentinel_ari
            ORDER BY year, week_no
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        # year, week_no로부터 week_start 계산
        df = df.with_columns([
            (pl.lit("") + pl.col("year").cast(pl.Utf8) + "-01-01").str.to_date().alias("year_start")
        ])
        df = df.with_columns([
            (pl.col("year_start") + pl.duration(days=(pl.col("week_no") - 1) * 7)).alias("week_start")
        ])

        # 병원체별 주간 집계
        weekly = (
            df.group_by(["week_start", "pathogen_group", "pathogen_nm"])
            .agg(pl.col("count").sum().alias("ari_count"))
            .sort("week_start")
        )

        # 병원체별 피봇
        pathogen_pivot = weekly.pivot(
            on="pathogen_nm",
            index="week_start",
            values="ari_count",
            aggregate_function="sum"
        ).fill_null(0)

        # 컬럼명 정리
        rename_map = {}
        for col in pathogen_pivot.columns:
            if col != "week_start":
                safe = col.replace(" ", "_").replace("/", "_").lower()
                rename_map[col] = f"ari_{safe}"
        pathogen_pivot = pathogen_pivot.rename(rename_map)

        # week_start를 datetime[μs]로 캐스트
        pathogen_pivot = pathogen_pivot.with_columns(pl.col("week_start").cast(pl.Datetime("us")))

        return pathogen_pivot
    except Exception as e:
        log.warning(f"  sentinel_ari 로드 실패: {e}")
        return pl.DataFrame()


def _load_weekly_disease(db_path: str) -> pl.DataFrame:
    """weekly_disease → 주간 감염병 발생 (전국 집계)."""
    try:
        df = _read_sql("""
            SELECT year, week_no, disease_nm, cases
            FROM weekly_disease
            WHERE week_no IS NOT NULL AND cases >= 0
            ORDER BY year, week_no, disease_nm
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        # year, week_no로부터 week_start 계산
        df = df.with_columns([
            (pl.lit("") + pl.col("year").cast(pl.Utf8) + "-01-01").str.to_date().alias("year_start")
        ])
        df = df.with_columns([
            (pl.col("year_start") + pl.duration(days=(pl.col("week_no") - 1) * 7)).alias("week_start")
        ])

        # 질병별 주간 전국 발생수
        weekly = (
            df.group_by(["week_start", "disease_nm"])
            .agg(pl.col("cases").sum().alias("cases_count"))
            .sort("week_start")
        )

        # 질병별 피봇
        disease_pivot = weekly.pivot(
            on="disease_nm",
            index="week_start",
            values="cases_count",
            aggregate_function="sum"
        ).fill_null(0)

        # 컬럼명 정리
        rename_map = {}
        for col in disease_pivot.columns:
            if col != "week_start":
                safe = col.replace(" ", "_").replace("(", "").replace(")", "").lower()
                rename_map[col] = f"dis_{safe}"
        disease_pivot = disease_pivot.rename(rename_map)

        # week_start를 datetime[μs]로 캐스트
        disease_pivot = disease_pivot.with_columns(pl.col("week_start").cast(pl.Datetime("us")))

        return disease_pivot
    except Exception as e:
        log.warning(f"  weekly_disease 로드 실패: {e}")
        return pl.DataFrame()


def _load_sentinel_sari(db_path: str) -> pl.DataFrame:
    """sentinel_sari → 주간 중증 급성호흡기감염 건수.

    역학 활용: 중증도 지표 — ILI는 경증 중심, SARI는 입원 필요 중증.
      SARI/ILI 비율 = 유행 심각도.
    """
    try:
        df = _read_sql("""
            SELECT year, week_no, count
            FROM sentinel_sari
            WHERE year IS NOT NULL AND week_no IS NOT NULL
            ORDER BY year, week_no
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        df = df.with_columns([
            (pl.col("year").cast(pl.Utf8) + "-01-01").str.to_date().alias("year_start")
        ])
        df = df.with_columns([
            (pl.col("year_start") + pl.duration(days=(pl.col("week_no") - 1) * 7)).alias("week_start")
        ])

        weekly = (
            df.group_by("week_start")
            .agg(pl.col("count").sum().alias("sari_count"))
            .with_columns(pl.col("week_start").cast(pl.Datetime("us")))
            .sort("week_start")
        )
        return weekly
    except Exception as e:
        log.warning(f"  sentinel_sari 로드 실패: {e}")
        return pl.DataFrame()


def _load_sentinel_hfmd(db_path: str) -> pl.DataFrame:
    """sentinel_hfmd → 주간 수족구병 발생률.

    역학 활용: 소아 감염 동시유행 지표 → 학교/어린이집 전파 활성도
    """
    try:
        df = _read_sql("""
            SELECT year, week_no, rate
            FROM sentinel_hfmd
            WHERE year IS NOT NULL AND week_no IS NOT NULL
            ORDER BY year, week_no
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        df = df.with_columns([
            (pl.col("year").cast(pl.Utf8) + "-01-01").str.to_date().alias("year_start")
        ])
        df = df.with_columns([
            (pl.col("year_start") + pl.duration(days=(pl.col("week_no") - 1) * 7)).alias("week_start")
        ])

        weekly = (
            df.group_by("week_start")
            .agg(pl.col("rate").mean().alias("hfmd_rate"))
            .with_columns(pl.col("week_start").cast(pl.Datetime("us")))
            .sort("week_start")
        )
        return weekly
    except Exception as e:
        log.warning(f"  sentinel_hfmd 로드 실패: {e}")
        return pl.DataFrame()


def _load_sentinel_enterovirus(db_path: str) -> pl.DataFrame:
    """sentinel_enterovirus → 주간 엔테로바이러스 검출 건수.

    역학 활용: 장바이러스 유행 = 소아 집단감염 활성 proxy
    """
    try:
        df = _read_sql("""
            SELECT year, week_no, count
            FROM sentinel_enterovirus
            WHERE year IS NOT NULL AND week_no IS NOT NULL
            ORDER BY year, week_no
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        df = df.with_columns([
            (pl.col("year").cast(pl.Utf8) + "-01-01").str.to_date().alias("year_start")
        ])
        df = df.with_columns([
            (pl.col("year_start") + pl.duration(days=(pl.col("week_no") - 1) * 7)).alias("week_start")
        ])

        weekly = (
            df.group_by("week_start")
            .agg(pl.col("count").sum().alias("enterovirus_count"))
            .with_columns(pl.col("week_start").cast(pl.Datetime("us")))
            .sort("week_start")
        )
        return weekly
    except Exception as e:
        log.warning(f"  sentinel_enterovirus 로드 실패: {e}")
        return pl.DataFrame()


__all__ = [
    "_load_sentinel_ili",
    "_load_sentinel_ari",
    "_load_weekly_disease",
    "_load_sentinel_sari",
    "_load_sentinel_hfmd",
    "_load_sentinel_enterovirus",
]
