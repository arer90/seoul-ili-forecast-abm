"""External / cross-cutting loaders — Google Trends, school closures,
WHO FluNet positivity, Korean holidays.

Sprint β Item 5 full migration (2026-05-26): real bodies moved here from
`loaders.py`. The legacy `loaders` module re-exports these for back-compat.
"""
from __future__ import annotations

import logging

import polars as pl

from ..utils import _read_sql

log = logging.getLogger(__name__)


# ── Google Trends 검색 트렌드 ───────────────────────────────────────────────
def _load_google_search_trends(db_path: str) -> pl.DataFrame:
    """google_search_trends → 주간 키워드별 검색 관심도.

    키워드를 pivot하여 주간 피처로 변환.
    Columns: gt_flu(독감), gt_fever(발열), gt_tamiflu(타미플루),
             gt_cold(감기), gt_hospital(소아과)
    """
    try:
        df = _read_sql("""
            SELECT period, keyword, interest
            FROM google_search_trends
            WHERE period IS NOT NULL AND interest IS NOT NULL
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        # 핵심 키워드만 선택하고 pivot
        kw_map = {
            "독감": "gt_flu",
            "인플루엔자": "gt_influenza",
            "발열": "gt_fever",
            "기침": "gt_cough",
            "타미플루": "gt_tamiflu",
            "감기": "gt_cold",
            "소아과": "gt_pediatric",
            "응급실": "gt_er",
            "콧물": "gt_runny_nose",
            "몸살": "gt_bodyache",
        }

        # 키워드 필터 + 이름 변환
        df = df.filter(pl.col("keyword").is_in(list(kw_map.keys())))
        df = df.with_columns([
            pl.col("keyword").replace_strict(kw_map, default="gt_other").alias("feat_name"),
        ])

        # week_start 계산
        df = df.with_columns([
            pl.col("period").str.to_datetime("%Y-%m-%d").alias("date"),
        ])
        df = df.with_columns([
            (pl.col("date") - pl.duration(days=pl.col("date").dt.weekday())).alias("week_start"),
        ])

        # pivot: 각 키워드를 컬럼으로
        pivoted = (
            df.group_by(["week_start", "feat_name"])
            .agg(pl.col("interest").mean())
            .pivot(on="feat_name", index="week_start", values="interest")
            .with_columns(pl.col("week_start").cast(pl.Datetime("us")))
            .sort("week_start")
        )
        return pivoted
    except Exception:
        return pl.DataFrame()


# ── 학교 휴업/휴교 데이터 ───────────────────────────────────────────────────
def _load_school_closure(db_path: str) -> pl.DataFrame:
    """school_closure_seoul → 주간 학교 휴업 건수.

    NEIS 학사일정에서 휴업/휴교 이벤트 수를 주간 집계.
    Columns: sch_closure_count (주간 휴업 이벤트 수)
    """
    try:
        df = _read_sql("""
            SELECT date, is_closure, event_name
            FROM school_closure_seoul
            WHERE date IS NOT NULL AND is_closure = 1
        """, db_path)

        if df.is_empty():
            return pl.DataFrame()

        df = df.with_columns([
            pl.col("date").str.to_datetime("%Y%m%d").alias("dt"),
        ])
        df = df.with_columns([
            (pl.col("dt") - pl.duration(days=pl.col("dt").dt.weekday())).alias("week_start"),
        ])

        weekly = (
            df.group_by("week_start")
            .agg([
                pl.len().alias("sch_closure_count"),
            ])
            .with_columns(pl.col("week_start").cast(pl.Datetime("us")))
            .sort("week_start")
        )
        return weekly
    except Exception:
        return pl.DataFrame()


def _load_flunet_positivity(db_path: str) -> pl.DataFrame:
    """WHO FluNet KR — influenza positivity + subtype shares.

    Phase D.3 (sprint 2026-05-06) — G1=A3 implementation. KDCA → WHO FluID
    official reporting channel (paper §4.5). KDCA virological surveillance
    data accessed via WHO FluNet (2014–2026, n=443 valid weeks).

    Computed weekly features at ISO_WEEKSTARTDATE:
        flu_positivity      = INF_ALL / SPEC_PROCESSED_NB
        flu_AH3_share       = AH3 / INF_ALL
        flu_AH1_share       = (AH1 + AH1N12009) / INF_ALL
        flu_BVic_share      = (BVIC_2DEL + ...4 variants) / INF_ALL
        flu_BYam_share      = BYAM / INF_ALL

    Subsequent lag1 transformation in builder.py for walk-forward leakage
    prevention (G-186). Influenza-only ILI proxy I_t* = flu_positivity_lag1 *
    ILI_rate(t) computed downstream as sensitivity target (paper §sensitivity).

    Returns:
        polars DataFrame keyed on `week_start` (ISO week Monday) with 5 features,
        OR empty DataFrame on failure.
    """
    try:
        df = _read_sql("""
            SELECT
                sdate,
                CAST(spec_processed AS REAL) AS specs,
                CAST(inf_total AS REAL) AS inf_all,
                CAST(inf_a_h3 AS REAL) AS ah3,
                CAST(inf_a_h1 AS REAL) AS ah1,
                CAST(inf_a_h1n1pdm09 AS REAL) AS ah1pdm,
                CAST(inf_b_victoria AS REAL) AS bvic,
                CAST(inf_b_yamagata AS REAL) AS byam
            FROM who_flunet
            WHERE country = 'Republic of Korea'
              AND sdate IS NOT NULL
            ORDER BY sdate
        """, db_path)
    except Exception as e:
        log.warning(f"  [_load_flunet_positivity] SQL fail: {e}")
        return pl.DataFrame()

    if df.is_empty():
        return pl.DataFrame()

    try:
        df = df.with_columns([
            pl.col("sdate")
            .str.strptime(pl.Datetime("us"), "%Y-%m-%d", strict=False)
            .alias("week_start")
        ]).drop("sdate")
    except Exception as e:
        log.warning(f"  [_load_flunet_positivity] datetime parse fail: {e}")
        return pl.DataFrame()

    # Aggregate Influenza A H1 (sum H1 + H1N1pdm09)
    df = df.with_columns([
        (pl.col("ah1").fill_null(0.0)
         + pl.col("ah1pdm").fill_null(0.0)).alias("ah1_tot"),
    ])

    for c in ["specs", "inf_all", "ah3", "ah1_tot", "bvic", "byam"]:
        df = df.with_columns([
            pl.col(c).fill_null(0.0).fill_nan(0.0).alias(c)
        ])

    df = df.with_columns([
        pl.when(pl.col("specs") > 0)
        .then(pl.col("inf_all") / pl.col("specs"))
        .otherwise(0.0).alias("flu_positivity"),
        pl.when(pl.col("inf_all") > 0)
        .then(pl.col("ah3") / pl.col("inf_all"))
        .otherwise(0.0).alias("flu_AH3_share"),
        pl.when(pl.col("inf_all") > 0)
        .then(pl.col("ah1_tot") / pl.col("inf_all"))
        .otherwise(0.0).alias("flu_AH1_share"),
        pl.when(pl.col("inf_all") > 0)
        .then(pl.col("bvic") / pl.col("inf_all"))
        .otherwise(0.0).alias("flu_BVic_share"),
        pl.when(pl.col("inf_all") > 0)
        .then(pl.col("byam") / pl.col("inf_all"))
        .otherwise(0.0).alias("flu_BYam_share"),
    ])

    return df.select([
        "week_start",
        "flu_positivity",
        "flu_AH3_share", "flu_AH1_share",
        "flu_BVic_share", "flu_BYam_share",
    ]).sort("week_start")


def _load_korean_holiday(db_path: str = "") -> pl.DataFrame:
    """Korean holiday calendar — weekly count + Lunar/Chuseok flag.

    Phase D.2 (sprint 2026-05-06). Uses `holidays.KR()` python library
    (no DB query — date-based). Subsequent lag1 transformation in builder.py
    (clinic closure → ILI reporting artifact 1-2 week lag, paper §4.5).

    Args:
        db_path: ignored (kept for loader signature consistency).

    Returns:
        polars DataFrame keyed on `week_start` (ISO week Monday) with:
          holiday_count       (int, # of public holidays in week)
          is_lunar_chuseok    (int, 1 if 설/추석 in week; else 0)
    """
    try:
        import holidays
    except ImportError:
        log.warning("  [_load_korean_holiday] 'holidays' lib 미설치 "
                     "(pip install holidays) — fallback empty DataFrame")
        return pl.DataFrame()

    from datetime import datetime as _dt, timedelta as _td

    try:
        kr_holidays = holidays.KR(years=range(2014, 2030))
    except Exception as e:
        log.warning(f"  [_load_korean_holiday] holidays.KR init fail: {e}")
        return pl.DataFrame()

    rows = []
    for date, name in kr_holidays.items():
        week_monday = date - _td(days=date.weekday())
        ws = _dt(week_monday.year, week_monday.month, week_monday.day)
        is_lunar = ("설" in name or "추석" in name
                    or "Lunar" in name or "Chuseok" in name)
        rows.append({
            "week_start": ws,
            "holiday_count": 1,
            "is_lunar_chuseok": 1 if is_lunar else 0,
        })

    if not rows:
        return pl.DataFrame()

    df = pl.from_dicts(rows)
    df = (df.group_by("week_start")
            .agg([pl.col("holiday_count").sum(),
                  pl.col("is_lunar_chuseok").max()])
            .sort("week_start"))
    df = df.with_columns([pl.col("week_start").cast(pl.Datetime("us"))])
    return df


__all__ = [
    "_load_google_search_trends",
    "_load_school_closure",
    "_load_flunet_positivity",
    "_load_korean_holiday",
]
