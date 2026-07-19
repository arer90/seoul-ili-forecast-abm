"""
sanitize.py — NaN / Inf / null 방어 유틸리티 (G-121)
=====================================================

Polars 1.x 에서 numpy NaN ≠ Polars null 이므로,
feature engineering 파이프라인 곳곳에서 float NaN 이
fill_null(0) 을 통과해 sklearn / DL 모델을 크래시시킨다.

이 모듈은 3 레이어 방어를 제공한다:
  1. sanitize_polars_df  — Polars DataFrame (builder.py 출구)
  2. sanitize_numpy      — numpy array     (phase1 출구, phase7/8 입구)
  3. sanitize_predictions — 예측 배열        (runner.py 모델 출력)
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
#  1. Polars DataFrame sanitizer
# ──────────────────────────────────────────────────────────────

def sanitize_polars_df(df, fill_value: float = 0.0):
    """Polars DataFrame 의 float NaN + Polars null 을 한 번에 정리.

    Parameters
    ----------
    df : pl.DataFrame
    fill_value : float, default 0.0

    Returns
    -------
    df : pl.DataFrame  (cleaned, same schema)
    n_fixed : int       (수정된 셀 수, 0 이면 클린)
    """
    import polars as pl

    float_cols = [c for c in df.columns
                  if df.schema[c] in (pl.Float32, pl.Float64)]
    non_float_cols = [c for c in df.columns if c not in float_cols]

    # float 컬럼: fill_nan → fill_null
    # non-float 컬럼: fill_null 만
    exprs = (
        [pl.col(c).fill_nan(fill_value).fill_null(fill_value) for c in float_cols]
        + [pl.col(c).fill_null(fill_value) for c in non_float_cols]
    )
    if exprs:
        df_clean = df.with_columns(exprs)
    else:
        df_clean = df

    # 수정 개수 추정 (null_count 기반 — NaN 은 정확히 못 세지만 0 이면 클린)
    n_null_before = sum(df[c].null_count() for c in df.columns)
    n_nan_before = sum(
        int(df[c].is_nan().sum()) for c in float_cols
    ) if float_cols else 0
    n_fixed = n_null_before + n_nan_before

    if n_fixed > 0:
        log.info(f"  [sanitize] Polars DF: {n_fixed} dirty cells → {fill_value}")

    return df_clean, n_fixed


# ──────────────────────────────────────────────────────────────
#  2. Numpy array sanitizer
# ──────────────────────────────────────────────────────────────

def sanitize_numpy(
    X: np.ndarray,
    feature_cols: Optional[Sequence[str]] = None,
    fill_value: float = 0.0,
    label: str = "",
) -> np.ndarray:
    """numpy 배열의 NaN / Inf / -Inf 를 fill_value 로 대체.

    Parameters
    ----------
    X : np.ndarray, shape (n, p) or (n,)
    feature_cols : list[str], optional — 로깅용 (어떤 피처가 dirty 인지)
    fill_value : float, default 0.0
    label : str — 로깅 접두사 (예: "Phase1", "Phase7")

    Returns
    -------
    X : np.ndarray (cleaned, same shape)
    """
    nan_mask = np.isnan(X)
    inf_mask = np.isinf(X)
    n_nan = int(nan_mask.sum())
    n_inf = int(inf_mask.sum())

    if n_nan == 0 and n_inf == 0:
        return X

    prefix = f"  [{label}]" if label else "  [sanitize]"

    if X.ndim == 2 and feature_cols is not None:
        nan_cols = np.where(nan_mask.any(axis=0))[0]
        inf_cols = np.where(inf_mask.any(axis=0))[0]
        if len(nan_cols) > 0:
            names = [feature_cols[i] for i in nan_cols[:8]]
            counts = [int(nan_mask[:, i].sum()) for i in nan_cols[:8]]
            log.warning(
                f"{prefix} NaN: {n_nan} cells in {len(nan_cols)} features — "
                f"top: {list(zip(names, counts))}"
            )
        if len(inf_cols) > 0:
            inf_names = [feature_cols[i] for i in inf_cols[:8]]
            log.warning(f"{prefix} Inf: {n_inf} cells — features: {inf_names}")
    else:
        log.warning(f"{prefix} NaN={n_nan}, Inf={n_inf}")

    X = np.nan_to_num(X, nan=fill_value, posinf=fill_value, neginf=fill_value)
    log.info(f"{prefix} → {fill_value} 대체 완료 ({n_nan} NaN, {n_inf} Inf)")
    return X


# Sprint B B3 (2026-05-26, Gemini MD audit):
# sanitize_predictions() removed — dead code, 0 active importers.
# Canonical implementation: simulation.models.base.sanitize_predictions (G-159).
