"""
Feature transformation functions for the feature engineering pipeline.

All _add_* functions that create new features from existing data.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import polars as pl

log = logging.getLogger(__name__)


# Numba JIT path for wavelet convolution. Falls back to numpy.
try:
    from numba import njit
    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore[misc]
        def _wrap(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return _wrap


@njit(cache=True, fastmath=True)
def _causal_convolve_ricker_jit(y: np.ndarray, scale: int) -> np.ndarray:
    """Causal Ricker/Mexican-Hat CWT at a single scale.

    Returns array shape (len(y),) where position i uses only y[0..i].
    Result is pre-shifted by 1 (i.e. y[i-1] is the most recent sample used).
    """
    n = y.shape[0]
    half = 3 * scale
    klen = 2 * half + 1

    # Build normalized kernel
    kernel = np.empty(klen, dtype=np.float64)
    k_sum_sq = 0.0
    for j in range(klen):
        tt = (j - half) / float(scale)
        w = (1.0 - tt * tt) * np.exp(-0.5 * tt * tt)
        kernel[j] = w
        k_sum_sq += w * w
    norm = np.sqrt(k_sum_sq + 1e-10)
    for j in range(klen):
        kernel[j] /= norm

    # "full" convolution, take first n entries (= causal convolution)
    out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        s = 0.0
        jmax = klen - 1 if klen - 1 < i else i
        for j in range(jmax + 1):
            s += y[i - j] * kernel[j]
        out[i] = s

    # shift(1): position 0 = 0, otherwise previous value
    shifted = np.empty(n, dtype=np.float64)
    shifted[0] = 0.0
    for i in range(1, n):
        shifted[i] = out[i - 1]
    return shifted


def _add_lag_features(df: pl.DataFrame, col: str, lags: list[int]) -> pl.DataFrame:
    """Lag 피처 생성."""
    for lag in lags:
        df = df.with_columns([
            pl.col(col).shift(lag).alias(f"{col}_lag{lag}")
        ])
    return df


def _add_rolling_features(df: pl.DataFrame, col: str, windows: list[int]) -> pl.DataFrame:
    """Rolling 통계 (mean, std, min, max) 피처.

    Leak-free: polars ``rolling_*(window_size=w)`` is a TRAILING (right-closed)
    window — position t aggregates x[t-w+1 .. t] (past + current only), never
    centered. The ``.shift(1)`` then makes row t strictly PAST-only
    (x[t-w .. t-1]). This makes the feature truncation-invariant: building on
    the full series and slicing ``[:k]`` equals building on the prefix ``[:k]``
    (max|Δ|=0), which the expanding-window protocol (run_expanding_multihorizon
    X_all[:k]) relies on. Guarded by tests/test_expanding_window_leakfree.py.
    """
    for w in windows:
        df = df.with_columns([
            pl.col(col).rolling_mean(window_size=w).shift(1).alias(f"{col}_rmean{w}"),
            pl.col(col).rolling_std(window_size=w).shift(1).fill_null(0).alias(f"{col}_rstd{w}"),
            pl.col(col).rolling_min(window_size=w).shift(1).alias(f"{col}_rmin{w}"),
            pl.col(col).rolling_max(window_size=w).shift(1).alias(f"{col}_rmax{w}"),
        ])
    return df


def _add_diff_features(df: pl.DataFrame, col: str, orders: list[int] = [1, 2]) -> pl.DataFrame:
    """차분 피처."""
    for d in orders:
        df = df.with_columns([
            pl.col(col).diff(n=d).shift(1).alias(f"{col}_diff{d}")
        ])
    return df


def _add_log_features(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Log1p 변환 피처 (right-skewed 분포 보정)."""
    for col in cols:
        df = df.with_columns([
            (pl.col(col).clip(lower_bound=0) + 1).log().alias(f"{col}_log1p")
        ])
    return df


def _add_quantile_encoding(df: pl.DataFrame, col: str, n_bins: int = 10,
                          train_end: Optional[int] = None) -> pl.DataFrame:
    """Quantile bin encoding."""
    if train_end is not None:
        vals = df[col].slice(0, train_end).drop_nulls()
    else:
        vals = df[col].drop_nulls()

    if len(vals) < n_bins:
        return df

    vals_np = vals.to_numpy()
    bins = np.quantile(vals_np, np.linspace(0, 1, n_bins + 1))
    bins = np.unique(bins)

    if len(bins) < 2:
        return df

    col_vals = df[col].to_numpy()
    qbin = np.searchsorted(bins[1:-1], col_vals).astype(float)

    df = df.with_columns([
        pl.lit(qbin).alias(f"{col}_qbin"),
        pl.lit(qbin / max(len(bins) - 2, 1)).alias(f"{col}_qnorm"),
    ])
    return df


def _add_binary_encoding(df: pl.DataFrame, col: str, n_bits: int = 8) -> pl.DataFrame:
    """Binary (bit-level) encoding."""
    int_vals = (df[col].fill_null(0).clip(lower_bound=0) * 10).cast(pl.Int64).clip(upper_bound=2**n_bits - 1).to_numpy().astype(int)

    for bit in range(n_bits):
        bit_values = ((int_vals >> bit) & 1).astype(float)
        df = df.with_columns([
            pl.lit(bit_values).alias(f"{col}_bit{bit}")
        ])

    return df


def _add_multi_resolution_seasonal(df: pl.DataFrame) -> pl.DataFrame:
    """Multi-resolution 주기 피처 (Fourier basis)."""
    if "week_seq" not in df.columns:
        return df

    ws = df["week_seq"].to_numpy().astype(float)

    for period in [52, 26, 13, 6.5]:
        name = str(period).replace(".", "_")
        sin_vals = np.sin(2 * np.pi * ws / period)
        cos_vals = np.cos(2 * np.pi * ws / period)
        df = df.with_columns([
            pl.lit(sin_vals).alias(f"sin_p{name}"),
            pl.lit(cos_vals).alias(f"cos_p{name}"),
        ])

    # Month-of-year encoding (circular)
    if "month" in df.columns:
        m = df["month"].to_numpy().astype(float)
        sin_month = np.sin(2 * np.pi * m / 12)
        cos_month = np.cos(2 * np.pi * m / 12)
        df = df.with_columns([
            pl.lit(sin_month).alias("sin_month"),
            pl.lit(cos_month).alias("cos_month"),
        ])

    return df


def _add_wavelet_features(df: pl.DataFrame, col: str, scales: list[int] = [4, 8, 16]) -> pl.DataFrame:
    """Continuous Wavelet Transform (Ricker/Mexican Hat) 근사 -- causal convolution.

    G-091 fix: mode="same" (centered) → mode="full" + 왼쪽 패딩(causal).
    과거 데이터만 사용하여 미래 누수를 방지.

    Numba-JIT path (`_causal_convolve_ricker_jit`) used when available —
    eliminates `np.convolve` full-array allocation + shift + roll round trip.
    """
    y = df[col].to_numpy().astype(np.float64)
    y_clean = np.nan_to_num(y, nan=0.0)
    n = y_clean.shape[0]

    for scale in scales:
        if _HAS_NUMBA:
            shifted = _causal_convolve_ricker_jit(
                np.ascontiguousarray(y_clean), int(scale)
            )
        else:
            t = np.arange(-3 * scale, 3 * scale + 1, dtype=float)
            wavelet = (1 - (t / scale) ** 2) * np.exp(-0.5 * (t / scale) ** 2)
            wavelet = wavelet / np.sqrt(np.sum(wavelet ** 2) + 1e-10)
            conv_full = np.convolve(y_clean, wavelet, mode="full")
            causal_conv = conv_full[:n]
            shifted = np.roll(causal_conv, 1)
            shifted[0] = 0

        df = df.with_columns([
            pl.lit(shifted).alias(f"{col}_wavelet{scale}")
        ])

    return df


def _add_interaction_features(df: pl.DataFrame) -> pl.DataFrame:
    """기상×ILI 교차 피처."""
    if "temp_avg" in df.columns and "ili_rate_lag1" in df.columns:
        cold_ili = (df["temp_avg"].clip(upper_bound=0).abs() * df["ili_rate_lag1"].fill_null(0)).to_numpy()
        df = df.with_columns(pl.lit(cold_ili).alias("cold_ili"))

        if "humidity" in df.columns:
            humid_ili = (df["humidity"].fill_null(50) * df["ili_rate_lag1"].fill_null(0) / 100).to_numpy()
            df = df.with_columns(pl.lit(humid_ili).alias("humid_ili"))

    if "pop_inflow" in df.columns and "ili_rate_lag1" in df.columns:
        pop_norm = (df["pop_inflow"].fill_null(0) / (df["pop_inflow"].max() + 1)).to_numpy()
        inflow_ili = pop_norm * df["ili_rate_lag1"].fill_null(0).to_numpy()
        df = df.with_columns(pl.lit(inflow_ili).alias("inflow_ili"))

    lag1_vals = df["ili_rate_lag1"].fill_null(0).to_numpy() if "ili_rate_lag1" in df.columns else None

    if lag1_vals is not None:
        if "subway_total_avg" in df.columns:
            sub_norm = (df["subway_total_avg"].fill_null(0) / (df["subway_total_avg"].max() + 1)).to_numpy()
            subway_ili = sub_norm * lag1_vals
            df = df.with_columns(pl.lit(subway_ili).alias("subway_ili"))

        if "bus_total_avg" in df.columns:
            bus_norm = (df["bus_total_avg"].fill_null(0) / (df["bus_total_avg"].max() + 1)).to_numpy()
            bus_ili = bus_norm * lag1_vals
            df = df.with_columns(pl.lit(bus_ili).alias("bus_ili"))

        if "hpop_peak_ratio" in df.columns:
            peak_ratio_ili = df["hpop_peak_ratio"].fill_null(1).to_numpy() * lag1_vals
            df = df.with_columns(pl.lit(peak_ratio_ili).alias("peak_ratio_ili"))

        if "er_bed_avg" in df.columns:
            er_inv = 1.0 / (df["er_bed_avg"].fill_null(1).clip(lower_bound=0.1)).to_numpy()
            er_inv_norm = er_inv / (er_inv.max() + 1e-6)
            er_burden_ili = er_inv_norm * lag1_vals
            df = df.with_columns(pl.lit(er_burden_ili).alias("er_burden_ili"))

        if "emp_contact_ratio" in df.columns:
            emp_contact_ili = df["emp_contact_ratio"].fill_null(0).to_numpy() * lag1_vals
            df = df.with_columns(pl.lit(emp_contact_ili).alias("emp_contact_ili"))

        if "wp_commuter_inflow" in df.columns:
            wp_norm = (df["wp_commuter_inflow"].fill_null(0) / (df["wp_commuter_inflow"].max() + 1)).to_numpy()
            wp_inflow_ili = wp_norm * lag1_vals
            df = df.with_columns(pl.lit(wp_inflow_ili).alias("wp_inflow_ili"))

        if "hs_congestion_ratio" in df.columns:
            hs_norm = (df["hs_congestion_ratio"].fill_null(1) / (df["hs_congestion_ratio"].max() + 1)).to_numpy()
            hs_congestion_ili = hs_norm * lag1_vals
            df = df.with_columns(pl.lit(hs_congestion_ili).alias("hs_congestion_ili"))

        # ── RT 확장 상호작용 (2026-04-11) ──

        # 지하철 밀집도 × ILI: 밀폐 공간 접촉 강도
        if "rt_sub_acml_total_avg" in df.columns:
            sub_vals = df["rt_sub_acml_total_avg"].fill_null(0).to_numpy()
            sub_norm = sub_vals / (sub_vals.max() + 1e-6)
            sub_crowd_ili = sub_norm * lag1_vals
            df = df.with_columns(pl.lit(sub_crowd_ili).alias("rt_subcrowd_ili"))

        # 도로 혼잡도 × ILI: 이동 밀도
        if "rt_road_cong_avg" in df.columns:
            road_vals = df["rt_road_cong_avg"].fill_null(0).to_numpy()
            road_norm = road_vals / (road_vals.max() + 1e-6)
            road_cong_ili = road_norm * lag1_vals
            df = df.with_columns(pl.lit(road_cong_ili).alias("rt_roadcong_ili"))

        # 비거주자 비율 × ILI: 외부 유입 감염 위험
        if "rt_popdet_nonresnt_avg" in df.columns:
            nonresnt = df["rt_popdet_nonresnt_avg"].fill_null(0).to_numpy()
            nonresnt_norm = nonresnt / (nonresnt.max() + 1e-6)
            nonresnt_ili = nonresnt_norm * lag1_vals
            df = df.with_columns(pl.lit(nonresnt_ili).alias("rt_nonresnt_ili"))

        # 고위험 연령 비율 × ILI: 감수성 인구 밀도
        if "rt_popdet_highrisk_age" in df.columns:
            hrisk = df["rt_popdet_highrisk_age"].fill_null(0).to_numpy()
            hrisk_norm = hrisk / (hrisk.max() + 1e-6)
            hrisk_ili = hrisk_norm * lag1_vals
            df = df.with_columns(pl.lit(hrisk_ili).alias("rt_highrisk_ili"))

    return df


def _add_epidemic_phase_features(df: pl.DataFrame, train_ratio: float = 0.8) -> pl.DataFrame:
    """유행 단계(phase) 지표.

    G-093 fix: 전체 데이터 median → train-only median 사용.
    """
    if "ili_rate" not in df.columns:
        return df

    ili = df["ili_rate"].to_numpy().copy()
    n_train = int(len(ili) * train_ratio)

    # train-only median으로 baseline 계산 (G-093)
    baseline = np.nanmedian(ili[:n_train]) if n_train > 0 else np.nanmedian(ili)
    threshold = baseline * 2

    above = (ili > threshold).astype(float)
    above_rolled = np.roll(above, 1)
    above_rolled[0] = 0
    df = df.with_columns(pl.lit(above_rolled).alias("above_threshold"))

    diffs = np.diff(ili, prepend=ili[0])
    rising = (diffs > 0).astype(int)
    consec_rise = np.zeros_like(ili)
    for i in range(1, len(ili)):
        consec_rise[i] = (consec_rise[i - 1] + 1) * rising[i]
    consec_rise_rolled = np.roll(consec_rise, 1)
    consec_rise_rolled[0] = 0
    df = df.with_columns(pl.lit(consec_rise_rolled).alias("consec_rise"))

    cumili = []
    prev_season = None
    cum = 0
    season_col = df["season_start"].to_numpy() if "season_start" in df.columns else np.zeros(len(df))

    for i, s in enumerate(season_col):
        if s != prev_season:
            cum = 0
            prev_season = s
        cum += ili[i]
        cumili.append(cum)

    cumili_rolled = np.roll(cumili, 1)
    cumili_rolled[0] = 0
    df = df.with_columns(pl.lit(cumili_rolled).alias("season_cum_ili"))

    return df


def _add_multi_resolution_agg(df: pl.DataFrame) -> pl.DataFrame:
    """Multi-resolution 집계 피처."""
    if "ili_rate" not in df.columns:
        return df

    ili = df["ili_rate"].to_numpy().copy()
    n = len(ili)

    # ── 월별 집계 ──
    if "month" in df.columns:
        month_vals = df["month"].to_numpy()
        monthly_avg = np.full(n, np.nan)
        monthly_max = np.full(n, np.nan)
        monthly_std = np.full(n, np.nan)

        for i in range(1, n):
            same_month = [ili[j] for j in range(max(0, i - 5), i)
                          if month_vals[j] == month_vals[i]]
            if same_month:
                monthly_avg[i] = np.mean(same_month)
                monthly_max[i] = np.max(same_month)
                monthly_std[i] = np.std(same_month) if len(same_month) > 1 else 0

        df = df.with_columns([
            pl.lit(monthly_avg).alias("mr_month_avg"),
            pl.lit(monthly_max).alias("mr_month_max"),
            pl.lit(monthly_std).alias("mr_month_std"),
        ])

    # ── 분기별 (최근 13주) 통계 ──
    q_avg = np.full(n, np.nan)
    q_max = np.full(n, np.nan)
    q_trend = np.full(n, np.nan)

    for i in range(13, n):
        window = ili[i - 13:i]
        q_avg[i] = np.mean(window)
        q_max[i] = np.max(window)
        x = np.arange(13, dtype=float)
        q_trend[i] = np.polyfit(x, window, 1)[0]

    df = df.with_columns([
        pl.lit(q_avg).alias("mr_quarter_avg"),
        pl.lit(q_max).alias("mr_quarter_max"),
        pl.lit(q_trend).alias("mr_quarter_trend"),
    ])

    # ── 시즌 컨텍스트 ── (Leakage fix: 완료된 과거 시즌만 사용)
    if "season_start" in df.columns:
        seasons = df["season_start"].to_numpy()

        prev_season_mean = np.full(n, np.nan)
        season_ratio = np.full(n, np.nan)

        # expanding: 시점 i까지 관측된 과거 완료 시즌의 평균만 사용
        completed_season_means = {}

        for i in range(n):
            s = seasons[i]
            # 이전 시즌이 끝났는지 체크
            if i > 0 and seasons[i] != seasons[i - 1]:
                prev_s = seasons[i - 1]
                mask = seasons[:i] == prev_s
                completed_season_means[prev_s] = np.mean(ili[:i][mask])

            prev_s = s - 1
            if prev_s in completed_season_means:
                prev_season_mean[i] = completed_season_means[prev_s]
                if completed_season_means[prev_s] > 0:
                    cur_season_so_far = [ili[j] for j in range(max(0, i)) if seasons[j] == s and j < i]
                    if cur_season_so_far:
                        season_ratio[i] = np.mean(cur_season_so_far) / completed_season_means[prev_s]

        df = df.with_columns([
            pl.lit(prev_season_mean).alias("mr_prev_season_mean"),
            pl.lit(season_ratio).alias("mr_season_ratio"),
        ])

    # ── 전년 동기 대비 (52주 전과 비교) ──
    yoy_ratio = np.full(n, np.nan)
    yoy_diff = np.full(n, np.nan)

    for i in range(52, n):
        if ili[i - 52] > 0:
            yoy_ratio[i] = ili[i - 1] / ili[i - 52]
            yoy_diff[i] = ili[i - 1] - ili[i - 52]

    df = df.with_columns([
        pl.lit(yoy_ratio).alias("mr_yoy_ratio"),
        pl.lit(yoy_diff).alias("mr_yoy_diff"),
    ])

    # ── 장기 추세 (26주 causal 이동평균) ── (G-090 fix: centered→causal)
    if n > 27:
        causal_ma26 = np.full(n, np.nan)
        for i in range(25, n):
            causal_ma26[i] = np.mean(ili[i - 25:i + 1])  # ili[i-25] ~ ili[i]
        # shift by 1: 시점 t에서는 t-1까지의 MA만 사용
        causal_ma26_shifted = np.roll(causal_ma26, 1)
        causal_ma26_shifted[0] = np.nan

        # gradient도 causal shifted 기준
        causal_grad = np.full(n, np.nan)
        for i in range(2, n):
            if not np.isnan(causal_ma26_shifted[i]) and not np.isnan(causal_ma26_shifted[i - 1]):
                causal_grad[i] = causal_ma26_shifted[i] - causal_ma26_shifted[i - 1]

        df = df.with_columns([
            pl.lit(np.nan_to_num(causal_ma26_shifted, nan=0.0)).alias("mr_trend_26w"),
            pl.lit(np.nan_to_num(causal_grad, nan=0.0)).alias("mr_trend_26w_diff"),
        ])

    # NaN interpolation — G-121: numpy NaN → fill_nan 먼저, 그 다음 fill_null
    # pl.lit(np.nan) 은 float NaN 으로 들어가므로 forward_fill/fill_null 에 안 잡힘.
    mr_cols = [c for c in df.columns if c.startswith("mr_")]
    for c in mr_cols:
        if c in df.columns:
            df = df.with_columns([
                pl.col(c).fill_nan(None).forward_fill().fill_null(0).alias(c)
            ])

    log.info(f"  Multi-resolution features: {len(mr_cols)}")
    return df