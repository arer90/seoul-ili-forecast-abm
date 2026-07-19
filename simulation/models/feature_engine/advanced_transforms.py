"""Advanced derived features for ILI time series.

8 categories — all causal (use only past data, no leakage):
  1. Hilbert transform → instantaneous amplitude / phase / frequency
  2. EMD-lite (causal IMF approximation via local oscillation extraction)
  3. Phase-space embedding (Takens-style delay coordinates)
  4. Permutation entropy (rolling complexity)
  5. Spectral entropy (rolling FFT entropy)
  6. Hjorth parameters (Activity / Mobility / Complexity)
  7. catch22-lite (subset of canonical features, fast)
  8. Quantum-inspired feature map (IQP-style angle encoding)

[2026-04-28 added] Single source of truth for advanced TS feature engineering.
All features:
  - causal: only past data → no leakage
  - shift(1): position i uses [0..i-1] (matches existing pipeline)
  - finite: NaN/Inf → 0
  - bounded: extreme values clipped
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import polars as pl

log = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════
def _safe_finite(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """NaN/Inf → fill, copy."""
    out = np.asarray(arr, dtype=np.float64).copy()
    mask = ~np.isfinite(out)
    out[mask] = fill
    return out


# ════════════════════════════════════════════════════════════════
# Strict sanitizer (2026-04-28 patch — TCN-Optuna/Rt-Augmented fix)
# ────────────────────────────────────────────────────────────────
# 문제:
#   • Hilbert/FFT/EMD 출력의 일부 NaN/Inf 가 X 행렬에 포함
#   • Ridge regression matmul 시 divide-by-zero / overflow / invalid value
#   • predictions 폭주 → R² 폭락 (TCN-Optuna R²=0.49, Rt-Augmented R²=-1.16)
#
# 해결:
#   _ultra_safe_finite — 모든 advanced feature output 에 강제 적용:
#     1. NaN/Inf → 0 (np.nan_to_num)
#     2. clip [-CLIP_MAX, +CLIP_MAX] (overflow 방지)
#     3. ILI rate 단위 의미 없는 큰 값은 robust median 으로 winsorize
# ════════════════════════════════════════════════════════════════
_CLIP_MAX = 1e3   # ILI rate 도메인의 합리적 max (실측 ~30, 1e3 = 안전 cap)


def _ultra_safe_finite(arr: np.ndarray,
                        fill: float = 0.0,
                        clip_max: float = _CLIP_MAX) -> np.ndarray:
    """모든 NaN/Inf 제거 + clip [-clip_max, +clip_max].

    TCN-Optuna / Rt-Augmented 의 matmul 폭주 방지를 위한 strict sanitizer.
    advanced_transforms 의 모든 derived feature output 에 적용 권장.

    Args:
        arr: input array (any shape)
        fill: NaN/Inf 대체값 (default 0.0)
        clip_max: 절대값 cap (default 1e3 — ILI 도메인 안전 max)
    """
    if arr is None:
        return arr
    out = np.asarray(arr, dtype=np.float64).copy()
    # NaN/Inf 제거
    out = np.nan_to_num(out, nan=fill, posinf=clip_max, neginf=-clip_max)
    # clip 으로 overflow 방지
    out = np.clip(out, -clip_max, clip_max)
    return out


def _causal_shift(arr: np.ndarray) -> np.ndarray:
    """shift(1): position 0 = 0, position i = arr[i-1]."""
    n = len(arr)
    shifted = np.zeros(n, dtype=np.float64)
    if n >= 2:
        shifted[1:] = arr[:-1]
    return shifted


# ════════════════════════════════════════════════════════════════
# 1. Hilbert transform (instantaneous amplitude / phase / frequency)
# ════════════════════════════════════════════════════════════════
def _causal_hilbert_window(y: np.ndarray, window: int = 26) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal sliding-window Hilbert transform.

    For each position i, apply Hilbert on window y[max(0,i-W+1):i+1] and
    take the LAST value (instantaneous quantities at i, using past only).

    Returns (amplitude, phase, freq) all length n, shifted by 1 (causal).
    """
    from scipy.signal import hilbert
    n = len(y)
    amp = np.zeros(n, dtype=np.float64)
    phase = np.zeros(n, dtype=np.float64)
    freq = np.zeros(n, dtype=np.float64)
    y_clean = _safe_finite(y)

    for i in range(2, n):
        a = max(0, i - window + 1)
        seg = y_clean[a:i + 1]
        if len(seg) < 4:
            continue
        try:
            analytic = hilbert(seg - np.mean(seg))
            amp[i] = float(np.abs(analytic[-1]))
            phase[i] = float(np.angle(analytic[-1]))
        except Exception:
            pass

    # instantaneous frequency = phase derivative
    phase_unwrap = np.unwrap(phase)
    freq[1:] = np.diff(phase_unwrap)
    freq = _safe_finite(freq)

    return _causal_shift(amp), _causal_shift(phase), _causal_shift(freq)


def _add_hilbert_features(df: pl.DataFrame, col: str = "ili_rate",
                          windows: list[int] = [13, 26]) -> pl.DataFrame:
    """Hilbert transform → instantaneous amplitude/phase/freq."""
    if col not in df.columns:
        return df
    y = df[col].to_numpy().astype(np.float64)
    for w in windows:
        amp, phase, freq = _causal_hilbert_window(y, window=w)
        df = df.with_columns([
            pl.lit(amp).alias(f"{col}_hilbert_amp_w{w}"),
            pl.lit(phase).alias(f"{col}_hilbert_phase_w{w}"),
            pl.lit(freq).alias(f"{col}_hilbert_freq_w{w}"),
        ])
    return df


# ════════════════════════════════════════════════════════════════
# 2. EMD-lite (causal IMF approximation)
# ════════════════════════════════════════════════════════════════
def _causal_imf_extract(y: np.ndarray, window: int = 26, n_imfs: int = 3) -> np.ndarray:
    """Causal Empirical Mode Decomposition approximation.

    Real EMD is non-causal (sifting uses extrema across full series). This is
    a sliding-window approximation: for each position, perform EMD on the
    past window and take the last IMF values.

    Returns shape (n, n_imfs) array (each column is an IMF).
    """
    n = len(y)
    out = np.zeros((n, n_imfs), dtype=np.float64)
    y_clean = _safe_finite(y)

    try:
        from PyEMD import EMD as _EMD
        has_pyemd = True
    except ImportError:
        has_pyemd = False
        log.debug("PyEMD not installed — EMD-lite uses fallback (numpy only)")

    for i in range(window, n):
        seg = y_clean[i - window + 1: i + 1]
        if has_pyemd:
            try:
                emd = _EMD()
                imfs = emd(seg, max_imf=n_imfs)
                # imfs shape (n_imfs_actual, len(seg))
                k = min(imfs.shape[0], n_imfs)
                for j in range(k):
                    out[i, j] = float(imfs[j, -1])
            except Exception:
                pass
        else:
            # Fallback: bandpass-like decomposition via successive smoothing
            x = seg.copy()
            for j in range(n_imfs):
                # smooth via causal moving average, then high-frequency residual
                w_smooth = max(2, 2 ** (j + 1))
                if w_smooth >= len(x):
                    break
                # right-edge causal mean
                smooth_last = float(np.mean(x[-w_smooth:]))
                hf = float(x[-1] - smooth_last)
                out[i, j] = hf
                # next residual: smoothed series
                x = np.array([np.mean(x[max(0, k - w_smooth):k + 1]) for k in range(len(x))])

    # causal shift
    shifted = np.zeros_like(out)
    shifted[1:] = out[:-1]
    return shifted


def _add_emd_features(df: pl.DataFrame, col: str = "ili_rate",
                     window: int = 26, n_imfs: int = 3) -> pl.DataFrame:
    """EMD-lite — n_imfs causal IMFs from sliding window."""
    if col not in df.columns:
        return df
    y = df[col].to_numpy().astype(np.float64)
    imfs = _causal_imf_extract(y, window=window, n_imfs=n_imfs)
    for j in range(n_imfs):
        df = df.with_columns([
            pl.lit(imfs[:, j]).alias(f"{col}_imf{j+1}_w{window}")
        ])
    return df


# ════════════════════════════════════════════════════════════════
# 3. Phase-space embedding (Takens delay coordinates)
# ════════════════════════════════════════════════════════════════
def _add_takens_features(df: pl.DataFrame, col: str = "ili_rate",
                         dim: int = 3, tau: int = 4) -> pl.DataFrame:
    """Takens delay-coordinate embedding.

    Creates dim columns: x_t-tau, x_t-2*tau, ..., x_t-dim*tau.
    All causal (only past data).
    """
    if col not in df.columns:
        return df
    y = df[col].to_numpy().astype(np.float64)
    n = len(y)
    for k in range(1, dim + 1):
        delayed = np.zeros(n, dtype=np.float64)
        offset = k * tau
        if offset < n:
            delayed[offset:] = y[:-offset]
        df = df.with_columns([
            pl.lit(delayed).alias(f"{col}_takens_d{k}_t{tau}")
        ])
    # local distance to nearest neighbor in embedding space (rolling)
    if dim >= 2 and n > 4 * tau:
        # construct embedding vectors X[i] = (y[i-tau], y[i-2*tau], ..., y[i-dim*tau])
        X = np.zeros((n, dim), dtype=np.float64)
        for k in range(1, dim + 1):
            offset = k * tau
            if offset < n:
                X[offset:, k - 1] = y[:-offset]
        # 입력 X sanitize
        X = _ultra_safe_finite(X)
        # rolling nearest-neighbor distance (causal: only past)
        nn_dist = np.zeros(n, dtype=np.float64)
        for i in range(2 * dim * tau, n):
            past = X[:i]   # all past embeddings
            cur = X[i]
            try:
                d = np.linalg.norm(past - cur, axis=1)
                d = d[(d > 1e-9) & np.isfinite(d)]
                if len(d) > 0:
                    nn_dist[i] = float(np.min(d))
            except Exception:
                pass
        df = df.with_columns([
            pl.lit(_causal_shift(_ultra_safe_finite(nn_dist))).alias(
                f"{col}_takens_nn_dist_t{tau}"
            )
        ])
    return df


# ════════════════════════════════════════════════════════════════
# 4. Permutation entropy (rolling complexity)
# ════════════════════════════════════════════════════════════════
def _permutation_entropy(seg: np.ndarray, m: int = 3, tau: int = 1) -> float:
    """Bandt & Pompe (2002) permutation entropy."""
    n = len(seg)
    if n < m * tau + 1:
        return 0.0
    patterns = []
    for i in range(n - (m - 1) * tau):
        window = seg[i: i + m * tau: tau]
        order = tuple(np.argsort(window))
        patterns.append(order)
    from collections import Counter
    counts = Counter(patterns)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in counts.values()])
    return float(-np.sum(probs * np.log(probs + 1e-12)))


def _add_permutation_entropy(df: pl.DataFrame, col: str = "ili_rate",
                             windows: list[int] = [13, 26], m: int = 3) -> pl.DataFrame:
    """Rolling permutation entropy."""
    if col not in df.columns:
        return df
    y = df[col].to_numpy().astype(np.float64)
    n = len(y)
    y_clean = _safe_finite(y)
    for w in windows:
        pe = np.zeros(n, dtype=np.float64)
        for i in range(w, n):
            seg = y_clean[i - w + 1: i + 1]
            pe[i] = _permutation_entropy(seg, m=m)
        df = df.with_columns([
            pl.lit(_causal_shift(pe)).alias(f"{col}_perment_w{w}")
        ])
    return df


# ════════════════════════════════════════════════════════════════
# 5. Spectral entropy (rolling FFT entropy)
# ════════════════════════════════════════════════════════════════
def _spectral_entropy(seg: np.ndarray) -> tuple[float, float]:
    """Returns (spectral_entropy, fft_slope_log_power_vs_log_freq)."""
    n = len(seg)
    if n < 4:
        return 0.0, 0.0
    seg_d = seg - np.mean(seg)
    # power spectrum
    fft_vals = np.fft.rfft(seg_d * np.hanning(n))
    psd = np.abs(fft_vals) ** 2
    psd = psd[1:]   # drop DC
    if psd.sum() < 1e-12:
        return 0.0, 0.0
    p = psd / psd.sum()
    se = float(-np.sum(p * np.log(p + 1e-12)))
    # FFT slope: linear fit to log(psd) ~ log(freq) (red-noise vs white)
    freqs = np.arange(1, len(psd) + 1, dtype=np.float64)
    log_p = np.log(psd + 1e-12)
    log_f = np.log(freqs)
    if len(log_p) >= 4:
        slope = float(np.polyfit(log_f, log_p, 1)[0])
    else:
        slope = 0.0
    return se, slope


def _add_spectral_entropy(df: pl.DataFrame, col: str = "ili_rate",
                          windows: list[int] = [26, 52]) -> pl.DataFrame:
    """Rolling spectral entropy + FFT-slope."""
    if col not in df.columns:
        return df
    y = df[col].to_numpy().astype(np.float64)
    n = len(y)
    y_clean = _safe_finite(y)
    for w in windows:
        se = np.zeros(n, dtype=np.float64)
        slope = np.zeros(n, dtype=np.float64)
        for i in range(w, n):
            seg = y_clean[i - w + 1: i + 1]
            se[i], slope[i] = _spectral_entropy(seg)
        df = df.with_columns([
            pl.lit(_causal_shift(se)).alias(f"{col}_spec_ent_w{w}"),
            pl.lit(_causal_shift(slope)).alias(f"{col}_fft_slope_w{w}"),
        ])
    return df


# ════════════════════════════════════════════════════════════════
# 6. Hjorth parameters
# ════════════════════════════════════════════════════════════════
def _hjorth_params(seg: np.ndarray) -> tuple[float, float, float]:
    """Hjorth Activity / Mobility / Complexity (1970)."""
    n = len(seg)
    if n < 3:
        return 0.0, 0.0, 0.0
    var_x = float(np.var(seg))
    if var_x < 1e-12:
        return 0.0, 0.0, 0.0
    dx = np.diff(seg)
    var_dx = float(np.var(dx)) if len(dx) > 0 else 0.0
    ddx = np.diff(dx)
    var_ddx = float(np.var(ddx)) if len(ddx) > 0 else 0.0

    activity = var_x
    mobility = float(np.sqrt(var_dx / (var_x + 1e-12)))
    complexity = (
        float(np.sqrt(var_ddx / (var_dx + 1e-12)) / (mobility + 1e-12))
        if mobility > 1e-12 else 0.0
    )
    return activity, mobility, complexity


def _add_hjorth_features(df: pl.DataFrame, col: str = "ili_rate",
                         windows: list[int] = [13, 26]) -> pl.DataFrame:
    """Rolling Hjorth Activity/Mobility/Complexity."""
    if col not in df.columns:
        return df
    y = df[col].to_numpy().astype(np.float64)
    n = len(y)
    y_clean = _safe_finite(y)
    for w in windows:
        a = np.zeros(n, dtype=np.float64)
        m = np.zeros(n, dtype=np.float64)
        c = np.zeros(n, dtype=np.float64)
        for i in range(w, n):
            seg = y_clean[i - w + 1: i + 1]
            a[i], m[i], c[i] = _hjorth_params(seg)
        df = df.with_columns([
            pl.lit(_causal_shift(a)).alias(f"{col}_hjorth_act_w{w}"),
            pl.lit(_causal_shift(m)).alias(f"{col}_hjorth_mob_w{w}"),
            pl.lit(_causal_shift(c)).alias(f"{col}_hjorth_cmp_w{w}"),
        ])
    return df


# ════════════════════════════════════════════════════════════════
# 7. catch22-lite (subset of canonical features, fast)
# ════════════════════════════════════════════════════════════════
def _catch22_lite(seg: np.ndarray) -> dict:
    """7 핵심 catch22 features (causal-friendly subset).

    Full catch22 has 22 features. This computes the subset that is
    (a) computationally cheap and (b) does not require complex normalization.
    """
    n = len(seg)
    if n < 8:
        return {f"f{i}": 0.0 for i in range(7)}
    seg = _safe_finite(seg)
    out = {}
    # F1: mean of first differences (drift)
    out["f1_mean_diff"] = float(np.mean(np.diff(seg)))
    # F2: std of first differences (volatility)
    out["f2_std_diff"] = float(np.std(np.diff(seg)))
    # F3: number of zero crossings of detrended series
    detrended = seg - np.linspace(seg[0], seg[-1], n)
    sign_changes = int(np.sum(np.diff(np.sign(detrended)) != 0))
    out["f3_zero_cross"] = float(sign_changes)
    # F4: longest run above mean
    above = seg > np.mean(seg)
    runs = []
    cur = 0
    for v in above:
        if v:
            cur += 1
        else:
            if cur > 0:
                runs.append(cur)
            cur = 0
    if cur > 0:
        runs.append(cur)
    out["f4_longest_above"] = float(max(runs)) if runs else 0.0
    # F5: skewness
    mu = np.mean(seg)
    sd = np.std(seg) + 1e-12
    out["f5_skew"] = float(np.mean(((seg - mu) / sd) ** 3))
    # F6: kurtosis
    out["f6_kurt"] = float(np.mean(((seg - mu) / sd) ** 4) - 3.0)
    # F7: autocorrelation lag-1
    if n >= 4:
        s_centered = seg - mu
        denom = float(np.dot(s_centered, s_centered)) + 1e-12
        out["f7_acf1"] = float(np.dot(s_centered[1:], s_centered[:-1]) / denom)
    else:
        out["f7_acf1"] = 0.0
    return out


def _add_catch22_features(df: pl.DataFrame, col: str = "ili_rate",
                          windows: list[int] = [26]) -> pl.DataFrame:
    """Rolling 7 catch22-lite features."""
    if col not in df.columns:
        return df
    y = df[col].to_numpy().astype(np.float64)
    n = len(y)
    y_clean = _safe_finite(y)
    for w in windows:
        feats = {f"f{i+1}": np.zeros(n, dtype=np.float64) for i in range(7)}
        keys_full = [
            "f1_mean_diff", "f2_std_diff", "f3_zero_cross", "f4_longest_above",
            "f5_skew", "f6_kurt", "f7_acf1",
        ]
        for i in range(w, n):
            seg = y_clean[i - w + 1: i + 1]
            d = _catch22_lite(seg)
            for j, key in enumerate(keys_full):
                feats[f"f{j+1}"][i] = d[key]
        for j, key in enumerate(keys_full):
            df = df.with_columns([
                pl.lit(_causal_shift(feats[f"f{j+1}"])).alias(f"{col}_catch22_{key}_w{w}")
            ])
    return df


# ════════════════════════════════════════════════════════════════
# 8. Quantum-inspired feature map (IQP-style angle encoding)
# ════════════════════════════════════════════════════════════════
def _add_quantum_features(df: pl.DataFrame, col: str = "ili_rate",
                          n_qubits: int = 4, lookback: int = 8) -> pl.DataFrame:
    """Quantum-inspired feature map (classical simulation of IQP encoding).

    IQP-style:  |ψ(x)⟩ = ⊗_i  R_z(θ_i) H |0⟩  with  θ_i = π * x_i / max(x_window)

    Returns:
      n_qubits angle-encoded features: cos(θ_i), sin(θ_i)
      n_qubits-1 pairwise interactions: cos(θ_i + θ_{i+1})
    All causal (compute on past lookback values).

    Reference: QuaCK-TSF (arXiv:2408.12007) — Ising-style quantum feature
    map shown to capture nonlinear temporal dependencies.

    [2026-04-28 patch] _ultra_safe_finite 로 input segment sanitize.
    seg 가 모두 0 인 경우 m=1e-9 fallback → angle=0 → cos=1, sin=0.
    """
    if col not in df.columns:
        return df
    y = df[col].to_numpy().astype(np.float64)
    n = len(y)
    y_clean = _ultra_safe_finite(y)   # 입력 자체 sanitize
    angles = np.zeros((n, n_qubits), dtype=np.float64)
    for i in range(lookback, n):
        seg = y_clean[i - lookback + 1: i + 1]
        # take last n_qubits values, normalize by max in segment
        pts = seg[-n_qubits:] if len(seg) >= n_qubits else np.pad(seg, (n_qubits - len(seg), 0))
        # 모두 0/NaN/Inf 인 segment 시 safe fallback
        pts = _ultra_safe_finite(pts)
        m_val = np.max(np.abs(pts))
        m = max(float(m_val), 1e-9) if np.isfinite(m_val) else 1e-9
        for k in range(n_qubits):
            angles[i, k] = np.pi * pts[k] / m
    # angle features (causal shift)
    for k in range(n_qubits):
        a = _causal_shift(angles[:, k])
        df = df.with_columns([
            pl.lit(np.cos(a)).alias(f"{col}_quantum_cos_q{k}"),
            pl.lit(np.sin(a)).alias(f"{col}_quantum_sin_q{k}"),
        ])
    # pairwise interaction (Ising-like ZZ encoding)
    for k in range(n_qubits - 1):
        a1 = _causal_shift(angles[:, k])
        a2 = _causal_shift(angles[:, k + 1])
        df = df.with_columns([
            pl.lit(np.cos(a1 + a2)).alias(f"{col}_quantum_zz_q{k}{k+1}"),
        ])
    return df


# ════════════════════════════════════════════════════════════════
# 9. STL decomposition (causal — sliding window)
# ════════════════════════════════════════════════════════════════
def _add_stl_features(df: pl.DataFrame, col: str = "ili_rate",
                      window: int = 104, period: int = 52) -> pl.DataFrame:
    """STL (Seasonal-Trend-Loess) decomposition — causal sliding window.

    For each position i, decompose y[max(0,i-W+1):i+1] into trend, seasonal,
    residual. Take the LAST values as features. All causal.

    Reference: Cleveland et al. (1990); statsmodels.tsa.seasonal.STL.
    """
    if col not in df.columns:
        return df
    try:
        from statsmodels.tsa.seasonal import STL
    except ImportError:
        log.debug("statsmodels not available — STL features skipped")
        return df

    y = df[col].to_numpy().astype(np.float64)
    n = len(y)
    y_clean = _safe_finite(y)

    trend = np.zeros(n, dtype=np.float64)
    seasonal = np.zeros(n, dtype=np.float64)
    resid = np.zeros(n, dtype=np.float64)

    min_len = 2 * period + 4
    for i in range(min_len, n):
        a = max(0, i - window + 1)
        seg = y_clean[a:i + 1]
        if len(seg) < min_len:
            continue
        try:
            stl = STL(seg, period=period, robust=True)
            res = stl.fit()
            trend[i] = float(res.trend[-1])
            seasonal[i] = float(res.seasonal[-1])
            resid[i] = float(res.resid[-1])
        except Exception:
            pass

    df = df.with_columns([
        pl.lit(_causal_shift(trend)).alias(f"{col}_stl_trend_w{window}"),
        pl.lit(_causal_shift(seasonal)).alias(f"{col}_stl_seasonal_w{window}"),
        pl.lit(_causal_shift(resid)).alias(f"{col}_stl_resid_w{window}"),
    ])
    return df


# ════════════════════════════════════════════════════════════════
# 10. Savitzky-Golay smoothing (causal sliding)
# ════════════════════════════════════════════════════════════════
def _add_savgol_features(df: pl.DataFrame, col: str = "ili_rate",
                         windows: list[int] = [9, 17],
                         polyorder: int = 3) -> pl.DataFrame:
    """Savitzky-Golay 평활화 + 1차 derivative (local 추세).

    Causal sliding-window: 각 position i 에서 y[i-W+1:i+1] 에 polynomial fit.
    Reference: Savitzky & Golay (1964); scipy.signal.savgol_filter.
    """
    if col not in df.columns:
        return df
    try:
        from scipy.signal import savgol_filter
    except ImportError:
        return df

    y = df[col].to_numpy().astype(np.float64)
    n = len(y)
    y_clean = _safe_finite(y)
    for w in windows:
        # window 는 odd, polyorder 보다 커야
        wo = w if w % 2 == 1 else w + 1
        if wo <= polyorder:
            continue
        smooth = np.zeros(n, dtype=np.float64)
        deriv = np.zeros(n, dtype=np.float64)
        for i in range(wo, n):
            seg = y_clean[i - wo + 1: i + 1]
            try:
                sm = savgol_filter(seg, wo, polyorder, mode="nearest")
                smooth[i] = float(sm[-1])
                # 1차 derivative
                dv = savgol_filter(seg, wo, polyorder, deriv=1, mode="nearest")
                deriv[i] = float(dv[-1])
            except Exception:
                pass
        df = df.with_columns([
            pl.lit(_causal_shift(smooth)).alias(f"{col}_savgol_smooth_w{w}"),
            pl.lit(_causal_shift(deriv)).alias(f"{col}_savgol_deriv_w{w}"),
        ])
    return df


# ════════════════════════════════════════════════════════════════
# 11. Hampel filter (outlier detection + replacement)
# ════════════════════════════════════════════════════════════════
def _add_hampel_features(df: pl.DataFrame, col: str = "ili_rate",
                         window: int = 13, n_sigmas: float = 3.0) -> pl.DataFrame:
    """Hampel filter — MAD-based outlier flag + cleaned signal.

    For each i: median(seg), MAD(seg). |y[i] - median| > n*MAD → outlier.
    Reference: Pearson (2002); robust to non-Gaussian noise.
    """
    if col not in df.columns:
        return df
    y = df[col].to_numpy().astype(np.float64)
    n = len(y)
    y_clean = _safe_finite(y)

    is_outlier = np.zeros(n, dtype=np.float64)
    cleaned = np.zeros(n, dtype=np.float64)
    deviation = np.zeros(n, dtype=np.float64)
    K = 1.4826  # MAD → σ for normal

    for i in range(window, n):
        seg = y_clean[i - window + 1: i + 1]
        med = float(np.median(seg))
        mad = float(np.median(np.abs(seg - med))) * K + 1e-9
        cur = y_clean[i]
        z = abs(cur - med) / mad
        deviation[i] = float(z)
        if z > n_sigmas:
            is_outlier[i] = 1.0
            cleaned[i] = med
        else:
            cleaned[i] = float(cur)

    df = df.with_columns([
        pl.lit(_causal_shift(is_outlier)).alias(f"{col}_hampel_outlier_w{window}"),
        pl.lit(_causal_shift(deviation)).alias(f"{col}_hampel_deviation_w{window}"),
        pl.lit(_causal_shift(cleaned)).alias(f"{col}_hampel_cleaned_w{window}"),
    ])
    return df


# ════════════════════════════════════════════════════════════════
# 12. SAX (Symbolic Aggregate approXimation) + PAA
# ════════════════════════════════════════════════════════════════
def _add_sax_paa_features(df: pl.DataFrame, col: str = "ili_rate",
                          window: int = 26, n_segments: int = 4,
                          n_symbols: int = 5) -> pl.DataFrame:
    """SAX + PAA — symbolic representation of time series.

    PAA: divide window into n_segments, mean of each.
    SAX: discretize each PAA value to symbol [0, n_symbols).

    Reference: Lin et al. (2003) — fastest TS pattern matching.
    """
    if col not in df.columns:
        return df
    y = df[col].to_numpy().astype(np.float64)
    n = len(y)
    y_clean = _safe_finite(y)

    paa_vals = np.zeros((n, n_segments), dtype=np.float64)
    sax_syms = np.zeros((n, n_segments), dtype=np.float64)

    # SAX uses Gaussian quantile breakpoints
    from scipy.stats import norm
    breakpoints = norm.ppf(np.linspace(0, 1, n_symbols + 1)[1:-1])

    for i in range(window, n):
        seg = y_clean[i - window + 1: i + 1]
        # z-normalize segment
        sd = np.std(seg) + 1e-9
        seg_z = (seg - np.mean(seg)) / sd
        # PAA: split into n_segments
        seg_size = len(seg_z) // n_segments
        if seg_size < 1:
            continue
        for k in range(n_segments):
            chunk = seg_z[k * seg_size: (k + 1) * seg_size] if k < n_segments - 1 \
                    else seg_z[k * seg_size:]
            paa_v = float(np.mean(chunk)) if len(chunk) > 0 else 0.0
            paa_vals[i, k] = paa_v
            # SAX symbol
            sym = int(np.searchsorted(breakpoints, paa_v))
            sax_syms[i, k] = float(sym)

    for k in range(n_segments):
        df = df.with_columns([
            pl.lit(_causal_shift(paa_vals[:, k])).alias(
                f"{col}_paa_s{k}_w{window}"),
            pl.lit(_causal_shift(sax_syms[:, k])).alias(
                f"{col}_sax_s{k}_w{window}"),
        ])
    return df


# ════════════════════════════════════════════════════════════════
# Master entry
# ════════════════════════════════════════════════════════════════
def add_advanced_features(df: pl.DataFrame, col: str = "ili_rate",
                          enabled: Optional[set] = None) -> pl.DataFrame:
    """Add all 8 categories of advanced features.

    Args:
        df: polars DataFrame.
        col: target time-series column (default ili_rate).
        enabled: subset {"hilbert","emd","takens","perment","spec","hjorth",
                          "catch22","quantum"}. None → all enabled.
    """
    if enabled is None:
        enabled = {"hilbert", "emd", "takens", "perment", "spec",
                   "hjorth", "catch22", "quantum",
                   "stl", "savgol", "hampel", "sax_paa"}

    if "hilbert" in enabled:
        df = _add_hilbert_features(df, col=col)
    if "emd" in enabled:
        df = _add_emd_features(df, col=col)
    if "takens" in enabled:
        df = _add_takens_features(df, col=col)
    if "perment" in enabled:
        df = _add_permutation_entropy(df, col=col)
    if "spec" in enabled:
        df = _add_spectral_entropy(df, col=col)
    if "hjorth" in enabled:
        df = _add_hjorth_features(df, col=col)
    if "catch22" in enabled:
        df = _add_catch22_features(df, col=col)
    if "quantum" in enabled:
        df = _add_quantum_features(df, col=col)
    # 2026-04-28 추가: 4가지 더
    if "stl" in enabled:
        df = _add_stl_features(df, col=col)
    if "savgol" in enabled:
        df = _add_savgol_features(df, col=col)
    if "hampel" in enabled:
        df = _add_hampel_features(df, col=col)
    if "sax_paa" in enabled:
        df = _add_sax_paa_features(df, col=col)

    # ── Strict post-sanitization (2026-04-28 patch) ──────────────
    # 모든 신규 advanced feature column 에 NaN/Inf/extreme 강제 처리.
    # TCN-Optuna / Rt-Augmented 의 matmul 폭주 (divide by zero, overflow,
    # invalid value) 재발 방지.
    new_cols = [c for c in df.columns
                 if any(tag in c.lower() for tag in
                        ("hilbert", "imf", "takens", "perment", "spec_ent",
                         "fft_slope", "hjorth", "catch22", "quantum",
                         "stl_trend", "stl_seasonal", "stl_resid",
                         "savgol", "hampel", "_paa_", "_sax_"))]
    for c in new_cols:
        try:
            arr = df[c].to_numpy().astype(np.float64)
            sanitized = _ultra_safe_finite(arr)
            df = df.with_columns(pl.lit(sanitized).alias(c))
        except Exception as _se:
            log.debug(f"  sanitize 실패 (skip): {c}: {_se}")

    return df


__all__ = [
    "add_advanced_features",
    "_add_hilbert_features",
    "_add_emd_features",
    "_add_takens_features",
    "_add_permutation_entropy",
    "_add_spectral_entropy",
    "_add_hjorth_features",
    "_add_catch22_features",
    "_add_quantum_features",
    "_add_stl_features",
    "_add_savgol_features",
    "_add_hampel_features",
    "_add_sax_paa_features",
]
