"""
P1 real_forecaster: Real Forecast Evaluation (HWP §3 4-way split)
=================================================================

Evaluate the final pipeline on the **real** slab — weeks AFTER the HWP
analysis cutoff (paper_cutoff_week=337). The real slab is truly out-of-sample:
never seen by R2 baseline training, R7 conformal calibration, R4 WF-CV,
or R11 SHAP. This satisfies HWP §3's "forecasting performance demonstration"
clause.

R6 audit MINOR #7 (2026-05-26) — P1 (real_forecaster) vs R10 (per_model_eval) role clarification:
  P1 real_forecaster = OPERATIONAL evaluation on 8-week real-data slab (post-cutoff).
             Use case: monitoring + rolling-origin 1-step-ahead forecast.
             Metric set: 8 keys (point + regime PI coverage).
             NOT a replacement for R10 (per_model_eval) — runs on a different (smaller)
             slab to mimic real-world deployment.
  R10 per_model_eval = FULL evaluation on test slab (n=37 for KDCA 2025).
             Use case: paper-grade reporting with 122-metric SSOT.
  Both are reported in the R12 (comprehensive_eval) report.

──────────────────────────────────────────────────────────────────────────────
Evaluation strategy
──────────────────────────────────────────────────────────────────────────────

For each candidate model (best in-sample model + naive baselines):

  1. ROLLING-ORIGIN 1-STEP-AHEAD FORECAST
     For each real week i in [0, real_n):
       train_window = X_in ⊕ X_real[:i]    (in-sample + already-revealed real)
       y_train      = y_in ⊕ y_real[:i]
       y_pred[i]    = model.fit(train_window, y_train).predict(X_real[i:i+1])
     This mimics what a real surveillance system does: re-train each week
     as new observations land, predict 1 week ahead.

  2. CONFORMAL PI from in-sample OOF residuals
     Lower/upper at α = {0.10, 0.20, 0.50} (= 90/80/50 % PI)
     Quantile uses the split-conformal ceiling-based formula
     k = ⌈(n+1)(1−α)⌉ − 1 (0-indexed) of Lei et al. (2018) JASA 113:1094 /
     Vovk-Gammerman-Shafer (2005) for finite-sample coverage guarantee.

  3. NAIVE BASELINES (sanity check)
     - persistence:    y_t = y_{t-1}
     - seasonal_naive: y_t = y_{t-52}   (same week last year)
     - ar1:            y_t = α + β · y_{t-1}    (refit each step)

──────────────────────────────────────────────────────────────────────────────
Metrics (all 21 available analytics functions wired in)
──────────────────────────────────────────────────────────────────────────────

  AI / point-forecast (7):
    R², MAE, MSE, RMSE, MAPE, sMAPE, direction_accuracy

  Probabilistic / interval (6):
    CRPS (Gaussian), Pinball loss, PI coverage, PI calibration table,
    Weighted Interval Score (WIS), PIT histogram

  Epidemic-curve (3):
    Peak-week error, Peak-intensity error, KDCA threshold-crossing F1

  Clinical / alert (3):
    Brier score, Brier skill score, Binary clinical rates
    (sensitivity / specificity / PPV / NPV / F1) at KDCA threshold

  Statistical inference (2):
    Diebold-Mariano (vs each baseline), McNemar (direction)

  Stability (1):
    Bootstrap 95% CI for MAE, RMSE, R²

──────────────────────────────────────────────────────────────────────────────
Outputs
──────────────────────────────────────────────────────────────────────────────
  simulation/results/real_eval/
    ├─ summary.json              top-line metrics
    ├─ metrics_full.json         per-model × per-metric grid
    ├─ predictions.csv           date, y_true, all model predictions, PI bounds
    ├─ baselines.json            persistence / seasonal_52 / ar1 metrics
    └─ report.md                 markdown summary for thesis appendix
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

log = logging.getLogger(__name__)

# KDCA epidemiological alert threshold for ILI (per 1,000 outpatient consults).
# KDCA recomputes this each season as: mean(non-epidemic-season ILI) + 2σ
# over the prior 3 non-epidemic seasons.  Hard-coding a single value is wrong;
# use the season-specific value matching the real-slab dates.
#
# Sources:
#   - 2023-24: 6.5  (KDCA 2023 인플루엔자 관리지침)
#   - 2024-25: 8.6  (KDCA 2024-25절기 인플루엔자 관리지침)
#   - 2025-26: 9.1  (KDCA 2025-10-17 유행주의보 발령)
KDCA_THRESHOLD_BY_SEASON: dict[int, float] = {
    2019: 6.0,
    2020: 6.0,
    2021: 6.0,
    2022: 6.5,
    2023: 6.5,
    2024: 8.6,
    2025: 9.1,
}
# Default to the latest published value for unknown seasons.
KDCA_ALERT_THRESHOLD_DEFAULT = 9.1


def _season_for(date_or_season) -> Optional[int]:
    """Return the flu-season start year for a given date.
    A flu "season" runs Sep S → Aug S+1, so 2026-02-22 → 2025."""
    import datetime as _dt
    if isinstance(date_or_season, (int, np.integer)):
        return int(date_or_season)
    try:
        d = (date_or_season.astype("datetime64[D]").item()
             if isinstance(date_or_season, np.datetime64)
             else date_or_season)
        if isinstance(d, _dt.datetime):
            d = d.date()
        return d.year if d.month >= 9 else d.year - 1
    except Exception:
        return None


def _kdca_threshold_for(date_or_season) -> float:
    """Return the KDCA epidemic threshold valid at the given date/season."""
    season = _season_for(date_or_season)
    if season is None:
        return KDCA_ALERT_THRESHOLD_DEFAULT
    return KDCA_THRESHOLD_BY_SEASON.get(season, KDCA_ALERT_THRESHOLD_DEFAULT)


# Back-compat alias — points to the latest season's value.
KDCA_ALERT_THRESHOLD = KDCA_ALERT_THRESHOLD_DEFAULT


# ═══════════════════════════════════════════════════════════════════════════
# Naive baselines — give the audience a yardstick for "how good is good".
# ═══════════════════════════════════════════════════════════════════════════
def _persistence_forecast(y_in: np.ndarray, y_real_so_far: np.ndarray) -> float:
    """y_t = y_{t-1}: just repeat the last observed value."""
    if len(y_real_so_far) > 0:
        return float(y_real_so_far[-1])
    return float(y_in[-1])


def _seasonal_naive_forecast(y_in: np.ndarray, y_real_so_far: np.ndarray,
                              lag: int = 52) -> float:
    """y_t = y_{t-52}: same week last year."""
    full = np.concatenate([y_in, y_real_so_far])
    if len(full) >= lag:
        return float(full[-lag])
    return float(y_in[-1])


def _ar1_forecast(y_in: np.ndarray, y_real_so_far: np.ndarray) -> float:
    """y_t = α + β · y_{t-1}: re-fit each step (closed-form OLS)."""
    full = np.concatenate([y_in, y_real_so_far])
    if len(full) < 3:
        return float(full[-1])
    y_t = full[1:]
    y_lag = full[:-1]
    cov = np.cov(y_lag, y_t, ddof=0)[0, 1]
    var = np.var(y_lag)
    if var < 1e-10:
        return float(y_t[-1])
    beta = cov / var
    alpha = float(y_t.mean() - beta * y_lag.mean())
    return float(alpha + beta * full[-1])


# ═══════════════════════════════════════════════════════════════════════════
# Rolling-origin 1-step-ahead forecast for an sklearn-compatible model.
# ═══════════════════════════════════════════════════════════════════════════
def _rolling_origin_multihorizon(
    runner_factory,
    model_name: str,
    X_in: np.ndarray, y_in: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    horizons: tuple = (1, 2, 3, 4),
    feature_cols: Optional[list] = None,
) -> dict:
    """Multi-horizon rolling-origin **RECURSIVE** forecast on the TEST slab (leakage-free).

    G-325 (2026-06-19, 3-AI 검토 + 사용자 #3): 이전 구현은 ``X_query=X_test[i+h-1]`` 로 **실제 미래
    lag 를 feature 로 사용** → h≥2 에서 누수(낙관적, 무효 decay table). 진짜 h-step 예측 = origin
    에서 미관측 주의 lag 를 **모델 자신의 예측으로 recursive** 채워야 함. 여기서 수정.

    각 origin t (관측 = X_in ⊕ X_test[:t]) 에서 1회 학습한 모델로 t, t+1, …, t+max_h-1 재귀 예측:
    target 주의 lag 컬럼(ili_rate_lag{1,2,4,52})을 buffer(관측+이전 예측)로 override, calendar/
    seasonal feature 는 실제값(미래에도 known) 사용. h-step 예측이 horizon-decay 를 노출.

    Args:
        feature_cols: lag 컬럼 식별용(ili_rate_lag{n}). None 이면 lag override 없음(univariate sequence
                      모델은 X 무시하므로 무관).

    Returns:
        {h: predictions (n_test,)} — preds[t] = origin t 에서 h-step 후(t+h-1 주) 예측. NaN=범위밖.

    Side effects: 모델 1회 학습(고정) + n_test×max_h forward pass (refit-per-origin 아님 = 경량).
    Caller responsibility: 1-step champion 선정과 별개의 **진단(decay table)** 용 — 선정엔 미사용.
    """
    import re as _re
    n_test = len(y_test)
    horizons = tuple(horizons)
    max_h = max(horizons) if horizons else 1
    out: dict[int, np.ndarray] = {h: np.full(n_test, np.nan, dtype=np.float64) for h in horizons}

    lag_map: dict[int, int] = {}    # {lag_steps: col_idx} — calendar 은 미래에도 known → override X
    if feature_cols is not None:
        for _idx, _c in enumerate(feature_cols):
            _m = _re.match(r"ili_rate_lag(\d+)$", str(_c))
            if _m:
                lag_map[int(_m.group(1))] = _idx

    try:    # 모델 1회 학습(고정 — decay table 진단용; refit-per-origin 은 비싸고 선정과 무관)
        runner = runner_factory()
        runner.fit(np.asarray(X_in, dtype=np.float64), np.asarray(y_in, dtype=np.float64))
    except Exception as e:  # noqa: BLE001
        log.debug(f"  [multihorizon] {model_name} fit fail: {e}")
        return out

    X_test = np.asarray(X_test, dtype=np.float64)
    y_obs = np.asarray(y_test, dtype=np.float64)
    _yin = np.asarray(y_in, dtype=np.float64)
    for t in range(n_test):
        buffer = list(_yin) + list(y_obs[:t])   # origin t 에서 known(관측만)
        for k in range(max_h):
            target = t + k
            if target >= n_test:
                break
            X_q = X_test[target].copy()
            for _lag_steps, _col in lag_map.items():     # 미관측 주는 buffer(예측 포함)로 override
                _src = len(buffer) - _lag_steps
                if 0 <= _src < len(buffer):
                    X_q[_col] = buffer[_src]
            try:
                _yh = runner.predict(X_q[None, :], model_name=model_name)
                _pred = (float(np.asarray(_yh).ravel()[0]) if _yh is not None and len(_yh)
                         else (buffer[-1] if buffer else 0.0))
            except Exception:  # noqa: BLE001
                _pred = buffer[-1] if buffer else 0.0
            _pred = max(_pred, 0.0)
            buffer.append(_pred)
            _h = k + 1
            if _h in out:
                out[_h][t] = _pred
    return out


def _rolling_origin_forecast(
    runner_factory,
    model_name: str,
    X_in: np.ndarray, y_in: np.ndarray,
    X_real: np.ndarray, y_real: np.ndarray,
    feature_cols: Optional[list] = None,
) -> Optional[np.ndarray]:
    """For each real week i, refit on (in-sample ⊕ real[:i]) and predict real[i].

    S0-3 leakage fix: rebuild quantile-bin / interaction / above-threshold
    features at each step using ONLY rows revealed up to that step. Without
    this, transforms.py:227-256 (and similar) bake in `pop_inflow.max()` /
    `subway_total_avg.max()` etc. computed over the FULL 345-week clock —
    including the real slab itself — which contaminates X_in's normalised
    columns with post-cutoff statistics.

    `feature_cols`: required for the per-fold recoders. If None, recoding
    is skipped (legacy behaviour, leakage-prone).

    Returns y_pred of length len(y_real), or None on failure.
    """
    n_real = len(y_real)
    n_in = len(y_in)
    preds = np.full(n_real, np.nan, dtype=np.float64)

    # Lazy import — phase6_wfcv depends on a lot of stuff.
    try:
        from simulation.pipeline.wfcv import (
            _recode_quantile_features_per_fold,
            _recode_above_threshold_per_fold,
            _recode_interaction_features_per_fold,
        )
        _recoders_ok = feature_cols is not None
    except Exception as _ie:
        log.warning(f"  [phase12] recoders unavailable ({_ie}) — leakage-prone fit")
        _recoders_ok = False

    for i in range(n_real):
        try:
            X_train = np.vstack([X_in, X_real[:i]]) if i > 0 else X_in
            y_train = np.concatenate([y_in, y_real[:i]]) if i > 0 else y_in
            X_query = X_real[i:i+1]

            # Rebuild leakage-prone features using only [:n_in+i]
            if _recoders_ok:
                X_full = np.vstack([X_train, X_query])
                train_end = n_in + i
                X_full = _recode_quantile_features_per_fold(X_full, feature_cols, train_end)
                X_full = _recode_above_threshold_per_fold(
                    X_full, np.concatenate([y_train, [0.0]]),  # query y unknown
                    feature_cols, train_end,
                )
                X_full = _recode_interaction_features_per_fold(X_full, feature_cols, train_end)
                X_train = X_full[:train_end]
                X_query = X_full[train_end:train_end+1]

            runner = runner_factory()
            runner.fit(X_train, y_train)
            y_hat = runner.predict(X_query, model_name=model_name)
            if y_hat is not None and len(y_hat) >= 1:
                preds[i] = float(y_hat[0])
        except Exception as e:
            log.warning(f"  [phase12] {model_name} step {i}: {type(e).__name__}: {e}")
    return preds if np.isfinite(preds).any() else None


# ═══════════════════════════════════════════════════════════════════════════
# Aggregate all 21 metrics for one (model, predictions) pair.
# ═══════════════════════════════════════════════════════════════════════════
def _weather_observed_columns(feature_cols: list) -> list[int]:
    """Indices of feature columns that are observed-week weather (perfect-
    foresight risk). Excludes KMA forecast columns (`fcst_*`), lag columns
    (`*_lag*`), rolling stats (`rmean/rstd`), and quantile-binned versions.
    """
    out: list[int] = []
    for i, c in enumerate(feature_cols):
        cl = c.lower()
        if "fcst" in cl or "lag" in cl or "rmean" in cl or "rstd" in cl:
            continue
        if "qbin" in cl or "qnorm" in cl:
            continue
        # Names matching observed-weather aggregates (built by _load_weather)
        if cl in ("temp_avg", "temp_min", "humidity", "wind_speed", "rainfall",
                 "pressure", "sunshine", "temp_std"):
            out.append(i)
    return out


def _all_pf_risk_columns(feature_cols: list) -> dict[str, list[int]]:
    """Return all current-week observed feature column indices, grouped by
    family. These are the perfect-foresight risk surface beyond just weather.

    Families:
      - weather: temp_*, humidity, wind_*, rainfall, pressure, sunshine
      - mobility: subway_*, bus_*, sub_h_*, bus_h_*  (daily / monthly transit)
      - population: pop_*, dong_*, hpop_*           (daily living population)
      - realtime: rt_*  (excluding rt_fcst_* — those are forecasts)

    Excludes lag, rolling, qbin, qnorm, fcst_* (those are causal / future-OK).
    """
    families = {"weather": [], "mobility": [], "population": [], "realtime": []}
    for i, c in enumerate(feature_cols):
        cl = c.lower()
        # Skip causal / future-OK columns
        if "fcst" in cl or "lag" in cl or "rmean" in cl or "rstd" in cl:
            continue
        if "qbin" in cl or "qnorm" in cl:
            continue
        # Classify
        if cl in ("temp_avg", "temp_min", "humidity", "wind_speed", "rainfall",
                  "pressure", "sunshine", "temp_std"):
            families["weather"].append(i)
        elif cl.startswith(("subway_", "bus_", "sub_h_", "bus_h_")):
            families["mobility"].append(i)
        elif cl.startswith(("pop_", "dong_", "hpop_")):
            families["population"].append(i)
        elif cl.startswith("rt_"):
            families["realtime"].append(i)
    return families


def _substitute_weather_in_real(
    X_real: np.ndarray,
    real_dates: Optional[np.ndarray],
    X_in: np.ndarray,
    dates_in: Optional[np.ndarray],
    feature_cols: list,
    mode: str,
) -> np.ndarray:
    """Apply the configured weather-handling strategy to X_real for P1 (real_forecaster).

    See SplitConfig.real_weather_mode for the contract:
      - "observed":   no change (perfect-foresight bias)
      - "climatology": week-of-year mean from in-sample
      - "hybrid":     keep KMA `fcst_*` columns as-is (they're already
                      in the feature matrix and represent forecast),
                      replace observed-weather columns with climatology

    Returns a (possibly new) X_real array with substitutions applied.
    """
    # Back-compat alias: "observed" → "oracle"
    if mode == "observed":
        mode = "oracle"
    if mode == "oracle":
        log.warning(
            "  [phase12] weather_mode=oracle: PERFECT-FORESIGHT (upper bound)"
            " — not operationally achievable, demonstrates ceiling only."
        )
        return X_real
    if real_dates is None or dates_in is None:
        log.warning(
            f"  [phase12] weather_mode={mode} requested but dates "
            f"unavailable — falling back to oracle (PF upper bound)."
        )
        return X_real

    # Substitute ALL PF-risk feature families, not just weather.
    # Mobility / population / realtime are also "current-week observed"
    # and would otherwise leak post-cutoff information when read off X_real.
    families = _all_pf_risk_columns(feature_cols)
    weather_idx = (families["weather"] + families["mobility"]
                   + families["population"] + families["realtime"])
    if not weather_idx:
        log.info("  [phase12] no PF-risk columns to substitute")
        return X_real
    log.info(
        f"  [phase12] PF-risk surface: weather={len(families['weather'])} "
        f"mobility={len(families['mobility'])} "
        f"population={len(families['population'])} "
        f"realtime={len(families['realtime'])} "
        f"(total {len(weather_idx)} columns)"
    )

    # Compute week-of-year (1..52) for in-sample rows
    def _woy(date_arr: np.ndarray) -> np.ndarray:
        date_arr = (date_arr if date_arr.dtype.kind == "M"
                    else np.array(date_arr, dtype="datetime64[D]"))
        # ISO week number — week 1 starts Mon containing the first Thu.
        import datetime as _dt
        out = np.empty(len(date_arr), dtype=np.int32)
        for i, d in enumerate(date_arr):
            try:
                py = d.astype("datetime64[D]").item()
                if isinstance(py, _dt.datetime):
                    py = py.date()
                out[i] = py.isocalendar()[1]  # week-of-year 1..53
            except Exception:
                out[i] = 1
        return out

    woy_in = _woy(dates_in)
    woy_real = _woy(real_dates)

    X_real_new = X_real.copy()
    n_substituted = 0
    for col_i in weather_idx:
        for r in range(len(X_real_new)):
            target_woy = int(woy_real[r])
            mask = (woy_in == target_woy)
            if mask.any():
                X_real_new[r, col_i] = float(np.nanmean(X_in[mask, col_i]))
                n_substituted += 1
            # else leave observed value (degenerate fallback)

    log.info(
        f"  [phase12] weather_mode={mode}: substituted {n_substituted} "
        f"cells across {len(weather_idx)} observed-weather columns × "
        f"{len(X_real_new)} real weeks"
    )
    return X_real_new


def _baseline_sigma(name: str, y_in: np.ndarray) -> float:
    """Per-baseline σ for CRPS/WIS/PIT.  Each naive baseline has its own
    error variance — they should not share the best-model OOF σ.
    """
    y_in = np.asarray(y_in, dtype=np.float64)
    if name == "persistence":
        # σ = std of y_t - y_{t-1} = AR(1) one-step naive error
        return float(np.std(np.diff(y_in))) or 1e-3
    if name == "seasonal_naive":
        if len(y_in) > 52:
            return float(np.std(y_in[52:] - y_in[:-52])) or 1e-3
        return float(np.std(y_in)) or 1e-3
    if name == "ar1":
        # σ = std of in-sample residuals after fitting AR(1)
        if len(y_in) < 3:
            return float(np.std(y_in)) or 1e-3
        y_t, y_lag = y_in[1:], y_in[:-1]
        var = float(np.var(y_lag))
        if var < 1e-10:
            return float(np.std(y_in)) or 1e-3
        beta = float(np.cov(y_lag, y_t, ddof=0)[0, 1] / var)
        alpha = float(y_t.mean() - beta * y_lag.mean())
        residuals = y_t - (alpha + beta * y_lag)
        return float(np.std(residuals)) or 1e-3
    return float(np.std(np.diff(y_in))) or 1e-3


def _gate_forecast(
    pred: np.ndarray,
    y_train: np.ndarray,
    fallback: Optional[np.ndarray] = None,
    k: float = 3.0,
) -> dict:
    """Hard reject-and-replace contract gate on an operational forecast (A1/M7).

    An evaluation-best champion can extrapolate-collapse on the real slab
    (real R²=−2684; pred ≈1007 vs observed ≈21 — REAL_FORECAST_STABILITY). This
    gate REJECTS (not clips) a forecast that violates the deployment contract and
    REPLACES it with the stable ``fallback`` so the collapse never reaches the
    ABM / ARIA downstream. Loud per G-237 — the replacement is counted + reasoned,
    never silent.

    Contract (all must hold): finite ∧ nonneg ∧ pred ≤ k·max(y_train) ∧
    max|Δpred| ≤ q99.5(|Δy_train|) (no step bigger than the worst historical one).

    Args:
        pred: (h,) candidate forecast.
        y_train: in-sample history used for the caps.
        fallback: replacement forecast when the gate trips (e.g. the stable
            median ensemble). If None, the violating ``pred`` is returned with
            ``replaced=False`` so the caller decides.
        k: train-max multiplier, k∈[2,3] (3 = lenient).

    Returns:
        ``{pred, replaced, n_violations, reason}``. Never raises.
    """
    out = {"pred": np.asarray(pred, dtype=np.float64), "replaced": False,
           "n_violations": 0, "reason": "ok"}
    try:
        p = np.asarray(pred, dtype=np.float64)
        yt = np.asarray(y_train, dtype=np.float64)
        yt = yt[np.isfinite(yt)]
        if yt.size == 0:
            return out
        cap = k * float(np.max(yt))
        dcap = (float(np.quantile(np.abs(np.diff(yt)), 0.995))
                if yt.size > 2 else float("inf"))
        viol: list[str] = []
        if not np.all(np.isfinite(p)):
            viol.append("non-finite")
        if np.any(p < 0):
            viol.append("negative")
        if np.any(p > cap):
            viol.append(f"exceeds {k:g}×train_max ({np.nanmax(p):.1f}>{cap:.1f})")
        if p.size > 1 and float(np.nanmax(np.abs(np.diff(p)))) > dcap:
            viol.append(f"|Δ|>{dcap:.1f} (max {np.nanmax(np.abs(np.diff(p))):.1f})")
        out["n_violations"] = len(viol)
        out["reason"] = "; ".join(viol) if viol else "ok"
        if viol and fallback is not None:
            out["pred"] = np.asarray(fallback, dtype=np.float64)
            out["replaced"] = True
    except Exception as e:   # gate must never break the run
        out["reason"] = f"gate-error: {type(e).__name__}"
    return out


def _evaluate_model(
    name: str,
    y_true: np.ndarray, y_pred: np.ndarray,
    *,
    sigma: Optional[float] = None,        # for CRPS / WIS / PIT
    pi_lower: Optional[np.ndarray] = None,
    pi_upper: Optional[np.ndarray] = None,
    pi_levels: Optional[dict] = None,     # {alpha: (lower, upper)}
    alert_threshold: float = KDCA_ALERT_THRESHOLD,
    baseline_pred: Optional[np.ndarray] = None,   # for DM/McNemar
    y_train_pool: Optional[np.ndarray] = None,    # G-326: full-129 SSOT merge(MASE 등) 용 in-sample
) -> dict:
    """단일 model 의 21+ metric + predictions 계산 (G-169 predictions 키, D-4).

    P1 (real_forecaster) 의 real-slab evaluation. NaN-safe (모든 sub-metric 실패 시 NaN 반환).
    G-169 (2026-05-03): `predictions` 키 자동 보존 → `real_eval/per_model/
    <model>.json` dir 저장 (G-164 plot_forecast_full real_pred 의 진짜 source).

    Args:
        name: model name (e.g., "XGBoost", "DNN-Conformer").
        y_true: 실제 ILI rate (n_real,) — finite mask 자동 적용.
        y_pred: 예측 ILI rate (n_real,) — `sanitize_predictions` 후 호출 권장.
        sigma: PI / WIS / CRPS / PIT 계산용 sigma (in-sample residual std). None →
               1e-3 fallback.
        pi_lower: 95% PI lower bound (옵션). None 시 sigma 기반 normal-quantile.
        pi_upper: 95% PI upper bound (옵션).
        pi_levels: K-multi-level PI dict `{alpha: (lower, upper)}` (Bracher 2021 K=4).
        alert_threshold: KDCA alert threshold (default per-season). binary classification.
        baseline_pred: persistence baseline prediction (DM test reference).

    Returns:
        dict (21+ keys + `predictions`):
          - **name** (str), **n_valid** (int), **predictions** (list[float], G-169)
          - Point: r2, mae, rmse, mse, mape, smape, mdape, mase
          - Bias: bias_mean_error
          - Probabilistic: wis, log_wis, crps_gaussian, log_score_gauss, pinball_q05/q50/q95
          - PIT: pit_mean, pit_std, pit_ks_p
          - PI coverage: pi95_coverage, pi95_width, pi80_coverage, pi80_width, pi50_coverage, pi50_width
          - Epi: peak_week_err, peak_int_relerr, direction_acc
          - Alert: alert_threshold, brier_score, brier_skill, sensitivity, specificity, ppv, npv, alert_f1
          - Stability: mae_ci95 (BCa bootstrap)

    Raises:
        절대 raise X — sub-metric 실패는 NaN 반환.

    Performance: O(n) — n=8 (real slab) ≈ 50ms (bootstrap 2000회 포함).
    Side effects: 없음 (pure function — file write 는 caller `run_real_eval` 책임).

    Caller responsibility:
        - sigma 가 합리적 in-sample residual std (else PI/PIT 부정확).
        - pi_levels 의 alpha key 가 FLUSIGHT_ALPHAS 와 일치 (K=11).
        - run_real_eval 가 `predictions` 키를 real_eval/per_model/<m>.json 저장 (G-169).

    See: G-169 (predictions 키 신규 + per_model/ dir 저장),
         G-164 (plot_forecast_full real_pred 의 진짜 source),
         per_model_eval._evaluate_model (52 metric, test-slab uniform 별도).
    """
    from simulation.analytics import metrics as M
    from simulation.analytics import diagnostics as D

    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not mask.any():
        return {"name": name, "n_valid": 0,
                "predictions": yp.tolist() if yp.size else []}  # G-169: 빈 mask 도 prediction 보존
    a, p = yt[mask], yp[mask]
    n = len(a)

    # G-169 (2026-05-03): per-model real prediction 항상 저장 (G-164 진짜 source).
    # 이전: metrics 만 저장 → plot_forecast_full real_pred placeholder 의존.
    # 이제: predictions 키에 raw prediction array 보존 → per_model/<m>.json 저장 가능.
    out: dict[str, Any] = {
        "name": name, "n_valid": int(n),
        "predictions": yp.tolist(),     # raw (mask 적용 전 — date alignment 보존)
    }

    # ── 1. Point-forecast (7 metrics) ─────────────────────────────────────
    err = p - a
    sse = float(np.sum(err ** 2))
    sst = float(np.sum((a - a.mean()) ** 2))
    out["r2"]    = 1.0 - sse / sst if sst > 0 else float("nan")
    out["mae"]   = float(np.mean(np.abs(err)))
    out["mse"]   = float(np.mean(err ** 2))
    out["rmse"]  = float(np.sqrt(out["mse"]))
    nz = a > 0
    out["mape"]  = float(np.mean(np.abs(err[nz] / a[nz])) * 100) if nz.any() else float("nan")
    den = np.abs(a) + np.abs(p)
    keep = den > 0
    out["smape"] = float(np.mean(2.0 * np.abs(err[keep]) / den[keep]) * 100) if keep.any() else float("nan")
    try:
        out["direction_accuracy"] = M.direction_accuracy(a, p).get("accuracy", float("nan"))
    except Exception:
        out["direction_accuracy"] = float("nan")

    # ── 2. Probabilistic (6 metrics) — needs σ or explicit PI ────────────
    # σ default: residual std on in-sample, fall back to point-forecast |error|
    if sigma is None or sigma <= 0:
        sigma = float(np.std(err)) or 1e-3
    out["sigma_used"] = float(sigma)
    sig_arr = np.full_like(a, sigma)
    try:
        out["crps_gaussian"] = float(np.mean(M.crps_gaussian(a, p, sig_arr)))
    except Exception:
        out["crps_gaussian"] = float("nan")
    try:
        out["wis"] = float(np.mean(D.weighted_interval_score(a, p, sigma)))
    except Exception:
        out["wis"] = float("nan")
    try:
        out["pinball_q05"] = M.pinball_loss(a, p - 1.645 * sigma, 0.05)
        out["pinball_q50"] = M.pinball_loss(a, p, 0.50)
        out["pinball_q95"] = M.pinball_loss(a, p + 1.645 * sigma, 0.95)
    except Exception:
        out["pinball_q05"] = out["pinball_q50"] = out["pinball_q95"] = float("nan")
    if pi_lower is not None and pi_upper is not None:
        try:
            cov = M.pi_coverage(a, pi_lower[mask], pi_upper[mask], nominal=0.95)
            out["pi95_coverage"] = cov["empirical"]
            out["pi95_width"]    = cov["mean_width"]
            out["pi95_dev"]      = cov["deviation"]
        except Exception:
            pass
    if pi_levels:
        try:
            cal = M.pi_calibration_table(
                a,
                {lvl: lo[mask] for lvl, (lo, _) in pi_levels.items()},
                {lvl: hi[mask] for lvl, (_, hi) in pi_levels.items()},
            )
            out["pi_calibration"] = cal
        except Exception:
            pass
    try:
        # PIT histogram (uniformity = good calibration)
        pit = D.pit_values(a, p, sigma)
        out["pit_mean"] = float(np.mean(pit))
        out["pit_std"]  = float(np.std(pit))
    except Exception:
        out["pit_mean"] = out["pit_std"] = float("nan")

    # ── 3. Epi-curve (3 metrics) ─────────────────────────────────────────
    try:
        out["peak_week"]      = M.peak_week_error(a, p, tolerance_weeks=1)
    except Exception:
        out["peak_week"] = {}
    try:
        out["peak_intensity"] = M.peak_intensity_error(a, p, log_scale=True)
    except Exception:
        out["peak_intensity"] = {}
    # Threshold-crossing F1 / sensitivity / specificity
    try:
        ev_true = (a > alert_threshold).astype(int)
        ev_pred = (p > alert_threshold).astype(int)
        out["alert_f1"] = float(2 * np.sum(ev_true & ev_pred)
                                / max(1, np.sum(ev_true) + np.sum(ev_pred)))
        out["alert_threshold"] = float(alert_threshold)
        out["alert_n_actual_pos"]    = int(np.sum(ev_true))
        out["alert_n_predicted_pos"] = int(np.sum(ev_pred))
    except Exception:
        out["alert_f1"] = float("nan")

    # ── 4. Clinical (3 metrics) — alert as binary classification ─────────
    # Gaussian-tail probability of crossing the threshold, NOT ratio of
    # magnitudes. Ratio-of-magnitudes (a) wasn't a probability (b) used
    # `a.max()` from the test slab → leakage. Gaussian tail uses only μ̂
    # and the in-sample σ.
    try:
        from scipy.stats import norm as _N
        ev_true_bin = (a > alert_threshold).astype(int)
        # P(Y > τ | μ̂, σ̂) = 1 - Φ((τ - μ̂) / σ̂)
        sigma_safe = max(float(sigma), 1e-6)
        z = (alert_threshold - p) / sigma_safe
        ev_prob = (1.0 - _N.cdf(z)).astype(np.float64)
        out["brier_score"] = float(M.brier_score(ev_true_bin, ev_prob))
        # BSS reference = climatology prevalence (in-sample, but here we
        # use the slab base-rate as a within-slab anchor; on n=8 with
        # prevalence=1 this collapses, which is itself a finding).
        ref_p = float(np.mean(ev_true_bin))
        if 0.0 < ref_p < 1.0:
            out["brier_skill"] = float(
                M.brier_skill_score(ev_true_bin, ev_prob, ref_p)
            )
        else:
            out["brier_skill"] = float("nan")  # degenerate (all-pos or all-neg)
            out["brier_skill_note"] = (
                f"slab prevalence={ref_p:.3f} → BSS undefined; "
                f"see report.md caveat"
            )
    except Exception as _be:
        out["brier_score"] = out["brier_skill"] = float("nan")
    try:
        clin = M.binary_clinical_rates(a, p, threshold=alert_threshold)
        out["sensitivity"] = clin.get("sensitivity")
        out["specificity"] = clin.get("specificity")
        out["ppv"]         = clin.get("ppv")
        out["npv"]         = clin.get("npv")
        out["clinical_f1"] = clin.get("f1")
    except Exception:
        out["sensitivity"] = out["specificity"] = out["ppv"] = out["npv"] = float("nan")

    # ── 5. Statistical inference (2) — vs naive baseline ─────────────────
    if baseline_pred is not None and len(baseline_pred) == len(yp):
        try:
            bp = baseline_pred[mask]
            stat, pval = M.diebold_mariano(a, p, bp, h=1)
            out["dm_stat"] = float(stat)
            out["dm_pval"] = float(pval)
        except Exception:
            out["dm_stat"] = out["dm_pval"] = float("nan")
        try:
            # McNemar on direction (up vs down vs actual)
            actual_dir = np.sign(np.diff(a))
            pred_dir   = np.sign(np.diff(p))
            base_dir   = np.sign(np.diff(bp))
            corr_a = (actual_dir == pred_dir).astype(int)
            corr_b = (actual_dir == base_dir).astype(int)
            stat, pval = M.mcnemar_test(corr_a, corr_b)
            out["mcnemar_stat"] = float(stat)
            out["mcnemar_pval"] = float(pval)
        except Exception:
            out["mcnemar_stat"] = out["mcnemar_pval"] = float("nan")

    # ── 6. Stability (1) — bootstrap CI for headline metric ──────────────
    try:
        ae = np.abs(err)
        ci = M.bootstrap_ci(ae, statistic=np.mean, n_boot=2000, alpha=0.05)
        # bootstrap_ci returns dict — inspect keys lazily for back-compat
        lo = ci.get("lower", ci.get("ci_lo", ci.get("low", float("nan"))))
        hi = ci.get("upper", ci.get("ci_hi", ci.get("high", float("nan"))))
        out["mae_ci95"] = (float(lo), float(hi))
    except Exception as _be:
        out["mae_ci95"] = (float("nan"), float("nan"))

    # G-326 (2026-06-19, 사용자: 전체 eval 통일 R+P): real-slab eval(52-subset)을 full 129-metric
    #   SSOT(evaluate_predictions_full)로 확장 — 기존 키 우선 merge(r2/wis 등 정의 보존, 신규 metric만
    #   추가). y_train_pool 전달 시 MASE 등도 non-NaN.
    try:
        from simulation.pipeline.phase_evaluator import evaluate_predictions_full as _epf
        _yt = np.asarray(y_true, dtype=np.float64)
        _yp = np.asarray(y_pred, dtype=np.float64)
        _mk = np.isfinite(_yt) & np.isfinite(_yp)
        if int(_mk.sum()) >= 3:
            _ytp = (np.asarray(y_train_pool, dtype=np.float64)
                    if y_train_pool is not None and len(y_train_pool) else None)
            _f129 = _epf(_yt[_mk], _yp[_mk], sigma=float(sigma) if sigma else 1.0,
                         y_train_pool=_ytp, phase_id=f"real_{name}")
            out = {**_f129, **out}   # 기존(out) 우선 — 정의 보존, 신규만 추가
    except Exception as _e129:   # noqa: BLE001
        log.debug(f"  [real_eval] {name} full-129 SSOT merge skip: {_e129}")

    return out


from simulation.utils.resource_tracker import track_resources


def _select_champion_and_real_pred(all_results, wf_best_name, n_real):
    """Pick the operational best-model name + (optionally) its OPTIMIZED real-slab
    prediction, preferring post-optimization artifacts over the WF-CV reference.

    real_eval (P1 real_forecaster) historically evaluated the WF-CV best-R² model with a fresh
    DEFAULT-HP instance. When real_eval runs AFTER per_model_optimize (R9) +
    per_model_eval (R10), the FINAL champion (best-WIS) and its OPTIMIZED real-slab
    rolling-origin prediction already exist — this returns those so the operational /
    deployment forecast reflects the final champion (G-306). Order-robust: when those
    artifacts are absent (real_eval still dispatched before R9/R10) it returns the
    WF-CV fallback name + None, so the caller does the legacy default-HP rolling.

    Args:
        all_results: pipeline outputs. Optionally carries "per_model_eval"
            (``ranking_top10`` — best-WIS first) and "per_model_optimize"
            (``per_model_configs[name]["refit_real_predictions"]``).
        wf_best_name: WF-CV best-R² fallback name (current behaviour), or None.
        n_real: real-slab length; an optimized prediction is accepted only when its
            length matches (guards stale / partial arrays).

    Returns:
        (name, optimized_real_pred | None, source). source ∈
        {"phase13_optimized" (use the array as best_pred, skip rolling),
         "champion_default_hp" (champion known, no optimized pred → default rolling),
         "wfcv_default_hp" (no post-opt artifacts → legacy fallback)}.
        When optimized_real_pred is None the caller MUST run its own rolling forecast.

    Side effects: none (pure read of all_results).
    Caller responsibility: when optimized_real_pred is not None, use it verbatim as
        best_pred — it is already the rolling-origin 1-step real-slab forecast.
    """
    pme = all_results.get("per_model_eval") if isinstance(all_results, dict) else None
    pmo = all_results.get("per_model_optimize") if isinstance(all_results, dict) else None

    champ = None
    if isinstance(pme, dict) and not pme.get("error") and not pme.get("skipped"):
        top = pme.get("ranking_top10")
        if isinstance(top, (list, tuple)) and len(top) > 0 and isinstance(top[0], str):
            champ = top[0]
            # G-306b (3자 감사 2026-06-18): R10 (per_model_eval) 가 feature-selected refit 을 'name[fs]' 로
            #   키잉(per_model_eval._collect_fs_test_preds) → ranking_top10[0] 가 'name[fs]' 일 수
            #   있다. per_model_configs 와 REGISTRY 는 BASE name 키 → [fs] 를 strip 해야 챔피언의
            #   최적화 real 예측(cfgs.get) + fallback REGISTRY.instantiate 가 둘 다 resolve.
            #   안 하면 cfgs miss / RuntimeError(L1001) → best_pred=None 으로 운영·배포 예측 silent 소실.
            if champ.endswith("[fs]"):
                champ = champ[: -len("[fs]")]
    if champ is None:
        return wf_best_name, None, "wfcv_default_hp"

    if isinstance(pmo, dict):
        cfgs = pmo.get("per_model_configs")
        if isinstance(cfgs, dict) and isinstance(cfgs.get(champ), dict):
            pred = cfgs[champ].get("refit_real_predictions")
            if pred is not None:
                arr = np.asarray(pred, dtype=np.float64).ravel()
                if arr.shape[0] == int(n_real) and np.isfinite(arr).any():
                    return champ, arr, "phase13_optimized"
    return champ, None, "champion_default_hp"


@track_resources("real_eval")
def run_real_eval(phase1: dict, all_results: dict, config) -> dict:
    """Refit + 1-step-ahead rolling forecast on the post-IRB real slab.

    Args:
        phase1: dict from run_data — must contain real_X / real_y / real_dates
                / X_all / y_all / feature_cols.
        all_results: collected pipeline outputs (baseline, wfcv, etc.)
        config: pipeline config.
    """
    t0 = time.time()
    real_X = phase1.get("real_X")
    real_y = phase1.get("real_y")
    real_dates = phase1.get("real_dates")

    if real_X is None or real_y is None or len(real_y) == 0:
        log.info("  [phase12] real slab 비어있음 — phase10 skip")
        return {"skipped": True, "reason": "no real slab", "elapsed": time.time() - t0}

    if not bool(getattr(config.split, "real_eval_enabled", True)):
        log.info("  [phase12] real_eval_enabled=False — skip")
        return {"skipped": True, "reason": "disabled", "elapsed": time.time() - t0}

    n_real = len(real_y)
    log.info(f"  [phase12] real slab: {n_real} weeks, "
             f"{real_dates[0] if real_dates is not None else '?'} → "
             f"{real_dates[-1] if real_dates is not None else '?'}")

    X_in = phase1["X_all"]
    y_in = phase1["y_all"]
    feature_cols = phase1["feature_cols"]
    n_in = len(y_in)
    dates_in = phase1.get("dates")

    # ── Apply weather-handling strategy to real slab ────────────────────
    weather_mode = str(getattr(config.split, "real_weather_mode", "observed"))
    if weather_mode != "observed":
        log.info(f"  [phase12] weather_mode = {weather_mode}")
        real_X = _substitute_weather_in_real(
            real_X, real_dates, X_in, dates_in, feature_cols, weather_mode,
        )

    # ── Locate best model from in-sample results ─────────────────────────
    # Helper: extract R² (or another scalar metric) from heterogeneous result
    # shapes safely. R4 (WF-CV) returns wf_results with model→{overall_metrics:
    # {r2}}, R2 (baseline) returns model_results with model→{r2}.
    def _safe_r2(node) -> float:
        if not isinstance(node, dict):
            return -np.inf
        if "r2" in node and isinstance(node["r2"], (int, float)):
            return float(node["r2"])
        om = node.get("overall_metrics")
        if isinstance(om, dict) and isinstance(om.get("r2"), (int, float)):
            return float(om["r2"])
        return -np.inf

    best_name = None
    if "wfcv" in all_results and isinstance(all_results["wfcv"], dict):
        # R4 (WF-CV) returns key 'wf_results' (NOT 'model_results' — that's R2 baseline)
        wf = (all_results["wfcv"].get("wf_results")
              or all_results["wfcv"].get("model_results")
              or {})
        if isinstance(wf, dict) and wf:
            best_name = max(wf.keys(), key=lambda k: _safe_r2(wf.get(k)))
            log.info(f"  [phase12] best from WF-CV: {best_name} "
                     f"(R²={_safe_r2(wf.get(best_name)):.4f})")
    if best_name is None and "baseline" in all_results:
        bl = (all_results["baseline"] or {}).get("model_results", {})
        if isinstance(bl, dict) and bl:
            best_name = max(bl.keys(), key=lambda k: _safe_r2(bl.get(k)))
            log.info(f"  [phase12] best from Baseline: {best_name} "
                     f"(R²={_safe_r2(bl.get(best_name)):.4f})")

    if best_name is None:
        log.warning("  [phase12] no model results — only naive baselines will be evaluated")

    # ── G-306: prefer the FINAL champion (R10 per_model_eval best-WIS) + its OPTIMIZED R9
    #    (per_model_optimize) real-slab prediction when available (real_eval dispatched AFTER
    #    R9/R10). Falls back to the WF-CV best-R² name + default-HP rolling otherwise (current order).
    #    Makes the operational / deployment forecast use the final champion, not a
    #    default-HP stand-in. ``_opt_real_pred`` (if not None) short-circuits the rolling.
    best_name, _opt_real_pred, _champ_src = _select_champion_and_real_pred(
        all_results, best_name, n_real)
    if best_name is not None and _champ_src != "wfcv_default_hp":
        log.info(f"  [phase12] champion source = {_champ_src} → '{best_name}'"
                 + ("  (reusing R9 per_model_optimize optimized real prediction)"
                    if _opt_real_pred is not None else "  (default-HP rolling)"))

    # ── Compute naive baselines on real slab ─────────────────────────────
    log.info("  [phase12] computing naive baselines + hhh4-equivalent")
    base_persist = np.array([
        _persistence_forecast(y_in, real_y[:i]) for i in range(n_real)
    ])
    base_seasonal = np.array([
        _seasonal_naive_forecast(y_in, real_y[:i]) for i in range(n_real)
    ])
    base_ar1 = np.array([
        _ar1_forecast(y_in, real_y[:i]) for i in range(n_real)
    ])
    # ── S2-B: hhh4-equivalent (Held & Paul 2012) — ROLLING-ORIGIN h=1
    # Apples-to-apples with persistence/AR1: refit on (in-sample ⊕ real[:i]),
    # predict real[i] only. Static recursive 1..n_real prediction lets drift
    # compound and inflates the comparison artificially.
    base_hhh4 = None
    try:
        from simulation.models.hhh4_benchmark import HHH4Equivalent
        hhh4_preds = []
        for i in range(n_real):
            y_train = np.concatenate([y_in, real_y[:i]]) if i > 0 else y_in
            m_hhh4 = HHH4Equivalent(harmonics=2, period=52, ar_order=1)
            m_hhh4.fit(y_train)
            hhh4_preds.append(float(m_hhh4.predict_h(h=1)[0]))
        base_hhh4 = np.array(hhh4_preds, dtype=np.float64)
        log.info(f"  [phase12] hhh4-equivalent rolling: σ={m_hhh4.sigma_:.4f}")
    except Exception as e:
        log.warning(f"  [phase12] hhh4 baseline failed ({e}) — skipping")

    # ── In-sample residual std for σ (used for CRPS / WIS / PIT) ────────
    # Use OOF predictions if available (best), else fall back to naive AR(1) residuals.
    sigma_in = None
    try:
        oof = all_results.get("wfcv", {}).get("oof_predictions", {})
        if oof and best_name and best_name in oof:
            res = y_in - np.asarray(oof[best_name], dtype=np.float64)
            sigma_in = float(np.std(res[np.isfinite(res)]))
    except Exception:
        sigma_in = None
    if sigma_in is None or sigma_in <= 0:
        # Fallback: AR(1) in-sample residual std
        sigma_in = float(np.std(np.diff(y_in)))
        log.info(f"  [phase12] σ fallback (AR1 diff std): {sigma_in:.4f}")
    else:
        log.info(f"  [phase12] σ from in-sample OOF residuals: {sigma_in:.4f}")

    # ── Conformal PI from in-sample OOF residuals (S1-1, S1-2 fix) ──────
    # Compute ALL relevant α with the split-conformal ceiling formula
    # k = ⌈(n+1)(1−α)⌉ − 1 (0-indexed) due to Lei et al. (2018) JASA 113:1094
    # / Vovk-Gammerman-Shafer (2005). Note: Bracher (2021) is cited only for
    # the WIS / interval-score formulation, NOT for the conformal quantile.
    # `conformal_q[α]` = absolute residual quantile ⇒ PI half-width at level
    # (1-α). Stored as a dict so headline 95% PI uses its OWN q (not max(...)).
    conformal_q: dict[float, float] = {}
    try:
        oof = all_results.get("wfcv", {}).get("oof_predictions", {})
        if oof and best_name and best_name in oof:
            res = y_in - np.asarray(oof[best_name], dtype=np.float64)
            res = np.abs(res[np.isfinite(res)])
            n_res = len(res)
            sorted_res = np.sort(res)
            # Lei 2018 / Vovk 2005 split-conformal ceiling:
            # k = ceil((n+1)(1-α)) - 1, 0-indexed, clipped to [0, n-1]
            for alpha in (0.05, 0.10, 0.20, 0.50):
                k = int(np.ceil((n_res + 1) * (1 - alpha))) - 1
                k = max(0, min(k, n_res - 1))
                conformal_q[alpha] = float(sorted_res[k])
            log.info(
                f"  [phase12] conformal PI half-widths (in-sample OOF, n={n_res}): "
                f"95%={conformal_q[0.05]:.3f}  90%={conformal_q[0.10]:.3f}  "
                f"80%={conformal_q[0.20]:.3f}  50%={conformal_q[0.50]:.3f}"
            )
    except Exception as e:
        log.warning(f"  [phase12] conformal PI skipped: {e}")

    # ── Best model: rolling-origin 1-step-ahead forecast ────────────────
    best_pred = None
    _real_eval_champion_decay: dict = {}   # G-325: champion multi-horizon decay (summary 로 전파)
    if _opt_real_pred is not None:
        # G-306: R9 (per_model_optimize) already produced the OPTIMIZED champion's real-slab
        # rolling-origin prediction — reuse it (no duplicate default-HP refit).
        best_pred = _opt_real_pred
        log.info(f"  [phase12] best_pred = R9 per_model_optimize optimized real prediction for "
                 f"'{best_name}' ({len(best_pred)} wk) — default-HP rolling skipped")
    elif best_name is not None:
        log.info(f"  [phase12] rolling-origin 1-step-ahead forecast for '{best_name}'...")
        try:
            from simulation.models.base import REGISTRY
            # Auto-register every model module — they self-register on import
            for _m in ("epi_models", "dl_models", "tree_models", "linear_models",
                       "negbin_glm", "graph_models", "phase_ensemble",
                       "conformal", "cqr_models", "bayesian_seir",
                       "seir_forced", "pinn_model"):
                try:
                    __import__(f"simulation.models.{_m}")
                except Exception as _ie:
                    log.debug(f"  [phase12] could not import {_m}: {_ie}")

            def _factory():
                # Fresh per-step instance. Falls back to None if not registered.
                inst = REGISTRY.instantiate(best_name)
                if inst is None:
                    raise RuntimeError(f"model {best_name!r} not in REGISTRY")
                # Wrap as a thin shim so _rolling_origin_forecast can call
                # .fit(X, y) and .predict(X, model_name=name) uniformly.
                class _Shim:
                    def __init__(self, m): self._m = m
                    def fit(self, X, y, **kw):
                        self._m.fit(X, y); return self
                    def predict(self, X, model_name=None, **kw):
                        return np.asarray(self._m.predict(X), dtype=np.float64)
                return _Shim(inst)
            best_pred = _rolling_origin_forecast(
                _factory, best_name, X_in, y_in, real_X, real_y,
                feature_cols=feature_cols,
            )
            # G-325 (사용자 #3 + 3-AI): champion 의 multi-horizon decay table — 1-step 평가가
            #   multi-week 일반화를 과대평가하지 않는지 가시화(test slab=X_in 마지막 ~68주, recursive
            #   leakage-free). 진단용 — champion 선정엔 미사용(선정은 1-step OOF-shortlist→hold-out 유지).
            try:
                _n_dec = int(min(68, max(8, len(y_in) // 3)))
                if len(y_in) > _n_dec + 20:
                    _mh = _rolling_origin_multihorizon(
                        _factory, best_name, X_in[:-_n_dec], y_in[:-_n_dec],
                        X_in[-_n_dec:], y_in[-_n_dec:], horizons=(1, 2, 3, 4),
                        feature_cols=feature_cols)
                    _yt_d = np.asarray(y_in[-_n_dec:], dtype=np.float64)
                    _yin_d = np.asarray(y_in[:-_n_dec], dtype=np.float64)
                    from simulation.pipeline.phase_evaluator import evaluate_predictions_full
                    for _h in (1, 2, 3, 4):
                        _p = _mh.get(_h)
                        if _p is None:
                            continue
                        # 정렬: out[h][t] = origin t 의 h-step 후(t+h-1 주) 예측 → y_test[t+h-1] 과 짝
                        _pred_h, _true_h = [], []
                        for _t in range(len(_yt_d)):
                            _tgt = _t + _h - 1
                            if _tgt < len(_yt_d) and np.isfinite(_p[_t]):
                                _pred_h.append(float(_p[_t]))
                                _true_h.append(float(_yt_d[_tgt]))
                        if len(_pred_h) < 3:
                            continue
                        _pred_h = np.asarray(_pred_h, dtype=np.float64)
                        _true_h = np.asarray(_true_h, dtype=np.float64)
                        _sig_h = float(np.std(_true_h - _pred_h)) or 1.0   # horizon h 잔차 std → WIS/PI 폭
                        try:    # G-168 SSOT: 129/134-metric 전부 (MAE 하나가 아님 — 사용자 정정)
                            _mfull = evaluate_predictions_full(
                                _true_h, _pred_h, sigma=_sig_h, y_train_pool=_yin_d,
                                phase_id=f"champion_horizon_h{_h}")
                        except Exception as _me:  # noqa: BLE001
                            log.debug(f"  [phase12] h{_h} full-metric 실패: {_me}")
                            _mfull = {}
                        _real_eval_champion_decay[_h] = {
                            "metrics": _mfull,                  # 129/134 metric 전부
                            "predictions": _pred_h.tolist(),    # 예측값
                            "y_true": _true_h.tolist(),         # test값
                            "n": int(len(_pred_h)), "sigma": _sig_h,
                        }
                    if _real_eval_champion_decay:
                        def _r2h(_h):
                            return float((_real_eval_champion_decay.get(_h, {}).get("metrics", {})
                                          or {}).get("r2", float("nan")))
                        def _wish(_h):
                            return float((_real_eval_champion_decay.get(_h, {}).get("metrics", {})
                                          or {}).get("wis", float("nan")))
                        log.info("  [phase12] ★ champion '%s' multi-horizon decay (G-325 recursive, "
                                 "leakage-free, 129-metric/horizon — test값·예측값·전체지표 보존):" % best_name)
                        for _h in (1, 2, 3, 4):
                            if _h in _real_eval_champion_decay:
                                log.info(f"      h={_h}: R²={_r2h(_h):+.3f}  WIS={_wish(_h):.3f}  "
                                         f"(n={_real_eval_champion_decay[_h]['n']})")
            except Exception as _de:  # noqa: BLE001
                log.warning(f"  [phase12] horizon-decay 계산 실패: {type(_de).__name__}: {_de}")
        except Exception as e:
            log.error(f"  [phase12] rolling forecast failed: {type(e).__name__}: {e}")

    # ── S2-A: Adaptive Conformal Inference if requested ─────────────────
    # Computed before metrics loop so we can pass aci_results into PI eval.
    conformal_method = str(getattr(config.split, "real_conformal_method", "split"))
    aci_results: dict = {}
    if conformal_method in ("aci", "agaci") and best_pred is not None:
        try:
            from simulation.analytics.conformal_aci import (
                AdaptiveConformal, AggregatedACI,
            )
            oof = all_results.get("wfcv", {}).get("oof_predictions", {})
            if best_name in oof:
                res = y_in - np.asarray(oof[best_name], dtype=np.float64)
                cls = (AggregatedACI if conformal_method == "agaci"
                       else AdaptiveConformal)
                aci_inst = cls(alpha_star=0.05,
                               gamma=float(getattr(config.split, "aci_gamma", 0.05))
                               ) if conformal_method == "aci" else cls(alpha_star=0.05)
                aci_inst.calibrate(res[np.isfinite(res)])
                lo_arr, hi_arr = [], []
                for t, (yp, yo) in enumerate(zip(best_pred, real_y)):
                    lo, hi = aci_inst.predict_interval(float(yp))
                    lo_arr.append(lo); hi_arr.append(hi)
                    aci_inst.update(float(yo))
                aci_results = {
                    "lower": np.array(lo_arr),
                    "upper": np.array(hi_arr),
                    "method": f"{conformal_method.upper()} (Gibbs&Candès 2021"
                              + (" / Zaffran 2022" if conformal_method=="agaci" else "")
                              + ")",
                    "realized_coverage": (aci_inst.realized_coverage
                                           if hasattr(aci_inst, "realized_coverage") else None),
                }
                log.info(
                    f"  [phase12] {aci_results['method']} realized "
                    f"coverage = {aci_results.get('realized_coverage', float('nan'))}"
                )
        except Exception as e:
            log.warning(f"  [phase12] ACI/AgACI failed ({e}) — using split-conformal")

    # ── Resolve season-specific KDCA threshold from real-slab dates ─────
    if real_dates is not None and len(real_dates) > 0:
        slab_threshold = _kdca_threshold_for(real_dates[0])
    else:
        slab_threshold = KDCA_ALERT_THRESHOLD_DEFAULT
    log.info(f"  [phase12] KDCA threshold for this slab: {slab_threshold} per 1,000")

    # ── S1-4: pull ensemble forecast if the scoring phase (R8) produced one ────────────
    candidates: dict[str, np.ndarray] = {
        "persistence":      base_persist,
        "seasonal_naive":   base_seasonal,
        "ar1":              base_ar1,
    }
    if base_hhh4 is not None and np.isfinite(base_hhh4).any():
        candidates["hhh4_equivalent"] = base_hhh4
    if best_pred is not None and np.isfinite(best_pred).any():
        candidates[best_name] = best_pred

    # ── S2-C: Stacking-on-CRPS ensemble (built from in-sample OOF) ─────
    ensemble_method = str(getattr(config.split, "ensemble_method", "stacking"))
    if ensemble_method in ("stacking", "median") and len(candidates) >= 3:
        try:
            from simulation.ensembles.stacking_crps import (
                stacking_weights_crps, predict_with_stacking,
                equally_weighted_median_ensemble,
            )
            oof = all_results.get("wfcv", {}).get("oof_predictions", {}) or {}
            stack_inputs = {}
            for nm in candidates:
                if nm in oof:
                    stack_inputs[nm] = np.asarray(oof[nm], dtype=np.float64)
            if ensemble_method == "median":
                med = equally_weighted_median_ensemble(candidates)
                if med is not None:
                    candidates["median_ensemble"] = med
                    log.info("  [phase12] median ensemble (Sherratt 2023)")
            elif len(stack_inputs) >= 2:
                stk = stacking_weights_crps(stack_inputs, y_in)
                applicable = {k: v for k, v in candidates.items()
                              if k in stk["weights"]}
                if applicable:
                    candidates["stacking_CRPS"] = predict_with_stacking(stk["weights"], applicable)
                    log.info(f"  [phase12] stacking-on-CRPS weights: {stk['weights']}")
            # A6 (M7): Caruana forward-stepwise (FluSight-standard ensemble builder)
            # as a first-class candidate — greedy OOF selection, robust to the
            # 53-model pool. Caruana already exists (ensembles/caruana.py) but was
            # never wired into the real-slab candidates.
            if len(stack_inputs) >= 2:
                from simulation.ensembles import caruana_forward_stepwise
                car = caruana_forward_stepwise(stack_inputs, y_in, n_steps=25,
                                               random_state=42)
                applicable_c = {k: v for k, v in candidates.items()
                                if k in car.model_weights}
                if applicable_c:
                    candidates["caruana_ensemble"] = predict_with_stacking(
                        car.model_weights, applicable_c)
                    log.info(f"  [phase12] Caruana ensemble ({len(car.model_weights)} members)")
        except Exception as e:
            log.warning(f"  [phase12] ensemble combination failed: {e}")
    # Ensemble (NNLS / BMA) if available — FluSight expects ensemble
    # alongside individual models.
    try:
        ens_results = all_results.get("scoring", {}).get("ensemble", {}) \
                      or all_results.get("ensemble", {})
        ens_pred = ens_results.get("real_predictions") if isinstance(ens_results, dict) else None
        if ens_pred is not None and len(ens_pred) == n_real:
            candidates["ensemble_NNLS"] = np.asarray(ens_pred, dtype=np.float64)
            log.info("  [phase12] ensemble (NNLS) wired into real-slab eval")
    except Exception:
        pass

    # ── A1 (M7): designate the DEPLOYMENT forecast separately from the
    # retrospective leaderboard. The champion's raw forecast stays in
    # `candidates` for honest evaluation, but what flows downstream (ABM/ARIA)
    # is the GATED forecast: a champion that violates the deployment contract
    # (extrapolation collapse) is REPLACED by the stable median ensemble.
    deployment: dict = {"model": best_name, "replaced": False, "reason": "ok"}
    if best_name in candidates and best_pred is not None:
        _fallback = candidates.get("median_ensemble", candidates.get("seasonal_naive"))
        _g = _gate_forecast(candidates[best_name], y_in, _fallback, k=3.0)
        deployment = {
            "model": ("median_ensemble" if _g["replaced"] else best_name),
            "champion": best_name,
            "replaced": _g["replaced"],
            "n_violations": _g["n_violations"],
            "reason": _g["reason"],
            "forecast": np.asarray(_g["pred"], dtype=np.float64).tolist(),
        }
        if _g["replaced"]:
            log.warning(
                f"  [phase12] ⚠ DEPLOYMENT GATE: champion '{best_name}' forecast "
                f"REJECTED ({_g['reason']}) → deploying stable fallback "
                f"'{deployment['model']}' to ABM/ARIA (champion kept for eval only)"
            )
        else:
            log.info(f"  [phase12] deployment gate: champion '{best_name}' passes "
                     f"contract — deploying champion")

    # pre-compute ABM eval (사용자 2026-06-07: "abm까지 사전에 다 했으면"). Gated by
    # MPH_PRECOMPUTE_ABM=1 to respect cost + the 2-engine separation; when on, the
    # anchored ABM for the DEPLOYMENT champion is computed here and saved so the
    # downstream ABM/ARIA basis is ready without a separate `sim` run. Lazy-import
    # keeps real_eval decoupled from the ABM engine by default.
    if os.environ.get("MPH_PRECOMPUTE_ABM") == "1" and deployment.get("forecast"):
        try:
            from simulation.abm.forecast_anchor import anchor_abm_to_forecast
            from simulation.utils.paths import get_results_dir
            _agents = int(os.environ.get("MPH_PRECOMPUTE_ABM_AGENTS", "20000"))
            _abm = anchor_abm_to_forecast(
                np.asarray(deployment["forecast"], dtype=np.float64),
                n_agents=_agents, seeds=[0, 1])
            _p = get_results_dir() / "abm_precomputed.json"
            _p.write_text(json.dumps({
                "deployment_model": deployment["model"],
                "champion": deployment.get("champion"),
                "wis": _abm["wis"], "corr_sim_vs_forecast": _abm["corr_sim_vs_forecast"],
                "degenerate": _abm["degenerate"], "fitted_forcing": _abm["fitted_forcing"],
            }, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
            log.info(f"  [phase12] ABM pre-computed (deployment '{deployment['model']}', "
                     f"{_agents} agents) → {_p}")
        except Exception as _e:   # pre-compute is best-effort; never fail real_eval
            log.warning(f"  [phase12] ABM pre-compute skipped: {type(_e).__name__}: {_e}")

    results: dict[str, dict] = {}
    for nm, pred in candidates.items():
        # ── S0-4: per-baseline σ for CRPS/WIS/PIT (naive baselines have
        #    different error variance than the best model)
        if nm in ("persistence", "seasonal_naive", "ar1"):
            sigma_nm = _baseline_sigma(nm, y_in)
        else:
            sigma_nm = sigma_in  # best/ensemble use in-sample OOF σ

        # ── S1-2: PI from per-α conformal quantile (not max(...))
        lo = up = None
        pi_levels_dict = None
        if nm in (best_name, "ensemble_NNLS") and conformal_q:
            # 95% PI half-width — explicitly use α=0.05 (the right one)
            q95 = conformal_q.get(0.05, conformal_q.get(0.10))
            lo = pred - q95
            up = pred + q95
            # Multi-level table (Bracher 2021 K=4)
            pi_levels_dict = {
                1 - alpha: (pred - conformal_q[alpha], pred + conformal_q[alpha])
                for alpha in conformal_q
            }
        # Use persistence as DM baseline reference (universal yardstick)
        ref = candidates["persistence"] if nm != "persistence" else None
        results[nm] = _evaluate_model(
            nm, real_y, pred,
            sigma=sigma_nm,
            pi_lower=lo, pi_upper=up,
            pi_levels=pi_levels_dict,
            alert_threshold=slab_threshold,    # S0-1: season-specific
            baseline_pred=ref,
            y_train_pool=y_in,                 # G-326: full-129 SSOT merge(MASE 등) 용
        )
        m = results[nm]
        log.info(
            f"  [phase12] {nm:18s} MAE={m.get('mae', float('nan')):.3f}  "
            f"RMSE={m.get('rmse', float('nan')):.3f}  R²={m.get('r2', float('nan')):.3f}  "
            f"WIS={m.get('wis', float('nan')):.3f}  alertF1={m.get('alert_f1', float('nan')):.3f}"
        )

        # R8.2 (2026-05-26): full 134-key SSOT eval on real-slab predictions.
        # Trajectory: R4 (WF-CV) OOF → P1 (real_forecaster) real-slab → R12 (comprehensive_eval) final SSOT.
        # ljung_box_p / residual_acf_lag1 → real-slab regime calibration evidence.
        try:
            from simulation.pipeline.phase_evaluator import evaluate_predictions_full
            mask = np.isfinite(real_y) & np.isfinite(pred)
            if mask.sum() >= 5:
                full_r8 = evaluate_predictions_full(
                    y_test=np.asarray(real_y, dtype=np.float64)[mask],
                    y_pred=np.asarray(pred, dtype=np.float64)[mask],
                    residuals=(np.asarray(real_y, dtype=np.float64)[mask]
                               - np.asarray(pred, dtype=np.float64)[mask]),
                    sigma=sigma_nm,
                    y_train_pool=y_in,
                    threshold=float(slab_threshold),
                    phase_id=f"phase10_real_{nm}",
                    enable_bootstrap_ci=False,
                )
                results[nm]["phase_eval_r8"] = full_r8
        except Exception as _e:
            results[nm]["phase_eval_r8_err"] = str(_e)

    # ── S1-3: coverage_gap_by_regime (already-existing in diagnostics) ──
    try:
        from simulation.analytics.diagnostics import coverage_gap_by_regime
        # Wire only for best/ensemble where we have PI bounds
        for nm in (best_name, "ensemble_NNLS"):
            if nm not in candidates or conformal_q is None or not conformal_q:
                continue
            pred = candidates[nm]
            q = conformal_q.get(0.05, conformal_q.get(0.10))
            try:
                regime = coverage_gap_by_regime(
                    real_y,
                    pred - q, pred + q,
                    real_dates if real_dates is not None else None,
                    nominal=0.95,
                )
                results[nm]["regime_coverage"] = regime
            except Exception as _re:
                log.debug(f"  [phase12] coverage_gap_by_regime({nm}): {_re}")
    except ImportError:
        pass

    # ── Persist all artifacts ────────────────────────────────────────────
    out_dir = Path(getattr(config, "save_dir", "simulation/results")) / "real_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # G-326 (2026-06-19, 사용자: horizon별 예측값·test값 CSV 완성): champion multi-horizon decay → CSV 2종.
    #   ① predictions: (horizon, week_idx, prediction, y_true) — 사용자 명시 요구.
    #   ② metrics: (horizon, n, sigma, + 129 metric 전부 — G-168 SSOT).
    try:
        if _real_eval_champion_decay:
            import csv as _csv
            _pred_path = out_dir / "champion_horizon_decay_predictions.csv"
            with open(_pred_path, "w", encoding="utf-8", newline="") as _pf:
                _w = _csv.writer(_pf)
                _w.writerow(["horizon", "week_idx", "prediction", "y_true"])
                for _h in sorted(_real_eval_champion_decay):
                    _d = _real_eval_champion_decay[_h]
                    for _i, (_p, _t) in enumerate(zip(_d.get("predictions", []), _d.get("y_true", []))):
                        _w.writerow([_h, _i, _p, _t])
            _met_path = out_dir / "champion_horizon_decay_metrics.csv"
            _all_keys = sorted({_k for _h in _real_eval_champion_decay
                                for _k in (_real_eval_champion_decay[_h].get("metrics") or {})})
            with open(_met_path, "w", encoding="utf-8", newline="") as _mf:
                _w = _csv.writer(_mf)
                _w.writerow(["horizon", "n", "sigma"] + _all_keys)
                for _h in sorted(_real_eval_champion_decay):
                    _d = _real_eval_champion_decay[_h]
                    _m = _d.get("metrics") or {}
                    _w.writerow([_h, _d.get("n"), _d.get("sigma")] + [_m.get(_k) for _k in _all_keys])
            log.info(f"  [phase12] ★ champion horizon-decay CSV: {_pred_path.name}(예측·test/horizon) "
                     f"+ {_met_path.name}({len(_all_keys)} metric/horizon)")
    except Exception as _ce:   # noqa: BLE001
        log.warning(f"  [phase12] horizon-decay CSV 작성 실패: {type(_ce).__name__}: {_ce}")

    # 2026-04-28 Bug fix: best_wis was None — extract from metrics dict.
    # 2026-05-30 (G-237): bare `metrics` was undefined (module is `import metrics
    # as M`; the per-model accumulator is `results`) → NameError silently voided
    # the champion gate for a 10h run. Use `results` (cf. identical pattern :1312).
    _best_metrics = results.get(best_name, {}) if best_name else {}
    summary = {
        "best_model": best_name,
        # G-325 (사용자 #3 + 3-AI): champion 의 multi-horizon decay {h: {metrics(129), predictions,
        #   y_true, n, sigma}} (recursive, leakage-free). 1-step 선정이 multi-week 일반화를 과대평가
        #   하는지 가시화 — G-168 SSOT 대로 horizon 별 test값·예측값·129 metric 전부 보존(진단; 선정 무관).
        "champion_horizon_decay": (_real_eval_champion_decay or None),
        # A1 (M7): the robust DEPLOYMENT forecast (gated champion → stable
        # fallback on contract violation). ABM/ARIA consume this, NOT best_model's
        # raw forecast — separates the retrospective champion from what we deploy.
        "deployment": deployment,
        "best_wis":  _best_metrics.get("wis"),                    # ← 추가
        "best_r2":   _best_metrics.get("r2"),                     # ← 추가
        "best_mae":  _best_metrics.get("mae"),                    # ← 추가
        "best_rmse": _best_metrics.get("rmse"),                   # ← 추가
        "best_crps": _best_metrics.get("crps_gaussian"),          # ← 추가
        "in_sample_n": int(n_in),
        "real_n": int(n_real),
        "real_dates_start": str(real_dates[0]) if real_dates is not None else None,
        "real_dates_end":   str(real_dates[-1]) if real_dates is not None else None,
        "weather_mode": weather_mode,
        "alert_threshold": float(slab_threshold),
        "alert_threshold_source": (
            f"KDCA {_season_for(real_dates[0]) if real_dates is not None else '?'}-"
            f"{(_season_for(real_dates[0]) + 1) if (real_dates is not None and _season_for(real_dates[0]) is not None) else '?'}"
            f" 절기 유행기준 (KDCA_THRESHOLD_BY_SEASON lookup)"
        ),
        "sigma_in_sample": float(sigma_in),
        "conformal_pi_widths": {f"alpha={a}": q for a, q in conformal_q.items()},
        "metrics": {nm: {k: v for k, v in m.items()
                          if not isinstance(v, (np.ndarray, list, dict))
                          or k in ("mae_ci95", "peak_week", "peak_intensity",
                                   "regime_coverage", "pi_calibration")}
                    for nm, m in results.items()},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (out_dir / "metrics_full.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    # G-169 (2026-05-03): per-model real prediction 디렉토리 저장.
    # plot_forecast_full.py 의 G-164 fix 가 우선 읽는 source.
    # 형식: {predictions: [...], r2, mae, rmse, wis, mape, ...} (model 별 JSON)
    per_model_dir = out_dir / "per_model"
    per_model_dir.mkdir(parents=True, exist_ok=True)
    # G-169 orphan fix: wipe stale *.json first so a prior run's model set
    # (e.g. ElasticNet.json from a different run) can't be mis-read as this
    # run's real_pred by plot_forecast_full (G-164 consumer).
    for _stale in per_model_dir.glob("*.json"):
        try:
            _stale.unlink()
        except OSError:
            pass
    n_saved = 0
    for nm, m in results.items():
        if not isinstance(m, dict):
            continue
        try:
            # JSON-safe filter (numpy scalar → float, list/dict 유지)
            entry = {k: (float(v) if isinstance(v, (np.integer, np.floating)) else v)
                     for k, v in m.items()
                     if not isinstance(v, np.ndarray)}
            # numpy array → list 변환 (predictions 등)
            for k, v in m.items():
                if isinstance(v, np.ndarray):
                    entry[k] = v.tolist()
            (per_model_dir / f"{nm}.json").write_text(
                json.dumps(entry, indent=2, default=str), encoding="utf-8",
            )
            n_saved += 1
        except Exception as _pe:
            log.warning(f"  [phase12] per_model save {nm} fail: {_pe}")
    log.info(f"  [phase12] per_model/ saved: {n_saved} models "
             f"(G-164 plot_forecast_full source)")

    # CSV: date, y_true, plus all model predictions
    import csv
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["date", "y_true"] + list(candidates.keys())
        w.writerow(header)
        for i in range(n_real):
            row = [str(real_dates[i]) if real_dates is not None else i,
                   float(real_y[i])]
            row.extend(float(candidates[nm][i]) if np.isfinite(candidates[nm][i])
                       else "" for nm in candidates)
            w.writerow(row)

    # ── Markdown report ───────────────────────────────────────────────
    threshold_season = (_season_for(real_dates[0])
                        if real_dates is not None else None)
    threshold_season_label = (
        f"{threshold_season}-{threshold_season + 1}"
        if threshold_season is not None else "?-?"
    )
    md = [
        "# P1 real_forecaster — Real Forecast Evaluation (HWP §3)",
        "",
        f"- **Best in-sample model**: `{best_name}`  ",
        f"- **In-sample n**: {n_in}  (HWP analysis period 2019–2025)",
        f"- **Real slab n**: {n_real}  "
        f"({real_dates[0] if real_dates is not None else '?'} → "
        f"{real_dates[-1] if real_dates is not None else '?'})",
        f"- **σ (in-sample OOF, best model)**: {sigma_in:.4f}",
        f"- **KDCA alert threshold (season-specific)**: {slab_threshold} per 1,000  "
        f"_(KDCA {threshold_season_label} 절기 유행기준)_",
        "",
        "## Forecast strategy",
        "**Rolling-origin 1-step-ahead refit**: each real week refits on "
        "(in-sample ⊕ already-revealed real) and predicts t+1. Per-step "
        "leakage-prone features (`pop_inflow_max` etc.) are recomputed "
        "using only data revealed up to that step, via "
        "`_recode_quantile_features_per_fold` / `_recode_above_threshold` / "
        "`_recode_interaction_features`.",
        "",
        "### Deterministic future-knowable features (35 of " + str(len(feature_cols)) + ")",
        "These categories are perfectly knowable at any forecast time and "
        "are used WITHOUT substitution:",
        "- **Calendar (19)**: `sin_p52`, `cos_p52`, `sin_p26/p13/p6_5`, "
        "`sin_month/cos_month`, `season_idx/norm`, `mr_month_*`, "
        "`mr_prev_season_mean`, `season_cum_ili` — perfect future info.",
        "- **KMA forecast (10)**: `fcst_tmp/reh/pcp/pty/pop/sky/wsd`, "
        "`rt_fcst_*` — KMA short-range forecast data, available in advance.",
        "- **Climatology (6)**: `ili_rate_rmean4/8/13/26`, `temp_avg_qnorm`, "
        "`ili_rate_lag1_qnorm` — historical week-of-year means, available.",
        "",
        f"### Weather-handling mode: **{weather_mode}**",
        ({
            "observed":    "Observed weather columns (`temp_avg`, `humidity`, "
                           "`rainfall`, etc.) are left at their **actual real-slab values**. "
                           "Convenient but assumes perfect-foresight weather — "
                           "**not realistic for live deployment**.",
            "climatology": "Observed weather columns are **replaced by week-of-year "
                           "means** computed from in-sample data only. No foresight. "
                           "Conservative — represents 'no weather information' deployment.",
            "hybrid":      "KMA `fcst_*` columns retained (they already carry forecast "
                           "data); other observed-weather columns replaced with "
                           "climatology. **Closest to live-deployment performance**.",
        }).get(weather_mode, "(unknown mode)"),
        "",
        "## ⚠️ Statistical caveats (n = " + str(n_real) + ")",
        f"- Real slab is **{n_real} weeks** — most metrics are descriptive,",
        "  not inferential. Specifically:",
        "  - Bootstrap CIs on n=8 use BCa but only ~8! distinct resamples.",
        "  - Diebold-Mariano with t-distribution df=" + str(max(0, n_real-2)) +
        " has very low power.",
        "  - `peak_week_error` is meaningful only if the slab spans an actual peak.",
        f"  - `alert_F1`: real-slab prevalence = {sum(real_y > slab_threshold)/n_real:.0%} "
        f"above {slab_threshold} → if = 100%, F1 collapses to trivially 1.",
        "",
        "## Section A — Epi-hub metrics (CDC FluSight / Bracher 2021 / RespiCast standard)",
        "",
        "| model | n | WIS | CRPS | 95% cov | 95% width | PIT μ | peak-wk | peak-int | alert-F1 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for nm, m in results.items():
        pw = m.get("peak_week", {}) or {}
        pi = m.get("peak_intensity", {}) or {}
        md.append(
            f"| {nm} | {m.get('n_valid', 0)} | "
            f"{m.get('wis', float('nan')):.3f} | "
            f"{m.get('crps_gaussian', float('nan')):.3f} | "
            f"{m.get('pi95_coverage', float('nan')):.3f} | "
            f"{m.get('pi95_width', float('nan')):.3f} | "
            f"{m.get('pit_mean', float('nan')):.2f} | "
            f"{pw.get('abs_weeks', float('nan'))} | "
            f"{pi.get('rel_err', float('nan')):.3f} | "
            f"{m.get('alert_f1', float('nan')):.3f} |"
        )
    md += [
        "",
        "## Section B — Point-forecast diagnostics (ML convention)",
        "",
        "_Note: hubs (FluSight, RespiCast) report MAE only; R²/MSE/RMSE/sMAPE_",
        "_are ML-side diagnostics not standardised in epi-forecast literature._",
        "",
        "| model | MAE | MAE 95% CI (BCa) | RMSE | R² | MAPE % | sMAPE % | dir-acc |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for nm, m in results.items():
        ci_lo, ci_hi = m.get("mae_ci95", (float("nan"), float("nan")))
        md.append(
            f"| {nm} | "
            f"{m.get('mae', float('nan')):.3f} | "
            f"({ci_lo:.3f}, {ci_hi:.3f}) | "
            f"{m.get('rmse', float('nan')):.3f} | "
            f"{m.get('r2', float('nan')):.3f} | "
            f"{m.get('mape', float('nan')):.2f} | "
            f"{m.get('smape', float('nan')):.2f} | "
            f"{m.get('direction_accuracy', float('nan')):.3f} |"
        )
    md += [
        "",
        "## Section C — Clinical / alert diagnostics",
        "",
        f"_Threshold = {slab_threshold} per 1,000 outpatient consultations "
        f"(KDCA {threshold_season_label} 절기)._",
        "_Brier probability uses Gaussian tail P(Y>τ|μ̂,σ̂) — not magnitude ratio._",
        "",
        "| model | Brier | Brier skill | sens | spec | PPV | NPV | F1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for nm, m in results.items():
        md.append(
            f"| {nm} | "
            f"{m.get('brier_score', float('nan')):.3f} | "
            f"{m.get('brier_skill', float('nan')):.3f} | "
            f"{m.get('sensitivity', float('nan')):.3f} | "
            f"{m.get('specificity', float('nan')):.3f} | "
            f"{m.get('ppv', float('nan')):.3f} | "
            f"{m.get('npv', float('nan')):.3f} | "
            f"{m.get('clinical_f1', float('nan')):.3f} |"
        )
    md += [
        "",
        "## Section D — Statistical comparison vs persistence baseline",
        "",
        "_DM and McNemar are methodological extensions; both Sherratt 2023 and_",
        "_FluSight explicitly do NOT use them as primary forecast metrics._",
        "_p-values at n=8 should be treated as exploratory only._",
        "",
        "| model | DM stat | DM p | McNemar stat | McNemar p |",
        "|---|---|---|---|---|",
    ]
    for nm, m in results.items():
        if nm == "persistence":
            continue
        md.append(
            f"| {nm} | "
            f"{m.get('dm_stat', float('nan')):.3f} | "
            f"{m.get('dm_pval', float('nan')):.4f} | "
            f"{m.get('mcnemar_stat', float('nan')):.3f} | "
            f"{m.get('mcnemar_pval', float('nan')):.4f} |"
        )
    md += [
        "",
        "## Provenance",
        f"- Real slab carved at phase1_data.py from idx {n_in} (paper_cutoff_week)",
        "  forward — in-sample 학습/WF-CV/test phase never see these rows (real_eval only).",
        "- σ for best/ensemble: in-sample OOF residual std.",
        "- σ per naive baseline: that baseline's own in-sample residual std.",
        "- Conformal PI: split-conformal ceiling-quantile from in-sample OOF",
        "  residuals (Lei et al. 2018 JASA 113:1094 / Vovk 2005).",
        "- Per-fold leakage recoder applied at each rolling step.",
        "- KDCA threshold: season-aware lookup (`KDCA_THRESHOLD_BY_SEASON`).",
        "",
        "## External-standard alignment",
        "- Bracher 2021 (PLOS Comp Bio): WIS, PIT, K-level coverage ✓",
        "- CDC FluSight 2024-25: WIS / 50-95% PI / peak metrics ✓",
        "  (FluSight uses 23 quantiles vs our 4; defensible for thesis.)",
        "- RespiCast / Sherratt 2023: pairwise relative WIS — TODO for future work.",
        "- KDCA 2025-26 유행주의보 (2025-10-17): threshold = 9.1 per 1,000 ✓",
    ]
    report_path = out_dir / "report.md"
    report_path.write_text("\n".join(md), encoding="utf-8")
    log.info(f"  [phase12] wrote: {out_dir}")

    # 2026-04-28: best_wis 도 함께 반환 (이전엔 None)
    _best_m_ret = results.get(best_name, {}) if best_name else {}
    return {
        "metrics": results,
        "best_model": best_name,
        "best_wis": _best_m_ret.get("wis"),
        "best_r2":  _best_m_ret.get("r2"),
        "n_real": n_real,
        "alert_threshold": KDCA_ALERT_THRESHOLD,
        "sigma_in_sample": sigma_in,
        "summary_path": str(out_dir / "summary.json"),
        "metrics_full_path": str(out_dir / "metrics_full.json"),
        "predictions_path": str(out_dir / "predictions.csv"),
        "report_path": str(report_path),
        "elapsed": time.time() - t0,
    }


# back-compat aliases (2026-06-02 semantic rename — 옛 run_phaseN)
run_phase12 = run_real_eval
