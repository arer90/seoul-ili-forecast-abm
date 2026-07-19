# Feature Engine Causality Audit (, S1-1 full pass)

**Date**: 2026-04-17
**Scope**: `simulation/models/feature_engine/transforms.py` + call-site in `builder.py`
**Target convention**: `ili_rate[t]` (nowcasting). `phase1_data.py:135` checks
`corr(X[:, i][t], y[t])` at the same index — i.e. the model learns `X[t] -> y[t]`,
not `X[t] -> y[t+1]`. All causality statements below assume this convention.

---

## 1. Transform-level verdicts

| # | Function | Verdict | Mechanism |
|---|----------|---------|-----------|
| 1 | `_add_lag_features` | **Causal** | `pl.col(col).shift(lag)` — values at row t equal col[t-lag]. |
| 2 | `_add_rolling_features` | **Causal** | `rolling_{mean,std,min,max}(w).shift(1)` — row t = stat over `[t-w, t-1]`. |
| 3 | `_add_diff_features` | **Causal** | `diff(n=d).shift(1)` — row t = col[t-1] − col[t-1-d]. |
| 4 | `_add_log_features` | **Causal** (per-row) | Element-wise `log1p`; no temporal context. |
| 5 | `_add_quantile_encoding` | Build-time global → **fold-recoded** | Bins come from `df[col].slice(0, train_end)` — global `train_end` at build time leaks fold-future distribution into early folds. `phase7_wfcv._recode_quantile_features_per_fold` rebuilds `*_qbin` / `*_qnorm` per fold using only `X_all[:train_end]`. |
| 6 | `_add_binary_encoding` | **Causal** (per-row) | Bit-level encoding of the (already-lagged) column; no temporal context. |
| 7 | `_add_multi_resolution_seasonal` | **Causal** (deterministic) | Sin/cos of `week_seq` and `month` — calendar index only. |
| 8 | `_add_wavelet_features` | **Causal** (G-091 fix) | `np.convolve(mode="full")` keeps past-only support; result sliced to `[:n]`, then `np.roll(1)` with `shifted[0]=0`. Applied to `ili_rate_lag1`, so feature at t uses y[t-2] and earlier. |
| 9 | `_add_interaction_features` | **Causal w/ caveat** (§2) | Multiplies `ili_rate_lag1[t]` (strict past) by `exogenous[t]` (contemporaneous). Legitimate under nowcasting. Global `max()` normalization is a minor documented leak (§2.B). |
| 10 | `_add_epidemic_phase_features` | Build-time global → **fold-recoded** (above_threshold) + **causal** (consec_rise, season_cum_ili) | `threshold = 2 · median(ili[:int(n·0.8)])` baked at build time leaks future distribution. `phase7_wfcv._recode_above_threshold_per_fold` rebuilds `above_threshold` per fold using `median(y[:train_end])`. `consec_rise` and `season_cum_ili` are `np.roll(·, 1)` at build time — already strictly past, no recode needed. |
| 11 | `_add_multi_resolution_agg` | **Causal** | All aggregates use half-open ranges ending at `i` (strict past) and/or explicit `np.roll(·, 1)`. Per-sub-feature verdicts in §3. |

---

## 2. Outstanding caveats

### 2.A `_add_interaction_features` — contemporaneous exogenous × lag-1 target

The block at `transforms.py:161-243` forms products like
`subway_ili[t] = subway_total_avg[t] · ili_rate_lag1[t]`. Because the target
is `ili_rate[t]` (nowcasting), using `subway_total_avg[t]` is defensible —
subway traffic for week t is observable by end of week t, the same observability
horizon as `ili_rate[t]` itself.

**Nowcasting** (current pipeline): legitimate.
**One-step-ahead forecasting** (not currently used): would require shifting
`ili_rate` forward or shifting all contemporaneous exogenous back by one step
before interacting.

### 2.B Global `max()` normalization inside `_add_interaction_features`

Lines 172, 180, 185, 195, 204, 209, 218, 225, 232, 239 use
`df[col].max()` (global) as the normalization denominator. This leaks a
single scalar summary of the entire series into every fold.

**Status (, 2026-04-19): RESOLVED** — `phase7_wfcv._recode_interaction_features_per_fold`
rebuilds 9 of the 10 interaction columns (`inflow_ili`, `subway_ili`,
`bus_ili`, `wp_inflow_ili`, `hs_congestion_ili`, `rt_subcrowd_ili`,
`rt_roadcong_ili`, `rt_nonresnt_ili`, `rt_highrisk_ili`) using
`max(src[:train_end]) + eps` as the denominator at each fold and at the
holdout refit. The 10th (`er_burden_ili`) uses a composite `1/x → max`
formula and is intentionally left with build-time values — it is
documented in the `excluded` set of
`test_interaction_specs_cover_every_builder_interaction` and should be
revisited if residual analysis flags it. Tests in
`simulation/tests/test__cv_splits.py::test_interaction_recode_*`
(5 tests) lock the behaviour.

### 2.C `_add_epidemic_phase_features.season_cum_ili`

```python
cum += ili[i] # adds CURRENT row
cumili.append(cum)
cumili_rolled = np.roll(cumili, 1) # then shifts forward by 1
cumili_rolled[0] = 0
```

After the `np.roll(·, 1)`, the value at row t equals `sum(ili[j])` for
`j ∈ season(t), j ≤ t-1` — i.e. strictly past. Causal ✓.

---

## 3. `_add_multi_resolution_agg` sub-feature breakdown

| Column | Window | Strict past? |
|--------|--------|--------------|
| `mr_month_avg/max/std` | `range(max(0, i-5), i)` + month match | ✓ (j < i) |
| `mr_quarter_avg/max/trend` | `ili[i-13:i]` | ✓ (Python half-open) |
| `mr_prev_season_mean` | `completed_season_means[prev_s]` — only populated when a season ends (`seasons[i] != seasons[i-1]`), mask `seasons[:i] == prev_s` | ✓ (uses slice `[:i]`) |
| `mr_season_ratio` | numerator `[ili[j] for j in range(i) if seasons[j] == s]`, denominator is `mr_prev_season_mean` | ✓ (`j < i`) |
| `mr_yoy_ratio` / `mr_yoy_diff` | `ili[i-1] / ili[i-52]`, `ili[i-1] - ili[i-52]` | ✓ |
| `mr_trend_26w` | `causal_ma26[i] = mean(ili[i-25:i+1])`, then `np.roll(·, 1)` with `[0] = np.nan` | ✓ (G-090 fix) |
| `mr_trend_26w_diff` | first difference of `mr_trend_26w` | ✓ |

---

## 4. Fold-wise recode coverage (phase7_wfcv)

```python
_QUANTILE_SPECS = (
 ("ili_rate_lag1", 10),
 ("temp_avg", 8),
)
_ABOVE_THRESHOLD_COL = "above_threshold"
```

**Invariant**: if the only features whose build-time values depend on
a train/future-distribution summary are `{*_qbin, *_qnorm, above_threshold}`,
and the fold-wise recode rewrites all of them using only `[:train_end]`, then
the WFCV loop is free of look-ahead leakage through this class of features.

The current `_QUANTILE_SPECS` is comprehensive: `builder.py:860-863` only
applies `_add_quantile_encoding` to `ili_rate_lag1` and `temp_avg`. If this
call site ever adds a new target (e.g. `humidity`, `subway_total_avg`), the
spec tuple must be updated in lock-step — otherwise early-fold folds silently
use global bins. A regression test for this would be valuable.

---

## 5. Summary

**Confirmed causal at build time** (no recode needed):
`_add_lag_features`, `_add_rolling_features`, `_add_diff_features`,
`_add_log_features`, `_add_binary_encoding`, `_add_multi_resolution_seasonal`,
`_add_wavelet_features`, `_add_multi_resolution_agg`,
`_add_epidemic_phase_features.{consec_rise, season_cum_ili}`.

**Needs fold-wise recode, and already has it**:
`_add_quantile_encoding` (all 4 output columns for both specs),
`_add_epidemic_phase_features.above_threshold`.

**Known minor leak, documented, not fixed**:
`_add_interaction_features` global `max()` normalization constants (§2.B).

**Convention caveat**:
`_add_interaction_features` is leak-free under nowcasting (`y = ili_rate[t]`).
Switching to h-step forecasting requires an explicit horizon shift.

**Next guardrail** (low effort, high value): add a unit test that asserts
`set(cols_with_train_end_dependent_stats) ⊆ set(fold_recoded_cols)` so that
adding a new quantile target in `builder.py` without updating
`_QUANTILE_SPECS` fails CI instead of silently leaking.
