# P1 real_forecaster — Real Forecast Evaluation (HWP §3)

- **Best in-sample model**: `None`  
- **In-sample n**: 337  (HWP analysis period 2019–2025)
- **Real slab n**: 17  (2026-02-22T00:00:00.000000 → 2026-06-14T00:00:00.000000)
- **σ (in-sample OOF, best model)**: 5.4907
- **KDCA alert threshold (season-specific)**: 9.1 per 1,000  _(KDCA 2025-2026 절기 유행기준)_

## Forecast strategy
**Rolling-origin 1-step-ahead refit**: each real week refits on (in-sample ⊕ already-revealed real) and predicts t+1. Per-step leakage-prone features (`pop_inflow_max` etc.) are recomputed using only data revealed up to that step, via `_recode_quantile_features_per_fold` / `_recode_above_threshold` / `_recode_interaction_features`.

### Deterministic future-knowable features (35 of 13)
These categories are perfectly knowable at any forecast time and are used WITHOUT substitution:
- **Calendar (19)**: `sin_p52`, `cos_p52`, `sin_p26/p13/p6_5`, `sin_month/cos_month`, `season_idx/norm`, `mr_month_*`, `mr_prev_season_mean`, `season_cum_ili` — perfect future info.
- **KMA forecast (10)**: `fcst_tmp/reh/pcp/pty/pop/sky/wsd`, `rt_fcst_*` — KMA short-range forecast data, available in advance.
- **Climatology (6)**: `ili_rate_rmean4/8/13/26`, `temp_avg_qnorm`, `ili_rate_lag1_qnorm` — historical week-of-year means, available.

### Weather-handling mode: **hybrid**
KMA `fcst_*` columns retained (they already carry forecast data); other observed-weather columns replaced with climatology. **Closest to live-deployment performance**.

## ⚠️ Statistical caveats (n = 17)
- Real slab is **17 weeks** — most metrics are descriptive,
  not inferential. Specifically:
  - Bootstrap CIs on n=8 use BCa but only ~8! distinct resamples.
  - Diebold-Mariano with t-distribution df=15 has very low power.
  - `peak_week_error` is meaningful only if the slab spans an actual peak.
  - `alert_F1`: real-slab prevalence = 53% above 9.1 → if = 100%, F1 collapses to trivially 1.

## Section A — Epi-hub metrics (CDC FluSight / Bracher 2021 / RespiCast standard)

| model | n | WIS | CRPS | 95% cov | 95% width | PIT μ | peak-wk | peak-int | alert-F1 |
|---|---|---|---|---|---|---|---|---|---|
| persistence | 17 | 2.703 | 2.774 | 1.000 | 47.371 | 0.37 | 0.0 | 0.820 | 0.842 |
| seasonal_naive | 17 | 5.938 | 6.210 | 1.000 | 36.429 | 0.46 | 6.0 | 0.036 | 0.857 |
| ar1 | 17 | 2.602 | 2.675 | 1.000 | 44.745 | 0.37 | 0.0 | 0.775 | 0.842 |
| hhh4_equivalent | 17 | 2.551 | 2.623 | 1.000 | 47.259 | 0.42 | 0.0 | 0.818 | 0.889 |

## Section B — Point-forecast diagnostics (ML convention)

_Note: hubs (FluSight, RespiCast) report MAE only; R²/MSE/RMSE/sMAPE_
_are ML-side diagnostics not standardised in epi-forecast literature._

| model | MAE | MAE 95% CI (BCa) | RMSE | R² | MAPE % | sMAPE % | dir-acc |
|---|---|---|---|---|---|---|---|
| persistence | 3.134 | (1.611, 7.723) | 6.172 | 0.392 | 21.09 | 18.16 | 0.562 |
| seasonal_naive | 7.536 | (5.283, 10.319) | 9.173 | -0.344 | 68.62 | 54.18 | 0.625 |
| ar1 | 3.055 | (1.633, 7.657) | 5.854 | 0.453 | 21.90 | 18.64 | 0.562 |
| hhh4_equivalent | 2.710 | (1.217, 8.233) | 5.980 | 0.429 | 17.58 | 15.58 | 0.562 |

## Section C — Clinical / alert diagnostics

_Threshold = 9.1 per 1,000 outpatient consultations (KDCA 2025-2026 절기)._
_Brier probability uses Gaussian tail P(Y>τ|μ̂,σ̂) — not magnitude ratio._

| model | Brier | Brier skill | sens | spec | PPV | NPV | F1 |
|---|---|---|---|---|---|---|---|
| persistence | 0.092 | 0.632 | nan | nan | nan | nan | nan |
| seasonal_naive | 0.201 | 0.192 | nan | nan | nan | nan | nan |
| ar1 | 0.097 | 0.612 | nan | nan | nan | nan | nan |
| hhh4_equivalent | 0.086 | 0.657 | nan | nan | nan | nan | nan |

## Section D — Statistical comparison vs persistence baseline

_DM and McNemar are methodological extensions; both Sherratt 2023 and_
_FluSight explicitly do NOT use them as primary forecast metrics._
_p-values at n=8 should be treated as exploratory only._

| model | DM stat | DM p | McNemar stat | McNemar p |
|---|---|---|---|---|
| seasonal_naive | 1.777 | 0.0946 | 3.000 | 1.0000 |
| ar1 | -1.040 | 0.3137 | 0.000 | 1.0000 |
| hhh4_equivalent | -2.132 | 0.0489 | 0.000 | 1.0000 |

## Provenance
- Real slab carved at phase1_data.py from idx 337 (paper_cutoff_week)
  forward — in-sample 학습/WF-CV/test phase never see these rows (real_eval only).
- σ for best/ensemble: in-sample OOF residual std.
- σ per naive baseline: that baseline's own in-sample residual std.
- Conformal PI: split-conformal ceiling-quantile from in-sample OOF
  residuals (Lei et al. 2018 JASA 113:1094 / Vovk 2005).
- Per-fold leakage recoder applied at each rolling step.
- KDCA threshold: season-aware lookup (`KDCA_THRESHOLD_BY_SEASON`).

## External-standard alignment
- Bracher 2021 (PLOS Comp Bio): WIS, PIT, K-level coverage ✓
- CDC FluSight 2024-25: WIS / 50-95% PI / peak metrics ✓
  (FluSight uses 23 quantiles vs our 4; defensible for thesis.)
- RespiCast / Sherratt 2023: pairwise relative WIS — TODO for future work.
- KDCA 2025-26 유행주의보 (2025-10-17): threshold = 9.1 per 1,000 ✓