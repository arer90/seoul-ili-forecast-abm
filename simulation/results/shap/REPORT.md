# R11 — SHAP + Feature Importance (all families)

- Models explained: **41** (permutation 18, native SHAP 27)
- Eval slab: n=68 (held-out test)


> **Corrected 2026-07-19.** Rows previously showed ✓ for models whose attributions were entirely zero, with the first four feature columns (`temp_avg, temp_min, humidity, wind_speed` and similar) printed as their top drivers — an artifact of sorting a zero vector, not a measurement. Those rows now read *not measured*. See `scripts/correct_shap_report.py`.

## Per-model

| model | family | permutation | native SHAP | top features |
|-------|--------|-------------|-------------|--------------|
| ARIMA | other | not measured | ✓ | temp_avg, humidity, wind_speed, rainfall |
| BayesianMCMC | linear | ✓ | ✓ | rt_subcrowd_ili, ili_rate_lag1, ili_rate_lag1_log1p, ili_age_50_64_lag1 |
| BayesianRidge | linear | ✓ | ✓ | temp_avg, humidity, rainfall, temp_min |
| CQR-LightGBM | tree | ✓ | ✓ | ili_rate_lag1, ili_rate_lag1_log1p, humid_ili, ili_age_1_6_lag1 |
| CQR-QuantReg | other | ✓ | ✓ | ili_rate_savgol_smooth_w9, ili_age_1_6_lag1, humid_ili, rt_nonresnt_ili |
| DLinear | linear | not measured | not measured | — |
| DNN | dl | not measured | not measured | — |
| DNN-Conformal | other | ✓ | ✓ | ili_age_1_6_lag1, ili_age_65p_lag1, ili_age_50_64_lag1, rt_nonresnt_ili |
| ElasticNet | linear | not measured | ✓ | ili_rate_lag1, ili_age_0_lag1, ili_rate_lag2, gt_bodyache_lag1 |
| EpiEstim | other | not measured | ✓ | temp_min, temp_avg, pressure, sunshine |
| FluSight-Baseline | other | not measured | not measured | — |
| FusedEpi | other | ✓ | ✓ | ili_rate_lag1_wavelet16, ili_rate_lag1, ili_rate_lag1_bit7, ili_rate_lag1_bit6 |
| GAM-Spline | other | not measured | not measured | — |
| GAT | dl | ✓ | ✓ | rt_nonresnt_ili, rt_highrisk_ili, ili_rate_lag1, rt_subcrowd_ili |
| GCN | dl | ✓ | ✓ | ili_rate_lag1, rt_nonresnt_ili, rt_subcrowd_ili, above_threshold |
| KRR | kernel | ✓ | ✓ | rt_subcrowd_ili, rt_roadcong_ili, ili_rate_lag1, ili_age_1_6_lag1 |
| LightGBM | other | ✓ | ✓ | gt_bodyache_lag1, fourier_cos_h1, sch_closure_lag1, mr_yoy_ratio |
| Mamba | dl | not measured | not measured | — |
| N-BEATS | dl | not measured | ✓ | sunshine, rainfall, wind_speed, temp_avg |
| N-HiTS | dl | ✓ | ✓ | rt_nonresnt_ili, ili_rate_lag4, vax_coverage, ili_rate_lag6 |
| NegBinGLM | linear | not measured | not measured | — |
| OverseasTransfer | dl | ✓ | ✓ | ili_rate_lag1_bit9, rt_nonresnt_ili, rt_subcrowd_ili, rt_roadcong_ili |
| PatchTST | dl | not measured | not measured | — |
| PoissonAutoreg | linear | not measured | not measured | — |
| RandomForest | tree | ✓ | ✓ | rt_highrisk_ili, ili_rate_lag1, rt_roadcong_ili, rt_subcrowd_ili |
| SARIMA | other | not measured | ✓ | temp_min, sunshine, temp_std, wind_speed |
| SARIMAX | other | ✓ | ✓ | pressure, temp_std, sunshine, temp_avg |
| SVR-Linear | kernel | not measured | ✓ | temp_avg, humidity, temp_min, rainfall |
| SVR-RBF | kernel | not measured | not measured | — |
| SeirCount-TabPFN | other | not measured | ✓ | wind_speed, temp_avg, sunshine, rainfall |
| TCN | dl | ✓ | ✓ | ili_rate_lag1, above_threshold, rt_roadcong_ili, rt_nonresnt_ili |
| TabPFN | other | ✓ | ✓ | fourier_cos_h1, fourier_sin_h3, fourier_cos_h3, fourier_cos_h2 |
| Theta | other | not measured | ✓ | temp_std, temp_min, temp_avg, sunshine |
| TiDE | dl | ✓ | ✓ | rt_nonresnt_ili, rt_roadcong_ili, rt_subcrowd_ili, ili_rate_lag1 |
| TiRex | dl | not measured | not measured | — |
| TimesFM-2.5 | other | not measured | not measured | — |
| TimesNet | dl | not measured | not measured | — |
| Wallinga-Teunis | other | not measured | ✓ | temp_avg, sunshine, temp_min, humidity |
| XGBoost | tree | ✓ | ✓ | rt_fcst_ppltn_max_avg, ili_rate_lag1, ili_rate_lag4, fourier_sin_h1 |
| hhh4-equivalent | other | not measured | not measured | — |
| iTransformer | dl | not measured | not measured | — |
