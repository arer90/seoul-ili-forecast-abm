# P2 — Inference on new data using champion `.pt` artifacts

- Models used: 39
- Inference window: n = 4 weeks
- Date range: 2026-05-24T00:00:00.000000 → 2026-06-14T00:00:00.000000

Each champion ships its **fitted scaler** and **transform state** (boxcox λ / PowerTransformer / log1p), so inference replays the exact training-time pipeline without re-fitting on inference X.

## Aggregate inference metrics (all horizons combined)

| model | n | WIS | MAE | RMSE | R² |
|---|---|---|---|---|---|
| FusedEpi | 4 | 0.219 | 0.343 | 0.393 | -0.014 |
| FluSight-Baseline | 4 | 0.220 | 0.321 | 0.400 | -0.048 |
| EpiEstim | 4 | 0.228 | 0.346 | 0.413 | -0.116 |
| TiRex | 4 | 0.361 | 0.503 | 0.596 | -1.326 |
| Theta | 4 | 0.395 | 0.639 | 0.704 | -2.253 |
| RandomForest | 4 | 0.490 | 0.697 | 0.781 | -2.998 |
| NegBinGLM | 4 | 0.613 | 0.812 | 1.079 | -6.630 |
| DLinear | 4 | 0.634 | 0.867 | 0.996 | -5.497 |
| GCN | 4 | 0.721 | 1.135 | 1.291 | -9.930 |
| Wallinga-Teunis | 4 | 0.925 | 1.077 | 1.110 | -7.074 |
| TimesFM-2.5 | 4 | 0.992 | 1.232 | 1.297 | -10.023 |
| OverseasTransfer | 4 | 1.072 | 1.164 | 1.180 | -8.125 |
| KRR | 4 | 1.182 | 1.613 | 1.837 | -21.127 |
| DNN-Conformal | 4 | 1.862 | 2.064 | 2.101 | -27.933 |
| N-BEATS | 4 | 1.947 | 2.558 | 2.805 | -50.582 |
| GAT | 4 | 2.172 | 2.652 | 2.769 | -49.254 |
| N-HiTS | 4 | 2.635 | 3.469 | 3.861 | -96.744 |
| TCN | 4 | 3.440 | 3.710 | 3.755 | -91.433 |
| TiDE | 4 | 3.634 | 4.780 | 6.838 | -305.510 |
| ARIMA | 4 | 4.053 | 5.529 | 6.356 | -263.862 |
| BayesianMCMC | 4 | 4.523 | 4.614 | 4.631 | -139.571 |
| GAM-Spline | 4 | 4.523 | 4.614 | 4.631 | -139.571 |
| SVR-Linear | 4 | 4.523 | 4.614 | 4.631 | -139.571 |
| CQR-QuantReg | 4 | 4.791 | 4.947 | 4.972 | -161.064 |
| SARIMA | 4 | 7.848 | 9.726 | 10.220 | -683.694 |
| CQR-LightGBM | 4 | 9.793 | 9.665 | 9.673 | -612.381 |
| PatchTST | 4 | 15.749 | 15.374 | 15.379 | -1549.369 |
| Mamba | 4 | 24.925 | 24.167 | 24.170 | -3828.411 |
| LightGBM | 4 | 26.708 | 26.741 | 26.789 | -4703.164 |
| TabPFN | 4 | 28.659 | 28.324 | 28.350 | -5267.365 |
| DNN | 4 | 34.504 | 35.138 | 35.256 | -8147.046 |
| iTransformer | 4 | 35.011 | 33.832 | 33.835 | -7503.257 |
| SARIMAX | 4 | 43.213 | 42.790 | 42.833 | -12025.410 |
| XGBoost | 4 | 45.063 | 43.651 | 43.656 | -12492.122 |
| SVR-RBF | 4 | 96.013 | 92.293 | 92.293 | -55836.567 |
| PoissonAutoreg | 4 | 115.107 | 150.120 | 162.738 | -173604.879 |
| BayesianRidge | 4 | 145.324 | 181.276 | 190.509 | -237911.286 |
| ElasticNet | 4 | 205.049 | 196.786 | 196.786 | -253846.990 |
| SeirCount-TabPFN | 4 | 328.211 | 318.938 | 318.997 | -667049.663 |

## Per-horizon breakdown (h=1 = next-week, primary KPI)

Note: forecasting accuracy *naturally* degrades over horizons (compounding uncertainty). h=1 is what KDCA weekly alerts use; later horizons are scenario-planning only.

### Absolute error per horizon (lower = better)

| model | h1 | h2 | h3 | h4 |
|---|---|---|---|---|
| ARIMA | 1.20 | 4.10 | 7.41 | 9.40 |
| BayesianMCMC | 5.14 | 4.73 | 4.06 | 4.53 |
| BayesianRidge | 165.07 | 137.10 | 141.85 | 281.09 |
| CQR-LightGBM | 9.14 | 9.55 | 10.22 | 9.75 |
| CQR-QuantReg | 4.11 | 5.41 | 5.19 | 5.07 |
| DLinear | 1.33 | 1.12 | 0.97 | 0.05 |
| DNN | 34.08 | 32.91 | 33.47 | 40.09 |
| DNN-Conformal | 1.54 | 1.95 | 2.62 | 2.15 |
| ElasticNet | 196.26 | 196.67 | 197.34 | 196.87 |
| EpiEstim | 0.67 | 0.25 | 0.42 | 0.05 |
| FluSight-Baseline | 0.61 | 0.20 | 0.47 | 0.00 |
| FusedEpi | 0.44 | 0.11 | 0.61 | 0.22 |
| GAM-Spline | 5.14 | 4.73 | 4.06 | 4.53 |
| GAT | 3.93 | 2.68 | 2.14 | 1.86 |
| GCN | 1.97 | 1.02 | 0.25 | 1.29 |
| KRR | 0.67 | 2.45 | 2.54 | 0.79 |
| LightGBM | 28.00 | 28.45 | 26.06 | 24.45 |
| Mamba | 23.64 | 24.05 | 24.72 | 24.25 |
| N-BEATS | 0.69 | 2.77 | 2.96 | 3.82 |
| N-HiTS | 0.56 | 4.73 | 4.06 | 4.53 |
| NegBinGLM | 0.15 | 1.38 | 1.65 | 0.07 |
| OverseasTransfer | 1.48 | 0.99 | 1.06 | 1.13 |
| PatchTST | 14.84 | 15.26 | 15.93 | 15.46 |
| PoissonAutoreg | 56.21 | 131.92 | 192.82 | 219.53 |
| RandomForest | 0.76 | 0.99 | 0.93 | 0.10 |
| SARIMA | 7.85 | 13.51 | 11.92 | 5.63 |
| SARIMAX | 43.95 | 41.32 | 45.31 | 40.58 |
| SVR-Linear | 5.14 | 4.73 | 4.06 | 4.53 |
| SVR-RBF | 91.76 | 92.18 | 92.85 | 92.38 |
| SeirCount-TabPFN | 312.41 | 314.37 | 320.90 | 328.07 |
| TCN | 4.43 | 4.04 | 3.48 | 2.89 |
| TabPFN | 29.39 | 29.58 | 27.57 | 26.75 |
| Theta | 1.06 | 0.23 | 0.58 | 0.68 |
| TiDE | 0.26 | 3.91 | 12.95 | 2.01 |
| TiRex | 0.92 | 0.62 | 0.04 | 0.43 |
| TimesFM-2.5 | 1.79 | 1.22 | 0.65 | 1.27 |
| Wallinga-Teunis | 1.23 | 1.06 | 0.65 | 1.36 |
| XGBoost | 43.45 | 43.87 | 44.54 | 42.75 |
| iTransformer | 33.30 | 33.72 | 34.39 | 33.92 |

### Prediction vs actual per horizon

| model | metric | h1 | h2 | h3 | h4 |
|---|---|---|---|---|---|
| _ground truth_ | actual | 5.142857142857143 | 4.728571428571429 | 4.057142857142857 | 4.5285714285714285 |
| ARIMA | pred | 6.34 | 8.83 | 11.47 | 13.93 |
| BayesianMCMC | pred | 0.00 | 0.00 | 0.00 | 0.00 |
| BayesianRidge | pred | 170.22 | 141.83 | 145.90 | 285.62 |
| CQR-LightGBM | pred | 14.28 | 14.28 | 14.28 | 14.28 |
| CQR-QuantReg | pred | 9.26 | 10.14 | 9.25 | 9.60 |
| DLinear | pred | 3.82 | 3.60 | 3.09 | 4.58 |
| DNN | pred | 39.23 | 37.64 | 37.52 | 44.62 |
| DNN-Conformal | pred | 6.68 | 6.68 | 6.68 | 6.68 |
| ElasticNet | pred | 201.40 | 201.40 | 201.40 | 201.40 |
| EpiEstim | pred | 4.48 | 4.48 | 4.47 | 4.48 |
| FluSight-Baseline | pred | 4.53 | 4.53 | 4.53 | 4.53 |
| FusedEpi | pred | 4.71 | 4.62 | 4.66 | 4.75 |
| GAM-Spline | pred | 0.00 | 0.00 | 0.00 | 0.00 |
| GAT | pred | 9.08 | 7.40 | 6.20 | 6.38 |
| GCN | pred | 7.12 | 3.70 | 3.80 | 3.24 |
| KRR | pred | 5.82 | 7.17 | 6.59 | 5.32 |
| LightGBM | pred | 33.14 | 33.18 | 30.11 | 28.98 |
| Mamba | pred | 28.78 | 28.78 | 28.78 | 28.78 |
| N-BEATS | pred | 4.46 | 1.96 | 1.10 | 0.71 |
| N-HiTS | pred | 4.58 | 0.00 | 0.00 | 0.00 |
| NegBinGLM | pred | 5.29 | 6.11 | 5.71 | 4.59 |
| OverseasTransfer | pred | 6.63 | 5.72 | 5.12 | 5.65 |
| PatchTST | pred | 19.99 | 19.98 | 19.99 | 19.99 |
| PoissonAutoreg | pred | 61.35 | 136.65 | 196.87 | 224.06 |
| RandomForest | pred | 5.90 | 5.72 | 4.99 | 4.42 |
| SARIMA | pred | 12.99 | 18.24 | 15.97 | 10.16 |
| SARIMAX | pred | 49.09 | 46.04 | 49.37 | 45.11 |
| SVR-Linear | pred | 0.00 | 0.00 | 0.00 | 0.00 |
| SVR-RBF | pred | 96.91 | 96.91 | 96.91 | 96.91 |
| SeirCount-TabPFN | pred | 317.55 | 319.10 | 324.96 | 332.60 |
| TCN | pred | 9.57 | 8.76 | 7.54 | 7.42 |
| TabPFN | pred | 34.53 | 34.31 | 31.63 | 31.28 |
| Theta | pred | 4.08 | 4.50 | 4.64 | 5.21 |
| TiDE | pred | 4.88 | 8.63 | 17.01 | 6.53 |
| TiRex | pred | 4.22 | 4.11 | 4.10 | 4.09 |
| TimesFM-2.5 | pred | 3.35 | 3.51 | 3.41 | 3.26 |
| Wallinga-Teunis | pred | 3.91 | 3.67 | 3.40 | 3.17 |
| XGBoost | pred | 48.60 | 48.60 | 48.60 | 47.28 |
| iTransformer | pred | 38.45 | 38.45 | 38.45 | 38.45 |

## Champions used

| model | version | test_WIS@promotion | promoted_at | transform | scaler | n_features |
|---|---|---|---|---|---|---|
| ARIMA | v1 | None | 2026-06-30T21:14:03Z | identity | none | ? |
| BayesianMCMC | v1 | None | 2026-06-30T21:13:34Z | hier_none | none | 20 |
| BayesianRidge | v1 | None | 2026-06-30T21:07:02Z | hier_individual | none | 12 |
| CQR-LightGBM | v1 | None | 2026-07-01T01:40:03Z | hier_none | none | 20 |
| CQR-QuantReg | v1 | None | 2026-07-01T01:40:11Z | hier_none | none | 20 |
| DLinear | v1 | None | 2026-07-01T01:39:52Z | hier_none | none | 32 |
| DNN | v1 | None | 2026-06-30T21:36:56Z | hier_none | none | 12 |
| DNN-Conformal | v1 | None | 2026-06-30T21:37:17Z | hier_individual | none | 20 |
| ElasticNet | v1 | None | 2026-06-30T21:06:55Z | hier_none | none | 20 |
| EpiEstim | v1 | None | 2026-06-30T21:13:39Z | identity | none | ? |
| FluSight-Baseline | v1 | None | 2026-06-30T21:20:34Z | identity | none | ? |
| FusedEpi | v1 | None | 2026-07-01T02:40:45Z | hier_none | none | 32 |
| GAM-Spline | v1 | None | 2026-06-30T21:13:17Z | hier_individual | none | 20 |
| GAT | v1 | None | 2026-07-01T01:49:46Z | hier_none | none | 32 |
| GCN | v1 | None | 2026-07-01T02:11:27Z | hier_individual | none | 32 |
| KRR | v1 | None | 2026-06-30T21:08:15Z | hier_none | none | 20 |
| LightGBM | v1 | None | 2026-06-30T20:58:52Z | hier_none | none | 20 |
| Mamba | v1 | None | 2026-06-30T23:24:06Z | identity | none | 13 |
| N-BEATS | v1 | None | 2026-07-01T00:40:21Z | hier_none | none | 32 |
| N-HiTS | v1 | None | 2026-07-01T00:47:45Z | hier_none | none | 32 |
| NegBinGLM | v1 | None | 2026-06-30T21:07:09Z | identity | none | ? |
| OverseasTransfer | v1 | None | 2026-07-01T02:19:01Z | hier_none | none | 32 |
| PatchTST | v1 | None | 2026-06-30T22:33:38Z | hier_individual | none | 32 |
| PoissonAutoreg | v1 | None | 2026-06-30T21:07:15Z | identity | none | ? |
| RandomForest | v1 | None | 2026-06-30T21:06:19Z | hier_none | none | 20 |
| SARIMA | v1 | None | 2026-06-30T21:15:39Z | identity | none | ? |
| SARIMAX | v1 | None | 2026-06-30T21:20:10Z | identity | none | ? |
| SVR-Linear | v1 | None | 2026-06-30T21:10:13Z | hier_none | none | 12 |
| SVR-RBF | v1 | None | 2026-06-30T21:13:05Z | hier_none | none | 348 |
| SeirCount-TabPFN | v1 | None | 2026-06-30T22:15:56Z | hier_individual | none | 348 |
| TCN | v1 | None | 2026-07-01T01:39:09Z | hier_none | none | 32 |
| TabPFN | v1 | None | 2026-06-30T21:45:36Z | hier_none | none | 32 |
| Theta | v1 | None | 2026-06-30T21:20:28Z | identity | none | ? |
| TiDE | v1 | None | 2026-07-01T00:55:59Z | hier_individual | none | 32 |
| TiRex | v1 | None | 2026-07-01T02:23:47Z | hier_none | none | 32 |
| TimesFM-2.5 | v1 | None | 2026-07-01T02:17:42Z | hier_none | none | 32 |
| Wallinga-Teunis | v1 | None | 2026-06-30T21:13:52Z | identity | none | ? |
| XGBoost | v1 | None | 2026-06-30T20:52:36Z | hier_none | none | 20 |
| iTransformer | v1 | None | 2026-06-30T22:56:58Z | identity | none | 13 |

> **Legacy** flag: champion was a bare-model pickle (pre-artifact). Inference falls back to identity transform + no scaler — these predictions may differ from training-time pipeline. Re-run R9 to upgrade to a `ChampionArtifact`.