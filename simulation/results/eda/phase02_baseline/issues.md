# Phase EDA issues


- **ARIMA**: outliers: 2 (|z| > 3.0)

- **SARIMA**: MAPE = 30.3% (> 30%); outliers: 2 (|z| > 3.0)

- **Theta**: |ACF lag-1| = 0.62 (> 0.3); outliers: 2 (|z| > 3.0)

- **SARIMAX**: outliers: 2 (|z| > 3.0)

- **SVR-Linear**: |ACF lag-1| = 0.49 (> 0.3); outliers: 3 (|z| > 3.0)

- **SVR-RBF**: R² = -0.502 (< 0.5) catastrophic; MAPE = 54.5% (> 30%); |ACF lag-1| = 0.92 (> 0.3); outliers: 2 (|z| > 3.0)

- **BayesianRidge**: |ACF lag-1| = 0.37 (> 0.3); outliers: 2 (|z| > 3.0)

- **ElasticNet**: |ACF lag-1| = 0.60 (> 0.3); outliers: 1 (|z| > 3.0)

- **CQR-QuantReg**: outliers: 1 (|z| > 3.0)

- **KRR**: |ACF lag-1| = 0.38 (> 0.3); outliers: 2 (|z| > 3.0)

- **NegBinGLM**: |ACF lag-1| = 0.34 (> 0.3); outliers: 2 (|z| > 3.0)

- **EpiEstim**: MAPE = 33.2% (> 30%); |ACF lag-1| = 0.77 (> 0.3); outliers: 2 (|z| > 3.0)
- **hhh4-equivalent**: R² = -3.557 (< 0.5) catastrophic; MAPE = 243.7% (> 30%); |ACF lag-1| = 0.90 (> 0.3)

- **Wallinga-Teunis**: MAPE = 30.9% (> 30%); |ACF lag-1| = 0.78 (> 0.3); outliers: 2 (|z| > 3.0)

- **XGBoost**: |ACF lag-1| = 0.86 (> 0.3); outliers: 2 (|z| > 3.0)

- **GAM-Spline**: MAPE = 30.6% (> 30%); |ACF lag-1| = 0.69 (> 0.3); outliers: 1 (|z| > 3.0)

- **LightGBM**: |ACF lag-1| = 0.84 (> 0.3); outliers: 2 (|z| > 3.0)

- **PoissonAutoreg**: |ACF lag-1| = 0.48 (> 0.3); outliers: 2 (|z| > 3.0)

- **RandomForest**: |ACF lag-1| = 0.83 (> 0.3); outliers: 2 (|z| > 3.0)

- **CQR-LightGBM**: MAPE = 61.2% (> 30%); |ACF lag-1| = 0.86 (> 0.3); outliers: 3 (|z| > 3.0)

- **DNN**: MAPE = 32.1% (> 30%); |ACF lag-1| = 0.81 (> 0.3); outliers: 2 (|z| > 3.0)

- **BayesianMCMC**: |ACF lag-1| = 0.40 (> 0.3); outliers: 2 (|z| > 3.0)

- **DLinear**: outliers: 2 (|z| > 3.0)

- **N-BEATS**: R² = -1.141 (< 0.5) catastrophic; MAPE = 98.4% (> 30%); |ACF lag-1| = 0.93 (> 0.3); outliers: 4 (|z| > 3.0)

- **N-HiTS**: R² = -1.026 (< 0.5) catastrophic; MAPE = 87.7% (> 30%); |ACF lag-1| = 0.93 (> 0.3); outliers: 2 (|z| > 3.0)

- **TCN**: R² = 0.374 (< 0.5); MAPE = 49.7% (> 30%); |ACF lag-1| = 0.89 (> 0.3); outliers: 2 (|z| > 3.0)

- **PatchTST**: MAPE = 30.6% (> 30%); |ACF lag-1| = 0.87 (> 0.3); outliers: 2 (|z| > 3.0)

- **iTransformer**: R² = 0.018 (< 0.5); MAPE = 49.3% (> 30%); |ACF lag-1| = 0.89 (> 0.3); outliers: 2 (|z| > 3.0)

- **DNN-Conformal**: |ACF lag-1| = 0.32 (> 0.3); outliers: 2 (|z| > 3.0)

- **TimesFM-2.5**: outliers: 2 (|z| > 3.0)

- **TabPFN**: |ACF lag-1| = 0.65 (> 0.3); outliers: 2 (|z| > 3.0)

- **TiRex**: outliers: 2 (|z| > 3.0)

- **GAT**: MAPE = 43.7% (> 30%); |ACF lag-1| = 0.73 (> 0.3); outliers: 1 (|z| > 3.0)

- **Mamba**: MAPE = 41.4% (> 30%); |ACF lag-1| = 0.78 (> 0.3); outliers: 1 (|z| > 3.0)

- **TimesNet**: |ACF lag-1| = 0.83 (> 0.3); outliers: 2 (|z| > 3.0)

- **OverseasTransfer**: |ACF lag-1| = 0.80 (> 0.3); outliers: 2 (|z| > 3.0)

- **TiDE**: R² = -0.512 (< 0.5) catastrophic; MAPE = 76.0% (> 30%); |ACF lag-1| = 0.92 (> 0.3); outliers: 2 (|z| > 3.0)
