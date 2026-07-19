# R12 — Comprehensive Evaluation Report

- Models evaluated: 48
- Per-model deep-dive reports: 48
- Master grid CSV: `MASTER_GRID.csv`
- Statistical tables: ['dm_pvalues']
- Figures: ['forest_plot_wis', 'heatmap_model_x_metric', 'calibration_curve', 'horizon_decay']

## Consolidated ranking (Borda-count across phases)

| rank | model | Borda score |
|------|-------|-------------|
| 1 | FusedEpi | 1.0 |
| 2 | GAM-Spline | 2.0 |
| 3 | TiRex | 3.0 |
| 4 | SVR-RBF | 4.0 |
| 5 | NegBinGLM | 5.0 |
| 6 | PoissonAutoreg | 6.0 |
| 7 | ElasticNet | 7.0 |
| 8 | KRR | 8.0 |
| 9 | CQR-QuantReg | 9.0 |
| 10 | ARIMA | 10.0 |

## R/P coverage

| R/P | Result key | Status |
|-----|------------|--------|
| R2 | `baseline` | missing/skipped |
| R4 | `wfcv` | missing/skipped |
| R5 | `diagnostics` | OK |
| R6 | `dm_tests` | OK |
| R7 | `prediction_intervals` | OK |
| R8 | `scoring` | OK |
| R9 | `per_model_optimize` | OK |
| R10 | `per_model_eval` | OK |
| R11 | `feature_importance` | OK |

## Per-age Fairness (TRIPOD-AI 5g) — auto-loaded from phase11_fairness/

_(no fairness output found — run `python -m simulation.scripts.phase11_fairness`)_

## Cross-Season LOSO (TRIPOD-AI 5j) — auto-loaded from phase11_loso/

_(no LOSO output found — run `python -m simulation.scripts.phase11_loso_full`)_

## Per-model deep-dive index

- [ARIMA](per_model/ARIMA.md)
- [BayesianMCMC](per_model/BayesianMCMC.md)
- [BayesianRidge](per_model/BayesianRidge.md)
- [CQR-LightGBM](per_model/CQR-LightGBM.md)
- [CQR-QuantReg](per_model/CQR-QuantReg.md)
- [DLinear](per_model/DLinear.md)
- [DNN](per_model/DNN.md)
- [DNN-Conformal](per_model/DNN-Conformal.md)
- [ElasticNet](per_model/ElasticNet.md)
- [Ensemble-Adaptive](per_model/Ensemble-Adaptive.md)
- [Ensemble-BMA](per_model/Ensemble-BMA.md)
- [Ensemble-Diversity](per_model/Ensemble-Diversity.md)
- [Ensemble-InvRMSE](per_model/Ensemble-InvRMSE.md)
- [Ensemble-NNLS](per_model/Ensemble-NNLS.md)
- [Ensemble-NNLS-Filtered](per_model/Ensemble-NNLS-Filtered.md)
- [Ensemble-ResidualAR](per_model/Ensemble-ResidualAR.md)
- [EpiEstim](per_model/EpiEstim.md)
- [FluSight-Baseline](per_model/FluSight-Baseline.md)
- [FusedEpi](per_model/FusedEpi.md)
- [GAM-Spline](per_model/GAM-Spline.md)
- [GAT](per_model/GAT.md)
- [GCN](per_model/GCN.md)
- [KRR](per_model/KRR.md)
- [LightGBM](per_model/LightGBM.md)
- [Mamba](per_model/Mamba.md)
- [N-BEATS](per_model/N-BEATS.md)
- [N-HiTS](per_model/N-HiTS.md)
- [NegBinGLM](per_model/NegBinGLM.md)
- [OverseasTransfer](per_model/OverseasTransfer.md)
- [PatchTST](per_model/PatchTST.md)
- [PoissonAutoreg](per_model/PoissonAutoreg.md)
- [RandomForest](per_model/RandomForest.md)
- [SARIMA](per_model/SARIMA.md)
- [SARIMAX](per_model/SARIMAX.md)
- [SVR-Linear](per_model/SVR-Linear.md)
- [SVR-RBF](per_model/SVR-RBF.md)
- [SeirCount-TabPFN](per_model/SeirCount-TabPFN.md)
- [TCN](per_model/TCN.md)
- [TabPFN](per_model/TabPFN.md)
- [Theta](per_model/Theta.md)
- [TiDE](per_model/TiDE.md)
- [TiRex](per_model/TiRex.md)
- [TimesFM-2.5](per_model/TimesFM-2.5.md)
- [TimesNet](per_model/TimesNet.md)
- [Wallinga-Teunis](per_model/Wallinga-Teunis.md)
- [XGBoost](per_model/XGBoost.md)
- [hhh4-equivalent](per_model/hhh4-equivalent.md)
- [iTransformer](per_model/iTransformer.md)

## Reproducibility

- Audit metadata: `simulation/results/eval_logs/{run_id}_audit.json`
- Per-record evaluation log: `simulation/results/eval_logs/{run_id}.jsonl`
- All-runs index: `simulation/results/eval_logs/INDEX.csv`

## Metric interpretation rubric

_Per-metric thresholds with literature citations. Quality tag (`✓ excellent` /
`good` / `⚠ acceptable` / `✗ poor`) is applied automatically in per-model_
_deep-dive reports. Direction column: `lower=lower-is-better`, `higher=higher-is-better`,_
_`calibration=closer-to-nominal-better`._

| Metric | Direction | Excellent | Good | Acceptable | Citation |
|---|---|---|---|---|---|
| **R²** | higher | ≥ 0.9 | ≥ 0.8 | ≥ 0.5 | Hyndman & Athanasopoulos (2021), Forecasting Principles & Practice |
| **MAE** | lower | ≤ 2.0 | ≤ 4.0 | ≤ 6.0 | FluSight 2024-25; Bracher et al. 2021 PLOS Comp Bio |
| **RMSE** | lower | ≤ 3.0 | ≤ 6.0 | ≤ 10.0 | Hyndman & Koehler 2006 IJF 22:679 |
| **MAPE (%)** | lower | ≤ 10.0 | ≤ 20.0 | ≤ 50.0 | Lewis 1982 'International and Business Forecasting Methods'; Hyndman & Koehler 2006 |
| **sMAPE (%)** | lower | ≤ 10.0 | ≤ 20.0 | ≤ 40.0 | Hyndman & Koehler 2006 IJF 22:679 |
| **MdAPE (%)** | lower | ≤ 8.0 | ≤ 15.0 | ≤ 30.0 | Armstrong & Collopy 1992 IJF 8:69 |
| **MASE (h=1)** | lower | ≤ 0.5 | ≤ 1.0 | ≤ 1.5 | Hyndman & Koehler 2006 IJF 22:679 (canonical scaled error) |
| **MASE (h=52, seasonal)** | lower | ≤ 0.3 | ≤ 0.7 | ≤ 1.0 | Hyndman & Koehler 2006 IJF 22:679 |
| **MASE (h=4, monthly seasonality)** | lower | ≤ 0.5 | ≤ 1.0 | ≤ 1.5 | Hyndman & Koehler 2006 IJF 22:679 |
| **MASE (h=13, quarterly)** | lower | ≤ 0.4 | ≤ 0.8 | ≤ 1.2 | Hyndman & Koehler 2006 IJF 22:679 |
| **Bias (mean signed error)** | calibration | |target−value| ≤ 0.5 | ≤ 2.0 | else | Hyndman & Koehler 2006 |
| **MSLE** | lower | ≤ 0.05 | ≤ 0.2 | ≤ 1.0 | Tofallis 2015 J Operational Research Soc 66:1352 |
| **Theil's U2** | lower | ≤ 0.5 | ≤ 1.0 | ≤ 1.5 | Theil 1966; Bliemel 1973 Mgmt Sci 19:444 |
| **Log score (Gaussian NLL)** | lower | ≤ 2.5 | ≤ 3.5 | ≤ 5.0 | Gneiting & Raftery 2007 JASA 102:359 |
| **MAE skill vs persistence** | higher | ≥ 0.5 | ≥ 0.3 | ≥ 0.1 | Murphy 1973 J Appl Meteor 12:595; Bracher 2021 PLOS Comp Bio |
| **WIS** | lower | ≤ 2.0 | ≤ 4.0 | ≤ 6.0 | Bracher J et al. 2021 PLOS Comp Bio 17:e1008618 |
| **log-WIS** | lower | ≤ 0.1 | ≤ 0.2 | ≤ 0.4 | Bosse NI et al. 2023 PLoS Comp Bio 19:e1011393 |
| **CRPS (Gaussian)** | lower | ≤ 2.0 | ≤ 4.0 | ≤ 6.0 | Gneiting & Raftery 2007 JASA 102(477):359, Eq.(5) |
| **Pinball loss q=0.50** | lower | ≤ 1.0 | ≤ 2.5 | ≤ 4.0 | Tibshirani 2023 lecture notes (statlearn) |
| **PIT mean** | calibration | |target−value| ≤ 0.05 | ≤ 0.1 | else | Bracher 2021 §3, Gneiting Balabdaoui Raftery 2007 JRSS-B 69:243 |
| **PIT KS p-value** | higher | ≥ 0.2 | ≥ 0.05 | ≥ 0.01 | Gneiting Balabdaoui Raftery 2007 JRSS-B |
| **95% PI empirical coverage** | calibration | |target−value| ≤ 0.02 | ≤ 0.05 | else | Bracher 2021; FluSight 2024-25 |
| **80% PI coverage** | calibration | |target−value| ≤ 0.05 | ≤ 0.1 | else | Bracher 2021 |
| **50% PI coverage** | calibration | |target−value| ≤ 0.05 | ≤ 0.1 | else | FluSight 2024-25 evaluation report |
| **Direction accuracy** | higher | ≥ 0.75 | ≥ 0.65 | ≥ 0.55 | FluSight 2024-25 categorical 'rate-trend' target |
| **Peak week error (|Δweeks|)** | lower | ≤ 0.0 | ≤ 1.0 | ≤ 2.0 | CDC FluSight 2018-19 onward (peak-week target) |
| **Peak intensity rel-err** | lower | ≤ 0.1 | ≤ 0.2 | ≤ 0.4 | CDC FluSight peak-intensity target |
| **Alert F1 (KDCA threshold)** | higher | ≥ 0.9 | ≥ 0.75 | ≥ 0.5 | KDCA 인플루엔자 표본감시 운영지침; Reich Lab benchmark |
| **Brier score (alert event)** | lower | ≤ 0.1 | ≤ 0.2 | ≤ 0.3 | Brier 1950 MWR 78:1 |
| **Brier skill score (vs climatology)** | higher | ≥ 0.5 | ≥ 0.3 | ≥ 0.1 | Murphy 1973 J Appl Meteor 12:595 |
| **Sensitivity (recall)** | higher | ≥ 0.9 | ≥ 0.8 | ≥ 0.7 | Standard public-health alert system metric |
| **Specificity** | higher | ≥ 0.9 | ≥ 0.8 | ≥ 0.7 | Standard public-health alert system metric |
| **PPV (precision)** | higher | ≥ 0.85 | ≥ 0.7 | ≥ 0.5 | Standard prevalence-dependent metric |
| **NPV** | higher | ≥ 0.95 | ≥ 0.85 | ≥ 0.7 | Standard prevalence-dependent metric |
| **Clinical F1 (binary at threshold)** | higher | ≥ 0.85 | ≥ 0.7 | ≥ 0.55 | van Rijsbergen 1979 'Information Retrieval' |
| **Relative WIS (pairwise tournament)** | lower | ≤ 0.5 | ≤ 0.8 | ≤ 1.0 | Sherratt K et al. 2023 eLife 12:e81916; Bosse 2022 scoringutils |

_Caveats:_
- Thresholds reflect mid-tier-journal expectations; reviewers may disagree.
- For `calibration` metrics, the value column shows |empirical − nominal|.
- For ILI rate forecasting at n=68, MAE in the 3-5 range is competitive
  (NegBinGLM achieves 3.92, the lowest in our 61-model leaderboard).
- WIS thresholds calibrated against FluSight 2024-25 typical scores.
