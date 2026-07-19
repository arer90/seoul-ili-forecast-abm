### TABLE 0
Master’s Thesis

Multi-Agent Simulation of Adaptive Behavioral Responses to Infectious Disease Transmission

Real-Time Influenza (ILI) Forecasting and Commuter-Coupled District Transmission Modeling for Seoul's 25 Districts
Seungjin Lee

Department of Epidemiology and Health Informatics

Graduate School of Public Health

Korea University

June 2026
Multi-Agent Simulation of Adaptive Behavioral Responses to Infectious Disease Transmission

Real-Time Influenza (ILI) Forecasting and Commuter-Coupled District Transmission Modeling for Seoul's 25 Districts



by                                                                               Seungjin Lee

_____________________________________

under the supervision of Professor Won Jin Lee

A thesis submitted in partial fulfillment of                       the requirements for the degree of                                   Master of  Public Health
### TABLE 1
__________________________
Committee Chair: Won Jin Lee

__________________________
Committee Member: Byung-Chul Chun

__________________________
Committee Member: Seung-A Choe
### TABLE 2
Model | WIS | R2 | RMSE | MAE | AUC-ROC | C-index
*NegBinGLM-V7 | 3.25 | 0.928 | 6.97 | 4.20 | 0.995 | 0.924
PoissonAutoreg | 3.41 | 0.917 | 7.49 | 4.35 | 0.997 | 0.922
ARIMA | 14.98 | -0.37 | 30.45 | 19.78 | 0.485 | 0.595
Theta | 17.71 | -0.82 | 35.07 | 23.65 | 0.483 | 0.603
SARIMAX | 18.66 | -1.01 | 36.81 | 25.00 | 0.199 | 0.286
SARIMA | 19.00 | -0.99 | 36.62 | 26.41 | 0.341 | 0.465
### TABLE 3
Model | MAE | RMSE | MAPE % | SMAPE %
*NegBinGLM-V7 | 4.20 | 6.97 | 15.95 | 16.41
PoissonAutoreg | 4.35 | 7.49 | 16.30 | 16.95
ARIMA | 19.78 | 30.45 | 61.70 | 71.99
Theta | 23.65 | 35.07 | 64.37 | 106.20
SARIMAX | 25.00 | 36.81 | 69.17 | 120.46
SARIMA | 26.41 | 36.62 | 86.93 | 165.57
### TABLE 4
Region | Shape correlation with Seoul | Note
Korea (national) | 0.90 | independent KR source - consistency check
Japan | 0.57 | East Asian temperate
China | 0.56 | East Asian temperate
France | 0.55 | Northern-Hemisphere temperate
Singapore | 0.54 | tropical, partial alignment
United States | 0.49 | aggregated national
Australia | 0.24 | Southern Hemisphere - season offset
### TABLE 5
Dimension | Champion result | Assessment
Point accuracy (R2 / RMSE / MAE) | 0.928 / 6.97 / 4.20 | strong
Probabilistic accuracy (WIS) | 3.25 | strong
Outbreak discrimination (AUC-ROC / C-index) | 0.995 / 0.924 | strong
95% interval coverage | under-nominal (raw) | recalibrated by CI
### TABLE 6
Category (n) | Models | Role
tree (4) | XGBoost, LightGBM, RandomForest, CatBoost | Gradient-boosted and bagged decision-tree ensembles
linear (5) | ElasticNet, BayesianRidge, NegBinGLM, NegBinGLM-V7, PoissonAutoreg | Regularized and count-regression linear models (champion family)
kernel (3) | KRR, SVR-Linear, SVR-RBF | Kernel-ridge and support-vector regression
other (1) | GAM-Spline | Generalized additive spline model
epi-extended (7) | EpiEstim, hhh4-equivalent, Wallinga-Teunis, GLARMA, EARS-C1, EARS-C2, EARS-C3 | Reproduction-number, count-time-series, and aberration-detection surveillance models
ts (5) | ARIMA, SARIMA, SARIMAX, Theta, FluSight-Baseline | Classical seasonal time series and the FluSight reference baseline
dl-tabular (4) | DNN, DNN-Optuna, DNN-Conformal, TabularDNN | Feed-forward and conformalized deep tabular networks
modern-ts (9) | PatchTST, iTransformer, Mamba, TimesNet, N-BEATS, N-HiTS, TiDE, TCN, TCN-Optuna | Transformer, state-space, and basis-expansion sequence networks
cqr (3) | CQR-LightGBM, CQR-GBR, CQR-QuantReg | Conformalized quantile regressors (interval-native)
graph (2) | GAT, GCN | Graph neural networks over the district adjacency
foundation (3) | Chronos-2, Chronos-2-FT, OverseasTransfer | Pretrained time-series foundation models and cross-country transfer
ensemble (7) | Ensemble-NNLS, Ensemble-NNLS-Filtered, Ensemble-BMA, Ensemble-InvRMSE, Ensemble-Diversity, Ensemble-Adaptive, Ensemble-ResidualAR | Stacking, NNLS, and Bayesian-model-averaging combiners
### TABLE 7
Family (n) | Metrics
Point error & scale (17) | r2, mae, rmse, mse, mape, smape, mdape, mase_h1, mase_h4, mase_h13, mase_h26, mase_h52, bias_mean_error, msle, theils_u, auprc, tp
Probabilistic scores (8) | wis, log_wis, crps_gaussian, pinball_q05, pinball_q95, wis_underpred, wis_overpred, wis_total_decomp
Interval calibration (22) | pit_mean, pit_std, pit_ks_p, sigma_in_sample, pi95_coverage, pi95_width, pi80_coverage, pi80_width, pi50_coverage, pi50_width, brier_reliability, calibration_slope, calibration_intercept, pi50_rel_width, pi80_rel_width, pi95_rel_width, wis_sharpness, pi99_coverage, pi99_width, pi95_relia, pi80_relia, pi_sharpness_ratio
Discrimination & alert (34) | direction_acc, alert_threshold, brier_score, brier_skill, brier_resolution, brier_uncertainty, pearson_r, spearman_r, c_index, roc_auc, npv, f1, partial_auc_high_spec, f2_score, f05_score, tn, fp, fn, accuracy, balanced_accuracy, prevalence, g_mean, dor, markedness, youden_j, sensitivity, specificity, ppv, alert_f1, mcc, cohens_kappa, lr_positive, lr_negative, net_benefit_default
Epidemic shape (9) | peak_week_err, peak_int_relerr, epi_peak_mae, epi_season_total_mae, lead_time_weeks, attack_rate_relerr, growth_rate_corr, epidemic_duration_err, season_onset_err
Significance, skill & ranking (21) | cost_skill_3to1, cost_skill_5to1, cost_skill_10to1, dm_z_stat, dm_p_value, dm_z_vs_climatology, dm_p_vs_climatology, dm_z_vs_lag52, dm_p_vs_lag52, dm_p_value_bh, dm_p_vs_climatology_bh, dm_p_vs_lag52_bh, skill_mae_vs_persist, skill_wis_vs_persist, skill_crps_vs_persist, skill_mae_vs_snaive, relative_wis_pairwise, rank_wis, rank_log_wis, rank_mae, rank_r2
Residual diagnostics (8) | ljung_box_q, ljung_box_p, residual_acf_lag1, shapiro_wilk_p, jarque_bera_p, durbin_watson, residual_skew, residual_kurtosis
Bookkeeping & identifiers (5) | n_test, n_valid, phase_id, champion_best_wis, champion_eligible
### TABLE 8
Metric | NegBin-V7 | Poisson | ARIMA | Theta | SARIMAX | SARIMA
WIS | 3.250 | 3.410 | 14.98 | 17.71 | 18.66 | 19.00
log-WIS | 0.112 | 0.118 | 0.624 | 1.312 | 1.784 | 2.600
CRPS | 3.302 | 3.440 | 13.75 | 13.78 | 14.64 | 14.04
R2 | 0.928 | 0.917 | -0.372 | -0.820 | -1.006 | -0.985
RMSE | 6.968 | 7.489 | 30.45 | 35.06 | 36.81 | 36.62
MAE | 4.202 | 4.345 | 19.78 | 23.65 | 25.00 | 26.41
MAPE % | 15.95 | 16.30 | 61.70 | 64.37 | 69.17 | 86.93
sMAPE % | 16.41 | 16.95 | 71.99 | 106.2 | 120.5 | 165.6
MASE h1 | 2.338 | 2.418 | 11.01 | 13.16 | 13.91 | 14.70
PI95 cov | 0.971 | 0.971 | 0.971 | 0.971 | 0.971 | 0.971
PI95 width | 35.01 | 35.64 | 137.8 | 153.8 | 161.9 | 163.2
PIT KS p | 1.000 | 0.999 | 1.000 | 1.000 | 1.000 | 1.000
Calib slope | 4.582 | 4.268 | 1.486 | -1.168 | -4.099 | -0.195
AUC-ROC | 0.995 | 0.997 | 0.485 | 0.483 | 0.199 | 0.341
AUPRC | 0.998 | 0.999 | 0.704 | 0.708 | 0.525 | 0.627
C-index | 0.924 | 0.922 | 0.595 | 0.603 | 0.286 | 0.465
Brier | 0.045 | 0.041 | 0.210 | 0.220 | 0.254 | 0.255
Alert F1 | 0.978 | 0.966 | 0.844 | 0.000 | 0.000 | 0.192
Peak-wk err | 0.000 | 1.000 | 46.00 | 59.00 | 26.00 | 1.000
Lead wk | -1.000 | -1.000 | 0.000 | - | - | 1.000
Dir acc | 0.672 | 0.687 | 0.507 | 0.522 | 0.448 | 0.224