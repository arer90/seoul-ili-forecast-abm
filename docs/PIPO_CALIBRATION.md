# PIPO Calibration Report — Seoul Influenza Models

> Calibration reporting per the **PIPO (Purpose–Inputs–Process–Outputs)** 16-item
> framework (Dankwa et al. 2025, *PLOS Comput Biol*; the reporting standard for
> transmission-dynamic model calibration). Addresses the external-review
> recommendation to report calibration completely. Code: `simulation/abm/
> calibrate.py`, `simulation/sim/`, the forecasting registry, `identifiability.py`,
> `sensitivity.py`. See [ABM.md](ABM.md), [ODD_PROTOCOL.md](ODD_PROTOCOL.md).

> The four PIPO dimensions are mapped below; the full 16-item checklist is in
> Dankwa et al. (2025) — every reportable item is addressed or marked as a stated
> limitation.

---

## P — Purpose
- **Goal**: (a) calibrate the 53-model probabilistic ILI forecasters to weekly
  per-gu ILI; (b) calibrate the metapop SEIR-V-D + four-parameter behavioral
  layer to a real Seoul wave; (c) **assess identifiability** of the behavioral
  parameters (when does behavior add value).
- **Target outputs**: weekly ILI per 1,000 (city + 25 gu), peak timing, attack
  rate, and — for interventions — counterfactual incidence/deaths.
- **Intended use**: spatially-targeted decision support (not individual diagnosis).

## I — Inputs
- **Calibration data**: KDCA `sentinel_influenza` ILI (the legal §4 sentinel-surveillance target),
  per season/week; per-gu covariates (mobility, weather, administrative).
- **Model**: metapop SEIR-V-D (commuter-coupled FoI) + behavioral α/κ/τ/θ rule;
  the forecasting registry (count-regression / ML / DL / mechanistic).
- **Fixed vs calibrated parameters**:
  - *Fixed from data*: district populations, commuter matrix M, school affiliation,
    KNHANES comorbidity prevalence, generation/serial interval.
  - *Calibrated*: transmissibility β (+ seasonal forcing), behavioral α/κ/τ/θ;
    forecaster hyper-parameters (Optuna).
- **Parameter ranges / priors**: explicit bounds (e.g. SA ranges in
  `sensitivity.py`: β 0.5–1.3, θ 0.3–0.7, α 0.1–0.6); R0 anchored to the seasonal-
  influenza median R ≈ 1.28 (Biggerstaff et al. 2014).

## P — Process
- **Algorithm**: forecasters via Optuna (TPE) with walk-forward CV; SEIR/behavioral
  via likelihood-aligned optimization (`calibrate.py`).
- **Objective / likelihood**: forecasting champion selected by **weighted interval
  score (WIS)** (Bracher et al. 2021), not R²; SEIR fit on incidence + peak timing
  + cumulative incidence jointly.
- **Constraints**: epidemiological **validity gate** (Rt ∈ [0.3, 8], seasonal
  phase, S+E+I+R+V+D = N conservation) rejects invalid fits.
- **Train/validation/test split**: leakage-controlled per-fold re-computation;
  **within-season / holdout-season** and **leave-one-district-out** evaluation
  (`within_season_validation.py`, `stratified_validation.py`).

## O — Outputs
- **Calibrated values**: champion forecaster + parameters; behavioral α/κ/τ/θ
  estimates.
- **Calibrated-output uncertainty** (the item most often omitted):
  - forecasts: conformal / CQR prediction intervals with **coverage** reporting;
  - behavioral parameters: **per-parameter profile-likelihood confidence intervals**
    (Raue et al. 2009) via `identifiability.py` (`_ci_width`, `profile_nd`) —
    finite CI = identifiable, flat = non-identifiable;
  - stochastic spread: Monte-Carlo **ensemble** percentile CIs + variance
    stabilization (`sensitivity.py`, Lee et al. 2015).
- **Goodness-of-fit**: WIS, R², RMSE, MAE, PI coverage (the 129-metric battery),
  with **honest negative findings** reported.
- **Validation taxonomy** (ISPOR-SMDM TF-7): face validity (epi gate), internal
  verification (conservation/positivity tests), cross-validity (sim-vs-observed,
  P3), predictive validity (holdout season).
- **Global sensitivity**: LHS + PRCC over the parameter space (Marino et al. 2008)
  identifies the drivers (β +0.92, behavioral α −0.83 on attack rate).

---

## Reproducibility
Seeds fixed; deterministic ODE + seeded tau-leap; config hashes (`config_sha256`)
and run manifests recorded; calibration ranges and objective are versioned in
code. Data provenance: `epi_real_seoul.db` (KDCA/KOSIS/Seoul open data), per
[ARIA.md](ARIA.md) §5 for the legal/surveillance grounding.
