# DB → Web-App Full Pipeline — Complete Documentation of Functions, Metrics, and Criteria

> **Purpose**: Record the **end-to-end chain** from data collection to the web dashboard,
> at the granularity of functions, metrics, and decision criteria.
> **Honesty rule**: Every statement here was **verified by reading** the actual sources (`simulation/`, `web/`). Fabrication: 0.
> **Single entry point**: `python -m simulation <cmd>` — **26 commands**.
> **Source SSOT**: `docs/figures_proof/PIPELINE_FULL.md` · `simulation/pipeline/phase_evaluator.py` ·
> `simulation/pipeline/runner.py` · `simulation/collectors/orchestrator.py` · `web/scripts/build_*.py`.
> Created 2026-06-08.

---

## Table of Contents

- [§1. End-to-End Flow](#1-전체-흐름-end-to-end-flow)
- [§2. Functions per Stage](#2-각-단계의-기능-functions-per-stage)
  - [2.1 Data collection — 8 collectors](#21-데이터-수집--8개-collector-collect)
  - [2.2 DB feed & schema](#22-db-적재스키마-db-feed--schema)
  - [2.3 Forecast training — 13 semantically named phases](#23-forecast-학습--13-의미이름-phase-train)
  - [2.4 Downstream commands after champion selection](#24-champion-이후-downstream-명령)
- [§3. Metrics](#3-평가지표-metrics)
  - [3.1 Forecast 129-metric SSOT](#31-forecast-129-metric-ssot-phase_evaluatorevaluate_predictions_full)
  - [3.2 SEIR-V-D / ABM validity metrics](#32-seir-v-d--abm-타당성-지표-validity-metrics)
  - [3.3 ARIA LLM metrics](#33-aria-llm-지표)
- [§4. Criteria and Decision Rules](#4-기준판정-criteria)
- [§5. Web Wiring](#5-웹-배선-web-wiring)
- [Appendix A. What is missing from a train-only view (honesty)](#부록-a-train-only-뷰에서-빠진-것-정직성)

---

## §1. End-to-End Flow

```
collect (8 collectors)
   └─▶ DB  epi_real_seoul.db (85 tables, ~80M rows)   ◀── import-external · extract-pdf · bootstrap · db-init/status/optimize/migrate
          └─▶ train  (13 semantically named phases, run_pipeline)
                 └─▶ ★ CHAMPION (best-WIS, selected in real_eval + deployment gate)
                        ├─▶ inference     predict-real        (real-time ILI forecasting)
                        ├─▶ simulation    sim                 (SEIR-V-D 25-gu + behavioral ABM; champion→ABM forcing)
                        └─▶ overseas val. overseas-validate   (KR vs DE/FR/HK/US/JP, WHO FluNet §4.6)
                               └─▶ ARIA LLM  mcp-server     (MCP stdio, RAG+GraphRAG+grounding, 10 epi tools)
                                      └─▶ web dashboard  Next.js (web/public/aggregates/*.json + live MCP bridge)
```

**Key facts**:
- The DB has **85 tables** (measured: `sqlite_master` count = 85; the "79" in ENGINEERING_PRINCIPLES.md is stale, and this agrees with `docs/figures_proof/PIPELINE_FULL.md` §4).
- `train` is **1 of 26** commands, and the 13 phases inside it cover forecast training only.
- Every stage after the champion is produced (inference, simulation, overseas validation, ARIA, web) is an **independent CLI/package**; they share only the champion artifact
  (`models/champion_log.json` + the `.pt` bundle) or the DB (Appendix A).
- The web app is a **real-time ILI dashboard for the 25 gu of Seoul** (Next.js 14 App Router) — driven by DB→JSON snapshots plus a live MCP bridge.
  Build confirmed GREEN (`docs/figures_proof/README.md` §6).

---

## §2. Functions per Stage

### 2.1 Data collection — 8 collectors (`collect`)

Registered in `simulation/collectors/orchestrator.py:GROUP_INFO`. `run_collection` iterates over `DEFAULT_ORDER`
(20 groups). The **8 extended collectors** requested are listed below. The core domestic time series (groups E/D/S/B/C/…)
exist separately, earlier in `DEFAULT_ORDER`.

| Code | Module | Data source | Core content | Input→Output |
|------|------|--------------------------|-----------|-----------|
| **G** | `group_g_gtrends.py` | **Google Trends** (pytrends, no key) — geo KR/US/JP/EU-10 | Influenza search interest (0-100), a leading indicator for ILI | pytrends API → `gtrends_*` tables |
| **I** | `group_i_overseas.py` | **Overseas ILI** — WHO FluNet, CDC ILINet/FluSurv-NET, WHO FluID (EU-10), Sentiweb FR, ECDC ERVISS (9 sources) | Weekly overseas ILI at country level | scraper/API → `overseas_ili`, `who_flunet` |
| **J** | `group_j_population_density.py` | **Population density** — WorldBank EN.POP.DNST, US Census ACS5, Japan 2020 national census | Population density of overseas regions (a transmission-scale feature) | API → density tables |
| **K** | `group_k_weather_gu.py` | **Per-gu weather** — KMA ASOS multi-station (108 Seoul + 401/119/400) | Bracketing ASOS stations for the 25 gu of Seoul → GU-weighted weather | KMA ASOS → `weather_historical` |
| **N** | `group_n_hospital.py` | **HIRA/NEDIS hospitals** — NEMC real-time ED API + HIRA ILI claims | Weekly hospital ED burden index | API → `ed_weekly_burden`, `hira_inpat_opat` |
| **O** | `group_o_regional_ili.py` | **Regional ILI** — US (Delphi FluView/NSSP, CDC NHSN/NWSS), Japan NIID, Germany RKI, France | Sub-national (state/prefecture/Bundesland) ILI — a core feature for GU-level forecasting | API → regional ILI tables |
| **T** | `group_t_commuter_flows.py` | **Commuter flows** — US Census ACS commuting, Germany Bundesagentur Pendler, Japan e-Stat | region×region commuting OD (the mediator of epidemic spread) | API → `commuter_flows`, `commuter_matrix` |
| **W** | `group_w_overseas_weather.py` | **Overseas weather** — Open-Meteo Historical (no key) | Daily meteorology for US state capitals / Japanese prefectural capitals / EU-10 capitals (exogenous climate features) | Open-Meteo → overseas weather tables |

> **Automatic post-collection step**: on completion, `collect` automatically refreshes the web aggregates (`refresh_web_data.py`;
> opt out with `--no-web-refresh`). In other words, the JSON snapshots in §5 are regenerated from the latest DB.

### 2.2 DB feed & schema

| Command | Role | Input→Output |
|------|------|-----------|
| `db-init` | Create the schema (idempotent, `init_db`) | `--db-path` → creates tables, prints the count |
| `db-status` | Row count per table (`print_status`) | DB → table printed to stdout |
| `db-optimize` | WAL checkpoint (TRUNCATE) + optional VACUUM/ANALYZE | DB → compaction and cleanup |
| `db-migrate` | Schema additions (`apply_schema_migration`, idempotent) | DB → new tables/columns |
| `import-external` | Import WHO FluNet + metadata / KOSIS sex-disaggregated and registry data / commuter matrix | xlsx/csv → `who_flunet`, `commuter_matrix`, `kosis_*` |
| `extract-pdf` | Seoul infectious disease surveillance yearbook PDF → DB (`extract_pdf`) | yearbook PDF → `seoul_annual_report_district/monthly` |
| `bootstrap` | Empty DB → operational readiness (init → import → extract → maintain → verify → VACUUM) | empty DB → loaded and verified |

### 2.3 Forecast training — 13 semantically named phases (`train`)

`train` → `run_pipeline(config)` (`simulation/pipeline/runner.py:629`). 65 models registered (53 active), WF-CV OOF,
Optuna 3-stage (preproc→feature→HP), 129 metrics. **Dispatch order = the preserved phase numbers**; what is exposed to the
user and used for resume is the **semantic name**. Verified in `PHASE_NAME_TO_NUMBER` (`runner.py:35`).
Usage: `python -m simulation train --resume-from <name|number>`.

| Order | Semantic name (module) | No. | Role | Main function | Input→Output |
|:---:|-----------------|:---:|------|----------|-----------|
| 1 | `data` (`data.py`) | 1 | Feature building and cleaning | `run_data` / `build_enriched_features` | DB → feature matrix (train-pool / test-clean) |
| 2 | `baseline` (`baseline.py`) | 4 | Baseline on **BASIC features** (13 lag + seasonality features) | `run_baseline` | features → raw per-model predictions |
| 3 | `external` (`external.py`) | 5 | Models on non-basic (external) features | `run_external` | features → external-feature model predictions |
| 4 | `wfcv` (`wfcv.py`) | 6 | **Walk-Forward CV OOF** (the champion comparison panel) | `run_wfcv` | features → OOF predictions (chronological split) |
| 5 | `diagnostics` (`diagnostics.py`) | 7 | Residuals and diagnostics | `run_diagnostics` | predictions → residual diagnostic table |
| 6 | `dm_test` (`dm_test.py`) | 9 | Diebold-Mariano test | `run_dm_test` | prediction errors → pairwise DM z/p |
| 7 | `intervals` (`intervals.py`) | 10 | Prediction intervals (PI / split-conformal) | `run_intervals` | residuals → PI quantiles (K=11) |
| 8 | `scoring` (`scoring.py`) | 11 | Diagnostic scoring (composite ≠ champion) | `run_scoring` | metrics → overall diagnostic summary (for reporting) |
| 9 | **`real_eval`** (`real_eval.py`) | 12 | **★Real-data operational forecast (rolling-origin, 1-step) → champion = best-WIS + deployment gate** | `run_real_eval` / `_gate_forecast` | real slab → champion selection + deployment contract |
| 10 | `per_model_optimize` (`per_model_optimize.py`) | 13 | **Feature selection happens only here**: preproc→STABILITY feature→mc→HP (Optuna 3-stage, full pool) | `run_per_model_optimize` | full feature pool → per-model best-HP `.pt` |
| 11 | `per_model_eval` (`per_model_eval.py`) | 14 | Per-model **129-metric** evaluation (test slab) | `run_per_model_eval` | best-HP models → `per_model_metrics.csv` |
| 12 | `shap`/`xai` (`shap_analysis.py`) | 15 | SHAP/XAI (all families: Tree/Linear/Deep/Kernel universal permutation) | `run_shap_analysis` | models → SHAP plots + importances |
| 13 | `comprehensive_eval` (`comprehensive_eval.py`) | 16 | Overall aggregation + figures (forest/heatmap/calibration/horizon_decay) | `run_comprehensive_eval` | outputs of all phases → aggregated report |

> **Non-contiguous numbering**: phases 2, 3, and 8 are unused (retired). Active = 1·4·5·6·7·9·10·11·12·13·14·15·16
> → hence `dm_test` is 9. The empty pre-stage feature LOAD (former phases 2-3) is also retired. `inference` (17) and
> `overseas` (18) are not in the `train` dispatch and run only via their own CLIs (`predict-real` / `overseas-validate`).
> **★Evaluation feature policy**: phases 4-12 use **BASIC eval features** (13 lag + seasonality features,
> `MPH_EVAL_FEATURES=basic` by default). Feature optimization over the full pool happens only in phase 13
> (full = final fallback). (ENGINEERING_PRINCIPLES.md §Training operations standard)
> **★Critical-phase fail-loud**: real_eval (12), per_model_optimize (13), and per_model_eval (14) are in `_CRITICAL_PHASES`
> (`runner.py:415`) — on failure, `_collect_critical_failures` surfaces a `CHAMPION_GATE_FAILED` banner at the end of the run (G-237).

### 2.4 Downstream commands after champion selection

| Command | Role | Main function | Input→Output |
|------|------|----------|-----------|
| `predict-real` | **Real-time inference** with the champion artifact. Builds the same feature matrix as training → slices the inference window → loads the ChampionArtifact (model+scaler+transform_state+feature_indices) → replays the training-time pipeline | `cmd_predict_real` (`inference_commands.py:16`) | DB (latest) → `results/inference/<ts>/` (predictions.csv, inference_metrics.json, champions_used.json, REPORT.md) |
| `sim` | **Metapop SEIR-V-D + behavioral ABM**. Default = **forecast-anchored ABM** (basis = the operational champion = the real_eval best_model; champion forecast → ABM forcing). Use `--scenario` to opt out into a fixed scenario, `--anchor-forecast <model>` to change the basis, and `--allow-gate-bypass` to disable the epi-validity gate | `cmd_sim` (`sim_commands.py:15`) / `run_forecast_anchored` / `MetapopSEIRVD` | champion forecast → trajectories (`forecast_anchored.json`/`.npz`) + epi-validity gate result |
| `overseas-validate` | **Overseas cross-validation** (thesis §4.6). Applies the Seoul model to US/JP/DE/FR/HK/KR with identical features and metrics. KR = internal baseline, the rest = external generalizability | `pipeline/overseas.py` (subprocess) | DB (`overseas_ili`) → `results/phase18_overseas_{ts}/` (cross-country R²/RMSE/MAPE/WIS) |
| `mcp-server` | **The ARIA LLM advisory layer** — an MCP (JSON-RPC 2.0) stdio server. `EpiMCPServer` exposes **10 epi tools**. RAG + GraphRAG grounding, read-only SQL guard, static citations | `cmd_mcp_server` / `run_stdio_server` (`server/mcp_epi.py`) | stdin ndjson → stdout ndjson (queries / forecasts / read-only DB access / evidence citations) |

**The 10 ARIA epi tools** (`simulation/server/mcp_epi.py:10-19`):

| # | Tool | Function |
|---|------|------|
| 1 | `epi.query_db` | read-only DuckDB SQL over `epi_real_seoul.db` (SQL guard) |
| 2 | `epi.forecast` | ensemble point + 95% PI for a gu |
| 3 | `epi.model_compare` | Diebold-Mariano tests by regime |
| 4 | `epi.shap_features` | Top-N SHAP for a (gu, week) |
| 5 | `epi.rt_estimate` | EpiEstim bayesian Rt sliding window (Cori 2013) |
| 6 | `epi.lead_time_analysis` | skill vs horizon for a model |
| 7 | `epi.outbreak_detect` | EARS-C1 / CUSUM flagging |
| 8 | `epi.validity_check` | run epi-validity gate on a claim (§4) |
| 9 | `epi.literature_rag` | vector-RAG over project PDFs |
| 10 | `epi.scenario_run` | metapop SEIR-V-D run (Stage 5) |

---

## §3. Metrics

### 3.1 Forecast 129-metric SSOT (`phase_evaluator.evaluate_predictions_full`)

**SSOT**: `simulation/pipeline/phase_evaluator.py:46` `evaluate_predictions_full(y_test, y_pred, …)` —
a single function computes **129 metrics** (on 2026-06-05 the 5 g175 4-criteria keys were removed, 134→129). Every
evaluation phase that has predictions (baseline 4 → WF-CV 6 → intervals 10 → real_eval 12 → per_model_eval 14 → overseas 18)
calls it with the **same shape**, which makes phase-trajectory comparison possible. All computation lives in exactly 1
place, this function (a deep module, D-4).

| Group | Representative metrics | Location (lines) | Public-health epidemiological meaning |
|------|-----------|-----------------|-----------------|
| **Point error (~11)** | R², RMSE, MAE, MSE, MAPE, sMAPE, MdAPE, bias, MSLE, Theil's U | `phase_evaluator.py:149-171` | Accuracy of the central forecast. R² = explained variance, MAPE = relative error |
| **Scale-free (MASE, 5)** | MASE_h1/h4/h13/h26/h52 (Hyndman 2006) | `:173-186` | Accuracy relative to a seasonal naive baseline (multiple horizons) |
| **Threshold/alert (24)** | sensitivity, specificity, PPV, NPV, **F1=alert_f1**, F2, balanced_acc, MCC, Cohen's κ, Youden J, **lead_time_weeks** (epi-curve), DOR, LR+/LR− | `:188-239` | Classification of whether the KDCA alert threshold (8.6‰) is exceeded — the direct currency of public-health decision making |
| **Empirical PI (~10)** | pi99/95/80/50_coverage, _width, **PICP** (=pi95_coverage), pi95_relia, pi_sharpness_ratio, pi95_ci_lo/hi (Wilson) | `:241-292` | Uncertainty quantification. Actual coverage against the nominal 95% (calibration) |
| **Probabilistic (WIS family)** | **WIS** (weighted interval score), log_WIS, wis_sharpness/underpred/overpred decomposition, **CRPS**, pinball_q05/q95 | `:294-339` | **The operational standard of FluSight / the COVID-19 Forecast Hub**. The single criterion for champion selection (§4) |
| **Probabilistic classification (Brier/ROC/Calib)** | brier_score, **brier_skill** (WOY climatology baseline), Brier decomposition (reliability/resolution/uncertainty), roc_auc, auprc, partial_auc_high_spec, calibration_slope/intercept, PIT (mean/std/ks_p) | `:341-445` | Calibration and discrimination of alert probabilities (Murphy 1973, Czado 2009 PIT) |
| **Epi-curve (12)** | peak_week_err, peak_int_relerr, epi_peak_mae, **attack_rate_relerr**, growth_rate_corr, **lead_time_weeks**, season_onset_err, epidemic_duration_err, pearson_r, spearman_r, c_index (Harrell) | `:447-492` | Epidemic peak timing, magnitude, growth rate, and lead time — the core of epidemiological decision making |
| **Cost skill (3)** | cost_skill_3to1/5to1/10to1 | `:494-504` | Utility weighted by the FN (missed alert) to FP cost ratio |
| **Residual diagnostics (8)** | shapiro_wilk_p, jarque_bera_p, residual_skew/kurtosis, residual_acf_lag1, durbin_watson, ljung_box_q/p | `:506-550` | Residual normality and autocorrelation (model adequacy) |
| **Multi-model (NaN→phase14)** | DM vs persist/climatology/lag52 (+BH-FDR), skill_*_vs_persist/snaive, relative_wis_pairwise, rank_wis/log_wis/mae/r2, bootstrap CI (mae/wis ci95) | `:567-768` | NaN for a single model; filled in by the phase-14 multi-model post-loop (Sherratt 2023 pairwise WIS) |

> **Fast mode**: with `MPH_FAST_METRIC=1` or `MPH_FULL_EVAL_TRAJECTORY=0`, a skip-marker dict is returned
> (`phase_evaluator.py:107`). Default = full (all 129 keys computed in every phase).
> **Default threshold 8.6‰** = the 2024-25 KDCA alert threshold (`:54`).

### 3.2 SEIR-V-D / ABM validity metrics

| Metric | Measured value | Location | Meaning |
|------|--------|-----------|------|
| **Mass conservation rel-err** | **1.9e-16** | `verify_all.py:17` (≤1e-9 threshold), `epi_validity.py:68` COMPARTMENT_TOL=1e-6 | \|S+E+I+R+V+D − N\|/N — population conservation. 10¹⁰× margin against the threshold (machine precision) |
| **two-engine rmse** (kernel ↔ ABM city-I) | **4.5e-11** | E2E §6, `MetapopSEIRVD` vs `agent_kernel` | Cross-validation of two independent engines (Python and Rust), agreeing to machine precision |
| **Rt range** (reproduction number) | [0.3, 8.0] | `epi_validity.py:184` `check_rt_sequence` | Below 0.3 = extinction, above 8 = beyond observed influenza (Cori 2013, with a safety margin) |
| **\|ΔRt\| continuity** | ≤ 1.5 | `epi_validity.py:67` RT_DELTA_CAP | Blocks abrupt week-to-week jumps (a signal of data defects) |
| **Seasonal peak window** | W48–W8 | `epi_validity.py:70` KOREAN_FLU_SEASON_WEEKS | Korean influenza seasonality — a peak in months 7-8 (July-August) is non-physical |
| **ABM hysteresis loop area** | **+19.78, p≈0.000** (ON) / 0.000, p=1.000 (OFF) | *(source not distributed — see note)* | Behavior-ON produces path dependence (epidemic memory); OFF is a flat control |
| **ABM per-season R²** | 0.34–0.71 (best 2023-24 = **0.712**) | *(source not distributed — see note)* | rolling-origin operational forecast, city-level fit |
| **ABM anchored WIS / corr** | **2.872** / **0.846** | *(source not distributed — see note)* | Consistency of champion forecast → ABM forcing (FluSight WIS) |

> **Rows marked *(source not distributed)*.** These figures come from result files that exist in
> the working repository but are not part of this distribution, so a reader cannot open them to
> check the number. They are retained because the pipeline that produces them ships — running it
> regenerates the files — but until then treat them as reported, not verifiable here. Every other
> row in §3.2 and §3.3 cites a file you can open in this checkout; `scripts/check_doc_citations.py`
> enforces that.

### 3.3 ARIA LLM metrics

**SSOT**: `simulation/results/aria_grounding_multi_llm.json` — the only ARIA evaluation artifact in
this distribution. Two contexts drawn from real thesis outputs (ABM forward validation and champion
metrics), scored for numeric grounding and Self-Ask decomposition across 7 backends.

> Rewritten 2026-07-19. The previous table cited `llm_compare_latest/factual_report.json` and
> `aria_grounding.json` — neither ships here — and quoted accuracy figures for Gemini-CLI, which the
> run that produced the surviving artifact **skipped** (`gemini quota unavailable … daily quota
> exhausted`). It also named backends (Gemma-3-12B, Llama-8B, Qwen3, DeepSeek-R1) that this run
> never evaluated. Everything below is read out of the shipped file.

| Backend | Tier | numeric fact_recall | numeric faithfulness | spurious claims | Self-Ask faithfulness |
|---|---|---|---|---|---|
| **Claude CLI** | cli | **1.000** | **0.928** | 2 | 0.688 |
| Mistral-7B | ollama | 0.800 | 0.757 | 1 | 0.646 |
| Codex (OpenAI GPT) | cli | 0.650 | 0.817 | 2 | **0.771** |
| Llama3.2-1B | ollama | 0.425 | 0.752 | 3 | 0.728 |
| Phi3.5-3.8B | ollama | 0.350 | 0.550 | 1 | 0.742 |
| Gemma3-1B | ollama | 0.350 | 0.562 | 0 | 0.695 |
| Qwen2.5-3B | ollama | 0.163 | 0.291 | 0 | 0.576 |

**How to read this.** `fact_recall` is the share of the context's numbers a backend reproduced
correctly; `faithfulness` penalises claims not supported by the context. The CLI-vs-local gap is
real but narrower than the withdrawn table implied — Mistral-7B (local) beats Codex (CLI) on
fact_recall. Claude's 1.000 reflects a **two-context corpus**, not a general capability claim.

**Limits worth stating.** Gemini was not evaluated (quota). Self-Ask `mean_n_subq` is identical
(6.5) for every backend because it is a harness constant, not a measured property of the replies —
it is omitted from the table for that reason. With n=2 contexts, no significance test on these
differences would be meaningful, and none is claimed.

---

## §4. Criteria and Decision Rules

### 4.1 Champion selection = pure best-WIS

- **Definition**: the champion is the **best-WIS** model on the rolling-origin 1-step operational forecast in `real_eval` (phase 12).
- **Environment variable**: `MPH_BEST_BY=oof_cv` (single validation on n=27 is forbidden — avoids the G-132 catastrophic trap, ENGINEERING_PRINCIPLES.md §Reproducibility).
- **Decision of 2026-06-05 (explicit user instruction)**: the **4-criteria (g175) filter was removed entirely**. There is no
  composite flag, gate, tier, or promise_score requiring R²/MAPE/WIS/PICP95 to pass simultaneously. **WIS is the sole champion
  criterion**; R²/MAPE/PICP exist only as individual metrics. (Code: `phase_evaluator.py:743`, the 5 g175 keys deleted, 134→129.)
- The web `trained-models.json` follows the same rule — `build_trained_models.py:62` `sorted(rows, key=wis)` ascending (lower = better).

### 4.2 Deployment gate (reject a volatile champion → fallback)

`real_eval.py:461` `_gate_forecast(pred, y_train, fallback, k=3.0)` — separates **best-WIS for evaluation** from **deployment safety**.

- **Contract (all conditions must hold)**: finite ∧ nonneg ∧ `pred ≤ k·max(y_train)` (k=3, lenient) ∧
  `max|Δpred| ≤ q99.5(|Δy_train|)` (must not exceed the worst historical step).
- **On violation**: the champion forecast is **rejected (not clipped)** and **replaced** by the stable **fallback**
  (`median_ensemble`, `real_eval.py:1064`). The champion is retained for evaluation only, and the replacement is reported
  loudly with a count and a reason (G-237).
- **Background**: an eval-best champion can extrapolate-collapse on the real slab (real R² = −2684; pred ≈ 1007 vs observed ≈ 21),
  so the collapse is blocked before it reaches the ABM/ARIA downstream (`real_eval.py:469-473`).

### 4.3 Epi-validity gate (physical and epidemiological plausibility of sim)

`simulation/verifier/epi_validity.py` `run_epi_validity_gate` — runs automatically with `sim` (disable with `--allow-gate-bypass`).

| Gate | Criterion | Basis |
|--------|------|------|
| Population conservation | \|S+E+I+R+V+D − N\|/N ≤ **1e-6** | COMPARTMENT_TOL (`epi_validity.py:68`) |
| Rt range | each Rt_t ∈ **[0.3, 8.0]** | `check_rt_sequence` (`:184`); Cori 2013 + a pandemic safety margin |
| Rt continuity | \|Rt_{t+1} − Rt_t\| ≤ **1.5** | RT_DELTA_CAP (`:67`) |
| Seasonal peak | peak ∈ **W48–W8** | KOREAN_FLU_SEASON_WEEKS (`:70`) |
| ILI upper bound | max ≤ **100‰** | biological upper bound (`:138`) |
| Parameter ranges | R0∈[0.5,4.0], σ∈[1/4,1.0]/d, VE∈[0.10,0.95], ifr∈[0.0001,0.05] | literature values (`:54-61`, Biggerstaff/Carrat/CDC) |

### 4.4 Training feature and evaluation policy

- **Feature selection happens only in phase 13 (per_model_optimize)** — preproc → STABILITY feature → mc → HP (Optuna 3-stage,
  full pool, full = final fallback). No other phase performs feature selection.
- **Phases 4-12 = BASIC eval features** (13 lag + seasonality features, `MPH_EVAL_FEATURES=basic` by default) — the champion gate (12) is BASIC too.
- **mc (multicollinearity) = per-model by default** (`MPH_MC_PER_MODEL=1`) — each model applies whichever of the ④ candidates measures best on OOF (G-242).
- **Reproducibility**: seed np = torch = 42, deterministic GLM/ODE, bit-identical with max|Δ| = 0 (E2E §2).

---

## §5. Web Wiring

The web app is driven by two paths: a **DB→JSON snapshot layer** and a **live MCP bridge layer**. Next.js 14 App Router.

### 5.1 Snapshot layer — `web/scripts/build_*.py` → `web/public/aggregates/*.json`

The `build_*.py` scripts (~16 of them) read the DB (or training outputs) and emit static JSON. `refresh_web_data.py` refreshes them all at once (automatically after `collect`).

| JSON (aggregates/) | Generating script | Source | Content (actual keys) |
|--------------------|---------------|------|---------------|
| **trained-models.json** | `build_trained_models.py` | `per_model_eval/per_model_metrics.csv` | `{version, total_models:53, source, metric_hint:"WIS asc", top:[{rank,name,r2,rmse,wis,crps,cov95,mape,family}], all}` — **ranked by ascending WIS** (`:62`) |
| **seir-metapop-init.json** | `build_seir_*` / static-aggregates | DB population and commuting | `{district_names, populations, mobility_flat, n_gu, source}` — WASM SEIR-V-D initialization |
| **abm-scenarios.json** | `build-static-aggregates.py` | output of ABM scenarios.py | `{gu_names, days, scenarios}` — what-if player scenarios |
| **latest-choropleth.json** | static-aggregates | latest ILI by gu | `{metric, disclaimer, rows}` — 25-gu choropleth (includes the disclaimer that per-gu data is limited to 2024) |
| **disease-vax.json** | `build_disease_vax.py` | DB disease and vaccination data | Incidence and vaccination coverage per disease |
| **commuter-edges.json** | static-aggregates | `commuter_matrix` | Commuting OD edges (flow animator) |
| **weather.json** / **air-env.json** | `build_weather.py` / `build_air_env.py` | `weather_historical` and others | Exogenous weather and air-quality overlays |
| **live-overlays.json** / **realtime-poi.json** | `build_live_overlays.py` / `build_realtime_poi.py` | real-time APIs | Live overlays and POIs |
| geojson (boundary/subway/bus) | `build_seoul_boundary.py` / `build_subway_*` / `build_bus_*` | boundaries and transit | Map layers |

In addition, `export-turso.py` exports a DB subset to a libSQL (Turso) seed (`turso_seed.sql`) to support server-side live queries.

#### 5.1.1 §web-champion — research champion vs web operational champion (SSOT, evidence-sync #4 2026-06-27)

> **One line**: the research champion (R10 best-WIS) is **FusedEpi**, while the web operational forecast (`ili-forecast.json`) uses **NegBinGLM (V6)**. This is a **difference in operational choice**, *not* an indication that FusedEpi cannot produce forward forecasts.

| Axis | Research champion | Web operational champion |
|----|----------------|-----------------|
| Model | **FusedEpi** (TiRex+TabPFN fusion + NegBin/mc/mechanistic/dynamic α/conformal) | **NegBinGLM** (V6 salvage = RidgeCV+log1p) |
| Selection criterion | **R10 best-WIS** = 3.278 (reported alongside the interpretable NegBinGLM); `per_model_eval.py:select_champion_g318` (G-339) | **summary_metrics.csv rank-1 test_R²** = 0.9085 (interpretable epi-GLM) |
| Outputs | `simulation/results/per_model_eval/per_model_metrics.csv` (48 models) + `comprehensive_eval/REPORT.md` (regenerated) | `web/scripts/build_production_forecast.py` → `ili-forecast.json` (full in-sample refit → 1-step into the future) |
| Forward capability | **Yes** — `expanding_multihorizon/result.json` (53 origins, recursive leak-free h=1..28; latest origin 2026-06-14, in_sample_n=354 → FusedEpi h=1 = **4.70/1k**, PI [0.0, 9.53], actual=null = genuinely the future) + `abm_forward_validation/result.json` (forward_R²=0.722) | Yes (NegBinGLM full-data refit → future row) |
| ABM/ARIA anchor | **FusedEpi** (champion forecast → ABM forcing, ARIA grounding) | — |

**Why the web uses NegBinGLM (honest framing)**: `build_production_forecast.py` is a separate production-track CLI that produces the operational forecast by refitting an interpretable epi-GLM on the full data (fast and deterministic). FusedEpi (the R10 WIS champion) is **not incapable of forward forecasting** — as the expanding-origin and ABM-forward outputs above demonstrate, it already produces forward forecasts from a single-champion refit — it is simply that the web production script is wired to use NegBinGLM.

**Wiring the web consistently to FusedEpi (optional, requires user approval)**: either replace `build_production_forecast.py` with a single `FusedEpiForecaster` refit, or convert the latest-origin forward output of `expanding_multihorizon` (4.70/1k) into the web aggregate format; then web = ABM = ARIA = the research champion, all FusedEpi consistently. ⚠ This is not a retraining of all 48 models (a single-champion refit, a few minutes). evidence-sync #4 (2026-06-27) went only as far as **leaving the numbers unchanged and documenting the divergence honestly** — the wiring swap is deferred.

### 5.2 Live layer — MCP bridge (ARIA)

- `web/scripts/mcp-bridge.ts` (a Node HTTP gateway, dependency-free, run with `tsx`, node≥20):
  1. Spawns `python -m simulation mcp-server` (the ARIA stdio MCP) **1 time only** (`PY_ARGS="-m simulation mcp-server"`)
  2. Correlates request and response ids
  3. `POST /rpc` — receives JSON-RPC 2.0 requests from the Next.js `lib/mcp-client.ts` → invokes the 10 epi tools
  4. `POST /report` — produces docx/pdf/pptx/xlsx output via `epi.generate_report`
- Default port 8787 (`MCP_BRIDGE_PORT`). Next.js Edge routes cannot spawn Python, so this Node process acts as the relay.
- `web/lib/validity.ts` — the web calls `epi.validity_check` (the §4.3 gate) → displays Rt / peak I / final D / VE verification after the fact.

### 5.3 Frontend (Next.js)

A 25-gu choropleth plus 5 simulators: WASM SEIR-V-D (`lib/seir-wasm`), the 53-model forecaster, the what-if player,
the commuter-flow animator, and the LLM advisor (ARIA). Build confirmed GREEN (`docs/figures_proof/README.md` §6, `next build`).
<!-- KO: 66-model → 53-model (2026-06-08, 53-roster live, 0 retired -pf) -->

---

## Appendix A. What is missing from a train-only view (honesty)

| Item | Included in `train`? | Actual location |
|------|:---:|-----------|
| Data collection (8 collectors) | ✗ | `collect` (separate) |
| DB feed and schema | ✗ | import-external/extract-pdf/bootstrap/db-* (each separate) |
| Forecast 13-phase training | ✓ | `train` → `run_pipeline` |
| Champion selection (best-WIS) | ✓ | `real_eval` (phase 12) inside `train` |
| Real-time inference | ✗ | `predict-real` (separate, consumes the champion artifact) |
| SEIR-V-D + behavioral ABM | ✗ | `sim` (separate, champion→ABM forcing) |
| Overseas cross-validation | ✗ | `overseas-validate` (separate) |
| ARIA LLM advisory | ✗ | `mcp-server` (separate, MCP stdio) |
| Web dashboard | ✗ | `web/` (Next.js, DB→JSON + live MCP bridge) |
| OOF ensemble tournament | ✗ | `orchestrate` (separate, takes OOF files as input) |
| Full lifecycle automation | ✗ | `run-all` (bootstrap→collect→db-optimize→train-all) |
| 10 maintenance and utility commands | ✗ | doctor/maintain/prune/auto-update/visualize/rehydrate/list-models/feature-importance/verify-audit/freeze-paper-primary |

**In short**: everything after the champion is produced — inference, simulation, overseas validation, ARIA, and the web —
consists of independent CLIs/packages that share only the champion artifact (or the DB). `train` is just 1 of the 26 commands.

---

*Evidence files: `simulation/pipeline/runner.py` (phase map, critical phases) · `phase_evaluator.py` (129 metrics) ·
`real_eval.py` (champion + gate) · `verifier/epi_validity.py` (Rt/conservation gates) · `cli/{sim,inference}_commands.py` ·
`server/mcp_epi.py` (10 tools) · `collectors/orchestrator.py` (GROUP_INFO) · `web/scripts/build_trained_models.py` ·
`web/scripts/mcp-bridge.ts` · `web/lib/validity.ts` · `docs/figures_proof/{PIPELINE_FULL,README,MODEL_ROSTER_53}.md`.
The 85-table DB count is measured. Honesty note: Appendix A states explicitly that the post-champion stages are separate CLIs.*
