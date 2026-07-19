# General Performance Optimization Framework

> **Extracted from `docs/_archive/SPRINT_FRESH_TRAIN_20260527_v2/SPRINT_REVIEW.md` §17 on 2026-05-27 per Round 4 audit F4.**
> SPRINT_REVIEW.md now contains only sprint status; the general performance framework lives here.

This section specializes the *general performance report* the user supplied alongside — universal diagnostic principles for an unspecified target — into the context of this sprint (ILI forecasting + distributed training pipeline). It synthesizes the official performance-efficiency axes of AWS/Azure/Google Cloud + Brendan Gregg's USE method + the three OpenTelemetry signals + the Google "Tail at Scale" principles.

### 17.1 5-Axis Performance Framework (AWS Well-Architected Performance Efficiency Pillar + Azure + GCP)

> **NB (Round 4 audit F4 correction, 2026-05-27)**: The AWS Well-Architected Framework currently has **6 pillars** (Operational Excellence + Security + Reliability + Performance Efficiency + Cost Optimization + Sustainability), and the pillar this framework maps to directly is the **Performance Efficiency Pillar**. The earlier V1-V3 notation "5-pillar" is stale (the Sustainability pillar was added in 2021-12). The 5 axes of this framework are *diagnostic axes* distinct from the 5 design principles of the Performance Efficiency Pillar (Democratize advanced technologies / Go global in minutes / Use serverless architectures / Experiment more often / Consider mechanical sympathy).

| Axis | Application in this sprint | Key indicators |
|------|--------------|----------|
| **Architecture selection** | 53 model × 12 category × Phase 1-18 dispatch order (RENUMBER 2026-05-28) | model registry coverage, phase dispatch order |
| **Compute / Hardware** | Mac MPS + CPU fallback, n_jobs≤2, OPTUNA_ISOLATE=1 | %CPU, %MEM, MPS thermal throttle, OOM events |
| **Data management** | DB 12GB + cache + DuckDB ATTACH READ_ONLY | bulk_insert chunk_size=20_000, cache parquet, WAL mode |
| **Networking / Content** | Pinf (inference) DB write (forecast_*), external dataset read | bulk insert latency, DB lock contention |
| **Process / Culture** | Sprint task tracker (33 task), audit chain, OSF preregistration | task completion rate, audit Round 1→2→3 progression |

### 17.2 USE Method (Brendan Gregg) — per-resource application in this sprint

| Resource | Utilization | Saturation | Errors | Monitor in this sprint |
|------|------------|-----------|--------|----------------|
| CPU | top, ps | run queue, throttling | (none) | `top -l 1 -o cpu` |
| Memory | %MEM | swap, page fault | OOM kill | `top -o mem`, kube events |
| Disk I/O | iostat %util | await, queue depth | I/O errors | `df -h .`, iostat |
| Network | NIC %util | dropped packets | retransmit | (DB only) |
| **Optuna trial** | n_running trials | queue depth | trial pruned/failed | study.trials_dataframe() |
| **DB connection** | active connections | pool wait | timeout | pg_stat_activity equivalent |
| **GPU (MPS)** | MPS util | thermal throttle | OOM | activity monitor |

### 17.3 OpenTelemetry 3-signal application in this sprint

| Signal | Use in this sprint | Tooling |
|--------|--------------|------|
| **Traces** | R9 (per_model_optimize) trial → fit → predict span | (not applied in the current sprint — OTel adoption recommended as follow-up) |
| **Metrics** | 129 metric per model (g175 5 keys removed, 134→129, 2026-06-05) + bootstrap CI | metric_eval.compute_full_metrics |
<!-- KO: g175 binding → 129 metric (4-criteria/g175 abolished). EN: g175 binding removed; 129 metrics (was 134; 5 g175 keys deleted 2026-06-05) -->
| **Logs** | Phase elapsed log + audit warning | `/tmp/training_resume_*.log` + simulation/logs/ |

Recommended future extensions (responding to the audit critique's point about insufficient observability):
- `simulation/observability/otel.py` new — span manager for R9 (per_model_optimize) trial / R10 (per_model_eval) binding / mc_filter
- Prometheus histogram (n_round per WF-CV, trial duration distribution)
- Grafana dashboard for `freshness_lag`, `queue_age`, `trial_pruned_rate`

### 17.4 Google "Tail at Scale" application in this sprint

**Dean & Barroso (2013) Communications of the ACM 56(2):74-80** — at fleet scale, **p99 / p99.9** rather than mean latency dominates the user experience:

> **NB (Round 4 audit F4 disclaimer, 2026-05-27)**: Dean & Barroso 2013 "The Tail at Scale" originally describes the statistical dynamics of *warehouse-scale* (1000+ machine) tail latency (the multiplicative effect of single slow component × N machines → SLA degradation). This sprint's single-machine MPS / CUDA setup has no warehouse-scale dynamics — we borrow only the "micro-fault budget" framing (1 slow trial → the whole training run is delayed). **The principle ports, but the scale does not** — this sprint's p99 trial duration is not equivalent to warehouse-scale p99 service latency (no quantitative comparison, qualitative framing only ✓).

| Measurement in this sprint | Tail-at-scale perspective |
|--------------|-------------------|
| 53 model × Optuna trial | trial-level p99 duration (1 slow trial delays the whole training run) |
| R9 (per_model_optimize) hierarchical preproc (100 trial) | p99 worst path of the hierarchical chain |
| WF-CV 10 round | round-level p99 (1 slow round triggers a timeout for the entire trial) |
| Mac MPS thermal throttle | throttle frequency under sustained workload (Notebookcheck Cinebench variation 43%) |

**Mitigations (response to audit HIGH-2)**:
- `MPH_LIGHTNING_MAX_TIME_PER_MODEL=1800` (G-152) — blocks a timeout on a single model
- `MPH_SEEDS_MIN=2` fallback (additionally recommended in V3) — avoids thermal cascading from 5-seed multi-seed runs
- HyperbandPruner — automatically prunes slow trials

### 17.5 Bottleneck candidates in this sprint (Round 3 audit + general framework integrated)

| Area | Dominant cause | Measurement | Recommended action (Stage) |
|------|-----------|------|-----------------|
| Training time | R9 (per_model_optimize) hierarchical preproc 100 × main HP 50 × 53 model × n_round 10 | trial duration p95/p99 | Stage 2: HyperbandPruner tuning + bounded joblib n_jobs=2 |
| Memory | bootstrap CI n=1000 × 4 metric × 53 model | R10 (per_model_eval) inline-computation memory | Stage 3: share pre-computed bootstrap indices (~400KB saved) |
| MPS thermal | 60-100h sustained workload | thermal throttle frequency | Stage 2: MPH_SEEDS_MIN=2 fallback (HIGH-2) |
| DB I/O | bulk_insert chunk_size=20_000 + DuckDB overlay | bulk insert latency | OK (G-120 applied) |
| R4 (WF-CV) reopt | scaffolded only (HIGH-9) | no unit test | Stage 2: unit test + explicitly mark as disabled |
| External API | (none) | (no external dependency) | OK |

### 17.6 ROI-based Priority Score (framework from the user's general report)

Priority computation for this sprint's Round 3 audit fixes:

\[
PriorityScore = 20 \times (0.20 \times Cost^{-1} + 0.15 \times Difficulty^{-1} + 0.30 \times Improvement + 0.15 \times Risk^{-1} + 0.20 \times ROI)
\]

| Audit item | Cost⁻¹ | Diff⁻¹ | Improve | Risk⁻¹ | ROI | Score | Rank |
|-----------|--------|--------|---------|--------|-----|-------|------|
| CRITICAL-1 remove Murphy 1973 | 5 | 5 | 4 | 5 | 5 | **94** | 1 (V3 ✓) |
| CRITICAL-2 mc_filter train leak test | 5 | 4 | 5 | 5 | 5 | **93** | 2 (V3 ✓) |
| HIGH-5 FluSight persistence baseline | 4 | 3 | 5 | 4 | 5 | **84** | 3 (V5 G1 ✓) |
| HIGH-1 multi_seed narrative | 5 | 5 | 3 | 5 | 4 | 84 | 4 |
| HIGH-3 OSF retrospective disclosure | 5 | 4 | 3 | 4 | 4 | 76 | 5 |
| HIGH-9 R4 (WF-CV) reopt unit test | 4 | 3 | 4 | 4 | 4 | 74 | 6 |
| HIGH-8 raise cohens_d to 0.5 | 4 | 4 | 3 | 3 | 4 | 70 | 7 |
| HIGH-7 OverseasTransfer covariate | 3 | 3 | 4 | 3 | 4 | 67 | 8 |
| HIGH-2 2-seed minimum mode | 4 | 4 | 3 | 4 | 3 | 66 | 9 |
| HIGH-6 R²/MAPE cutoff calibration-in-the-large | 3 | 3 | 4 | 3 | 4 | 67 | 10 |
| HIGH-4 PROBAST self-assessment caveat | 5 | 5 | 2 | 5 | 3 | 64 | 11 |

→ **Stage 1 V3 fix (CRITICAL 1+2) complete**, **V5 G1 fix (FluSight baseline registered) complete**.

### 17.7 Training launch checklist (immediately before Stage 4, V3 consolidated)

| # | Item | Verification |
|---|------|------|
| 1 | DB 12GB intact | `stat -f "size=%z" simulation/data/db/epi_real_seoul.db` |
| 2 | requirements.lock 202 packages | `wc -l requirements.lock` |
| 3 | CITATION.cff Murphy 1973 removed | `grep -c "Murphy" CITATION.cff` = 0 active reference |
| 4 | mc_filter train leak test 12/12 PASS | `python simulation/tests/test_mc_filter_stage3.py` |
| 5 | KDCA threshold helper season auto-detect | `python -c "from simulation.analytics.kdca_threshold import detect_current_season_threshold; print(detect_current_season_threshold())"` |
| 6 | conditional_calibration family="continuous" default | `grep "family.*continuous" simulation/analytics/conditional_calibration.py` |
| 7 | multi_seed.py Bergstra primes naming fix | `grep -c "arbitrary values" simulation/analytics/multi_seed.py` ≥ 1 |
| 8 | 23 sh active exports + audit 9 = 32 | `grep -c "^export" run_resume_phase12.sh` |
| 9 | preflight check | `bash scripts/preflight_check.sh` exit 0 |
| 10 | Optuna DB fresh (after archiving) | `ls simulation/results/optuna_*.db 2>/dev/null \| wc -l` = 0 |
| 11 | FluSight-Baseline registered (R4 G1) | `python -c "from simulation.models.registry import CATEGORY_MODELS; assert 'FluSight-Baseline' in CATEGORY_MODELS['ts']"` |

→ When all are ✓, training launch is possible.

---
