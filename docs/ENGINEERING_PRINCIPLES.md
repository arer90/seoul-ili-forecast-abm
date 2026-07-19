# MPH Infectious Disease Simulation Project

> 🧭 **New here, or lost? → [`docs/START_HERE.md`](REPOSITORY_MAP.md)** (one-page navigation: current status, per-task routing, absolute rules, document map). This ENGINEERING_PRINCIPLES.md = the rules SSOT (principles); START_HERE = where to find what.

**Goal**: Seoul ILI rate multi-model forecasting + Metapop SEIR-V-D + ARIA LLM Layer
**Source-of-truth**: `docs/internal/INDEX.md` (routing) + `docs/MASTER_REFERENCE_20260529.md` (full synthesis) + `docs/FINAL_INSPECTION_20260529.md` (inspection checklist)
**History / transition / retirement records**: `docs/history/` — this file holds **current rules only**. For why and when something was decided, see history.

---

## One-page summary

```
simulation/                ← single code root
├── __main__.py            python -m simulation <cmd> (25 cmd_* re-import from cli/)
├── cli/                   25 cmd_* handlers — 9 modules
│   ├── _scenarios.py      SCENARIOS + ALL_MODELS (49 active = CATEGORY_MODELS mirror)
│   ├── db_commands.py · maintenance_commands.py · utility_commands.py · sim_commands.py
│   ├── data_commands.py · pipeline_commands.py · inference_commands.py · training_commands.py
├── data/db/               epi_real_seoul.db (85 tables)
├── database/              safe_connect · bulk_insert · DuckDB overlay
├── collectors/            group_*.py + import + extract_pdf
├── models/                49 active (final lineup) / 66 registered (17 deferred/retired) — live count SSOT = verify_registry_coverage() · 53→49 = G-319f(NegBinGLM-V7)·G-323(EARS×3) · retirement/reduction history → docs/history/
│                          foundation = TimesFM-2.5(Google) + TiRex(NX-AI xLSTM) — both transformers-free (G-261/265); avoids the Chronos⊥mlx-lm conflict
│                          added (measured to beat the incumbent in a SOTA survey): NegBinGLM-Glum(G-263, 0.878)·TabPFN(G-264, 0.917)·DLinear(G-265, rolling 0.935)·TiRex(G-265, rolling 0.944, best)
├── pipeline/              active = research track(R1-R12) + production(P1-P5). phases.py = SSOT (label ↔ name).
│                          R1 data → R2 baseline → R3 external → R4 WF-CV → R5 diagnostics → R6 DM →
│                          R7 intervals → R8 scoring →
│                          R9 per_model_optimize (feature selection happens only here: actual order preproc→mc→STABILITY feature→HP — mc→feature keeps small-sample OOF stable; mc = cheap pre-pass, HP = study inside the refit) →
│                          R10 per_model_eval(129-metric) → R11 SHAP/XAI → R12 comprehensive (consumes only the R9/R10 champion and families; real_eval decoupled 2026-06-20) →
│                          P1 real_forecaster (operational rolling-origin 1-step; starts only after the entire R-track finishes = start of the production track. Uses the R9 champion + ABM/ARIA deployment gate) →
│                          P2 family_deploy·P3 ABM·P4 ARIA·P5 web = production track. inference·overseas = separate CLI.
│                          Files/functions use meaningful names (data.py/baseline.py/per_model_optimize.py, run_<name>). R/P ↔ name SSOT: simulation/pipeline/phases.py
├── server/                MCP epi server + epimas_adapter.py
└── scripts/               audit/retrain/compare/launch runner (champion = **G-339 LEAK-FREE**: within the OOF 1-SE band, fold stability → parsimony → OOF-WIS, **hold-out test not used for selection**; the old G-318 test-shortlist = winner's curse, discarded; pure OOF-argmin G-307 = overfitting)
```

> R2~R8 (baseline~scoring) = **BASIC eval features** (13 lag + seasonality features, `MPH_EVAL_FEATURES=basic`); only R9 (per_model_optimize) uses the full pool.
> Label ↔ name SSOT = `simulation/pipeline/phases.py` (R = research / P = production track).

---

## Quick start

```bash
.venv/bin/python -m simulation db-status        # environment check
.venv/bin/python -m simulation collect --groups all
bash scripts/launch_full_run.sh                  # full training run (clean+detached, PPID=1, preflight gate)
bash scripts/audit_and_retrain.sh                # post-training audit + retrain + compare
```

---

## Routing (`docs/internal/INDEX.md` + `paper/INDEX.md` are the main entry points)

Before starting work, consult the INDEX tables and load only the 1-2 MD files you need.

| Frequency | First load |
|------|---------|
| **Before every task (top-level self-verification)** | **ENGINEERING_PRINCIPLES.md §"Five design discipline principles" — D-1~D-5 (Grill / Ubiquitous / TDD / Deep / Gray-box)** |
| **Before every code change (agent behavior)** | **ENGINEERING_PRINCIPLES.md §"Four coding-agent behavior principles" — K-1~K-4** |
| **Domain term lookup (D-2)** | **`docs/internal/ubiquitous_language.md`** |
| 5-second check on every task | `docs/internal/gotchas_active.md` (§"Quick check" + Critical-Recent) |
| **Interrupting and restarting training** | **Full run = `bash scripts/launch_full_run.sh` (clean+detached, PPID=1). ⚠ Partial resume (skipping completed models) = `--no-clean --resume-from N` — `--no-clean` alone still re-runs phases 1-12 (it only skips the archive step). Code verification: `run_pipeline.sh:63`, G-315 / gotchas item 14** |
| **Avoiding wasted time (G-150)** | **Before changing a training path, inspect every caller per site (no mechanical bulk changes)** |
| New analysis / report | `docs/internal/role.md` |
| Code / pipeline work branching | `docs/internal/INDEX.md` (topic → file lookup, includes the **principle** column) |
| Thesis / Figure / Table work | `paper/INDEX.md` |
| **Why/when of earlier decisions, deprecation and retirement history** | `docs/history/` |

---

## Five design discipline principles (D-1 ~ D-5)

> **Top layer — the working method itself**. Apply in the order `D-1 → D-2 → D-3 → D-4 → D-5` before starting any task.
> Then `K-1 → K-4` (agent behavior), and finally `#1 → #5` (engineering). (Decision sources/dates → `docs/history/`)

### D-1. Grill Me — design-tree dependency interview (interview-driven design)
- **Trigger = automatic, on every user message**. Include 1-3 sharp questions.
- **Interview the user relentlessly**: walk down each branch of the design tree and **resolve the dependencies between decisions one at a time**.
- **No silent assumptions** (reinforces K-1) — ask explicitly: "Is X an assumption here?" "Is the intent of Y A or B?"
- **Question format**: a table (options A/B/C + impact + time) so the user can answer with a single character.
- **Exemptions**: an obviously single interpretation = grill exempt
  - status queries: "What's the status?", "How far along?", "What are the results?"
  - clear commands: "kill <PID>", "next", "keep going", "do it that way"
  - simple sub-tasks: "show me the log", "fix this (diagnosis already done)"
  - **immediately after the user has answered the previous grill** (dependencies already resolved)
- **Stopping rule**: once all dependencies are resolved → confirm with "Here's my understanding (A/B/C). Proceed?" before writing code.

### D-2. Ubiquitous Language — unified domain vocabulary (DDD, Eric Evans)
- **Use a single vocabulary**: code / MD / logs / commit messages / test names all refer to the same concept with the same word.
- **Repository**: `docs/internal/ubiquitous_language.md` — every domain term + definition + aliases (Korean/English) + code location.
- **No aliases**: do not mix `multi-criteria`, `multi_criterion`, `4-criteria filter` for the same concept — **one definition** only.
- **Translation policy**:
  - Domain terms = Korean with English alongside (e.g. "예측 구간 PI" / prediction interval PI)
  - Code identifiers = English (`pi95_coverage`)
  - MD body text = **follow each file's existing language** (Korean-dominant files stay Korean, English-dominant files stay English)
  - Docstrings = keep the existing language (back-compat)
- **Conflict handling**: **grep immediately and unify everything** — on finding an alias, do a 1-2 hour sprint to replace all occurrences.
- **Procedure for adding a new term**: ① register it in ubiquitous_language.md → ② use it in code/MD/tests → ③ search for existing aliases and deprecate them.

### D-3. TDD — Test-Driven Development (Red → Green → Refactor)
- **Strictness policy (mixed)**:
  - **New features = strict TDD**: commit the failing test first → then the fix commit (the Red → Green → Refactor order is enforced)
  - **Bug fixes = test-after smoke**: write the helper, then validate on a 6-8 case sample — adding the reproduction case retrospectively is OK
- **Red**: before writing a new feature, first write a **failing test** (including the reproduction case).
- **Green**: write the **minimum** code that makes the test pass (fused with K-2 simplicity).
- **Refactor**: improve the code while keeping the tests green.
- **Smoke test standard**: matched / mismatch / empty / NaN / edge, 6-8 cases, and keep the test after it goes to production (regression guard).
- **A G-### entry is itself a TDD pattern**: incident (red) → permanent fix (green) → smoke test (refactor verification). Every G-### states its reproduction case.
- **Triggers**: bug fix / new helper / new model class / new feature. Simple typos and log-message changes are exempt.

### D-4. Deep Module — small interface, rich implementation (Ousterhout)
- **Scope**: every existing shallow module is a refactoring target, and new code must be a deep module from the start.
- **Reject shallow modules** (thin wrappers where the interface ≈ the implementation in size): zero abstraction value.
- **Deep module standard**: small interface (1-3 functions) + large implementation (hundreds of lines, encapsulating complex logic).
- **Examples**:
  - `simulation.database.safe_connect()` ✓ deep (1 function; WAL + retry + corruption check + thread-local, 200+ lines)
  - `evaluate_predictions_full(y_test, y_pred, …)` ✓ deep (1 function, the 129-key R10 SSOT computation, ~750 lines; 128 metrics + phase_id)
  - `_validate_shapes(X, y, X_test, name, min_n)` ✓ deep (1 function, 6 validation rules, 70 lines)
  - **Counter-example**: `def get_value(d, k): return d.get(k)` ✗ shallow — do not use.
- **Wrapping pattern**: when abstracting across several layers, every layer must be deep — no pass-through wrappers.
- **"Information hiding"** (Parnas 1972): the caller only needs to know the interface, and internal changes have zero impact on callers.

### D-5. Gray-Box Delegation — partly white + partly black (appropriate abstraction)
- **Docstring standard strictness (enforced for all public functions)**: every public function (one that does not start with an uppercase letter or `_`) states its contract — Google-style:
  ```python
  def func(arg1: T, arg2: T2) -> RT:
      """One-line summary.

      Args:
          arg1: meaning + unit + range
          arg2: same

      Returns:
          return format + shape

      Raises:
          ValueError: condition under which it is raised

      Performance: O(n) time, ~100MB peak memory
      Side effects: writes to disk / DB / global state (be specific)
      Caller responsibility: enforce domain constraints such as y ≥ 0
      """
  ```
- **Reject black boxes** (full abstraction): impossible to debug, no memory/performance guarantees.
- **Reject white boxes** (full exposure): internal changes break every caller.
- **Gray-box standard**: ① a clear interface (contract), ② **explicitly guaranteed** performance / memory / side effects, ③ the internal implementation stays encapsulated.
- **State the guarantees when delegating**: "this helper guarantees `n_jobs ≤ 2`" (G-049) · "`subprocess.Popen` isolation — child memory is 100% reclaimed" (G-158) · "shape mismatch → ValueError in 0 seconds" (G-166). The user must know the gray parts (performance/memory/contract) to use it safely.

> **Self-verification procedure**: D-1 (grill) → D-2 (terminology) → D-3 (test first) → D-4 (deep) → D-5 (gray-box) → K-1~K-4 → #1~#5 → `gotchas_active.md` quick check.

---

## Four coding-agent behavior principles (Karpathy)

> **Below D-1~D-5, above the 5 engineering principles (agent behavior layer)**. A filter every code/MD change must pass.
> Core: *"models make wrong assumptions and run along without checking"* + *"overcomplicate / bloat abstractions"*.

### K-1. Think Before Coding (state assumptions + ask when unsure)
- **No silent assumptions** — state them explicitly. When multiple interpretations are possible, surface them and let the user choose.
- **Do not guess at data** — verify with `PRAGMA table_info` / `grep` / `Read` before writing code (avoids G-002, G-131).
- **When uncertain, run a sample test** — edit 1 file, verify with a smoke test, and confirm with the user before a full sweep.

### K-2. Simplicity First (only what was asked)
- **No speculative features or unnecessary abstractions** — "in case we need it later" is banned. YAGNI is enforced.
- **The doubt test**: would a senior engineer call this "overcomplicated"? → If yes, rewrite it.
- Fused with principle **#4 KISS**: single source-of-truth + single entry point + single DB.

### K-3. Surgical Changes (change only the requested code, clean up only orphans)
- **No refactoring of adjacent code** — modify only the file/function the user asked about (G-150 = the same principle).
- **Preserve the existing style** — match indentation / naming / comment patterns.
- **Do not remove dead code** (only on explicit user request) — remove only the imports/variables your change left unused.
- **Use `replace_all` carefully**: outside of renames, use limited Edits with unique context.

### K-4. Goal-Driven Execution (verifiable success criteria)
- **"fix the bug" → test first, then confirm it passes** — every fix carries a reproduction case + verification.
- **State the multi-step plan + checkpoints** — specify in advance how each step will be verified.
- **Smoke test first** — after adding a helper, validate on a 6-8 case sample before applying it in production.
- Models are good at *"looping until goals"* — give clear success criteria, not imperative commands.

> **Self-verification procedure** (before changing code): K-1 → K-2 → K-3 → K-4 → the 5 engineering principles (#1~#5) → `gotchas_active.md` quick check.

---

## Five engineering principles (first principles)

> This section holds *principles* only. The *gotchas* (G-###) that result from violating them are already enforced in code
> (`safe_connect`/`bulk_insert`/`runner.py`/`per_model_optimize.py`) — the list lives in `gotchas_active.md`, and the incident history in `docs/history/gotchas-lineage.md`.

### 1. OS / hardware independence (portability)
- macOS (MPS) / Linux (CUDA) / Windows (CUDA·CPU) run identical code → no environment assumptions.
- `simulation.models.base.pick_device()` (cuda > mps > cpu) is automatic — code never declares the device directly.
- Use `pathlib.Path` + an explicit `encoding="utf-8"` — handles Windows cp949 / POSIX.
- Optional accelerator (Rust `seir_core`) — falls back to numba automatically when absent.
- Single entry point `python -m simulation <cmd>` — no dependence on shell-specific redirection.

### 2. Memory / CPU hygiene for long runs (6-24h)
- **Reclaim through isolation**: isolate trials/models with `subprocess.Popen` → the OS reclaims 100% of child memory (`OPTUNA_ISOLATE=1` enforced, G-158).
- **Explicit cleanup**: `del obj; gc.collect(); gc.collect()` twice + `torch.{cuda,mps}.empty_cache()` + `malloc_trim(0)` on Linux.
- **CPU restraint**: `n_jobs ≤ 2` (never -1 — causes a CPU 0% deadlock).
- **Stall defense**: subprocess returncode + adaptive `stall_timeout` (kill after 300s with no change) + `MPH_LIGHTNING_MAX_TIME_PER_MODEL=1800` (G-152).
- **WAL hygiene**: periodic `checkpoint_wal()`.
- **Preflight verification**: the training entry script calls `bash scripts/preflight_check.sh` automatically → training is blocked when an environment variable is missing (G-158).

### 3. Light load + accurate
- **Data processing**: prefer Polars (lazy / streaming) → pandas only as the sklearn bridge.
- **DB writes**: `bulk_insert(rows, chunk_size=20_000)` in a single transaction — per-row inserts rejected (5-50× slower).
- **DB reads**: analysis queries go through a DuckDB ATTACH READ_ONLY overlay — pandas full scans rejected.
- **Keep single-writer SQLite + bulk dict inserts** (SQLAlchemy ORM rejected, G-120).
- **Cache strategy**: `simulation/cache/*.parquet` (FE) + a persisted Optuna study DB → warm start on re-runs.

### 4. Simple is best (KISS — single source-of-truth)
- **Single entry point**: `python -m simulation <cmd>` — do not create scattered scripts.
- **Single DB**: `simulation/data/db/epi_real_seoul.db` — do not create other .db files.
- **Single code root**: `simulation/`.
- **One MD per topic**: follow the `docs/internal/INDEX.md` table — duplicate definitions are rejected.
- **Data-driven decisions**: textbook guesswork (hardcoded per-group mappings) is rejected → use per-feature stats + model-aware logic (G-131).

### 5. Reproducibility
- **Determinism**: fix the seeds (`np.random.seed(42)`, `torch.manual_seed(42)`).
- **OOF / WF-CV best selection**: `MPH_BEST_BY=oof_cv` — a single n=27 val split is rejected (G-132).
- **Transform safety**: log1p inverse cap `expm1(clip(x, -2, log1p(y_max × 10)))` (G-146).
- **Prediction sanitize** (G-159): `sanitize_predictions` — only NaN/None/±inf become 0.0 (legitimate values such as negatives are preserved), nonneg=False by default.
- **Shape validation fail-fast** (G-166/160): `_validate_shapes` (base.py:80) → a caller bug raises a ValueError in 0 seconds.
- **Full metric preservation** (G-168/167): the SSOT `evaluate_predictions_full` computes all **129 metrics** and preserves every key. Champion = **pure best-WIS** (R²/MAPE/PICP are reported only as individual metrics).
- **Trial cleanup callback** (G-161): `study.optimize` is required to use a cleanup callback + gc_after_trial.
- **comprehensive(R12) figures + plot honesty** (G-163/164): all 4 figure types (calibration·forest_plot_wis·heatmap·horizon_decay) are code-generated (from the metric grid); placeholders are never allowed. R12 is decoupled from real_eval (2026-06-20), so the operational real_pred belongs to P1 (real_forecaster) — R12 figures consume only the R9/R10 champion and families.
- **Anchor blend per-model α** (G-141/144): learned as an Optuna HP — a single floor value is banned.
- **Champion artifact bundle**: `.pt` = `{model + scaler + transform_state + feature_indices + meta}`.
- **Critical-stage fail-loud** (G-237): silent voids are absolutely forbidden — ① catch+mark+banner = P1·R9·R10 (real_forecaster·per_model_optimize·per_model_eval), ② loud crash = R1·R7·R8 (data·intervals·scoring). Never allow a silent `{"error":…} → "Pipeline Complete!"`.

> **A gotcha is the consequence of violating a principle**. Follow the five above and most G-### issues are avoided automatically. Before new code/analysis: verify the five → `gotchas_active.md` quick check.

---

## Training operations standard (summary)

> Full training run = `bash scripts/launch_full_run.sh` (clean+detached, preflight gate). The env vars below are exported
> automatically by `run_pipeline.sh`. For the decision source / measured evidence behind each value → `docs/history/`.

```bash
export OPTUNA_ISOLATE=1                   # trial isolation (G-158)
export MPH_BEST_BY=oof_cv                 # a single n=27 val split is banned (G-132)
export MPH_STABLE_TRANSFORMS=1            # per_model_optimize = all of STABLE_Y opened up (log1p/sqrt/asinh/laplace/mcmc), preproc = **pure grid** dynamic OOF selection (G-330→**G-335: Optuna study/sampler/pruner removed; flat grid, 7 transforms, 1 OOF run each + 1-SE + fourth_root**; ⚠ "preproc 100 trial/Optuna" is stale, pre-G-333); force-identity applies only to Poisson/hhh4/pf; explosion backstop = G-328/G-334 cap. ※ HP uses a separate Optuna study (TPE). (boxcox/yeo remain non-stable, G-133/146)
export MPH_EVAL_FEATURES=basic            # baseline~real_eval = BASIC (13 lag + seasonality features); the full pool is per_model_optimize only
export MPH_FEAT_PATH=nested               # feature guard = NESTED size-path + 1-SE (deep NNs switch to binary automatically via G-250)
export MPH_MC_PER_MODEL=1                 # mc = per-model OOF best among none/vif/corr/pca (global selection retired)
export MPH_MC_MARGIN=0.02                 # minimum relative OOF-WIS improvement over none (when ambiguous, none = overfit guard)
export MPH_LIGHTNING_MAX_TIME_PER_MODEL=1800   # Lightning final fit timeout (G-152)
export MPH_PJ_ALPHA_LO=0.0 MPH_PJ_ALPHA_HI=1.5 # anchor blend α per-model HP (G-144)
export MPH_PI_AUGMENT_LO=0 MPH_PI_AUGMENT_HI=3 # PI augment, avoids leakage (G-144)
.venv/bin/python -m simulation.scripts.cleanup_optuna_studies --threshold 100   # Optuna cleanup before training (G-143)
```

- champion = **G-339 LEAK-FREE (hold-out test not used for selection)** (2026-06-24, correction from external AI reviewer #1): the old G-318 (OOF top-8 shortlist → **hold-out test WIS argmin**) merely shrank K to 8 while still picking 1-of-8 using the test set, which **reintroduces winner's curse** (Cawley & Talbot 2010 · Varma & Simon 2006). G-339 removes the test set from selection entirely: **within the OOF 1-SE statistical-tie band (Breiman 1984), rank by fold stability (`_oof_fold_cv` = a leak-free proxy for robustness to distribution shift) → parsimony (n_features) → OOF-WIS**. The G-307 SVR-RBF OOF-noise win is absorbed by the band + stability. **The hold-out test is a diagnostic reported alongside only, via `select_champion_holdout_best`** (not for deployment); the test set is reported once, at the end. SSOT = `per_model_eval.py:select_champion_g318` (which contains the G-339 body); tooling = `scripts/rerank_champion.py` (zero retraining). Guard = `tests/test_g339_champion_leakfree.py` (leak-free · SVR-noise rejection · parsimony). There is **one champion (FusedEpi)**; NegBinGLM is **reported alongside** as an epidemiologically interpretable count model (it is not the champion). (4-criteria/g175 retired → `docs/history/metrics-registry.md`)
- Paper top-priority metric: TOP_3 = {wis, alert_f1, lead_time_weeks} (`metric_rubric.py:PAPER_TOP_*`, `docs/METRIC_TOP_PRIORITY.md`).
- Details: `docs/internal/pipeline_3stage.md` · `docs/internal/best_decision.md` · `docs/FEATURE_SELECTION_STABILITY_DECISION_20260601.md`.

---

> **This file (ENGINEERING_PRINCIPLES.md) = routing + D-1~D-5 + K-1~K-4 + the 5 engineering principles (current rules only)**. Details are in `docs/internal/INDEX.md`, history in `docs/history/`.
> **Self-verification order**: D-1 → D-2 → D-3 → D-4 → D-5 → K-1 → K-2 → K-3 → K-4 → #1~#5 → `gotchas_active.md` quick check.
