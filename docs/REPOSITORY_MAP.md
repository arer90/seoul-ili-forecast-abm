# Repository map

Where things live, and — just as usefully — which parts of the layout are confusing and why.

The tree grew over a year of thesis work, so it is not uniformly tidy. Rather than reorganise it and
risk breaking paths that produced published results, this document explains it as it is. Every count
below was measured from the shipped tree, not estimated.

---

## Top level

| Path | What it is |
|---|---|
| `simulation/` | The single code root. Everything importable lives here. |
| `scripts/` | Standalone research and operations scripts, run directly, not imported. |
| `tests/` | Repository-level tests (there is a second suite inside `simulation/tests/` — see below). |
| `web/` | Next.js dashboard and the LLM advisory API routes. |
| `paper/` | Figure and table source scripts for the thesis. |
| `docs/` | Documentation. `docs/internal/` holds working notes; `docs/history/` holds decision lineage. |
| `run_pipeline.sh`, `run_resume_phase12.sh` | Training entry points. |

---

## Inside `simulation/`

Grouped by what they do rather than alphabetically, with the measured file count.

### The pipeline itself

| Directory | `.py` | Role |
|---|---|---|
| `pipeline/` | 39 | The R1–R12 research track and P1–P5 production track. `phases.py` is the single source of truth for phase labels. |
| `models/` | 95 | The forecasting registry — every model class, plus the registry that discovers them. |
| `ensembles/` | 5 | Ensemble combiners. |
| `analytics/` | 28 | Metrics, conformal prediction, statistical tests. |
| `verifier/` | 7 | Epidemiological validity gates (mass conservation, Rt bounds). |

### Data

| Directory | `.py` | Role |
|---|---|---|
| `collectors/` | 43 | API collectors, one module per source group. **`collectors/legacy/` is the active set** despite the name — `orchestrator.py` imports from it dynamically. |
| `database/` | 7 | SQLite access: `safe_connect`, bulk insert, schema, the DuckDB read overlay. |
| `data/` | — | The database file location (not shipped) and the API-key example file. |
| `cache/` | 1 | Feature cache helpers. |

### Simulation and advisory

| Directory | `.py` | Role |
|---|---|---|
| `abm/` | 53 | The agent-based metapopulation: agent kernel, contact network, behaviour layer, ABC-SMC and EnKF calibration. |
| `sim/` | 11 | The deterministic compartmental stepper the ABM builds on. |
| `llm_compare/` | 24 | The ARIA advisory layer — LLM backends, grounding, multi-agent orchestration. |
| `server/` | 13 | The MCP server exposing the pipeline as tools. |

### Support

| Directory | `.py` | Role |
|---|---|---|
| `cli/` | 12 | The `python -m simulation <cmd>` command handlers. |
| `utils/` | 19 | Shared helpers (device selection, artifact bundling, doctor). |
| `scripts/` | 175 | Analysis and figure scripts scoped to the package. |
| `tests/` | 270 | The main test suite. |
| `benchmarks/`, `tools/`, `tracking/` | 2 / 3 / 1 | Benchmarks, small utilities, experiment tracking. |
| `r_verification/` | 9 `.R` | Independent R cross-checks of the key statistics — stationarity, residual diagnostics, the Diebold–Mariano test, WIS/CRPS/PIT, negative-binomial dispersion, Rt via EpiEstim, ITS attribution, TBATS/ETS baselines. Not Python; run separately in R. |
| `c/`, `rust/` | 3 / 4 | Optional native accelerators. Both fall back to Numba automatically if unbuilt. |
| `results/` | — | Analysis outputs backing the paper: metrics, checkpoints, SHAP, training history. |

---

## Known confusions

These are the parts that will trip you up. They are documented rather than fixed because renaming
them would change import paths that produced published results.

### `db/` is not `database/`

`simulation/database/` is the real module. `simulation/db/` is a **tombstone**: its `__init__.py`
does nothing but raise

```
ImportError: simulation.db was removed in . Use `from simulation.database import ...` instead.
```

It exists so that any surviving caller of the old path fails loudly instead of silently importing
something stale. The other three files inside it are leftovers with no importers. (The empty version
string in that message is a pre-existing cosmetic bug, not a truncation.)

### There are two `scripts/` directories

| | Count | Character |
|---|---|---|
| `scripts/` (top level) | 115 `.py` + 14 `.sh` | Research one-offs and operations scripts, including the shell entry points. |
| `simulation/scripts/` | 175 `.py` | Analysis and figure scripts that import the package. |

The split is not clean — both contain `_`-prefixed experiment scripts. If you are looking for a
script and it is not in one, check the other.

### There are two test suites

`tests/` at the top level and `simulation/tests/`. The top-level one holds repository-level guards
(for example the check that the thesis numbers match the shipped results); `simulation/tests/` holds
the package's own unit tests.

**Run tests one file at a time.** A whole-suite run segfaults on macOS through the LightGBM/OpenMP
interaction, and without the database several ABM test files cannot even be collected — see
[SETUP.md §6](../SETUP.md).

### `collectors/legacy/` is current

The directory name suggests otherwise, but `simulation/collectors/orchestrator.py` imports these
modules by computed name at runtime. Treat them as live code.

### Modules are loaded by computed name

Several directories are swept dynamically — `simulation.models.{name}`,
`simulation.collectors.legacy.{name}`, and others. A static "who imports this?" search will report
false negatives for models and collectors. Check the dynamic import sites before concluding a module
is unused.

---

## Where results come from

| Output | Produced by |
|---|---|
| `simulation/results/per_model_eval/` | R10 evaluation — the 124-metric battery per model |
| `simulation/results/per_model_optimal/` | R9 optimisation — the selected config per model |
| `simulation/results/checkpoints/` | Per-phase checkpoints (`checkpoint_R1.json` …) |
| `simulation/results/shap/` | R11 post-hoc explanation |
| `simulation/results/abm_*` | ABM validation, calibration, counterfactual and ablation runs |
| `web/public/aggregates/` | Aggregates the dashboard reads |

Figures are **not** shipped as images — the scripts under `paper/` and `simulation/scripts/`
regenerate them from the result data above.
