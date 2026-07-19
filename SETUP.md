# SETUP — add your API keys and the database builds itself

This repository ships **all source code, analysis outputs and paper assets**, but **not the database**
(`epi_real_seoul.db`, ~13 GB). The underlying data comes from public agency APIs, so
**you supply your own keys, run the collectors once, and the same database is rebuilt locally.**

---

## 1. Environment

This project is **[uv](https://docs.astral.sh/uv/)-based** and requires **Python ≥ 3.12**
(`pyproject.toml`: `requires-python = ">=3.12"`, with a `[tool.uv]` section).

### Recommended — with uv

```bash
git clone https://github.com/arer90/seoul-ili-forecast-abm.git
cd seoul-ili-forecast-abm

uv venv --python 3.12                      # create .venv with a uv-managed Python
uv sync                                    # resolve from pyproject.toml
```

Or, for a bit-exact reproduction of the published environment:

```bash
uv pip install -r requirements.lock
```

### Alternative — standard venv + pip

```bash
python3.12 -m venv .venv
source .venv/bin/activate                  # Windows: .venv\Scripts\activate
pip install -r requirements.lock
```

> ### ℹ `uv sync` vs `requirements.lock`
> **Both work. They give you different things.**
>
> `uv sync` resolves from `pyproject.toml` and gives you a current-compatible environment. It used to
> break this project: several packages the ARIA layer imports — `openai`, `ag2`, `dspy`, the
> `langchain-*` set, `tantivy`, `datasets`, `krippendorff`, `sbi`, `mlx-lm` — had been installed
> ad-hoc with `pip` and were never declared, so `uv sync` correctly pruned them as unrequested and
> the layer stopped importing. **They are all declared now**, so `uv sync` produces a working
> environment. `mlx-lm` carries a `darwin`/`arm64` marker, which keeps Linux and Windows resolution
> clean.
>
> `requirements.lock` pins the exact versions that produced the published numbers. Use it when you
> want the results to match bit-for-bit rather than merely to run.
>
> One thing `uv` does not handle either way: the optional Rust accelerator, built separately with
> maturin (see below).

> ### ℹ The Rust accelerator is optional
> The `-e ./simulation/rust` line in `requirements.lock` builds the Rust accelerator (`seir_core`).
> **Installation fails at that line if you have no Rust toolchain.** Either install Rust
> (<https://rustup.rs>) or comment that line out — without the accelerator the code
> **falls back to Numba automatically** and results are unchanged.

All commands below use `.venv/bin/python`.

---

## 2. Add your API keys (either method)

### Method A — key file (recommended)

```bash
cp simulation/data/api_key.example.txt simulation/data/api_key.txt
# open it and paste your key after each label
```

**Keep the labels exactly as they are.** `_KEY_LABEL_MAP` in `simulation/database/config.py`
maps those labels to internal names (`seoul_general`, `kosis`, …).

### Method B — environment variables (these override Method A)

```bash
cp .env.example .env
# fill in the values, then
set -a; source .env; set +a
```

### Which keys, and where to get them

| Tier | Data | Sign-up |
|------|------|---------|
| **Required** | Seoul living population, commercial activity | <https://data.seoul.go.kr> |
| **Required** | Seoul subway ridership | same portal (subway key) |
| **Required** | Seoul air quality | same portal (air-quality key) |
| **Required** | Korea Public Data Portal (many services) | <https://www.data.go.kr> |
| **Required** | KOSIS population and statistics | <https://kosis.kr/openapi> |
| **Required** | KMA weather observation and forecast | <https://apihub.kma.go.kr> |
| Recommended | NEIS school calendars | <https://open.neis.go.kr> |
| Recommended | Public holidays (KASI) | data.go.kr — Special Day Information |
| Recommended | KDCA notifiable disease counts | data.go.kr — KDCA |
| Optional | Overseas comparison (Japan e-Stat, US Census) | e-stat.go.jp / api.census.gov |
| Optional | ARIA LLM backends | Anthropic / OpenAI / Google / Ollama |

Any collector whose key is missing is skipped; the rest still run.
The **core Seoul ILI pipeline** needs only the six “Required” entries.

### Things worth knowing

1. **You need two Seoul Open Data general keys.**
   Write the `일반인증키(인구/상점)` (general authentication key — population/commercial activity)
   label on **two lines** in `api_key.txt`: the first becomes
   `seoul_general`, the second `seoul_general2`. Collectors fall back
   `seoul_subway → seoul_general → seoul_general2`, so two distinct keys avoid rate limits.
   `seoul_general2` has **no environment variable** — it can only be set in the file.
2. **The two KMA keys are different.**
   `KMA_HUB` (apihub.kma.go.kr — observation and forecast) and `KMA_API_KEY` (data.go.kr ASOS,
   six per-district stations) are **different keys from different portals**, despite the similar names.
3. **One data.go.kr key, but per-service approval.**
   KDCA notifiable disease, HIRA, emergency care, hospital information and vaccination each require
   their own 활용신청 (usage request) before that collector works.
4. **Some sources need no key at all** — the KDCA dportal sentinel ILI signal (the core target),
   WHO FluNet, Open-Meteo, CDC and ECDC. **The core dependent variable is collectable without any key.**
5. **Known gap.** The `google_search_trends` feature is only obtainable through the unofficial
   `pytrends` scraper, for which no API key exists. If you are rate-limited (HTTP 429) there is no
   supported fallback — run without that feature or source it separately.

---

## 3. Build the database

```bash
# 0) check the environment (the database file is created here if absent)
.venv/bin/python -m simulation db-status

# 1) collect everything (calls the agency APIs — this takes a while)
.venv/bin/python -m simulation collect --groups all

# 2) check again
.venv/bin/python -m simulation db-status
```

- Path: `simulation/data/db/epi_real_seoul.db` — schema SSOT in
  `simulation/database/schema.py` (85 tables).
- Collecting the raw hourly bus ridership table (`monthly_bus_hourly`, ~78 M rows) is what makes the
  database reach **13 GB**. To keep it small, skip that group and set
  `include_monthly_bus_hourly=False` in the feature builder.
- Subsets: `--groups a,b,c`.

---

## 4. Run the pipeline

```bash
# full training run (clean + detached, preflight-gated)
bash scripts/launch_full_run.sh

# post-run audit, retrain and compare
bash scripts/audit_and_retrain.sh
```

The single entry point is `python -m simulation <cmd>`; see `python -m simulation --help`.

---

## 5. What is included and what is not

**Included**

- Full source code (`simulation/`, `scripts/`, `tests/`, `web/`)
- Analysis outputs backing the paper (`simulation/results/` — metrics, figures, checkpoints,
  SHAP, training history)
- Paper assets (`paper/` — figure and table sources, presentation decks)
- **All PowerPoint sources**
- Documentation (`docs/`, `ENGINEERING_PRINCIPLES.md`, `API_KEYS_LAYOUT.md`)

**Deliberately excluded**

| Item | Why | Recovery |
|------|-----|----------|
| `simulation/data/db/*.db` (13 GB) | Size, and avoiding source-data redistribution | Section 3 above |
| `simulation/data/collected/**` (raw CSVs) | Individual files exceed 100 MB | The collectors re-fetch them |
| `models/*.pt` (2.9 GB) | GitHub's 100 MB per-file limit | Regenerated by training |
| `simulation/results/_archive_fullrun_*` | Archives of superseded runs | Not needed |
| Thesis manuscript (docx/pdf) | Pending degree conferral and library submission | Released later |
| `api_key.txt`, `.env` | **Secrets** | Create your own |
| `.venv/`, `node_modules/`, `rust/target/` | Build artifacts | Reinstall / rebuild |

---

## 6. Known behaviour before the database exists

This distribution intentionally omits the database, so the following are **expected** until you
complete section 3.

| Symptom | Cause | What to do |
|---------|-------|------------|
| Eight `simulation.abm.*` modules fail to import<br>(`RuntimeError: expected 25 Seoul gu names from DB, found 0`) | `simulation/abm/synthetic_population.py` **reads the database at module load** to get the 25 district names, so the import itself fails without data. | Resolved once the database exists. Until then, do not import the ABM modules. |
| `pytest tests/` aborts collection<br>(`Interrupted: 12 errors during collection`) | The import failure above happens during **test collection**, stopping the whole run. | Run per file: `pytest tests/test_docx_numbers_match_results.py -q`. macOS needs per-file runs anyway (LightGBM/OpenMP). |
| `simulation rehydrate` exits with `FileNotFoundError: models/champion_log.json` | `simulation/utils/rehydrate.py` guards the read but **not the write**, and never creates `models/`, which is absent because model weights were excluded. | `mkdir -p models` and re-run. |
| `visualize`, `predict-real` exit with `sqlite3.OperationalError: no such table` | Deliberate fail-loud policy on an empty database (silent failure is banned). | Resolved once the database exists. |

**The test suite, measured on this checkout (301 files, no database, no API keys):**

| | | |
|---|---|---|
| **237** | run in CI and pass | on Linux, Windows and macOS |
| 29 | need the database or an excluded result file | the symptoms in the table above |
| 16 | need a package outside the light CI set | usually torch |
| 10 | pass with the full environment, not the light one | |
| 9 | contain no test functions | |

CI runs **everything except a measured exclusion list**
(`scripts/ci_test_exclusions.txt`), so a new test file is covered the day it lands
rather than when someone remembers to register it. Re-measure the list after an
environment change:

```bash
python scripts/ci_run_tests.py --list      # what would run
python scripts/ci_run_tests.py --survey    # re-measure the exclusions
python scripts/ci_run_tests.py             # run them
```

Eleven defects were fixed on 2026-07-19 while wiring this up. Four were real: the
hysteresis detector returned zero for a textbook loop and degraded silently
without scipy; the Rust-vs-NumPy comparison demanded agreement finer than float32
can represent; SHAP counted all-zero attributions as measured explanations; and
ARIA crashed on the shipped `ranking.json` because it read a list of dicts as a
list of strings. The rest were tests still asserting a superseded configuration —
model counts of 51 or 53 against a lineup that is now 45, a transform set from
before `fourth_root`, a sampler-seed policy that had been reversed, and models
since moved to `DEFER_MODELS`. Each now asserts the invariant rather than the
literal, so the next documented lineup change does not read as a regression.

> **Every tracked path is ASCII and space-free**, enforced by
> `check_portability.py --only filename`. This is not cosmetic: `git ls-files`
> quotes non-ASCII names as `"\352\260\220..."` unless `core.quotepath` is false,
> which is a per-machine setting, so tooling that parses that output sees a
> different file set on a CI runner than on the machine a baseline was measured
> on. Two Korean-named files under `simulation/data/external/` were renamed or
> removed on 2026-07-20 for this reason.

> **Windows note.** Python decodes and encodes with the locale encoding when a
> call omits `encoding=`, which is cp1252 on Windows and UTF-8 elsewhere. Every
> such call site this repository actually executes has been fixed and is pinned by
> `simulation/tests/test_no_reached_default_encoding.py`, but 256 unexercised ones
> remain in the source (baselined in CI). If you hit a `UnicodeDecodeError` on
> Windows, `set PYTHONUTF8=1` makes Python use UTF-8 regardless of locale and is a
> safe workaround; please report the file so the site can be fixed properly.

**Works without a database**: `db-status`, `doctor`, `list-models`, `sim --list-scenarios`,
`--help`, and the result validator below.

```bash
python scripts/validate_results.py     # standard library only — no install needed
```

This is the check to run first on a fresh clone. It confirms the shipped results are present,
parseable and correctly shaped; that every model is ranked on a leak-free out-of-fold WIS; and that
the numbers quoted in the README trace back to a result file. It runs on Linux, Windows and macOS
in CI (`.github/workflows/reproducibility.yml`).

---

## 6.1 What reproduction actually guarantees

Worth being exact about, because the four links in the chain have different strengths.

| Link | Status | What limits it |
|------|--------|----------------|
| **Install** | Reproducible | `requirements.lock` pins the exact versions that produced the published numbers. `uv sync` gives a working but current-compatible environment. |
| **Database rebuild** | Reproducible in structure, not bit-for-bit | The collectors rebuild the same schema from the same public APIs, but those APIs revise history: KDCA sentinel figures are provisional and get corrected for several weeks, and back-fill windows differ per source. A rebuild today yields the same tables, not identical row values. |
| **Training** | Mostly reproducible given the same database — see §6.3 | Per-model seeding, not global. GPU kernel non-determinism drifts across hardware, and the three foundation models fetch weights from a remote hub at run time (TabPFN via `hf_hub_download`, TimesFM via `from_pretrained`, TiRex via `load_model("NX-AI/TiRex")`), so an upstream checkpoint change moves their output. |
| **Published numbers** | Verifiable, not re-derivable in CI | No hosted runner can re-run the pipeline — the database is not distributed and a full run needs a GPU and many hours. `validate_results.py` verifies the shipped results are intact and self-consistent instead. |

The honest summary: **the code and the method reproduce; the exact decimals depend on when you
collect and on what hardware you train.** Rank ordering and the qualitative conclusions are the
level at which results should be expected to match.

---

## 6.2 Restarting an interrupted run

A full run takes many hours, so it will get interrupted. It does resume — but resume is
**phase-level, and you must name the phase**.

```bash
bash scripts/launch_full_run.sh --no-clean --resume-from R9
```

`--no-clean` and `--resume-from` do different jobs and you generally need both. `--no-clean` only
skips archiving the previous results; on its own it does not move the starting phase, so the earlier
phases re-run. `--resume-from <label>` is what actually skips them. Labels are the R/P names in
`simulation/pipeline/phases.py` (`R1`…`R12`, `P1`) — the older numeric form no longer works.

**What survives a kill.** Three caches make a restart cheap rather than free:

| Artifact | Written | Effect on restart |
|---|---|---|
| Feature-engineering parquet (`simulation/cache/`) | R1 | Reused when the database has not changed |
| Optuna study (`simulation/results/optuna_study.db`) | R9 | Trials warm-start via `load_if_exists=True` |
| Per-model configs (`simulation/results/per_model_optimal/<MODEL>.json`) | R9 | **Per-model skip** — a model already optimised is not redone. This is the one place resume is finer than a phase, and it is why a crash at model 30 of 51 does not cost you models 1–29. `MPH_FORCE_REDO_PHASE13=1` forces a redo. |

**Two things to know before you rely on it.**

*What a resume restores, and what it cannot.* The in-memory `all_results` dict used to start empty on
every launch, so a run resumed at R10 reached the evaluation phases with nothing to score and still
printed "Pipeline Complete!". It is now rebuilt from disk on resume
(`simulation/pipeline/rehydrate.py`), and the fail-loud gate treats an involuntary skip of a critical
phase (P1/R9/R10) as a failure rather than a success.

Two phases still cannot come back, because their checkpoints were written as a progress log rather
than as a resume source:

| Phase | Checkpoint holds | Consequence on resume |
|---|---|---|
| R2 baseline | `{"baseline_n_models": N}` only | Its predictions are gone |
| R4 WF-CV | a subset of the walk-forward result | Partial |
| everything else | the full phase result | Restored |

R9 is a special case in your favour: it is rebuilt from `per_model_optimal/<MODEL>.json` rather than
from its checkpoint, so it survives even a targeted re-run that overwrote the checkpoint.

The practical consequence: **R10 selects the champion from two sources** — the BASIC-feature baseline
predictions from R2, and the feature-selected `[fs]` refits from R9. Resume at R9 and the R2 source is
absent, so the champion comes from the `[fs]` half of the pool. The run says so in the log
(`[resume] baseline NOT restored — …`) instead of reporting a confident result built on half the
candidates. **If the champion matters for what you are doing, run from R1.**

*Running one model, or a few.* `--models A,B` is for re-evaluating specific models, and it is **not
isolated from a previous full run**: the phases it touches overwrite that run's checkpoints with
their own narrow state. This has happened — a three-model GLM re-evaluation left R9 listing 3 models
instead of 46, R10 recording `skipped: filter excluded all`, and R12 regenerating its report from an
empty dict (`Models evaluated: 0`) while `per_model_metrics.csv` still held all 48.

Two ways to avoid it:

```bash
# isolate the targeted run's output entirely
MPH_OUTPUT_ROOT=/path/to/scratch .venv/bin/python -m simulation train --models NegBinGLM,FusedEpi

# or rebuild the aggregate report afterwards from the artifacts that survived
python scripts/regenerate_r12.py --sync-checkpoints
```

`regenerate_r12.py` reads the per-model files and the metrics CSV — which a targeted run does not
destroy — and re-runs the real R12 code over them, so the report and the checkpoints end up
describing the same run as the results tree. It stamps what it writes with `regenerated_from`.
`--dry-run` shows what it would rebuild.

Deployment stages P2–P5 (family deploy, ABM, ARIA, web) sit outside the phase dispatch loop and are
driven by their own CLI commands, so `--resume-from P2` and later are not meaningful.

**Windows.** `launch_full_run.sh` and `run_pipeline.sh` are POSIX shell, so driving a run on Windows
means WSL. The underlying `python -m simulation train --resume-from R9` is portable; only the
launcher and its environment-variable block are not.

---

## 6.3 How far determinism actually goes

Worth stating precisely, because "seeds are fixed" is true of most of this pipeline and not all of
it.

Seeding is **per-model, not global**. `simulation/pipeline/runner.py` does seed `random`, `numpy`
and `torch` in the parent process, but R9 fits each model in a **freshly spawned subprocess**
(`MPH_PHASE13_ISOLATE`, default on — it is what keeps a model's memory from leaking into the next).
A spawned child is a new interpreter, so the parent's `torch.manual_seed` does not carry into it,
and the generic worker script does not re-seed on entry. What does carry is the environment, which
is why `run_pipeline.sh` exports `PYTHONHASHSEED=42` before Python starts rather than setting it in
Python.

So determinism rests on each model seeding itself:

| Component | Seeded | Where |
|---|---|---|
| Tree models (XGBoost, LightGBM, RF, …) | Yes | `random_state=42` at construction |
| Deep models (DNN, TCN, LSTM, …) | Yes | `torch.manual_seed` inside `fit` |
| ABM | Yes, per stream | `SeedSequence(seed).spawn(...)` per day and district |
| **Graph models (GCN, GAT)** | **No** | No `manual_seed` or `random_state` anywhere in `simulation/models/graph_models.py` — weights initialise from OS entropy, so these two differ run to run |

Two consequences worth planning around. Re-running the pipeline reproduces the tree, deep and
mechanistic results but not GCN/GAT point-for-point. And the multi-seed spread reported for the
champion comes from `_multi_seed_metrics`, which refits under five explicit seeds — that is a
diagnostic for the champion, not the path every model takes.

This is documented rather than patched on purpose: adding a seed to the graph models would change
their output, and the numbers in `simulation/results/` were produced without one.

---

## 7. Key handling — read this before you deploy

Every key here is your own, issued to you and revoked by you at the portal that issued it. This
section is the set of cautions that are specific to *this* codebase — the things that are not
obvious from the provider documentation. They were established by reading the code, and each one
cites where.

### 7.1 The repository itself is clean

`api_key.txt` and `.env` have never been tracked, on any branch or in any earlier commit
(`git log --all --full-history` over both paths returns nothing). `.gitignore` covers `.env`,
`.env.*` and `**/api_key.txt` while deliberately re-including `.env.example` and
`api_key.example.txt`. Both shipped example files contain placeholders only. There are no hardcoded
credentials — keys are read from the file or the environment, nowhere else.

If you fork and add your own keys, that stays true only as long as you do not `git add -f`.

### 7.2 The environment silently beats the file

Keys are resolved in two tiers (`simulation/database/config.py:141-187`): `api_key.txt` is parsed
first, then environment variables overwrite matching entries. **The environment always wins**, for
exactly seven names — `SEOUL_KEY`, `SEOUL_SUBWAY`, `SEOUL_AIR`, `DATA_GO_KR`, `KMA_HUB`, `KOSIS`,
`NEIS`.

The failure mode this produces: you edit `api_key.txt`, the run still uses an old exported value
from a shell you sourced an hour ago, and nothing is logged to tell you. An *empty* environment
variable is harmless — the override is guarded by `if _val:` — so blank lines in a sourced `.env`
do not shadow the file.

### 7.3 `seoul_general2` cannot be set by environment variable

It is produced by writing the label `일반인증키(인구/상점)` (general authentication key —
population/commercial activity) **twice** in `api_key.txt`; the second
occurrence becomes `seoul_general2` (`config.py:156-161`). There is no environment variable for it.

This matters if you deploy in a container or CI that injects only environment variables: you can
never set this key that way, and `simulation/collectors/legacy/group_c_daily.py:340`, `:428` and
`:573` index it **unguarded** as `KEYS['seoul_general2']` — a missing entry is a `KeyError`, not a
graceful skip. `simulation/scripts/sanity_check.py:65` also lists it as required.

### 7.4 Duplicate labels overwrite each other without warning

Two different labels map to `data_go_kr` (`공공데이터포털 서비스키`, the Public Data Portal service
key, and `기상청_생활기상지수 조회서비스`, the KMA living-weather-index query service;
`config.py:117-118`), and two map to `kma_hub` (`기상청 api허브`, the KMA API hub, and
`기상청 인증키`, the KMA authentication key; `:119-120`). The parse loop assigns unconditionally, so the
**last line wins** — silently.

Reusing the same portal key on both lines is fine and is what the example file suggests. Pasting a
*different* key on the second line, or leaving one of the two at its `YOUR_...` placeholder, gives
you a hard-to-diagnose failure: the breakage surfaces in an unrelated collector, and no message
names the duplicated label.

### 7.5 Your keys travel inside the URL, and failures log the URL

Korean government APIs authenticate by URL, not by header. The Seoul key is in the **path**
(`{SEOUL_BASE}/{KEYS['seoul_general2']}/...`); data.go.kr and US Census take the key as a
**query-string parameter**. This is imposed upstream and cannot be fixed locally.

This used to mean every failure wrote a plaintext key to `simulation/logs/collect_*.log`, because
the collector logged the full URL on any 4xx, 5xx or timeout. **Failure logging now redacts.**
`redact_secrets()` (`simulation/database/config.py`) replaces every loaded key value — and its
percent-encoded form, since the key sits in a URL — with `***REDACTED***`, and
`simulation/collectors/legacy/base.py` passes both the URL and the response body through it before
logging.

Two limits worth knowing. Redaction only covers keys the process actually loaded, so a key you pass
some other way is not masked. And it is a log-time defence, not transport security: the key is still
in the request URL, so a proxy, a TLS-terminating gateway or the provider's own access log sees it
regardless.

Practical rules: still do not run the collectors through a shared or logging egress proxy, and
rotate any key that was in use before this redaction existed if those older logs left the host.

> `simulation/collectors/legacy/` is the **active** collector set despite the directory name —
> `simulation/collectors/orchestrator.py:191` imports from it. Do not dismiss these as dead code.

### 7.6 Quota exhaustion is retried, and leaves a marker

The retry branch in `legacy/base.py` used to fire only on `status_code >= 500`. Everything else fell
through to `break  # 4xx 등 클라이언트 오류는 재시도 불필요` (“client errors such as 4xx do not need
a retry”) — and **HTTP 429 is a 4xx**, so exhausting a daily quota stopped the collector dead. It
returned `None`, skipped every remaining item, and finished with partial data that looked exactly
like a genuinely sparse API response.

Now 429 is retried alongside 5xx. The wait honours the server's `Retry-After` header when it is
longer than the normal backoff, capped at `RETRY_AFTER_CAP = 120.0` seconds so a hostile or
mistaken header cannot stall a run indefinitely. If all `MAX_RETRY` attempts are exhausted the
collector logs a greppable `QUOTA-EXHAUSTED` line, so a quota failure is distinguishable from an
empty response after the fact:

```bash
grep -c QUOTA-EXHAUSTED simulation/logs/collect_*.log
```

This raises the quota ceiling; it does not remove it. A daily cap you have genuinely spent is not
recoverable by retrying, so still check row counts with `db-status` after a large collection rather
than trusting exit status, and use two *distinct* Seoul general keys (§2) to stay under the limit in
the first place.

### 7.7 The dashboard's map key cannot be kept secret

`NEXT_PUBLIC_VWORLD_KEY` is read inside a `"use client"` component and interpolated into tile URLs
(`web/components/MapPanel.tsx:493-503`). Next.js inlines `NEXT_PUBLIC_*` into the client bundle at
build time, and the key also travels in the tile request path — so any visitor can read it from the
bundle or from their network tab.

Secrecy is not available here. Restrict it at the **VWorld console** by registering your exact
deployment domain as the allowed service URL, and treat it as a public identifier with a quota
attached. **This repository does not do that for you** — there is no tile proxy and no referrer
configuration. Deploy as-is and you ship an unrestricted key.

### 7.8 `PUBLIC_DEMO=1` now refuses to start without a rate limiter

The chat routes layer three protections, and two of them used to **fail open**. With
`UPSTASH_URL` / `UPSTASH_TOKEN` unset, `publicRatelimit()` returned `null` and the call site guarded
with `if (rl)`, so the per-IP limit was skipped entirely; `checkDailyGlobalCap()` returned
`{ allowed: true }`, so there was no daily cap; and the auth gate was already off by definition in
public mode. A public deployment with `PUBLIC_DEMO=1` and no Upstash was therefore an
**unauthenticated, unmetered proxy to your paid Anthropic/OpenAI key** — nothing errored and nothing
was logged to tell you.

This now **fails closed**. `llmRateGuard()` (`web/lib/rate-guard.ts`) returns HTTP 503 when
`PUBLIC_DEMO=1` and no limiter is configured, and both LLM routes call it before reaching a
provider. `ALLOW_UNMETERED_PUBLIC_DEMO=1` overrides for local development and logs a warning each
time — never set it in production.

The guard lives in one shared helper for a reason. The first version of this check was written
inline in `/api/chat` only, and `/api/chat/parallel` — which reaches the same entry point and can
fan out to several providers per request — was left completely unguarded, so the hole stayed open
on the more expensive of the two routes. **Any new route that reaches an LLM must call
`llmRateGuard(req)`.**

Still set a hard monthly spend cap in the Anthropic or OpenAI console. The application-layer cap
protects against traffic, not against a bug or a misconfiguration in the layer itself.

### 7.9 Rotation

Rotation is per-deployment and per-key — do it at the issuing portal (data.seoul.go.kr, data.go.kr,
apihub.kma.go.kr, kosis.kr, open.neis.go.kr, or your LLM provider's console). Rotate whenever a key
may have been exposed: shared log files (§7.5), a public build containing a `NEXT_PUBLIC_*` value
(§7.7), or a demo deployment that ran without limits (§7.8).

---

## Citation

This code and these results support the following thesis:

> Seung Jin Lee (2026). *Real-Time Influenza Forecasting and Commuter-Coupled District Transmission
> Modeling for Seoul's 25 Districts: Multi-Agent Simulation of Adaptive Behavioral Responses to
> Infectious Disease Transmission.* Master of Public Health thesis, Graduate School of Public
> Health, Korea University.

> The thesis manuscript (docx/pdf) is **not included** in this repository pending degree conferral
> and university library submission. What is included is the code, analysis outputs, figure and
> table sources, and presentation decks that support it. The manuscript will be made available
> through the Korea University library after conferral.
