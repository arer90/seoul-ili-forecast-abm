# PHASE_MAPPING — number ↔ semantic name ↔ file (SSOT bridge)

> **Purpose** (codex+Gemini 2026-06-01): phase numbers are embedded in file names, functions, checkpoints,
> resume logic, banners, and comments, so every RENUMBER causes drift. This table is the **single bridge** —
> the thesis uses "Phase N" (fixed numbers), the code uses **semantic-name modules + run_<name> functions**,
> and the two are connected here. **Number = display only (this document, the thesis, log labels).
> Semantic name = stable identity.**
>
> **2026-06-02 full semantic rename completed** (user: "let the code use plain names without numbering too"):
> 19 phase module files + run_phaseN functions → semantic names. The old numbered names
> (`phase13_per_model_optimize.py`, `run_phase13`) are kept as **back-compat deprecation aliases**, so existing
> callers/checkpoints/tests/thesis references keep working without interruption. Dispatch order, checkpoint
> numbers, and log phase labels **keep the numbers** (behavior; 134-metric history keys). When inserting a new
> step, do not rename files → update only this table.

## Main pipeline (single `train` run, dispatch order = number)

| Thesis Phase | Semantic name (stable) | canonical file | canonical function | Status | feature |
|---|---|---|---|---|---|
| 1 | data | `data.py` | `run_data` | active | generates 399 raw |
| **2-3** | **(empty)** | — | — | **empty** (B2 2026-06-01) | feature selection and multicollinearity moved to **phase 13** |
| 4 | baseline | `baseline.py` | `run_baseline` | active | **BASIC** (lag + seasonality) anchor |
| 5 | external | `external.py` | `run_external` | optional (`optuna.mode=external/all`; **default none → inert**) | loads external Optuna JSON (unused by default) |
| 6 | wfcv | `wfcv.py` | `run_wfcv` | active | **BASIC** — champion comparison panel |
| 7 | diagnostics | `diagnostics.py` | `run_diagnostics` | active | BASIC (inherits phase6 OOF) |
| 8 | ar_correction | `ar_correction.py` | `run_ar_correction` | active | BASIC (inherited) |
| 9 | dm_test | `dm_test.py` | `run_dm_test` | active | BASIC (inherited) |
| 10 | intervals | `intervals.py` | `run_intervals` (+`run_intervals_extended`) | active | BASIC (inherited) |
| 11 | scoring | `scoring.py` | `run_scoring` | active | BASIC (inherited) |
| 12 | real_eval (★champion gate) | `real_eval.py` | `run_real_eval` | active | **BASIC** (real_X+X_all slice) |
| 13 | per_model_optimize (★feature) | `per_model_optimize.py` | `run_per_model_optimize` | active | **full pool** STABILITY+nested guard (full = final fallback) |
| — | (★eval-features) | runner `_resolve_eval_features`/`X_eval`/`_phase1_eval` | — | `MPH_EVAL_FEATURES=basic` default | 4-12 = BASIC (lag + seasonality, 13 features), 13 = full. Setting =full regresses to the old behavior |
| 14 | per_model_eval (134-metric) | `per_model_eval.py` | `run_per_model_eval` | active | selected features |
| 15 | **shap / xai** | `shap_analysis.py`, `xai.py` | `run_shap`, `run_xai` | active | — (`shap_analysis`: avoids a name clash with the `import shap` library) |
| 16 | comprehensive | `comprehensive_eval.py` | `run_comprehensive_eval` | active | — |

## Separate CLI (split off from the main train run)

| Thesis Phase | Semantic name | canonical file | Notes |
|---|---|---|---|
| 17 | inference | `inference.py` | `run_inference`. Writes forecasts to the DB |
| 18 | **overseas (domestic vs. overseas ILI comparison)** | `overseas.py` (+`seoul_gu.py`, `true_ili_cohort.py`) | **★ the old "phase 15 overseas" was moved to 18 by the RENUMBER** (US/JP/DE/FR/KR cross-country R²/RMSE/MAPE/WIS comparison + HTML). Phase 15 was reassigned to SHAP/XAI. |

## Old numbered names → canonical (back-compat deprecation alias, 2026-06-02)

> The 19 old `phaseN_X.py` modules are thin re-export shims (DeprecationWarning + full `dir()` re-export: public
> and single-underscore private). Old imports and function names work 100%. New code should use the canonical
> names. **Do not delete** (caller/checkpoint/thesis compatibility).

| Old numbered module (alias) | canonical | Old function (alias) | canonical function |
|---|---|---|---|
| `phase1_data.py` | `data.py` | `run_phase1` | `run_data` |
| `phase4_baseline.py` | `baseline.py` | `run_phase4_baseline` | `run_baseline` |
| `phase5_external.py` | `external.py` | `run_phase5_external` | `run_external` |
| `phase6_wfcv.py` | `wfcv.py` | `run_phase6` | `run_wfcv` |
| `phase7_diagnostics.py` | `diagnostics.py` | `run_phase7` | `run_diagnostics` |
| `phase8_ar_correction.py` | `ar_correction.py` | `run_phase8` | `run_ar_correction` |
| `phase9_dm_test.py` | `dm_test.py` | `run_phase9` | `run_dm_test` |
| `phase10_intervals.py` | `intervals.py` | `run_phase10`(+`_extended`) | `run_intervals`(+`_extended`) |
| `phase11_scoring.py` | `scoring.py` | `run_phase11` | `run_scoring` |
| `phase12_real_eval.py` | `real_eval.py` | `run_phase12` | `run_real_eval` |
| `phase13_per_model_optimize.py` | `per_model_optimize.py` | `run_phase13` | `run_per_model_optimize` |
| `phase14_per_model_eval.py` | `per_model_eval.py` | `run_phase14` | `run_per_model_eval` |
| `phase15_shap.py` | `shap_analysis.py` | `run_phase15` | `run_shap` |
| `phase15_xai.py` | `xai.py` | `run_phase15_xai` | `run_xai` |
| `phase16_comprehensive_eval.py` | `comprehensive_eval.py` | `run_phase16` | `run_comprehensive_eval` |
| `phase17_inference.py` | `inference.py` | `run_phase17` | `run_inference` |
| `phase18_overseas.py` / `phase18_seoul_gu.py` / `phase18_true_ili_cohort.py` | `overseas.py` / `seoul_gu.py` / `true_ili_cohort.py` | (no function rename) | — |

## Semantic-name modules (number-independent from the start)

| canonical | Old number alias | Notes |
|---|---|---|
| `mc_filter_stage3.py` | `phase2_multicollinearity.py` (alias) | 4-way multicollinearity. **alias = thin re-export shim** — verified by `test_mc_filter_stage3`. Do not delete. |
| `_inline_optuna_3stage.py` | `phase3_feature_optuna.py` (alias) | inline preproc/feature Optuna. alias = back-compat. |
| `feature_select_corr1se.py` | (no number) | STABILITY/nested/guard — the `corr1se` in the file name is historical. |
| `phase_evaluator.py` | (no number) | 134-metric SSOT evaluation. |

## resume (by name or by number)

- `--resume-from 13` (numeric) **or** `--resume-from per_model_optimize` (semantic name) both work — 2026-06-02.
- Implementation: `runner.PHASE_NAME_TO_NUMBER` + `runner.resolve_resume_from` (case- and whitespace-insensitive). `__main__` wires up a lazy `_resume_type`.
- `shap`/`shap_analysis`/`xai` → all map to phase 15. Dispatch numbers and checkpoints stay numeric.

## Historical RENUMBERs (drift tracking)
- **2026-05-28 (a3648f0)**: dispatch order unified with the numbering. HP-optimize→13, real_eval→12, SHAP→15.
- **old 15 overseas → 18**: because of this move the user mistakenly thought "the phase 15 domestic/overseas comparison disappeared" — in reality it was relocated to 18 (this table prevents that confusion).
- **2026-06-01 (B2)**: phases 2-3 emptied. Scrubbed stale runner comments and resume labels + created this mapping table.
- **2026-06-02 full semantic rename**: 19 module files + 17 run_phaseN functions → semantic names (Step A `git mv`+alias /
  Step B function rename + canonical imports / Step C resume name aliases). Old numbered names = deprecation aliases. shap→shap_analysis.

## Thesis citation rule
- The thesis methods section cites **"Phase N"** (the numbers in the table above) verbatim — this table **fixes** the numbers, so the thesis stays safe even if the code structure changes.
- On the code side, writing **number + semantic name** together, as in "Phase 13 (per_model_optimize)", is recommended.
- **Do not change the numbers again.** Code should use the canonical semantic names (old numbered names are aliases). Add steps via a semantic name plus an update to this table.

## Completed (2026-06-02 semantic rename)
- ✅ Full semantic file rename (`phase13_per_model_optimize.py`→`per_model_optimize.py`, 19 in total) + function rename
  (17 run_phaseN→run_<name>) + `--resume-from` number↔name compatibility aliases. As recommended by codex+Gemini: **isolated
  step-by-step commits (Step A/B/C), zero behavior change, mechanical grep verification, back-compat aliases**. Full sweep: 544 passed.
- (Not adopted) converting dispatch into a `PIPELINE_ORDER` registry loop — each phase takes heterogeneous arguments
  (run_wfcv(X, y, cols, per_model_feature_map) vs run_diagnostics(y, oof)), which makes a generic loop hard and high-risk →
  kept the hardcoded if-block sequential dispatch (number = order, call = run_<name>). Revisit in a separate PR if needed.
