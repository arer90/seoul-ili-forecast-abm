# ⚠ Stale — do not use

These web-dashboard aggregates were generated **before the FusedEpi champion was finalised**.
They contradict the current results, so they are quarantined out of the served data path.

| File | Stale content | Current (paper) |
|------|---------------|-----------------|
| `ili-forecast-models.json` | `champion: NegBinGLM`, `champion_version: "V6 salvage (RidgeCV+log1p …)"` | champion = **FusedEpi** |
| `ili-forecast.json` | `model: NegBinGLM` | champion = **FusedEpi** |

Two things are out of date:

1. **Champion** — the paper and pipeline champion is `FusedEpi`, selected leak-free on out-of-fold
   WIS. `NegBinGLM` is reported alongside it as the interpretable count model, not as the champion.
2. **NegBinGLM implementation** — `V6 salvage (RidgeCV+log1p)` was a count-GLM in name only; the
   internals were RidgeCV. It was later replaced with a genuine log-link GLM, so the numbers above
   are not what the current code produces.

## Regenerating

Web aggregates are build artifacts produced by the pipeline's web stage (P5) from current results.
Build the database (see `SETUP.md`) and run the pipeline to regenerate them against the current
champion. Until then the dashboard's forecast panel may be empty.

Integrity check: `tests/test_champion_consistency.py` catches this mismatch.
