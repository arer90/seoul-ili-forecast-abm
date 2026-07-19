"""TDD guards for the 2026-06-21 search-quality fixes (post-run codex+workflow diagnosis).

Two confirmed defects under-optimized the run + corrupted champion selection:
  PART F (_inline_optuna_3stage.py): non-identity y-transform SEED trials (asinh/laplace/
    mcmc_robust) were pruned mid-fold in ~1/3 of models → their best transform never competed
    (TabPFN asinh R²0.927 lost to a defaulted identity 0.892). Fix: seed trials (trial.number
    < _n_seeded) skip fold-level pruning (pass optuna_trial=None) → full fair OOF per transform.
  PART G (per_model_optimize.py): the do-no-harm/baseline-floor + META skip set `best` to a
    metric-bare {transform:identity} dict → val_metrics.oof_wis defaulted to inf → the G-318
    champion selector skipped 35/49 models (incl the count/epi champions). Fix: backfill the
    model's best completed-trial OOF-WIS as the finite shortlist signal.

These tests pin the MECHANISM (not a full model R9) so they run fast + standalone.
"""
import math

import numpy as np
import optuna
import pytest


# ───────────────────────── PART F: seed-trial prune exemption ─────────────────────────
def test_seed_trials_exempt_from_pruning():
    """Seed trials (trial.number < n_seeded) must COMPLETE even when their interim loss is bad
    enough that a non-seed trial would be pruned. Mirrors _inline_optuna_3stage _preproc_objective:
    seeds pass optuna_trial=None to _oof_cv_wis_hier (its L500-504 only report/should_prune when
    optuna_trial is not None)."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    n_seeded = 3
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(n_startup_trials=2, seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=1, n_warmup_steps=0),
    )
    for v in ("a", "b", "c"):            # the enqueued startup seeds (run first, numbers 0..2)
        study.enqueue_trial({"t": v})

    def obj(trial):
        t = trial.suggest_categorical("t", ["a", "b", "c", "good"])
        base = 0.1 if t == "good" else 10.0      # seeds (a/b/c) are deliberately BAD
        is_seed = trial.number < n_seeded         # PART F exemption condition
        for step in range(5):
            if not is_seed:                       # seeds skip report/should_prune (= optuna_trial=None)
                trial.report(base + step, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return base

    study.optimize(obj, n_trials=14)
    seeds = [tr for tr in study.trials if tr.number < n_seeded]
    assert len(seeds) == n_seeded
    # ★ every seed must have COMPLETED (a fair full OOF), none pruned
    assert all(tr.state == optuna.trial.TrialState.COMPLETE for tr in seeds), \
        f"seed trials must complete, got {[tr.state for tr in seeds]}"
    # and the seeds' bad values ARE recorded (so the OOF can compare transforms)
    assert all(tr.value is not None for tr in seeds)


def test_nonseed_bad_trials_still_prunable():
    """Control: WITHOUT the exemption (is_seed always False), a bad trial CAN be pruned — proves
    the pruner is live and the exemption (above) is what protects seeds, not a dead pruner."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0),
    )

    def obj(trial):
        x = trial.suggest_float("x", 0.0, 1.0)
        for step in range(8):
            trial.report(x * 100 + step, step)   # high x => bad => prunable after warmup
            if trial.should_prune():
                raise optuna.TrialPruned()
        return x

    study.optimize(obj, n_trials=25)
    pruned = [tr for tr in study.trials if tr.state == optuna.trial.TrialState.PRUNED]
    assert len(pruned) >= 1, "pruner must be able to prune non-exempt trials (else exemption is moot)"


# ───────────────────────── PART G: oof_wis backfill for floored/meta best ─────────────────────────
def _backfill(best: dict, trial_results):
    """Mirror of per_model_optimize.py PART G backfill (kept in sync with that inline block)."""
    _bof = best.get("oof_wis")
    if (not isinstance(_bof, (int, float))) or (not np.isfinite(_bof)):
        _fin = [t for t in (trial_results or [])
                if isinstance(t, dict) and isinstance(t.get("oof_wis"), (int, float))
                and np.isfinite(t.get("oof_wis"))]
        if _fin:
            _bt = min(_fin, key=lambda t: t["oof_wis"])
            best["oof_wis"] = float(_bt["oof_wis"])
            if (not isinstance(best.get("wis"), (int, float))
                    or not np.isfinite(best.get("wis", float("nan")))):
                _bw = _bt.get("wis")
                if isinstance(_bw, (int, float)):
                    best["wis"] = float(_bw)
            best["_oof_wis_source"] = "min_trial_backfill"
    return best


def test_backfill_floored_identity_recovers_finite_oof():
    """Floored best (metric-bare identity) + finite trial OOFs → oof_wis backfilled to the min,
    so the G-318 selector no longer skips the model."""
    best = {"transform": "identity", "scaler": "none", "preproc_optuna_params": None}
    trials = [{"oof_wis": 2.438, "wis": 1.1}, {"oof_wis": 1.617, "wis": 0.9},
              {"oof_wis": float("inf"), "wis": float("nan")}]
    out = _backfill(best, trials)
    assert out["oof_wis"] == pytest.approx(1.617)
    assert out["wis"] == pytest.approx(0.9)
    assert out["_oof_wis_source"] == "min_trial_backfill"


def test_backfill_meta_zero_trials_stays_nonfinite():
    """META/mechanistic baselines (0 trials) keep non-finite oof_wis by design (reference models,
    not champion candidates) — no spurious finite value injected."""
    best = {"transform": "identity", "scaler": "none", "oof_wis": float("inf")}
    out = _backfill(best, [])
    assert not np.isfinite(out["oof_wis"])
    assert "_oof_wis_source" not in out


def test_backfill_finite_oof_unchanged():
    """A model that genuinely carried a finite oof_wis (HIER preproc path) is left untouched."""
    best = {"transform": "HIER_none", "oof_wis": 1.757, "wis": 0.73}
    out = _backfill(best, [{"oof_wis": 1.0}])
    assert out["oof_wis"] == pytest.approx(1.757)
    assert "_oof_wis_source" not in out


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
