"""G-308 (audit #2): masked best preproc trial binds its OWN trial_params, not study.best_params.

Bug: study.best_params is Optuna's internal best (by returned oof_wis). An error trial (whose
_evaluate_config failed but whose independent _oof_cv_wis_hier returned a finite oof) could be
Optuna's best → refit replays that INVALID preproc → N-HiTS/TiDE refit-null. Fix:
_pick_masked_best_preproc masks error trials and binds the masked best's own trial_params.

macOS: run PER-FILE.
"""
from simulation.pipeline._inline_optuna_3stage import _pick_masked_best_preproc


def test_g308_error_trial_masked_uses_own_params():
    """Error trial has the LOWEST oof (would be study.best) but is masked → best = valid trial,
    preproc_optuna_params = the VALID trial's own params (NOT the error trial's)."""
    trial_results = [
        {"oof_wis": 1.0, "wis": float("inf"), "error": "boom",
         "trial_params": {"y_transform": "BROKEN"}},        # lowest oof but ERROR
        {"oof_wis": 2.0, "wis": 2.0,
         "trial_params": {"y_transform": "log1p"}},          # valid, higher oof
    ]
    best, best_idx = _pick_masked_best_preproc(trial_results, "oof_cv")
    assert best_idx == 1, "error trial (idx 0) masked despite lowest oof"
    assert best["preproc_optuna_params"] == {"y_transform": "log1p"}, \
        "binds masked-best's OWN params, NOT the error trial's (refit-null source)"


def test_g308_all_valid_picks_lowest_oof():
    trial_results = [
        {"oof_wis": 3.0, "wis": 3.0, "trial_params": {"t": "a"}},
        {"oof_wis": 1.0, "wis": 1.0, "trial_params": {"t": "b"}},   # best
        {"oof_wis": 2.0, "wis": 2.0, "trial_params": {"t": "c"}},
    ]
    best, best_idx = _pick_masked_best_preproc(trial_results, "oof_cv")
    assert best_idx == 1
    assert best["preproc_optuna_params"] == {"t": "b"}


def test_g308_best_by_val_ranks_by_wis():
    trial_results = [
        {"oof_wis": 1.0, "wis": 5.0, "trial_params": {"t": "a"}},
        {"oof_wis": 9.0, "wis": 1.0, "trial_params": {"t": "b"}},   # best by wis
    ]
    best, best_idx = _pick_masked_best_preproc(trial_results, "val")
    assert best_idx == 1, "best_by='val' ranks by wis, not oof_wis"
    assert best["preproc_optuna_params"] == {"t": "b"}


def test_g308_missing_trial_params_defaults_empty():
    trial_results = [{"oof_wis": 1.0, "wis": 1.0}]   # no trial_params key
    best, best_idx = _pick_masked_best_preproc(trial_results, "oof_cv")
    assert best["preproc_optuna_params"] == {}
