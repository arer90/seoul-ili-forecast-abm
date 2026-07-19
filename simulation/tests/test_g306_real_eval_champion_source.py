"""G-306: real_eval(phase 12) must use the FINAL champion + OPTIMIZED real prediction.

real_eval historically evaluated the WF-CV best-R² model with a fresh DEFAULT-HP
instance — NOT the optimized champion. When real_eval runs AFTER per_model_optimize(13)
+ per_model_eval(14), the best-WIS champion (per_model_eval.ranking_top10[0]) and its
optimized real-slab prediction (per_model_optimize.per_model_configs[champ]
.refit_real_predictions) are available. `_select_champion_and_real_pred` prefers those,
falling back to the WF-CV name + default-HP rolling when absent (order-robust).

macOS: run PER-FILE.
"""
import numpy as np

from simulation.pipeline.real_eval import _select_champion_and_real_pred


def _pme(top):
    return {"ranking_top10": top}


def _pmo(name, pred):
    return {"per_model_configs": {name: {"refit_real_predictions": pred}}}


def test_g306_fallback_when_no_post_opt_artifacts():
    """Current dispatch order (real_eval before 13/14): no per_model_eval/optimize →
    WF-CV fallback name + None + 'wfcv_default_hp' (legacy behaviour preserved)."""
    name, pred, src = _select_champion_and_real_pred({}, "WF_BEST", n_real=8)
    assert name == "WF_BEST"
    assert pred is None
    assert src == "wfcv_default_hp"


def test_g306_optimized_champion_preferred():
    """Champion (best-WIS) + matching-length optimized real prediction → reuse it."""
    opt = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    all_results = {
        "per_model_eval": _pme(["ChampX", "Runner", "Third"]),
        "per_model_optimize": _pmo("ChampX", opt),
    }
    name, pred, src = _select_champion_and_real_pred(all_results, "WF_BEST", n_real=8)
    assert name == "ChampX", "must switch to phase-14 best-WIS champion"
    assert src == "phase13_optimized"
    assert pred is not None and np.allclose(pred, opt), "must reuse the optimized array"


def test_g306_champion_but_no_optimized_pred():
    """Champion known but no optimized real prediction → champion name + None +
    'champion_default_hp' (caller does default-HP rolling on the champion)."""
    all_results = {
        "per_model_eval": _pme(["ChampX"]),
        "per_model_optimize": {"per_model_configs": {"ChampX": {}}},
    }
    name, pred, src = _select_champion_and_real_pred(all_results, "WF_BEST", n_real=8)
    assert name == "ChampX"
    assert pred is None
    assert src == "champion_default_hp"


def test_g306_length_mismatch_rejected():
    """An optimized pred whose length != n_real is rejected (stale/partial guard)."""
    all_results = {
        "per_model_eval": _pme(["ChampX"]),
        "per_model_optimize": _pmo("ChampX", [1.0, 2.0, 3.0]),  # len 3 != n_real 8
    }
    name, pred, src = _select_champion_and_real_pred(all_results, "WF_BEST", n_real=8)
    assert name == "ChampX"
    assert pred is None
    assert src == "champion_default_hp"


def test_g306_all_nan_pred_rejected():
    """An all-NaN optimized pred is rejected → champion_default_hp."""
    all_results = {
        "per_model_eval": _pme(["ChampX"]),
        "per_model_optimize": _pmo("ChampX", [np.nan] * 8),
    }
    name, pred, src = _select_champion_and_real_pred(all_results, "WF_BEST", n_real=8)
    assert pred is None
    assert src == "champion_default_hp"


def test_g306_error_or_skipped_per_model_eval_falls_back():
    """A per_model_eval that errored or skipped must NOT be mined for a champion."""
    for bad in ({"error": "boom"}, {"skipped": True, "reason": "n<10"}):
        name, pred, src = _select_champion_and_real_pred(
            {"per_model_eval": bad}, "WF_BEST", n_real=8)
        assert name == "WF_BEST"
        assert pred is None
        assert src == "wfcv_default_hp"


def test_g306_none_wf_fallback_stays_none():
    """When there is no WF-CV fallback either, returns (None, None, fallback)."""
    name, pred, src = _select_champion_and_real_pred({}, None, n_real=8)
    assert name is None and pred is None and src == "wfcv_default_hp"


def test_g306b_fs_champion_stripped_to_base_finds_optimized_pred():
    """G-306b (audit #3): champion='name[fs]' → strip [fs] → find BASE config's
    refit_real_predictions. Without the strip, cfgs.get('name[fs]') misses → silent loss."""
    opt = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    all_results = {
        "per_model_eval": _pme(["ChampX[fs]", "Runner"]),   # ranking_top10[0] carries [fs]
        "per_model_optimize": _pmo("ChampX", opt),           # per_model_configs keyed by BASE name
    }
    name, pred, src = _select_champion_and_real_pred(all_results, "WF_BEST", n_real=8)
    assert name == "ChampX", "must strip [fs] → base name (was 'ChampX[fs]')"
    assert src == "phase13_optimized"
    assert pred is not None and np.allclose(pred, opt)


def test_g306b_fs_champion_returns_base_name_for_fallback():
    """[fs] champion with no optimized pred → returns BASE name so REGISTRY.instantiate
    (the default-HP fallback) resolves instead of raising on 'name[fs]'."""
    all_results = {
        "per_model_eval": _pme(["ChampX[fs]"]),
        "per_model_optimize": {"per_model_configs": {"ChampX": {}}},
    }
    name, pred, src = _select_champion_and_real_pred(all_results, "WF_BEST", n_real=8)
    assert name == "ChampX", "base name (not 'ChampX[fs]') so instantiate works"
    assert pred is None
    assert src == "champion_default_hp"
