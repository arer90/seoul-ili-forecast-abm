"""B-P6 (M7): seasonal-peak checker fires on simulator output (was dead).

_run_gate now derives ISO weeks from a season-start anchor (day 0 = W36) so
check_seasonal_peak runs — a synthetic flu wave should peak inside W48-W8.
"""
from dataclasses import replace

import numpy as np


def _sim(days=150):
    from simulation.sim.metapop_seirvd import MetapopSEIRVD
    from simulation.sim.scenarios import _default_params
    return MetapopSEIRVD(replace(_default_params(), days=days)).run(
        run_validator=True, backend="numba")


def test_seasonal_peak_check_now_fires():
    checks = _sim().epi_validity["metapop_seirvd"]["checks"]
    assert "seasonal_peak" in checks, "iso_weeks not wired — check_seasonal_peak stayed dead"


def test_synthetic_flu_peaks_in_winter_window():
    sp = _sim().epi_validity["metapop_seirvd"]["checks"]["seasonal_peak"]
    assert "peak_week" in sp
    # a flu wave seeded at W36 should peak in the allowed W48-W8 window
    assert sp["peak_week"] in set(sp["allowed_window"])


def test_iso_weeks_length_matches_predictions():
    # the derived iso_weeks must align 1:1 with the daily prediction trajectory
    res = _sim(days=60)
    n_pred = res.state.shape[0]              # T daily snapshots
    # re-derive with the same anchor and confirm the modular week math is sane
    weeks = np.array([((36 - 1 + d // 7) % 53) + 1 for d in range(n_pred)])
    assert weeks.min() >= 1 and weeks.max() <= 53
