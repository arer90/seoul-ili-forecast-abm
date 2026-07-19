"""B-P2 (M7 SCI-grade): Rt (Cori 2013 EpiEstim) wired into the simulator gate.

The estimator (`simulation.models.rt_estimator.RtEstimator`) already existed and
is solid; the bug was that `MetapopSEIRVD._run_gate` supplied predictions/
compartments/params but **never `rt`**, so `check_rt_sequence` was dead code for
every simulator run. After the fix, city-wide daily incidence (Σ_gu σ·E) is fed
through the existing RtEstimator and passed to the gate.

Pure-estimator tests (synthetic incidence) verify correctness independent of the
ODE engine; the sim-based test verifies the wiring end-to-end.
"""
from dataclasses import replace

import numpy as np


# ── pure estimator (engine-independent) ───────────────────────────────────
def test_rt_estimator_recovers_growth_phase():
    """Exponential growth incidence → median Rt > 1 (renewal-equation sanity)."""
    from simulation.models.rt_estimator import RtEstimator

    inc = 1.3 ** np.arange(40, dtype=float)  # clear R>1
    rt = RtEstimator(window_size=7).estimate(inc)["Rt_mean"].to_numpy()
    finite = rt[np.isfinite(rt)]
    assert finite.size > 0
    assert np.median(finite) > 1.0, "growth-phase Rt should exceed 1"


def test_rt_estimator_distinguishes_growth_from_decline():
    """Growth incidence yields a higher Rt than declining incidence.

    (Relative test — robust to the weak Gamma prior + discrete SI weights that
    float the decline tail toward ~1; the meaningful property is monotonicity.)
    """
    from simulation.models.rt_estimator import RtEstimator

    est = RtEstimator(window_size=7)
    rt_up = est.estimate(1.3 ** np.arange(40, dtype=float))["Rt_mean"].to_numpy()
    rt_dn = est.estimate(50.0 * 0.7 ** np.arange(40, dtype=float) + 1.0)["Rt_mean"].to_numpy()
    med_up, med_dn = np.nanmedian(rt_up), np.nanmedian(rt_dn)
    assert med_up > 1.0, "growth-phase Rt should exceed 1"
    assert med_up > med_dn, f"growth Rt ({med_up:.2f}) should exceed decline Rt ({med_dn:.2f})"


# ── sim wiring (end-to-end) ────────────────────────────────────────────────
def _short_sim():
    from simulation.sim.metapop_seirvd import MetapopSEIRVD
    from simulation.sim.scenarios import _default_params

    params = replace(_default_params(), days=120)
    return MetapopSEIRVD(params).run(run_validator=True, backend="numba")


def test_rt_sequence_check_now_fires_on_sim_output():
    """The rt_sequence checker runs on simulator output (dead before B-P2)."""
    res = _short_sim()
    report = res.epi_validity["metapop_seirvd"]
    assert "rt_sequence" in report["checks"], (
        "rt not wired into _run_gate — check_rt_sequence stayed dead"
    )


def test_sim_rt_is_finite_and_in_plausible_range():
    """A seeded epidemic yields estimable, finite, plausibly-bounded Rt."""
    from simulation.models.rt_estimator import RtEstimator

    res = _short_sim()
    city_incidence = res.incidence.sum(axis=1)
    rt = RtEstimator(window_size=7).estimate(city_incidence)["Rt_mean"].to_numpy()
    finite = rt[np.isfinite(rt)]
    assert finite.size > 0, "no estimable Rt from a growing epidemic"
    # upper bound only — the tail can legitimately approach the elimination floor
    assert float(np.nanmax(finite)) <= 8.0, f"Rt exceeds Cori ceiling: max={finite.max():.2f}"


def test_sim_rt_tracks_analytic_re():
    """Estimated Rt tracks the SEIR's own analytic Re = R0·S(t)/N.

    As S depletes over the wave both decline monotonically, so the Cori
    (incidence-based, lag-smoothed) estimate must be strongly positively
    correlated with the mechanistic Re — the consistency check the B-survey
    asked for, validating the wired estimate is mechanistically meaningful.
    """
    from simulation.models.rt_estimator import RtEstimator

    res = _short_sim()
    comps = {c: res.city_total(c) for c in ("S", "E", "I", "R", "V", "D")}
    N = sum(comps.values())  # elementwise total (conserved)
    re_full = res.params.disease.R0 * comps["S"] / N

    rt_df = RtEstimator(window_size=7).estimate(res.incidence.sum(axis=1))
    t_idx = rt_df["t"].to_numpy().astype(int)
    rt = rt_df["Rt_mean"].to_numpy()
    re_at_t = re_full[t_idx]
    m = np.isfinite(rt) & np.isfinite(re_at_t)
    # Spearman (rank/monotone): the Cori estimate lags Re, so Pearson is only
    # moderate, but both decline monotonically as S depletes → strong rank corr.
    from scipy.stats import spearmanr

    rho = float(spearmanr(rt[m], re_at_t[m]).statistic)
    assert rho > 0.6, f"estimated Rt should track analytic Re monotonically (rho={rho:.2f})"
