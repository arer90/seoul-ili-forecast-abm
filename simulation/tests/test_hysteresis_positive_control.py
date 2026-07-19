"""Guard: the hysteresis detector must resolve a loop wherever the series starts.

The detector reported the textbook maximal loop — a unit circle, driver cos and
response sin, 90 degrees out of phase — as `area=0.000, p=1.000, "single-valued /
too few cycles to resolve a loop"`. That is the same output it gives the
behaviour-OFF control, so the two were indistinguishable.

Two compounding causes, both about WHERE the series begins:

  `_branch_area` split each span at `argmax`. A driver sampled from its maximum
  has argmax at index 0, the guard `pk < 2` fired, and it returned 0.

  `_cycle_bounds` cut at the single interior trough, handing `_branch_area` two
  monotone half-spans whose turning points were both on edges — hence the
  "2 cycle(s)" in the verdict for what is one full cycle.

This matters beyond the unit test: `docs/PIPELINE_DB_TO_WEB.md` quotes this
detector's loop area as evidence that behaviour-ON produces path dependence,
against a behaviour-OFF control reported as `0.000, p=1.000`. A detector that
also returns `0.000, p=1.000` for a real loop cannot support that contrast.

Run standalone — macOS needs per-file pytest runs (LightGBM/OpenMP):
    .venv/bin/python -m pytest simulation/tests/test_hysteresis_positive_control.py -q
"""

import numpy as np
import pytest

from simulation.abm.dynamical_signatures import hysteresis_loop_area

# The inputs are z-normalised before integration, so a unit circle becomes radius
# sqrt(2) and encloses 2*pi rather than pi.
EXPECTED = 2.0 * np.pi


def _circle(n=40, phase=0.0, reverse=False):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False) + phase
    return np.cos(t), (-np.sin(t) if reverse else np.sin(t))


def test_unit_circle_is_detected():
    d, r = _circle()
    out = hysteresis_loop_area(d, r, n_null=2000)
    assert out["significant"] is True, out["verdict"]
    assert out["null_p"] < 0.05
    assert out["abs_area"] == pytest.approx(EXPECTED, rel=0.05), (
        f"enclosed area {out['abs_area']} != {EXPECTED:.4f}"
    )


def test_one_traversal_counts_as_one_cycle():
    """The 2-cycle miscount is what routed a full loop into the monotone path."""
    d, r = _circle()
    assert hysteresis_loop_area(d, r, n_null=200)["n_cycles"] == 1


@pytest.mark.parametrize("phase", [0.0, np.pi])
def test_extremum_aligned_starts_both_give_the_full_loop(phase):
    """Starting at the driver's max exercises the trough split, at its min the peak
    split. Both resolve a complete cycle, so both must return the full enclosed
    area with the same sign."""
    base = hysteresis_loop_area(*_circle(phase=0.0), n_null=200)
    out = hysteresis_loop_area(*_circle(phase=phase), n_null=200)
    assert out["abs_area"] == pytest.approx(base["abs_area"], rel=0.02)
    assert np.sign(out["loop_area"]) == np.sign(base["loop_area"])


@pytest.mark.parametrize("phase", [0.0, np.pi / 6, np.pi / 3, np.pi / 2, np.pi, 1.5 * np.pi])
def test_multi_cycle_detection_is_phase_independent(phase):
    """The shape real data takes: several seasons, response lagged.

    A single truncated cycle cannot be measured in full — the branch integral is
    restricted to the driver range both branches share, and a window that starts
    mid-slope shares only a sliver (see the docstring's Limitations note). With
    two or more complete cycles the loop is resolvable wherever the record begins,
    and detection must not depend on that accident.
    """
    t = np.linspace(0, 6 * np.pi, 150) + phase
    out = hysteresis_loop_area(-np.cos(t), -np.cos(t - 0.6), n_null=1000)
    assert out["significant"] is True, out["verdict"]
    per_cycle = out["abs_area"] / max(out["n_cycles"], 1)
    assert 2.5 < per_cycle < 4.5, (
        f"area per resolved cycle {per_cycle:.3f} is off the ~3.5 the same "
        f"trajectory gives at other phases"
    )


def test_reversed_circulation_flips_the_sign():
    fwd = hysteresis_loop_area(*_circle(), n_null=200)
    rev = hysteresis_loop_area(*_circle(reverse=True), n_null=200)
    assert np.sign(fwd["loop_area"]) == -np.sign(rev["loop_area"])
    assert fwd["abs_area"] == pytest.approx(rev["abs_area"], rel=0.02)


# ── the negative controls must stay negative ─────────────────────────────────
def test_memoryless_single_valued_stays_at_zero():
    """y = x^2 retraces its own path: no memory, no loop."""
    up = np.linspace(0, 1, 20)
    d = np.concatenate([up, up[::-1]])
    out = hysteresis_loop_area(d, d ** 2, n_null=2000)
    assert out["abs_area"] < 0.05
    assert out["significant"] is False, out["verdict"]


def test_constant_response_is_not_a_loop():
    """Behaviour-OFF: no modulation to be path-dependent about."""
    t = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    out = hysteresis_loop_area(np.cos(t), np.full(40, 3.0), n_null=200)
    assert out["significant"] is False
    assert out["abs_area"] == 0.0


def test_monotone_driver_has_no_loop_to_find():
    """An open ramp never returns, so there is no enclosed area to report."""
    d = np.linspace(0, 1, 40)
    out = hysteresis_loop_area(d, d + 0.1 * np.sin(8 * d), n_null=200)
    assert out["significant"] is False, out["verdict"]


def test_multi_season_lagged_driver_is_detected():
    """The shape the real ILI/mobility pair takes: several cycles, response lagged."""
    t = np.linspace(0, 6 * np.pi, 150)
    out = hysteresis_loop_area(-np.cos(t), -np.cos(t - 0.6), n_null=2000)
    assert out["significant"] is True, out["verdict"]
    assert out["null_p"] < 0.05
