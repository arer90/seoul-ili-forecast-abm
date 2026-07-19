"""Guard: FusedEpi default interval = NegBin+PID, Tweedie remains selectable (2026-07-12 decision).

Rationale: in the CANONICAL thesis protocol (242 train + 27 val + 68 test, pool_end=269, test[269,337)),
NegBin+PID beats Tweedie on WIS/PICP95/PICP50 (2.744/0.927/0.662 vs 2.796/0.897/0.603, DM tie) — verified in
scripts/_canonical_split_verify.py. The earlier "Tweedie wins" was a non-canonical 132-origin window artifact.
So NegBin+PID stays the committed default; Tweedie stays selectable for supplementary/robustness reporting.
Lightweight (no TiRex/TabPFN fit): guards the default + the option only.
"""
from __future__ import annotations
from simulation.models.fused_epi import FusedEpiForecaster


def test_default_pi_method_is_negbin():
    m = FusedEpiForecaster()
    assert m.pi_method == "negbin", f"default pi_method should be 'negbin' (canonical-test winner), got {m.pi_method!r}"
    assert m.tweedie_p == 1.5


def test_tweedie_still_selectable():
    m = FusedEpiForecaster(pi_method="tweedie")
    assert m.pi_method == "tweedie", "Tweedie must remain a selectable interval option (supplementary/robustness)"


def test_both_methods_are_valid_choices():
    for method in ("tweedie", "negbin"):
        m = FusedEpiForecaster(pi_method=method)
        assert m.pi_method == method
        # tweedie-specific state slot exists regardless of method (set during fit)
        assert hasattr(m, "_fused_cal")
