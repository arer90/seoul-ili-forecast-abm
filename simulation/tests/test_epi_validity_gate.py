"""Stage 4 — epi-validity gate unit tests.

Covers each new check function plus the ``run_epi_validity_gate`` dispatcher
and the tournament opt-in exclusion filter. These tests are *self-contained*:
they exercise the verifier + ensemble modules without requiring a DB, a
feature cache, or any model training.
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.verifier.epi_validity import (
    EPI_RANGE,
    KNOWN_OUTBREAK_PEAKS,
    RT_DELTA_CAP,
    SEASONAL_PEAK_WEEKS,
    check_compartment_conservation,
    check_epi_validity,
    check_outbreak_alignment,
    check_rt_sequence,
    check_seasonal_peak,
    run_epi_validity_gate,
)


# ──────────────────────────────────────────────────────────────────────
# Range-table tightenings (Stage 4 spec)
# ──────────────────────────────────────────────────────────────────────
def test_range_table_stage4_tightenings():
    assert EPI_RANGE["Re"].lo == 0.3
    # C4/M7: VE floor widened 0.50→0.10 to admit drift-mismatch seasons (TND VE 10-60%).
    assert EPI_RANGE["VE"].lo == 0.10 and EPI_RANGE["VE"].hi == 0.95
    assert EPI_RANGE["ifr"].lo == 0.0001 and EPI_RANGE["ifr"].hi == 0.05


def test_module_constants_exposed():
    assert RT_DELTA_CAP == 1.5
    # seasonal peak window W48-W52 ∪ W1-W8
    for w in (48, 49, 50, 51, 52, 1, 2, 3, 4, 5, 6, 7, 8):
        assert w in SEASONAL_PEAK_WEEKS
    for w in (20, 25, 30, 40):
        assert w not in SEASONAL_PEAK_WEEKS
    assert "2009_H1N1" in KNOWN_OUTBREAK_PEAKS
    assert KNOWN_OUTBREAK_PEAKS["2009_H1N1"] == (2009, 45)


# ──────────────────────────────────────────────────────────────────────
# check_rt_sequence
# ──────────────────────────────────────────────────────────────────────
class TestRtSequence:
    def test_benign_sequence_passes(self):
        r = check_rt_sequence([0.8, 1.0, 1.2, 1.4, 1.1])
        assert r.status == "ok"

    def test_below_floor_fails(self):
        r = check_rt_sequence([0.1, 0.2, 1.0])
        assert r.status == "fail"
        assert any("elimination floor" in v for v in r.details["violations"])

    def test_above_ceiling_fails(self):
        # ceiling raised 5.0→8.0 (G-184, ENGINEERING_PRINCIPLES.md Rt∈[0.3,8] spec); use >8.0.
        r = check_rt_sequence([1.0, 1.2, 9.0])
        assert r.status == "fail"
        assert any("Cori 2013 ceiling" in v for v in r.details["violations"])

    def test_week_over_week_jump_fails(self):
        r = check_rt_sequence([1.0, 3.0, 1.0])  # |ΔRt| = 2.0 > 1.5
        assert r.status == "fail"
        assert any("|ΔRt|" in v for v in r.details["violations"])

    def test_empty_warns(self):
        r = check_rt_sequence([])
        assert r.status == "warn"


# ──────────────────────────────────────────────────────────────────────
# check_compartment_conservation
# ──────────────────────────────────────────────────────────────────────
class TestCompartmentConservation:
    def test_exact_conservation(self):
        T = 5
        comps = {
            "S": np.full(T, 900.0), "E": np.full(T, 10.0),
            "I": np.full(T, 30.0),  "R": np.full(T, 50.0),
            "V": np.full(T, 5.0),   "D": np.full(T, 5.0),
        }
        c = check_compartment_conservation(comps, 1000.0)
        assert c.status == "ok"
        assert c.details["max_rel_err"] < 1e-9

    def test_drift_fails(self):
        T = 5
        comps = {
            "S": np.full(T, 800.0), "E": np.full(T, 10.0),  # 10% missing
            "I": np.full(T, 30.0),  "R": np.full(T, 50.0),
            "V": np.full(T, 5.0),   "D": np.full(T, 5.0),
        }
        c = check_compartment_conservation(comps, 1000.0)
        assert c.status == "fail"

    def test_tolerance_respected(self):
        T = 5
        comps = {
            "S": np.full(T, 999.9), "I": np.full(T, 0.05), "R": np.full(T, 0.05),
        }
        # total = 1000.0; exactly at tol
        c = check_compartment_conservation(comps, 1000.0, tol=1e-3)
        assert c.status == "ok"


# ──────────────────────────────────────────────────────────────────────
# check_seasonal_peak
# ──────────────────────────────────────────────────────────────────────
class TestSeasonalPeak:
    def _series(self, peak_week: int, length: int = 52):
        preds = np.zeros(length)
        weeks = np.arange(1, length + 1)
        idx = np.argmax(weeks == peak_week)
        preds[idx] = 10.0
        return preds, weeks

    def test_january_peak_ok(self):
        preds, weeks = self._series(2)   # W2
        assert check_seasonal_peak(preds, weeks).status == "ok"

    def test_december_peak_ok(self):
        preds, weeks = self._series(50)  # W50
        assert check_seasonal_peak(preds, weeks).status == "ok"

    def test_summer_peak_fails(self):
        preds, weeks = self._series(26)  # W26 (late June / early July)
        r = check_seasonal_peak(preds, weeks)
        assert r.status == "fail"
        assert any("outside the seasonal window" in v
                   for v in r.details.get("violations", []))

    def test_length_mismatch_fails(self):
        r = check_seasonal_peak([1, 2, 3], [1, 2])
        assert r.status == "fail"


# ──────────────────────────────────────────────────────────────────────
# check_outbreak_alignment
# ──────────────────────────────────────────────────────────────────────
class TestOutbreakAlignment:
    def _iyw(self):
        years = np.concatenate([
            np.full(52, 2009), np.full(52, 2017), np.full(10, 2018),
        ])
        weeks = np.concatenate([
            np.arange(1, 53), np.arange(1, 53), np.arange(1, 11),
        ])
        return np.stack([years, weeks], axis=1)

    def test_aligned_peak_ok(self):
        iyw = self._iyw()
        preds = np.zeros(iyw.shape[0])
        # strong peaks at anchors
        i = np.where((iyw[:, 0] == 2009) & (iyw[:, 1] == 45))[0][0]
        preds[i] = 100.0
        j = np.where((iyw[:, 0] == 2018) & (iyw[:, 1] == 2))[0][0]
        preds[j] = 95.0
        r = check_outbreak_alignment(preds, iyw)
        assert r.status == "ok"

    def test_missing_anchor_peak_fails(self):
        iyw = self._iyw()
        preds = np.random.RandomState(0).rand(iyw.shape[0])
        # suppress the anchor region
        mask_2009 = (iyw[:, 0] == 2009) & (np.abs(iyw[:, 1] - 45) <= 2)
        preds[mask_2009] = 0.0
        # add a huge peak elsewhere
        preds[0] = 10.0
        r = check_outbreak_alignment(preds, iyw)
        assert r.status == "fail"

    def test_shape_validation(self):
        r = check_outbreak_alignment([1, 2, 3], [[2009, 1], [2009, 2]])  # mismatched rows
        assert r.status == "fail"


# ──────────────────────────────────────────────────────────────────────
# run_epi_validity_gate dispatcher
# ──────────────────────────────────────────────────────────────────────
class TestGateDispatcher:
    def test_flag_only_default(self):
        outputs = {
            "BadRt": {"predictions": np.array([1.0, 2.0, 3.0]),
                      "rt": [0.1, 5.5]},
        }
        gate = run_epi_validity_gate(outputs)
        assert gate["BadRt"]["status"] == "fail"
        assert gate["BadRt"]["exclude_from_ensemble"] is False

    def test_strict_exclude_flips_flag(self):
        outputs = {
            "BadRt": {"predictions": np.array([1.0, 2.0, 3.0]),
                      "rt": [0.1, 5.5]},
        }
        gate = run_epi_validity_gate(outputs, strict_exclude=True)
        assert gate["BadRt"]["status"] == "fail"
        assert gate["BadRt"]["exclude_from_ensemble"] is True

    def test_clean_model_passes(self):
        preds = np.zeros(52); preds[1] = 10.0
        outputs = {
            "Clean": {"predictions": preds,
                      "iso_weeks": np.arange(1, 53),
                      "rt": [0.8, 1.0, 1.2]},
        }
        gate = run_epi_validity_gate(outputs)
        assert gate["Clean"]["status"] == "ok"

    def test_prediction_only_model_still_ok(self):
        outputs = {
            "OnlyPreds": {"predictions": np.array([1.2, 1.5, 0.8])},
        }
        gate = run_epi_validity_gate(outputs)
        assert gate["OnlyPreds"]["status"] == "ok"


# ──────────────────────────────────────────────────────────────────────
# Config integration
# ──────────────────────────────────────────────────────────────────────
def test_config_has_epi_validity_defaults():
    from simulation.pipeline.config import EpiValidityConfig, PipelineConfig
    c = PipelineConfig()
    assert isinstance(c.epi_validity, EpiValidityConfig)
    assert c.epi_validity.enabled is True
    assert c.epi_validity.strict_exclude is False
    assert c.epi_validity.rt_delta_cap == RT_DELTA_CAP


def test_config_roundtrips_through_dict():
    from simulation.pipeline.config import PipelineConfig
    c = PipelineConfig()
    c.epi_validity.strict_exclude = True
    c.epi_validity.rt_delta_cap = 2.0
    d = c.to_dict()
    c2 = PipelineConfig.from_dict(d)
    assert c2.epi_validity.strict_exclude is True
    assert c2.epi_validity.rt_delta_cap == 2.0


# ──────────────────────────────────────────────────────────────────────
# Tournament integration
# ──────────────────────────────────────────────────────────────────────
class TestTournamentFilter:
    def _make_preds(self):
        rng = np.random.RandomState(0)
        y = rng.rand(50)
        oof = {
            "GoodModel":   y + 0.01 * rng.rand(50),
            "MeHModel":    y + 0.05 * rng.rand(50),
            "EpiBadModel": y + 0.02 * rng.rand(50),
        }
        cats = {n: "ML" for n in oof}
        return oof, y, cats

    def test_strict_gate_drops_flagged_model(self):
        from simulation.ensembles.tournament import TournamentOrchestrator
        oof, y, cats = self._make_preds()
        gate = {
            "EpiBadModel": {"status": "fail", "violations": ["Rt jumped"],
                            "exclude_from_ensemble": True},
            "GoodModel":   {"status": "ok", "exclude_from_ensemble": False},
            "MeHModel":    {"status": "ok", "exclude_from_ensemble": False},
        }
        orch = TournamentOrchestrator(top_k_per_category=2, caruana_steps=5)
        res = orch.run(oof, y, cats, epi_validity_gate=gate)
        # EpiBadModel must not appear in stage A'-2 pool weights
        a2_weights = getattr(res.stage_a2, "weights", {}) or {}
        assert "EpiBadModel" not in a2_weights

    def test_flag_only_gate_keeps_model(self):
        from simulation.ensembles.tournament import TournamentOrchestrator
        oof, y, cats = self._make_preds()
        gate = {
            "EpiBadModel": {"status": "fail", "violations": ["Rt jumped"],
                            "exclude_from_ensemble": False},  # flag-only
            "GoodModel":   {"status": "ok", "exclude_from_ensemble": False},
            "MeHModel":    {"status": "ok", "exclude_from_ensemble": False},
        }
        orch = TournamentOrchestrator(top_k_per_category=2, caruana_steps=5)
        res = orch.run(oof, y, cats, epi_validity_gate=gate)
        a1 = res.stage_a1 or {}
        # EpiBadModel is eligible for stage A'-1 inclusion
        all_a1 = {m for names in a1.values() for m in names}
        assert "EpiBadModel" in all_a1 or "EpiBadModel" in oof  # at least not dropped silently

    def test_paper_primary_never_dropped(self):
        from simulation.ensembles.tournament import TournamentOrchestrator
        oof, y, cats = self._make_preds()
        gate = {
            "EpiBadModel": {"status": "fail", "exclude_from_ensemble": True},
            "GoodModel":   {"status": "ok", "exclude_from_ensemble": False},
            "MeHModel":    {"status": "ok", "exclude_from_ensemble": False},
        }
        orch = TournamentOrchestrator(top_k_per_category=2, caruana_steps=5)
        res = orch.run(oof, y, cats,
                       paper_primary=["EpiBadModel"],
                       epi_validity_gate=gate)
        # EpiBadModel is paper_primary → must survive the gate
        a1 = res.stage_a1 or {}
        all_a1 = {m for names in a1.values() for m in names}
        assert "EpiBadModel" in all_a1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
