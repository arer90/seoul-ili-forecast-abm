"""Phase banner labels must stay synced with the R/P registry (Phase 0b: label-keyed).

`eta.PHASE_ETAS` is now keyed by R/P label (numbers removed). This guards that every banner
key is a real R/P label and its display name matches the registry's semantic name, so the
progress banner never drifts from code identity (SSOT = simulation.pipeline.phases).
"""
from __future__ import annotations

from simulation.utils.eta import PHASE_ETAS
from simulation.pipeline import phases


def test_banner_keys_are_known_labels():
    for key in PHASE_ETAS:
        assert phases.is_known(key), f"PHASE_ETAS key '{key}' is not a known R/P label"


def test_banner_names_match_registry():
    """Each banner's display name == the registry's semantic name for that label."""
    for label, (name, _eta) in PHASE_ETAS.items():
        assert name == phases.name_of(label), (
            f"{label}: banner name '{name}' DESYNCED from registry '{phases.name_of(label)}'")


def test_numbers_fully_gone():
    # no integer keys remain in the banner table
    assert all(isinstance(k, str) for k in PHASE_ETAS)
    # retired/empty phases never reappear as labels
    for gone in ("ar_correction", "2", "3", "8"):
        assert gone not in PHASE_ETAS


def test_dispatched_research_phases_have_banner():
    """Every non-CLI registry phase has a banner entry (so the run shows progress)."""
    for label, _track, _name, is_cli in phases.PHASES:
        if not is_cli and label != "P1":          # P1 banner exists; P2-P5 are Phase B
            assert label in PHASE_ETAS, f"{label} missing from PHASE_ETAS"


if __name__ == "__main__":
    for fn in (test_banner_keys_are_known_labels, test_banner_names_match_registry,
               test_numbers_fully_gone, test_dispatched_research_phases_have_banner):
        fn()
        print(f"  ✓ {fn.__name__}")
    print("ALL PASS")
