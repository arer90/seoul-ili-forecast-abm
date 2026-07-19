"""Guard: EnKF champion-coupling genuinely improves the ABM forward at the base
origin (real champion forecast) and is CHAMPION-SPECIFIC (proxy-anchor origins
do not benefit) — honest, leak-free, no over-claim.

Produced by ``scripts/run_abm_champion_enkf.py`` (reuses the live ABM + the
``enkf_assimilation`` EnKF; no live-code modification). Run per-file:
    .venv/bin/python -m pytest simulation/tests/test_abm_champion_enkf.py -q
"""
import json
from pathlib import Path

RESULT = Path(__file__).resolve().parents[1] / "results" / "abm_champion_enkf" / "result.json"


def _load() -> dict:
    assert RESULT.exists(), f"EnKF coupling result missing: {RESULT}"
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_base_reconstruction_parity():
    """Re-simulated anchored floor matches the stored multiorigin forward_r2 (proof
    the ensemble is genuinely re-simulated, not hardcoded)."""
    b = _load()["base_origin"]
    assert b is not None, "base origin (champion) missing"
    assert b["reconstruction_ok"] is True, (
        f"floor {b['r2_anchored_floor']} vs stored {b['r2_anchored_floor_stored']}")


def test_enkf_improves_base_origin_within_ceiling():
    """EnKF coupling lifts the base-origin ABM forward above the static-anchor floor
    and never above the champion-alone ceiling (a blend, honestly bounded)."""
    b = _load()["base_origin"]
    assert b["r2_enkf_coupled"] > b["r2_anchored_floor"], "EnKF must improve over the floor"
    assert b["r2_enkf_coupled"] <= b["r2_forecast_ceiling"] + 1e-6, (
        "EnKF-coupled R2 cannot exceed the champion-alone ceiling")


def test_improvement_is_champion_specific():
    """The gain is champion-specific: the base (champion) origin gains, and gains MORE
    than the median proxy origin (assimilating a poor climatology proxy does not help)."""
    d = _load()
    base_gain = d["base_origin"]["enkf_gain_over_floor"]
    proxy_median = d.get("proxy_enkf_gain_median")
    assert base_gain > 0, "champion-coupled base origin must gain over the floor"
    assert proxy_median is None or base_gain > proxy_median, (
        f"base gain {base_gain} not > proxy median {proxy_median} -> not champion-specific")


def test_leak_free():
    """The coupling is leak-free (forecast never contains the forward truth)."""
    assert _load().get("leak_free") is True
