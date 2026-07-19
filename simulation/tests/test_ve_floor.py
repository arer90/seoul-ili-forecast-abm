"""C4 (M7): VE gate floor widened to admit drift-mismatch seasons (TND VE 10-60%)."""
from simulation.verifier.epi_validity import EPI_RANGE, check_epi_validity


def test_ve_floor_lowered_to_admit_mismatch_seasons():
    lo, hi = EPI_RANGE["VE"].lo, EPI_RANGE["VE"].hi
    assert lo <= 0.10 < 0.50, f"VE floor not widened: {lo}"
    assert hi == 0.95


def test_low_ve_season_now_passes_gate():
    # a documented drift-mismatch VE (0.15) must NOT be flagged any more
    params = {"R0": 1.4, "sigma": 0.5, "gamma": 0.25, "VE": 0.15}
    res = check_epi_validity(params=params, predictions=None)
    ve_viol = [v for v in res.details.get("violations", []) if "VE" in v]
    assert ve_viol == [], f"low-VE season wrongly flagged: {ve_viol}"


def test_impossible_ve_still_rejected():
    res = check_epi_validity(params={"VE": 1.5}, predictions=None)
    assert any("VE" in v for v in res.details.get("violations", []))
