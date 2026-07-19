"""within-season adaptive 검증 smoke TDD (early 보정 → late held-out).

실 DB + ABM 통합 — 구조 + discrimination metric(AUC/C-index) 존재 + held-out 분리 검증.
macOS: run PER-FILE.
"""
import os
import numpy as np
import pytest

from simulation.abm.within_season_validation import run_within_season

DB = "simulation/data/db/epi_real_seoul.db"


@pytest.mark.skipif(not os.path.exists(DB), reason="DB 없음")
def test_within_season_structure_and_discrimination():
    r = run_within_season(DB, cal_frac=0.6, r0_grid=(1.4, 1.8, 2.2))
    assert r["cal_weeks"] > 0 and r["eval_weeks"] > 0
    assert r["t_split"] == r["cal_weeks"]                 # early=보정, late=held-out 분리
    sci = r["sci_validation"]
    for arm in ("adaptive", "static"):
        for k in ("wis", "rmse", "mae", "auc_roc", "c_index", "coverage95"):
            assert k in sci[arm] and np.isfinite(sci[arm][k])
            if k in ("auc_roc", "c_index", "coverage95"):
                assert 0.0 <= sci[arm][k] <= 1.0
    assert "best_r0" in r["calibration"]
