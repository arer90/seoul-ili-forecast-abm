"""G-318b (2026-06-19): rerank_champion._y_test reconstructs the EXACT phase-13 hold-out target.

The original `_y_test` used `SELECT ili_rate ... ORDER BY week_start` — `week_start` does not
exist (schema = season_start/week_seq/week_label; ili_rate has 7 age_group rows). That raised
OperationalError → DM gate silently skipped → G-318 could not actually test significance.

The fix reuses the canonical loader (load_kr_sentinel_ili = 7-age AVG, ORDER BY season_start,
week_seq) and the real split `[train | test n_test | real REAL_HORIZON]`, so the test target is
`y[-(n_test+REAL_HORIZON):-REAL_HORIZON]`, NOT the last n_test weeks.

TDD guard: a correct reconstruction must reproduce the STORED test_metrics.r2 of already-trained
models (the pipeline computed r2 against the true y_test; if our y_test matches, r2 matches).

macOS: run PER-FILE.
"""
import json
import os

import numpy as np
import pytest

from simulation.scripts.rerank_champion import REAL_HORIZON, _resolve_real_horizon, _y_test

OPT = "simulation/results/per_model_optimal"


def _r2(pred, true):
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    return 1.0 - ((true - pred) ** 2).sum() / ((true - true.mean()) ** 2).sum()


@pytest.mark.skipif(not os.path.exists("simulation/data/db/epi_real_seoul.db"), reason="needs DB")
def test_real_horizon_resolved_from_split_not_hardcoded():
    """G-322b: REAL_HORIZON 은 하드코딩(15) 아니라 split(test_window_idx)에서 유도 — 그리고 그 유도값이
    stored R² 를 정확 재현해야(하드코딩 15 는 1주 어긋나 R² −0.29). REAL_HORIZON 상수는 fallback 전용."""
    from simulation.database import safe_connect
    from simulation.pipeline.true_ili_cohort import load_kr_sentinel_ili
    con = safe_connect("simulation/data/db/epi_real_seoul.db")
    y = np.asarray([v for _, _, v in load_kr_sentinel_ili(con)], float)
    con.close()
    H = _resolve_real_horizon(y, 68)
    assert H > 0
    if os.path.exists(f"{OPT}/FusedEpi.json"):
        d = json.load(open(f"{OPT}/FusedEpi.json", encoding="utf-8"))
        recon = _r2(d["refit_test_predictions"], y[-(68 + H):-H])
        assert abs(recon - d["test_metrics"]["r2"]) < 1e-3, f"H={H} 재현 실패"
        # 회귀 가드: 옛 하드코딩 15 는 어긋나야 (그래야 이 테스트가 의미)
        assert abs(_r2(d["refit_test_predictions"], y[-83:-15]) - d["test_metrics"]["r2"]) > 0.1


@pytest.mark.skipif(not os.path.exists("simulation/data/db/epi_real_seoul.db"),
                    reason="needs real DB")
def test_y_test_length_is_n_test():
    yt = _y_test(68)
    assert yt is not None and len(yt) == 68


@pytest.mark.skipif(not os.path.exists(f"{OPT}/TabPFN.json"),
                    reason="needs trained per_model_optimal results")
@pytest.mark.parametrize("model", ["FusedEpi", "TabPFN", "ARIMA", "SeirCount-TabPFN"])
def test_y_test_reproduces_stored_r2(model):
    """Reconstructed y_test must reproduce the stored hold-out R² (proves exact alignment)."""
    p = f"{OPT}/{model}.json"
    if not os.path.exists(p):
        pytest.skip(f"{model} not trained")
    d = json.load(open(p, encoding="utf-8"))
    pred = d.get("refit_test_predictions")
    stored_r2 = (d.get("test_metrics") or {}).get("r2")
    if not pred or stored_r2 is None:
        pytest.skip(f"{model} has no test predictions")
    yt = _y_test(len(pred))
    assert yt is not None, "y_test reconstruction returned None (DM would be skipped)"
    recon_r2 = _r2(pred, yt)
    assert abs(recon_r2 - stored_r2) < 0.01, (
        f"{model}: reconstructed R²={recon_r2:.4f} != stored {stored_r2:.4f} "
        f"→ y_test misaligned, DM would compare against wrong target"
    )


@pytest.mark.skipif(not os.path.exists(f"{OPT}/TabPFN.json"), reason="needs results")
def test_y_test_is_not_naive_last_n():
    """Guard against the naive `y[-n:]` bug: it gives a wildly wrong (negative) R²."""
    from simulation.database import safe_connect
    from simulation.pipeline.true_ili_cohort import load_kr_sentinel_ili
    con = safe_connect("simulation/data/db/epi_real_seoul.db")
    y = np.asarray([v for _, _, v in load_kr_sentinel_ili(con)], float)
    con.close()
    d = json.load(open(f"{OPT}/TabPFN.json", encoding="utf-8"))
    pred = d["refit_test_predictions"]
    naive_r2 = _r2(pred, y[-len(pred):])      # the OLD buggy alignment
    correct_r2 = _r2(pred, _y_test(len(pred)))
    assert naive_r2 < 0 < correct_r2, (
        f"naive last-n R²={naive_r2:.3f} should be wrong/negative; correct={correct_r2:.3f}"
    )
