"""G-265b (2026-06-13, codex+gemini+claude 3자 리뷰): champion 선택 OOF 집계 = regime-conditional 통일.

3자 리뷰가 적발: G-256b(2026-06-12)가 inline 경로 OOF 집계를 median→regime-conditional mean 으로
바꿔 peak 캠페인을 완성했으나, **per_model_optimize 의 champion 선택 경로(1-SE OOF helper·config OOF)는
여전히 np.median(=D5) 사용** → champion 선택이 peak-blind 로 남는 불일치. `_oof_regime_aggregate` 로 통일.
(mc-probe median 은 fold 가 아닌 probe-model across 집계라 의도적 비대상.)

Run: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python -m pytest simulation/tests/test_g265b_regime_aggregate.py -x -q
"""
from __future__ import annotations

import numpy as np


def test_regime_aggregate_not_peak_blind():
    """outbreak fold 가 있으면 regime-conditional 은 median(peak-blind) 보다 크게 = peak 반영."""
    from simulation.pipeline.per_model_optimize import _oof_regime_aggregate
    wis = [2.0, 2.1, 1.9, 8.0, 9.0]          # 3 quiet + 2 outbreak
    fmax = [10, 11, 9, 90, 100]               # outbreak fold 의 y_val max 큼
    ytr = np.concatenate([np.full(250, 15.0), np.full(40, 80.0)])  # 75pct ≈ 15~ → outbreak threshold
    agg = _oof_regime_aggregate(wis, fmax, ytr)
    assert agg > float(np.median(wis)) + 1.0, "regime-conditional 이 median(peak-blind) 과 거의 같음"
    # 0.5·mean(quiet=2.0)+0.5·mean(outbreak=8.5)=5.25
    assert abs(agg - 5.25) < 0.5, f"regime-conditional 값 이상: {agg}"


def test_regime_aggregate_fallback_mean():
    """fold_maxes/y_train 없으면 mean fallback (median 아님)."""
    from simulation.pipeline.per_model_optimize import _oof_regime_aggregate
    wis = [2.0, 2.1, 1.9, 8.0, 9.0]
    assert abs(_oof_regime_aggregate(wis, None, None) - float(np.mean(wis))) < 1e-9


def test_regime_aggregate_empty_inf():
    from simulation.pipeline.per_model_optimize import _oof_regime_aggregate
    assert _oof_regime_aggregate([], None, None) == float("inf")


def test_regime_aggregate_single_regime_is_mean():
    """모든 fold 가 quiet(한 regime)면 = mean (regime 분리 불가)."""
    from simulation.pipeline.per_model_optimize import _oof_regime_aggregate
    wis = [2.0, 2.1, 1.9, 2.2]
    fmax = [10, 11, 9, 12]                     # 전부 quiet
    ytr = np.full(200, 15.0)
    agg = _oof_regime_aggregate(wis, fmax, ytr)
    assert abs(agg - float(np.mean(wis))) < 0.5


if __name__ == "__main__":
    test_regime_aggregate_not_peak_blind(); print("PASS not peak-blind")
    test_regime_aggregate_fallback_mean(); print("PASS fallback mean")
    test_regime_aggregate_empty_inf(); print("PASS empty inf")
    test_regime_aggregate_single_regime_is_mean(); print("PASS single regime")
    print("=== ALL PASS ===")
