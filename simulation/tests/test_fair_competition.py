"""TDD — phase 14 FAIR-COMPETITION II (2026-06-02, codex+Gemini+Claude).

`_collect_fs_test_preds` 가 phase-13 feature-선택 refit 예측을 "name[fs]" 로 노출해 BASIC 과
test slab head-to-head 경쟁시킴(champion=진짜 best, 강제 X). 정렬·방어성·라벨 고정.

run PER-FILE on macOS: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 pytest <thisfile>
"""
import numpy as np
import pytest

from simulation.pipeline.per_model_eval import _collect_fs_test_preds


def _ar(vals):
    return list(np.asarray(vals, dtype=float))


def test_basic_fs_configs_added_with_label():
    ar = {"per_model_optimize": {"per_model_configs": {
        "XGBoost":  {"refit_test_predictions": _ar(range(10)), "test_metrics": {"wis": 3.0}},
        "LightGBM": {"refit_test_predictions": _ar(range(10, 20)), "test_metrics": {"wis": 2.0}},
    }}}
    out = _collect_fs_test_preds(ar, n_real_test=10)
    assert set(out) == {"XGBoost[fs]", "LightGBM[fs]"}, "feature-선택은 [fs] 라벨로"
    assert np.allclose(out["XGBoost[fs]"], np.arange(10.0))
    assert out["LightGBM[fs]"].shape == (10,)


def test_alignment_takes_last_n_real_test():
    # refit 예측이 test slab 보다 길면 마지막 n_real_test 정렬
    ar = {"per_model_optimize": {"per_model_configs": {
        "M": {"refit_test_predictions": _ar(range(20))},
    }}}
    out = _collect_fs_test_preds(ar, n_real_test=8)
    assert out["M[fs]"].shape == (8,)
    assert np.allclose(out["M[fs]"], np.arange(12.0, 20.0))  # 마지막 8개


def test_defensive_missing_or_bad_inputs_never_raise():
    # phase 13 부재
    assert _collect_fs_test_preds({}, 10) == {}
    assert _collect_fs_test_preds({"per_model_optimize": {}}, 10) == {}
    assert _collect_fs_test_preds({"per_model_optimize": None}, 10) == {}
    # res 가 dict 아님 / 예측 None / 짧음 / NaN-only → 전부 skip (raise X)
    ar = {"per_model_optimize": {"per_model_configs": {
        "bad_type": "not-a-dict",
        "no_pred":  {"test_metrics": {"wis": 1.0}},
        "none_pred": {"refit_test_predictions": None},
        "too_short": {"refit_test_predictions": _ar(range(3))},
        "all_nan":   {"refit_test_predictions": [float("nan")] * 10},
        "good":      {"refit_test_predictions": _ar(range(10))},
    }}}
    out = _collect_fs_test_preds(ar, n_real_test=10)
    assert set(out) == {"good[fs]"}, f"유효한 것만: {set(out)}"


def test_fs_can_compete_and_win_or_lose():
    """심판=test slab: BASIC 과 feature-선택이 같은 slab 에서 경쟁 가능함을 고정
    (champion 강제 없음 — 둘 다 ranking pool 진입)."""
    n = 12
    y = np.arange(n, dtype=float)
    test_preds = {"XGBoost": y + 5.0}  # BASIC (큰 오차)
    ar = {"per_model_optimize": {"per_model_configs": {
        "XGBoost": {"refit_test_predictions": _ar(y + 0.1)},  # feature-선택 (작은 오차)
    }}}
    test_preds.update(_collect_fs_test_preds(ar, n_real_test=n))
    # 둘 다 pool 에 → 경쟁. feature-선택이 더 정확하면 그게 이길 수 있고, 아니면 BASIC.
    assert "XGBoost" in test_preds and "XGBoost[fs]" in test_preds
    mae_basic = np.mean(np.abs(test_preds["XGBoost"] - y))
    mae_fs = np.mean(np.abs(test_preds["XGBoost[fs]"] - y))
    assert mae_fs < mae_basic, "이 fixture 에선 feature-선택이 더 정확 (경쟁 성립 확인)"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
