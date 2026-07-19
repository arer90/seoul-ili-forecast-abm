"""TDD — factorial NNLS ensemble 결합 (2026-06-03).

run PER-FILE on macOS: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 pytest <thisfile>
"""
import numpy as np
import pytest

from simulation.analytics.ensemble_combine import nnls_ensemble


def test_nnls_recovers_known_weights():
    """y_val = 0.7·A + 0.3·B 면 NNLS 가 ~[0.7,0.3] 회복 → test 도 그 가중 결합."""
    rng = np.random.default_rng(0)
    n_val, n_test = 50, 30
    A_val = rng.normal(0, 1, n_val); B_val = rng.normal(0, 1, n_val)
    y_val = 0.7 * A_val + 0.3 * B_val
    A_test = rng.normal(0, 1, n_test); B_test = rng.normal(0, 1, n_test)
    ens, w = nnls_ensemble({"A": A_val, "B": B_val}, y_val,
                           {"A": A_test, "B": B_test})
    assert abs(w["A"] - 0.7) < 0.05 and abs(w["B"] - 0.3) < 0.05   # 가중치 회복
    assert abs(w["A"] + w["B"] - 1.0) < 1e-9                        # 합=1
    assert np.allclose(ens, 0.7 * A_test + 0.3 * B_test, atol=0.05) # test 결합


def test_nnls_weights_nonnegative_and_normalized():
    rng = np.random.default_rng(1)
    n = 40
    preds = {m: rng.normal(0, 1, n) for m in ("A", "B", "C")}
    y = rng.normal(0, 1, n)
    test = {m: rng.normal(0, 1, 20) for m in ("A", "B", "C")}
    ens, w = nnls_ensemble(preds, y, test)
    assert all(wi >= -1e-9 for wi in w.values())     # 비음
    assert abs(sum(w.values()) - 1.0) < 1e-9         # 정규화
    assert len(ens) == 20


def test_nnls_only_models_in_both():
    rng = np.random.default_rng(2)
    n = 30
    # A,B 는 val+test, C 는 val 만 (test 없음) → 제외
    ens, w = nnls_ensemble(
        {"A": rng.normal(0, 1, n), "B": rng.normal(0, 1, n), "C": rng.normal(0, 1, n)},
        rng.normal(0, 1, n),
        {"A": rng.normal(0, 1, 15), "B": rng.normal(0, 1, 15)})
    assert set(w) == {"A", "B"}      # C 제외 (test 없음)
    assert len(ens) == 15


def test_nnls_empty_returns_none():
    ens, w = nnls_ensemble({}, [1, 2, 3], {})
    assert ens is None and w == {}


def test_nnls_skips_none_preds():
    rng = np.random.default_rng(3)
    n = 30
    ens, w = nnls_ensemble(
        {"A": rng.normal(0, 1, n), "B": None},
        rng.normal(0, 1, n),
        {"A": rng.normal(0, 1, 15), "B": rng.normal(0, 1, 15)})
    assert set(w) == {"A"}          # B(None) 제외


def test_nnls_beats_worst_base_on_val():
    """앙상블 val MSE ≤ 최악 base val MSE (NNLS 가 나쁜 모델 가중↓)."""
    rng = np.random.default_rng(4)
    n = 60
    y = rng.normal(10, 2, n)
    good = y + rng.normal(0, 0.3, n)     # 좋은 base
    bad = y + rng.normal(0, 5, n)        # 나쁜 base
    ens, w = nnls_ensemble({"good": good, "bad": bad}, y, {"good": good, "bad": bad})
    mse = lambda p: float(np.mean((y - p) ** 2))
    assert mse(ens) <= mse(bad) + 1e-6   # 최악보다 나음
    assert w["good"] > w["bad"]          # 좋은 모델 가중 큼


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
