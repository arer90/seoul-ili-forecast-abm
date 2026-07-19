"""Part 1 generator TDD — 합성데이터가 의도한 구조(AR + 상호작용 전용 신호)를 가지는지 증명.

make_synth: y=0.6·lag1 + 1.2·(xa·xb) + 0.6·xc + noise. 핵심 단언:
  · xa, xb 개별 marginal |corr(.,y)| ≈ 0 (상호작용으로만 기여 → |corr| 필터가 놓침)
  · xa·xb (곱) 은 |corr| 높음 (트리/model-based 가 잡을 신호)
  · y_lag1 |corr| 높음 (AR)
  · 결정론(seed)
이게 성립해야 crossover 실험(model-based 가 큰 n서 xa,xb 복원)이 유의미.

run: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python -m pytest <this> -q -s
"""
import numpy as np
import pytest

from simulation.scripts._feature_idea_proof_crossover import make_synth

pytestmark = pytest.mark.filterwarnings("ignore")


def _abscorr(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return 0.0
    return abs(float(np.corrcoef(a, b)[0, 1]))


def test_deterministic():
    X1, y1, _, _ = make_synth(300, seed=0)
    X2, y2, _, _ = make_synth(300, seed=0)
    assert np.array_equal(X1, X2) and np.array_equal(y1, y2)


def test_shape_and_names():
    X, y, names, sig = make_synth(500, p_noise=40, seed=1)
    assert X.shape == (500, 44) and len(y) == 500 and len(names) == 44
    assert sig == {"lag1": 0, "xa": 1, "xb": 2, "xc": 3}


def test_interaction_only_low_marginal_corr():
    """xa, xb 개별 |corr(.,y)| 은 작고(상호작용 전용), 곱 xa·xb 는 큼 — |corr| 필터의 사각지대."""
    X, y, names, sig = make_synth(3000, seed=0)
    ca = _abscorr(X[:, sig["xa"]], y)
    cb = _abscorr(X[:, sig["xb"]], y)
    cprod = _abscorr(X[:, sig["xa"]] * X[:, sig["xb"]], y)
    print(f"\n  |corr| xa={ca:.3f} xb={cb:.3f}  vs  xa·xb={cprod:.3f}")
    assert ca < 0.15 and cb < 0.15, f"xa/xb marginal |corr| 가 큼 (상호작용 전용 아님): {ca:.3f},{cb:.3f}"
    assert cprod > 0.30, f"xa·xb 곱 |corr| 가 작음 (신호 약함): {cprod:.3f}"


def test_ar_and_linear_signal_present():
    """y_lag1(AR)·xc(선형) 은 marginal |corr| 가 있어야 (|corr| 필터가 잡는 신호)."""
    X, y, names, sig = make_synth(3000, seed=0)
    assert _abscorr(X[:, sig["lag1"]], y) > 0.30, "AR 신호 약함"
    assert _abscorr(X[:, sig["xc"]], y) > 0.10, "선형 xc 신호 약함"


def test_noise_features_uncorrelated():
    X, y, names, sig = make_synth(3000, seed=0)
    noise_corrs = [_abscorr(X[:, j], y) for j in range(4, X.shape[1])]
    assert np.median(noise_corrs) < 0.10, "노이즈 feature 가 y 와 상관 (구조 오류)"
