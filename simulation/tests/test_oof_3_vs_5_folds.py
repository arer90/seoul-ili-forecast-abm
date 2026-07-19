"""oof_cv = 3-fold vs research_5fold = 5-fold (사용자 "oof_cv 3과 5 TDD").

n_pool 작을 때(여기 ~280) fold 수가 선택에 영향? 3-fold(~93/fold) vs 5-fold(~47/fold):
  - 같은 best preproc-config 를 고르나? (선택 일관성)
  - OOF WIS 추정값 차이? (5-fold = 더 작은 slab = 더 noisy?)
  - config 간 분산(=판별력) 차이?
실제 `_oof_cv_wis` (flat, n_folds 인자) 로 3 vs 5 직접 비교. 재현성 단언.

macOS: run PER-FILE with KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore")
_TS = ["identity", "log1p", "sqrt", "asinh"]
_SC = ["none", "robust"]


def _data(n=300, p=8, seed=11):
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(n, 1))
    X = np.hstack([g, g + 0.05 * rng.normal(size=(n, 1)), 2 * g, rng.normal(size=(n, 5))])
    y = 12.0 + 3.0 * X[:, 0] + 2.0 * X[:, 3] - 1.5 * X[:, 4] + rng.gamma(1.2, 0.6, n)
    return X, y, [f"f{i}" for i in range(p)]


def _oofs(name, X, y, cols, n_folds):
    """{config: oof_wis} for one model at n_folds."""
    from simulation.models.registry import verify_registry_coverage
    verify_registry_coverage(force_import=True)
    from simulation.models.base import REGISTRY
    from simulation.pipeline.per_model_optimize import _oof_cv_wis
    cls = REGISTRY.get(name)
    fac = (lambda c=cls: c())
    out = {}
    for t in _TS:
        for s in _SC:
            out[(t, s)] = _oof_cv_wis(fac, X, y, t, s, feature_cols=cols, n_folds=n_folds)
    return out


def _report():
    from simulation.models.registry import verify_registry_coverage
    verify_registry_coverage(force_import=True)
    X, y, cols = _data()
    reps = ["ElasticNet", "BayesianRidge", "KRR", "XGBoost"]
    lines = ["", "=" * 78, "oof_cv 3-fold vs 5-fold — 선택 일관성 + 추정 + 안정성", "=" * 78,
             f"  {'model':13s} {'best@3':>16s} {'best@5':>16s} {'동일?':>6s}"]
    agree = 0; n = 0; gaps = []
    for name in reps:
        try:
            o3 = _oofs(name, X, y, cols, 3); o5 = _oofs(name, X, y, cols, 5)
        except Exception as e:
            lines.append(f"  {name}: 실패 {str(e)[:30]}"); continue
        b3 = min(o3, key=o3.get); b5 = min(o5, key=o5.get)
        same = b3 == b5; agree += int(same); n += 1
        gaps.append(abs(o3[b3] - o5[b5]))
        lines.append(f"  {name:13s} {str(b3):>16s} {str(b5):>16s} {'YES' if same else 'no':>6s}"
                     f"   (WIS@3={o3[b3]:.3f} @5={o5[b5]:.3f})")
    lines.append("-" * 78)
    lines.append(f"  {n} models | best-config 3=5 동일: {agree}/{n}  |  평균 |OOF@3−OOF@5|={np.mean(gaps) if gaps else float('nan'):.4f}")
    lines.append(f"  해석: 동일 많으면 fold 수 선택 무관 (3 충분). |gap| 크면 5가 다른 추정.")
    lines.append("=" * 78)
    return "\n".join(lines), agree, n


def test_oof_3_vs_5_comparison(capsys):
    rep, agree, n = _report()
    print(rep)
    assert n >= 3, f"need ≥3 models, got {n}"


def test_oof_reproducible():
    """같은 seed → 3-fold OOF 동일 (TDD 재현성)."""
    X, y, cols = _data()
    a = _oofs("ElasticNet", X, y, cols, 3)
    b = _oofs("ElasticNet", X, y, cols, 3)
    assert a == b, "non-reproducible oof"
