"""Sprint α R1 (2026-05-26): mc_filter_stage3 SoT + alias deprecation tests.

phase2_multicollinearity (legacy alias) → mc_filter_stage3 (canonical).
Codex § 3.5 worst case: alias 누락 시 G-234 auto-select silently → none fallback.
"""
from __future__ import annotations

import warnings

import numpy as np


def test_canonical_module_imports():
    """new path import 동작."""
    from simulation.pipeline.mc_filter_stage3 import apply_multicollinearity_filter
    assert apply_multicollinearity_filter is not None


# (removed 2026-06-06: test_legacy_alias_module_* / test_alias_re_exports_* —
#  the phaseN deprecation-alias modules were deleted; canonical imports only now.)


def test_valid_methods_set():
    """4-method enum 확정."""
    from simulation.pipeline.mc_filter_stage3 import _VALID_METHODS
    assert _VALID_METHODS == {"none", "vif", "corr", "pca"}


def test_method_none_passthrough():
    """method=none → X unchanged + kept=None."""
    from simulation.pipeline.mc_filter_stage3 import apply_multicollinearity_filter

    rng = np.random.default_rng(42)
    X_tr = rng.normal(size=(40, 5))
    X_va = rng.normal(size=(10, 5))
    X_te = rng.normal(size=(10, 5))
    y_tr = rng.normal(size=40)
    feat_cols = ["a", "b", "c", "d", "e"]

    Xt, Xv, Xt2, kept, meta = apply_multicollinearity_filter(
        X_tr, X_va, X_te, y_tr, feature_cols=feat_cols, method="none",
    )
    assert Xt.shape == X_tr.shape
    assert meta["method"] == "none"
    # kept index list (none → all kept) — implementation 의존
    assert meta.get("n_dropped", 0) == 0


def test_method_vif_drops_multicollinear_features():
    """X3 = 2*X1 + noise 같은 collinear feature → VIF drop."""
    from simulation.pipeline.mc_filter_stage3 import apply_multicollinearity_filter

    rng = np.random.default_rng(1)
    n = 100
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    X3 = 2.0 * X1 + 0.01 * rng.normal(size=n)   # near-perfect collinear with X1
    X4 = rng.normal(size=n)

    X = np.column_stack([X1, X2, X3, X4])
    y = X1 + X2 + 0.1 * rng.normal(size=n)
    feat = ["X1", "X2", "X3", "X4"]

    Xt, Xv, Xt2, kept, meta = apply_multicollinearity_filter(
        X[:60], X[60:80], X[80:], y[:60],
        feature_cols=feat, method="vif",
    )
    assert meta["method"] == "vif"
    # X3 (collinear with X1) 가 drop 되어야 함
    assert meta.get("n_dropped", 0) >= 1


def test_method_corr_drops_high_correlation_pair():
    """|corr| > 0.9 → corr method 가 drop."""
    from simulation.pipeline.mc_filter_stage3 import apply_multicollinearity_filter

    rng = np.random.default_rng(2)
    n = 80
    X1 = rng.normal(size=n)
    X2 = X1 + 0.001 * rng.normal(size=n)  # near-perfect correlation
    X3 = rng.normal(size=n)

    X = np.column_stack([X1, X2, X3])
    y = X1 + X3 + 0.1 * rng.normal(size=n)
    feat = ["A", "B", "C"]

    Xt, Xv, Xt2, kept, meta = apply_multicollinearity_filter(
        X[:50], X[50:65], X[65:], y[:50],
        feature_cols=feat, method="corr",
    )
    assert meta["method"] == "corr"
    assert meta.get("n_dropped", 0) >= 1


def test_method_pca_returns_components():
    """PCA mode → X 가 component 공간으로 변환됨."""
    from simulation.pipeline.mc_filter_stage3 import apply_multicollinearity_filter

    rng = np.random.default_rng(3)
    n = 80
    X = rng.normal(size=(n, 10))
    y = X[:, 0] + X[:, 1] + 0.1 * rng.normal(size=n)
    feat = [f"f{i}" for i in range(10)]

    Xt, Xv, Xt2, kept, meta = apply_multicollinearity_filter(
        X[:50], X[50:65], X[65:], y[:50],
        feature_cols=feat, method="pca",
    )
    assert meta["method"] == "pca"
    # PCA 결과는 fewer or equal columns (variance retention)
    assert Xt.shape[1] <= 10


def test_phase12_caller_path_intact():
    """G-232 path: phase13_per_model_optimize.py 가 import 가능."""
    from simulation.pipeline.per_model_optimize import optimize_one_model
    assert optimize_one_model is not None


def test_runner_caller_path_intact():
    """runner.py 의 G-232a path: mc_filter_stage3 import 가능."""
    from simulation.pipeline.runner import run_pipeline
    assert run_pipeline is not None


# ────────────────────────────────────────────────────────────────────
# Round 3 audit CRITICAL-2 (2026-05-27): explicit train leak verification
# audit V1 D8 unfixed → V3 학습 launch 전 강제.
# 검증 패턴: train 만 fit, val/test 는 transform — synthetic outlier shift
# 가 train fit 만 사용했을 때 X_val/X_test 에 그대로 propagate 됨을 verify.
# ────────────────────────────────────────────────────────────────────


def test_no_train_leak_vif_train_only_fit():
    """VIF method: StandardScaler.fit() 가 X_train 만 사용.

    Synthetic dataset: X_val/X_test 에 극단적 mean shift (+1000) 적용.
    train-only fit 의 경우 _method_vif 는 X_val/X_test 를 X_val[:, kept_idx]
    로 그대로 반환 (별도 transform X). 결과: val 의 outlier mean 1000 이 유지됨.
    만약 별도 val/test fit 됐다면 outlier 가 정규화되어 mean≈0.
    """
    import numpy as np
    from simulation.pipeline.mc_filter_stage3 import _method_vif

    np.random.seed(42)
    X_train = np.random.normal(0, 1, (100, 5))
    y_train = np.random.normal(0, 1, 100)
    X_val = np.random.normal(1000, 1, (30, 5))
    X_test = np.random.normal(1000, 1, (30, 5))
    feature_cols = [f"f{i}" for i in range(5)]

    X_train_f, X_val_f, X_test_f, kept, meta = _method_vif(
        X_train, X_val, X_test, y_train, feature_cols
    )

    assert X_val_f.shape[0] == 30, "val sample count 보존"
    assert X_test_f.shape[0] == 30, "test sample count 보존"
    if X_val_f.shape[1] > 0:
        val_mean = float(np.mean(X_val_f))
        # train-only fit 의 증거: X_val 의 mean shift 1000 가 X_val_f 에 살아남음
        # 별도 val fit 시 정규화로 mean≈0 됨 (train leak inverse pattern)
        assert val_mean > 100, (
            f"val_mean = {val_mean:.2f} — train leak? "
            f"val 이 train fit 만 사용해야 X_val 의 원래 mean 보존"
        )


def test_no_train_leak_pca_train_only_fit():
    """PCA method: StandardScaler + PCA fit 모두 X_train 만 사용.

    동일 패턴: X_val/X_test 의 outlier shift 가 PCA transform 후 살아남음.
    """
    import numpy as np
    from simulation.pipeline.mc_filter_stage3 import _method_pca

    np.random.seed(42)
    X_train = np.random.normal(0, 1, (100, 10))
    y_train = np.random.normal(0, 1, 100)
    X_val = np.random.normal(1000, 1, (30, 10))
    X_test = np.random.normal(1000, 1, (30, 10))
    feature_cols = [f"f{i}" for i in range(10)]

    X_train_pca, X_val_pca, X_test_pca, kept, meta = _method_pca(
        X_train, X_val, X_test, y_train, feature_cols
    )

    if X_val_pca.shape[1] > 0:
        train_pc1_max = float(np.max(np.abs(X_train_pca[:, 0])))
        val_pc1_max = float(np.max(np.abs(X_val_pca[:, 0])))
        # val_pc1_max 가 train 의 5배 이상 = train-only fit 증거
        # (train 만 fit 했으므로 val 의 outlier shift 가 PCA score 에 propagate)
        assert val_pc1_max > train_pc1_max * 5, (
            f"val PC1 max ({val_pc1_max:.2f}) vs train PC1 max "
            f"({train_pc1_max:.2f}) — train leak? "
            f"train-only fit 시 val 의 mean shift 가 PC1 score 에 propagate 되어야 함"
        )
