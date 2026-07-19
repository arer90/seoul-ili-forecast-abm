"""
simulation/tests/test_v22_cv_splits.py
======================================
— CV splitter regression tests for S2-2.

`test_leakage.py` 는 `LeakageChecker` 와 정적 (date, gu_nm) split 을 검사한다.
이 파일은 **CV splitter 함수 자체**가 아래 불변식을 만족하는지 고정한다:

 1. train / test 인덱스 집합이 서로소 (disjoint)
 2. 경계 인접성 — test_start == train_end (expanding window, no gap/overlap)
 3. train 은 단조 증가 (expanding)
 4. S0-1 holdout 슬랩 이 WF-CV fold 에 의해 절대 침범되지 않음
 5. 모든 test 블록 이 합집합 == holdout 직전 까지 (coverage)
 6. 최소 학습 조건 (min_train, horizon) 이 모든 fold 에서 지켜짐

대상: `simulation.models.expanding_cv.ExpandingWindowCV._get_folds`
 `simulation.pipeline.wfcv._generate_wf_folds`

실행:
 .venv\\Scripts\\python.exe -m pytest simulation/tests/test_v22_cv_splits.py -v
"""
from __future__ import annotations

import numpy as np
import pytest


# ══════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════

def _train_idx(train_end: int) -> set[int]:
    return set(range(0, train_end))


def _test_idx(test_start: int, test_end: int) -> set[int]:
    return set(range(test_start, test_end))


# ══════════════════════════════════════════════════════════════════════════
# ExpandingWindowCV._get_folds
# ══════════════════════════════════════════════════════════════════════════

def test_expanding_cv_folds_are_disjoint():
    """모든 fold 에서 train ∩ test == ∅."""
    from simulation.models.expanding_cv import ExpandingWindowCV
    cv = ExpandingWindowCV(min_train_weeks=104, step_weeks=26, horizon=13)
    folds = cv._get_folds(n_samples=343)
    assert folds, "no folds generated — check fixture size / params"
    for i, (tr_end, te_start, te_end) in enumerate(folds):
        tr = _train_idx(tr_end)
        te = _test_idx(te_start, te_end)
        inter = tr & te
        assert not inter, (
            f"fold {i}: train ∩ test overlap = {len(inter)} "
            f"(tr_end={tr_end}, te=[{te_start},{te_end}))"
        )


def test_expanding_cv_boundary_is_adjacent():
    """expanding window 는 no-gap, no-overlap — test_start == train_end."""
    from simulation.models.expanding_cv import ExpandingWindowCV
    cv = ExpandingWindowCV(min_train_weeks=104, step_weeks=26, horizon=13)
    for tr_end, te_start, te_end in cv._get_folds(343):
        assert te_start == tr_end, (
            f"gap/overlap at boundary: tr_end={tr_end}, te_start={te_start}"
        )
        assert te_end - te_start == 13, (
            f"test window length != horizon: te=[{te_start},{te_end})"
        )


def test_expanding_cv_is_monotonically_expanding():
    """fold i+1 의 train 구간은 fold i 의 train 구간을 포함한다."""
    from simulation.models.expanding_cv import ExpandingWindowCV
    cv = ExpandingWindowCV(min_train_weeks=104, step_weeks=26, horizon=13)
    folds = cv._get_folds(343)
    for a, b in zip(folds, folds[1:]):
        assert b[0] > a[0], f"train_end not expanding: {a[0]} → {b[0]}"


def test_expanding_cv_respects_min_train_and_horizon():
    """첫 fold 의 train_end >= min_train, 모든 fold 의 te_end <= n."""
    from simulation.models.expanding_cv import ExpandingWindowCV
    cv = ExpandingWindowCV(min_train_weeks=104, step_weeks=26, horizon=13)
    n = 343
    folds = cv._get_folds(n)
    assert folds[0][0] >= cv.min_train_weeks
    for tr_end, te_start, te_end in folds:
        assert te_end <= n, f"fold extends past end: te_end={te_end} > n={n}"
        assert tr_end >= cv.min_train_weeks


def test_expanding_cv_empty_on_too_few_samples():
    """데이터가 min_train + horizon 에도 못 미치면 fold 리스트는 비어있다."""
    from simulation.models.expanding_cv import ExpandingWindowCV
    cv = ExpandingWindowCV(min_train_weeks=104, step_weeks=26, horizon=13)
    assert cv._get_folds(50) == []
    assert cv._get_folds(116) == []  # 104 + 13 = 117 필요


# ══════════════════════════════════════════════════════════════════════════
# phase6_wfcv._generate_wf_folds — holdout-aware
# ══════════════════════════════════════════════════════════════════════════

def test_wfcv_folds_disjoint_without_holdout():
    from simulation.pipeline.wfcv import _generate_wf_folds
    folds = _generate_wf_folds(n_total=343, min_train=104, step=26, holdout_start=None)
    assert folds
    for i, fold in enumerate(folds):
        # Fold 형식이 (tr_end, val_start, val_end) 또는 유사한 3-tuple 이라 가정
        tr_end, val_start, val_end = fold[:3]
        tr = _train_idx(tr_end)
        va = _test_idx(val_start, val_end)
        assert not (tr & va), (
            f"fold {i}: train ∩ val overlap "
            f"(tr_end={tr_end}, val=[{val_start},{val_end}))"
        )


def test_wfcv_folds_never_touch_holdout_slab():
    """S0-1 회귀 가드: holdout_start 이후 인덱스는 어떤 fold 에도 나타나지 않는다."""
    from simulation.pipeline.wfcv import _generate_wf_folds
    n, holdout_start = 343, 317   # last 26 weeks = conformal holdout
    folds = _generate_wf_folds(n_total=n, min_train=104, step=26,
                               holdout_start=holdout_start)
    assert folds, "folds are empty — check min_train vs holdout_start"
    for i, fold in enumerate(folds):
        tr_end, val_start, val_end = fold[:3]
        assert val_end <= holdout_start, (
            f"fold {i}: val_end={val_end} extends into holdout "
            f"(start={holdout_start})"
        )
        assert tr_end <= holdout_start, (
            f"fold {i}: tr_end={tr_end} extends into holdout "
            f"(start={holdout_start})"
        )


def test_wfcv_holdout_covers_exactly_last_slab():
    """holdout slab [holdout_start, n) 는 어떤 fold 에도 속하지 않는다."""
    from simulation.pipeline.wfcv import _generate_wf_folds
    n, holdout_start = 343, 317
    folds = _generate_wf_folds(n_total=n, min_train=104, step=26,
                               holdout_start=holdout_start)
    covered: set[int] = set()
    for fold in folds:
        tr_end, val_start, val_end = fold[:3]
        covered |= _train_idx(tr_end)
        covered |= _test_idx(val_start, val_end)
    holdout = set(range(holdout_start, n))
    assert not (covered & holdout), (
        f"holdout slab [{holdout_start},{n}) touched by WF-CV: "
        f"{sorted(covered & holdout)[:5]}..."
    )


def test_wfcv_monotonic_train_end():
    from simulation.pipeline.wfcv import _generate_wf_folds
    folds = _generate_wf_folds(n_total=343, min_train=104, step=26,
                               holdout_start=317)
    for a, b in zip(folds, folds[1:]):
        assert b[0] > a[0], f"train_end not monotonic: {a[0]} → {b[0]}"


# ══════════════════════════════════════════════════════════════════════════
# Panel-leakage generic guard — future-proof for (date, gu_nm) re-introduction
# ══════════════════════════════════════════════════════════════════════════

def test_panel_split_key_disjointness_helper():
    """(date, gu_nm) key 기반 disjointness 유틸 자체 smoke.

    현재 runner 는 서울 집계 weekly 로 학습하지만 metapop_seir / per-gu 실험
    경로에서 panel split 을 재도입할 때 회귀 방지를 위한 최소 검증.
    """
    gus = ["강남구", "종로구", "마포구"]
    dates = [f"2024-W{w:02d}" for w in range(1, 21)]
    keys = [(d, g) for d in dates for g in gus]

    n = len(keys)
    tr, va, te = keys[:int(0.7 * n)], keys[int(0.7 * n):int(0.85 * n)], keys[int(0.85 * n):]

    s_tr, s_va, s_te = set(tr), set(va), set(te)
    assert not (s_tr & s_va)
    assert not (s_tr & s_te)
    assert not (s_va & s_te)
    assert s_tr | s_va | s_te == set(keys)


def test_panel_split_detects_within_entity_temporal_shuffle():
    """동일 gu_nm 안에서 주가 섞여 train∩val 이 발생하면 감지된다."""
    train_keys = {(f"2024-W{w:02d}", "강남구") for w in range(1, 8)}
    val_keys = {(f"2024-W{w:02d}", "강남구") for w in [5, 8, 9]}    # W05 누출
    overlap = train_keys & val_keys
    assert overlap == {("2024-W05", "강남구")}, (
        f"panel leakage detection failed: expected W05 overlap, got {overlap}"
    )


# ══════════════════════════════════════════════════════════════════════════
# S1-1 — fold-wise above_threshold recode
# ══════════════════════════════════════════════════════════════════════════

def test_above_threshold_recode_uses_fold_train_median():
    """fold 별 median(y[:train_end]) 기반 threshold 로 above_threshold 재계산 확인."""
    from simulation.pipeline.wfcv import _recode_above_threshold_per_fold

    # Synthetic ili: 점점 커지는 추세 (fold 초기/후기 중앙값 크게 달라야 함)
    y = np.arange(1, 101, dtype=np.float64)          # 1..100
    feature_cols = ["x0", "above_threshold", "x1"]
    X = np.zeros((100, 3), dtype=np.float64)
    X[:, 1] = 99.0   # build-time 값 자리 표시자 (어떤 값이든 덮어써져야 함)

    # train_end=20 → median(y[:20])=10.5 → threshold=21 → above[i] = (y[i]>21)
    X_f = _recode_above_threshold_per_fold(X, y, feature_cols, train_end=20)
    expected_above = (y > 21.0).astype(np.float64)
    expected_rolled = np.roll(expected_above, 1); expected_rolled[0] = 0.0
    np.testing.assert_array_equal(X_f[:, 1], expected_rolled)

    # train_end=80 → median(y[:80])=40.5 → threshold=81
    X_f2 = _recode_above_threshold_per_fold(X, y, feature_cols, train_end=80)
    expected2 = (y > 81.0).astype(np.float64)
    expected2_rolled = np.roll(expected2, 1); expected2_rolled[0] = 0.0
    np.testing.assert_array_equal(X_f2[:, 1], expected2_rolled)

    # fold 가 달라지면 threshold 도 달라져야 한다 (sanity)
    assert not np.array_equal(X_f[:, 1], X_f2[:, 1])


def test_above_threshold_recode_passthrough_when_column_absent():
    """feature_cols 에 above_threshold 가 없으면 입력 배열을 그대로 반환한다."""
    from simulation.pipeline.wfcv import _recode_above_threshold_per_fold
    y = np.arange(50, dtype=np.float64)
    feature_cols = ["x0", "x1", "x2"]
    X = np.random.default_rng(0).standard_normal((50, 3))
    X_f = _recode_above_threshold_per_fold(X, y, feature_cols, train_end=20)
    # identity (array-wise) — recode must not mutate when column missing
    np.testing.assert_array_equal(X_f, X)


def test_above_threshold_recode_does_not_mutate_input():
    """원본 X 는 절대 덮어쓰지 않는다 (copy-on-write 불변식)."""
    from simulation.pipeline.wfcv import _recode_above_threshold_per_fold
    y = np.arange(1, 51, dtype=np.float64)
    feature_cols = ["above_threshold"]
    X = np.full((50, 1), 99.0, dtype=np.float64)
    X_snapshot = X.copy()
    _ = _recode_above_threshold_per_fold(X, y, feature_cols, train_end=10)
    np.testing.assert_array_equal(X, X_snapshot)


def test_above_threshold_recode_guards_tiny_train_end():
    """train_end<10 이면 build-time 값을 유지한다 (stable median 불가)."""
    from simulation.pipeline.wfcv import _recode_above_threshold_per_fold
    y = np.arange(1, 51, dtype=np.float64)
    feature_cols = ["above_threshold"]
    X = np.full((50, 1), 42.0, dtype=np.float64)
    X_f = _recode_above_threshold_per_fold(X, y, feature_cols, train_end=5)
    np.testing.assert_array_equal(X_f, X)


# ══════════════════════════════════════════════════════════════════════════
# S1-1 — quantile encoding coverage invariant (CAUSALITY_AUDIT §4)
#
# Protects against silent drift: if someone adds a new
#     _add_quantile_encoding(df, "<col>", n_bins=<k>, ...)
# call to builder.py without updating `_QUANTILE_SPECS` in phase6_wfcv, the
# new *_qbin / *_qnorm columns keep their build-time bins (which use the
# global train_end) on every fold — silent look-ahead leakage for early
# folds. This AST-level test makes the failure mode loud.
# ══════════════════════════════════════════════════════════════════════════

def test_quantile_specs_cover_every_builder_call():
    """Every _add_quantile_encoding(col, n_bins=k) in builder.py must
    appear in phase6_wfcv._QUANTILE_SPECS so the per-fold recode can
    rebuild its bins."""
    import ast
    from pathlib import Path
    from simulation.pipeline.wfcv import _QUANTILE_SPECS

    builder_path = (
        Path(__file__).resolve().parent.parent
        / "models" / "feature_engine" / "builder.py"
    )
    assert builder_path.exists(), f"builder.py not found at {builder_path}"
    tree = ast.parse(builder_path.read_text(encoding="utf-8"))

    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        fn_name = (
            fn.attr if isinstance(fn, ast.Attribute)
            else (fn.id if isinstance(fn, ast.Name) else None)
        )
        if fn_name != "_add_quantile_encoding":
            continue
        # positional: (df, col_name, n_bins=...)
        if len(node.args) < 2 or not isinstance(node.args[1], ast.Constant):
            continue
        col_name = node.args[1].value
        n_bins = None
        for kw in node.keywords:
            if kw.arg == "n_bins" and isinstance(kw.value, ast.Constant):
                n_bins = kw.value.value
        if n_bins is None:
            # fallback: 3rd positional
            if len(node.args) >= 3 and isinstance(node.args[2], ast.Constant):
                n_bins = node.args[2].value
        assert n_bins is not None, (
            f"_add_quantile_encoding({col_name!r}) has no n_bins — "
            "this test needs to be extended to parse the new call style."
        )
        found.append((col_name, int(n_bins)))

    assert found, (
        "Expected at least one _add_quantile_encoding call in builder.py; "
        "none found. Did the call site move?"
    )
    spec_set = set(_QUANTILE_SPECS)
    missing = [pair for pair in found if pair not in spec_set]
    assert not missing, (
        f"builder.py applies _add_quantile_encoding{missing!r} but "
        f"phase6_wfcv._QUANTILE_SPECS={_QUANTILE_SPECS} does not cover it. "
        "Add the (col, n_bins) pair to _QUANTILE_SPECS so the fold-wise "
        "recode can rebuild its bins; otherwise early-fold bins leak "
        "future-distribution info (see CAUSALITY_AUDIT.md §4)."
    )


# ══════════════════════════════════════════════════════════════════════════
# S1-1 — fold-wise interaction-feature max-norm recode
#
# CAUSALITY_AUDIT §2.B flagged `_add_interaction_features` global-max()
# normalization as a documented minor leak. recodes it per fold.
# These tests lock the behavior so future builder.py changes don't regress.
# ══════════════════════════════════════════════════════════════════════════

def test_interaction_recode_uses_fold_train_max():
    """fold 별 max(src[:train_end]) 기반 denominator 로 interaction 재계산."""
    from simulation.pipeline.wfcv import (
        _recode_interaction_features_per_fold,
    )
    # Synthetic subway_total_avg: step-up at t=50 so train_end=20 and
    # train_end=80 produce very different maxes.
    n = 100
    feature_cols = ["ili_rate_lag1", "subway_total_avg", "subway_ili"]
    X = np.zeros((n, 3), dtype=np.float64)
    X[:, 0] = np.arange(n, dtype=np.float64) * 0.1          # lag1 ∈ [0, 9.9]
    src = np.concatenate([
        np.full(50, 10.0, dtype=np.float64),                 # first half flat
        np.full(50, 100.0, dtype=np.float64),                # second half spike
    ])
    X[:, 1] = src
    X[:, 2] = 99.9                                            # placeholder

    # train_end=20: src[:20].max() == 10 → denom = 10 + 1
    X_f = _recode_interaction_features_per_fold(X, feature_cols, train_end=20)
    exp20 = (src / 11.0) * X[:, 0]
    np.testing.assert_allclose(X_f[:, 2], exp20, rtol=1e-10)

    # train_end=80: src[:80].max() == 100 → denom = 101
    X_f2 = _recode_interaction_features_per_fold(X, feature_cols, train_end=80)
    exp80 = (src / 101.0) * X[:, 0]
    np.testing.assert_allclose(X_f2[:, 2], exp80, rtol=1e-10)

    # Different fold, different denominator → different output
    assert not np.allclose(X_f[:, 2], X_f2[:, 2])


def test_interaction_recode_future_perturbation_invariant():
    """S1-1 증명: train_end 이후의 src 값을 교란해도 [:train_end] 구간의
    interaction 출력은 변하지 않는다 (= 미래 정보를 참조하지 않는다)."""
    from simulation.pipeline.wfcv import (
        _recode_interaction_features_per_fold,
    )
    rng = np.random.default_rng(42)
    n = 100
    feature_cols = ["ili_rate_lag1", "pop_inflow", "inflow_ili"]
    X = np.zeros((n, 3), dtype=np.float64)
    X[:, 0] = rng.uniform(0, 5, n)
    X[:, 1] = rng.uniform(0, 50, n)
    X[:, 2] = 0.0
    train_end = 40

    X_a = _recode_interaction_features_per_fold(X, feature_cols, train_end=train_end)
    # Perturb future rows of the source column
    X_pert = X.copy()
    X_pert[train_end:, 1] *= 1000.0            # huge bump after train_end
    X_b = _recode_interaction_features_per_fold(X_pert, feature_cols, train_end=train_end)
    # Past interaction output must be byte-identical
    np.testing.assert_array_equal(X_a[:train_end, 2], X_b[:train_end, 2])


def test_interaction_recode_passthrough_when_multiplier_absent():
    """feature_cols 에 ili_rate_lag1 이 없으면 입력을 그대로 반환."""
    from simulation.pipeline.wfcv import (
        _recode_interaction_features_per_fold,
    )
    feature_cols = ["pop_inflow", "inflow_ili"]
    X = np.ones((30, 2), dtype=np.float64)
    X_f = _recode_interaction_features_per_fold(X, feature_cols, train_end=10)
    np.testing.assert_array_equal(X_f, X)


def test_interaction_recode_does_not_mutate_input():
    """원본 X 를 절대 덮어쓰지 않는다 (copy-on-write 불변식)."""
    from simulation.pipeline.wfcv import (
        _recode_interaction_features_per_fold,
    )
    feature_cols = ["ili_rate_lag1", "pop_inflow", "inflow_ili"]
    X = np.ones((30, 3), dtype=np.float64)
    X[:, 0] = 2.0; X[:, 1] = 3.0; X[:, 2] = 42.0
    X_snapshot = X.copy()
    _ = _recode_interaction_features_per_fold(X, feature_cols, train_end=15)
    np.testing.assert_array_equal(X, X_snapshot)


def test_interaction_recode_guards_tiny_train_end():
    """train_end<10 이면 build-time 값을 유지한다 (안정적 max 불가)."""
    from simulation.pipeline.wfcv import (
        _recode_interaction_features_per_fold,
    )
    feature_cols = ["ili_rate_lag1", "pop_inflow", "inflow_ili"]
    X = np.ones((30, 3), dtype=np.float64)
    X[:, 2] = 777.0  # sentinel
    X_f = _recode_interaction_features_per_fold(X, feature_cols, train_end=5)
    np.testing.assert_array_equal(X_f, X)


def test_interaction_specs_cover_every_builder_interaction():
    """_add_interaction_features 의 with_columns(...alias("..._ili")) 출력 중
    `src / (src.max()+eps) * lag1` 패턴을 따르는 모든 컬럼은
    _INTERACTION_SPECS 에 등록되어야 한다. (CAUSALITY_AUDIT §2.B 보호)

    이 테스트는 SPEC 의 source 컬럼이 실제 transforms.py 소스 안에서
    `_add_interaction_features` 함수 본문에 등장하는지만 검사한다 —
    builder.py 가 새 interaction 을 추가하면 AST 매치로 누락 여부가 드러남.
    """
    import ast
    from pathlib import Path
    from simulation.pipeline.wfcv import _INTERACTION_SPECS

    transforms_path = (
        Path(__file__).resolve().parent.parent
        / "models" / "feature_engine" / "transforms.py"
    )
    src = transforms_path.read_text(encoding="utf-8")
    tree = ast.parse(src)

    interaction_fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_add_interaction_features":
            interaction_fn = node
            break
    assert interaction_fn is not None, (
        "_add_interaction_features not found in transforms.py — "
        "audit target moved, update this test."
    )
    # Collect all `.alias("..._ili")` literals inside the function body.
    aliases: set[str] = set()
    for node in ast.walk(interaction_fn):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "alias"
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and isinstance(node.args[0].value, str)
                and node.args[0].value.endswith("_ili")):
            aliases.add(node.args[0].value)

    # The non-max-norm interactions that are intentionally excluded from
    # SPEC (ratio / clip-based formulas — see phase6_wfcv._INTERACTION_SPECS
    # docstring).
    excluded = {
        "cold_ili",          # clip(upper=0).abs() · lag1
        "humid_ili",         # humidity/100 · lag1
        "peak_ratio_ili",    # already-ratio column · lag1
        "er_burden_ili",     # 1/er_bed then max (composite)
        "emp_contact_ili",   # already-ratio column · lag1
    }
    spec_outputs = {out for (out, _src, _eps) in _INTERACTION_SPECS}
    missing = aliases - excluded - spec_outputs
    assert not missing, (
        f"_add_interaction_features emits {missing!r} but "
        f"phase6_wfcv._INTERACTION_SPECS does not cover them. "
        "Either add to _INTERACTION_SPECS (max-norm pattern) or to the "
        "excluded set in this test (non-max-norm pattern) with a comment "
        "explaining the formula."
    )
