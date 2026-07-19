"""TDD for the nested size-path + 1-SE/parsimony guard (codex+Gemini 2026-06-01 recommendation).

The per-model feature guard was binary {STABILITY subset ~9, full ~399}. The user found this
"too extreme" (2 cases). codex+Gemini converged: the only SAFE enrichment at n=242 is an ORDERED
NESTED size-path (π ladder) chosen by a 1-SE/parsimony rule (NOT an unordered method menu, which
overfits). These tests pin:
  - build_nested_size_path: nested, ascending size, deduped, full last.
  - select_size_path_1se: picks the SMALLEST candidate within 1-SE / margin of the best (parsimony),
    not the raw argmin (which would overfit OOF noise).

Run PER-FILE on macOS: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 pytest <thisfile>
"""
import numpy as np
import pytest

from simulation.pipeline.feature_select_corr1se import (
    build_nested_size_path, select_size_path_1se, resolve_feature_path,
    derive_min_keep_from_stability)


# ── build_nested_size_path ────────────────────────────────────────────────────
def test_nested_path_is_nested_and_ascending():
    p = 10
    freq = np.zeros(p)
    freq[[0, 1]] = 0.9      # π≥0.8 keeps these
    freq[[2, 3]] = 0.7      # π≥0.6 adds these
    freq[[4, 5, 6]] = 0.5   # π≥0.4 adds these
    path = build_nested_size_path(freq, p, pi_levels=(0.8, 0.6, 0.4), min_keep=1)
    sizes = [len(c) for c in path]
    assert sizes == sorted(sizes), f"sizes must be ascending, got {sizes}"
    # strictly nested: each candidate ⊆ next
    for a, b in zip(path, path[1:]):
        assert set(a).issubset(set(b)), f"{a} not subset of {b} — path not nested"
    # full set is last
    assert path[-1] == list(range(p)), "full feature set must be the last candidate"
    # expected sizes: {0,1}=2, {0,1,2,3}=4, {0,1,2,3,4,5,6}=7, full=10
    assert sizes == [2, 4, 7, 10], f"unexpected emergent sizes {sizes}"


def test_nested_path_dedups_identical_levels():
    p = 6
    freq = np.zeros(p)
    freq[[0, 1]] = 0.95     # all three π levels keep exactly {0,1} (nothing between 0.4 and 0.95)
    path = build_nested_size_path(freq, p, pi_levels=(0.8, 0.6, 0.4), min_keep=1)
    # {0,1} appears once (deduped) + full → 2 candidates
    assert len(path) == 2, f"identical π levels must dedup; got {path}"
    assert path[0] == [0, 1] and path[-1] == list(range(p))


def test_nested_path_empty_level_falls_back_to_min_keep():
    p = 8
    freq = np.zeros(p)       # no feature reaches any π → every level empty
    freq[3] = 0.1            # feature 3 has the highest (tiny) frequency
    path = build_nested_size_path(freq, p, pi_levels=(0.8, 0.6, 0.4), min_keep=2)
    # empty levels → top-frequency min_keep=2 fallback; deduped → {top2} + full
    assert len(path) == 2
    assert len(path[0]) == 2 and path[-1] == list(range(p))


def test_min_keep_is_data_derived_from_stability_inner_k(monkeypatch):
    monkeypatch.delenv("MPH_FEAT_MIN_KEEP", raising=False)
    monkeypatch.delenv("MPH_FEAT_FLOOR", raising=False)
    meta = {"inner_k": 12, "p_eff": 80, "n_forced_mandatory": 4}
    assert derive_min_keep_from_stability(meta, p=100) == 12


def test_nested_path_uses_data_derived_min_to_block_single_feature(monkeypatch):
    monkeypatch.delenv("MPH_FEAT_MIN_KEEP", raising=False)
    monkeypatch.delenv("MPH_FEAT_FLOOR", raising=False)
    p = 20
    freq = np.zeros(p)
    freq[0] = 0.95
    meta = {"inner_k": 6, "p_eff": p, "n_forced_mandatory": 0}
    min_keep = derive_min_keep_from_stability(meta, p=p)
    path = build_nested_size_path(freq, p, pi_levels=(0.8, 0.6, 0.4), min_keep=min_keep)
    assert len(path[0]) >= 6
    assert 1 not in [len(c) for c in path]


# ── select_size_path_1se ──────────────────────────────────────────────────────
def test_1se_picks_smallest_within_margin_not_argmin():
    # idx1 is the raw best (4.0); idx0 (4.05) is within 2% margin (4.08) and SMALLER → parsimony picks idx0.
    means = [4.05, 4.00, 4.50, 6.00]
    sizes = [3, 9, 20, 399]
    chosen = select_size_path_1se(means, sizes, fold_scores=None, margin=0.02)
    assert chosen == 0, f"parsimony should pick the smaller within-margin candidate, got {chosen}"


def test_1se_picks_global_best_when_clearly_better():
    # idx0 small but clearly worse (beyond margin); idx1 is best and others worse → pick idx1.
    means = [5.00, 4.00, 4.50, 6.00]
    sizes = [3, 9, 20, 399]
    chosen = select_size_path_1se(means, sizes, fold_scores=None, margin=0.02)
    assert chosen == 1, f"should pick best when smaller candidate is clearly worse, got {chosen}"


def test_1se_smallest_is_best_picks_it():
    means = [3.00, 4.00, 4.50, 6.00]
    sizes = [3, 9, 20, 399]
    assert select_size_path_1se(means, sizes, margin=0.02) == 0


def test_1se_wide_fold_se_increases_parsimony():
    # idx1 best (4.0) but with a WIDE fold spread → SE large → threshold admits the smaller idx0 (4.3).
    means = [4.30, 4.00, 5.00, 7.00]
    sizes = [3, 9, 20, 399]
    fold_scores = [
        [4.2, 4.4, 4.3],            # idx0
        [2.0, 6.0, 4.0],            # idx1 best mean 4.0 but huge spread → big SE
        [4.9, 5.1, 5.0],
        [6.9, 7.1, 7.0],
    ]
    chosen = select_size_path_1se(means, sizes, fold_scores=fold_scores, margin=0.0, se_mult=1.0)
    assert chosen == 0, f"wide SE on best should admit the smaller candidate via 1-SE, got {chosen}"


def test_1se_narrow_fold_se_keeps_best():
    # same means but TIGHT fold spread on best → SE tiny, margin 0 → smaller idx0 (4.30) NOT admitted.
    means = [4.30, 4.00, 5.00, 7.00]
    sizes = [3, 9, 20, 399]
    fold_scores = [
        [4.29, 4.31, 4.30],
        [3.99, 4.01, 4.00],         # tiny SE
        [4.99, 5.01, 5.00],
        [6.99, 7.01, 7.00],
    ]
    chosen = select_size_path_1se(means, sizes, fold_scores=fold_scores, margin=0.0, se_mult=1.0)
    assert chosen == 1, f"tight SE should keep the best, got {chosen}"


def test_1se_handles_nonfinite():
    means = [np.inf, 4.0, np.nan, 6.0]
    sizes = [3, 9, 20, 399]
    chosen = select_size_path_1se(means, sizes, margin=0.02)
    assert chosen == 1


# ── resolve_feature_path: deep-NN family override (dl/modern-ts → binary) ──────
def test_resolve_path_binary_env_always_binary():
    assert resolve_feature_path("binary", category="tree", model_name="XGBoost") == "binary"
    assert resolve_feature_path("binary", category="dl", model_name="TabularDNN") == "binary"


def test_resolve_path_nested_for_regularized_families():
    for cat, name in [("tree", "XGBoost"), ("linear", "ElasticNet"), ("epi", "GAM-Spline"),
                      ("linear", "KRR")]:
        assert resolve_feature_path("nested", category=cat, model_name=name) == "nested", \
            f"{name}({cat}) should stay nested"


def test_resolve_path_dl_category_forced_to_binary():
    # deep-NN (dl-tabular + modern-ts all category='dl') → binary even when env=nested
    for name in ["TabularDNN", "TCN", "N-BEATS", "PatchTST", "DeepAR", "TFT"]:
        assert resolve_feature_path("nested", category="dl", model_name=name) == "binary", \
            f"{name}(dl) must fall back to binary (unreliable small-fold OOF)"


def test_resolve_path_name_fallback_when_category_missing():
    # category 누락이어도 NN 이름이면 binary (방어)
    assert resolve_feature_path("nested", category="", model_name="TabularDNN") == "binary"
    assert resolve_feature_path("nested", category="", model_name="TCN-forecaster") == "binary"
    # 비-NN 이름 + category 없음 → nested 유지
    assert resolve_feature_path("nested", category="", model_name="XGBoost") == "nested"


def test_dl_family_fallback_when_category_empty():
    """REGRESSION (2026-06-02): meta.category 가 전 모델 '' 이고 이름-힌트가 TimesNet/TiDE/N-HiTS
    를 못 잡아 nested(느림)로 빠져 phase 13 이 3h 정체. _is_dl_family(CATEGORY_MODELS family)로 근본 차단."""
    from simulation.pipeline.feature_select_corr1se import _is_dl_family
    # 과거 nested 로 잘못 빠지던 모델들 (이름-힌트 누락) — 이제 family 로 binary
    for m in ["TimesNet", "TiDE", "N-HiTS"]:
        assert resolve_feature_path("nested", category="", model_name=m) == "binary", \
            f"{m}: category='' 에서도 binary 여야 (family fallback)"
        assert _is_dl_family(m), f"{m} 은 DL family"
    # 전 DL/foundation/graph family = binary (category 없이도)
    # G-261 (2026-06-13): Chronos-2 → TimesFM-2.5/TiRex 대체 (Chronos retire).
    for m in ["Mamba", "PatchTST", "iTransformer", "N-BEATS", "TCN", "TabularDNN", "DNN",
              "GAT", "GCN", "TimesFM-2.5", "TiRex", "OverseasTransfer"]:
        assert resolve_feature_path("nested", category="", model_name=m) == "binary", m
    # 비-DL = nested 유지 (회귀 없음)
    for m in ["XGBoost", "ElasticNet", "KRR", "EpiEstim", "ARIMA", "GAM-Spline"]:
        assert resolve_feature_path("nested", category="", model_name=m) == "nested", m
        assert not _is_dl_family(m), f"{m} 은 DL family 아님"


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
