"""feature_select_corr1se TDD — LIVE feature 선택 함수 단위 검증.

(2026-06-01 청소: 옛 derive_k_bounds/pick_k_1se/top_k_indices/select_features_fixed_epv/corr1se
 (폐기된 1-SE/EPV size-search) + 그 테스트 제거 — codex+Gemini 청소. 아래는 LIVE 함수만 검증:
 select_features_stability(재표본 빈도·n-adaptive) / forward_select·backward_select(wrapper) /
 make_model_importance_fn(model-based importance) / feature_guard_keep(Stage-2 guard).)

run: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python -m pytest <this> -q
"""
import numpy as np
import pytest

from simulation.pipeline.feature_select_corr1se import (
    forward_select, backward_select, select_features_stability,
)

pytestmark = pytest.mark.filterwarnings("ignore")


def _signal_data(n, p=80, n_signal=60, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    X = rng.normal(size=(n, p))
    for j in range(n_signal):                       # n_signal 개 feature = y + 노이즈 (상관 큼)
        X[:, j] = y + (0.3 + 0.02 * j) * rng.normal(size=n)
    return X, y


# ── forward / backward wrapper: 합성 정답 복원 증명 ────────────────────
def _synth_score(true_set, irr_pen=0.1):
    """score = (정답 누락 수)·1.0 + (무관 포함 수)·irr_pen — 정확히 true_set 서 최소(=0)."""
    ts = set(true_set)
    def score(idx):
        s = set(idx)
        return len(ts - s) * 1.0 + len(s - ts) * irr_pen
    return score


def test_forward_recovers_true_set():
    """forward(쌓기)가 known 정답 {1,3,5} 를 복원."""
    sel = forward_select(_synth_score({1, 3, 5}), list(range(10)), k_cap=8)
    assert set(sel) == {1, 3, 5}, f"forward 복원 실패: {sel}"


def test_backward_recovers_true_set():
    """backward(줄이기)가 전체서 무관 제거하며 정답 {1,3,5} 로 수렴."""
    sel = backward_select(_synth_score({1, 3, 5}), list(range(10)), k_min=1)
    assert set(sel) == {1, 3, 5}, f"backward 복원 실패: {sel}"


def test_forward_respects_cap():
    sel = forward_select(_synth_score({1, 3, 5}), list(range(10)), k_cap=2)
    assert len(sel) == 2 and set(sel).issubset({1, 3, 5}), f"cap 위반: {sel}"


def test_backward_reduces_from_full():
    sel = backward_select(_synth_score({1, 3, 5}), list(range(10)), k_min=1)
    assert len(sel) < 10, "backward 가 줄이지 않음"


def test_forward_stops_when_no_improvement():
    """무관만 있으면(정답 0개 관련) forward 는 1개도 안 쌓거나 최소만 (개선 없음)."""
    sel = forward_select(_synth_score(set()), list(range(10)), k_cap=8)
    assert len(sel) == 0, f"개선 없는데 추가함: {sel}"


# ── STABILITY selection (추천 메커니즘) — 합성 증명 ─────────────────────
def test_stability_recovers_strong_drops_noise():
    """강한(상관 큰) feature 0,1,2 는 빈도≥π 로 선택, noise 는 드롭."""
    X, y = _signal_data(242, p=60, n_signal=3, seed=0)   # feature 0,1,2 = y 상관 강
    out = select_features_stability(X, y, B=40, pi=0.6, epv_ratio=20, seed=1)
    sel = set(out["selected_indices"])
    print(f"\n  stability: selected={sorted(sel)} (strong 0,1,2 의 freq="
          f"{[round(out['stability'][j], 2) for j in (0, 1, 2)]})")
    assert {0, 1, 2}.issubset(sel), f"강한 feature 누락: {sel}"
    assert len(sel) < 60, "noise 전부 선택(선택 안 함)"
    assert all(out["stability"][j] >= 0.9 for j in (0, 1, 2)), "강한 feature 빈도 낮음"


def test_stability_inner_k_n_adaptive():
    """inner_k = n//epv_ratio — n 따라 자동 (하드코드 아님)."""
    o242 = select_features_stability(*_signal_data(242, p=60, n_signal=3), B=15, epv_ratio=20, seed=1)
    o500 = select_features_stability(*_signal_data(500, p=60, n_signal=3), B=15, epv_ratio=20, seed=1)
    assert o242["inner_k"] == 242 // 20 and o500["inner_k"] == 500 // 20
    assert o500["inner_k"] > o242["inner_k"], "inner_k 가 n 에 안 적응"


def test_stability_dynamic_size_not_fixed():
    """출력 size = 빈도 창발 (고정 k 아님): pi 낮추면 더 많이 선택."""
    X, y = _signal_data(242, p=60, n_signal=8, seed=0)
    s_strict = len(select_features_stability(X, y, B=30, pi=0.8, epv_ratio=20, seed=1)["selected_indices"])
    s_loose = len(select_features_stability(X, y, B=30, pi=0.4, epv_ratio=20, seed=1)["selected_indices"])
    assert s_loose >= s_strict, f"pi 완화 시 더 많이 선택돼야 (dynamic): strict={s_strict} loose={s_loose}"


def test_stability_deterministic():
    X, y = _signal_data(242, p=60, n_signal=3)
    a = select_features_stability(X, y, B=15, seed=7)["selected_indices"]
    b = select_features_stability(X, y, B=15, seed=7)["selected_indices"]
    assert a == b


# ── n-adaptive 전환 (C): 작은 n=|corr| filter, massive n=model-based per-model ─────────
# 사용자 결정(2026-06-01): "data 작다고 무시 말고 massive 대비." threshold = epv_ratio×p (도출).
from simulation.pipeline.feature_select_corr1se import _abs_corr as _ac_test


def test_stability_threshold_is_derived_from_p_and_epv():
    """model_based_min_n = epv_ratio × p_eff (도출) — 하드코드 n 아님."""
    X, y = _signal_data(300, p=50, n_signal=3, seed=0)
    out = select_features_stability(X, y, B=10, epv_ratio=20, seed=1)
    assert out["model_based_min_n"] == 20 * out["p_eff"], (
        f"threshold 가 epv×p_eff 도출 아님: {out['model_based_min_n']} vs 20×{out['p_eff']}")
    print(f"\n  derived threshold: epv(20)×p_eff({out['p_eff']}) = {out['model_based_min_n']}")


def test_stability_corr_mode_at_small_n_ignores_model():
    """작은 n (< 도출 threshold) → corr 모드, importance_fn 호출 안 함 (현 동작 보존)."""
    X, y = _signal_data(242, p=60, n_signal=3, seed=0)   # threshold=20×60=1200 > 242
    called = {"n": 0}
    def spy_imp(Xs, ys):
        called["n"] += 1
        return np.zeros(Xs.shape[1])
    out = select_features_stability(X, y, B=20, importance_fn=spy_imp, seed=1)
    assert out["mode"] == "corr", f"작은 n 인데 model_based: {out['mode']}"
    assert called["n"] == 0, "작은 n 에서 model importance 를 부르면 안 됨 (비용+과적합)"
    assert {0, 1, 2}.issubset(set(out["selected_indices"]))   # |corr| 로 신호 복원


def test_stability_model_based_when_n_exceeds_threshold():
    """n ≥ threshold + importance_fn 제공 → model_based 모드, importance 로 신호 복원."""
    X, y = _signal_data(242, p=60, n_signal=3, seed=0)
    def imp_fn(Xs, ys):
        return _ac_test(Xs, ys)            # 결정론 importance (테스트용)
    out = select_features_stability(X, y, B=20, importance_fn=imp_fn,
                                    model_based_min_n=100, seed=1)   # 강제 낮춤
    assert out["mode"] == "model_based", f"강제 threshold 인데 corr: {out['mode']}"
    assert {0, 1, 2}.issubset(set(out["selected_indices"]))


def test_stability_model_based_is_per_model():
    """서로 다른 모델 importance → 서로 다른 선택 = per-model 분화 (|corr| 로는 53모델 동일)."""
    X, y = _signal_data(242, p=60, n_signal=8, seed=0)
    def imp_A(Xs, ys):                       # 모델 A: 짝수 feature 선호
        s = _ac_test(Xs, ys).copy(); s[1::2] *= 0.05; return s
    def imp_B(Xs, ys):                       # 모델 B: 홀수 feature 선호
        s = _ac_test(Xs, ys).copy(); s[0::2] *= 0.05; return s
    a = select_features_stability(X, y, B=20, importance_fn=imp_A, model_based_min_n=100, seed=1)
    b = select_features_stability(X, y, B=20, importance_fn=imp_B, model_based_min_n=100, seed=1)
    assert set(a["selected_indices"]) != set(b["selected_indices"]), (
        f"per-model 분화 실패 (둘 다 {a['selected_indices']})")
    print(f"\n  per-model: modelA={a['selected_indices']} vs modelB={b['selected_indices']}")


def test_stability_model_based_falls_back_on_bad_importance():
    """importance_fn 이 NaN/shape mismatch/예외 → 그 subsample 은 |corr| fallback (crash 0)."""
    X, y = _signal_data(242, p=60, n_signal=3, seed=0)
    def bad_imp(Xs, ys):
        raise RuntimeError("model fit failed")
    out = select_features_stability(X, y, B=20, importance_fn=bad_imp,
                                    model_based_min_n=100, seed=1)
    assert out["mode"] == "model_based"
    assert {0, 1, 2}.issubset(set(out["selected_indices"])), "fallback 후에도 신호 복원돼야"


# ── make_model_importance_fn: 적용 모델 기반 importance 추출 (model-based stability 글루) ──
class _FakeForecaster:
    """duck-typed BaseForecaster: _model(sklearn) + fit/predict (OpenMP-safe 모델만)."""
    def __init__(self, kind):
        self.kind = kind; self._model = None; self._hidden = None
    def fit(self, X, y):
        if self.kind == "linear":
            from sklearn.linear_model import Ridge
            self._model = Ridge(alpha=1.0).fit(X, y)
        elif self.kind == "tree":
            from sklearn.tree import DecisionTreeRegressor
            self._model = DecisionTreeRegressor(max_depth=6, random_state=0).fit(X, y)
        elif self.kind == "blackbox":            # _model 없음 → permutation 경로 강제
            from sklearn.linear_model import Ridge
            self._hidden = Ridge(alpha=1.0).fit(X, y)
        elif self.kind == "raise":
            raise RuntimeError("fit boom")
        return self
    def predict(self, X):
        m = self._model if self._model is not None else self._hidden
        return m.predict(X)


def test_model_importance_linear_uses_coef():
    from simulation.pipeline.feature_select_corr1se import make_model_importance_fn
    X, y = _signal_data(120, p=20, n_signal=3, seed=0)
    imp = make_model_importance_fn(lambda: _FakeForecaster("linear"))(X, y)
    assert imp.shape[0] == 20
    assert set(np.argsort(imp)[::-1][:5]) & {0, 1, 2}, "선형 coef_ 로 강한 feature 상위"


def test_model_importance_tree_uses_feature_importances():
    from simulation.pipeline.feature_select_corr1se import make_model_importance_fn
    X, y = _signal_data(200, p=20, n_signal=3, seed=0)
    imp = make_model_importance_fn(lambda: _FakeForecaster("tree"))(X, y)
    assert imp.shape[0] == 20 and float(imp.sum()) > 0
    assert set(np.argsort(imp)[::-1][:5]) & {0, 1, 2}, "트리 importances 로 강한 feature 상위"


def test_model_importance_blackbox_uses_permutation():
    from simulation.pipeline.feature_select_corr1se import make_model_importance_fn
    X, y = _signal_data(150, p=12, n_signal=3, seed=0)
    imp = make_model_importance_fn(lambda: _FakeForecaster("blackbox"))(X, y)
    assert imp.shape[0] == 12, "permutation 이 per-feature 점수 줘야"
    assert set(np.argsort(imp)[::-1][:4]) & {0, 1, 2}, "permutation 으로 강한 feature 상위"


def test_model_importance_raise_returns_empty_for_fallback():
    from simulation.pipeline.feature_select_corr1se import make_model_importance_fn
    imp = make_model_importance_fn(lambda: _FakeForecaster("raise"))(np.zeros((10, 5)), np.zeros(10))
    assert imp.shape[0] == 0, "fit 실패 → 길이-0 → stability 가 |corr| fallback"


# ── feature_guard_keep: Stage-2 엄격 개선 guard (사용자 "이전 대비 개선 보장") ──────────
from simulation.pipeline.feature_select_corr1se import feature_guard_keep


def test_guard_keeps_subset_only_if_improves_by_margin():
    # subset 이 full 대비 크게 개선 (0.8 ≤ 1.0×0.98) → 유지
    assert feature_guard_keep(oof_full=1.0, oof_sel=0.80, rel_margin=0.02) is True
    # subset 이 약간만 개선 (0.99 > 1.0×0.98=0.98) → margin 미달 → full 복원
    assert feature_guard_keep(oof_full=1.0, oof_sel=0.99, rel_margin=0.02) is False
    # subset 이 더 나쁨 → full 복원
    assert feature_guard_keep(oof_full=1.0, oof_sel=1.10, rel_margin=0.02) is False


def test_guard_parsimony_keeps_subset_unless_clearly_worse():
    """PARSIMONY 모드(2026-06-01 사용자): subset 기본 유지, full 은 subset 이 ≥margin 명백히 나쁠 때만."""
    # 동등 → subset 유지 (strict 면 full 이었을 것)
    assert feature_guard_keep(1.0, 1.00, 0.02, prefer_subset=True) is True
    # 약간 나쁨 (1.01 ≤ 1.0×1.02) → subset 유지 (parsimony)
    assert feature_guard_keep(1.0, 1.01, 0.02, prefer_subset=True) is True
    # 명백히 나쁨 (1.05 > 1.02) → full 복원
    assert feature_guard_keep(1.0, 1.05, 0.02, prefer_subset=True) is False
    # 개선 → 당연히 subset
    assert feature_guard_keep(1.0, 0.80, 0.02, prefer_subset=True) is True


def test_guard_parsimony_vs_strict_differ_on_tie():
    """동등(개선 없음)일 때 두 모드가 다름: strict→full, parsimony→subset (= full 안 쏟아짐)."""
    assert feature_guard_keep(1.0, 1.0, 0.02, prefer_subset=False) is False   # strict → full
    assert feature_guard_keep(1.0, 1.0, 0.02, prefer_subset=True) is True     # parsimony → subset


def test_guard_parsimony_nonfinite_still_full():
    """parsimony 라도 비교 불가(subset 실패)면 full (안전)."""
    assert feature_guard_keep(1.0, float("inf"), 0.02, prefer_subset=True) is False
    assert feature_guard_keep(float("nan"), 0.5, 0.02, prefer_subset=True) is False


def test_guard_margin_zero_keeps_any_improvement():
    assert feature_guard_keep(1.0, 0.999, rel_margin=0.0) is True
    assert feature_guard_keep(1.0, 1.0, rel_margin=0.0) is True   # 동등 (≤) → 유지
    assert feature_guard_keep(1.0, 1.001, rel_margin=0.0) is False


def test_guard_nonfinite_reverts_to_full():
    assert feature_guard_keep(float("inf"), 0.5, 0.02) is False
    assert feature_guard_keep(1.0, float("nan"), 0.02) is False


def test_cumulative_guard_chain_never_degrades():
    """비교 TDD: greedy 보장 체인 — 각 단계 후보를 '개선시만 채택, 아니면 이전 유지' →
    accepted 시퀀스는 단조 비증가(=이전 대비 개선 보장, 악화 candidate 는 자동 reject).
    (사용자 파이프라인 모델 2→7 의 핵심 불변식.)"""
    candidates = [6.80, 1.67, 1.67, 1.67, 0.66, 1.20]  # 단계별 후보 OOF-WIS (마지막은 악화)
    accepted = candidates[0]
    seq = [accepted]
    for cand in candidates[1:]:
        if feature_guard_keep(accepted, cand, rel_margin=0.0):   # 개선(≤)시만 채택
            accepted = cand
        seq.append(accepted)
    # 보장: accepted 단조 비증가 (악화 후보 1.20 은 reject → 0.66 유지)
    assert all(seq[i + 1] <= seq[i] + 1e-9 for i in range(len(seq) - 1)), f"보장 위반: {seq}"
    assert seq[-1] == 0.66, f"최종 = best-so-far 여야 (악화 reject): {seq}"
    print(f"\n  보장 체인 accepted: {seq} (악화 1.20 reject → 0.66 유지)")


def test_guard_ili_like_reverts_to_full():
    """ILI 실측 유사(full 0.847 ≤ sel 0.839? 아니 — WIS 가 아니라 R² 예시면 반대. WIS lower=better:
    full OOF-WIS 가 sel 보다 **낮으면**(full 우수) subset 개선 못 함 → full 복원)."""
    # WIS lower=better: full=5.26 (우수), sel=5.31 (열위) → 개선 못 함 → full
    assert feature_guard_keep(oof_full=5.26, oof_sel=5.31, rel_margin=0.02) is False
