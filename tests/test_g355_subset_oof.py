"""G-355 (2026-06-25, 설계 #8): 챔피언 선정 OOF = 배포 SUBSET 의 OOF (선정=배포 일치).

근본: per_model_optimize preproc-first 경로에서 Stage-1(_stage1_preproc_optuna_inline,
feature_indices=None)이 best['oof_wis']/folds 를 FULL feature pool 로 채웠다. 이후 STABILITY
feature-guard 가 feature_indices 를 SUBSET 으로 줄이고 그 subset 의 OOF(nested=_means[_pick],
binary=_oof_sel/_oof_full)를 계산하지만 feature_indices 선택에만 쓰고 best 에 기록 안 함
→ return dict(oof_wis/oof_wis_folds)이 FULL-pool 수치 보고 → G-339 챔피언 selector 가 배포될
모델(subset)과 다른 구성의 OOF 로 1-SE band·fold안정성·parsimony 판정(선정≠배포).

fix: feature_indices 확정 직후 각 분기에서 그 subset 의 OOF(mean+folds)를 best 에 기록.

회귀 가드(핵심): guard 후 best['oof_wis'] 가 배포 subset 의 OOF(_means[_pick])와 일치.

macOS: per-file (.venv/bin/python -m pytest tests/test_g355_subset_oof.py -p no:cacheprovider).
"""
import inspect

import numpy as np

import simulation.pipeline.per_model_optimize as PMO


# ─────────────────────────────────────────────────────────────────────────
# 1. SOURCE-LEVEL 가드: subset OOF 기록 배선이 코드에 존재
# ─────────────────────────────────────────────────────────────────────────
def test_nested_branch_records_subset_oof():
    """NESTED 분기: feature_indices=_cands[_pick] 직후 best['oof_wis']=_means[_pick] 기록."""
    src = inspect.getsource(PMO)
    assert 'best["oof_wis"] = float(_means[_pick])' in src, "NESTED subset OOF 미기록"
    assert "_folds_list[_pick]" in src, "NESTED subset fold 벡터 미기록"
    assert '"subset_guard_nested"' in src, "NESTED _oof_wis_source 태그 누락"


def test_binary_branch_records_picked_subset_oof():
    """BINARY 분기: _guard_oof return_folds=True + 선택 feature set OOF 기록."""
    src = inspect.getsource(PMO)
    assert "_guard_oof(None, return_folds=True)" in src, "BINARY full OOF folds 미포착"
    assert "_guard_oof(_sel_idx, return_folds=True)" in src, "BINARY subset OOF folds 미포착"
    assert 'best["oof_wis"] = float(_picked_oof)' in src, "BINARY picked OOF 미기록"
    assert '"subset_guard_binary"' in src, "BINARY _oof_wis_source 태그 누락"


def test_subset_oof_recorded_after_feature_indices_set():
    """선정=배포 일치: 기록이 feature_indices 확정 직후(같은 분기 안)에 위치."""
    src = inspect.getsource(PMO)
    # NESTED: feature_indices = _cands[_pick] 와 best 기록 사이에 다른 분기 진입 없음
    i_fi = src.index("feature_indices = _cands[_pick]")
    i_rec = src.index('best["oof_wis"] = float(_means[_pick])')
    assert 0 < (i_rec - i_fi) < 800, "NESTED 기록이 feature_indices 확정 직후가 아님"


# ─────────────────────────────────────────────────────────────────────────
# 2. BEHAVIOURAL 가드: subset 기록 로직(=배포 OOF) 이 full-pool 과 다름을 입증
#    (NESTED 분기 계약의 self-contained 복제)
# ─────────────────────────────────────────────────────────────────────────
def _record_subset_oof_nested(best, _means, _folds_list, _pick):
    """per_model_optimize NESTED 분기 기록 계약의 복제."""
    if np.isfinite(_means[_pick]):
        best["oof_wis"] = float(_means[_pick])
        best["oof_wis_folds"] = (list(_folds_list[_pick]) if _folds_list[_pick] else None)
        best["_oof_wis_source"] = "subset_guard_nested"
    return best


def test_subset_oof_overrides_full_pool_value():
    """★ guard 후 best['oof_wis'] = 배포 subset OOF(_means[_pick]), full-pool 값 아님."""
    # Stage-1 이 채운 FULL-pool 값(선정에 쓰이면 선정≠배포)
    best = {"oof_wis": 8.291, "oof_wis_folds": [8.0, 8.5, 8.4, 8.3]}
    # guard 가 산출한 subset 사다리: pick=1 (작은 feature set, 약간 우수)
    _means = [8.291, 1.5154, 9.7]      # [full, subset_pick, tiny]
    _folds_list = [[8.0, 8.5], [1.4, 1.6, 1.5], [9.5, 9.9]]
    _pick = 1
    best = _record_subset_oof_nested(best, _means, _folds_list, _pick)
    assert best["oof_wis"] == 1.5154, "배포 subset OOF 미반영(여전히 full-pool)"
    assert best["oof_wis"] != 8.291, "선정 OOF 가 여전히 full-pool = 선정≠배포(버그)"
    assert best["oof_wis_folds"] == [1.4, 1.6, 1.5]
    assert best["_oof_wis_source"] == "subset_guard_nested"


def test_full_pick_records_full_oof_consistently():
    """1-SE 가 full(_pick=0)을 고르면 기록값도 full = 정합(선정=배포, 둘 다 full)."""
    best = {"oof_wis": 8.291, "oof_wis_folds": [8.0, 8.5, 8.4, 8.3]}
    _means = [8.291, 9.5, 10.0]
    _folds_list = [[8.0, 8.5, 8.4, 8.3], [9.4, 9.6], [9.9, 10.1]]
    _pick = 0
    best = _record_subset_oof_nested(best, _means, _folds_list, _pick)
    assert best["oof_wis"] == 8.291
    assert best["_oof_wis_source"] == "subset_guard_nested"


def test_empty_folds_normalized_to_none():
    """빈 fold 리스트는 None 으로 정규화(champion _oof_fold_cv 가 size<2 를 일관 처리)."""
    best = {"oof_wis": 8.291}
    _means = [8.291, 1.5]
    _folds_list = [[8.0, 8.5], []]   # subset fold 결손
    best = _record_subset_oof_nested(best, _means, _folds_list, 1)
    assert best["oof_wis"] == 1.5
    assert best["oof_wis_folds"] is None, "빈 folds 가 None 으로 정규화 안 됨"


def test_nonfinite_pick_leaves_best_untouched():
    """선택 subset OOF 가 비유한이면 best 미변경(do-no-harm, Stage-1 값 유지)."""
    best = {"oof_wis": 8.291, "oof_wis_folds": [8.0, 8.5]}
    _means = [8.291, float("inf")]
    _folds_list = [[8.0, 8.5], None]
    best = _record_subset_oof_nested(best, _means, _folds_list, 1)
    assert best["oof_wis"] == 8.291, "비유한 subset OOF 가 best 를 덮음(do-no-harm 위반)"
    assert "_oof_wis_source" not in best
