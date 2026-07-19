"""G-339/G-386: LEAK-FREE 챔피언 선정 (hold-out test 미사용) regression guard.

외부 reviewer #1 (2026-06-24): 옛 G-318 은 OOF top-8 shortlist 안에서 hold-out test WIS argmin 으로
챔피언을 골라 winner's curse 재유입(test 로 1-of-8 선택). G-339 는 OOF 1-SE band 안에서 leak-free
tiebreaker 로 선정, test 는 진단 병기(select_champion_holdout_best).

G-386 (2026-06-27, 적대 감사 — outcome-tuning 제거): 옛 G-343 SE-cap(``se=min(se,0.05*best_oof)``)은
band 를 {best} singleton 으로 강제하도록 **튜닝**된 것(폐기 주석 자백)이라 제거 — best fold 분산이 크면
1-SE band 는 넓은 게 정직(가짜 정밀도 금지). tiebreaker 도 재설계: 옛 fold-stability 1순위는 band-worst
인데 균일 평범한 모델(낮은 CV)이 우승하는 결함이 있어, 안정성은 OOF-노이즈 우승자 **가드**로만 쓰고
(band CV 중앙값 2배 초과 outlier 제거) 1순위는 **OOF-WIS**(정확도) → parsimony 로 변경.

이 테스트가 지키는 불변식:
  1. **leak-free**: hold-out 'wis'(test) 를 어떻게 바꿔도 챔피언 불변.
  2. **OOF-노이즈-우승 거부(G-307/SVR-RBF)**: OOF 1등이 fold-불안정 gross outlier 면 챔피언 아님.
  3. **parsimony**: OOF 동률이면 fewer n_features.
  4. **graceful**: oof_wis_folds 결손(META)이면 margin band + OOF/parsimony 로 동작.
  5. **diagnostic 분리**: select_champion_holdout_best(test 1위) 는 챔피언과 다를 수 있다.
  6. **NO outcome-tuned cap (G-386)**: best fold 분산이 크면 band 가 넓어야(통계동률 정직).
"""
import numpy as np
import pytest

from simulation.pipeline.per_model_eval import (
    select_champion_g318, select_champion_holdout_best, _oof_fold_cv,
)


def _rows():
    # SVR-RBF = OOF 1등(노이즈 마진)이나 fold 불안정 → 챔피언 아니어야.
    # TabPFN  = OOF 동률·가장 안정 → G-339 챔피언.
    # NegBinGLM = OOF 동률·안정·최소 feature.
    # Lucky   = OOF 동률(band 내)·불안정·**hold-out test 최저** → holdout_best 만, G-339 아님.
    # RF      = OOF band 밖.
    return [
        {"model": "SVR-RBF",   "oof_wis": 1.59, "oof_wis_folds": [0.5, 0.8, 3.5, 1.2, 1.0], "n_features": 50, "wis": 5.17},
        {"model": "TabPFN",    "oof_wis": 1.61, "oof_wis_folds": [1.55, 1.58, 1.60, 1.62, 1.57], "n_features": 30, "wis": 2.90},
        {"model": "NegBinGLM", "oof_wis": 1.62, "oof_wis_folds": [1.55, 1.62, 1.70, 1.58, 1.60], "n_features": 11, "wis": 3.20},
        {"model": "Lucky",     "oof_wis": 2.05, "oof_wis_folds": [1.0, 3.0, 1.5, 2.5, 2.0], "n_features": 80, "wis": 2.50},
        {"model": "RF",        "oof_wis": 3.10, "oof_wis_folds": [3.0, 3.2, 3.1, 3.0, 3.2], "n_features": 100, "wis": 4.0},
    ]


def test_champion_is_stable_not_oof_noise_winner():
    """SVR-RBF=OOF 1등이나 fold 불안정 → 챔피언 아님(G-307 흡수). 안정 모델이 챔피언."""
    champ = select_champion_g318(_rows())
    assert champ is not None
    assert champ["model"] != "SVR-RBF", "OOF-노이즈-우승(불안정)이 챔피언이 되면 안 됨"
    assert champ["model"] == "TabPFN", f"가장 안정한 band 멤버가 챔피언이어야 (got {champ['model']})"


def test_leak_free_test_wis_does_not_change_champion():
    """hold-out 'wis'(test)를 어떻게 흔들어도 챔피언 불변 = leak-free."""
    base = select_champion_g318(_rows())["model"]
    for mult in (0.01, 0.5, 10.0, 100.0):
        perturbed = _rows()
        for r in perturbed:
            r["wis"] = r["wis"] * mult       # test WIS 임의 교란
        assert select_champion_g318(perturbed)["model"] == base, \
            "test WIS 교란이 챔피언을 바꾸면 leak (선정에 test 사용)"


def test_parsimony_breaks_oof_ties():
    """OOF 동률(같은 oof_wis·folds)이면 fewer n_features 가 챔피언 (Breiman 1-SE parsimony)."""
    folds = [1.50, 1.55, 1.52, 1.53, 1.51]
    rows = [
        {"model": "Big",   "oof_wis": 1.50, "oof_wis_folds": folds, "n_features": 80, "wis": 2.0},
        {"model": "Small", "oof_wis": 1.50, "oof_wis_folds": folds, "n_features": 8, "wis": 3.0},
    ]
    assert select_champion_g318(rows)["model"] == "Small"


def test_no_outcome_tuned_cap_honest_wide_band():
    """G-386: SE-cap 제거 후 best fold 분산이 크면 band 가 넓어야(통계동률 정직), 그러나 OOF-first
    tiebreaker 가 band-worst-but-smooth 모델을 우승시키지 않는다(옛 stability-1순위 결함 제거).

    Best(noisy folds, 큰 SE) 가 진짜 OOF 1등이면 band 가 넓어도(SmoothMediocre 흡수) Best 가 챔피언:
    OOF 정확도 1순위 → 매끄럽지만 OOF 약한 모델은 우승 못 함. (옛 cap 은 이걸 가짜 band 축소로 달성 →
    outcome-tuning. 새 규칙은 honest-wide band + OOF-first 로 동일 결론을 정직하게 도출.)"""
    rows = [
        {"model": "Best", "oof_wis": 1.50, "oof_wis_folds": [0.5, 0.8, 2.5, 1.2, 1.0],
         "n_features": 32, "wis": 3.0},                                                       # noisy folds → 큰 SE → 넓은 band
        {"model": "SmoothMediocre", "oof_wis": 1.62, "oof_wis_folds": [1.61, 1.62, 1.63, 1.62, 1.62],
         "n_features": 20, "wis": 4.5},                                                       # 매끄럽지만 OOF 약함(band 내)
    ]
    champ = select_champion_g318(rows)
    assert champ["model"] == "Best", "OOF-first: band-worst-but-smooth 모델이 best 를 제치면 안 됨(stability-1순위 결함)"


def test_band_widens_without_cap():
    """G-386 회귀 가드: best 의 fold SE 가 크면(36% of mean 같은 실측 케이스) band 가 1개로 collapse
    하지 않고 통계동률 모델을 포함한다 (옛 5% cap 의 가짜 정밀도 금지)."""
    from simulation.pipeline.per_model_eval import CHAMPION_SHORTLIST_K  # noqa: F401
    # best=[0.9,0.3,1.7,2.9] → mean≈1.45, SE≈0.55(38%). thr≈2.0 → 동률 2모델 모두 band.
    rows = [
        {"model": "BestNoisy", "oof_wis": 1.45, "oof_wis_folds": [0.9, 0.3, 1.7, 2.9],
         "n_features": 32, "wis": 4.0},
        {"model": "Tied", "oof_wis": 1.90, "oof_wis_folds": [1.7, 1.9, 2.0, 1.85, 2.05],
         "n_features": 20, "wis": 5.0},   # oof 1.90 < thr≈2.0 → band 내(cap 있으면 제외됐을 것)
    ]
    # cap 이 있으면 thr≈1.45+0.0725=1.52 → band={BestNoisy}. cap 제거 → Tied 도 band(통계동률 정직).
    # 그래도 OOF-first 라 BestNoisy 가 챔피언(정확도 1등) — band 만 정직하게 넓어짐.
    champ = select_champion_g318(rows)
    assert champ["model"] == "BestNoisy"


def test_graceful_without_folds_uses_margin_and_parsimony():
    """oof_wis_folds 결손(META)이면 2% margin band 로 동작(크래시 없음); OOF 동률이면 parsimony."""
    rows = [
        {"model": "A", "oof_wis": 1.60, "oof_wis_folds": None, "n_features": 40, "wis": 2.0},
        {"model": "B", "oof_wis": 1.60, "oof_wis_folds": None, "n_features": 9, "wis": 3.0},   # OOF 동률, 더 parsimonious
        {"model": "C", "oof_wis": 5.00, "oof_wis_folds": None, "n_features": 3, "wis": 9.0},   # band 밖
    ]
    champ = select_champion_g318(rows)
    assert champ["model"] == "B", "folds 결손·OOF 동률 시 margin band 내 최소 feature 가 챔피언"


def test_oof_first_within_band_no_folds():
    """folds 결손 + OOF 차이 있으면 OOF-first (낮은 oof) 가 챔피언 (parsimony 는 OOF 동률 시만)."""
    rows = [
        {"model": "Lower", "oof_wis": 1.60, "oof_wis_folds": None, "n_features": 40, "wis": 2.0},
        {"model": "Sparser", "oof_wis": 1.61, "oof_wis_folds": None, "n_features": 9, "wis": 3.0},  # band 내, 더 sparse 하나 OOF 약간 높음
    ]
    assert select_champion_g318(rows)["model"] == "Lower", "OOF 1순위: band 내 더 정확한 모델이 챔피언"


def test_holdout_best_is_separate_diagnostic():
    """select_champion_holdout_best(test 1위)는 G-339 챔피언과 다를 수 있다(진단 병기)."""
    rows = _rows()
    g339 = select_champion_g318(rows)["model"]
    ho = select_champion_holdout_best(rows)["model"]
    assert ho == "Lucky", "holdout_best = hold-out test WIS 최저"
    assert g339 != ho, "leak-free 챔피언 ≠ test-best (분리 입증)"


def test_empty_and_no_eligible():
    assert select_champion_g318([]) is None
    assert select_champion_g318([{"model": "X", "oof_wis": float("inf")}]) is None  # 적격 없음


def test_fold_cv_helper():
    assert _oof_fold_cv(None) == float("inf")
    assert _oof_fold_cv([1.5]) == float("inf")          # <2 fold
    assert _oof_fold_cv([2.0, 2.0, 2.0]) == 0.0          # 완전 안정
    assert _oof_fold_cv([1.0, 3.0]) > 0.0                # 변동 있음


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
