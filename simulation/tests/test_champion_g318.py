"""G-339 (옛 G-318 supersede) — LEAK-FREE 챔피언, hold-out test 선정 미사용.

옛 G-318 은 OOF top-8 shortlist 안에서 hold-out test WIS argmin 으로 1위를 골랐다 = test 로 1-of-8
선택 = winner's curse 재유입(외부 reviewer #1, 2026-06-24). G-339 는 test 를 선정서 완전 제거:
OOF 1-SE 통계동률 band(Breiman) 안에서 fold 안정성(분포이동 견고성 proxy)→parsimony→OOF-WIS.
hold-out test 는 select_champion_holdout_best 진단 병기만. (포괄 계약 = tests/test_g339_champion_leakfree.py)

macOS: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 pytest <thisfile>
"""
from simulation.pipeline.per_model_eval import (
    select_champion_g318, select_champion_holdout_best,
    _designate_best_wis_champion, CHAMPION_SHORTLIST_K,
)


def test_leak_free_picks_stable_in_band_not_test_or_noise():
    # G-339/G-386: OOF 1-SE band 안에서 OOF-노이즈 gross-outlier 는 stability-guard 로 제외, 나머지 중
    #   OOF-best 가 챔피언. (a) OOF_noise=OOF 1등이나 fold CV 가 band 중앙값 2배 초과(gross outlier=SVR-RBF
    #   패턴) → guard 로 탈락 (b) TestLucky=test 최고지만 OOF 33% 나쁨(band 밖) → winner's curse 제외.
    rows = [
        # OOF_noise: folds CV≈1.0 (gross outlier — 0.05..2.5 극단 분산), Stable CV≈0.01 → guard 제거.
        {"model": "OOF_noise", "oof_wis": 1.00, "oof_wis_folds": [0.05, 0.1, 2.5, 0.2, 0.15], "n_features": 40, "wis": 2.0},
        {"model": "Stable",    "oof_wis": 1.01, "oof_wis_folds": [1.00, 1.02, 0.99, 1.01, 1.00], "n_features": 30, "wis": 3.0},
        {"model": "TestLucky", "oof_wis": 1.50, "oof_wis_folds": [1.5, 1.5, 1.5, 1.5, 1.5], "n_features": 10, "wis": 0.1},
    ]
    champ = select_champion_g318(rows, shortlist_k=8)
    assert champ["model"] == "Stable", "G-386: gross-outlier noise-winner 는 guard 로 제외, band 내 OOF-best 챔피언"


def test_shortlist_excludes_low_oof_even_if_holdout_great():
    # winner's curse 통제: OOF 랭크 > K 인 모델은 hold-out 최고여도 챔피언 아님(shortlist 밖).
    rows = [{"model": f"m{i}", "oof_wis": float(i), "wis": 5.0} for i in range(1, 9)]  # m1..m8 = shortlist
    rows.append({"model": "lucky", "oof_wis": 100.0, "wis": 0.01})  # hold-out 최고지만 OOF 꼴찌
    champ = select_champion_g318(rows, shortlist_k=8)
    assert champ["model"] != "lucky", "winner's curse: hold-out-lucky 모델은 OOF-shortlist 밖이면 제외"
    assert champ["model"] in {f"m{i}" for i in range(1, 9)}


def test_fallback_to_oof_best_when_holdout_all_missing():
    rows = [{"model": "a", "oof_wis": 2.0, "wis": float("inf")},
            {"model": "b", "oof_wis": 1.0, "wis": float("inf")}]
    champ = select_champion_g318(rows)
    assert champ["model"] == "b", "hold-out 전부 결손이면 OOF-best fallback"


def test_meta_inf_oof_excluded_from_shortlist():
    rows = [{"model": "meta", "oof_wis": float("inf"), "wis": 0.1},   # META/결손 → 후순위
            {"model": "real", "oof_wis": 3.0, "wis": 4.0}]
    champ = select_champion_g318(rows)
    assert champ["model"] == "real"


def test_holdout_best_picks_test_wis_min_ignoring_oof():
    # 순수 hold-out best = test WIS 최저 (OOF 무시). lucky(OOF 꼴찌지만 test 최고)도 후보.
    rows = [{"model": "a", "oof_wis": 1.0, "wis": 5.0},
            {"model": "lucky", "oof_wis": 100.0, "wis": 0.5}]
    ho = select_champion_holdout_best(rows)
    assert ho["model"] == "lucky", "hold-out best 는 OOF 무관 test WIS 최저"


def test_designate_g339_leakfree_and_holdout_diagnostic():
    # G-339: champion_eligible = leak-free(band 안 fold-안정 'stable'). holdout_best = test 1위
    #   ('unstable', test=1.5), 진단 병기라 G-339 와 다를 수 있음.
    rows = [{"model": "stable",   "oof_wis": 1.00, "oof_wis_folds": [1.00, 1.01, 0.99, 1.00, 1.00], "n_features": 20, "wis": 3.0, "rank_wis": 1},
            {"model": "unstable", "oof_wis": 1.01, "oof_wis_folds": [0.1, 2.0, 0.2, 1.8, 0.15], "n_features": 25, "wis": 1.5, "rank_wis": 2}]
    g339 = _designate_best_wis_champion(rows)
    ho = select_champion_holdout_best(rows)
    assert g339["model"] == "stable"                          # leak-free: band 안 안정 모델
    assert ho["model"] == "unstable"                          # hold-out best = test 1위 (진단, G-339와 다름)
    s = next(r for r in rows if r["model"] == "stable")
    u = next(r for r in rows if r["model"] == "unstable")
    assert s["champion_eligible"] is True and s["champion_best_wis"] is True
    assert s["champion_holdout_best"] is False                # test-best 아님(진단 분리)
    assert u["champion_eligible"] is False and u["champion_holdout_best"] is True


def test_designate_flags_disagreement():
    # G-318(shortlist 내 gen) ≠ hold-out best(lucky, OOF 밖) → 둘 다 플래그
    rows = [{"model": f"m{i}", "oof_wis": float(i), "wis": 5.0, "rank_wis": i} for i in range(1, 9)]
    rows.append({"model": "lucky", "oof_wis": 100.0, "wis": 0.1, "rank_wis": 9})
    g318 = _designate_best_wis_champion(rows)
    ho = select_champion_holdout_best(rows)
    assert g318["model"] != "lucky"        # G-318 shortlist 밖 제외
    assert ho["model"] == "lucky"          # hold-out best 는 lucky
    assert next(r for r in rows if r["model"] == "lucky")["champion_holdout_best"] is True
    assert next(r for r in rows if r["model"] == "lucky")["champion_eligible"] is False


def test_rerank_cli_uses_same_function():
    # consistency: rerank_champion.py 가 동일 select_champion_g318 SSOT 를 import
    import simulation.scripts.rerank_champion as rc
    assert hasattr(rc, "select_champion_g318"), "rerank CLI must reuse the shared G-318 selector"
    assert rc.SHORTLIST_K == CHAMPION_SHORTLIST_K, "shortlist K must be the single SSOT value"
