"""G-353 (2026-06-25, 감사 P1): count-invariant fair-OOF 는 **진단 전용**(선정 미사용).

발견: FusedEpi 는 min_data 미달 small-train fold drop → nfold=4(경쟁자 5) → stored oof_wis(regime-balanced
mean)가 2-2 even split 서 plain mean 과 항등 → outbreak penalty 면제 = 비대칭. 그러나 이를 plain-mean 으로
"고치면"(G-353 1차안) regime penalty(outbreak fold 가중)가 전 모델서 사라져 **노이즈-우승자**(SVR-RBF, fold
[0.5,0.8,3.5,1.2,1.0] 의 나쁜 3.5 평탄화)를 오히려 도움 → G-339 노이즈-거부 깨짐(test_g339 RED 로 포착).

결론(검증가 correct=False 옳음): plain-mean 선정은 **틀림**. 챔피언 선정은 robust stored regime-agg 유지
(노이즈-거부). _selection_oof_wis 는 **진단/투명성 컬럼**으로만 — fold-count 비대칭을 보고에 가시화(논문
정직성). 실측: stored 선정 챔피언=FusedEpi(noise-rejecting), fair 진단서도 band+fold-stability 로 FusedEpi.
#2 비대칭은 FusedEpi 가 5번째 fold 를 구조적으로 못 구하는 **근본 한계**(clean fix 부재) → 투명 보고가 정답.

macOS: per-file.
"""
import glob
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

from simulation.pipeline.per_model_eval import _selection_oof_wis, select_champion_g318


def test_count_invariant_uses_folds_plain_mean():
    """folds 있으면 plain-mean(+var penalty), regime no-op → stored regime-agg 와 독립."""
    from simulation.pipeline.per_model_optimize import _fold_variance_penalize
    r = {"oof_wis": 99.0, "oof_wis_folds": [1.0, 1.0, 3.0, 3.0]}   # stored 99 무시, folds 사용
    assert abs(_selection_oof_wis(r) - _fold_variance_penalize(2.0, [1.0, 1.0, 3.0, 3.0])) < 1e-9


def test_folds_none_falls_back_to_stored():
    """META(folds=None)는 stored oof_wis 로 graceful fallback(crash 없음)."""
    assert _selection_oof_wis({"oof_wis": 1.617, "oof_wis_folds": None}) == 1.617
    assert _selection_oof_wis({"oof_wis": float("inf"), "oof_wis_folds": None}) == float("inf")


@pytest.mark.skipif(not os.path.exists("simulation/results/per_model_optimal/FusedEpi.json"),
                    reason="needs trained results")
def test_real_data_fair_aggregation_champion_is_disclosed():
    """★ 회귀 가드: fair 집계 챔피언이 **CSV 에 기록**돼 있는가.

    2026-07-19 정정: 이 테스트의 옛 이름은 "fair 집계서도 챔피언=FusedEpi" 였으나
    **fair 스칼라를 쓰지 않았다** — 51행이 저장된 ``oof_wis`` 를 그대로 먹여서
    배포 경로를 재확인할 뿐이었고, 이름과 docstring 이 주장하는 바를 검증한 적이
    없다(공허 통과). 실제로 fair 집계를 적용하면 챔피언은 FusedEpi 가 아니다.

    G-353 의 결정(선정은 robust regime-agg 유지, fair 는 진단 컬럼)은 그대로 두되,
    그 결정이 의존하는 **투명성 컬럼이 실제로 배포되는지**를 여기서 강제한다.
    """
    from simulation.models.registry import DEFER_MODELS
    defer = set(DEFER_MODELS)
    rows = []
    for f in glob.glob("simulation/results/per_model_optimal/*.json"):
        n = os.path.basename(f)[:-5]
        if n == "summary" or n in defer:
            continue
        d = json.load(open(f, encoding="utf-8"))
        vm = d.get("val_metrics") or {}
        bc = d.get("best_config") or {}
        rows.append({"model": n, "oof_wis": vm.get("oof_wis", float("inf")),
                     "oof_wis_folds": vm.get("oof_wis_folds"), "n_features": bc.get("n_features"),
                     "wis": (d.get("test_metrics") or {}).get("wis")})
    champ_stored = select_champion_g318(rows)
    assert champ_stored, "stored 집계서 챔피언 선정 실패"

    fair_rows = [dict(r, oof_wis=_selection_oof_wis(r)) for r in rows]
    champ_fair = select_champion_g318(fair_rows)
    assert champ_fair, "fair 집계서 챔피언 선정 실패"

    # 두 집계가 갈리는 것 자체는 결함이 아니다(1-SE band 안 통계적 동률).
    # 결함은 그것을 **감추는** 것이다 — 배포 CSV 가 반드시 기록해야 한다.
    import csv as _csv
    csv_path = ROOT / "simulation/results/per_model_eval/per_model_metrics.csv"
    if not csv_path.exists():
        pytest.skip("per_model_metrics.csv 부재")
    with csv_path.open(encoding="utf-8") as fh:
        shipped = list(_csv.DictReader(fh))
    for col in ("selection_oof_wis", "n_oof_folds", "champion_plain_mean_agg"):
        assert col in shipped[0], (
            f"G-353 투명성 컬럼 {col} 이 배포 CSV 에 없다 — fair 스칼라를 '진단 컬럼'으로 "
            f"강등한 근거가 성립하려면 그 컬럼이 실제로 배포돼야 한다"
        )
    flagged = [r["model"] for r in shipped
               if str(r.get("champion_plain_mean_agg", "")).strip().lower() in ("true", "1")]
    assert flagged == [champ_fair["model"]], (
        f"champion_plain_mean_agg 가 실제 fair 챔피언과 불일치: CSV={flagged}, "
        f"계산={champ_fair['model']}"
    )
