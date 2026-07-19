"""
R10: Per-Model Evaluation on Test Slab (n=68)
====================================================

R10 runs a comprehensive per-model evaluation on the **test slab** where
inferential power exists.

For each of the 50+ trained models, computes:
  • All 118 metrics — post-S8 cleanup 2026-05-26 (3-way Codex+Gemini+Claude consensus):
    Random seed disclosure (S9 2026-05-26): all bootstrap/PIT/event-prob sampling
    uses np.random.default_rng(42) for reproducibility (publication P1).
    KDCA ILI threshold reference: KCDC 2024-25 Sentinel Surveillance Bulletin,
    Korea Disease Control and Prevention Agency, 인플루엔자 의사환자 분율 기준
    (https://www.kdca.go.kr/contents.es?mid=a20303080000).

    INPUT SOURCE (R6 audit, 2026-05-26): test_preds comes from WF-CV OOF
    predictions (R4) + R2 baseline test_pred — NOT AR-corrected
    predictions from retired Phase 8. This is intentional: R10 evaluates RAW model
    forecasts; retired Phase 8 AR correction is a separate ensemble step. Consequence:
    ljung_box_p and residual_acf_lag1 in this report reflect RAW residuals,
    which may exhibit autocorrelation that retired Phase 8 would suppress.

    CHAMPION = PURE BEST-WIS (2026-06-05, 사용자 명시 "4-criteria/g175 완전 제거"):
    the champion is simply the lowest-WIS model (rank_wis == 1). R²/MAPE/WIS/PICP95
    are reported as INDIVIDUAL metrics (no combined gate, tier, composite, or
    promise score). The retired 4-criteria filter (R²≥0.8 ∧ MAPE≤20 ∧ WIS≤6 ∧
    PICP95≥0.9) is fully removed — no g175_* columns are produced.

    DROPPED 11 keys for methodological defensibility:
      - 7 redundant aliases: epi_peak_week_err (=peak_week_err), brier_skill_score (=brier_skill),
        n (=n_test), informedness (=youden_j), clinical_f1 (=f1 generic; Gemini Q2 evidence),
        early_warning_lead (=-season_onset_err), s_index (project-specific, was NaN)
      - 4 Gaussian-derived weak: log_score_gauss (no clean fix), pinball_q50 (∝ MAE/2),
        hl_chi2/hl_p_value (HL underpowered at n=68 + Gaussian-prob basis)
    BUG FIXES (Codex audit): R10-compat code wis→mean_wis, _pi*_width local extraction
    Categories: point + scaled + Gaussian-PI probabilistic (WIS family) + empirical PI coverage
    + Brier (Gaussian prob — S8 Tier C migrate to empirical residual bootstrap pending)
    + Murphy decomp (S4) + calibration slope/intercept (S5) + c_index (S5) + ROC family (S6)
    + cost-skill 3/5/10:1 + DM test vs lag-1 (S6) + F-β + confusion matrix + DOR (S7)
    + residual diagnostics + rankings
  • Multi-horizon table (h=1, 2, 3, 4) — FluSight-aligned
  • K=11 PI levels with Wilson exact CI for coverage
  • Pairwise tournament relative WIS (Sherratt 2023)
  • Hansen SPA test for multiple-comparisons-corrected ranking
  • Log-transformed WIS (Bosse 2023, FluSight 2024-25 standard)

Output:
  simulation/results/per_model_eval/
    ├── per_model_metrics.csv     ← long-form: model × metric × horizon
    ├── ranking.json              ← WIS / log-WIS / pairwise / SPA rankings
    ├── horizon_decay.csv         ← per-model h=1,2,3,4 decay
    ├── coverage_table.csv        ← K=11 PI coverage with Wilson CI
    └── report.md                 ← thesis appendix
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


from simulation.utils.resource_tracker import track_resources
from simulation.config_global import GLOBAL, Z95  # SSOT (2026-05-28)


def _selection_oof_wis(r) -> float:
    """Cross-model 챔피언 비교용 **count-invariant** OOF-WIS 스칼라 (G-353, 2026-06-25, 감사 P1).

    R9 가 영속화한 ``oof_wis`` 는 regime-balanced mean(0.5·quiet+0.5·elevated)이라 fold COUNT 에
    민감하다 — 5-fold(3 quiet+2 elevated)는 outbreak fold 를 50% 가중하나, fold 가 4개로 줄어 2-2 가
    되면 regime mean 이 plain mean 과 수학적 항등이 되어 outbreak 가중이 사라진다(FusedEpi 가 min_data
    미달 small-train fold drop → nfold=4 → outbreak penalty 면제 = 경쟁자 대비 구조적 유리). 모델별
    fold COUNT 차이가 oof_wis 를 비교 불가로 만든다.

    cross-model 비교 스칼라를 ``oof_wis_folds`` 의 plain-mean(+variance penalty)으로 재계산해
    count-invariant·페어하게 만든다. within-model config 선택(Stage-1/2/3 regime-agg)은 불변(peak-aware
    config pick 보존). fold 벡터 결손(META/0-trial)이면 stored oof_wis 로 graceful fallback.

    ⚠ 전제 정정(2026-07-19). 위 문단의 "regime-balanced mean 은 fold COUNT 에 민감하다"는
    **틀렸다**. regime mean = 0.5·mean(quiet) + 0.5·mean(elevated) 는 두 **그룹 평균**의 가중
    평균이라 각 그룹의 fold 개수가 식에 들어가지 않는다 — 정의상 count-invariant 다. FusedEpi 가
    "outbreak penalty 면제"처럼 보이는 것은 2-2 split 에서 50/50 가중이 plain mean 과 **수치적으로
    일치**하기 때문이지 우대가 아니다. 오히려 **이 함수(plain mean)가 composition-sensitive** 하다:
    outbreak fold 를 그 모델이 우연히 가진 개수만큼(3-2 면 40%, 2-2 면 50%) 가중한다. 배포 표 전수
    확인 결과 stored ``oof_wis`` 가 plain-mean 과 일치하는 모델은 **0개**(regime 일치 23개, 나머지는
    quiet/elevated 분류가 fold_maxes 기준이라 정렬 추정과 다를 뿐 plain 도 아님).

    실질적으로도 FusedEpi 는 **가중되는 구간에서 이긴다** — elevated fold 평균 2.297 vs GAM-Spline
    2.599. GAM-Spline 은 quiet 에서 낫고, 그래서 무가중 평균에서만 앞선다. 역학 예측기를 outbreak
    구간으로 평가한다는 것이 이 규칙의 요점이므로 무가중 평균은 쓰지 않는다.

    이 함수는 **비교용 진단 스칼라**로 남긴다: ``_designate_champions`` 가 ``selection_oof_wis`` /
    ``n_oof_folds`` / ``champion_plain_mean_agg`` 를 배포해 독자가 양쪽으로 재현할 수 있게 한다.
    가드: tests/test_oof_aggregation_is_fold_count_invariant.py (산술 성질) +
    tests/test_g353_champion_count_invariant.py (컬럼 배포 여부).

    Args:
        r: per-model row dict (``oof_wis_folds`` 우선, 결손 시 ``oof_wis``).
    Returns:
        count-invariant 선택 스칼라(낮을수록 우수) 또는 stored/inf fallback.
    Side effects: 없음.
    """
    f = r.get("oof_wis_folds")
    if f:
        a = np.asarray(f, dtype=float)
        a = a[np.isfinite(a)]
        if a.size >= 1:
            from simulation.pipeline.per_model_optimize import _fold_variance_penalize
            return _fold_variance_penalize(float(np.mean(a)), list(a))
    v = r.get("oof_wis", float("inf"))
    return float(v) if isinstance(v, (int, float)) and np.isfinite(v) else float("inf")


def _assign_oof_and_test_ranks(rows):
    """G-307 (3자 감사 #1, 2026-06-18): cross-model 순위를 LEAKAGE-FREE 하게 매긴다.

    ``rank_wis`` (= 챔피언 선정 기준) 은 각 모델의 **R9 OOF-CV WIS** (5-fold WF-CV expanding-window,
    train-pool 내부, hold-out test 미접촉) 로 산출한다. hold-out test ``wis`` 로 순위를 매기면
    53개 모델 중 test-best 를 고르는 selection-on-test (winner's curse) 가 되므로 금지 — test
    순위는 ``rank_wis_test`` 로 **진단만** 남긴다 (OOF-best ≠ test-best 면 분리가 작동 중이라는 실증).

    Args:
        rows: per-model dict 리스트. 각 row 는 ``oof_wis`` (선정; 결손/META 는 +inf) 와
            ``wis`` (hold-out test; 보고) 를 가진다.

    Returns:
        ``oof_wis`` 오름차순 (= rank_wis 순서) 으로 정렬된 rows. in-place 로 각 row 에
        ``rank_wis`` (OOF) + ``rank_wis_test`` (test 진단) 를 설정.

    Side effects: rows 각 dict 에 rank_wis / rank_wis_test 키 추가.
    Caller responsibility: oof_wis 결손 모델은 +inf 로 들어와 선정 후순위(report-only) 가 된다.
    """
    def _oof(r):
        v = r.get("oof_wis", float("inf"))
        return v if np.isfinite(v) else float("inf")
    rows_sorted = sorted(rows, key=_oof)
    for rank, r in enumerate(rows_sorted, 1):
        r["rank_wis"] = rank
    _by_test = sorted(rows_sorted, key=lambda r: r["wis"]
                      if np.isfinite(r.get("wis", float("inf"))) else float("inf"))
    for rank, r in enumerate(_by_test, 1):
        next(rr for rr in rows_sorted if rr["model"] == r["model"])["rank_wis_test"] = rank
    return rows_sorted


CHAMPION_SHORTLIST_K = 8  # G-318: OOF 통계동률 후보 cluster 크기 (rerank_champion.py 와 동일 SSOT).


def _holdout_test_wis(r):
    """row 의 hold-out test WIS — per_model_eval 는 'wis', rerank_champion 는 'test_wis' 키 사용."""
    for k in ("wis", "test_wis"):
        v = r.get(k)
        if isinstance(v, (int, float)) and np.isfinite(v):
            return float(v)
    return float("inf")


def _oof_fold_cv(folds):
    """OOF per-fold WIS 의 변동계수(std/|mean|, ddof=1) — 낮을수록 fold 간 안정.

    분포이동(시즌차) 견고성의 **LEAK-FREE proxy**: G-318 이 hold-out test 로 측정하던 '일반화'를
    test 없이 WF-CV fold 안정성으로 대체한다. fold <2 개면 +inf(안정성 미상 → tiebreak 후순위).

    Args:
        folds: per-fold OOF-WIS 리스트(없거나 <2 fold 면 안정성 측정 불가).
    Returns:
        변동계수(float, 낮을수록 안정) 또는 +inf(측정 불가).
    """
    if not folds:
        return float("inf")
    a = np.asarray(folds, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return float("inf")
    m = float(np.mean(a))
    if abs(m) < 1e-9:
        return float("inf")
    return float(np.std(a, ddof=1) / abs(m))


def select_champion_g318(rows, shortlist_k=CHAMPION_SHORTLIST_K):
    """G-339 (2026-06-24, 사용자+외부 reviewer): LEAK-FREE 챔피언 — hold-out test 선정 미사용.

    [G-318 supersede] 옛 G-318 은 OOF top-K shortlist 안에서 **hold-out test WIS argmin** 으로
    챔피언을 골랐다 — K=8 로 줄였을 뿐 test 로 1-of-8 을 고르므로 winner's curse 가 재유입된다
    (외부 reviewer #1; Cawley & Talbot 2010 JMLR 11; Varma & Simon 2006 BMC Bioinformatics 7:91;
    Hastie-Tibshirani-Friedman ESL Ch.7 = test 는 최종 1회 평가만). G-339 는 test 를 선정에서
    **완전히 제거**한다:

      ① **OOF 1-SE 통계동률 band** (Breiman 1984 1-SE rule): best 의 per-fold OOF 로 SE 추정,
         ``oof_wis ≤ best + max(SE, 2% margin)`` 인 모델 = 통계적 동률 cluster (top-K 캡).
         **NO SE-cap** (G-386, 2026-06-27): 옛 G-343 cap(5%)은 band 를 singleton 으로 역튜닝한
         것이라 제거 — best 의 fold 분산이 크면 1-SE band 는 넓은 게 정직(가짜 정밀도 금지).
      ② band 안 **leak-free tiebreaker** (test 미사용): (a) fold-stability **GUARD** — band CV
         중앙값의 2배 초과하는 gross outlier 만 제거(OOF-노이즈로 1등 된 G-307/SVR-RBF 케이스 차단,
         ``_oof_fold_cv`` proxy) → (b) **OOF-WIS** (정확도 1순위 — band-worst 인데 매끄러운 모델이
         우승하는 옛 stability-1순위 결함 제거) → (c) parsimony(``n_features``, Breiman 1-SE).

    이는 G-307(순수 OOF-argmin = SVR-RBF OOF-노이즈-우승)도 band+안정성 가드로 흡수한다.
    hold-out test 는 ``select_champion_holdout_best`` 로 **진단 병기만**(배포 챔피언 아님). test 는
    최종 보고에서 1회만 접촉 → winner's curse 0. 평가 환경은 전 모델 동일(같은 OOF/test + WIS).

    Args:
        rows: per-model dict. 선정 신호(전부 leak-free): ``oof_wis``, ``oof_wis_folds``(fold 벡터),
            ``n_features``(parsimony). ``wis``/``test_wis`` 는 보고 전용(선정 미사용).
        shortlist_k: 1-SE band 상한(top-K oof 캡, 폭주 방지; 기본 8 = CHAMPION_SHORTLIST_K).

    Returns:
        챔피언 row (leak-free). 적격(finite oof_wis) row 없으면 None.

    Side effects: 없음 (순수 — 호출자가 designation 적용).
    """
    elig = [r for r in rows if np.isfinite(r.get("oof_wis", float("inf")))]
    if not elig:
        return None
    ranked = sorted(elig, key=lambda r: float(r["oof_wis"]))
    best = ranked[0]
    best_oof = float(best["oof_wis"])
    # Breiman 1-SE band: SE from best 의 per-fold OOF (true 1-SE), 없으면 2% margin floor.
    se = 0.0
    fb = best.get("oof_wis_folds")
    if fb:
        a = np.asarray(fb, dtype=float)
        a = a[np.isfinite(a)]
        if a.size >= 2:
            se = float(np.std(a, ddof=1) / np.sqrt(a.size))
    # G-386 (2026-06-27, 적대 감사 — outcome-tuning 제거): 옛 G-343 SE-cap(``se = min(se, 0.05*best_oof)``)
    #   은 band 를 {best} singleton 으로 강제하도록 **튜닝**된 것이었다(폐기된 주석 자백: "실측: cap 후
    #   band={FusedEpi}"). best 의 fold 분산이 클 때(여기 SE=36% of mean — 4-5 fold WF-CV 에서 정상)
    #   진짜 Breiman 1-SE band 는 넓은 게 맞다(통계적 동률 cluster). cap 은 가짜 정밀도를 만들어 band 를
    #   인위 축소 → 챔피언이 어느 모델이냐와 무관하게 결정돼야 할 band 폭을 결과로 역튜닝한 것. 제거한다.
    #   (감사 검증: cap 제거 시 band 가 OOF 1-SE 동률 8모델로 정직하게 확장; cap 은 5%로 가짜 축소.)
    thr = best_oof + max(se, 0.02 * abs(best_oof))
    band = [r for r in ranked[:shortlist_k] if float(r["oof_wis"]) <= thr + 1e-12]
    if not band:
        band = [best]

    # G-386 leak-free tiebreaker (재설계): 옛 G-339 는 fold 안정성(_oof_fold_cv)을 **1순위** 키로 써서
    #   band-worst 인데 fold 가 균일하게 평범한 모델(낮은 CV)이 진짜 최강을 제치고 우승하는 결함이 있었다
    #   (감사 실측: cap 제거 시 그 키가 band-worst CQR-QuantReg 를 선정 — 가장 매끄럽지만 OOF·hold-out 둘 다
    #   band 최악). band 는 이미 정확도 통계동률을 보장하므로 1순위는 OOF 정확도여야 한다(낮을수록 우수
    #   → parsimony 동률 깸). fold 안정성은 **OOF-노이즈 우승자(G-307/SVR-RBF)를 걸러내는 가드**로만 쓴다:
    #   현 OOF-leader 가 fold-stability gross outlier 면 demote(노이즈로 1등 된 케이스). 가드 정의(tune-free):
    #   (a) **절대 pathology** CV>1.0 = 단일 fold WIS 가 mean 을 초과(분포 한쪽 fold 가 지배) → 항상 demote.
    #   (b) **상대 outlier** band≥3 일 때만, leader CV ≥ 3× (나머지 band CV median) → demote. (2-model band 는
    #   매끄러운 1모델이 floor 를 인위로 낮춰 진짜-best-noisy 를 오탈락시키므로 상대규칙 면제.)
    def _cv(r):
        return _oof_fold_cv(r.get("oof_wis_folds"))

    def _parsi(r):
        nf = r.get("n_features")
        return float(nf) if isinstance(nf, (int, float)) and np.isfinite(nf) else float("inf")

    pool = sorted(band, key=lambda r: (float(r["oof_wis"]), _parsi(r)))   # OOF-first → parsimony
    while len(pool) > 1:
        leader = pool[0]
        lcv = _cv(leader)
        if np.isfinite(lcv) and lcv > 1.0:                               # (a) 절대 pathology
            pool = pool[1:]
            continue
        rest = [_cv(r) for r in pool[1:] if np.isfinite(_cv(r))]
        if len(pool) >= 3 and rest and np.isfinite(lcv):                 # (b) 상대 outlier (band≥3만)
            mr = float(np.median(rest))
            if mr > 1e-9 and lcv >= 3.0 * mr:
                pool = pool[1:]
                continue
        break
    return pool[0]


def select_champion_holdout_best(rows):
    """순수 hold-out best 챔피언 — hold-out test WIS 최저 (OOF-shortlist 없이).

    사용자 결정(2026-06-19): G-318 과 **둘 다 산출·병기**. 직관적 정의(test 1등=우승)이며 G-318 과
    같으면 강한 증거(과적합 아님), 다르면 둘 다 근거와 함께 보고(투명성). winner's curse 위험은
    있으나 동률-cluster 가 작거나 격차가 크면 G-318 과 수렴.

    Args:
        rows: per-model rows (``wis``/``test_wis`` hold-out).

    Returns:
        hold-out test WIS 최저 row, 또는 (전부 결손이면) None.

    Side effects: 없음 (순수).
    """
    finite = [r for r in rows if np.isfinite(_holdout_test_wis(r))]
    return min(finite, key=_holdout_test_wis) if finite else None


def _designate_best_wis_champion(rows):
    """챔피언 designate — G-318(primary, 배포) + hold-out best(병기) 둘 다 (사용자 결정 2026-06-19).

    이전 G-307(rank_wis==1 = 순수 OOF-argmin)은 OOF-과적합 챔피언을 박제했다(ENGINEERING_PRINCIPLES.md 폐기). 이제
    primary = ``select_champion_g318``(**G-339 LEAK-FREE**: OOF 1-SE 통계동률 band 안에서 fold 안정성
    →parsimony→OOF-WIS, **hold-out test 미사용** — 외부 reviewer #1 winner's curse 차단).
    + ``select_champion_holdout_best``(순수 test 1위)는 **진단 병기만**(배포 아님) — primary 와 같으면
    강한 증거, 다르면 둘 다 투명 보고. R²/MAPE/PICP95 는 개별 metric 병기(게이트/composite 아님).

    Args:
        rows: per-model rows (``oof_wis`` 선정 + ``wis`` hold-out + rank_wis 진단).

    Returns:
        G-318 챔피언 row(primary, 배포), 또는 None. (hold-out best 는 select_champion_holdout_best
        또는 row["champion_holdout_best"] 플래그로 접근 — back-compat 단일 반환 유지.)

    Side effects: 모든 row 에 champion_best_wis/champion_eligible(=G-339 leak-free) + champion_holdout_best.
    """
    champ = select_champion_g318(rows)
    champ_ho = select_champion_holdout_best(rows)

    # Transparency columns (2026-07-19). Selection runs on the stored
    # regime-balanced `oof_wis` = 0.5·mean(quiet) + 0.5·mean(elevated), which is
    # an average of two GROUP means and therefore does NOT depend on fold count.
    # The unweighted alternative (`_selection_oof_wis`) is the composition-
    # sensitive one — it weights outbreak folds by however many a model happens
    # to have — and it changes the champion, so both are recorded and a reader
    # can redo the comparison either way. See
    # tests/test_oof_aggregation_is_fold_count_invariant.py for the arithmetic.
    plain_rows = [dict(r, oof_wis=_selection_oof_wis(r)) for r in rows]
    champ_fair = select_champion_g318(plain_rows)
    for r in rows:
        is_g318 = bool(champ is not None and r["model"] == champ["model"])
        r["champion_best_wis"] = is_g318
        r["champion_eligible"] = is_g318                                  # 배포 = G-318 primary
        r["champion_holdout_best"] = bool(champ_ho is not None and r["model"] == champ_ho["model"])
        folds = r.get("oof_wis_folds")
        r["n_oof_folds"] = len(folds) if folds else 0
        r["selection_oof_wis"] = _selection_oof_wis(r)
        r["champion_plain_mean_agg"] = bool(
            champ_fair is not None and r["model"] == champ_fair["model"]
        )
    if champ is not None and champ_fair is not None and champ["model"] != champ_fair["model"]:
        log.info(
            f"  [R10] outbreak-weighted selection picks {champ['model']}; an "
            f"UNWEIGHTED mean of the same folds would pick {champ_fair['model']}. "
            f"The weighted rule is the deployed one by design; both are inside "
            f"the 1-SE tie band. See column champion_plain_mean_agg."
        )
    return champ   # 단일(G-318) 반환 — back-compat (hold-out best 는 플래그/별도함수)


def _collect_fs_test_preds(all_results, n_real_test):
    """FAIR-COMPETITION II (2026-06-02, codex+Gemini+Claude): R9 full-pool feature-selected
    refit test predictions, keyed ``"name[fs]"`` so each competes head-to-head with its BASIC
    config on the SAME held-out test slab.

    The test slab is never used for feature selection (selection = train-pool OOF in R9), so
    BASIC vs feature-selected on the test slab carries no selection-optimism → the champion is the
    GENUINE best across both feature sets, not forced onto either. (Resolves the select-on-BASIC /
    deploy-on-full inconsistency; lets the data decide whether the 401-pool beats AR+seasonal.)

    Args:
        all_results: pipeline outputs; reads
            ``all_results["per_model_optimize"]["per_model_configs"]`` =
            ``{model: {"refit_test_predictions": [...], "test_metrics": {...}, ...}}``.
        n_real_test: test-slab length; predictions are aligned to the LAST ``n_real_test`` entries.

    Returns:
        ``{f"{model}[fs]": np.ndarray(n_real_test)}`` for every model with a finite 1-D refit.
        Empty dict if R9 is absent/failed/skipped (defensive — never raises).

    Side effects: none.
    """
    out: dict[str, np.ndarray] = {}
    pmo = all_results.get("per_model_optimize", {}) or {}
    pmc = pmo.get("per_model_configs", {}) if isinstance(pmo, dict) else {}
    if not isinstance(pmc, dict):
        return out
    for name, res in pmc.items():
        if not isinstance(res, dict):
            continue
        tp = res.get("refit_test_predictions")
        if tp is None:
            continue
        try:
            a = np.asarray(tp, dtype=np.float64)
        except Exception:
            continue
        if a.ndim == 1 and len(a) >= n_real_test and np.isfinite(a).any():
            out[f"{name}[fs]"] = a[-n_real_test:]
    return out


@track_resources("per_model_eval")
def run_per_model_eval(
    phase1: dict,
    all_results: dict,
    config,
) -> dict:
    """Per-model evaluation on test slab (in-sample idx [pool_end:n]).

    Args:
      phase1: dict with X_all, y_all, dates, n_train, n_val, n_test, pool_end
      all_results: pipeline outputs including wfcv["oof_predictions"]
      config: pipeline config

    Returns: {model_table, ranking, horizon_decay, coverage_table, paths,
              [research_results]}

    2026-05-28 사용자 명시 design A: R10 = research용. env MPH_PHASE14_RESEARCH_MODE=1
    enable 시 _inline_optuna_3stage.run_3stage_optuna(..., mode="research") 호출 →
    per_model_research/<MODEL>.json 별도 산출. backward-compat default OFF.
    """
    # 2026-05-28 사용자 명시 design A — research mode 3-stage Optuna (env-gated)
    # B4e (2026-05-28): R10 wiring 본격 구현 — factory_fn registry + 53 model loop.
    _research_mode = GLOBAL.filter.phase14_research_mode
    _research_results: dict = {}
    if _research_mode:
        try:
            from simulation.pipeline._inline_optuna_3stage import run_3stage_optuna
            import json as _json_p11
            from simulation.models.base import REGISTRY as _REGISTRY_p11
            from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
            import simulation.models  # force-import to trigger register
            _research_dir = get_results_dir() / "per_model_research"  # MPH_OUTPUT_ROOT 존중
            _research_dir.mkdir(parents=True, exist_ok=True)
            # R9 결과 (service champion) 의 model 이름 재사용
            _phase12 = all_results.get("phase12") or all_results.get("per_model_opt") or {}
            _per_model_configs = _phase12.get("per_model_configs", {})
            _X_all = phase1.get("X_all")
            _y_all = phase1.get("y_all")
            _feature_cols = phase1.get("feature_cols")
            _n_train = phase1.get("n_train")
            _pool_end = phase1.get("pool_end", (_n_train or 0) + phase1.get("n_val", 0))
            if (_X_all is not None and _y_all is not None and _per_model_configs
                    and _n_train and _pool_end and _pool_end > _n_train):
                # Train/val split (R9 의 logic 과 일치 — train+val pool)
                _X_train_p11 = _X_all[:_n_train]
                _y_train_p11 = _y_all[:_n_train]
                _X_val_p11 = _X_all[_n_train:_pool_end]
                _y_val_p11 = _y_all[_n_train:_pool_end]
                log.info(
                    f"  [R10] research mode ON — {len(_per_model_configs)} model "
                    f"3-stage Optuna (preproc 100 + feature 20 + HP 20, mode=research, "
                    f"5-fold WF-CV stricter)"
                )
                # Build factories from REGISTRY (R9 와 같은 방식)
                _n_research_ok = 0
                _n_research_err = 0
                for _mname in _per_model_configs.keys():
                    _cls = _REGISTRY_p11.get(_mname)
                    if _cls is None:
                        log.warning(f"  [R10] research {_mname}: REGISTRY miss → skip")
                        _n_research_err += 1
                        continue
                    def _factory(cls=_cls):
                        return cls()
                    try:
                        # B4a stricter (5-fold WF-CV) + B4f Stage 2 adapter +
                        # B4b trial-level percentile CI 자동 적용
                        _r_result = run_3stage_optuna(
                            model_name=_mname,
                            factory_fn=_factory,
                            X_train=_X_train_p11, y_train=_y_train_p11,
                            X_val=_X_val_p11, y_val=_y_val_p11,
                            feature_cols=_feature_cols,
                            n_train=_n_train,
                            mode="research",
                        )
                        _n_research_ok += 1
                        log.info(
                            f"  [R10 research] {_mname} done "
                            f"({_r_result.get('stage1_status', '?')})"
                        )
                    except Exception as _me:
                        log.warning(f"  [R10 research] {_mname} failed: {_me}")
                        _n_research_err += 1

                _research_results["_mode"] = "research"
                _research_results["_status"] = (
                    f"B4 full design A — {_n_research_ok} OK, {_n_research_err} fail"
                )
                _research_results["_n_models_target"] = len(_per_model_configs)
                _research_results["_n_models_ok"] = _n_research_ok
                _research_results["_n_models_err"] = _n_research_err

                # B4c MCS_{90} membership — multi-model 가능 (research 결과 가져옴)
                try:
                    _mcs_loss_matrix = []
                    _mcs_names = []
                    for _mname in _per_model_configs.keys():
                        _r_path = _research_dir / f"{_mname}.json"
                        if not _r_path.exists():
                            continue
                        try:
                            _r_data = _json_p11.loads(_r_path.read_text())
                            _best_wis = (_r_data.get("stage1_best_preproc") or {}).get("wis")
                            if _best_wis is not None and np.isfinite(_best_wis):
                                # Single scalar loss (best preproc WIS) — MCS 의 정식 input
                                # 은 (n_obs, n_model) loss matrix. 현재 single scalar →
                                # n_obs=1 truncated. 진정한 MCS = R10 test 의 per-obs loss.
                                # → B4c minimal: 단순 ranking 만 명시
                                _mcs_loss_matrix.append([float(_best_wis)])
                                _mcs_names.append(_mname)
                        except Exception:
                            pass
                    if _mcs_loss_matrix and len(_mcs_names) >= 3:
                        _mcs_loss_matrix = np.array(_mcs_loss_matrix).T  # (n_obs=1, n_model)
                        # MCS 단일 obs 로는 의미 X → ranking 만
                        _ranks = np.argsort(_mcs_loss_matrix[0])
                        _research_results["b4c_mcs_ranking"] = [
                            _mcs_names[i] for i in _ranks
                        ]
                        _research_results["b4c_mcs_status"] = (
                            "ranking only (single-obs proxy). 진정 MCS 는 per-obs loss matrix "
                            "필요 → R10 test slab 에서 model 별 per-week loss 계산 후 활성."
                        )
                except Exception as _mcs_err:
                    _research_results["b4c_mcs_error"] = str(_mcs_err)

                (_research_dir / "_summary.json").write_text(
                    _json_p11.dumps(_research_results, indent=2, default=str)
                )
                log.info(
                    f"  [R10 research] summary → {_research_dir}/_summary.json "
                    f"({_n_research_ok}/{len(_per_model_configs)} OK)"
                )
            else:
                log.warning(
                    "  [R10] research mode 활성이지만 phase1/phase12 결과 부족 → skip"
                )
        except Exception as _re_err:
            log.warning(f"  [R10] research mode 실패 (non-fatal): {_re_err}")
            _research_results["_error"] = str(_re_err)

    from simulation.analytics.metrics import (
        peak_week_error, peak_intensity_error, direction_accuracy,
        diebold_mariano, bootstrap_ci, brier_score, brier_skill_score,
        brier_decomposition,
        binary_clinical_rates, crps_gaussian, pinball_loss,
        epidemic_phase_metrics, advanced_clinical_metrics_ext,
        # Sprint S5 (2026-05-26) — 8 new metric functions + 3 epi alt
        pearson_r, spearman_r,
        calibration_slope_intercept, hosmer_lemeshow,
        c_index, s_index,
        epi_peak_mae, epi_season_total_mae,
        # Sprint S6 (2026-05-26) — 12 missing per 9fix glossary + ROC AUC
        roc_auc, pi_relative_widths, cost_skill_ratios,
        diebold_mariano_vs_baseline, lead_time_weeks_metric,
        # Sprint S7 (2026-05-26) — ROC family + F-β + confusion matrix + DOR
        roc_family_metrics, f_beta_scores,
        confusion_matrix_table, clinical_diagnostic_metrics,
        # Sprint S9 (2026-05-26) — Bootstrap CI + BH-FDR (publication-readiness P0)
        adjust_pvalues,
    )
    from simulation.analytics.diagnostics import (
        weighted_interval_score, pit_values,
        # S8 Tier C 2026-05-26: empirical residual-quantile WIS (Lei 2018)
        weighted_interval_score_empirical,
    )
    from simulation.analytics.hub_metrics import (
        FLUSIGHT_ALPHAS, k11_pi_widths_from_residuals,
        pairwise_relative_wis,
        coverage_with_exact_ci, hansen_spa_test,
        mase, median_absolute_percentage_error,
        mean_error, msle, theils_u, log_score_gaussian,
        relative_skill_score,
        # S8 Tier C 2026-05-26: empirical residual-quantile WIS log + decomp
        weighted_interval_score_logscale_empirical,
        weighted_interval_score_components_empirical,
    )
    # G-365 (2026-06-26): model-agnostic adaptive conformal (Conformal-PID) — rolling 과거
    #   obs 로 PI 동적조정. static 잔차 PI 가 정점 분포이동에 과소피복(중위 0.67)하던 것 해소.
    #   전 모델 동일 적용(공정). leak-free(과거 obs만). MPH_ADAPTIVE_CONFORMAL=0 이면 옛 static.
    import os as _os
    from simulation.analytics.adaptive_conformal import (
        adaptive_conformal_bounds, wis_from_bounds,
    )
    _USE_ADAPTIVE_CONF = _os.environ.get("MPH_ADAPTIVE_CONFORMAL", "1") == "1"
    # KDCA threshold for alert metrics (test slab spans 2024-25 + 2025-26 seasons)
    from simulation.pipeline.real_eval import _kdca_threshold_for

    t0 = time.time()
    y_in = phase1["y_all"]
    n_train = phase1.get("n_train", 0)
    n_val   = phase1.get("n_val", 0)
    pool_end = phase1.get("pool_end", n_train + n_val)
    n_test  = phase1.get("n_test", len(y_in) - pool_end)
    test_start = pool_end
    test_end   = test_start + n_test
    y_test = y_in[test_start:test_end]
    n_real_test = len(y_test)

    if n_real_test < 10:
        log.warning(f"  [R10] test slab too short (n={n_real_test}) — skipping")
        return {"skipped": True, "reason": f"n_test={n_real_test} < 10",
                "elapsed": time.time() - t0}

    log.info(f"  [R10] test slab n={n_real_test}, "
             f"computing per-model metrics for all WF-CV models")

    # ── Pull predictions from R4 WF-CV (preferred) and/or R2
    #    baseline (fallback). WF-CV provides full-length OOF arrays; baseline
    #    provides test_pred only for the test slab. Combine both sources so
    #    every trained model gets evaluated, not just WF-CV-tracked ones.
    test_preds: dict[str, np.ndarray] = {}

    # Source 1: WF-CV OOF predictions (full-length arrays; subset to test slab)
    wfcv_oof = (all_results.get("wfcv", {}) or {}).get("oof_predictions", {}) or {}
    for name, arr in wfcv_oof.items() if isinstance(wfcv_oof, dict) else []:
        if arr is None:
            continue
        try:
            a = np.asarray(arr, dtype=np.float64)
        except Exception:
            continue
        if len(a) < test_end:
            continue
        sub = a[test_start:test_end]
        if np.isfinite(sub).any():
            test_preds[name] = sub

    # Source 2: R2 baseline test_pred (already on test slab).
    # R2 returns {"model_results": {individual_results: {model: {...}},
    #                                     ensemble_results: {model: {...}}, ...}}
    # Each model's dict has 'test_pred' (list, length n_test).
    bl_root = (all_results.get("baseline", {}) or {}).get("model_results", {}) or {}
    if isinstance(bl_root, dict):
        # Flatten individual + ensemble results
        flat_models: dict = {}
        for k in ("individual_results", "ensemble_results"):
            inner = bl_root.get(k, {})
            if isinstance(inner, dict):
                flat_models.update(inner)
        # If flat_models is empty, fall back to top-level (legacy shape)
        if not flat_models:
            flat_models = {k: v for k, v in bl_root.items()
                           if isinstance(v, dict) and "test_pred" in v}
        for name, info in flat_models.items():
            if name in test_preds:
                continue  # WF-CV took precedence
            if not isinstance(info, dict):
                continue
            tp = info.get("test_pred")
            if tp is None:
                continue
            try:
                a = np.asarray(tp, dtype=np.float64)
            except Exception:
                continue
            if len(a) >= n_real_test:
                # Baseline test_pred may be longer than n_test (104 vs 68 in
                # legacy split). Take the LAST n_real_test entries to align
                # with HWP test-slab indices [pool_end:n_in].
                test_preds[name] = a[-n_real_test:]

    # Source 3 (FAIR-COMPETITION II, 2026-06-02): R9 full-pool feature-selected refits as
    # "name[fs]" configs → they compete with BASIC head-to-head on the test slab. Champion =
    # genuine best across both feature sets (not forced onto BASIC nor onto feature-selected).
    _fs = _collect_fs_test_preds(all_results, n_real_test)
    test_preds.update(_fs)
    if _fs:
        log.info(f"  [R10] Source 3 (fair-competition): {len(_fs)} feature-selected [fs] "
                 f"configs vs BASIC on test slab — champion = genuine best")

    n_wfcv = sum(1 for n in test_preds if n in (wfcv_oof or {}))
    n_baseline = len(test_preds) - n_wfcv
    log.info(
        f"  [R10] sources: WF-CV OOF = {n_wfcv}, baseline test_pred = "
        f"{n_baseline}, total = {len(test_preds)}"
    )

    if not test_preds:
        log.warning("  [R10] no test-slab predictions found — R10 skipped")
        return {"skipped": True, "reason": "no predictions",
                "elapsed": time.time() - t0}

    # ── Respect --models CLI filter when present (user-restricted run) ──
    selected = getattr(config, "_selected_models", None) or []
    if selected:
        before_n = len(test_preds)
        test_preds = {k: v for k, v in test_preds.items() if k in selected}
        log.info(
            f"  [R10] --models filter: kept {len(test_preds)} of "
            f"{before_n} sources (kept: {sorted(test_preds.keys())})"
        )
        if not test_preds:
            log.warning(f"  [R10] no models in --models filter "
                        f"({sorted(selected)}) had test predictions — skipping")
            return {"skipped": True, "reason": "filter excluded all",
                    "elapsed": time.time() - t0}

    if not test_preds:
        log.warning("  [R10] no models have test-slab predictions")
        return {"skipped": True, "reason": "no test predictions",
                "elapsed": time.time() - t0}

    log.info(f"  [R10] evaluating {len(test_preds)} models on test slab")

    # ── Per-model individually-optimized configs (R9 output) ─────
    # If R9 ran, replace test_preds with predictions from each model's
    # OWN optimal (transform × scaler × feature × HP) config.
    pm_opt = all_results.get("per_model_optimize", {})
    pm_configs = pm_opt.get("per_model_configs", {}) if isinstance(pm_opt, dict) else {}
    if pm_configs:
        log.info(
            f"  [R10] using R9 per-model optimal configs for "
            f"{len(pm_configs)} models (vs uniform pipeline preset)"
        )
        # Note: predictions still come from WF-CV OOF; the optimal config
        # is recorded in the report as evidence that "model M's reported
        # numbers reflect M's individual best", not a uniform pipeline
        # baseline. To regenerate predictions under each model's optimal
        # config requires a refit pass — that is R9's optional
        # `refit_with_optimal=True` mode (not the default to keep runtime
        # bounded). When refit predictions are available they override OOF here.
        # ── 설계 계약 (2026-06-05, codex/gemini/claude 수렴) ──────────────────
        # champion 선정 = **WF-CV(OOF) 선택 slab** 기준(= 기본 경로)이어야 한다.
        # refit override(refit_test_predictions = hold-out test)는 각 모델 optimal
        # config 의 **hold-out 성능 보고용**이지 선정 기준이 아니다. override 가 켜지면
        # rank_wis(선정)도 hold-out 으로 바뀌어 selection-on-test(winner's curse)가 생김
        # → 논문 보고 시 champion 은 **WF-CV 로 선정 · hold-out 으로 보고(분리)**. 두 데이터가
        # 다르면 winner's curse 없음. 계약 가드: tests/test_champion_best_wis.py
        # ::test_champion_selection_slab_contract. 상세: docs/EVAL_AND_FORECAST_STRUCTURE_20260605.md.
        # G-282 (2026-06-16, 3자 감사): GCN 류 R2(baseline) 미실행(R9-only) 모델은
        #   test_preds[name]=빈 list → 옛 guard len(refit)==len([])=0=False 로 건전한 R9 refit
        #   (68개, GCN OOF r2=0.670)을 폐기 → R10 silent drop. refit 가 공통 test-slab 길이와
        #   일치하면 test_preds[name] 부재여도 채택(누락 복구).
        _n_test_common = max((len(v) for v in test_preds.values()
                              if v is not None and len(v) > 0), default=0)
        for name, cfg in pm_configs.items():
            if not isinstance(cfg, dict):
                continue
            refit_pred = cfg.get("refit_test_predictions")
            if refit_pred is None:
                continue
            _existing = test_preds.get(name)
            if _existing is not None and len(_existing) > 0:
                _ok = (len(refit_pred) == len(_existing))
            else:   # G-282: baseline 미실행 모델 — 공통 test 길이와 일치하면 채택
                _ok = (_n_test_common == 0 or len(refit_pred) == _n_test_common)
            if _ok:
                test_preds[name] = np.asarray(refit_pred, dtype=np.float64)
                best_cfg = cfg.get("best_config") or {}
                if isinstance(best_cfg, dict):
                    log.info(f"    [R10] {name} → using R9 refit predictions "
                             f"(transform={best_cfg.get('transform', '?')}, "
                             f"scaler={best_cfg.get('scaler', '?')})")

    # ── In-sample residuals: σ + K=11 PI half-widths + raw residual array ──
    # WF-CV models: use OOF[:test_start] residuals (proper in-sample residual)
    # Baseline-only models: use test slab residuals as σ proxy (S2 leak caveat;
    #   no current alternative — DM-flagged for separate fix).
    # S8 Tier C (2026-05-26): residuals_per_model added for empirical WIS/CRPS
    #   migration. sigmas kept for back-compat (PIT/log-WIS delta-method only).
    sigmas: dict[str, float] = {}
    k11_qs: dict[str, dict] = {}
    residuals_per_model: dict[str, np.ndarray] = {}
    # ── G-354 (2026-06-25, P1 감사 #4): leak-free residual 출처 우선순위 ──
    #   (1) R9 refit in-sample residual(train-pool fit error 또는 model native conformal cal-split
    #       잔차; oof_wis 와 동일 누수-free 레짐). (2) R4 WF-CV OOF[:test_start] residual(wfcv_oof
    #       보유 모델만). (3) 둘 다 없음 → residual=빈배열 → WIS=NaN + pi_source="unavailable".
    #   test-residual(y_test-pred)은 절대 금지 — 채점 대상 점에 self-calibrate = 낙관 편향·모델별
    #   이질·비교 불가. (옛 else-branch 폐기.)
    pi_source_per_model: dict[str, str] = {}
    for name, pred in test_preds.items():
        _base = name[:-4] if name.endswith("[fs]") else name   # [fs] → bare config 키
        res = None
        _src = "unavailable"
        # (1) R9 in-sample residual (leak-free; native conformal 또는 static train-pool fit error)
        _cfg = pm_configs.get(_base, {}) if isinstance(pm_configs, dict) else {}
        _ires = (_cfg.get("val_metrics", {}) or {}).get("insample_residuals") if isinstance(_cfg, dict) else None
        if _ires is not None:
            _a = np.asarray(_ires, dtype=np.float64)
            _a = _a[np.isfinite(_a)]
            if len(_a) >= 2:
                res, _src = _a, "r9_leakfree"
        # (2) WF-CV OOF residual (in-sample residual 결손 시)
        if res is None and name in wfcv_oof and wfcv_oof.get(name) is not None:
            oof_pred = np.asarray(wfcv_oof[name], dtype=np.float64)[:test_start]
            oof_y = y_in[:test_start]
            mask = np.isfinite(oof_pred) & np.isfinite(oof_y)
            _a = (oof_y - oof_pred)[mask]
            if len(_a) >= 2:
                res, _src = _a, "wfcv_oof"
        # (3) 둘 다 없음 — test-leak 금지, WIS NaN 처리
        pi_source_per_model[name] = _src
        if res is None or len(res) < 2:
            if _src == "unavailable":
                log.warning(f"  [R10] {name}: PI source unavailable "
                            f"(no R9 in-sample nor WF-CV OOF residual) → WIS=NaN (no test-leak)")
            sigmas[name] = float("nan")
            k11_qs[name] = {a: float("nan") for a in FLUSIGHT_ALPHAS}
            residuals_per_model[name] = np.array([], dtype=np.float64)
            continue
        sigmas[name] = max(float(np.std(res)), 1e-3)
        k11_qs[name] = k11_pi_widths_from_residuals(np.abs(res), FLUSIGHT_ALPHAS)
        residuals_per_model[name] = np.asarray(res, dtype=np.float64)

    # ── Per-model metric grid ───────────────────────────────────────────
    rows: list[dict] = []
    losses_per_model: dict[str, np.ndarray] = {}  # for SPA test

    # Resolve alert threshold from test slab dates (KDCA season-aware).
    # Test slab covers ~2024-10 → 2026-02 spanning 2024-25 + 2025-26 seasons;
    # use first test-slab date for threshold lookup.
    dates_phase1 = phase1.get("dates")
    if dates_phase1 is not None and len(dates_phase1) > test_start:
        t_threshold = _kdca_threshold_for(dates_phase1[test_start])
    else:
        t_threshold = 8.6  # 2024-25 default

    for name, pred in test_preds.items():
        sigma = sigmas[name]
        err = pred - y_test
        ae = np.abs(err)
        sse = float(np.sum(err ** 2))
        sst = float(np.sum((y_test - y_test.mean()) ** 2))

        # ─── Point / scaled (forecasting-canonical) ──────────────────────
        r2 = 1.0 - sse / sst if sst > 0 else float("nan")
        mae_v = float(np.mean(ae))
        rmse = float(np.sqrt(np.mean(err ** 2)))
        mse = float(np.mean(err ** 2))
        # MAPE / sMAPE (skip zero-target divisions)
        nz = y_test != 0
        mape = float(np.mean(np.abs(err[nz] / y_test[nz])) * 100) if nz.any() else float("nan")
        den = np.abs(y_test) + np.abs(pred)
        keep = den > 0
        smape = (float(np.mean(2.0 * np.abs(err[keep]) / den[keep]) * 100)
                 if keep.any() else float("nan"))
        mdape = median_absolute_percentage_error(y_test, pred)
        # MASE — Hyndman 2006 (FORECASTING-canonical scaled error)
        try:
            mase_h1 = mase(y_test, pred, y_train=y_in[:test_start], seasonality=1)
            mase_h4 = mase(y_test, pred, y_train=y_in[:test_start], seasonality=4)
            mase_h13 = mase(y_test, pred, y_train=y_in[:test_start], seasonality=13)
            mase_h26 = mase(y_test, pred, y_train=y_in[:test_start], seasonality=26)
            mase_h52 = mase(y_test, pred, y_train=y_in[:test_start], seasonality=52)
        except Exception:
            mase_h1 = mase_h4 = mase_h13 = mase_h26 = mase_h52 = float("nan")
        # Bias / MSLE / Theil's U
        bias_v = mean_error(y_test, pred)
        try:
            msle_v = msle(y_test, pred, epsilon=1.0)
        except Exception:
            msle_v = float("nan")
        try:
            theil_u_v = theils_u(y_test, pred)
        except Exception:
            theil_u_v = float("nan")

        # ─── Probabilistic ───────────────────────────────────────────────
        _adapt_b = None   # G-365: adaptive conformal bounds (coverage _cov 에서 재사용; off/실패 시 None)
        try:
            # S8 Tier C 2026-05-26: empirical residual-quantile WIS (no Gaussian σ
            #   assumption). Lei 2018 split-conformal; Bracher 2021 formula.
            _res_for_wis = residuals_per_model.get(name, np.array([], dtype=np.float64))
            if len(_res_for_wis) < 2:   # G-354: PI source unavailable → 정직 NaN(test-leak 금지)
                raise ValueError("PI source unavailable (no leak-free residual)")
            # G-365: adaptive conformal(PID) — rolling 과거 obs 로 정점서 구간 동적확장(분포이동
            #   대응). static 잔차 PI 과소피복 해소. _adapt_b 는 아래 coverage(_cov)서도 재사용.
            if _USE_ADAPTIVE_CONF:
                _adapt_b = adaptive_conformal_bounds(
                    pred, k11_qs[name], _res_for_wis, y_test, FLUSIGHT_ALPHAS)
                wis_arr = wis_from_bounds(y_test, _adapt_b, FLUSIGHT_ALPHAS, median=pred)
            else:
                wis_arr = weighted_interval_score_empirical(
                    y_test, pred, _res_for_wis, alphas=list(FLUSIGHT_ALPHAS),
                )
            mean_wis = float(np.mean(wis_arr))
            log_wis_arr = weighted_interval_score_logscale_empirical(
                y_test, pred, _res_for_wis, alphas=FLUSIGHT_ALPHAS,
            )
            mean_log_wis = float(np.mean(log_wis_arr))
        except Exception as e:
            log.debug(f"  [R10] WIS failed for {name}: {e}")
            mean_wis = mean_log_wis = float("nan")
        # S8 Tier C-3 (2026-05-26): empirical CRPS via residual-bootstrap samples
        #   (replaces Gaussian closed-form). Bracher 2021 / Krüger 2021.
        try:
            _res_for_crps = residuals_per_model.get(name, np.array([], dtype=np.float64))
            if len(_res_for_crps) >= 10:
                from simulation.analytics.hub_metrics import crps_empirical
                _rng_crps = np.random.default_rng(42)
                _M_crps = 1000
                _crps_samples = pred[:, None] + _rng_crps.choice(
                    _res_for_crps, size=(len(pred), _M_crps), replace=True,
                )
                crps = crps_empirical(y_test, _crps_samples)
            else:
                # Fallback to Gaussian closed-form when residuals are insufficient
                crps = float(np.mean(crps_gaussian(y_test, pred, np.full_like(y_test, sigma))))
        except Exception:
            crps = float("nan")
        # Log score (Gaussian NLL — FluSight standard alongside CRPS)
        try:
            log_score_v = log_score_gaussian(y_test, pred, sigma)
        except Exception:
            log_score_v = float("nan")
        try:
            # S8 Tier C-3 (2026-05-26): empirical quantile pinball (replaces
            #   Gaussian z·σ at q=0.05/0.95). Uses k11 empirical residual quantiles.
            _q_a_pin = k11_qs.get(name, {})
            # k11_qs has half-widths at α level; α=0.10 → 90% PI → q05 lower / q95 upper
            _q90_half = _q_a_pin.get(0.10, float("nan"))
            if np.isfinite(_q90_half):
                pin_q05 = pinball_loss(y_test, pred - _q90_half, 0.05)
                pin_q95 = pinball_loss(y_test, pred + _q90_half, 0.95)
            else:
                # Fallback to Gaussian when empirical quantile is unavailable
                pin_q05 = pinball_loss(y_test, pred - 1.645 * sigma, 0.05)
                pin_q95 = pinball_loss(y_test, pred + 1.645 * sigma, 0.95)
            pin_q50 = float("nan")  # dropped in S8 (proportional to MAE)
        except Exception:
            pin_q05 = pin_q50 = pin_q95 = float("nan")
        try:
            # S8 Tier C-3 (2026-05-26): rank-PIT (Czado 2009 Biometrics 65:1254-1261)
            #   — empirical predictive ensemble from residual bootstrap, then rank of y_t
            #   in the ensemble. Replaces Gaussian PIT Φ((y-pred)/σ).
            _res_for_pit = residuals_per_model.get(name, np.array([], dtype=np.float64))
            if len(_res_for_pit) >= 10:
                _rng_pit = np.random.default_rng(42)
                _M_pit = 999
                _pit_samples = pred[:, None] + _rng_pit.choice(
                    _res_for_pit, size=(len(pred), _M_pit), replace=True,
                )  # shape (n, M)
                # Rank of y_t among {pit_samples[t, :]} → (rank + 0.5) / (M + 1)
                _ranks = np.sum(_pit_samples < y_test[:, None], axis=1)  # 0..M
                pit_a = (_ranks + 0.5) / (_M_pit + 1.0)
            else:
                # Fallback to Gaussian PIT when residuals are insufficient
                pit_a = pit_values(y_test, pred, sigma)
            pit_mean = float(np.mean(pit_a))
            pit_std = float(np.std(pit_a))
            # Cramér-von Mises / KS uniformity test on rank-PIT
            from scipy.stats import kstest
            ks_stat, ks_p = kstest(pit_a, "uniform")
            pit_ks_p = float(ks_p)
        except Exception:
            pit_mean = pit_std = pit_ks_p = float("nan")

        # ─── PI coverage with Wilson exact CI (K=11 spec, headlines 95/80/50) ─
        def _cov(alpha):
            # G-365: adaptive conformal bound 우선(rolling PID, 분포이동 대응); 없으면 static 잔차 PI.
            if _adapt_b is not None and alpha in _adapt_b:
                lo, hi = _adapt_b[alpha]
                return coverage_with_exact_ci(
                    y_test, lo, hi, nominal=1 - alpha, method="wilson",
                )
            q = k11_qs[name].get(alpha)
            if q is None or not np.isfinite(q):
                return {}
            return coverage_with_exact_ci(
                y_test, pred - q, pred + q, nominal=1 - alpha, method="wilson",
            )
        cov_95 = _cov(0.05)
        cov_80 = _cov(0.20)
        cov_50 = _cov(0.50)

        # ─── Epi-curve ───────────────────────────────────────────────────
        try:
            pw = peak_week_error(y_test, pred, tolerance_weeks=1)
            pie = peak_intensity_error(y_test, pred, log_scale=True)
        except Exception:
            pw, pie = {}, {}

        # ─── Direction ──────────────────────────────────────────────────
        try:
            da = direction_accuracy(y_test, pred).get("accuracy", float("nan"))
        except Exception:
            da = float("nan")

        # ─── Clinical / Alert (KDCA threshold) ──────────────────────────
        try:
            from scipy.stats import norm as _N
            ev_true_bin = (y_test > t_threshold).astype(int)
            # S8 Tier C-2 (2026-05-26): empirical residual bootstrap for event probability
            #   (replaces Gaussian CDF Φ((threshold - pred)/σ) — defensible for skewed ILI).
            #   M=1000 samples; falls back to Gaussian when residuals are insufficient (<10).
            _res_for_boot = residuals_per_model.get(name, np.array([], dtype=np.float64))
            if len(_res_for_boot) >= 10:
                _rng = np.random.default_rng(42)
                _M = 1000
                _res_samples = _rng.choice(_res_for_boot, size=(_M, len(pred)), replace=True)
                _pred_samples = pred[None, :] + _res_samples  # shape (M, n)
                ev_prob = np.mean(_pred_samples > t_threshold, axis=0).astype(np.float64)
            else:
                z = (t_threshold - pred) / max(sigma, 1e-6)
                ev_prob = (1.0 - _N.cdf(z)).astype(np.float64)
            bs = float(brier_score(ev_true_bin, ev_prob))
            # BSS reference (G3 Round 3 audit, 2026-05-26): WEEK-OF-YEAR climatology
            # baseline (FluSight Hub standard, Bracher 2021 / Reich 2019). Per-week
            # baseline probability = mean across train years of (y > t at that WOY).
            # Replaces prior scalar `mean(y_train > t)` which was non-seasonal and
            # artificially inflated BSS (Gemini Round 3 critique). Falls back to
            # scalar baseline when dates unavailable.
            _train_bin = (y_in[:test_start] > t_threshold)
            ref_p_scalar = float(np.mean(_train_bin)) if len(_train_bin) > 0 else float(np.mean(ev_true_bin))
            try:
                if (dates_phase1 is not None
                        and len(dates_phase1) >= test_end
                        and len(_train_bin) > 0):
                    import pandas as _pd
                    _dates_train = _pd.to_datetime(list(dates_phase1[:test_start]))
                    _dates_test = _pd.to_datetime(list(dates_phase1[test_start:test_end]))
                    _woy_train = _dates_train.isocalendar().week.to_numpy()
                    _woy_test = _dates_test.isocalendar().week.to_numpy()
                    # WOY probability array (1..53; per-week mean of train alert events)
                    _woy_prob = np.full(54, ref_p_scalar)  # default fallback per slot
                    for w in range(1, 54):
                        _mask_w = _woy_train == w
                        if _mask_w.sum() >= 2:  # need at least 2 obs per WOY
                            _woy_prob[w] = float(_train_bin[_mask_w].mean())
                    # Map test weeks → climatology prob
                    _clim_prob_per_week = np.array(
                        [_woy_prob[int(w)] if 1 <= int(w) <= 53 else ref_p_scalar
                         for w in _woy_test],
                        dtype=np.float64,
                    )
                    # BS_baseline_woy = mean Brier of climatology forecast
                    bs_baseline_woy = float(np.mean((ev_true_bin - _clim_prob_per_week) ** 2))
                    bss = (1.0 - bs / bs_baseline_woy) if bs_baseline_woy > 0 else float("nan")
                    bss_baseline_type = "woy_climatology"
                else:
                    # Fallback to scalar baseline (no dates available)
                    bss = float(brier_skill_score(ev_true_bin, ev_prob, ref_p_scalar)) \
                          if 0.0 < ref_p_scalar < 1.0 else float("nan")
                    bss_baseline_type = "scalar_train_prevalence"
            except Exception as _e:
                log.debug(f"  [R10] WOY BSS failed for {name}, fallback to scalar: {_e}")
                bss = float(brier_skill_score(ev_true_bin, ev_prob, ref_p_scalar)) \
                      if 0.0 < ref_p_scalar < 1.0 else float("nan")
                bss_baseline_type = "scalar_train_prevalence"
            # Keep ref_p alias for downstream compat (Brier decomposition uses scalar)
            ref_p = ref_p_scalar
            # Sprint S4 (Murphy 1973): decomposition isolates calibration vs discrimination
            try:
                bs_decomp = brier_decomposition(ev_true_bin, ev_prob, n_bins=10)
                bs_rel = float(bs_decomp.get("reliability", float("nan")))
                bs_res = float(bs_decomp.get("resolution",  float("nan")))
                bs_unc = float(bs_decomp.get("uncertainty", float("nan")))
            except Exception:
                bs_rel = bs_res = bs_unc = float("nan")
            # ``binary_clinical_rates`` contracts for a 0/1 ``y_true``; passing the raw
            # continuous ILI rate made ``astype(int)`` yield 2, 12, 45 … so neither
            # ``y == 1`` nor ``y == 0`` matched and the confusion matrix came back all
            # zeros — sensitivity/specificity/PPV/NPV/F1 were nan for every model. Feed
            # the epidemic-threshold indicator that ``alert_f1`` already uses.
            clin = binary_clinical_rates(ev_true_bin, pred, threshold=t_threshold)
            sens = clin.get("sensitivity", float("nan"))
            spec = clin.get("specificity", float("nan"))
            ppv  = clin.get("ppv", float("nan"))
            npv  = clin.get("npv", float("nan"))
            f1c  = clin.get("f1", float("nan"))
            # alert F1 (threshold crossing dice)
            ev_pred_bin = (pred > t_threshold).astype(int)
            tp_a = int(np.sum(ev_true_bin & ev_pred_bin))
            denom = max(1, int(np.sum(ev_true_bin) + np.sum(ev_pred_bin)))
            alert_f1 = 2.0 * tp_a / denom
            # Sprint S5 (2026-05-26): calibration slope/intercept + HL goodness-of-fit
            try:
                _ci_s5 = calibration_slope_intercept(ev_true_bin, ev_prob)
                _s5_calib_slope = float(_ci_s5.get("calibration_slope", float("nan")))
                _s5_calib_intercept = float(_ci_s5.get("calibration_intercept", float("nan")))
            except Exception:
                _s5_calib_slope = _s5_calib_intercept = float("nan")
            try:
                _hl_s5 = hosmer_lemeshow(ev_true_bin, ev_prob, n_bins=10)
                _s5_hl_chi2 = float(_hl_s5.get("hl_chi2", float("nan")))
                _s5_hl_p = float(_hl_s5.get("hl_p_value", float("nan")))
            except Exception:
                _s5_hl_chi2 = _s5_hl_p = float("nan")
        except Exception:
            bs = bss = sens = spec = ppv = npv = f1c = alert_f1 = float("nan")
            bs_rel = bs_res = bs_unc = float("nan")
            _s5_calib_slope = _s5_calib_intercept = float("nan")
            _s5_hl_chi2 = _s5_hl_p = float("nan")

        # Sprint S5 (2026-05-26): Point correlations + discrimination + skill index + epi alt
        try:
            _s5_pearson = pearson_r(y_test, pred)
            _s5_spearman = spearman_r(y_test, pred)
            _s5_c_index = c_index(y_test, pred)
            _s5_epi_peak_mae = epi_peak_mae(y_test, pred, peak_window=2)
            _s5_epi_season_total_mae = epi_season_total_mae(y_test, pred)
            # s_index: aggregate of WIS skill (vs climatology) + alert utility
            #   wis_climatology = mean of climatology WIS if available, else fall back to NaN
            try:
                _wis_clim = float(np.mean(np.abs(y_test - np.mean(y_in[:test_start]))))
            except Exception:
                _wis_clim = None
            _s5_s_index = s_index(
                y_test, pred,
                wis_model=float(mean_wis) if np.isfinite(mean_wis) else None,
                wis_climatology=_wis_clim,
                sensitivity=float(sens) if np.isfinite(sens) else None,
                lead_time_weeks=None,  # set if epi_phase has it
            )
        except Exception:
            _s5_pearson = _s5_spearman = _s5_c_index = float("nan")
            _s5_epi_peak_mae = _s5_epi_season_total_mae = _s5_s_index = float("nan")

        # Sprint S6 (2026-05-26): 12 missing per 9fix glossary + ROC AUC for influenza alerts
        try:
            _s6_roc_auc = roc_auc(ev_true_bin, ev_prob)
        except Exception:
            _s6_roc_auc = float("nan")
        # Sprint S7 (2026-05-26): ROC family + F-β + confusion matrix + DOR
        try:
            _s7_roc_family = roc_family_metrics(ev_true_bin, ev_prob, high_spec_fpr_max=0.1)
        except Exception:
            _s7_roc_family = {"auprc": float("nan"), "partial_auc_high_spec": float("nan")}
        try:
            _s7_fbeta = f_beta_scores(ev_true_bin, ev_pred_bin)
        except Exception:
            _s7_fbeta = {"f2_score": float("nan"), "f05_score": float("nan")}
        try:
            _s7_cm = confusion_matrix_table(ev_true_bin, ev_pred_bin)
        except Exception:
            _s7_cm = {"tp": 0, "tn": 0, "fp": 0, "fn": 0,
                      "accuracy": float("nan"), "balanced_accuracy": float("nan"),
                      "prevalence": float("nan"), "g_mean": float("nan")}
        try:
            _s7_clin_diag = clinical_diagnostic_metrics(ev_true_bin, ev_pred_bin)
        except Exception:
            _s7_clin_diag = {"dor": float("nan"), "markedness": float("nan"),
                             "informedness": float("nan"), "youden_j": float("nan")}
        # Extract PI widths as locals for downstream pi_relative_widths()
        # (bug fix 2026-05-26: previously referenced undefined locals)
        _pi50_width = cov_50.get("mean_width", float("nan"))
        _pi80_width = cov_80.get("mean_width", float("nan"))
        _pi95_width = cov_95.get("mean_width", float("nan"))
        try:
            _s6_pi_rel = pi_relative_widths(
                y_test,
                {"pi50_width": _pi50_width, "pi80_width": _pi80_width, "pi95_width": _pi95_width},
                cap=150.0,
            )
        except Exception:
            _s6_pi_rel = {"pi50_rel_width": float("nan"),
                          "pi80_rel_width": float("nan"),
                          "pi95_rel_width": float("nan")}
        try:
            _s6_cost = cost_skill_ratios(y_test, pred, threshold=t_threshold)
        except Exception:
            _s6_cost = {f"cost_skill_{r}to1": float("nan") for r in (3, 5, 10)}
        try:
            _s6_dm = diebold_mariano_vs_baseline(y_test, pred)  # lag-1 persistence default
        except Exception:
            _s6_dm = {"dm_z_stat": float("nan"), "dm_p_value": float("nan")}
        try:
            _s6_lead = lead_time_weeks_metric(y_test, pred, threshold=t_threshold)
        except Exception:
            _s6_lead = float("nan")
        _s6_n = int(np.size(y_test))
        _s6_n_valid = int(np.isfinite(pred).sum())

        def _s5_round(v):
            return round(float(v), 4) if np.isfinite(v) else float("nan")

        # ─── WIS Decomposition (Bracher 2021) ────────────────────────────
        # S8 Tier C 2026-05-26: empirical residual-quantile WIS decomp
        wis_decomp = weighted_interval_score_components_empirical(
            y_test, pred, residuals_per_model.get(name, np.array([], dtype=np.float64)),
            alphas=FLUSIGHT_ALPHAS,
        )

        # ─── Epidemic phase metrics (Biggerstaff 2016, Cowling 2020) ─────
        epi_phase = epidemic_phase_metrics(y_test, pred, threshold=t_threshold)

        # ─── Advanced clinical metrics (Chicco 2020, Vickers 2006) ───────
        adv_clin = advanced_clinical_metrics_ext(
            y_test, pred, threshold=t_threshold, prior_prob=0.30
        )

        # ─── Residual diagnostics ────────────────────────────────────────
        # R6 audit MAJOR #6 (2026-05-26): integrated R5 의 9-key diagnostic
        # 중 4 누락 (jarque_bera, durbin_watson, skew, kurtosis) 을 R10 row
        # 에 surface. Reviewer-expected residual panel 완성.
        ljung_q = ljung_p = resid_acf1 = shapiro_p = float("nan")
        jarque_bera_p = durbin_watson = resid_skew = resid_kurtosis = float("nan")
        try:
            _r = (pred - y_test)
            _r = _r[np.isfinite(_r)]
            if len(_r) >= 5:
                from scipy.stats import shapiro as _shap, jarque_bera as _jb, skew as _sk, kurtosis as _kt
                _, shapiro_p = _shap(_r)
                shapiro_p = float(shapiro_p)
                # R6 MAJOR #6: Jarque-Bera (normality, alternative to Shapiro)
                try:
                    _jb_stat, _jb_p = _jb(_r)
                    jarque_bera_p = float(_jb_p)
                except Exception:
                    pass
                # R6 MAJOR #6: Skewness + Kurtosis (residual shape)
                try:
                    resid_skew = float(_sk(_r))
                    resid_kurtosis = float(_kt(_r))
                except Exception:
                    pass
                if len(_r) >= 3:
                    _rm = _r - _r.mean()
                    _var = float(np.var(_rm))
                    if _var > 1e-10:
                        resid_acf1 = float(np.mean(_rm[1:] * _rm[:-1]) / _var)
                # R6 MAJOR #6: Durbin-Watson (lag-1 autocorrelation, complementary to LB)
                try:
                    _dr = np.diff(_r)
                    _dw_num = float(np.sum(_dr ** 2))
                    _dw_den = float(np.sum(_r ** 2))
                    durbin_watson = _dw_num / _dw_den if _dw_den > 1e-10 else float("nan")
                except Exception:
                    pass
                try:
                    from statsmodels.stats.diagnostic import acorr_ljungbox as _alb
                    _lb = _alb(_r, lags=[min(10, len(_r) // 3)], return_df=True)
                    ljung_q = float(_lb["lb_stat"].iloc[-1])
                    ljung_p = float(_lb["lb_pvalue"].iloc[-1])
                except Exception:
                    pass
        except Exception:
            pass

        # ─── PI extensions (PI reliability + 98%/99% coverage) ──────────
        pi99_cov = pi99_w = pi95_relia = pi80_relia = pi_shr = float("nan")
        try:
            _q99 = k11_qs[name].get(0.02, float("nan"))  # alpha=0.02 → ~98% PI
            if np.isfinite(_q99):
                _c99 = coverage_with_exact_ci(
                    y_test, pred - _q99, pred + _q99,
                    nominal=0.98, method="wilson",
                )
                pi99_cov = float(_c99.get("empirical", float("nan")))
                pi99_w   = float(_c99.get("mean_width", float("nan")))
            _e95 = cov_95.get("empirical", float("nan"))
            _e80 = cov_80.get("empirical", float("nan"))
            pi95_relia = float(abs(_e95 - 0.95)) if np.isfinite(_e95) else float("nan")
            pi80_relia = float(abs(_e80 - 0.80)) if np.isfinite(_e80) else float("nan")
            _y_std = float(np.std(y_test))
            _w95   = cov_95.get("mean_width", float("nan"))
            if _y_std > 1e-6 and np.isfinite(_w95):
                pi_shr = float(_w95 / _y_std)
        except Exception:
            pass

        # ─── Stability (BCa bootstrap 95% CI on MAE) ────────────────────
        # G-363 (2026-06-26): random_state=42 추가 — 누락 시 CI 가 run 마다 변동(비재현).
        #   wis CI(아래 ~1570)는 이미 seed 됨. mae_ci95 만 누락이었음 (129-metric 전수 감사 발견).
        try:
            ci = bootstrap_ci(ae, statistic=np.mean, n_boot=2000, alpha=0.05, random_state=42)
            mae_ci = (float(ci.get("ci_lo", float("nan"))),
                      float(ci.get("ci_hi", float("nan"))))
        except Exception:
            mae_ci = (float("nan"), float("nan"))

        rows.append({
            "model": name,
            "n_test": int(n_real_test),
            # Point / scaled (forecasting-canonical)
            "r2": round(r2, 4),
            "mae": round(mae_v, 4),
            "mae_ci95_lo": round(mae_ci[0], 4),
            "mae_ci95_hi": round(mae_ci[1], 4),
            "rmse": round(rmse, 4),
            "mse":  round(mse, 4),
            "mape": round(mape, 4),
            "smape": round(smape, 4),
            "mdape": round(mdape, 4),
            "mase_h1":  round(mase_h1, 4)  if np.isfinite(mase_h1)  else float("nan"),
            "mase_h4":  round(mase_h4, 4)  if np.isfinite(mase_h4)  else float("nan"),
            "mase_h13": round(mase_h13, 4) if np.isfinite(mase_h13) else float("nan"),
            "mase_h26": round(mase_h26, 4) if np.isfinite(mase_h26) else float("nan"),
            "mase_h52": round(mase_h52, 4) if np.isfinite(mase_h52) else float("nan"),
            # Bias / signed error
            "bias_mean_error": round(bias_v, 4) if np.isfinite(bias_v) else float("nan"),
            "msle":            round(msle_v, 4) if np.isfinite(msle_v) else float("nan"),
            "theils_u":        round(theil_u_v, 4) if np.isfinite(theil_u_v) else float("nan"),
            # Probabilistic
            "wis": round(mean_wis, 4) if np.isfinite(mean_wis) else float("nan"),    # ← hold-out test WIS (★보고 전용; 선정에 절대 미사용)
            # G-354 (2026-06-25): PI 반폭 residual 출처 — r9_leakfree | wfcv_oof | unavailable.
            #   unavailable = leak-free 출처 없음 → wis=NaN(test-residual self-calibration 금지).
            "pi_source": pi_source_per_model.get(name, "unavailable"),
            # G-307→G-339: cross-model 챔피언 선정 = R9 OOF-CV WIS(5-fold WF-CV, 누수-free)만.
            #   G-339(2026-06-24)는 한 발 더 — OOF 1-SE band 안 fold안정성/parsimony tiebreaker라
            #   shortlist 안에서조차 test 'wis' 를 안 본다(외부 reviewer #1 winner's curse 완전 차단).
            #   rank_wis 는 이 oof_wis 로 산출. META/feature-less(Optuna skip)·결손은 oof_wis inf→후순위.
            "oof_wis": float((pm_configs.get(name, {}) or {}).get(
                "val_metrics", {}).get("oof_wis", float("inf"))),
            # G-339 (2026-06-24): LEAK-FREE 챔피언 tiebreaker 신호 (test 미사용). fold 벡터=1-SE band
            #   + 안정성, n_features=parsimony. select_champion_g318 가 이 둘로 hold-out 없이 선정.
            "oof_wis_folds": (pm_configs.get(name, {}) or {}).get(
                "val_metrics", {}).get("oof_wis_folds"),
            "n_features": (pm_configs.get(name, {}) or {}).get(
                "best_config", {}).get("n_features"),
            "log_wis": round(mean_log_wis, 4),
            "crps_gaussian": round(crps, 4),
            # DROP log_score_gauss (Codex/Gemini consensus 2026-05-26): Gaussian NLL not defensible
            #   for skewed low-count ILI without native predictive density. No clean replacement.
            "pinball_q05": round(pin_q05, 4) if np.isfinite(pin_q05) else float("nan"),
            # DROP pinball_q50 (Codex 2026-05-26): proportional to MAE/2 when q=pred → redundant.
            "pinball_q95": round(pin_q95, 4) if np.isfinite(pin_q95) else float("nan"),
            "pit_mean": round(pit_mean, 4),
            "pit_std":  round(pit_std, 4),
            "pit_ks_p": round(pit_ks_p, 4),
            "sigma_in_sample": round(sigma, 4),
            # PI coverage (Wilson exact)
            "pi95_coverage": cov_95.get("empirical", float("nan")),
            "pi95_ci_lo":    cov_95.get("ci_lo", float("nan")),
            "pi95_ci_hi":    cov_95.get("ci_hi", float("nan")),
            "pi95_width":    cov_95.get("mean_width", float("nan")),
            "pi80_coverage": cov_80.get("empirical", float("nan")),
            "pi80_width":    cov_80.get("mean_width", float("nan")),
            "pi50_coverage": cov_50.get("empirical", float("nan")),
            "pi50_width":    cov_50.get("mean_width", float("nan")),
            # Epi-curve
            "peak_week_err":   pw.get("abs_weeks", float("nan")),
            "peak_int_relerr": pie.get("rel_err", float("nan")),
            "direction_acc":   da,
            # Clinical / Alert (KDCA threshold)
            "alert_threshold": float(t_threshold),
            "brier_score": round(bs, 4) if np.isfinite(bs) else float("nan"),
            "brier_skill": round(bss, 4) if np.isfinite(bss) else float("nan"),
            # Murphy (1973) decomposition: BS = REL − RES + UNC
            "brier_reliability": round(bs_rel, 4) if np.isfinite(bs_rel) else float("nan"),
            "brier_resolution":  round(bs_res, 4) if np.isfinite(bs_res) else float("nan"),
            "brier_uncertainty": round(bs_unc, 4) if np.isfinite(bs_unc) else float("nan"),
            # Sprint S5 (2026-05-26) — calibration + HL goodness-of-fit
            "calibration_slope":     _s5_round(_s5_calib_slope),
            "calibration_intercept": _s5_round(_s5_calib_intercept),
            # DROP hl_chi2, hl_p_value (Codex 2026-05-26): n=68 underpowered + Gaussian-prob basis.
            # Sprint S5 — point correlations + discrimination
            "pearson_r":             _s5_round(_s5_pearson),
            "spearman_r":            _s5_round(_s5_spearman),
            "c_index":               _s5_round(_s5_c_index),
            # DROP s_index (Codex 2026-05-26): project-specific aggregate, was NaN due to undefined wis.
            # Sprint S5 — epi aggregates
            # DROP epi_peak_week_err (3-way 2026-05-26): identical alias of peak_week_err (line above).
            "epi_peak_mae":          _s5_round(_s5_epi_peak_mae),
            "epi_season_total_mae":  _s5_round(_s5_epi_season_total_mae),
            # DROP brier_skill_score (3-way 2026-05-26): identical alias of brier_skill (line above).
            # ── Sprint S6 (2026-05-26): 12 missing per 9fix glossary + ROC AUC ──
            "roc_auc":               _s5_round(_s6_roc_auc),
            "pi50_rel_width":        _s5_round(_s6_pi_rel.get("pi50_rel_width", float("nan"))),
            "pi80_rel_width":        _s5_round(_s6_pi_rel.get("pi80_rel_width", float("nan"))),
            "pi95_rel_width":        _s5_round(_s6_pi_rel.get("pi95_rel_width", float("nan"))),
            "cost_skill_3to1":       _s5_round(_s6_cost.get("cost_skill_3to1", float("nan"))),
            "cost_skill_5to1":       _s5_round(_s6_cost.get("cost_skill_5to1", float("nan"))),
            "cost_skill_10to1":      _s5_round(_s6_cost.get("cost_skill_10to1", float("nan"))),
            "dm_z_stat":             _s5_round(_s6_dm.get("dm_z_stat", float("nan"))),
            "dm_p_value":            _s5_round(_s6_dm.get("dm_p_value", float("nan"))),
            "lead_time_weeks":       _s5_round(_s6_lead),
            # DROP n (3-way 2026-05-26): duplicate of n_test (line 552).
            "n_valid":               _s6_n_valid,
            "npv":                   npv,    # was computed in binary_clinical_rates but not surfaced
            "f1":                    f1c,    # sklearn/FluSight-standard generic F1 (Gemini Q2 2026-05-26)
            # ── Sprint S7 (2026-05-26): ROC family + F-β + confusion matrix + DOR ──
            # ROC family
            "auprc":                 _s5_round(_s7_roc_family.get("auprc", float("nan"))),
            "partial_auc_high_spec": _s5_round(_s7_roc_family.get("partial_auc_high_spec", float("nan"))),
            # F-β family
            "f2_score":              _s5_round(_s7_fbeta.get("f2_score", float("nan"))),
            "f05_score":             _s5_round(_s7_fbeta.get("f05_score", float("nan"))),
            # Confusion matrix raw cells + derived
            "tp":                    int(_s7_cm.get("tp", 0)),
            "tn":                    int(_s7_cm.get("tn", 0)),
            "fp":                    int(_s7_cm.get("fp", 0)),
            "fn":                    int(_s7_cm.get("fn", 0)),
            "accuracy":              _s5_round(_s7_cm.get("accuracy", float("nan"))),
            "balanced_accuracy":     _s5_round(_s7_cm.get("balanced_accuracy", float("nan"))),
            "prevalence":            _s5_round(_s7_cm.get("prevalence", float("nan"))),
            "g_mean":                _s5_round(_s7_cm.get("g_mean", float("nan"))),
            # Diagnostic test classics
            "dor":                   _s5_round(_s7_clin_diag.get("dor", float("nan"))),
            "markedness":            _s5_round(_s7_clin_diag.get("markedness", float("nan"))),
            # DROP informedness (3-way 2026-05-26): identical to youden_j (sens+spec-1).
            "youden_j":              _s5_round(_s7_clin_diag.get("youden_j", float("nan"))),
            "sensitivity": sens,
            "specificity": spec,
            "ppv": ppv,
            # DROP clinical_f1 (Gemini Q2 2026-05-26): patient-level diagnostic naming (Steyerberg
            #   2019 TRIPOD), not for population surveillance. Use generic `f1` instead.
            #   Duplicate npv removed too (already surfaced above at line 634).
            "alert_f1":    round(alert_f1, 4) if np.isfinite(alert_f1) else float("nan"),
            # ── WIS decomposition (Bracher 2021) — 4 keys ──────────────
            "wis_sharpness":    wis_decomp.get("wis_sharpness",    float("nan")),
            "wis_underpred":    wis_decomp.get("wis_underpred",    float("nan")),
            "wis_overpred":     wis_decomp.get("wis_overpred",     float("nan")),
            "wis_total_decomp": wis_decomp.get("wis_total_decomp", float("nan")),
            # ── Epidemic phase (Biggerstaff 2016) — 5 keys ─────────────
            "attack_rate_relerr":    epi_phase.get("attack_rate_relerr",    float("nan")),
            "growth_rate_corr":      epi_phase.get("growth_rate_corr",      float("nan")),
            "epidemic_duration_err": epi_phase.get("epidemic_duration_err", float("nan")),
            "season_onset_err":      epi_phase.get("season_onset_err",      float("nan")),
            # DROP early_warning_lead (3-way 2026-05-26): always = -season_onset_err
            #   (metrics.py:1494-1497 docstring). Redundant.
            # ── Advanced clinical (Chicco 2020, Vickers 2006) — 5 keys ─
            "mcc":                 adv_clin.get("mcc",                 float("nan")),
            "cohens_kappa":        adv_clin.get("cohens_kappa",        float("nan")),
            "lr_positive":         adv_clin.get("lr_positive",         float("nan")),
            "lr_negative":         adv_clin.get("lr_negative",         float("nan")),
            "net_benefit_default": adv_clin.get("net_benefit_default", float("nan")),
            # ── Residual diagnostics — 4 keys ──────────────────────────
            "ljung_box_q":       ljung_q,
            "ljung_box_p":       ljung_p,
            "residual_acf_lag1": resid_acf1,
            "shapiro_wilk_p":    shapiro_p,
            # R6 audit MAJOR #6 — 4 new residual diagnostics (2026-05-26)
            "jarque_bera_p":     jarque_bera_p,    # normality (alternative)
            "durbin_watson":     durbin_watson,    # lag-1 autocorr (complementary to LB)
            "residual_skew":     resid_skew,       # 3rd moment
            "residual_kurtosis": resid_kurtosis,   # 4th moment
            # ── PI extensions — 5 keys ─────────────────────────────────
            "pi99_coverage":      pi99_cov,
            "pi99_width":         pi99_w,
            "pi95_relia":         pi95_relia,
            "pi80_relia":         pi80_relia,
            "pi_sharpness_ratio": pi_shr,
        })
        # 2026-05-28 사용자 명시 R2: R8 evaluator (134 metric) merge into R10 row.
        # R10 자체 row (~118) 와 R8 unified (134) 의 unique key 추가 → ~150 unique.
        try:
            from simulation.pipeline.phase_evaluator import evaluate_predictions_full
            _r = rows[-1]
            _full_r8 = evaluate_predictions_full(
                y_test=y_test, y_pred=pred,
                # G-354 (2026-06-25): 129-metric 도 leak-free residual 사용 — 옛 y_test-pred 은 채점 대상
                #   test 점에 self-calibrate(누수). residuals_per_model[name] = 첫 loop 의 leak-free
                #   (r9_leakfree/wfcv_oof) 또는 빈배열(unavailable→coverage NaN, 정직). primary wis 와 동일 출처.
                residuals=residuals_per_model.get(name, np.asarray([], dtype=np.float64)),
                sigma=sigma,
                y_train_pool=None, threshold=t_threshold,
                phase_id="R10",   # per_model_eval = R10
            )
            for _k, _v in _full_r8.items():
                # R10 의 기존 key 보존 (우선), R8 의 새 key 만 추가
                if _k not in _r and not _k.startswith("_"):
                    _r[_k] = _v
        except Exception as _r8_err:
            log.warning(f"  [R10] R8 evaluator skip ({name}): {_r8_err}")
        losses_per_model[name] = ae  # AE used as loss for SPA

    # ── Relative MAE/WIS skill score vs persistence baseline ────────────
    # Skill = 1 − model/baseline. >0 beats persistence; <0 worse.
    # Add persistence as a synthetic candidate for comparison if not present.
    try:
        if "persistence" not in test_preds:
            persist_pred = np.concatenate([[y_in[-1]], y_test[:-1]])
            persist_mae = float(np.mean(np.abs(persist_pred - y_test)))
        else:
            persist_pred = test_preds["persistence"]
            persist_mae = float(np.mean(np.abs(persist_pred - y_test)))
        for r in rows:
            r["skill_mae_vs_persist"] = round(
                relative_skill_score(r.get("mae", float("nan")), persist_mae,
                                      lower_is_better=True), 4)
    except Exception as _se:
        log.debug(f"  [R10] skill score failed: {_se}")

    # ── Pairwise tournament relative WIS ───────────────────────────────
    log.info(f"  [R10] pairwise tournament relative WIS")
    wis_per_target: dict[str, np.ndarray] = {}
    for name, pred in test_preds.items():
        try:
            # G-354 (2026-06-25): empirical residual-quantile WIS — **report-only** hold-out
            #   토너먼트. (정정: 챔피언은 R9 oof_wis[누수-free]로 select_champion_g318 가 선정하며
            #   이 R10 WIS 로 선정하지 않음 — 옛 주석 "the SAME definition the champion is selected
            #   by" 는 거짓이었다.) PI 반폭은 G-354 leak-free residual(R9 in-sample → WF-CV OOF)에서만
            #   오며, 출처 결손 모델은 WIS=NaN(test-residual self-calibration 금지).
            _res = residuals_per_model.get(name, np.array([], dtype=np.float64))
            if len(_res) < 2:   # G-354: PI source unavailable → 토너먼트 제외(relative_wis NaN)
                continue
            wis_arr = weighted_interval_score_empirical(
                y_test, pred, _res, alphas=list(FLUSIGHT_ALPHAS))
            wis_per_target[name] = np.asarray(wis_arr, dtype=np.float64)
        except Exception as e:
            log.debug(f"  [R10] {name} WIS-per-target failed: {e}")
    rel_wis = pairwise_relative_wis(wis_per_target) if len(wis_per_target) >= 2 else {}

    # G-340 (2026-06-24, 외부 reviewer): Mathis 2024 / Cramer 2022 단일분모 relative WIS =
    #   gmean(WIS_model) / gmean(WIS_FluSight-Baseline) — Hub/FluSight 헤드라인 통화(<1=baseline 능가).
    #   pairwise(Sherratt 2023)는 토너먼트, 이건 명시적 baseline 대비 skill. baseline WIS 는 이미
    #   test_preds 에 있어 post-hoc(재학습 불필요). compute_relative_wis(flusight_baseline.py)는 구현됐으나
    #   orphan 이었음 → 여기 배선.
    from simulation.models.flusight_baseline import compute_relative_wis
    _baseline_wis = wis_per_target.get("FluSight-Baseline")

    # Embed pairwise + baseline-relative WIS into the row table
    for r in rows:
        r["relative_wis_pairwise"] = round(rel_wis.get(r["model"], float("nan")), 4)
        if _baseline_wis is not None and r["model"] in wis_per_target:
            _rw = compute_relative_wis(wis_per_target[r["model"]], _baseline_wis)
            r["relative_wis_vs_baseline"] = round(_rw["relative_wis"], 4)   # <1 = FluSight baseline 능가
        else:
            r["relative_wis_vs_baseline"] = float("nan")                     # baseline 결손/모델 WIS 없음

    # ── Hansen SPA test ────────────────────────────────────────────────
    log.info(f"  [R10] Hansen SPA test (n_bootstrap=2000)")
    spa_results: dict = {}
    if len(losses_per_model) >= 2:
        # G-350 (2026-06-25, 감사 P2): SPA benchmark — FluSight-Baseline(Mathis 2024 persistence,
        #   평가 풀에 실재) 우선. 옛 코드는 literal 'persistence'/'ar1'/'seasonal_naive'(losses_per_model 에
        #   절대 없는 이름)만 찾아 항상 dict-첫모델(SVR-Linear, 36위)로 fallback → "중위모델이 최선" trivial
        #   reject = vacuous. SPA 는 보고용(선정 무영향). else-first fallback 은 안전망 유지.
        bench = next((b for b in ("FluSight-Baseline", "persistence", "ar1", "seasonal_naive")
                      if b in losses_per_model), next(iter(losses_per_model)))
        try:
            spa_results = hansen_spa_test(losses_per_model, benchmark_name=bench,
                                            n_bootstrap=2000, seed=0)
            log.info(f"  [R10] SPA: {spa_results['interpretation']}")
        except Exception as e:
            log.warning(f"  [R10] SPA test failed: {e}")

    # ── Sort + ranking ─────────────────────────────────────────────────
    # G-307 (3자 감사 #1, 2026-06-18): rank_wis = R9 OOF-CV WIS(선정, 누수-free) — hold-out test 'wis'
    #   가 아니다(그건 winner's curse). rank_wis_test = test 진단. _assign_oof_and_test_ranks 가 캡슐화.
    rows_sorted = _assign_oof_and_test_ranks(rows)
    rows_by_log_wis = sorted(rows_sorted, key=lambda r: r["log_wis"]
                             if np.isfinite(r["log_wis"]) else float("inf"))
    for rank, r in enumerate(rows_by_log_wis, 1):
        next(rr for rr in rows_sorted if rr["model"] == r["model"])["rank_log_wis"] = rank

    # ── New rankings: rank_mae, rank_r2 ──────────────────────────────
    rows_by_mae = sorted(rows_sorted, key=lambda r: r.get("mae", float("inf"))
                         if np.isfinite(r.get("mae", float("nan"))) else float("inf"))
    for rank, r in enumerate(rows_by_mae, 1):
        next(rr for rr in rows_sorted if rr["model"] == r["model"])["rank_mae"] = rank
    rows_by_r2 = sorted(rows_sorted,
                        key=lambda r: r.get("r2", float("-inf"))
                        if np.isfinite(r.get("r2", float("nan"))) else float("-inf"),
                        reverse=True)
    for rank, r in enumerate(rows_by_r2, 1):
        next(rr for rr in rows_sorted if rr["model"] == r["model"])["rank_r2"] = rank

    # ── Skill scores: WIS/CRPS vs persistence ────────────────────────
    try:
        if "persistence" not in test_preds:
            _persist_pred = np.concatenate([[y_in[-1]], y_test[:-1]])
        else:
            _persist_pred = test_preds["persistence"]
        _persist_sigma = max(float(np.std(y_test - _persist_pred)), 1e-3)
        # A2 (M7): empirical WIS for the persistence benchmark too, so the
        # skill_wis_vs_persist ratio compares empirical/empirical (the model WIS
        # is empirical). σ kept for the Gaussian CRPS denominator just below.
        _persist_res = (y_test - _persist_pred)
        _persist_res = _persist_res[np.isfinite(_persist_res)]
        _persist_wis_arr = weighted_interval_score_empirical(
            y_test, _persist_pred, _persist_res, alphas=FLUSIGHT_ALPHAS
        )
        _persist_wis  = float(np.mean(_persist_wis_arr))
        _persist_crps = float(np.mean(crps_gaussian(
            y_test, _persist_pred, np.full_like(y_test, _persist_sigma)
        )))
        for r in rows_sorted:
            _m_wis  = r.get("wis",  float("nan"))
            _m_crps = r.get("crps_gaussian", float("nan"))
            r["skill_wis_vs_persist"] = (
                round(relative_skill_score(_m_wis, _persist_wis, lower_is_better=True), 4)
                if np.isfinite(_m_wis) else float("nan")
            )
            r["skill_crps_vs_persist"] = (
                round(relative_skill_score(_m_crps, _persist_crps, lower_is_better=True), 4)
                if np.isfinite(_m_crps) else float("nan")
            )
    except Exception as _se:
        log.debug(f"  [R10] skill WIS/CRPS vs persist failed: {_se}")
        for r in rows_sorted:
            r.setdefault("skill_wis_vs_persist",  float("nan"))
            r.setdefault("skill_crps_vs_persist", float("nan"))

    # ── DM test vs climatology (unconditional training mean) ─────────
    # dm_z_vs_climatology: baseline = global mean of y_in[:test_start]
    # dm_z_vs_lag52:       baseline = lag-52 seasonal naive (distinct)
    try:
        # Climatology baseline: unconditional historical mean — always available
        _clim_const = np.full(
            len(y_test),
            float(np.mean(y_in[:test_start])) if test_start > 0 else float(np.mean(y_test)),
            dtype=np.float64,
        )
        # Lag-52 seasonal naive — only when sufficient history exists
        _has_lag52 = len(y_in) > test_start + 52
        if _has_lag52:
            _lag52_pred = np.asarray(
                y_in[test_start - 52: test_end - 52], dtype=np.float64
            )[:len(y_test)]
        else:
            _lag52_pred = None

        for r in rows_sorted:
            _pred2 = test_preds.get(r["model"])
            if _pred2 is None:
                r.update(dm_z_vs_climatology=float("nan"),
                         dm_p_vs_climatology=float("nan"),
                         dm_z_vs_lag52=float("nan"),
                         dm_p_vs_lag52=float("nan"))
                continue
            # DM vs unconditional training mean (climatology)
            try:
                _zc, _pc = diebold_mariano(y_test, _pred2, _clim_const, h=1)
                r["dm_z_vs_climatology"] = round(float(_zc), 4)
                r["dm_p_vs_climatology"] = round(float(_pc), 4)
            except Exception:
                r["dm_z_vs_climatology"] = float("nan")
                r["dm_p_vs_climatology"] = float("nan")
            # DM vs lag-52 seasonal naive (distinct baseline)
            if _lag52_pred is not None:
                try:
                    _zs, _ps = diebold_mariano(y_test, _pred2, _lag52_pred, h=1)
                    r["dm_z_vs_lag52"] = round(float(_zs), 4)
                    r["dm_p_vs_lag52"] = round(float(_ps), 4)
                except Exception:
                    r["dm_z_vs_lag52"] = float("nan")
                    r["dm_p_vs_lag52"] = float("nan")
            else:
                r["dm_z_vs_lag52"] = float("nan")
                r["dm_p_vs_lag52"] = float("nan")
    except Exception as _se:
        log.debug(f"  [R10] DM vs climatology failed: {_se}")
        for r in rows_sorted:
            r.setdefault("dm_z_vs_climatology", float("nan"))
            r.setdefault("dm_p_vs_climatology", float("nan"))
            r.setdefault("dm_z_vs_lag52",        float("nan"))
            r.setdefault("dm_p_vs_lag52",        float("nan"))

    # ── S9 (2026-05-26): BH-FDR multiple testing correction for DM tests ──
    # Per Codex/Gemini publication-readiness P0: control FDR at 5% across the
    # 3 per-model DM tests (vs persist [lag-1], vs climatology, vs lag-52).
    # Adjusted p-values added as dm_p_value_bh, dm_p_vs_climatology_bh,
    # dm_p_vs_lag52_bh. Reference: Benjamini & Hochberg (1995) JRSSB 57:289.
    for r in rows_sorted:
        try:
            _raw_dm_ps = [
                r.get("dm_p_value", float("nan")),
                r.get("dm_p_vs_climatology", float("nan")),
                r.get("dm_p_vs_lag52", float("nan")),
            ]
            _finite_idx = [i for i, p in enumerate(_raw_dm_ps) if np.isfinite(p)]
            if len(_finite_idx) >= 2:
                _finite_ps = [_raw_dm_ps[i] for i in _finite_idx]
                _bh = adjust_pvalues(_finite_ps, method="fdr_bh")
                _adj = list(_bh.get("p_adj", _finite_ps))
                # Map back to original 3-slot order
                _out = [float("nan")] * 3
                for j, i in enumerate(_finite_idx):
                    _out[i] = round(float(_adj[j]), 4)
                r["dm_p_value_bh"]          = _out[0]
                r["dm_p_vs_climatology_bh"] = _out[1]
                r["dm_p_vs_lag52_bh"]       = _out[2]
            else:
                r["dm_p_value_bh"] = r.get("dm_p_value", float("nan"))
                r["dm_p_vs_climatology_bh"] = r.get("dm_p_vs_climatology", float("nan"))
                r["dm_p_vs_lag52_bh"] = r.get("dm_p_vs_lag52", float("nan"))
        except Exception:
            r["dm_p_value_bh"] = float("nan")
            r["dm_p_vs_climatology_bh"] = float("nan")
            r["dm_p_vs_lag52_bh"] = float("nan")

    # ── S9: Bootstrap CI (B=1000, block_len=int(sqrt(n))) for key metrics ──
    # Per publication-readiness P0: report 95% CI for primary probabilistic
    # metrics so reviewers see borderline-significance at small n. Block bootstrap
    # (Künsch 1989) preserves serial dependence in y_test. Seed=42 for reproducibility.
    _B_RES = 1000
    _SEED_BOOT = 42
    _block_len = int(np.sqrt(max(len(y_test), 2)))
    for r in rows_sorted:
        try:
            _pred_b = test_preds.get(r["model"])
            if _pred_b is None or len(_pred_b) != len(y_test):
                raise ValueError("pred not available")
            _ae = np.abs(_pred_b - y_test)
            # MAE CI (already computed earlier as mae_ci95_lo/hi — re-confirm seed)
            _wis_ci = bootstrap_ci(_ae, statistic=np.mean, n_boot=_B_RES,
                                   alpha=0.05, method="bca",
                                   random_state=_SEED_BOOT, block_len=_block_len)
            # Note: MAE CI already populated; bootstrap WIS/CRPS pair-resample
            # requires custom logic — store fast block-bootstrap MAE CI as fallback
            # for any model that didn't get earlier per-call CI.
            r.setdefault("mae_ci95_lo_bs", round(float(_wis_ci.get("ci_lo", float("nan"))), 4))
            r.setdefault("mae_ci95_hi_bs", round(float(_wis_ci.get("ci_hi", float("nan"))), 4))
            # WIS CI (block bootstrap on indices with empirical-q WIS recomputation)
            _res_b = residuals_per_model.get(r["model"], np.array([], dtype=np.float64))
            if len(_res_b) >= 10:
                _wis_boots = np.empty(_B_RES)
                _rng_b = np.random.default_rng(_SEED_BOOT)
                _L = _block_len
                _n_blocks = (len(y_test) + _L - 1) // _L
                _max_start = max(1, len(y_test) - _L + 1)
                for _b in range(_B_RES):
                    _starts = _rng_b.integers(0, _max_start, size=_n_blocks)
                    _idx = np.concatenate([np.arange(s, s + _L) for s in _starts])[:len(y_test)]
                    _wis_boots[_b] = weighted_interval_score_empirical(
                        y_test[_idx], _pred_b[_idx], _res_b,
                        alphas=list(FLUSIGHT_ALPHAS),
                    ).mean()
                _w_lo, _w_hi = float(np.quantile(_wis_boots, 0.025)), float(np.quantile(_wis_boots, 0.975))
                r["wis_ci95_lo"] = round(_w_lo, 4)
                r["wis_ci95_hi"] = round(_w_hi, 4)
            else:
                r["wis_ci95_lo"] = float("nan")
                r["wis_ci95_hi"] = float("nan")
        except Exception as _se:
            log.debug(f"  [R10] Bootstrap CI for {r.get('model')} failed: {_se}")
            r.setdefault("mae_ci95_lo_bs", float("nan"))
            r.setdefault("mae_ci95_hi_bs", float("nan"))
            r.setdefault("wis_ci95_lo", float("nan"))
            r.setdefault("wis_ci95_hi", float("nan"))

    # ── Skill score vs seasonal naive (lag-52) ───────────────────────
    try:
        if len(y_in) > test_start + 52:
            _snaive = y_in[test_start - 52: test_end - 52]
        else:
            _snaive = np.full(len(y_test), float(np.mean(y_in[:test_start])))
        _snaive = np.asarray(_snaive, dtype=np.float64)[:len(y_test)]
        _snaive_mae = float(np.mean(np.abs(_snaive - y_test)))
        for r in rows_sorted:
            _m_mae = r.get("mae", float("nan"))
            r["skill_mae_vs_snaive"] = (
                round(relative_skill_score(_m_mae, _snaive_mae, lower_is_better=True), 4)
                if (np.isfinite(_m_mae) and _snaive_mae > 1e-9) else float("nan")
            )
    except Exception as _se:
        log.debug(f"  [R10] skill vs snaive failed: {_se}")
        for r in rows_sorted:
            r.setdefault("skill_mae_vs_snaive", float("nan"))

    # ── Per-metric bootstrap CI (진단용; 4-criteria/g175 gate 완전 제거 2026-06-05) ──
    # champion = 순수 best-WIS. R²/MAPE/WIS/PICP95 의 bootstrap CI 를 개별 진단으로 산출
    # (comprehensive 가 pi95_ci 소비; r2/mape/wis_ci 일반 보고). gate/composite/tier 없음.
    #   - WIS: FluSight relative WIS<1 vs baseline (Mathis et al. 2024 Nat Commun 15:6289).
    #   - PICP95: FluSight nominal 0.95 (Bracher et al. 2021 PLoS Comput Biol 17(2):e1008618).
    from simulation.analytics.bootstrap_ci import bootstrap_metric_ci as _boot_ci
    from simulation.config_global import GLOBAL as _GLOBAL_CFG
    _N_BOOT = _GLOBAL_CFG.filter.ci_bootstrap_n
    _SEED   = _GLOBAL_CFG.filter.ci_bootstrap_seed

    # bootstrap CI metric fn factories (audit Stage 1.2, Task #14.b)
    def _r2_fn(yt: np.ndarray, yp: np.ndarray) -> float:
        err = yp - yt
        sse = float(np.sum(err * err))
        sst = float(np.sum((yt - yt.mean()) ** 2))
        return float(1.0 - sse / sst) if sst > 0 else float("nan")

    def _mape_fn(yt: np.ndarray, yp: np.ndarray) -> float:
        nz = np.abs(yt) > 1e-3
        if not nz.any():
            return float("nan")
        return float(np.mean(np.abs((yp[nz] - yt[nz]) / yt[nz])) * 100.0)

    def _make_wis_fn(residuals: np.ndarray):
        # A2 (M7): empirical residual-quantile WIS — same definition as the
        # champion / tournament WIS, so the bootstrap WIS-CI is on the same scale
        # (was Gaussian-σ closed form). PI half-widths fixed at the point
        # estimate's residuals across bootstraps (mirrors the prior fixed-σ CI).
        from simulation.analytics.diagnostics import weighted_interval_score_empirical
        from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
        _res = np.asarray(residuals, dtype=np.float64)
        def _fn(yt: np.ndarray, yp: np.ndarray) -> float:
            try:
                arr = weighted_interval_score_empirical(yt, yp, _res, alphas=list(FLUSIGHT_ALPHAS))
                return float(np.mean(arr))
            except Exception:
                return float("nan")
        return _fn

    def _make_pi95_fn(sigma: float):
        _s = max(float(sigma), 1e-6)
        def _fn(yt: np.ndarray, yp: np.ndarray) -> float:
            lo = yp - Z95 * _s
            hi = yp + Z95 * _s
            return float(np.mean((yt >= lo) & (yt <= hi)))
        return _fn

    # bootstrap CI per model (comprehensive 가 pi95_ci 소비; r2/mape/wis_ci 일반 진단).
    # champion = 순수 best-WIS (_designate_best_wis_champion). per-metric CI 만 유지.
    for r in rows_sorted:
        try:
            _pred_b = test_preds.get(r["model"])
            if _pred_b is None or len(_pred_b) != len(y_test):
                r["r2_ci_lo"] = r["r2_ci_hi"] = float("nan")
                r["mape_ci_lo"] = r["mape_ci_hi"] = float("nan")
                r["wis_ci_lo"] = r["wis_ci_hi"] = float("nan")
                r["pi95_ci_lo"] = r["pi95_ci_hi"] = float("nan")
            else:
                _res_b = residuals_per_model.get(r["model"], np.array([], dtype=np.float64))
                _sigma_b = float(np.std(_res_b, ddof=1)) if len(_res_b) > 1 else 1.0
                _wis_fn = _make_wis_fn(_res_b)        # A2: empirical (σ kept for pi95)
                _pi95_fn = _make_pi95_fn(_sigma_b)

                r2_ci   = _boot_ci(_r2_fn,   y_test, _pred_b, n_boot=_N_BOOT, seed=_SEED)
                mape_ci = _boot_ci(_mape_fn, y_test, _pred_b, n_boot=_N_BOOT, seed=_SEED)
                wis_ci  = _boot_ci(_wis_fn,  y_test, _pred_b, n_boot=_N_BOOT, seed=_SEED)
                pi95_ci = _boot_ci(_pi95_fn, y_test, _pred_b, n_boot=_N_BOOT, seed=_SEED)

                r["r2_ci_lo"], r["r2_ci_hi"]     = r2_ci["ci_lo"],   r2_ci["ci_hi"]
                r["mape_ci_lo"], r["mape_ci_hi"] = mape_ci["ci_lo"], mape_ci["ci_hi"]
                r["wis_ci_lo"], r["wis_ci_hi"]   = wis_ci["ci_lo"],  wis_ci["ci_hi"]
                r["pi95_ci_lo"], r["pi95_ci_hi"] = pi95_ci["ci_lo"], pi95_ci["ci_hi"]
        except Exception as _ci_e:
            log.debug(f"  [R10] {r.get('model','?')} CI 계산 실패: {_ci_e}")
            r["r2_ci_lo"] = r["r2_ci_hi"] = float("nan")
            r["mape_ci_lo"] = r["mape_ci_hi"] = float("nan")
            r["wis_ci_lo"] = r["wis_ci_hi"] = float("nan")
            r["pi95_ci_lo"] = r["pi95_ci_hi"] = float("nan")

    # ── champion = G-318 공정 챔피언 (2026-06-19, 사용자 결정): OOF-shortlist → hold-out 일반화.
    # 이전 G-307(순수 OOF-argmin)은 OOF-과적합 챔피언 박제 → 폐기. R²/MAPE/PICP 는 개별 metric 병기. ──
    _champ = _designate_best_wis_champion(rows_sorted)        # G-318 primary (배포) + 플래그 설정
    _champ_ho = select_champion_holdout_best(rows_sorted)     # hold-out best (병기 보고)
    if _champ is not None:
        log.info(
            f"  [R10] 🏆 Champion (G-318, 배포) = {_champ['model']} "
            f"(WIS={_champ.get('wis'):.4f} | R²={_champ.get('r2', float('nan')):.3f} "
            f"MAPE={_champ.get('mape', float('nan')):.2f} PICP95={_champ.get('pi95_coverage', float('nan')):.3f})"
        )
    if _champ_ho is not None:
        _agree = bool(_champ is not None and _champ_ho["model"] == _champ["model"])
        log.info(
            f"  [R10] 🥈 Champion (hold-out best, 병기) = {_champ_ho['model']} "
            f"(WIS={_champ_ho.get('wis'):.4f}) — "
            f"{'G-318과 일치 → 강한 증거(과적합 아님)' if _agree else 'G-318과 다름 → 둘 다 보고'}"
        )

    # 출력 (per_model_metrics.csv · metric_history): champion = best-WIS, g175/4-criteria
    # 컬럼 없음 (phase_evaluator 129-metric + 본 함수 모두 g175-free 2026-06-05).
    # R²/MAPE/WIS/PICP 는 개별 metric 으로 유지; bootstrap *_ci_lo/hi 는 comprehensive 소비.

    # ── Persist artifacts ──────────────────────────────────────────────
    out_dir = Path(getattr(config, "save_dir", "simulation/results")) / "per_model_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    import csv
    if rows_sorted:
        cols = list(rows_sorted[0].keys())
        with (out_dir / "per_model_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in rows_sorted:
                w.writerow(r)

    # JSON ranking summary. NOTE: rows_sorted is ordered by OOF-CV WIS (rank_wis,
    # the leak-free selection criterion), NOT by hold-out test WIS — so this list
    # is the OOF order. Keyed accordingly to avoid the "by WIS" mislabel.
    summary = {
        "n_models_evaluated": len(rows_sorted),
        "n_test_weeks": int(n_real_test),
        "test_window_idx": (int(test_start), int(test_end)),
        "top10_by_oof_wis": [r["model"] for r in rows_sorted[:10]],
        "spa_test": spa_results,
        "pairwise_relative_wis": rel_wis,
    }
    (out_dir / "ranking.json").write_text(json.dumps(summary, indent=2, default=str))

    # Markdown
    md = [
        "# R10 — Per-Model Evaluation on Test Slab",
        "",
        f"- **Test slab n** = {n_real_test} weeks",
        f"- **Models evaluated** = {len(rows_sorted)}",
        f"- **Headline metric** = WIS (Bracher 2021); also reported: log-WIS (Bosse 2023, FluSight 2024-25)",
        "",
        "## Top 10 by OOF-CV WIS (selection order)",
        "",
        "> `rank` = `rank_wis` = **R9 OOF-CV WIS** rank (leak-free selection criterion, G-307). "
        "Rows are ordered by OOF, NOT by the hold-out `WIS` column — so the WIS column is not "
        "monotonic here (that is expected: OOF-best ≠ test-best is the leakage separation working). "
        "For a hold-out-WIS-sorted view see `rank_wis_test`.",
        "",
        "| rank (OOF) | model | WIS (test) | log-WIS | MAE | MAE 95% CI | R² | 95% PI cov | rel-WIS-pair |",
        "|------|-------|-----|---------|-----|------------|-----|-----------|--------------|",
    ]
    for r in rows_sorted[:10]:
        md.append(
            f"| {r.get('rank_wis', '?')} | {r['model']} | "
            f"{r['wis']:.3f} | {r['log_wis']:.3f} | {r['mae']:.3f} | "
            f"({r['mae_ci95_lo']:.3f}, {r['mae_ci95_hi']:.3f}) | "
            f"{r['r2']:.3f} | "
            f"{r['pi95_coverage']:.3f} ({r['pi95_ci_lo']:.2f}, {r['pi95_ci_hi']:.2f}) | "
            f"{r['relative_wis_pairwise']:.3f} |"
        )
    if spa_results:
        md += [
            "",
            "## Hansen SPA Test (multiple-comparisons-corrected ranking)",
            f"- Benchmark: `{spa_results['benchmark']}`",
            f"- Test statistic: {spa_results.get('test_statistic', float('nan')):.3f}",
            f"- p-value: {spa_results['spa_p_value']:.4f}",
            f"- {spa_results['interpretation']}",
        ]
    md += [
        "",
        "## Notes",
        "- All metrics computed on test slab (in-sample idx [pool_end:n_in])",
        "  — n=68 by HWP §3, NOT the n=8 real slab.",
        "- 95% PI coverage uses Wilson exact CI (vs normal approximation).",
        "- WIS is the headline; pairwise relative WIS (Sherratt 2023) tournament",
        "  shows model-vs-rest performance.",
        "- Hansen SPA test corrects multiple-comparisons fishing across 50+ models.",
    ]
    (out_dir / "report.md").write_text("\n".join(md))
    log.info(f"  [R10] wrote per-model report to {out_dir}")

    # Sprint 3 EDA sidecar (2026-05-26) — non-fatal, atomic write
    from .eda_writer import write_phase_eda
    write_phase_eda(
        phase_id=11, phase_tag="per_model_eval",
        y_true=y_test, predictions=test_preds,
        save_dir=Path(getattr(config, "save_dir", "simulation/results")) / "eda",
        extra_meta={"n_models_evaluated": len(rows_sorted),
                     "top10_by_oof_wis": [r["model"] for r in rows_sorted[:10]]},
    )

    # G-318: ranking_top10[0] = 공정 챔피언(OOF-shortlist→hold-out 일반화). real_eval.py:797·web·.pt 가
    #   ranking_top10[0] 을 챔피언으로 소비하므로 OOF-best([0]=G-307)를 박으면 G-318 이 우회됨 → 챔피언을
    #   맨 앞에, 나머지는 OOF(rank_wis) 순. + 명시 champion 필드(소비자 robust resolve).
    _champ_name = _champ["model"] if _champ is not None else None
    _champ_ho_name = _champ_ho["model"] if _champ_ho is not None else None
    _top10 = ([_champ_name] + [r["model"] for r in rows_sorted if r["model"] != _champ_name])[:10] \
        if _champ_name is not None else [r["model"] for r in rows_sorted[:10]]
    return {
        "n_models": len(rows_sorted),
        "n_test_weeks": int(n_real_test),
        "ranking_top10": _top10,
        "champion": _champ_name,                    # G-318 (배포 = ranking_top10[0])
        "champion_holdout_best": _champ_ho_name,     # 순수 hold-out best (병기 보고)
        "champions_agree": bool(_champ_name is not None and _champ_name == _champ_ho_name),
        "spa_test": spa_results,
        "pairwise_relative_wis": rel_wis,
        "report_path": str(out_dir / "report.md"),
        "metrics_csv": str(out_dir / "per_model_metrics.csv"),
        "ranking_json": str(out_dir / "ranking.json"),
        "elapsed": time.time() - t0,
    }


# back-compat aliases (2026-06-02 semantic rename — 옛 run_phaseN)
run_phase14 = run_per_model_eval
