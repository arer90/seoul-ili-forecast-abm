"""
R6: Diebold-Mariano 쌍대 검정
====================================
모든 모델 쌍을 비교하여 통계적 유의성 판정.
"""
import logging
from simulation.config_global import GLOBAL  # SSOT (2026-05-28)
import time
import numpy as np
from itertools import combinations
from typing import Dict

log = logging.getLogger(__name__)


def _dm_test(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> dict:
    """Diebold-Mariano 검정 (HLN 소표본 보정). e1,e2=예측오차; 손실=제곱오차.

    G-341 (2026-06-24, 외부 reviewer): p-value 를 **Harvey-Leybourne-Newbold (1997) 소표본 보정**
    DM 으로 통일 — 표준 asymptotic Normal DM 은 작은 n 에서 과대확신(over-sized). HLN 은 보정계수
    √[(n+1-2h+h(h-1)/n)/n] + t_{n-1} 참조로 보수화. p-value SSOT = analytics.hln_dm_pvalue
    (D-2: 모든 DM = HLN-DM). statistic 은 보고용 별도 산출. R6 메인 DM 은 n=68 test 가 아니라
    WF-CV OOF(~343 주)에 작동 → HLN≈asymptotic 이나, regime subset(n≥20)·rigor 위해 보정.
    """
    from simulation.analytics.ablation_stats import hln_dm_pvalue
    la = np.asarray(e1, dtype=np.float64) ** 2
    lb = np.asarray(e2, dtype=np.float64) ** 2
    n = int(min(la.size, lb.size))
    if n < 10:
        return {"statistic": 0, "p_value": 1.0}
    p_value = hln_dm_pvalue(la, lb, h=h)          # HLN-corrected 양측 p-value (SSOT)
    d = la - lb
    d_mean = float(np.mean(d))
    var_d = float(np.var(d, ddof=1))
    dm_stat = d_mean / np.sqrt(var_d / n) if var_d > 0 else 0.0
    return {"statistic": float(dm_stat), "p_value": float(p_value)}


# G-341 (2026-06-24, 외부 reviewer): DM 은 **nested 모델쌍에 무효** — H0 하 손실차 분산→0 으로
#   점근 정규성이 붕괴한다(Clark & West 2007 의 MSPE-adjusted 가 정답). 아래 = 알려진 nested family
#   (보수적): ARIMA⊂SARIMA⊂SARIMAX, Poisson⊂NegBin(+NB 변형). 이 쌍은 result["nested"]=True 로
#   표시 + 글로벌 win-count 에서 제외 → '모든 쌍 DM 유의' 식 잘못된 유의성 주장 차단. 완전한
#   Clark-West 통계 배선은 follow-up(P1); 우선 mis-specified DM 보고를 막는다.
_NESTED_GROUPS = (
    frozenset({"ARIMA", "SARIMA", "SARIMAX"}),
    frozenset({"PoissonAutoreg", "NegBinGLM", "NegBinGLM-V7", "NegBinGLM-Glum"}),
)


def _is_nested_pair(m1: str, m2: str) -> bool:
    """m1, m2 가 알려진 nested family(서로 포함관계)면 True → DM 무효(Clark-West 필요)."""
    return any(m1 in g and m2 in g for g in _NESTED_GROUPS)


def _build_regime_masks(y_all, config):
    """S1-4 fix: split the series into pre-COVID / during-COVID / post-COVID
    regimes so DM tests are computed within each regime (error-loss
    stationarity is violated across the 2020 NPI structural break).

    Regime boundaries use config.data.dates if present; otherwise fall back
    to proportional index splits calibrated to the project's 2017-2024
    weekly window (~343 weeks): pre-COVID ≈ first 47%, during ≈ next 36%,
    post ≈ last 17%.
    """
    n = len(y_all)
    dates = None
    for attr_chain in (("data", "dates"), ("dates",)):
        obj = config
        for a in attr_chain:
            obj = getattr(obj, a, None)
            if obj is None:
                break
        if obj is not None:
            dates = obj
            break
    regimes = {}
    if dates is not None and len(dates) == n:
        import numpy as _np
        d = _np.asarray(dates, dtype="datetime64[D]")
        pre = d < _np.datetime64("2020-03-01")
        post = d >= _np.datetime64("2023-01-01")
        during = (~pre) & (~post)
        regimes["pre_covid"] = pre
        regimes["during_covid"] = during
        regimes["post_covid"] = post
    else:
        # proportional fallback
        i1 = int(round(n * 0.47))
        i2 = int(round(n * 0.83))
        mask_pre = np.zeros(n, dtype=bool);    mask_pre[:i1] = True
        mask_during = np.zeros(n, dtype=bool); mask_during[i1:i2] = True
        mask_post = np.zeros(n, dtype=bool);   mask_post[i2:] = True
        regimes["pre_covid"] = mask_pre
        regimes["during_covid"] = mask_during
        regimes["post_covid"] = mask_post
        log.warning("  R6: config has no dates -- regime split uses "
                    "proportional index fallback (47/36/17). Add "
                    "`config.data.dates` for calendar-accurate splits.")
    regimes["global"] = np.ones(n, dtype=bool)
    return regimes


from simulation.utils.resource_tracker import track_resources


@track_resources("phase9_dm_test")
def run_dm_test(y_all, oof_predictions: Dict[str, np.ndarray], config) -> dict:
    """R6: DM 검정 실행 (regime-split: pre/during/post-COVID + global)."""
    from .utils.logging_util import phase_banner, fmt_time
    phase_banner("R6", "Diebold-Mariano 쌍대 검정 (regime-split)")

    t0 = time.time()
    model_names = sorted(oof_predictions.keys())
    regimes = _build_regime_masks(y_all, config)

    dm_results: Dict[str, Dict[str, dict]] = {r: {} for r in regimes}
    win_count = {m: 0 for m in model_names}  # global-only win counts
    total_tests = 0

    for m1, m2 in combinations(model_names, 2):
        p1 = oof_predictions[m1]
        p2 = oof_predictions[m2]
        valid_base = ~np.isnan(p1) & ~np.isnan(p2)
        if valid_base.sum() < 20:
            continue
        key = f"{m1}_vs_{m2}"

        for regime_name, regime_mask in regimes.items():
            valid = valid_base & regime_mask
            if valid.sum() < 20:
                continue
            e1 = y_all[valid] - p1[valid]
            e2 = y_all[valid] - p2[valid]
            result = _dm_test(e1, e2)
            result["nested"] = _is_nested_pair(m1, m2)   # G-341: nested → DM 무효(Clark-West 필요)
            dm_results[regime_name][key] = result
            if regime_name == "global":
                total_tests += 1
                # nested 쌍은 DM 무효 → 유의 win-count 제외(잘못된 유의성 주장 차단)
                if result["p_value"] < 0.05 and not result["nested"]:
                    if result["statistic"] < 0:
                        win_count[m1] += 1
                    else:
                        win_count[m2] += 1

    # 승률 계산
    n_models = len(model_names)
    max_wins = n_models - 1 if n_models > 1 else 1
    dm_win_rates = {m: round(win_count[m] / max_wins, 4) for m in model_names}

    for regime_name in ("pre_covid", "during_covid", "post_covid", "global"):
        n_r = len(dm_results.get(regime_name, {}))
        log.info(f"  [regime={regime_name:13s}] pairs tested = {n_r}")
    log.info(f"  총 {total_tests}개 글로벌 쌍 비교 완료")
    log.info("  DM 승률 (global, 상위 5):")
    for m, rate in sorted(dm_win_rates.items(), key=lambda x: -x[1])[:5]:
        log.info(f"    {m:18s} 승률={rate:.2%} ({win_count[m]}승)")

    # R8.3 (2026-05-26): per-model full 134-key SSOT eval at R6 perspective.
    # Trajectory: R4 OOF → R6 post-DM-test 시점 134-key snapshot.
    # Toggle via env MPH_FULL_EVAL_TRAJECTORY (default '1').
    per_model_eval_r8: dict = {}
    from simulation.config_global import GLOBAL as _GCFG  # SSOT (2026-05-28)
    if _GCFG.filter.full_eval_trajectory:
        try:
            from simulation.pipeline.phase_evaluator import evaluate_predictions_full
            for _mname in model_names:
                _oof = oof_predictions.get(_mname)
                if _oof is None:
                    continue
                _valid = ~np.isnan(_oof)
                if _valid.sum() < 5:
                    continue
                try:
                    full_r8 = evaluate_predictions_full(
                        y_test=y_all[_valid],
                        y_pred=_oof[_valid],
                        residuals=y_all[_valid] - _oof[_valid],
                        y_train_pool=None,
                        threshold=GLOBAL.filter.alert_threshold,
                        phase_id=f"R6_dm_{_mname}",
                        enable_bootstrap_ci=False,
                    )
                    per_model_eval_r8[_mname] = full_r8
                except Exception as _e:
                    per_model_eval_r8[_mname] = {"phase_eval_r8_err": str(_e)}
        except Exception as _e:
            log.debug(f"  [phase9] phase_eval_r8 wiring skipped: {_e}")

    elapsed = time.time() - t0
    # ── audit Stage 1.3 (#15.b, 2026-05-27) — BH-FDR family correction ──
    # 52 choose 2 = 1326 pairs × 4 regime = 5304 DM tests.
    # Benjamini & Hochberg (1995) FDR(q=0.05) per regime family.
    # Reference: BH (1995) JRSSB 57:289-300, doi:10.1111/j.2517-6161.1995.tb02031.x
    dm_results_bh_fdr: dict = {}
    win_count_bh = {m: 0 for m in model_names}
    try:
        from simulation.analytics.multiple_testing import apply_bh_fdr
        for regime_name, regime_dict in dm_results.items():
            p_values = {k: v["p_value"] for k, v in regime_dict.items()}
            bh = apply_bh_fdr(p_values, q=0.05)
            dm_results_bh_fdr[regime_name] = {
                "n_tests": bh["n_tests"],
                "n_rejected": bh["n_rejected"],
                "reject": bh["reject"],
                "pvals_corrected": bh["pvals_corrected"],
                "method": "fdr_bh",
                "q": 0.05,
            }
            # global regime → BH-corrected win counts
            if regime_name == "global":
                for pair_key, rejected in bh["reject"].items():
                    if rejected:
                        m1, m2 = pair_key.split("_vs_")
                        stat = regime_dict[pair_key]["statistic"]
                        if stat < 0:
                            win_count_bh[m1] += 1
                        else:
                            win_count_bh[m2] += 1
        log.info(
            f"  [phase9] BH-FDR(q=0.05) per regime: "
            f"global={dm_results_bh_fdr['global']['n_rejected']}/"
            f"{dm_results_bh_fdr['global']['n_tests']}, "
            f"pre={dm_results_bh_fdr.get('pre_covid', {}).get('n_rejected', 0)}/{dm_results_bh_fdr.get('pre_covid', {}).get('n_tests', 0)}, "
            f"during={dm_results_bh_fdr.get('during_covid', {}).get('n_rejected', 0)}/{dm_results_bh_fdr.get('during_covid', {}).get('n_tests', 0)}, "
            f"post={dm_results_bh_fdr.get('post_covid', {}).get('n_rejected', 0)}/{dm_results_bh_fdr.get('post_covid', {}).get('n_tests', 0)}"
        )
    except Exception as _bh_e:
        log.warning(f"  [phase9] BH-FDR fail (non-fatal): {_bh_e}")
        dm_results_bh_fdr = {}

    dm_win_rates_bh = (
        {m: round(win_count_bh[m] / max_wins, 4) for m in model_names}
        if dm_results_bh_fdr else {}
    )

    log.info(f"  ✓ R6 완료 [{fmt_time(elapsed)}]")

    return {
        "dm_tests": dm_results.get("global", {}),   # back-compat flat view
        "dm_tests_by_regime": dm_results,            # full regime-split result
        "win_counts": win_count,                     # uncorrected (back-compat)
        "win_rates": dm_win_rates,                   # uncorrected (back-compat)
        # audit Stage 1.3 (#15.b) — BH-FDR corrected
        "dm_tests_bh_fdr": dm_results_bh_fdr,
        "win_counts_bh_fdr": win_count_bh,
        "win_rates_bh_fdr": dm_win_rates_bh,
        "per_model_eval_r8": per_model_eval_r8,   # R8.3 — per-model 134-key snapshot
        "elapsed": elapsed,
    }


# back-compat aliases (2026-06-02 semantic rename — 옛 run_phaseN)
run_phase9 = run_dm_test
