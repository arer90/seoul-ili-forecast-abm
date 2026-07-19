"""
R8: 종합 스코어 + 유효성 + 최종 랭킹
==========================================
R²(40%) + RMSE순위(20%) + DM승률(15%) + WF-CV 안정성(15%) + Conformal(10%)

Step-2: ``compute_pi_ab_summary`` compares split-conformal vs CV+
(Barber+2021) per model and emits an aggregate recommendation. Used to
decide which PI to present in the paper.
"""
import logging
from simulation.config_global import GLOBAL  # SSOT (2026-05-28)
import time
import numpy as np
from typing import Dict, Any, List

log = logging.getLogger(__name__)


def _rank_normalize(values: Dict[str, float], higher_is_better: bool = True) -> Dict[str, float]:
    """값을 0~1 정규화. higher_is_better=False면 낮을수록 높은 점수."""
    if not values:
        return {}
    sorted_items = sorted(values.items(), key=lambda x: x[1], reverse=higher_is_better)
    n = len(sorted_items)
    return {name: (n - rank) / max(n - 1, 1) for rank, (name, _) in enumerate(sorted_items)}


def compute_composite_scores(
    wf_results: dict,
    dm_win_rates: dict,
    pi_results: dict,
    config,
) -> Dict[str, dict]:
    """종합 점수 계산."""
    w = config.scoring
    model_names = list(wf_results.keys())
    scores = {}

    # 메트릭 수집
    r2_map = {}
    rmse_map = {}
    stability_map = {}

    for name in model_names:
        m = wf_results[name].get("overall_metrics", {})
        r2_map[name] = m.get("r2", 0)
        rmse_map[name] = m.get("rmse", 999)
        ts = m.get("temporal_stability", {})
        if ts:
            stability_map[name] = 1.0 - abs(ts.get("early_r2", 0) - ts.get("late_r2", 0))
        else:
            stability_map[name] = 0.5
    # Conformal coverage 점수 (|coverage - 0.95|가 작을수록 좋음)
    conformal_map = {}
    for name in model_names:
        pi = pi_results.get(name, {})
        conf = pi.get("conformal", {})
        cov = conf.get("coverage", 0.5)
        conformal_map[name] = 1.0 - abs(cov - 0.95) * 5  # 0.95 ±0.05 → 0.75~1.0

    # 정규화
    r2_norm = _rank_normalize(r2_map, higher_is_better=True)
    rmse_norm = _rank_normalize(rmse_map, higher_is_better=False)
    dm_norm = _rank_normalize(dm_win_rates, higher_is_better=True)
    stab_norm = _rank_normalize(stability_map, higher_is_better=True)
    conf_norm = _rank_normalize(conformal_map, higher_is_better=True)

    for name in model_names:
        composite = (
            w.w_r2 * r2_norm.get(name, 0)
            + w.w_rmse_rank * rmse_norm.get(name, 0)
            + w.w_dm_win_rate * dm_norm.get(name, 0)
            + w.w_stability * stab_norm.get(name, 0)
            + w.w_conformal * conf_norm.get(name, 0)
        )
        scores[name] = {
            "composite": round(composite, 4),
            "r2_score": round(r2_norm.get(name, 0), 4),
            "rmse_score": round(rmse_norm.get(name, 0), 4),
            "dm_score": round(dm_norm.get(name, 0), 4),
            "stability_score": round(stab_norm.get(name, 0), 4),
            "conformal_score": round(conf_norm.get(name, 0), 4),
            "raw_r2": r2_map.get(name, 0),
            "raw_rmse": rmse_map.get(name, 999),
        }

    return scores

def _safe_float(x, default=float("nan")):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def compute_pi_ab_summary(pi_by_model: Dict[str, dict]) -> Dict[str, Any]:
    """Build a side-by-side split-conformal vs CV+ comparison per model.

    Each entry in ``pi_by_model`` is expected to have a ``"conformal"`` dict
    (split conformal) and may have a ``"cv_plus"`` dict (Barber+2021 CV+).
    Returns per-model rows + an aggregate block with a recommendation that
    downstream consumers can use to pick the PI to present in the paper.

    Recommendation rubric (requires ≥3 models with both PIs):
      - prefer whichever covers closer to 0.95 AND is not materially wider;
      - if both cover (median ≥ 0.90), pick the tighter (width ratio < 0.95
        → CV+; > 1.05 → split; else "tie");
      - if neither covers, return "neither_covers" to flag the run.
    """
    per_model: List[dict] = []
    for name, entry in (pi_by_model or {}).items():
        if name.startswith("_") or not isinstance(entry, dict):
            continue
        conf = entry.get("conformal") or {}
        cvp = entry.get("cv_plus") or {}
        sw = _safe_float(conf.get("width"))
        sc = _safe_float(conf.get("coverage"))
        cw = _safe_float(cvp.get("width"))
        cc = _safe_float(cvp.get("coverage"))
        ratio = cw / sw if (np.isfinite(sw) and sw > 0 and np.isfinite(cw)) else float("nan")
        dcov = cc - sc if (np.isfinite(sc) and np.isfinite(cc)) else float("nan")
        per_model.append({
            "model": name,
            "split_width": sw,
            "split_coverage": sc,
            "split_source": conf.get("source", "n/a"),
            "cv_plus_width": cw,
            "cv_plus_coverage": cc,
            "cv_plus_n_folds": int(cvp.get("n_folds", 0)) if cvp else 0,
            "cv_plus_n_cal": int(cvp.get("n_cal", 0)) if cvp else 0,
            "width_ratio": ratio,
            "delta_coverage": dcov,
        })

    sw_a = np.array([r["split_width"] for r in per_model], dtype=float)
    cw_a = np.array([r["cv_plus_width"] for r in per_model], dtype=float)
    sc_a = np.array([r["split_coverage"] for r in per_model], dtype=float)
    cc_a = np.array([r["cv_plus_coverage"] for r in per_model], dtype=float)
    both = np.isfinite(sw_a) & np.isfinite(cw_a)
    n_both = int(both.sum())

    def _med(a):
        return float(np.nanmedian(a)) if np.isfinite(a).any() else float("nan")

    agg: Dict[str, Any] = {
        "n_total": len(per_model),
        "n_with_both": n_both,
        "n_with_split_only": int((np.isfinite(sw_a) & ~np.isfinite(cw_a)).sum()),
        "n_with_cv_plus_only": int((~np.isfinite(sw_a) & np.isfinite(cw_a)).sum()),
        "n_with_neither": int((~np.isfinite(sw_a) & ~np.isfinite(cw_a)).sum()),
        "split_width_median": _med(sw_a),
        "split_coverage_median": _med(sc_a),
        "cv_plus_width_median": _med(cw_a),
        "cv_plus_coverage_median": _med(cc_a),
    }

    if n_both >= 3:
        wr = cw_a[both] / sw_a[both]
        agg["width_ratio_median"] = float(np.nanmedian(wr))
        agg["cv_plus_tighter_pct"] = float(np.mean(wr < 1.0))
        sc_med = float(np.nanmedian(sc_a[both]))
        cc_med = float(np.nanmedian(cc_a[both]))
        sc_ok = sc_med >= 0.90
        cc_ok = cc_med >= 0.90
        if cc_ok and not sc_ok:
            rec = "cv_plus"
        elif sc_ok and not cc_ok:
            rec = "split"
        elif sc_ok and cc_ok:
            if agg["width_ratio_median"] < 0.95:
                rec = "cv_plus"
            elif agg["width_ratio_median"] > 1.05:
                rec = "split"
            else:
                rec = "tie"
        else:
            rec = "neither_covers"
        agg["recommendation"] = rec
    else:
        agg["width_ratio_median"] = float("nan")
        agg["cv_plus_tighter_pct"] = float("nan")
        agg["recommendation"] = "insufficient_data"

    return {"per_model": per_model, "aggregate": agg}


from simulation.utils.resource_tracker import track_resources


@track_resources("phase11_scoring")
def run_scoring(wf_results, dm_results, pi_results, config,
               oof_predictions=None, y_all=None, holdout_start=None) -> dict:
    """R8: 종합 스코어링 및 최종 랭킹.

 NNLS-stacking step: when ``oof_predictions`` + ``y_all`` are passed in,
 additionally compute NNLS stacking weights on the OOF slab
 ([0, holdout_start)) via ``.ensemble_weights.oof_nnls_weights``. The
 weights are surfaced as ``nnls_oof_weights`` in the returned dict so
 downstream consumers can blend OOF / holdout predictions without
 refitting the ensemble head on the 15 % val slab.
 """
    from .utils.logging_util import phase_banner, fmt_time
    phase_banner("R8", "종합 스코어 + 최종 랭킹")

    t0 = time.time()

    dm_win_rates = dm_results.get("win_rates", {})

    # PI 결과를 모델별로 재구성
    pi_by_model = pi_results.get("pi_results", {})

    scores = compute_composite_scores(
        wf_results=wf_results.get("wf_results", {}),
        dm_win_rates=dm_win_rates,
        pi_results=pi_by_model,
        config=config,
    )

    # D1: NNLS stacking on the OOF signal (no-op without the inputs).
    nnls_oof_weights: Dict[str, float] = {}
    if oof_predictions is not None and y_all is not None:
        try:
            from .ensemble_weights import oof_nnls_weights
            nnls_oof_weights = oof_nnls_weights(
                oof_predictions,
                np.asarray(y_all),
                holdout_start=holdout_start,
            )
            if nnls_oof_weights:
                tops = sorted(nnls_oof_weights.items(),
                              key=lambda kv: -kv[1])[:5]
                log.info(
                    "  [NNLS-OOF] top weights: "
                    + ", ".join(f"{k}={v:.3f}" for k, v in tops)
                )
        except Exception as e:
            log.warning(f"  [NNLS-OOF] skipped: {e}")

    # Step-2: split-conformal vs CV+ A/B diagnostic.
    ab_summary = compute_pi_ab_summary(pi_by_model)
    agg = ab_summary["aggregate"]
    if agg["n_with_both"] >= 3:
        log.info("  === PI A/B: split-conformal vs CV+ (nominal 0.95) ===")
        log.info(
            f"  {'model':20s}  {'split_w':>8s} {'split_c':>7s}  "
            f"{'cv+_w':>8s} {'cv+_c':>7s}  {'ratio':>6s}  {'Δcov':>7s}"
        )
        for r in sorted(ab_summary["per_model"], key=lambda x: x["model"]):
            if not (np.isfinite(r["split_width"]) and np.isfinite(r["cv_plus_width"])):
                continue
            log.info(
                f"  {r['model']:20s}  "
                f"{r['split_width']:8.2f} {r['split_coverage']:7.1%}  "
                f"{r['cv_plus_width']:8.2f} {r['cv_plus_coverage']:7.1%}  "
                f"{r['width_ratio']:6.3f}  {r['delta_coverage']:+7.1%}"
            )
        log.info(
            f"  [PI A/B] medians: split(w={agg['split_width_median']:.2f}, "
            f"cov={agg['split_coverage_median']:.1%}) | "
            f"CV+(w={agg['cv_plus_width_median']:.2f}, "
            f"cov={agg['cv_plus_coverage_median']:.1%}) | "
            f"ratio={agg['width_ratio_median']:.3f} | "
            f"CV+ tighter in {agg['cv_plus_tighter_pct']:.0%} of models | "
            f"pick={agg['recommendation']}"
        )
    elif agg["n_with_both"] == 0:
        log.info(
            "  [PI A/B] no CV+ entries — only split conformal is available. "
            "Expected when R7 (intervals) did not emit fold_holdout_predictions."
        )
    else:
        log.info(
            f"  [PI A/B] only {agg['n_with_both']} model(s) have both PIs "
            "— skipping comparison (need ≥3)."
        )

    # 랭킹 — DIAGNOSTIC ONLY (A4/M7). 이 composite 랭킹은 진단용 보고일 뿐
    # champion 선택 기준이 아니다. champion = 순수 best-WIS (per_model_eval).
    # composite 는 Borda 통합(comprehensive_eval._ranking_consolidated)에서 제외됨.
    ranking = sorted(scores.items(), key=lambda x: -x[1]["composite"])

    log.info("  === 진단용 랭킹 (Composite Score — champion 아님; champion=best-WIS) ===")
    log.info(f"  {'순위':>4s}  {'모델':18s}  {'Composite':>9s}  {'R²':>7s}  {'RMSE':>7s}  {'DM':>6s}  {'Stab':>6s}  {'Conf':>6s}")
    log.info("  " + "-" * 80)
    for rank, (name, s) in enumerate(ranking, 1):
        log.info(f"  {rank:4d}  {name:18s}  {s['composite']:9.4f}  "
                 f"{s['raw_r2']:7.4f}  {s['raw_rmse']:7.2f}  "
                 f"{s['dm_score']:6.3f}  {s['stability_score']:6.3f}  {s['conformal_score']:6.3f}")

    # 유효성 판정
    validity = {}
    for name, s in scores.items():
        issues = []
        if s["raw_r2"] < 0:
            issues.append("negative_r2")
        if s["raw_rmse"] > 50:
            issues.append("high_rmse")
        if s["stability_score"] < 0.3:
            issues.append("unstable")
        validity[name] = {
            "valid": len(issues) == 0,
            "issues": issues,
        }

    # 2026-05-28 사용자 명시 R3: R8 evaluator (134 metric) per-model on OOF predictions
    phase9_r8_per_model: Dict[str, dict] = {}
    if oof_predictions is not None and y_all is not None:
        try:
            from simulation.pipeline.phase_evaluator import evaluate_predictions_full
            _y_arr = np.asarray(y_all)
            _train_pool = (_y_arr[:holdout_start]
                            if holdout_start is not None and holdout_start > 0
                            else _y_arr)
            for _mname, _pred in oof_predictions.items():
                try:
                    _pred_arr = np.asarray(_pred)
                    if len(_pred_arr) != len(_y_arr):
                        continue
                    _mask = np.isfinite(_y_arr) & np.isfinite(_pred_arr)
                    if _mask.sum() < 5:
                        continue
                    _full_r8 = evaluate_predictions_full(
                        y_test=_y_arr[_mask], y_pred=_pred_arr[_mask],
                        residuals=_y_arr[_mask] - _pred_arr[_mask],
                        sigma=float(np.std(_y_arr)) or 1.0,
                        y_train_pool=_train_pool,
                        threshold=GLOBAL.filter.alert_threshold, phase_id=f"phase9_{_mname}",
                    )
                    phase9_r8_per_model[_mname] = _full_r8
                except Exception as _r8_err:
                    log.warning(f"  [R8] skip {_mname}: {_r8_err}")
        except Exception as _r8_outer:
            log.warning(f"  [R8] evaluator init failed: {_r8_outer}")

    elapsed = time.time() - t0
    log.info(f"\n  ✓ R8 scoring 완료 [{fmt_time(elapsed)}]")

    return {
        "scores": scores,
        "ranking": [(name, s) for name, s in ranking],
        "validity": validity,
        "nnls_oof_weights": nnls_oof_weights,
        "pi_ab": ab_summary,
        "elapsed": elapsed,
        # 2026-05-28 사용자 명시: R8 134-metric per model on OOF
        "r8_per_model": phase9_r8_per_model,
    }


# back-compat aliases (2026-06-02 semantic rename — 옛 run_phaseN)
run_phase11 = run_scoring
