"""
phase_evaluator.py — Unified 129-key evaluator (R8 2026-05-26)
==================================================================

사용자 R8 요청: 모든 evaluation phase (predictions 가 있는 곳) 에서 동일한
129-key set 을 계산. Phase 별 trajectory 비교 + ablation 가능하게.

Mirrors `simulation/pipeline/per_model_eval.py` row dict 구성을
single-model 단위 함수로 분리. R10 (per_model_eval) 의 multi-model rankings + Bootstrap
CI 는 본 evaluator 미포함 (model set 필요).

Application:
  R2 (baseline) ─→ raw predictions per model
  은퇴(구 Phase 8 AR correction) ─→ AR-corrected predictions
  R7 (intervals, PI calibration) ─→ calibrated PI predictions
  R4 (WF-CV) ─→ OOF predictions
  P1 (real_forecaster, 8-week real eval) ─→ rolling-origin predictions
  R10 per_model_eval (SSOT) ─→ final per-model row (full 129-key)
  R9 (per_model_optimize, HP Optuna) ─→ best-HP refit predictions
  Pov (overseas) ─→ per-country predictions

각 phase 가 본 evaluator 를 호출 → trajectory plot 가능:
  R2 WIS → 은퇴(구 Phase 8 AR) WIS → R9 WIS = 단조 감소 (개선) evidence

NOT included (model-set-level metrics, separate post-processing):
- relative_wis_pairwise (Sherratt 2023, needs all models)
- rank_wis / rank_log_wis / rank_mae / rank_r2 (pairwise ranks)
- skill_*_vs_persist (needs persistence baseline reference)
- DM tests vs climatology/lag52 (needs all model errors)
- Bootstrap CI (B=1000 expensive, only final R10 per_model_eval)
- BH-FDR adjusted p-values (multi-test family)

Reference: docs/MASTER_REFERENCE_20260529.md §4 (129-key, 12 sections;
문서 섹션별 합 = 18+28+5+5+9+5+9+32+9+8+5+1 = 134이나, **실제 코드 반환 = 129**
(4-criteria binding 5 flag 미산출, 2026-06-05; "134"는 docstring 섹션합 잔재 = stale)).
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def evaluate_predictions_full(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    *,
    residuals: Optional[np.ndarray] = None,
    sigma: Optional[float] = None,
    y_train_pool: Optional[np.ndarray] = None,
    threshold: float = 8.6,
    dates_test: Optional[list] = None,
    dates_train: Optional[list] = None,
    seed: int = 42,
    phase_id: str = "unknown",
    compute_paper_tier_flags: bool = True,
    # R8.1 multi-model context for FULL 129 SSOT keys (NaN if missing) ──
    baseline_predictions: Optional[dict] = None,
    all_model_wis: Optional[dict] = None,
    all_model_mae: Optional[dict] = None,
    enable_bootstrap_ci: bool = False,
    bootstrap_b: int = 1000,
    bh_fdr_dm_family: bool = True,
) -> dict:
    """Compute the 129-key SSOT battery per-model (single-model subset of R10 per_model_eval row).

    Args:
        y_test: observed (n,)
        y_pred: predictions (n,)
        residuals: OOF residuals (k,) for empirical PI/WIS/Brier. If None,
                   uses test residuals (caveat: leak — disclose downstream).
        sigma: residual std. Auto from `residuals` if None.
        y_train_pool: for MASE_h1/h4/h13/h26/h52 scaling. If None, MASE = NaN.
        threshold: KDCA alert threshold (default 8.6 for 2024-25).
        dates_test: ISO weekly dates for WOY climatology BSS (Reich 2019).
        dates_train: train period dates for WOY prob array.
        seed: RNG seed for bootstrap (default 42).
        phase_id: source phase identifier (added to output for trajectory tracking).
        compute_paper_tier_flags: if True, add PAPER_TOP_{2,3,5,10}_complete bool flags.

    Returns:
        dict 129 keys (subset of R10 per_model_eval row dict). Single-model metrics only.

    Note:
        Excludes:
          - Multi-model rankings (relative_wis_pairwise, rank_*, skill_*_vs_*)
          - DM tests vs baseline (need baseline predictions)
          - Bootstrap CI (expensive, R10 per_model_eval only)
          - BH-FDR adjusted p-values
          - G-175 binding flags (computed at R10 per_model_eval final step)

        These are added separately by R10 per_model_eval multi-model post-processing.

    Performance: O(n × K) where K=11 quantiles, n_test = 37-68. ~0.5s per model.
    """
    # R8.3 (2026-05-26): MPH_FULL_EVAL_TRAJECTORY env toggle.
    # Default '1' = enabled (full 129-key computation, per-phase trajectory analysis).
    # Set '0' to disable globally for cost reduction — returns minimal skip-marker dict.
    # 2026-05-28 사용자 명시 "fast argparse 했을때 주어진대로": MPH_FAST_METRIC alias 추가.
    #   - MPH_FAST_METRIC=1 (fast mode) → MPH_FULL_EVAL_TRAJECTORY=0 와 동일 동작
    #   - default = full (모든 phase 가 129-key SSOT 평가, 사용자 명시 일치)
    from simulation.config_global import GLOBAL as _GCFG  # SSOT (2026-05-28)
    _fast = _GCFG.filter.fast_metric
    _full_eval = _GCFG.filter.full_eval_trajectory
    if _fast or not _full_eval:
        return {
            "phase_id": phase_id,
            "_skipped": True,
            "_reason": "MPH_FAST_METRIC=1 or MPH_FULL_EVAL_TRAJECTORY=0",
        }

    from scipy.stats import (
        norm as _N, kstest as _ks, shapiro as _shap, jarque_bera as _jb,
        skew as _sk, kurtosis as _kt,
    )

    out: dict = {"phase_id": phase_id}

    # ─── Common setup ──────────────────────────────────────────────────
    y_test = np.asarray(y_test, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = len(y_test)
    if n < 5:
        log.warning(f"  [{phase_id}] n_test={n} < 5, returning NaN row")
        return out

    err = y_pred - y_test
    ae = np.abs(err)
    sse = float(np.sum(err ** 2))
    sst = float(np.sum((y_test - y_test.mean()) ** 2))

    # σ
    if sigma is None:
        if residuals is not None and len(residuals) >= 2:
            sigma = float(np.std(residuals))
        else:
            sigma = float(np.std(err)) if n > 1 else 1.0
    sigma = max(float(sigma), 1e-6)

    # residuals fallback
    if residuals is None:
        residuals = err  # leak caveat — disclose in phase callsite
    residuals = np.asarray(residuals, dtype=np.float64)
    residuals = residuals[np.isfinite(residuals)]

    # ─── TIER 1: NAKED (10) ────────────────────────────────────────────
    out["r2"] = 1.0 - sse / sst if sst > 0 else float("nan")
    out["mae"] = float(np.mean(ae))
    out["rmse"] = float(np.sqrt(np.mean(err ** 2)))
    out["mse"] = float(np.mean(err ** 2))
    nz = y_test != 0
    out["mape"] = (float(np.mean(np.abs(err[nz] / y_test[nz])) * 100)
                   if nz.any() else float("nan"))
    den = np.abs(y_test) + np.abs(y_pred)
    keep = den > 0
    out["smape"] = (float(np.mean(2.0 * np.abs(err[keep]) / den[keep]) * 100)
                    if keep.any() else float("nan"))
    out["mdape"] = (float(np.median(np.abs(err[nz] / y_test[nz])) * 100)
                    if nz.any() else float("nan"))
    out["bias_mean_error"] = float(err.mean())
    try:
        out["msle"] = float(np.mean((np.log1p(y_pred) - np.log1p(y_test)) ** 2))
    except Exception:
        out["msle"] = float("nan")
    try:
        out["theils_u"] = float(np.sqrt(np.mean(err ** 2)) /
                                 np.sqrt(np.mean(y_test ** 2)))
    except Exception:
        out["theils_u"] = float("nan")

    # ─── TIER 2: MASE (5, Hyndman 2006) ────────────────────────────────
    if y_train_pool is not None and len(y_train_pool) > 60:
        y_train_pool = np.asarray(y_train_pool, dtype=np.float64)
        for h in (1, 4, 13, 26, 52):
            try:
                diff = np.abs(np.diff(y_train_pool, n=h))
                den_h = diff.mean()
                out[f"mase_h{h}"] = (float(np.mean(ae) / den_h) if den_h > 0
                                     else float("nan"))
            except Exception:
                out[f"mase_h{h}"] = float("nan")
    else:
        for h in (1, 4, 13, 26, 52):
            out[f"mase_h{h}"] = float("nan")

    # ─── TIER 3: THRESHOLD (24) ────────────────────────────────────────
    ev_true = (y_test > threshold).astype(int)
    ev_pred = (y_pred > threshold).astype(int)
    tp = int(((ev_true == 1) & (ev_pred == 1)).sum())
    tn = int(((ev_true == 0) & (ev_pred == 0)).sum())
    fp = int(((ev_true == 0) & (ev_pred == 1)).sum())
    fn = int(((ev_true == 1) & (ev_pred == 0)).sum())
    out.update({"tp": tp, "tn": tn, "fp": fp, "fn": fn})

    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    out["sensitivity"] = sens
    out["specificity"] = spec
    out["ppv"] = ppv
    out["npv"] = npv
    out["accuracy"] = (tp + tn) / n
    out["prevalence"] = (tp + fn) / n
    out["balanced_accuracy"] = ((sens + spec) / 2
                                 if not (np.isnan(sens) or np.isnan(spec))
                                 else float("nan"))
    out["g_mean"] = (float(np.sqrt(sens * spec)) if sens > 0 and spec > 0
                     else float("nan"))
    out["f1"] = (2 * tp / (2 * tp + fp + fn)
                 if (2 * tp + fp + fn) > 0 else float("nan"))
    out["f2_score"] = (5 * tp / (5 * tp + 4 * fn + fp)
                       if (5 * tp + 4 * fn + fp) > 0 else float("nan"))
    out["f05_score"] = (1.25 * tp / (1.25 * tp + 0.25 * fn + fp)
                        if (1.25 * tp + 0.25 * fn + fp) > 0 else float("nan"))
    out["alert_f1"] = out["f1"]
    out["youden_j"] = (sens + spec - 1
                       if not (np.isnan(sens) or np.isnan(spec))
                       else float("nan"))
    mcc_d = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    out["mcc"] = (tp * tn - fp * fn) / mcc_d if mcc_d > 0 else float("nan")
    out["dor"] = (tp * tn) / max(fp * fn, 1)
    out["markedness"] = (ppv + npv - 1
                         if not (np.isnan(ppv) or np.isnan(npv))
                         else float("nan"))
    po = out["accuracy"]
    pe = (((tp + fp) / n) * ((tp + fn) / n) +
          ((tn + fp) / n) * ((tn + fn) / n))
    out["cohens_kappa"] = (po - pe) / (1 - pe) if (1 - pe) > 0 else float("nan")
    out["lr_positive"] = (sens / (1 - spec)
                          if not np.isnan(spec) and spec < 1
                          else float("inf"))
    out["lr_negative"] = ((1 - sens) / spec if not np.isnan(spec) and spec > 0
                          else float("nan"))
    dir_t = np.sign(np.diff(y_test))
    dir_p = np.sign(np.diff(y_pred))
    out["direction_acc"] = float(np.mean(dir_t == dir_p))

    # ─── TIER 4: EMPIRICAL PI (~10) — Lei 2018 ─────────────────────────
    try:
        from simulation.analytics.hub_metrics import (
            k11_pi_widths_from_residuals, FLUSIGHT_ALPHAS,
            coverage_with_exact_ci,
        )
        if len(residuals) >= 10:
            q = k11_pi_widths_from_residuals(np.abs(residuals), FLUSIGHT_ALPHAS)
            for alpha, name in [(0.02, "pi99"), (0.05, "pi95"),
                                  (0.20, "pi80"), (0.50, "pi50")]:
                qh = q.get(float(alpha))
                if qh is None or not np.isfinite(qh):
                    out[f"{name}_coverage"] = float("nan")
                    out[f"{name}_width"] = float("nan")
                    continue
                lo = y_pred - qh
                hi = y_pred + qh
                out[f"{name}_coverage"] = float(((y_test >= lo) &
                                                  (y_test <= hi)).mean())
                out[f"{name}_width"] = float(2 * qh)
            out["pi_sharpness_ratio"] = (2 * q.get(0.05, float("nan"))
                                          / max(float(y_test.std()), 1e-6))
            # Wilson exact CI for pi95
            try:
                cv = coverage_with_exact_ci(
                    y_test, y_pred - q[0.05], y_pred + q[0.05],
                    nominal=0.95, method="wilson",
                )
                out["pi95_ci_lo"] = float(cv.get("ci_lo", float("nan")))
                out["pi95_ci_hi"] = float(cv.get("ci_hi", float("nan")))
            except Exception:
                out["pi95_ci_lo"] = out["pi95_ci_hi"] = float("nan")
            # |relia| = |empirical - nominal|
            out["pi95_relia"] = (abs(out["pi95_coverage"] - 0.95)
                                  if np.isfinite(out["pi95_coverage"])
                                  else float("nan"))
            out["pi80_relia"] = (abs(out["pi80_coverage"] - 0.80)
                                  if np.isfinite(out["pi80_coverage"])
                                  else float("nan"))
        else:
            for alpha, name in [(0.02, "pi99"), (0.05, "pi95"),
                                  (0.20, "pi80"), (0.50, "pi50")]:
                out[f"{name}_coverage"] = float("nan")
                out[f"{name}_width"] = float("nan")
            out["pi95_relia"] = out["pi80_relia"] = float("nan")
            out["pi_sharpness_ratio"] = float("nan")
            out["pi95_ci_lo"] = out["pi95_ci_hi"] = float("nan")
    except Exception as _e:
        log.debug(f"  [{phase_id}] PI block failed: {_e}")

    out["sigma_in_sample"] = sigma
    out["alert_threshold"] = float(threshold)

    # ─── TIER 5: EMPIRICAL WIS family + Brier + ROC + Calibration ──────
    try:
        from simulation.analytics.diagnostics import (
            weighted_interval_score_empirical,
        )
        from simulation.analytics.hub_metrics import (
            weighted_interval_score_logscale_empirical,
            weighted_interval_score_components_empirical,
            crps_empirical, FLUSIGHT_ALPHAS,
        )
        if len(residuals) >= 10:
            wis_arr = weighted_interval_score_empirical(
                y_test, y_pred, residuals, alphas=list(FLUSIGHT_ALPHAS),
            )
            out["wis"] = float(np.mean(wis_arr))
            out["log_wis"] = float(
                weighted_interval_score_logscale_empirical(
                    y_test, y_pred, residuals, alphas=FLUSIGHT_ALPHAS,
                ).mean()
            )
            decomp = weighted_interval_score_components_empirical(
                y_test, y_pred, residuals, alphas=FLUSIGHT_ALPHAS,
            )
            out["wis_sharpness"] = decomp.get("wis_sharpness", float("nan"))
            out["wis_underpred"] = decomp.get("wis_underpred", float("nan"))
            out["wis_overpred"] = decomp.get("wis_overpred", float("nan"))
            out["wis_total_decomp"] = decomp.get("wis_total_decomp", float("nan"))
            # CRPS empirical
            rng = np.random.default_rng(seed)
            M = 1000
            samples = y_pred[:, None] + rng.choice(
                residuals, size=(n, M), replace=True,
            )
            out["crps_gaussian"] = float(crps_empirical(y_test, samples))
            # Pinball empirical
            from simulation.analytics.metrics import pinball_loss
            q90 = k11_pi_widths_from_residuals(np.abs(residuals), (0.10,))[0.10]
            out["pinball_q05"] = float(pinball_loss(y_test, y_pred - q90, 0.05))
            out["pinball_q95"] = float(pinball_loss(y_test, y_pred + q90, 0.95))
        else:
            for k in ("wis", "log_wis", "wis_sharpness", "wis_underpred",
                      "wis_overpred", "wis_total_decomp", "crps_gaussian",
                      "pinball_q05", "pinball_q95"):
                out[k] = float("nan")
    except Exception as _e:
        log.debug(f"  [{phase_id}] WIS/CRPS block failed: {_e}")

    # Brier + ROC + calibration (empirical bootstrap ev_prob)
    try:
        if len(residuals) >= 10:
            rng2 = np.random.default_rng(seed)
            M = 1000
            res_samples = rng2.choice(residuals, size=(M, n), replace=True)
            ev_prob = np.mean(y_pred[None, :] + res_samples > threshold, axis=0)
        else:
            z = (threshold - y_pred) / sigma
            ev_prob = (1.0 - _N.cdf(z)).astype(np.float64)
        out["brier_score"] = float(np.mean((ev_true - ev_prob) ** 2))
        # WOY climatology baseline (R6 G3)
        if (dates_test is not None and dates_train is not None
                and y_train_pool is not None
                and len(dates_train) == len(y_train_pool)
                and len(dates_test) == n):
            try:
                import pandas as _pd
                _dtr = _pd.to_datetime(list(dates_train))
                _dte = _pd.to_datetime(list(dates_test))
                _woy_tr = _dtr.isocalendar().week.to_numpy()
                _woy_te = _dte.isocalendar().week.to_numpy()
                _tb = (np.asarray(y_train_pool) > threshold)
                _woy_prob = np.full(54, float(np.mean(_tb)))
                for w in range(1, 54):
                    mw = _woy_tr == w
                    if mw.sum() >= 2:
                        _woy_prob[w] = float(_tb[mw].mean())
                _clim = np.array([_woy_prob[int(w)] if 1 <= int(w) <= 53
                                    else float(np.mean(_tb))
                                    for w in _woy_te])
                _bs_base = float(np.mean((ev_true - _clim) ** 2))
                out["brier_skill"] = (1.0 - out["brier_score"] / _bs_base
                                       if _bs_base > 0 else float("nan"))
            except Exception:
                ref_p = (float(np.mean(np.asarray(y_train_pool) > threshold))
                          if y_train_pool is not None else float(ev_true.mean()))
                _unc = ref_p * (1 - ref_p)
                out["brier_skill"] = (1.0 - out["brier_score"] / _unc
                                       if 0 < _unc < 1 else float("nan"))
        else:
            ref_p = (float(np.mean(np.asarray(y_train_pool) > threshold))
                      if y_train_pool is not None else float(ev_true.mean()))
            _unc = ref_p * (1 - ref_p)
            out["brier_skill"] = (1.0 - out["brier_score"] / _unc
                                   if 0 < _unc < 1 else float("nan"))
        # Brier decomposition (Murphy 1973)
        try:
            from simulation.analytics.metrics import brier_decomposition
            bd = brier_decomposition(ev_true, ev_prob, n_bins=10)
            out["brier_reliability"] = float(bd.get("reliability", float("nan")))
            out["brier_resolution"] = float(bd.get("resolution", float("nan")))
            out["brier_uncertainty"] = float(bd.get("uncertainty", float("nan")))
        except Exception:
            out["brier_reliability"] = float("nan")
            out["brier_resolution"] = float("nan")
            out["brier_uncertainty"] = float("nan")
        # ROC family
        try:
            from sklearn.metrics import (
                roc_auc_score, average_precision_score, roc_curve, auc as _auc,
            )
            if len(set(ev_true)) > 1:
                out["roc_auc"] = float(roc_auc_score(ev_true, ev_prob))
                out["auprc"] = float(average_precision_score(ev_true, ev_prob))
                fpr, tpr, _ = roc_curve(ev_true, ev_prob)
                mask = fpr <= 0.1
                out["partial_auc_high_spec"] = (float(_auc(fpr[mask], tpr[mask]))
                                                  if mask.sum() >= 2
                                                  else float("nan"))
            else:
                out["roc_auc"] = out["auprc"] = out["partial_auc_high_spec"] = float("nan")
        except Exception:
            out["roc_auc"] = out["auprc"] = out["partial_auc_high_spec"] = float("nan")
        # Calibration (Cox 1958)
        try:
            from sklearn.linear_model import LogisticRegression
            ep_c = np.clip(ev_prob, 1e-6, 1 - 1e-6)
            logit = np.log(ep_c / (1 - ep_c))
            lr_model = LogisticRegression(fit_intercept=True).fit(
                logit.reshape(-1, 1), ev_true,
            )
            out["calibration_slope"] = float(lr_model.coef_[0, 0])
            out["calibration_intercept"] = float(lr_model.intercept_[0])
        except Exception:
            out["calibration_slope"] = out["calibration_intercept"] = float("nan")
        # Rank-PIT (Czado 2009)
        try:
            if len(residuals) >= 10:
                rng_pit = np.random.default_rng(seed)
                Mp = 999
                ps = y_pred[:, None] + rng_pit.choice(
                    residuals, size=(n, Mp), replace=True,
                )
                ranks = np.sum(ps < y_test[:, None], axis=1)
                pit_a = (ranks + 0.5) / (Mp + 1.0)
                out["pit_mean"] = float(pit_a.mean())
                out["pit_std"] = float(pit_a.std())
                out["pit_ks_p"] = float(_ks(pit_a, "uniform")[1])
            else:
                out["pit_mean"] = out["pit_std"] = out["pit_ks_p"] = float("nan")
        except Exception:
            out["pit_mean"] = out["pit_std"] = out["pit_ks_p"] = float("nan")
    except Exception as _e:
        log.debug(f"  [{phase_id}] Brier/ROC/Calibration block failed: {_e}")

    # ─── TIER 6: EPI-CURVE (12) ────────────────────────────────────────
    try:
        peak_t = int(np.argmax(y_test))
        peak_p = int(np.argmax(y_pred))
        out["peak_week_err"] = abs(peak_p - peak_t)
        out["peak_int_relerr"] = (abs(float(np.max(y_pred)) - float(np.max(y_test)))
                                    / max(float(np.max(y_test)), 1e-6))
        out["epi_peak_mae"] = float(np.mean(np.abs(err[max(0, peak_t - 2):peak_t + 3])))
        out["epi_season_total_mae"] = abs(float(np.sum(y_pred)) - float(np.sum(y_test)))
        out["attack_rate_relerr"] = (out["epi_season_total_mae"]
                                       / max(float(np.sum(y_test)), 1e-6))
        gr_t = np.diff(y_test) / np.maximum(y_test[:-1], 1e-6)
        gr_p = np.diff(y_pred) / np.maximum(y_pred[:-1], 1e-6)
        out["growth_rate_corr"] = float(np.corrcoef(gr_t, gr_p)[0, 1])
        above_t = (np.argmax(y_test > threshold)
                    if (y_test > threshold).any() else -1)
        above_p = (np.argmax(y_pred > threshold)
                    if (y_pred > threshold).any() else -1)
        out["lead_time_weeks"] = (above_t - above_p
                                    if above_t >= 0 and above_p >= 0
                                    else float("nan"))
        out["season_onset_err"] = out["lead_time_weeks"]
        out["epidemic_duration_err"] = abs(int(np.sum(y_pred > threshold))
                                              - int(np.sum(y_test > threshold)))
        try:
            from scipy.stats import pearsonr as _pr, spearmanr as _sr
            out["pearson_r"] = float(_pr(y_test, y_pred)[0])
            out["spearman_r"] = float(_sr(y_test, y_pred)[0])
        except Exception:
            out["pearson_r"] = out["spearman_r"] = float("nan")
        # c-index (Harrell)
        try:
            c_conc = c_total = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if y_test[i] == y_test[j]:
                        continue
                    c_total += 1
                    if ((y_test[i] > y_test[j] and y_pred[i] > y_pred[j]) or
                            (y_test[i] < y_test[j] and y_pred[i] < y_pred[j])):
                        c_conc += 1
            out["c_index"] = c_conc / c_total if c_total > 0 else float("nan")
        except Exception:
            out["c_index"] = float("nan")
    except Exception as _e:
        log.debug(f"  [{phase_id}] EPI-CURVE block failed: {_e}")

    # ─── Cost skill ─────────────────────────────────────────────────────
    try:
        for ratio in (3, 5, 10):
            cost = fn * ratio + fp
            max_cost = n * ratio
            out[f"cost_skill_{ratio}to1"] = (1 - cost / max_cost
                                               if max_cost > 0
                                               else float("nan"))
    except Exception:
        for ratio in (3, 5, 10):
            out[f"cost_skill_{ratio}to1"] = float("nan")

    # ─── TIER 8: RESIDUAL DIAG (R6 — JB, DW, skew, kurt) ───────────────
    try:
        if len(err) >= 5:
            try:
                out["shapiro_wilk_p"] = float(_shap(err)[1])
            except Exception:
                out["shapiro_wilk_p"] = float("nan")
            try:
                out["jarque_bera_p"] = float(_jb(err)[1])
            except Exception:
                out["jarque_bera_p"] = float("nan")
            try:
                out["residual_skew"] = float(_sk(err))
                out["residual_kurtosis"] = float(_kt(err))
            except Exception:
                out["residual_skew"] = out["residual_kurtosis"] = float("nan")
            try:
                _rm = err - err.mean()
                _var = float(np.var(_rm))
                out["residual_acf_lag1"] = (float(np.mean(_rm[1:] * _rm[:-1]) / _var)
                                              if _var > 1e-10 else float("nan"))
            except Exception:
                out["residual_acf_lag1"] = float("nan")
            try:
                _dr = np.diff(err)
                _dw_num = float(np.sum(_dr ** 2))
                _dw_den = float(np.sum(err ** 2))
                out["durbin_watson"] = (_dw_num / _dw_den if _dw_den > 1e-10
                                          else float("nan"))
            except Exception:
                out["durbin_watson"] = float("nan")
            try:
                from statsmodels.stats.diagnostic import acorr_ljungbox as _alb
                _lb = _alb(err, lags=[min(10, n // 3)], return_df=True)
                out["ljung_box_q"] = float(_lb["lb_stat"].iloc[-1])
                out["ljung_box_p"] = float(_lb["lb_pvalue"].iloc[-1])
            except Exception:
                out["ljung_box_q"] = out["ljung_box_p"] = float("nan")
        else:
            for k in ("shapiro_wilk_p", "jarque_bera_p", "residual_skew",
                       "residual_kurtosis", "residual_acf_lag1",
                       "durbin_watson", "ljung_box_q", "ljung_box_p"):
                out[k] = float("nan")
    except Exception as _e:
        log.debug(f"  [{phase_id}] residual diag block failed: {_e}")

    # ─── Sample sizes ───────────────────────────────────────────────────
    out["n_test"] = int(n)
    out["n_valid"] = int(np.isfinite(y_pred).sum())

    # Sprint D2 (2026-05-26): PAPER_TOP_{2,3,5,10} 완전 폐기.
    # 사용자 명시 "paper top tier은 왜 만들어?!" — R7 의 paper_top*_complete
    # boolean flags 모두 제거. compute_paper_tier_flags kwarg 는 back-compat 위해
    # signature 에는 유지하되 silently ignored.

    # ════════════════════════════════════════════════════════════════════
    # R8.1 (2026-05-26) — FULL 129 SSOT extension
    # 모든 evaluation phase 에서 동일 129 shape 반환.
    # Multi-model context 없으면 NaN (R10 per_model_eval 가 post-loop 에서 채움).
    # ════════════════════════════════════════════════════════════════════

    # ─── Baseline DM tests (R6 dm_test style — needs baseline preds) ──────
    out["dm_z_stat"] = float("nan")
    out["dm_p_value"] = float("nan")
    out["dm_z_vs_climatology"] = float("nan")
    out["dm_p_vs_climatology"] = float("nan")
    out["dm_z_vs_lag52"] = float("nan")
    out["dm_p_vs_lag52"] = float("nan")
    if baseline_predictions is not None:
        try:
            from scipy.stats import norm as _Nbase
            def _dm(e1, e2):
                d = e1 ** 2 - e2 ** 2
                d = d[np.isfinite(d)]
                if len(d) < 5:
                    return float("nan"), float("nan")
                z = float(np.mean(d) / (np.std(d) / np.sqrt(len(d))))
                return z, float(2.0 * (1 - _Nbase.cdf(abs(z))))
            # vs persistence (lag-1)
            if "persist" in baseline_predictions:
                pp = np.asarray(baseline_predictions["persist"], dtype=np.float64)
                if len(pp) == n:
                    z, p = _dm(err, pp - y_test)
                    out["dm_z_stat"] = z
                    out["dm_p_value"] = p
            if "climatology" in baseline_predictions:
                cp = np.asarray(baseline_predictions["climatology"], dtype=np.float64)
                if len(cp) == n:
                    z, p = _dm(err, cp - y_test)
                    out["dm_z_vs_climatology"] = z
                    out["dm_p_vs_climatology"] = p
            if "lag52" in baseline_predictions:
                lp = np.asarray(baseline_predictions["lag52"], dtype=np.float64)
                if len(lp) == n:
                    z, p = _dm(err, lp - y_test)
                    out["dm_z_vs_lag52"] = z
                    out["dm_p_vs_lag52"] = p
        except Exception as _e:
            log.debug(f"  [{phase_id}] DM baseline block failed: {_e}")

    # BH-FDR adjusted DM p-values (R3 S9)
    out["dm_p_value_bh"] = float("nan")
    out["dm_p_vs_climatology_bh"] = float("nan")
    out["dm_p_vs_lag52_bh"] = float("nan")
    if bh_fdr_dm_family:
        try:
            from simulation.analytics.metrics import adjust_pvalues
            raw_ps = [out["dm_p_value"], out["dm_p_vs_climatology"],
                      out["dm_p_vs_lag52"]]
            finite_idx = [i for i, p in enumerate(raw_ps) if np.isfinite(p)]
            if len(finite_idx) >= 2:
                finite_ps = [raw_ps[i] for i in finite_idx]
                bh = adjust_pvalues(finite_ps, method="fdr_bh")
                _adj = list(bh["p_adj"])
                _out_bh = [float("nan")] * 3
                for j, i in enumerate(finite_idx):
                    _out_bh[i] = float(_adj[j])
                out["dm_p_value_bh"] = _out_bh[0]
                out["dm_p_vs_climatology_bh"] = _out_bh[1]
                out["dm_p_vs_lag52_bh"] = _out_bh[2]
        except Exception as _e:
            log.debug(f"  [{phase_id}] BH-FDR failed: {_e}")

    # ─── Skill scores vs baselines ──────────────────────────────────────
    out["skill_mae_vs_persist"] = float("nan")
    out["skill_wis_vs_persist"] = float("nan")
    out["skill_crps_vs_persist"] = float("nan")
    out["skill_mae_vs_snaive"] = float("nan")
    if baseline_predictions is not None:
        try:
            from simulation.analytics.hub_metrics import relative_skill_score
            if "persist" in baseline_predictions:
                pp = np.asarray(baseline_predictions["persist"], dtype=np.float64)
                mae_p = float(np.mean(np.abs(pp - y_test)))
                out["skill_mae_vs_persist"] = (
                    float(relative_skill_score(out["mae"], mae_p, lower_is_better=True))
                    if mae_p > 1e-12 else float("nan")
                )
                # WIS / CRPS vs persistence (if residuals available)
                if len(residuals) >= 10 and np.isfinite(out.get("wis", float("nan"))):
                    try:
                        from simulation.analytics.diagnostics import (
                            weighted_interval_score_empirical as _wis_emp,
                        )
                        from simulation.analytics.hub_metrics import (
                            FLUSIGHT_ALPHAS, crps_empirical as _crps_emp,
                        )
                        persist_res = y_test - pp
                        p_wis = float(_wis_emp(y_test, pp, persist_res,
                                                alphas=list(FLUSIGHT_ALPHAS)).mean())
                        out["skill_wis_vs_persist"] = (
                            float(relative_skill_score(out["wis"], p_wis,
                                                         lower_is_better=True))
                            if p_wis > 1e-12 else float("nan")
                        )
                        rng_sc = np.random.default_rng(seed)
                        p_samples = pp[:, None] + rng_sc.choice(
                            persist_res, size=(n, 1000), replace=True,
                        )
                        p_crps = float(_crps_emp(y_test, p_samples))
                        out["skill_crps_vs_persist"] = (
                            float(relative_skill_score(out["crps_gaussian"], p_crps,
                                                         lower_is_better=True))
                            if p_crps > 1e-12 else float("nan")
                        )
                    except Exception:
                        pass
            if "lag52" in baseline_predictions:
                lp = np.asarray(baseline_predictions["lag52"], dtype=np.float64)
                mae_l = float(np.mean(np.abs(lp - y_test)))
                out["skill_mae_vs_snaive"] = (
                    float(relative_skill_score(out["mae"], mae_l, lower_is_better=True))
                    if mae_l > 1e-12 else float("nan")
                )
        except Exception as _e:
            log.debug(f"  [{phase_id}] skill scores failed: {_e}")

    # ─── Multi-model rankings (needs all_model_*) ──────────────────────
    out["relative_wis_pairwise"] = float("nan")
    out["rank_wis"] = float("nan")
    if all_model_wis is not None:
        try:
            from simulation.analytics.hub_metrics import pairwise_relative_wis
            # Need WIS arrays per model
            wis_per_model = {k: np.atleast_1d(v) for k, v in all_model_wis.items()}
            if len(wis_per_model) >= 2:
                rel = pairwise_relative_wis(wis_per_model)
                this_model_key = phase_id  # caller can set phase_id=model_name
                if this_model_key in rel:
                    out["relative_wis_pairwise"] = float(rel[this_model_key])
                # Rank
                sorted_models = sorted(rel.items(), key=lambda kv: kv[1])
                for rank, (m, _) in enumerate(sorted_models, 1):
                    if m == this_model_key:
                        out["rank_wis"] = rank
                        break
        except Exception as _e:
            log.debug(f"  [{phase_id}] pairwise WIS failed: {_e}")

    # ─── Bootstrap CI (R3 S9) — expensive, opt-in ──────────────────────
    out["mae_ci95_lo"] = float("nan")
    out["mae_ci95_hi"] = float("nan")
    out["wis_ci95_lo"] = float("nan")
    out["wis_ci95_hi"] = float("nan")
    if enable_bootstrap_ci:
        try:
            from simulation.analytics.metrics import bootstrap_ci
            block_len = max(int(np.sqrt(n)), 2)
            ci = bootstrap_ci(ae, statistic=np.mean, n_boot=bootstrap_b,
                              alpha=0.05, method="bca", random_state=seed,
                              block_len=block_len)
            out["mae_ci95_lo"] = float(ci.get("ci_lo", float("nan")))
            out["mae_ci95_hi"] = float(ci.get("ci_hi", float("nan")))
            # WIS block bootstrap
            if np.isfinite(out.get("wis", float("nan"))) and len(residuals) >= 10:
                from simulation.analytics.diagnostics import (
                    weighted_interval_score_empirical as _wis_b,
                )
                from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
                rng_b = np.random.default_rng(seed)
                L = block_len
                nb = (n + L - 1) // L
                ms = max(1, n - L + 1)
                wis_boots = np.empty(bootstrap_b)
                for b in range(bootstrap_b):
                    starts = rng_b.integers(0, ms, size=nb)
                    idx = np.concatenate([np.arange(s, s + L)
                                            for s in starts])[:n]
                    wis_boots[b] = _wis_b(
                        y_test[idx], y_pred[idx], residuals,
                        alphas=list(FLUSIGHT_ALPHAS),
                    ).mean()
                out["wis_ci95_lo"] = float(np.quantile(wis_boots, 0.025))
                out["wis_ci95_hi"] = float(np.quantile(wis_boots, 0.975))
        except Exception as _e:
            log.debug(f"  [{phase_id}] Bootstrap CI failed: {_e}")

    # ─── G-175 4-criteria binding flags: REMOVED 2026-06-05 ───────────────
    # 사용자 명시: 4-criteria(g175) 완전 제거. champion = 순수 best-WIS, R²/MAPE/WIS/PICP
    # 는 개별 metric 으로만 존재(위). g175_r2_pass/mape_pass/wis_pass/pi95_coverage_pass/
    # 4criteria_pass 5개 종합 flag 산출 안 함 → metric count 134→129.
    # `enable_g175_binding` 인자도 제거 (전 call-site 정리 완료 2026-06-05).

    # PI rel widths (R6 surface)
    try:
        y_mean = float(np.abs(np.mean(y_test)))
        if y_mean > 1e-6:
            out["pi50_rel_width"] = out.get("pi50_width", float("nan")) / y_mean
            out["pi80_rel_width"] = out.get("pi80_width", float("nan")) / y_mean
            out["pi95_rel_width"] = out.get("pi95_width", float("nan")) / y_mean
        else:
            out["pi50_rel_width"] = out["pi80_rel_width"] = out["pi95_rel_width"] = float("nan")
    except Exception:
        out["pi50_rel_width"] = out["pi80_rel_width"] = out["pi95_rel_width"] = float("nan")

    # Additional rank flags (filled by R10 per_model_eval multi-model post-loop)
    out["rank_log_wis"] = float("nan")
    out["rank_mae"] = float("nan")
    out["rank_r2"] = float("nan")

    # Bootstrap CI aliases
    out["mae_ci95_lo_bs"] = out.get("mae_ci95_lo", float("nan"))
    out["mae_ci95_hi_bs"] = out.get("mae_ci95_hi", float("nan"))

    return out


__all__ = ["evaluate_predictions_full"]
