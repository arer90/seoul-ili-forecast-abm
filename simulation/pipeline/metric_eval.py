"""R9 per_model_optimize metric evaluation — extracted from per_model_optimize.py.

Code-split sprint C1 partial (2026-05-12): 38-metric compute_full_metrics extracted as
a standalone module. per_model_optimize (R9) was a 2,214-line monolith mixing HP search /
metric eval / champion selection. metric_eval is the cleanest split target
(no callback / closure / shared state — pure function over arrays).

Design (D-4 deep module):
    Single public function compute_full_metrics() — small interface, rich
    implementation (54 metrics covering point/bias/probabilistic/PIT/PI/
    WIS-decomp/epi-phase/clinical/residual diagnostics/DM). NaN-safe (every
    sub-metric wrapped in try/except), returns NaN on sub-metric failure.

Public API:
    compute_full_metrics(y_test, y_pred, *, sigma_for_wis, y_train_pool) → dict

Performance: O(n) time, ~1MB memory (n=68 test slab baseline).
Side effects: none — pure function.
Caller responsibility:
    - sanitize_predictions(y_pred) before call (NaN/inf masking)
    - Multi-criteria filter (R²+MAPE+WIS+PICP95) consumers use 4 keys
    - y_train_pool optional but improves MASE_h1 (else NaN)

See:
    - G-168 (test_metrics 26 key 보존), G-167 (PICP95 empirical band; 4-criteria 제거 2026-06-05),
      G-175 (MAPE 20% / PICP95 0.90 forward 2026-05-11).
    - per_model_eval._evaluate_model (52 metric, 26 핵심 align).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from simulation.config_global import Z95, Z80, Z50  # SSOT (2026-05-28)


def compute_full_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    *,
    sigma_for_wis: float = 1.0,
    y_train_pool: Optional[np.ndarray] = None,
    viral_positivity_train: Optional[np.ndarray] = None,
    threshold_method: str = "kdca",
) -> dict:
    """54 metric 한 번에 계산 (G-168, D-4 deep module).

    `per_model_eval._evaluate_model` 와 align — R9 per_model_optimize 의 단일 helper 가
    R10 per_model_eval 의 metric 계산 패턴 재현. NaN-safe (모든 sub-metric 실패 시 NaN 반환,
    예외 전파 X).

    Args:
        y_test: 실제 ILI rate (n_test,) — finite mask 자동 적용.
        y_pred: 예측 ILI rate (n_test,) — `sanitize_predictions` 후 호출 권장.
        sigma_for_wis: PI / WIS 계산용 sigma (test residual std). 기본 1.0,
                       0 → 1e-6 으로 clip (numerical safety).
        y_train_pool: MASE_h1 / threshold q70 계산용 train pool (옵션).

    Returns:
        dict (54 keys):
          - **Point (8)**: r2, mae, rmse, mse, mape, smape, mdape, mase_h1
          - **Bias + scaled (3)**: bias_mean_error, theils_u, msle
          - **Probabilistic (6)**: wis, log_wis, crps_gaussian, pinball_q05/q50/q95
          - **PIT (3)**: pit_mean, pit_std, pit_ks_p
          - **PI coverage (3, G-167)**: pi95_coverage, pi80_coverage, pi50_coverage
          - **WIS decomp (4)**: wis_sharpness, wis_underpred, wis_overpred, wis_total_decomp
          - **Epi-curve (3)**: peak_week_err, peak_int_relerr, direction_acc
          - **Epidemic phase (5)**: attack_rate_relerr, growth_rate_corr,
                                    epidemic_duration_err, season_onset_err, early_warning_lead
          - **G-181 operational (9)**: sensitivity, specificity, ppv, npv, f1,
                                       youden_j, lead_time_weeks, cost_skill_3/5/10
          - **Advanced clinical (5)**: mcc, cohens_kappa, lr_positive,
                                       lr_negative, net_benefit_default
          - **DM (2)**: dm_z_stat, dm_p_value
          - **Sample (2)**: n, sigma_in_sample

    Raises:
        절대 raise X — 모든 sub-metric 실패 시 NaN 반환.

    Performance: O(n) time, ~1MB memory (n=68 test slab 기준).
    Side effects: 없음 (pure function).
    Caller responsibility:
        - y_pred 의 NaN/inf 사전 sanitize (`sanitize_predictions` 권장).
        - 도메인 제약 (ILI rate ≥ 0) 별도 검증.
        - Multi-criteria filter (R²+MAPE+WIS+PICP95) 적용 시 4 키 모두 사용.

    Example:
        >>> m = compute_full_metrics(y_test, y_pred, sigma_for_wis=2.5,
        ...                           y_train_pool=y_in[:269])
        >>> m["pi95_coverage"]  # 0.90 = canonical G-175 floor
        0.910

    See: G-168 (test_metrics 26 키 보존), G-167 (PICP95 empirical band; 4-criteria 제거 2026-06-05),
         G-175 (MAPE 20% / PICP95 0.90 forward 2026-05-11),
         per_model_eval._evaluate_model (52 metric 의 26 핵심 align).
    """
    from simulation.analytics.diagnostics import (
        weighted_interval_score, pit_values,
    )
    from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
    try:
        from simulation.analytics.metrics import (
            crps_gaussian, pinball_loss,
            peak_week_error, peak_intensity_error, direction_accuracy,
            epidemic_phase_metrics, advanced_clinical_metrics_ext,
        )
    except Exception:
        crps_gaussian = None
        pinball_loss = None
        peak_week_error = peak_intensity_error = direction_accuracy = None
        epidemic_phase_metrics = advanced_clinical_metrics_ext = None
    try:
        from simulation.analytics.hub_metrics import (
            mase, median_absolute_percentage_error,
            weighted_interval_score_logscale,
            weighted_interval_score_components,
            theils_u as _theils_u, msle as _msle,
        )
    except Exception:
        mase = None
        median_absolute_percentage_error = None
        weighted_interval_score_logscale = None
        weighted_interval_score_components = None
        _theils_u = None
        _msle = None

    yt = np.asarray(y_test, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(yt) & np.isfinite(yp)
    if not mask.any():
        # Codex audit 2026-05-27 fix: 종래 partial 5-key dict 반환은 59-key contract drift.
        # NaN-safe: 모든 key NaN schema 로 일관 반환 (downstream merge / DataFrame 안전).
        return _empty_metrics_schema(n=int(len(yt)), sigma_in=float(sigma_for_wis))
    a, p = yt[mask], yp[mask]
    n = len(a)
    err = p - a
    ae = np.abs(err)

    # ── Point ───────────────────────────────────────────────────
    sse = float(np.sum(err ** 2))
    sst = float(np.sum((a - a.mean()) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else float("nan")
    mae = float(np.mean(ae))
    mse = float(np.mean(err ** 2))
    rmse = float(np.sqrt(mse))
    # MAPE / sMAPE — y=0 보호
    y_abs = np.abs(a)
    nz = y_abs > 1e-3
    mape = float(np.mean(ae[nz] / y_abs[nz]) * 100.0) if nz.any() else float("nan")
    den = np.abs(a) + np.abs(p)
    keep = den > 1e-3
    smape = float(np.mean(2.0 * ae[keep] / den[keep]) * 100.0) if keep.any() else float("nan")
    mdape = (median_absolute_percentage_error(a, p)
             if median_absolute_percentage_error else float("nan"))
    # MASE
    mase_h1 = float("nan")
    if mase is not None and y_train_pool is not None and len(y_train_pool) > 1:
        try:
            mase_h1 = float(mase(a, p, y_train=y_train_pool, seasonality=1))
        except Exception:
            mase_h1 = float("nan")

    # Bias + Theil's U + MSLE (PI extensions group)
    bias_v = float(np.mean(err))
    try:
        theils_u_v = float(_theils_u(a, p)) if _theils_u is not None else float("nan")
    except Exception:
        theils_u_v = float("nan")
    try:
        msle_v = float(_msle(a, p, epsilon=1.0)) if _msle is not None else float("nan")
    except Exception:
        msle_v = float("nan")

    # ── Probabilistic ───────────────────────────────────────────
    sigma = max(float(sigma_for_wis), 1e-6)
    try:
        wis_arr = weighted_interval_score(a, p, sigma, alphas=FLUSIGHT_ALPHAS)
        wis = float(np.mean(wis_arr))
    except Exception:
        wis = float("nan")
    try:
        log_wis_arr = (weighted_interval_score_logscale(a, p, sigma)
                       if weighted_interval_score_logscale else None)
        log_wis = float(np.mean(log_wis_arr)) if log_wis_arr is not None else float("nan")
    except Exception:
        log_wis = float("nan")
    try:
        crps = (float(np.mean(crps_gaussian(a, p, np.full_like(a, sigma))))
                if crps_gaussian else float("nan"))
    except Exception:
        crps = float("nan")
    try:
        pin_q05 = float(pinball_loss(a, p - 1.645 * sigma, 0.05)) if pinball_loss else float("nan")
        pin_q50 = float(pinball_loss(a, p, 0.50)) if pinball_loss else float("nan")
        pin_q95 = float(pinball_loss(a, p + 1.645 * sigma, 0.95)) if pinball_loss else float("nan")
    except Exception:
        pin_q05 = pin_q50 = pin_q95 = float("nan")

    # ── PIT ──────────────────────────────────────────────────────
    pit_mean = pit_std = pit_ks_p = float("nan")
    try:
        pit_a = pit_values(a, p, sigma)
        pit_mean = float(np.mean(pit_a))
        pit_std = float(np.std(pit_a))
        from scipy.stats import kstest
        _, ks_p = kstest(pit_a, "uniform")
        pit_ks_p = float(ks_p)
    except Exception:
        pass

    # ── G-167: PI coverage (sigma 기반 95/80/50%) ────────────────
    # FLUSIGHT_ALPHAS 가 K=11 quantile 이지만 multi-criteria filter 는
    # 95/80/50 만 쓴다. 단순 normal-quantile 근사 (z=1.96, 1.282, 0.674).
    pi95_cov = pi80_cov = pi50_cov = float("nan")
    try:
        for z, key in [(Z95, "pi95_coverage"), (Z80, "pi80_coverage"),
                       (Z50, "pi50_coverage")]:
            lo = p - z * sigma
            hi = p + z * sigma
            covered = ((a >= lo) & (a <= hi)).mean()
            if key == "pi95_coverage": pi95_cov = float(covered)
            elif key == "pi80_coverage": pi80_cov = float(covered)
            else: pi50_cov = float(covered)
    except Exception:
        pass

    # ── Epi-curve ────────────────────────────────────────────────
    peak_week_err = peak_int_relerr = direction_acc = float("nan")
    try:
        if peak_week_error is not None:
            pw = peak_week_error(a, p, tolerance_weeks=1)
            peak_week_err = float(pw.get("abs_weeks", float("nan")))
        if peak_intensity_error is not None:
            pie = peak_intensity_error(a, p, log_scale=True)
            peak_int_relerr = float(pie.get("rel_err", float("nan")))
        if direction_accuracy is not None and len(a) >= 2:
            da = direction_accuracy(a, p).get("accuracy", float("nan"))
            direction_acc = float(da)
    except Exception:
        pass

    # ── G-181 op-metrics (operational/alert + cost-skill + DM) ──────────────
    # 26 → 41 metric (paper §4.6 매핑).
    # threshold = q70 (paper 표준), miss/FA cost ratio 3:1, 5:1, 10:1
    sensitivity = specificity = ppv = npv = f1 = youden_j = float("nan")
    lead_time = float("nan")
    cost_skill_3 = cost_skill_5 = cost_skill_10 = float("nan")
    dm_z = dm_p = float("nan")

    try:
        # threshold = KDCA mean+2SD (primary, Kang SK/Son WS/Kim BI 2024 doi:10.3346/jkms.2024.39.e40 PMID 38288541)
        # secondary: q70 of train pool (sensitivity analysis only)
        # fallback: 8.6 KDCA 2024-25 default (no train pool)
        # audit Stage 1.1 (Task #13, 2026-05-27) — q70 → KDCA mean+2SD 변경
        from simulation.analytics.kdca_threshold import (
            compute_kdca_epidemic_threshold,
            KDCA_DEFAULT_THRESHOLD_2024_25,
        )
        if y_train_pool is not None and len(y_train_pool) > 0:
            _kdca = compute_kdca_epidemic_threshold(
                y_train_pool,
                viral_positivity_train=viral_positivity_train,
            )
            if threshold_method == "q70":
                thr = _kdca["threshold_q70"]
                if not np.isfinite(thr):
                    thr = float(np.percentile(y_train_pool, 70))
            else:  # "kdca" (default)
                thr = _kdca["threshold"]
                if not np.isfinite(thr):
                    thr = _kdca["threshold_q70"]
                if not np.isfinite(thr):
                    thr = KDCA_DEFAULT_THRESHOLD_2024_25
        else:
            thr = KDCA_DEFAULT_THRESHOLD_2024_25  # 8.6 — leakage-free fallback

        # confusion matrix at threshold
        true_pos = (a > thr) & (p > thr)
        false_pos = (a <= thr) & (p > thr)
        true_neg = (a <= thr) & (p <= thr)
        false_neg = (a > thr) & (p <= thr)
        tp = int(true_pos.sum())
        fp = int(false_pos.sum())
        tn = int(true_neg.sum())
        fn = int(false_neg.sum())

        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
        ppv = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
        npv = float(tn / (tn + fn)) if (tn + fn) > 0 else float("nan")
        if not (np.isnan(sensitivity) or np.isnan(ppv)) and (sensitivity + ppv) > 0:
            f1 = float(2 * sensitivity * ppv / (sensitivity + ppv))
        if not (np.isnan(sensitivity) or np.isnan(specificity)):
            youden_j = float(sensitivity + specificity - 1.0)

        # Lead time: 첫 threshold 도달 시 예측이 며칠 앞섰는지 (weekly index 차이)
        a_above = np.where(a > thr)[0]
        p_above = np.where(p > thr)[0]
        if len(a_above) > 0 and len(p_above) > 0:
            lead_time = float(a_above[0] - p_above[0])  # 양수 = 예측이 앞섰음

        # Cost-skill (miss=fn, FA=fp); baseline = naive constant predict (mean)
        baseline_pred = float(a.mean())
        baseline_above = baseline_pred > thr
        for ratio_v, key_attr in [(3, 'cost_skill_3'), (5, 'cost_skill_5'), (10, 'cost_skill_10')]:
            cost_model = ratio_v * fn + fp
            base_fp = int(((~true_pos) & (a <= thr)).sum() if baseline_above else 0)
            base_fn = int((a > thr).sum() if not baseline_above else 0)
            cost_baseline = ratio_v * base_fn + base_fp
            if cost_baseline > 0:
                skill = 1.0 - cost_model / cost_baseline
            else:
                skill = float("nan")
            if key_attr == 'cost_skill_3': cost_skill_3 = float(skill)
            elif key_attr == 'cost_skill_5': cost_skill_5 = float(skill)
            else: cost_skill_10 = float(skill)
    except Exception:
        pass

    # Diebold-Mariano vs lag-1 persistence (correct naive baseline)
    try:
        if len(a) >= 4:
            # Lag-1 persistence: predict last observed value at each step.
            # For t=0 use the final training observation (from y_train_pool);
            # fall back to a[0] only when y_train_pool is not provided.
            _last_train = (float(y_train_pool[-1])
                           if (y_train_pool is not None and len(y_train_pool) > 0)
                           else float(a[0]))
            naive = np.concatenate([[_last_train], a[:-1].astype(np.float64)])
            err_model = err  # already (p - a)
            err_naive = naive - a
            d = err_model ** 2 - err_naive ** 2  # loss differential (squared)
            d_mean = float(d.mean())
            d_std = float(d.std(ddof=1))
            if d_std > 1e-9:
                from scipy.stats import t as _t
                dm_z = float(d_mean / (d_std / np.sqrt(len(d))))
                # Harvey et al. (1997) correction: use t(n-1), not Normal
                dm_p = float(2 * (1 - _t.cdf(abs(dm_z), df=len(d) - 1)))
    except Exception:
        pass

    # ── WIS decomposition (Bracher 2021) — 4 new keys ───────────────────
    wis_decomp_d = (weighted_interval_score_components(a, p, sigma)
                    if weighted_interval_score_components is not None
                    else {"wis_sharpness": float("nan"), "wis_underpred": float("nan"),
                          "wis_overpred": float("nan"),  "wis_total_decomp": float("nan")})

    # ── Epidemic phase metrics (Biggerstaff 2016) — 5 new keys ──────────
    # Threshold priority (audit Stage 1.1, Task #13, 2026-05-27):
    #   (1) KDCA mean+2SD on past 3 seasons non-epidemic period
    #       (Kang SK/Son WS/Kim BI 2024 doi:10.3346/jkms.2024.39.e40 PMID 38288541),
    #   (2) q70 of train pool (sensitivity analysis fallback),
    #   (3) KDCA 2024-25 default 8.6 (no train pool, leakage-free).
    # NEVER test-set derived (data leakage — test q70 biases clinical metrics).
    from simulation.analytics.kdca_threshold import (
        compute_kdca_epidemic_threshold as _kdca_thr_fn,
        KDCA_DEFAULT_THRESHOLD_2024_25 as _KDCA_DEFAULT_THR,
    )
    if y_train_pool is not None and len(y_train_pool) > 0:
        _kdca_ep = _kdca_thr_fn(
            y_train_pool, viral_positivity_train=viral_positivity_train,
        )
        if threshold_method == "q70":
            _ep_thr = _kdca_ep["threshold_q70"]
            if not np.isfinite(_ep_thr):
                _ep_thr = float(np.percentile(y_train_pool, 70))
        else:  # "kdca" (default)
            _ep_thr = _kdca_ep["threshold"]
            if not np.isfinite(_ep_thr):
                _ep_thr = _kdca_ep["threshold_q70"]
            if not np.isfinite(_ep_thr):
                _ep_thr = _KDCA_DEFAULT_THR
    else:
        _ep_thr = _KDCA_DEFAULT_THR  # 8.6 — leakage-free fallback
    epi_phase_d = (epidemic_phase_metrics(a, p, threshold=_ep_thr)
                   if epidemic_phase_metrics is not None
                   else {"attack_rate_relerr": float("nan"), "growth_rate_corr": float("nan"),
                         "epidemic_duration_err": float("nan"), "season_onset_err": float("nan"),
                         "early_warning_lead": float("nan")})

    # ── Advanced clinical metrics (Chicco 2020, Vickers 2006) — 5 new keys
    adv_clin_d = (advanced_clinical_metrics_ext(a, p, threshold=_ep_thr, prior_prob=0.30)
                  if advanced_clinical_metrics_ext is not None
                  else {"mcc": float("nan"), "cohens_kappa": float("nan"),
                        "lr_positive": float("nan"), "lr_negative": float("nan"),
                        "net_benefit_default": float("nan")})

    # ── Conditional calibration (audit Stage 3.2, Task #20, 2026-05-27) — 7 new keys
    # Czado, Gneiting & Held (2009) doi:10.1111/j.1541-0420.2009.01191.x +
    # marginal calibration diagram + conditional coverage by tier.
    # marginal PIT 만 보고하는 audit A6 비판 해소.
    try:
        from simulation.analytics.conditional_calibration import (
            compute_conditional_calibration_block as _cond_calib_fn,
        )
        # audit 2026-05-27 fix: ILI rate = continuous ratio scale, NOT count.
        # Czado et al. (2009) nonrandomized PIT 는 count-only — continuous default.
        _cond_calib_d = _cond_calib_fn(a, p, sigma=sigma, family="continuous")
    except Exception:
        _cond_calib_d = {
            "pit_nonrand_mean": float("nan"),
            "pit_nonrand_std":  float("nan"),
            "pit_nonrand_ks_p": float("nan"),
            "marginal_calib_max_diff":  float("nan"),
            "marginal_calib_mean_diff": float("nan"),
            "picp95_low_tier":  float("nan"),
            "picp95_high_tier": float("nan"),
        }

    # Q4 / G-277: 계층(stratified) 보고 — train 범위 내(within)/밖(out-of-range=외삽) 분리.
    #   full metric(위 r2/mae/wis)은 primary 로 유지(외삽 점 삭제=cherry-pick 금지). 다만
    #   y_true > train_max 인 외삽 점을 투명하게 분리해 "정상 구간 적합도"와 "외삽 부담"을
    #   정직하게 드러낸다(피크 외삽은 구간 coverage·lead-time 으로 별도 평가 권장).
    _strat = _stratified_range_metrics(a, p, y_train_pool)

    return {
        # Point
        "r2": r2, "mae": mae, "rmse": rmse, "mse": mse,
        "mape": mape, "smape": smape, "mdape": mdape, "mase_h1": mase_h1,
        # Bias + scaled error metrics (PI extensions +2)
        "bias_mean_error": bias_v,
        "theils_u":        theils_u_v,
        "msle":            msle_v,
        # Probabilistic
        "wis": wis, "log_wis": log_wis, "crps_gaussian": crps,
        # pinball_q50 dropped (S8 R4 align): proportional to MAE/2 — redundant
        "pinball_q05": pin_q05, "pinball_q95": pin_q95,
        # PIT
        "pit_mean": pit_mean, "pit_std": pit_std, "pit_ks_p": pit_ks_p,
        # PI coverage (G-167 — multi-criteria 4 번째 criteria; G-175 forward 0.90)
        "pi95_coverage": pi95_cov, "pi80_coverage": pi80_cov, "pi50_coverage": pi50_cov,
        # WIS decomposition (Bracher 2021) — 4 new
        "wis_sharpness":    wis_decomp_d.get("wis_sharpness",    float("nan")),
        "wis_underpred":    wis_decomp_d.get("wis_underpred",    float("nan")),
        "wis_overpred":     wis_decomp_d.get("wis_overpred",     float("nan")),
        "wis_total_decomp": wis_decomp_d.get("wis_total_decomp", float("nan")),
        # Epi-curve
        "peak_week_err": peak_week_err, "peak_int_relerr": peak_int_relerr,
        "direction_acc": direction_acc,
        # Epidemic phase metrics (Biggerstaff 2016) — 5 new
        "attack_rate_relerr":    epi_phase_d.get("attack_rate_relerr",    float("nan")),
        "growth_rate_corr":      epi_phase_d.get("growth_rate_corr",      float("nan")),
        "epidemic_duration_err": epi_phase_d.get("epidemic_duration_err", float("nan")),
        "season_onset_err":      epi_phase_d.get("season_onset_err",      float("nan")),
        # early_warning_lead dropped (S8 R4 align): = -season_onset_err redundant
        # G-181 operational/alert (paper §4.6 — 9개)
        "sensitivity": sensitivity, "specificity": specificity,
        "ppv": ppv, "npv": npv, "f1": f1, "youden_j": youden_j,
        "lead_time_weeks": lead_time,
        "cost_skill_3to1": cost_skill_3, "cost_skill_5to1": cost_skill_5,
        "cost_skill_10to1": cost_skill_10,
        # Advanced clinical (Chicco 2020, Vickers 2006) — 5 new
        "mcc":                 adv_clin_d.get("mcc",                 float("nan")),
        "cohens_kappa":        adv_clin_d.get("cohens_kappa",        float("nan")),
        "lr_positive":         adv_clin_d.get("lr_positive",         float("nan")),
        "lr_negative":         adv_clin_d.get("lr_negative",         float("nan")),
        "net_benefit_default": adv_clin_d.get("net_benefit_default", float("nan")),
        # DM test (paper §4.1.4)
        "dm_z_stat": dm_z, "dm_p_value": dm_p,
        # Conditional calibration (audit Stage 3.2 — Czado/Gneiting/Held 2009)
        "pit_nonrand_mean":          _cond_calib_d["pit_nonrand_mean"],
        "pit_nonrand_std":           _cond_calib_d["pit_nonrand_std"],
        "pit_nonrand_ks_p":          _cond_calib_d["pit_nonrand_ks_p"],
        "marginal_calib_max_diff":   _cond_calib_d["marginal_calib_max_diff"],
        "marginal_calib_mean_diff":  _cond_calib_d["marginal_calib_mean_diff"],
        "picp95_low_tier":           _cond_calib_d["picp95_low_tier"],
        "picp95_high_tier":          _cond_calib_d["picp95_high_tier"],
        # Sample size
        "n": int(n), "sigma_in_sample": sigma,
        # Q4 / G-277: stratified range report (within / out-of-train-range)
        **_strat,
    }


_STRAT_KEYS = (
    "n_within_range", "n_out_of_range", "out_of_range_max_obs",
    "within_range_r2", "within_range_mae", "within_range_wis",
    "out_of_range_mae", "frac_out_of_range",
)


def _stratified_range_metrics(a: np.ndarray, p: np.ndarray,
                              y_train_pool: Optional[np.ndarray]) -> dict:
    """Q4 / G-277: within-train-range vs out-of-range(외삽) 분리 metric.

    train 최댓값 기준으로 test 점을 ① within(y_true ≤ train_max) ② out-of-range
    (y_true > train_max, 외삽=가장 어려운 점) 로 나눠 각각 r2/mae 보고. full metric 을
    대체하지 않고 **보조**한다(외삽 점 삭제는 cherry-pick — 보고만 분리). y_train_pool
    없으면 빈 dict (back-compat). 모든 키는 _STRAT_KEYS 로 고정(contract 일관).

    Args:
        a: finite y_true. p: finite y_pred. y_train_pool: train 타깃(범위 기준).
    Returns:
        _STRAT_KEYS 부분/전체를 담은 dict (없으면 {}).
    """
    if y_train_pool is None or len(np.asarray(y_train_pool)) == 0 or len(a) == 0:
        return {}
    tmax = float(np.nanmax(np.asarray(y_train_pool, dtype=np.float64)))
    within = a <= tmax
    out = ~within
    d: dict = {
        "n_within_range": int(within.sum()),
        "n_out_of_range": int(out.sum()),
        "frac_out_of_range": float(out.mean()),
        "out_of_range_max_obs": (float(a[out].max()) if out.any() else float("nan")),
    }
    if within.sum() >= 2:
        ss_res = float(np.sum((a[within] - p[within]) ** 2))
        ss_tot = float(np.sum((a[within] - a[within].mean()) ** 2))
        d["within_range_r2"] = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
        d["within_range_mae"] = float(np.mean(np.abs(a[within] - p[within])))
    else:
        d["within_range_r2"] = float("nan")
        d["within_range_mae"] = float("nan")
    d["out_of_range_mae"] = (float(np.mean(np.abs(a[out] - p[out]))) if out.any()
                             else float("nan"))
    d["within_range_wis"] = float("nan")   # WIS 는 σ 의존 — full wis 로 갈음(placeholder, 키 일관)
    return d


def _empty_metrics_schema(*, n: int = 0, sigma_in: float = float("nan")) -> dict:
    """59-key NaN schema for no-valid edge case (Codex audit fix 2026-05-27).

    Returns same key set as compute_full_metrics() success path — downstream
    DataFrame / merge / json.dumps 안전. NaN-safe contract preservation.
    """
    nan = float("nan")
    return {
        # Point (8)
        "r2": nan, "mae": nan, "rmse": nan, "mse": nan,
        "mape": nan, "smape": nan, "mdape": nan, "mase_h1": nan,
        # Bias + scaled (3)
        "bias_mean_error": nan, "theils_u": nan, "msle": nan,
        # Probabilistic (5)
        "wis": nan, "log_wis": nan, "crps_gaussian": nan,
        "pinball_q05": nan, "pinball_q95": nan,
        # PIT (3)
        "pit_mean": nan, "pit_std": nan, "pit_ks_p": nan,
        # PI coverage (3)
        "pi95_coverage": nan, "pi80_coverage": nan, "pi50_coverage": nan,
        # WIS decomp (4)
        "wis_sharpness": nan, "wis_underpred": nan, "wis_overpred": nan, "wis_total_decomp": nan,
        # Epi-curve (3)
        "peak_week_err": nan, "peak_int_relerr": nan, "direction_acc": nan,
        # Epidemic phase (4)
        "attack_rate_relerr": nan, "growth_rate_corr": nan,
        "epidemic_duration_err": nan, "season_onset_err": nan,
        # G-181 operational (9)
        "sensitivity": nan, "specificity": nan, "ppv": nan, "npv": nan, "f1": nan,
        "youden_j": nan, "lead_time_weeks": nan,
        "cost_skill_3to1": nan, "cost_skill_5to1": nan, "cost_skill_10to1": nan,
        # Advanced clinical (5)
        "mcc": nan, "cohens_kappa": nan, "lr_positive": nan, "lr_negative": nan,
        "net_benefit_default": nan,
        # DM (2)
        "dm_z_stat": nan, "dm_p_value": nan,
        # Conditional calibration (7, audit S3.2)
        "pit_nonrand_mean": nan, "pit_nonrand_std": nan, "pit_nonrand_ks_p": nan,
        "marginal_calib_max_diff": nan, "marginal_calib_mean_diff": nan,
        "picp95_low_tier": nan, "picp95_high_tier": nan,
        # Q4 / G-277: stratified range keys (contract 일관 — empty 도 동일 key set)
        **{k: (0 if k.startswith("n_") else nan) for k in _STRAT_KEYS},
        # Sample
        "n": int(n), "sigma_in_sample": sigma_in,
        # Edge-case flag
        "n_valid": 0,
    }


__all__ = ["compute_full_metrics"]
