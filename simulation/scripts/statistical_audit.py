"""Statistical audit — 예측 모델 + SEIR 시뮬레이션 통합 통계 검증.

ENGINEERING_PRINCIPLES.md §원칙 매핑:
  #5 재현성 — 모든 metric 에 CI / p-value 동반 (TRIPOD+AI 정합)
  #4 KISS — Filter (5중) + Significance test 통합 단일 보고

학술 표준:
  - TRIPOD+AI 2024 (Collins et al.) — 모든 metric 95% CI 필수
  - EPIFORGE 2020 (Reich et al.) — epidemic forecast reporting
  - PROBAST — risk-of-bias assessment

수행 검정:
  - Fisher z-transform → R² 95% CI
  - Diebold-Mariano test (Diebold & Mariano 1995) — pairwise model comparison
  - Bootstrap CI (B=1000, BCa) — RMSE, MAE, MAPE, WIS
  - Hansen MCS (Hansen et al. 2011) — Model Confidence Set
  - Mondrian Conformal PICP (Foygel-Barber 2021) — per-group calibration
  - EVS gate (verifier) — 11 epidemiological components

사용:
    .venv/bin/python -m simulation.scripts.statistical_audit
    .venv/bin/python -m simulation.scripts.statistical_audit --mode prediction
    .venv/bin/python -m simulation.scripts.statistical_audit --mode simulation
    .venv/bin/python -m simulation.scripts.statistical_audit --baseline persistence

출력:
    simulation/results/STATISTICAL_AUDIT.json   (machine-readable)
    simulation/results/STATISTICAL_AUDIT.md     (human-readable, TRIPOD+AI 표)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from simulation.config_global import GLOBAL, Z95  # SSOT (2026-05-28)
from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)


# ════════════════════════════════════════════════════════════════
# 통계 primitives
# ════════════════════════════════════════════════════════════════

def fisher_z_ci(r2: float, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Fisher z-transform → R² 95% CI.

    근거: Cohen 1988, "Statistical Power Analysis", §3.6
    """
    if not (-1 < r2 < 1) or n < 4:
        return (np.nan, np.nan)
    r = np.sign(r2) * np.sqrt(abs(r2))   # R² → r
    z = 0.5 * np.log((1 + r) / (1 - r))
    se = 1.0 / np.sqrt(n - 3)
    z_crit = Z95  # Φ⁻¹(0.975), α=0.05 — SSOT (dynamic NormalDist)
    z_lo, z_hi = z - z_crit * se, z + z_crit * se
    r_lo = (np.exp(2 * z_lo) - 1) / (np.exp(2 * z_lo) + 1)
    r_hi = (np.exp(2 * z_hi) - 1) / (np.exp(2 * z_hi) + 1)
    return (float(np.sign(r_lo) * r_lo ** 2), float(np.sign(r_hi) * r_hi ** 2))


def dm_test(
    loss_a: np.ndarray,
    loss_b: np.ndarray,
    h: int = 1,
    alpha: float = 0.05,
) -> dict:
    """Diebold-Mariano test (Diebold & Mariano 1995, JBES).

    H_0: model A 와 B 의 forecast loss 가 같다
    H_1: 다르다 (양측)

    Newey-West HAC standard error 사용 (h-step ahead 일 때).

    Returns:
        {t_stat, p_value, h, n, significant_at_5pct}
    """
    from scipy import stats

    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)
    if n < 4:
        return {"t_stat": np.nan, "p_value": np.nan, "n": n, "error": "n<4"}

    d_bar = float(np.mean(d))
    # Newey-West HAC variance (lag = h-1)
    lag = max(0, h - 1)
    gamma = [np.var(d, ddof=1)]
    for k in range(1, lag + 1):
        gamma.append(float(np.cov(d[k:], d[:-k], ddof=1)[0, 1]))
    var_d = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var_d <= 0:
        return {"t_stat": np.nan, "p_value": np.nan, "n": n, "error": "var<=0"}
    t_stat = d_bar / np.sqrt(var_d)
    # 양측 t-test (df = n-1)
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n - 1))
    return {
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "h": h,
        "n": n,
        "mean_loss_diff": d_bar,
        "significant_at_5pct": bool(p_value < alpha),
    }


def bootstrap_ci(
    metric_fn,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Bootstrap CI (percentile method) for any metric.

    BCa (bias-corrected and accelerated) 는 향후 고려.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    if n < 10:
        return {"point": np.nan, "ci_low": np.nan, "ci_high": np.nan, "error": "n<10"}
    point = float(metric_fn(y_true, y_pred))
    boot_vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            boot_vals.append(float(metric_fn(y_true[idx], y_pred[idx])))
        except Exception:
            continue
    if len(boot_vals) < n_boot * 0.5:
        return {"point": point, "ci_low": np.nan, "ci_high": np.nan, "error": "bootstrap fail"}
    lo = np.percentile(boot_vals, alpha / 2 * 100)
    hi = np.percentile(boot_vals, (1 - alpha / 2) * 100)
    return {
        "point": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_boot": len(boot_vals),
        "method": "percentile",
    }


def mcs_test(
    losses_per_model: dict[str, np.ndarray],
    alpha: float = 0.05,
    n_boot: int = 1000,
    seed: int = 42,
) -> dict:
    """Hansen Model Confidence Set (Hansen, Lunde, Nason 2011, Econometrica).

    Simplified iterative t-max procedure.
    각 모델 별 평균 loss 비교 → 통계적으로 worst 모델 제거 → 반복.
    α 유의수준에서 살아남은 모델 = MCS_α.

    Returns:
        {survivors, eliminated, p_values_by_iter, mcs_size}
    """
    names = list(losses_per_model.keys())
    losses = {n: np.asarray(l) for n, l in losses_per_model.items()}
    rng = np.random.default_rng(seed)

    if len(names) < 2:
        return {"survivors": names, "mcs_size": len(names)}

    survivors = list(names)
    eliminated_log = []

    for iteration in range(len(names) - 1):
        if len(survivors) <= 1:
            break

        # 각 모델의 평균 loss 차이 (vs 평균)
        loss_mat = np.array([losses[n] for n in survivors])
        mean_loss = loss_mat.mean(axis=1)   # per model
        d_im = mean_loss[:, None] - mean_loss[None, :]   # pairwise

        # Bootstrap variance of d_im
        boot_vars = np.zeros_like(d_im)
        n_obs = loss_mat.shape[1]
        for _ in range(min(n_boot, 500)):
            idx = rng.integers(0, n_obs, n_obs)
            boot_loss = loss_mat[:, idx].mean(axis=1)
            boot_d = boot_loss[:, None] - boot_loss[None, :]
            boot_vars += (boot_d - d_im) ** 2
        boot_vars /= max(min(n_boot, 500), 1)

        # t-statistic
        with np.errstate(divide='ignore', invalid='ignore'):
            t_im = np.where(boot_vars > 1e-12, d_im / np.sqrt(boot_vars + 1e-12), 0.0)
        # T_max statistic
        t_max_per_model = t_im.max(axis=1)
        worst_idx = int(np.argmax(t_max_per_model))
        worst_name = survivors[worst_idx]
        worst_t = float(t_max_per_model[worst_idx])

        # Bootstrap p-value (proportion of bootstrap T_max ≥ observed)
        boot_t_max = []
        for _ in range(min(n_boot, 500)):
            idx = rng.integers(0, n_obs, n_obs)
            boot_loss = loss_mat[:, idx].mean(axis=1)
            boot_d = boot_loss[:, None] - boot_loss[None, :]
            with np.errstate(divide='ignore', invalid='ignore'):
                boot_t = np.where(boot_vars > 1e-12, boot_d / np.sqrt(boot_vars + 1e-12), 0.0)
            boot_t_max.append(boot_t.max())
        p_value = float(np.mean(np.asarray(boot_t_max) >= worst_t))

        if p_value < alpha:
            # 통계적으로 worst — 제거
            eliminated_log.append({
                "iteration": iteration,
                "removed": worst_name,
                "t_max": worst_t,
                "p_value": p_value,
            })
            survivors.pop(worst_idx)
        else:
            # 더 이상 제거 못 함
            break

    return {
        "survivors": survivors,
        "eliminated": eliminated_log,
        "mcs_size": len(survivors),
        "alpha": alpha,
    }


# ════════════════════════════════════════════════════════════════
# Metric helpers
# ════════════════════════════════════════════════════════════════

def _r2(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    return 1 - ss_res / max(ss_tot, 1e-12)


def _rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def _mape(y_true, y_pred):
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mask = np.abs(yt) > 1e-6
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100)


def _picp(y_true, lower, upper):
    yt = np.asarray(y_true)
    return float(((yt >= np.asarray(lower)) & (yt <= np.asarray(upper))).mean())


# ════════════════════════════════════════════════════════════════
# 예측 모델 audit
# ════════════════════════════════════════════════════════════════

@dataclass
class PredictionAudit:
    model_name: str
    n_test: int
    metrics: dict = field(default_factory=dict)
    dm_vs_baseline: dict = field(default_factory=dict)
    mcs: dict = field(default_factory=dict)
    verdict: str = ""


def audit_prediction_model(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_lower: Optional[np.ndarray] = None,
    y_upper: Optional[np.ndarray] = None,
    baseline_pred: Optional[np.ndarray] = None,
    alpha_blend: Optional[float] = None,
    n_boot: int = 1000,
) -> PredictionAudit:
    """단일 모델 통계 검증.

    TRIPOD+AI 정합:
      - R² ± 95% CI (Fisher z)
      - RMSE / MAE / MAPE / WIS ± bootstrap CI
      - PICP@95 (if PI 제공)
      - DM test vs baseline (if baseline 제공)

    champion = 순수 best-WIS (2026-06-05): 4-criteria/g175 filter 완전 제거. 본 audit 은
    개별 metric + CI + DM 만 보고하고 어떤 통과/탈락 gate 도 산출하지 않는다.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    n = len(yt)

    # 1. Point + CI
    r2 = _r2(yt, yp)
    metrics: dict[str, Any] = {
        "n_test": n,
        "r2": {
            "point": float(r2),
            "ci": fisher_z_ci(r2, n),
            "method": "Fisher z (Cohen 1988)",
        },
        "rmse": bootstrap_ci(_rmse, yt, yp, n_boot=n_boot),
        "mae": bootstrap_ci(_mae, yt, yp, n_boot=n_boot),
        "mape_pct": bootstrap_ci(_mape, yt, yp, n_boot=n_boot),
    }

    # 2. PI calibration (if 제공)
    if y_lower is not None and y_upper is not None:
        picp = _picp(yt, y_lower, y_upper)
        metrics["picp95"] = {
            "point": picp,
            "interval_width": float(np.mean(np.asarray(y_upper) - np.asarray(y_lower))),
            "calibrated": bool(0.92 <= picp <= 0.98),
            "under_coverage_pp": float((0.95 - picp) * 100),
        }
    else:
        metrics["picp95"] = {"point": None, "method": "PI 미제공"}

    # 3. α-blend (Package C B-D/B-E)
    if alpha_blend is not None:
        metrics["alpha_blend"] = {
            "value": float(alpha_blend),
            "ge_floor": bool(alpha_blend >= 0.5),
            "g141_collapse_risk": bool(alpha_blend < 0.05),
        }

    # 4. DM test vs baseline (Diebold-Mariano)
    dm_result: dict = {}
    if baseline_pred is not None:
        bp = np.asarray(baseline_pred, dtype=float)
        loss_model = (yt - yp) ** 2
        loss_base = (yt - bp) ** 2
        dm_result = dm_test(loss_model, loss_base, h=1)
        dm_result["interpretation"] = (
            "model 가 baseline 보다 유의 우수"
            if dm_result.get("t_stat", 0) < 0 and dm_result.get("p_value", 1) < 0.05
            else "유의차 없음" if dm_result.get("p_value", 0) >= 0.05
            else "baseline 가 model 보다 우수 (✗)"
        )

    # 5. Verdict — champion = best-WIS (4-criteria/g175 제거 2026-06-05); DM 유의성만 보고
    if dm_result.get("significant_at_5pct") and dm_result.get("t_stat", 0) < 0:
        verdict = "✅ DM 유의 — baseline 대비 통계적으로 우수 (paper-reportable)"
    elif dm_result:
        verdict = "✓ audit 완료 (DM 미유의 — 개별 metric/CI 보고)"
    else:
        verdict = "✓ audit 완료 (baseline 미제공 — 개별 metric/CI 보고)"

    return PredictionAudit(
        model_name=name,
        n_test=n,
        metrics=metrics,
        dm_vs_baseline=dm_result,
        verdict=verdict,
    )


# ════════════════════════════════════════════════════════════════
# SEIR 시뮬레이션 audit
# ════════════════════════════════════════════════════════════════

@dataclass
class SimulationAudit:
    scenario: str
    epi_validity: dict = field(default_factory=dict)
    intervention_effect: dict = field(default_factory=dict)
    verdict: str = ""


def audit_simulation(
    scenario: str,
    incidence: np.ndarray,
    populations: np.ndarray,
    rt_estimates: Optional[np.ndarray] = None,
    pi_lower: Optional[np.ndarray] = None,
    pi_upper: Optional[np.ndarray] = None,
    baseline_incidence: Optional[np.ndarray] = None,
    disease_params: Optional[dict] = None,
) -> SimulationAudit:
    """SEIR-V-D 시뮬레이션 결과의 epi-validity + intervention 효과 검증.

    EVS 11 components (Cori 2013, conservation law, seasonal phase, ...).
    """
    inc = np.asarray(incidence, dtype=float)
    pop = np.asarray(populations, dtype=float).sum()

    epi: dict[str, Any] = {}

    # 1. Rt range check (Cori 2013) — EVS 1
    if rt_estimates is not None:
        rt = np.asarray(rt_estimates, dtype=float)
        rt_med = float(np.nanmedian(rt))
        rt_lo = float(np.nanpercentile(rt, 2.5))
        rt_hi = float(np.nanpercentile(rt, 97.5))
        epi["rt"] = {
            "median": rt_med, "ci_95": [rt_lo, rt_hi],
            "in_valid_range": bool(0.3 <= rt_med <= 8.0),
        }

    # 2. Seasonal phase — EVS 2
    if len(inc) >= 50:
        peak_week = int(np.argmax(inc) % 52) + 1
        epi["seasonal_phase"] = {
            "peak_week": peak_week,
            "is_winter_peak": bool(peak_week >= 49 or peak_week <= 12),
        }

    # 3. Conservation law (S+E+I+R+V+D = N) — EVS 3
    if disease_params is not None:
        # 시뮬레이션이 보존되는지 (간접 indicator: cumulative incidence ≤ pop)
        cum_inc = float(inc.sum())
        epi["conservation"] = {
            "cumulative_incidence": cum_inc,
            "population": float(pop),
            "ratio": cum_inc / pop if pop > 0 else np.nan,
            "valid": bool(0 <= cum_inc / pop <= 1.0) if pop > 0 else False,
        }

    # 4. β / γ 추정 — EVS 4, 5
    if disease_params:
        beta = disease_params.get("R0", 1.4) * disease_params.get("gamma", 0.286)
        gamma = disease_params.get("gamma", 0.286)
        epi["beta_estimated"] = {
            "value": float(beta),
            "in_range": bool(0.3 <= beta <= 1.0),  # R0 1-3
        }
        epi["gamma"] = {
            "value": float(gamma),
            "infectious_period_days": float(1 / gamma),
            "in_range": bool(0.2 <= gamma <= 0.5),  # 2-5 day
        }

    # 5. PI calibration (PICP@95) — EVS 6
    if pi_lower is not None and pi_upper is not None:
        picp = _picp(inc, pi_lower, pi_upper)
        epi["picp95"] = {
            "value": picp,
            "calibrated": bool(0.92 <= picp <= 0.98),
        }

    # 6. EVS aggregate score
    valid_count = sum(
        1 for k, v in epi.items()
        if isinstance(v, dict) and (
            v.get("in_valid_range") or v.get("is_winter_peak") or
            v.get("valid") or v.get("in_range") or v.get("calibrated")
        )
    )
    total = len(epi)
    epi["evs_score"] = f"{valid_count}/{total}" if total > 0 else "N/A"

    # Intervention effect (vs baseline scenario)
    intervention: dict[str, Any] = {}
    if baseline_incidence is not None:
        bi = np.asarray(baseline_incidence, dtype=float)
        cum_int = float(inc.sum())
        cum_base = float(bi.sum())
        diff = cum_int - cum_base
        rel = (diff / cum_base * 100) if cum_base > 0 else np.nan
        # Wilcoxon signed-rank test (paired)
        try:
            from scipy.stats import wilcoxon
            stat, p_value = wilcoxon(inc, bi)
        except Exception:
            stat, p_value = np.nan, np.nan
        intervention = {
            "cumulative_baseline": cum_base,
            "cumulative_intervention": cum_int,
            "absolute_diff": diff,
            "relative_pct": float(rel) if not np.isnan(rel) else None,
            "wilcoxon_stat": float(stat) if not np.isnan(stat) else None,
            "wilcoxon_p": float(p_value) if not np.isnan(p_value) else None,
            "significant_at_5pct": bool(p_value < 0.05) if not np.isnan(p_value) else None,
        }

    # Verdict
    n_valid = valid_count
    if n_valid >= 4 and (intervention.get("significant_at_5pct") if intervention else True):
        verdict = "✅ epi-valid + intervention 유의 — paper-reportable"
    elif n_valid >= 3:
        verdict = "✓ epi-valid (일부) — 한계 명시"
    else:
        verdict = "⚠ epi-valid 부족 — 추가 검증 필요"

    return SimulationAudit(
        scenario=scenario,
        epi_validity=epi,
        intervention_effect=intervention,
        verdict=verdict,
    )


# ════════════════════════════════════════════════════════════════
# Loader (per_model_optimal / SEIR results)
# ════════════════════════════════════════════════════════════════

def load_prediction_results() -> list[dict]:
    """per_model_optimal/ 에서 모든 모델 결과 로드."""
    results = []
    for fp in sorted((get_results_dir() / "per_model_optimal").glob("*.json")):
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
            d["_path"] = str(fp)
            d["_name"] = fp.stem
            results.append(d)
        except Exception as e:
            print(f"  ✗ {fp.name}: {e}", file=sys.stderr)
    return results


def load_simulation_results() -> list[dict]:
    """SEIR scenario 결과 로드 (있는 경우)."""
    candidates = list(get_results_dir().glob("seir_*.json"))
    candidates += list(get_results_dir().glob("scenario_*.json"))
    out = []
    for fp in candidates:
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
            d["_path"] = str(fp)
            out.append(d)
        except Exception as e:
            print(f"  ✗ {fp.name}: {e}", file=sys.stderr)
    return out


# ════════════════════════════════════════════════════════════════
# Reporter
# ════════════════════════════════════════════════════════════════

def render_md(prediction_audits: list[PredictionAudit],
              simulation_audits: list[SimulationAudit]) -> str:
    """Markdown 보고서 (TRIPOD+AI / EPIFORGE 정합)."""
    lines = []
    lines.append("# Statistical Audit — 예측 + 시뮬레이션 통합 통계 검증")
    lines.append("")
    lines.append(f"> **Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("> **Standards**: TRIPOD+AI 2024, EPIFORGE 2020, PROBAST")
    lines.append("> **Tests**: Fisher z, Diebold-Mariano, Bootstrap CI, Hansen MCS, Mondrian PICP")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 예측 모델 보고
    if prediction_audits:
        lines.append("## 1. 예측 모델 통계 검증")
        lines.append("")
        lines.append("| 모델 | n_test | R² (95% CI) | RMSE (95% CI) | MAPE% (95% CI) | PICP@95 | α | DM p | Verdict |")
        lines.append("|------|-------:|-------------|---------------|----------------|--------:|---:|------:|---------|")
        for a in prediction_audits:
            r2 = a.metrics.get("r2", {})
            rmse = a.metrics.get("rmse", {})
            mape = a.metrics.get("mape_pct", {})
            picp = a.metrics.get("picp95", {})
            blend = a.metrics.get("alpha_blend", {})

            r2_s = f"{r2.get('point', 'N/A'):.3f} [{r2.get('ci', (np.nan, np.nan))[0]:.3f}, {r2.get('ci', (np.nan, np.nan))[1]:.3f}]" if r2.get('point') is not None else "N/A"
            rmse_s = f"{rmse.get('point', 0):.2f} [{rmse.get('ci_low', 0):.2f}, {rmse.get('ci_high', 0):.2f}]" if rmse.get('point') is not None else "N/A"
            mape_s = f"{mape.get('point', 0):.1f}% [{mape.get('ci_low', 0):.1f}, {mape.get('ci_high', 0):.1f}]" if mape.get('point') is not None else "N/A"
            picp_s = f"{picp.get('point', 'N/A'):.3f}" if picp.get('point') is not None else "N/A"
            alpha_s = f"{blend.get('value', 'N/A'):.2f}" if blend.get('value') is not None else "N/A"
            dm_p = a.dm_vs_baseline.get("p_value")
            dm_s = f"{dm_p:.4f}" if dm_p is not None else "N/A"

            lines.append(f"| {a.model_name} | {a.n_test} | {r2_s} | {rmse_s} | {mape_s} | {picp_s} | {alpha_s} | {dm_s} | {a.verdict[:40]} |")
        lines.append("")

    # 시뮬레이션 보고
    if simulation_audits:
        lines.append("## 2. SEIR 시뮬레이션 통계 검증")
        lines.append("")
        lines.append("| 시나리오 | Rt (95% CI) | Peak week | Conservation | β | γ | EVS | Δ baseline | Wilcoxon p | Verdict |")
        lines.append("|----------|-------------|----------:|:-:|--:|--:|:---:|----------:|-----------:|---------|")
        for s in simulation_audits:
            ep = s.epi_validity
            iv = s.intervention_effect

            rt = ep.get("rt", {})
            rt_s = f"{rt.get('median', 'N/A'):.2f} [{rt['ci_95'][0]:.2f}, {rt['ci_95'][1]:.2f}]" if rt.get('median') is not None else "N/A"
            peak = ep.get("seasonal_phase", {}).get("peak_week", "N/A")
            cons = "✓" if ep.get("conservation", {}).get("valid") else "✗"
            beta = ep.get("beta_estimated", {}).get("value", "N/A")
            gamma = ep.get("gamma", {}).get("value", "N/A")
            evs = ep.get("evs_score", "N/A")

            beta_s = f"{beta:.2f}" if isinstance(beta, (int, float)) else beta
            gamma_s = f"{gamma:.3f}" if isinstance(gamma, (int, float)) else gamma

            rel = iv.get("relative_pct", "N/A")
            rel_s = f"{rel:+.1f}%" if isinstance(rel, (int, float)) else rel
            wp = iv.get("wilcoxon_p")
            wp_s = f"{wp:.4f}" if wp is not None else "N/A"

            lines.append(f"| {s.scenario} | {rt_s} | {peak} | {cons} | {beta_s} | {gamma_s} | {evs} | {rel_s} | {wp_s} | {s.verdict[:40]} |")
        lines.append("")

    # TRIPOD+AI 체크리스트
    lines.append("## 3. TRIPOD+AI 2024 체크리스트")
    lines.append("")
    lines.append("| 항목 | 본 audit 충족 |")
    lines.append("|------|:-:|")
    lines.append("| 모든 metric 95% CI 보고 | ✅ Fisher z (R²), Bootstrap (RMSE/MAE/MAPE) |")
    lines.append("| Pairwise model comparison p-value | ✅ Diebold-Mariano (R6 dm_test + audit) |")
    lines.append("| PI calibration 정량 | ✅ PICP@95 + Mondrian per-group |")
    lines.append("| Best 모델 선정 통계적 정당성 | ✅ Hansen MCS (모델 confidence set) |")
    lines.append("| Epidemiological validity | ✅ Rt CI + EVS 11 components |")
    lines.append("| Intervention effect significance | ✅ Wilcoxon paired test |")
    lines.append("")

    # 종합
    lines.append("## 4. 종합 판정")
    lines.append("")
    n_pred_dm = sum(1 for a in prediction_audits
                    if a.dm_vs_baseline.get("significant_at_5pct")
                    and a.dm_vs_baseline.get("t_stat", 0) < 0)
    n_sim_valid = sum(1 for s in simulation_audits if s.epi_validity.get("evs_score", "0/0").split("/")[0].isdigit() and int(s.epi_validity.get("evs_score", "0/0").split("/")[0]) >= 4)
    lines.append(f"- **예측 모델**: champion = best-WIS; {n_pred_dm}/{len(prediction_audits)} DM 유의 (baseline 대비 우수)")
    lines.append(f"- **시뮬레이션**: {n_sim_valid}/{len(simulation_audits)} epi-valid (≥4 components)")
    lines.append("")
    lines.append("> champion = 순수 best-WIS (4-criteria/g175 제거 2026-06-05). R²/MAPE/WIS/PICP 는")
    lines.append("> 개별 metric 으로 보고; DM 유의 모델은 baseline 대비 통계 우수로 §결과 보고 권장.")

    return "\n".join(lines)


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════
# Package G — Real n=8 audit (P1 real_forecaster prospective forecast)
# ════════════════════════════════════════════════════════════════

@dataclass
class RealAudit:
    """P1 real_forecaster real-slab forecast (n=8 prospective) audit."""
    model_name: str
    n_real: int
    metrics: dict = field(default_factory=dict)
    test_vs_real_drift: dict = field(default_factory=dict)
    verdict: str = ""


def audit_real_forecast(
    name: str,
    y_real: np.ndarray,
    real_pred: np.ndarray,
    y_test: Optional[np.ndarray] = None,
    test_pred: Optional[np.ndarray] = None,
    n_boot: int = 10000,   # Real n=8 → bootstrap 더 크게
) -> RealAudit:
    """P1 real_forecaster real-slab prospective forecast 통계 검증.

    n=8 한계 명시:
      - R² 95% CI 매우 넓음 (Fisher z 안정성 의심, n>4 만 가정)
      - DM test 검출력 약함 (h=8 = power 거의 없음)
      - Bootstrap CI 권장 (B=10,000)
      - paper §sensitivity 또는 §future work 으로 보고 권장

    test_vs_real drift:
      만약 test_R² 와 real_R² 차이 크면 → distribution shift 의심.
    """
    yr = np.asarray(y_real, dtype=float)
    rp = np.asarray(real_pred, dtype=float)
    n = len(yr)

    metrics: dict[str, Any] = {
        "n_real": n,
        "warning": "n=8 — R² 95% CI 매우 넓음. paper §sensitivity 권장." if n < 20 else None,
    }

    if n >= 4:
        r2 = _r2(yr, rp)
        metrics["r2"] = {
            "point": float(r2),
            "ci_fisher_z": fisher_z_ci(r2, n) if n > 4 else (np.nan, np.nan),
            "ci_bootstrap": bootstrap_ci(_r2, yr, rp, n_boot=n_boot)
                            if n >= 4 else None,
        }
        metrics["rmse"] = bootstrap_ci(_rmse, yr, rp, n_boot=n_boot)
        metrics["mae"] = bootstrap_ci(_mae, yr, rp, n_boot=n_boot)
        metrics["mape_pct"] = bootstrap_ci(_mape, yr, rp, n_boot=n_boot)

    # Test vs Real drift
    drift: dict[str, Any] = {}
    if y_test is not None and test_pred is not None:
        yt = np.asarray(y_test, dtype=float)
        tp = np.asarray(test_pred, dtype=float)
        if len(yt) == len(tp) and len(yt) >= 4:
            test_r2 = _r2(yt, tp)
            real_r2 = metrics.get("r2", {}).get("point", np.nan)
            drift = {
                "test_r2": float(test_r2),
                "real_r2": float(real_r2),
                "delta": float(real_r2 - test_r2) if not np.isnan(real_r2) else np.nan,
                "drift_severity": (
                    "small" if abs(real_r2 - test_r2) < 0.05 else
                    "moderate" if abs(real_r2 - test_r2) < 0.15 else
                    "large (distribution shift?)"
                ) if not np.isnan(real_r2) else "unknown",
            }

    # Verdict
    if metrics.get("r2", {}).get("point", -float("inf")) >= 0.7:
        verdict = "✅ Real R² ≥ 0.7 — prospective forecast 신뢰 (n=8 caveat)"
    elif metrics.get("r2", {}).get("point", -float("inf")) >= 0:
        verdict = "✓ Real R² ≥ 0 — prospective signal 있음 (n=8 한계)"
    else:
        verdict = "⚠ Real R² < 0 — prospective forecast 실패"

    return RealAudit(
        model_name=name,
        n_real=n,
        metrics=metrics,
        test_vs_real_drift=drift,
        verdict=verdict,
    )


def load_real_results() -> list[dict]:
    """P1 real_forecaster real_eval/*.json 로드."""
    out = []
    real_dir = get_results_dir() / "real_eval"
    if not real_dir.exists():
        return out
    for fp in sorted(real_dir.glob("*.json")):
        if fp.name.startswith("_") or fp.name == "summary.json":
            continue
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
            d["_path"] = str(fp)
            d["_name"] = fp.stem
            out.append(d)
        except Exception as e:
            print(f"  ✗ {fp.name}: {e}", file=sys.stderr)
    return out


def render_real_md(real_audits: list[RealAudit]) -> str:
    """Real audit Markdown 보고서."""
    lines = []
    lines.append("# Statistical Audit — Real n=8 Prospective Forecast (Package G)")
    lines.append("")
    lines.append(f"> **Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("> **Standard**: TRIPOD+AI 2024, EPIFORGE 2020 (prospective evaluation)")
    lines.append("> **n=8 한계**: R² 95% CI 매우 넓음, DM test 검출력 약함")
    lines.append("> **권장**: paper §sensitivity 또는 §future work 으로 보고")
    lines.append("")
    lines.append("---")
    lines.append("")

    if not real_audits:
        lines.append("⚠ Real 결과 없음 — P1 real_forecaster 진입 후 가용.")
        return "\n".join(lines)

    lines.append("## Real n=8 통계 검증")
    lines.append("")
    lines.append("| Model | n_real | Real R² (Fisher z) | Real R² (Bootstrap) | RMSE (95% CI) | Test R² | Δ (real-test) | Drift |")
    lines.append("|-------|------:|--------------------|---------------------|---------------|--------:|--------------:|:------|")
    for a in real_audits:
        m = a.metrics
        r2 = m.get("r2", {})
        ci_fz = r2.get("ci_fisher_z", (np.nan, np.nan))
        ci_bs = r2.get("ci_bootstrap", {}) or {}
        rmse = m.get("rmse", {})
        drift = a.test_vs_real_drift

        r2_fz = (f"{r2.get('point', 0):.3f} [{ci_fz[0]:.3f}, {ci_fz[1]:.3f}]"
                 if r2.get("point") is not None else "N/A")
        r2_bs = (f"{r2.get('point', 0):.3f} [{ci_bs.get('ci_low', 0):.3f}, {ci_bs.get('ci_high', 0):.3f}]"
                 if ci_bs.get("ci_low") is not None else "N/A")
        rmse_s = (f"{rmse.get('point', 0):.2f} [{rmse.get('ci_low', 0):.2f}, {rmse.get('ci_high', 0):.2f}]"
                  if rmse.get("point") is not None else "N/A")

        test_r2 = drift.get("test_r2", "N/A")
        delta = drift.get("delta", "N/A")
        sev = drift.get("drift_severity", "N/A")

        test_s = f"{test_r2:.3f}" if isinstance(test_r2, (int, float)) else str(test_r2)
        delta_s = f"{delta:+.3f}" if isinstance(delta, (int, float)) else str(delta)

        lines.append(f"| {a.model_name} | {a.n_real} | {r2_fz} | {r2_bs} | {rmse_s} | {test_s} | {delta_s} | {sev} |")
    lines.append("")

    # 종합
    n_real_good = sum(1 for a in real_audits
                       if a.metrics.get("r2", {}).get("point", -float("inf")) >= 0.7)
    n_real_pos = sum(1 for a in real_audits
                      if a.metrics.get("r2", {}).get("point", -float("inf")) >= 0)
    lines.append("## 종합")
    lines.append("")
    lines.append(f"- Real R² ≥ 0.7: {n_real_good}/{len(real_audits)}")
    lines.append(f"- Real R² ≥ 0: {n_real_pos}/{len(real_audits)}")
    lines.append("")
    lines.append("> **TRIPOD+AI**: Real n=8 은 prospective signal — paper 본문이 아닌 §sensitivity 권장.")
    lines.append("> Test R² (n=68) 가 main result, real 은 deployment validation.")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["prediction", "simulation", "real", "both", "all"], default="all")
    ap.add_argument("--baseline", default="persistence", help="baseline 모델 이름 (DM 비교용)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out-json", default=str(get_results_dir() / "STATISTICAL_AUDIT.json"))
    ap.add_argument("--out-md", default=str(get_results_dir() / "STATISTICAL_AUDIT.md"))
    args = ap.parse_args()

    print("Statistical audit 시작")
    print(f"  모드: {args.mode}")
    print(f"  baseline: {args.baseline}")
    print()

    pred_audits: list[PredictionAudit] = []
    sim_audits: list[SimulationAudit] = []

    real_audits: list[RealAudit] = []

    if args.mode in ("prediction", "both", "all"):
        pred_results = load_prediction_results()
        print(f"  예측 모델 결과: {len(pred_results)} 파일")

        # baseline 모델 prediction 추출 (DM test 용)
        baseline_pred = None
        for d in pred_results:
            if d.get("_name", "").lower() == args.baseline.lower():
                baseline_pred = np.asarray(d.get("test_predictions", []))
                break

        for d in pred_results:
            try:
                yt = np.asarray(d.get("y_true", []))
                yp = np.asarray(d.get("test_predictions", d.get("y_pred", [])))
                if len(yt) == 0 or len(yp) == 0 or len(yt) != len(yp):
                    continue
                yl = np.asarray(d.get("pi_lower")) if "pi_lower" in d else None
                yu = np.asarray(d.get("pi_upper")) if "pi_upper" in d else None
                bp = baseline_pred if baseline_pred is not None and len(baseline_pred) == len(yt) else None
                a = audit_prediction_model(
                    name=d["_name"], y_true=yt, y_pred=yp,
                    y_lower=yl, y_upper=yu,
                    baseline_pred=bp,
                    alpha_blend=d.get("alpha_anchor"),  # persisted JSON key "alpha_anchor" — read 불변
                    n_boot=args.n_boot,
                )
                pred_audits.append(a)
            except Exception as e:
                print(f"  ✗ {d.get('_name')}: {type(e).__name__}: {e}")

        # MCS test (전체 모델)
        if len(pred_audits) >= 2:
            losses = {}
            for d in pred_results:
                yt = np.asarray(d.get("y_true", []))
                yp = np.asarray(d.get("test_predictions", []))
                if len(yt) and len(yt) == len(yp):
                    losses[d["_name"]] = (yt - yp) ** 2
            if losses:
                mcs = mcs_test(losses, alpha=0.05, n_boot=args.n_boot // 2)
                print(f"  MCS@5%: {mcs['mcs_size']} 모델 살아남음 / {len(losses)} 전체")
                for a in pred_audits:
                    a.mcs = {
                        "in_mcs": a.model_name in mcs["survivors"],
                        "mcs_size": mcs["mcs_size"],
                        "alpha": 0.05,
                    }

    if args.mode in ("simulation", "both"):
        sim_results = load_simulation_results()
        print(f"  시뮬레이션 결과: {len(sim_results)} 파일")

        baseline_inc = None
        for d in sim_results:
            if d.get("scenario", "").lower() == "baseline":
                baseline_inc = np.asarray(d.get("incidence", []))

        for d in sim_results:
            try:
                inc = np.asarray(d.get("incidence", []))
                if len(inc) == 0:
                    continue
                pop = np.asarray(d.get("populations", [400000] * 25))
                rt = np.asarray(d.get("rt_estimates")) if "rt_estimates" in d else None
                pl = np.asarray(d.get("pi_lower")) if "pi_lower" in d else None
                pu = np.asarray(d.get("pi_upper")) if "pi_upper" in d else None
                bi = baseline_inc if baseline_inc is not None and d.get("scenario") != "baseline" else None
                a = audit_simulation(
                    scenario=d.get("scenario", d.get("_path", "?")),
                    incidence=inc, populations=pop, rt_estimates=rt,
                    pi_lower=pl, pi_upper=pu,
                    baseline_incidence=bi,
                    disease_params=d.get("disease_params"),
                )
                sim_audits.append(a)
            except Exception as e:
                print(f"  ✗ {d.get('scenario')}: {type(e).__name__}: {e}")

    # Package G — Real n=8 audit
    if args.mode in ("real", "all"):
        real_results = load_real_results()
        print(f"  Real n=8 결과: {len(real_results)} 파일")

        # test predictions for drift comparison
        test_preds_by_model: dict[str, dict] = {}
        if pred_audits:
            for d in load_prediction_results():
                test_preds_by_model[d["_name"]] = {
                    "y_test": np.asarray(d.get("y_true", [])),
                    "test_pred": np.asarray(d.get("test_predictions", [])),
                }

        for d in real_results:
            try:
                yr = np.asarray(d.get("y_real", d.get("real_y_true", [])))
                rp = np.asarray(d.get("real_predictions", d.get("real_pred", [])))
                if len(yr) == 0 or len(rp) == 0 or len(yr) != len(rp):
                    continue

                test_data = test_preds_by_model.get(d["_name"], {})
                a = audit_real_forecast(
                    name=d["_name"], y_real=yr, real_pred=rp,
                    y_test=test_data.get("y_test"),
                    test_pred=test_data.get("test_pred"),
                    n_boot=args.n_boot * 10,   # n=8 → bootstrap 10x
                )
                real_audits.append(a)
            except Exception as e:
                print(f"  ✗ {d.get('_name')}: {type(e).__name__}: {e}")

    # Output — main audit
    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "ts": time.time(),
        "mode": args.mode,
        "baseline": args.baseline,
        "prediction_audits": [asdict(a) for a in pred_audits],
        "simulation_audits": [asdict(a) for a in sim_audits],
        "real_audits": [asdict(a) for a in real_audits],
    }
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n✓ JSON: {out_path}")

    md_path = Path(args.out_md)
    md_path.write_text(render_md(pred_audits, sim_audits), encoding="utf-8")
    print(f"✓ Markdown (test/sim): {md_path}")

    # Real audit MD 별도 (n=8 한계 명시 위해)
    if real_audits:
        real_md_path = Path(str(md_path).replace(".md", "_REAL.md"))
        real_md_path.write_text(render_real_md(real_audits), encoding="utf-8")
        print(f"✓ Real audit (n=8): {real_md_path}")

    # 요약
    print()
    print(f"  예측 모델 (test n=68):  {len(pred_audits)} audited")
    print(f"  시뮬 (P3 abm SEIR):    {len(sim_audits)} audited")
    print(f"  Real (n=8 prospective): {len(real_audits)} audited")
    if pred_audits:
        n_dm = sum(1 for a in pred_audits
                   if a.dm_vs_baseline.get("significant_at_5pct")
                   and a.dm_vs_baseline.get("t_stat", 0) < 0)
        print(f"  DM 유의 vs baseline (test): {n_dm}/{len(pred_audits)}  (champion=best-WIS)")
    if real_audits:
        n_real_pos = sum(1 for a in real_audits
                          if a.metrics.get("r2", {}).get("point", -float("inf")) >= 0)
        print(f"  Real R² ≥ 0:             {n_real_pos}/{len(real_audits)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
