"""
Metric interpretation rubric — per-metric thresholds + citations.
==================================================================

For every metric reported in R8 scoring / R9 per_model_optimize, this module declares:
  • What the metric measures
  • The "lower is better" / "higher is better" / "calibration target" direction
  • Numerical thresholds for "excellent / good / acceptable / poor"
  • The canonical literature citation

Used by R9 per_model_optimize's REPORT.md generator to add a rubric section
("metric interpretation guide") and by per-model deep-dives to flag
each metric value with a quality tag (✓ excellent / ⚠ acceptable / ✗ poor).

These thresholds are conventions from the literature — they are NOT
hard truths. Reviewers may legitimately disagree; the citations let
them follow up. For ILI rate forecasting at n=68, the targets reflect
mid-tier journal expectations (Influenza and Other Respiratory Viruses,
BMC Infectious Diseases, JMIR Public Health and Surveillance, IJID).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class MetricRule:
    name: str
    direction: str       # "lower" | "higher" | "calibration"
    description: str
    excellent: Optional[float]   # threshold for ✓
    good: Optional[float]        # threshold for "good"
    acceptable: Optional[float]  # threshold for ⚠
    citation: str

    def quality(self, value: float) -> str:
        """Return one of {excellent, good, acceptable, poor, n/a}."""
        try:
            v = float(value)
        except (TypeError, ValueError):
            return "n/a"
        if v != v:  # NaN
            return "n/a"
        if self.direction == "lower":
            if self.excellent is not None and v <= self.excellent: return "excellent"
            if self.good is not None and v <= self.good: return "good"
            if self.acceptable is not None and v <= self.acceptable: return "acceptable"
            return "poor"
        if self.direction == "higher":
            if self.excellent is not None and v >= self.excellent: return "excellent"
            if self.good is not None and v >= self.good: return "good"
            if self.acceptable is not None and v >= self.acceptable: return "acceptable"
            return "poor"
        if self.direction == "calibration":
            # For calibration metrics, deviation from a target is the score.
            # Conventions:
            #   PIT mean: target 0.5, tolerance ±0.05 excellent / ±0.10 good / ±0.15 acceptable
            #   PI coverage: target = nominal, tolerance ±2pp / ±5pp / ±10pp
            target = self.excellent if self.excellent is not None else 0.5
            dev = abs(v - target)
            if self.good is not None and dev <= self.good: return "excellent"
            if self.acceptable is not None and dev <= self.acceptable: return "good"
            return "poor"
        return "n/a"


# Master rubric: per-metric interpretation
RUBRIC: dict[str, MetricRule] = {
    # ── Point/Scaled forecasting ─────────────────────────────────────
    "r2": MetricRule(
        name="R²", direction="higher",
        description="Coefficient of determination on test slab",
        excellent=0.90, good=0.80, acceptable=0.50,
        citation="Hyndman & Athanasopoulos (2021), Forecasting Principles & Practice",
    ),
    "mae": MetricRule(
        name="MAE", direction="lower",
        description="Mean Absolute Error in ILI rate units (/1000)",
        excellent=2.0, good=4.0, acceptable=6.0,
        citation="FluSight 2024-25; Bracher et al. 2021 PLOS Comp Bio",
    ),
    "rmse": MetricRule(
        name="RMSE", direction="lower",
        description="Root Mean Squared Error",
        excellent=3.0, good=6.0, acceptable=10.0,
        citation="Hyndman & Koehler 2006 IJF 22:679",
    ),
    "mape": MetricRule(
        name="MAPE (%)", direction="lower",
        description="Mean Absolute Percentage Error — Lewis (1982) interpretation",
        excellent=10.0, good=20.0, acceptable=50.0,
        citation=("Lewis 1982 'International and Business Forecasting Methods';"
                   " Hyndman & Koehler 2006"),
    ),
    "smape": MetricRule(
        name="sMAPE (%)", direction="lower",
        description="Symmetric MAPE, undefined-at-zero protection",
        excellent=10.0, good=20.0, acceptable=40.0,
        citation="Hyndman & Koehler 2006 IJF 22:679",
    ),
    "mdape": MetricRule(
        name="MdAPE (%)", direction="lower",
        description="Median APE — robust to outliers",
        excellent=8.0, good=15.0, acceptable=30.0,
        citation="Armstrong & Collopy 1992 IJF 8:69",
    ),
    "mase_h1": MetricRule(
        name="MASE (h=1)", direction="lower",
        description="MAE / naive 1-step-baseline MAE. <1 = beats persistence",
        excellent=0.5, good=1.0, acceptable=1.5,
        citation="Hyndman & Koehler 2006 IJF 22:679 (canonical scaled error)",
    ),
    "mase_h52": MetricRule(
        name="MASE (h=52, seasonal)", direction="lower",
        description="MAE / seasonal-naive (52w) MAE. <1 = beats seasonal naive",
        excellent=0.3, good=0.7, acceptable=1.0,
        citation="Hyndman & Koehler 2006 IJF 22:679",
    ),
    "mase_h4": MetricRule(
        name="MASE (h=4, monthly seasonality)", direction="lower",
        description="MAE / 4w-lag-naive MAE",
        excellent=0.5, good=1.0, acceptable=1.5,
        citation="Hyndman & Koehler 2006 IJF 22:679",
    ),
    "mase_h13": MetricRule(
        name="MASE (h=13, quarterly)", direction="lower",
        description="MAE / 13w-lag-naive MAE",
        excellent=0.4, good=0.8, acceptable=1.2,
        citation="Hyndman & Koehler 2006 IJF 22:679",
    ),
    "bias_mean_error": MetricRule(
        name="Bias (mean signed error)", direction="calibration",
        description="Mean(y_pred - y_true). Target = 0 (unbiased). |bias| < σ/2 = good.",
        excellent=0.0, good=0.5, acceptable=2.0,  # tolerances at scale of ILI rate
        citation="Hyndman & Koehler 2006",
    ),
    "msle": MetricRule(
        name="MSLE", direction="lower",
        description="Mean Squared Logarithmic Error — useful at low ILI rates",
        excellent=0.05, good=0.20, acceptable=1.0,
        citation="Tofallis 2015 J Operational Research Soc 66:1352",
    ),
    "theils_u": MetricRule(
        name="Theil's U2", direction="lower",
        description="Forecast accuracy ratio vs naive 1-step. <1 = beats naive",
        excellent=0.5, good=1.0, acceptable=1.5,
        citation="Theil 1966; Bliemel 1973 Mgmt Sci 19:444",
    ),
    "log_score_gauss": MetricRule(
        name="Log score (Gaussian NLL)", direction="lower",
        description="Negative log-likelihood under Gaussian predictive. CDC FluSight standard.",
        excellent=2.5, good=3.5, acceptable=5.0,
        citation="Gneiting & Raftery 2007 JASA 102:359",
    ),
    "skill_mae_vs_persist": MetricRule(
        name="MAE skill vs persistence", direction="higher",
        description="1 − MAE_model / MAE_persistence. >0 beats persistence baseline.",
        excellent=0.50, good=0.30, acceptable=0.10,
        citation="Murphy 1973 J Appl Meteor 12:595; Bracher 2021 PLOS Comp Bio",
    ),
    # ── Probabilistic ────────────────────────────────────────────────
    "wis": MetricRule(
        name="WIS", direction="lower",
        description="Weighted Interval Score (Bracher 2021); FluSight primary",
        excellent=2.0, good=4.0, acceptable=6.0,
        citation="Bracher J et al. 2021 PLOS Comp Bio 17:e1008618",
    ),
    "log_wis": MetricRule(
        name="log-WIS", direction="lower",
        description="WIS on log-scale (Bosse 2023, FluSight 2024-25 primary)",
        excellent=0.10, good=0.20, acceptable=0.40,
        citation="Bosse NI et al. 2023 PLoS Comp Bio 19:e1011393",
    ),
    "crps_gaussian": MetricRule(
        name="CRPS (Gaussian)", direction="lower",
        description="Continuous Ranked Probability Score, parametric Gaussian",
        excellent=2.0, good=4.0, acceptable=6.0,
        citation="Gneiting & Raftery 2007 JASA 102(477):359, Eq.(5)",
    ),
    "pinball_q50": MetricRule(
        name="Pinball loss q=0.50", direction="lower",
        description="Quantile loss at predictive median; equals 0.5×MAE for the median",
        excellent=1.0, good=2.5, acceptable=4.0,
        citation="Tibshirani 2023 lecture notes (statlearn)",
    ),
    "pit_mean": MetricRule(
        name="PIT mean", direction="calibration",
        description="Probability Integral Transform mean. Target = 0.5 (uniform)",
        excellent=0.5,    # target value
        good=0.05,        # tolerance for excellent
        acceptable=0.10,  # tolerance for good
        citation="Bracher 2021 §3, Gneiting Balabdaoui Raftery 2007 JRSS-B 69:243",
    ),
    "pit_ks_p": MetricRule(
        name="PIT KS p-value", direction="higher",
        description="Kolmogorov-Smirnov p for PIT uniformity. >0.05 = calibrated",
        excellent=0.20, good=0.05, acceptable=0.01,
        citation="Gneiting Balabdaoui Raftery 2007 JRSS-B",
    ),
    # ── PI Coverage ──────────────────────────────────────────────────
    "pi95_coverage": MetricRule(
        name="95% PI empirical coverage", direction="calibration",
        description="Should equal nominal 0.95. Wilson exact CI reported.",
        excellent=0.95, good=0.02, acceptable=0.05,
        citation="Bracher 2021; FluSight 2024-25",
    ),
    "pi80_coverage": MetricRule(
        name="80% PI coverage", direction="calibration",
        description="Should equal nominal 0.80",
        excellent=0.80, good=0.05, acceptable=0.10,
        citation="Bracher 2021",
    ),
    "pi50_coverage": MetricRule(
        name="50% PI coverage", direction="calibration",
        description="Should equal nominal 0.50; FluSight reports this alongside 95%",
        excellent=0.50, good=0.05, acceptable=0.10,
        citation="FluSight 2024-25 evaluation report",
    ),
    # ── Direction / Epi-curve ────────────────────────────────────────
    "direction_acc": MetricRule(
        name="Direction accuracy", direction="higher",
        description="Fraction of weeks with correct up/down sign (lag-1 differencing)",
        excellent=0.75, good=0.65, acceptable=0.55,
        citation="FluSight 2024-25 categorical 'rate-trend' target",
    ),
    "peak_week_err": MetricRule(
        name="Peak week error (|Δweeks|)", direction="lower",
        description="Absolute error of argmax week; 0 = perfect; ≤1 = within tolerance",
        excellent=0.0, good=1.0, acceptable=2.0,
        citation="CDC FluSight 2018-19 onward (peak-week target)",
    ),
    "peak_int_relerr": MetricRule(
        name="Peak intensity rel-err", direction="lower",
        description="|peak_pred - peak_true| / peak_true at the actual peak week",
        excellent=0.10, good=0.20, acceptable=0.40,
        citation="CDC FluSight peak-intensity target",
    ),
    # ── Clinical / Alert (KDCA threshold) ────────────────────────────
    "alert_f1": MetricRule(
        name="Alert F1 (KDCA threshold)", direction="higher",
        description="F1 of binary alert (predicted vs actual threshold crossing)",
        excellent=0.90, good=0.75, acceptable=0.50,
        citation="KDCA 인플루엔자 표본감시 운영지침; Reich Lab benchmark",
    ),
    "brier_score": MetricRule(
        name="Brier score (alert event)", direction="lower",
        description="Mean squared error of P(Y>τ) probability forecast",
        excellent=0.10, good=0.20, acceptable=0.30,
        citation="Brier 1950 MWR 78:1",
    ),
    "brier_skill": MetricRule(
        name="Brier skill score (vs climatology)", direction="higher",
        description="1 - BS/BS_ref. >0 beats climatology; >0.5 strong improvement",
        excellent=0.50, good=0.30, acceptable=0.10,
        citation="Murphy 1973 J Appl Meteor 12:595",
    ),
    "sensitivity": MetricRule(
        name="Sensitivity (recall)", direction="higher",
        description="TP / (TP + FN) at threshold",
        excellent=0.90, good=0.80, acceptable=0.70,
        citation="Standard public-health alert system metric",
    ),
    "specificity": MetricRule(
        name="Specificity", direction="higher",
        description="TN / (TN + FP) at threshold",
        excellent=0.90, good=0.80, acceptable=0.70,
        citation="Standard public-health alert system metric",
    ),
    "ppv": MetricRule(
        name="PPV (precision)", direction="higher",
        description="TP / (TP + FP) — depends on prevalence",
        excellent=0.85, good=0.70, acceptable=0.50,
        citation="Standard prevalence-dependent metric",
    ),
    "npv": MetricRule(
        name="NPV", direction="higher",
        description="TN / (TN + FN)",
        excellent=0.95, good=0.85, acceptable=0.70,
        citation="Standard prevalence-dependent metric",
    ),
    "clinical_f1": MetricRule(
        name="Clinical F1 (binary at threshold)", direction="higher",
        description="2·sens·PPV / (sens + PPV)",
        excellent=0.85, good=0.70, acceptable=0.55,
        citation="van Rijsbergen 1979 'Information Retrieval'",
    ),
    # ── Comparative ──────────────────────────────────────────────────
    "relative_wis_pairwise": MetricRule(
        name="Relative WIS (pairwise tournament)", direction="lower",
        description=("Geomean of WIS_M / WIS_M' over all opponents. "
                     "<1 = beats average opponent (Sherratt 2023)"),
        excellent=0.50, good=0.80, acceptable=1.00,
        citation="Sherratt K et al. 2023 eLife 12:e81916; Bosse 2022 scoringutils",
    ),
}


def render_rubric_markdown() -> str:
    """Return a markdown table of all metrics + interpretation thresholds."""
    md = [
        "## Metric interpretation rubric",
        "",
        "_Per-metric thresholds with literature citations. Quality tag (`✓ excellent` /",
        "`good` / `⚠ acceptable` / `✗ poor`) is applied automatically in per-model_",
        "_deep-dive reports. Direction column: `lower=lower-is-better`, `higher=higher-is-better`,_",
        "_`calibration=closer-to-nominal-better`._",
        "",
        "| Metric | Direction | Excellent | Good | Acceptable | Citation |",
        "|---|---|---|---|---|---|",
    ]
    for key, rule in RUBRIC.items():
        if rule.direction == "calibration":
            exc = f"|target−value| ≤ {rule.good}"
            gd = f"≤ {rule.acceptable}"
            ac = "else"
        else:
            arrow = "≤" if rule.direction == "lower" else "≥"
            exc = f"{arrow} {rule.excellent}" if rule.excellent is not None else "—"
            gd = f"{arrow} {rule.good}" if rule.good is not None else "—"
            ac = f"{arrow} {rule.acceptable}" if rule.acceptable is not None else "—"
        md.append(
            f"| **{rule.name}** | {rule.direction} | {exc} | {gd} | {ac} | "
            f"{rule.citation} |"
        )
    md += [
        "",
        "_Caveats:_",
        "- Thresholds reflect mid-tier-journal expectations; reviewers may disagree.",
        "- For `calibration` metrics, the value column shows |empirical − nominal|.",
        "- For ILI rate forecasting at n=68, MAE in the 3-5 range is competitive",
        "  (NegBinGLM achieves 3.92, the lowest in our 61-model leaderboard).",
        "- WIS thresholds calibrated against FluSight 2024-25 typical scores.",
        "",
    ]
    return "\n".join(md)


def annotate_row(row: dict) -> dict:
    """Attach a quality tag to each metric value in a row dict.

    Returns a new dict with `<metric>_q` keys added (e.g., `r2_q="good"`).
    """
    out = dict(row)
    for key, rule in RUBRIC.items():
        if key in row and row[key] is not None:
            out[f"{key}_q"] = rule.quality(row[key])
    return out


def quality_emoji(q: str) -> str:
    return {"excellent": "✓", "good": "·", "acceptable": "⚠", "poor": "✗", "n/a": "?"}.get(q, "?")


# ════════════════════════════════════════════════════════════════════════════
# Sprint D2 (2026-05-26): R7 PAPER_TOP_{2,3,5,10} 폐기
# 사용자 명시: "paper top tier은 왜 만들어?!"
# PAPER_TOP_*, classify_paper_tier, paper_top_summary 모두 제거.
# 2026-06-05 (사용자 명시): G-175 4-criteria filter (PRIMARY_FILTER_*) 도 완전 제거 —
# champion = 순수 best-WIS. BH-FDR (PRIMARY_DM_FAMILY) + Bootstrap CI (PRIMARY_CI_METRICS) 만 유지.
# ════════════════════════════════════════════════════════════════════════════

# DM test family — BH-FDR correction within model
PRIMARY_DM_FAMILY: list[str] = [
    "dm_p_value", "dm_p_vs_climatology", "dm_p_vs_lag52",
]
PRIMARY_DM_FAMILY_ADJUSTED: list[str] = [k + "_bh" for k in PRIMARY_DM_FAMILY]

# Bootstrap CI (BCa + Künsch block, B=1000, seed=42)
PRIMARY_CI_METRICS: list[str] = [
    "mae_ci95_lo", "mae_ci95_hi", "wis_ci95_lo", "wis_ci95_hi",
]


# ════════════════════════════════════════════════════════════════════════════
# Per-phase metric registry (2026-05-28) — phase 별 metric SSOT
# ────────────────────────────────────────────────────────────────────────────
# 각 phase 가 산출/책임지는 metric 단일 정의. 이전엔 코드·문서 어디에도 phase→metric
# 매핑이 없어 "각 phase 가 뭘 평가하는지" 파악 불가 → 본 registry 가 단일 source.
#
# 필드:
#   name       : phase 표시명 (새 번호 = dispatch 순서)
#   slab       : 평가 데이터 — oof(WF-CV out-of-fold) / oof_cv(3-fold 내부) /
#                test(in-sample held-out n≈68) / real(cutoff 이후 OOS) /
#                pool(train+val) / champion / new / country / all / N/A
#   full_134   : evaluate_predictions_full (134-key R8 evaluator) 를 해당 slab 에서 호출하는가
#   primary    : 그 phase 가 authoritative 하게 생산하는 domain metric (134 부분집합 또는 고유).
#                "ALL_134" = 전체 134, 비-134 항목은 설명 라벨.
# 전체 목록: phase_evaluator.evaluate_predictions_full. 현재 키 수 = **129** (g175 5키 제거 후 134→129,
#   2026-06-05). 아래 'full_134' 키명은 "전체 evaluator 호출 여부" 플래그의 역사적 라벨 — 의미는 '전체 129'.
PHASE_METRICS: dict[int, dict] = {
    1:  {"name": "data + FE",            "slab": "N/A",      "full_134": False, "primary": []},
    2:  {"name": "multicollinearity",    "slab": "pool",     "full_134": False, "primary": []},
    3:  {"name": "feature optuna load",  "slab": "N/A",      "full_134": False, "primary": []},
    4:  {"name": "baseline",             "slab": "test",     "full_134": True,  "primary": ["r2", "rmse", "mae", "mape", "wis"]},
    5:  {"name": "external optuna",      "slab": "test",     "full_134": True,  "primary": ["r2", "rmse", "mae", "mape", "wis"]},
    6:  {"name": "walk-forward CV",      "slab": "oof",      "full_134": True,  "primary": ["r2", "rmse", "mae", "mape", "wis", "pi95_coverage"]},
    7:  {"name": "residual diagnostics", "slab": "oof",      "full_134": True,
         "primary": ["shapiro_wilk_p", "jarque_bera_p", "ljung_box_p", "ljung_box_q",
                     "durbin_watson", "residual_acf_lag1", "residual_skew", "residual_kurtosis"]},
    8:  {"name": "AR correction",        "slab": "oof",      "full_134": True,  "primary": ["mae", "rmse", "r2"]},
    9:  {"name": "DM test",              "slab": "oof",      "full_134": True,
         "primary": ["dm_z_stat", "dm_p_value", "dm_p_value_bh", "dm_z_vs_climatology",
                     "dm_p_vs_climatology", "dm_p_vs_climatology_bh", "dm_z_vs_lag52",
                     "dm_p_vs_lag52", "dm_p_vs_lag52_bh"]},
    10: {"name": "PI + conformal",       "slab": "oof",      "full_134": True,
         "primary": ["pi50_coverage", "pi50_width", "pi50_rel_width", "pi80_coverage", "pi80_width",
                     "pi80_rel_width", "pi80_relia", "pi95_coverage", "pi95_width", "pi95_rel_width",
                     "pi95_relia", "pi99_coverage", "pi99_width", "pi_sharpness_ratio", "crps_gaussian",
                     "pit_mean", "pit_std", "pit_ks_p", "calibration_slope", "calibration_intercept",
                     "brier_score", "brier_skill", "brier_reliability", "brier_resolution", "brier_uncertainty"]},
    11: {"name": "composite scoring",    "slab": "oof",      "full_134": True,
         "primary": ["r2", "rmse", "wis"]},  # composite = R²40% + RMSE순위20% + DM승률15% + WFCV안정성15% + Conformal10%
    12: {"name": "real-slab eval",       "slab": "real",     "full_134": True,
         "primary": ["mae", "pi95_coverage", "peak_week_err", "peak_int_relerr", "epi_peak_mae",
                     "epi_season_total_mae", "epidemic_duration_err", "season_onset_err",
                     "lead_time_weeks", "attack_rate_relerr", "growth_rate_corr"]},
    13: {"name": "HP-optimize",          "slab": "oof_cv",   "full_134": True,
         "primary": ["wis", "r2", "mape", "pi95_coverage"]},  # OOF-WIS 최적화 (4-criteria gate 제거 2026-06-05)
    14: {"name": "per-model eval (SSOT)", "slab": "test",    "full_134": True,
         "primary": ["ALL_134"]},  # 전체 134 = 논문 Table 1 SSOT row
    15: {"name": "SHAP + XAI",           "slab": "champion", "full_134": False,
         "primary": ["shap_values", "permutation_importance", "feature_rank"]},  # 134 아님 (XAI)
    16: {"name": "comprehensive eval",   "slab": "all",      "full_134": False,
         "primary": ["aggregate(MASTER_GRID)", "figures", "fairness", "loso"]},
    17: {"name": "inference",            "slab": "new",      "full_134": False, "primary": ["point_forecast", "pi95"]},
    18: {"name": "overseas validation",  "slab": "country",  "full_134": True,  "primary": ["r2", "rmse", "mape", "wis"]},
}


__all__ = [
    "MetricRule", "RUBRIC",
    "render_rubric_markdown", "annotate_row", "quality_emoji",
    # BH-FDR + Bootstrap CI (PRIMARY_FILTER_* 제거 2026-06-05; PAPER_TOP_* 제거 2026-05-26)
    "PRIMARY_DM_FAMILY", "PRIMARY_DM_FAMILY_ADJUSTED",
    "PRIMARY_CI_METRICS",
    "PHASE_METRICS",
]
