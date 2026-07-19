"""
Analytics module: reusable epidemiological and forecast diagnostics.

Submodules:
  - metrics: Forecast comparison and accuracy metrics
  - diagnostics: Probabilistic forecast validation
  - epidemiological: Rt estimation and excess detection
"""

from .metrics import (
    diebold_mariano,
    crps_gaussian,
    _calc_metrics,
    pinball_loss,
    pi_coverage,
    pi_calibration_table,
    mcnemar_test,
    peak_week_error,
    peak_intensity_error,
    direction_accuracy,
    brier_score,
    brier_skill_score,
    binary_clinical_rates,
    decision_curve,
    bootstrap_ci,
    adjust_pvalues,
)

from .diagnostics import (
    pit_values,
    calibration_check,
    weighted_interval_score,
    model_confidence_set,
    coverage_gap_by_regime,
    coverage_gap_table,
)

from .epidemiological import (
    estimate_rt_cori,
    serfling_regression,
)

__all__ = [
    # metrics — point & comparison
    "diebold_mariano",
    "crps_gaussian",
    "_calc_metrics",
    "mcnemar_test",
    # metrics — probabilistic / interval
    "pinball_loss",
    "pi_coverage",
    "pi_calibration_table",
    # metrics — epi-curve & clinical
    "peak_week_error",
    "peak_intensity_error",
    "direction_accuracy",
    "brier_score",
    "brier_skill_score",
    "binary_clinical_rates",
    "decision_curve",
    # metrics — inference / CI
    "bootstrap_ci",
    "adjust_pvalues",
    # diagnostics
    "pit_values",
    "calibration_check",
    "weighted_interval_score",
    "model_confidence_set",
    "coverage_gap_by_regime",
    "coverage_gap_table",
    # epidemiological
    "estimate_rt_cori",
    "serfling_regression",
]
