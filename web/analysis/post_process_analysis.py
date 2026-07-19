#!/usr/bin/env python3
"""Run read-only post-processing analyses for the Seoul ILI web dashboard.

The script reads existing prediction artifacts only. It does not retrain models
or write derived data; the Markdown report is printed to stdout.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import LinearRegression


ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS_DIR = ROOT / "simulation" / "results" / "csv"
SUMMARY_PATH = PREDICTIONS_DIR / "summary_metrics.csv"
BACKTEST_PATH = ROOT / "web" / "public" / "aggregates" / "backtest.json"
REQUIRED_PREDICTION_COLUMNS = {"split", "idx", "y_true", "y_pred"}
QUANTILE_LEVELS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)


@dataclass(frozen=True)
class PointMetrics:
    """Point-forecast metrics used in the printed comparison."""

    mae: float
    rmse: float
    peak_mae: float


@dataclass(frozen=True)
class PredictionData:
    """Aligned predictions and common outcomes for all valid models."""

    model_predictions: dict[str, dict[str, np.ndarray]]
    y_val: np.ndarray
    y_test: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    warnings: tuple[str, ...]


def _warn(messages: list[str], message: str) -> None:
    messages.append(message)
    logging.warning(message)


def _load_predictions(directory: Path) -> PredictionData:
    """Load and align every valid predictions_*.csv file."""

    messages: list[str] = []
    model_predictions: dict[str, dict[str, np.ndarray]] = {}
    canonical: dict[str, pd.DataFrame] = {}

    for path in sorted(directory.glob("predictions_*.csv")):
        model_name = path.stem.removeprefix("predictions_")
        try:
            frame = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - artifact-dependent
            _warn(messages, f"Skipping {path.name}: read failed ({exc}).")
            continue

        missing = REQUIRED_PREDICTION_COLUMNS.difference(frame.columns)
        if missing:
            _warn(messages, f"Skipping {path.name}: missing columns {sorted(missing)}.")
            continue

        frame = frame.loc[:, ["split", "idx", "y_true", "y_pred"]].copy()
        frame["split"] = frame["split"].astype(str).str.lower()
        for column in ("idx", "y_true", "y_pred"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        if frame.isna().any().any() or not np.isfinite(
            frame[["idx", "y_true", "y_pred"]].to_numpy(dtype=float)
        ).all():
            _warn(messages, f"Skipping {path.name}: missing or non-finite values.")
            continue
        if frame.duplicated(["split", "idx"]).any():
            _warn(messages, f"Skipping {path.name}: duplicate split/idx rows.")
            continue
        if set(frame["split"]) != {"val", "test"}:
            _warn(messages, f"Skipping {path.name}: expected val and test splits.")
            continue

        aligned: dict[str, np.ndarray] = {}
        valid = True
        for split in ("val", "test"):
            part = frame.loc[frame["split"] == split].sort_values("idx").reset_index(drop=True)
            truth = part.loc[:, ["idx", "y_true"]]
            if split not in canonical:
                canonical[split] = truth
            else:
                reference = canonical[split]
                same_idx = np.array_equal(
                    reference["idx"].to_numpy(), truth["idx"].to_numpy()
                )
                same_truth = np.allclose(
                    reference["y_true"].to_numpy(dtype=float),
                    truth["y_true"].to_numpy(dtype=float),
                    rtol=0.0,
                    atol=1e-10,
                )
                if not same_idx or not same_truth:
                    _warn(messages, f"Skipping {path.name}: {split} rows do not align.")
                    valid = False
                    break
            aligned[split] = part["y_pred"].to_numpy(dtype=float)

        if valid:
            model_predictions[model_name] = aligned

    if not model_predictions or set(canonical) != {"val", "test"}:
        raise RuntimeError("No valid prediction files with aligned val/test splits were found.")

    return PredictionData(
        model_predictions=model_predictions,
        y_val=canonical["val"]["y_true"].to_numpy(dtype=float),
        y_test=canonical["test"]["y_true"].to_numpy(dtype=float),
        val_idx=canonical["val"]["idx"].to_numpy(dtype=int),
        test_idx=canonical["test"]["idx"].to_numpy(dtype=int),
        warnings=tuple(messages),
    )


def _load_backtest(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load backtest JSON and normalize its models into a name-keyed mapping."""

    raw = json.loads(path.read_text(encoding="utf-8"))
    container: Any = raw.get("models", raw) if isinstance(raw, dict) else raw
    models: dict[str, dict[str, Any]] = {}

    if isinstance(container, list):
        for entry in container:
            if isinstance(entry, dict) and entry.get("name"):
                models[str(entry["name"])] = entry
    elif isinstance(container, dict):
        for name, entry in container.items():
            if isinstance(entry, dict):
                normalized = dict(entry)
                normalized.setdefault("name", name)
                models[str(normalized["name"])] = normalized
    else:
        raise ValueError("backtest.json has no recognizable model collection.")

    if not models:
        raise ValueError("backtest.json contains no named models.")
    return raw if isinstance(raw, dict) else {}, models


def _single_best_name(summary_path: Path, available_models: set[str]) -> tuple[str, pd.Series]:
    """Resolve the single-best point model from summary_metrics.csv by test MAE."""

    summary = pd.read_csv(summary_path)
    required = {"name", "test_mae", "test_rmse"}
    missing = required.difference(summary.columns)
    if missing:
        raise ValueError(f"summary_metrics.csv is missing columns {sorted(missing)}.")
    eligible = summary.loc[summary["name"].isin(available_models)].copy()
    if eligible.empty:
        raise ValueError("No summary_metrics.csv model has a valid predictions file.")
    row = eligible.loc[eligible["test_mae"].astype(float).idxmin()]
    return str(row["name"]), row


def _peak_mask(y_test: np.ndarray) -> tuple[np.ndarray, float]:
    threshold = float(np.quantile(y_test, 0.75))
    return y_test >= threshold, threshold


def _point_metrics(y_true: np.ndarray, y_pred: np.ndarray, peak_mask: np.ndarray) -> PointMetrics:
    residual = y_true - y_pred
    return PointMetrics(
        mae=float(np.mean(np.abs(residual))),
        rmse=float(np.sqrt(np.mean(np.square(residual)))),
        peak_mae=float(np.mean(np.abs(residual[peak_mask]))),
    )


def _coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


def _requested_interval_score(
    y_true: np.ndarray, y_pred: np.ndarray, lower: np.ndarray, upper: np.ndarray
) -> float:
    """Compute the task-specified 95% interval score, not dashboard multi-PI WIS."""

    score = (
        np.abs(y_true - y_pred)
        + 2.0 * np.maximum(0.0, lower - y_true)
        + 2.0 * np.maximum(0.0, y_true - upper)
    )
    return float(np.mean(score))


def _point_arrays(entry: dict[str, Any], key: str) -> dict[str, np.ndarray]:
    points = entry.get(key, [])
    if not isinstance(points, list) or not points:
        raise ValueError(f"backtest model {entry.get('name')} has no {key}.")

    fields = set(points[0])
    required = {"actual", "predicted"}
    if not required.issubset(fields):
        raise ValueError(f"{key} is missing {sorted(required.difference(fields))}.")

    arrays: dict[str, np.ndarray] = {}
    for field in fields:
        try:
            arrays[field] = np.asarray([point[field] for point in points], dtype=float)
        except (KeyError, TypeError, ValueError):
            continue
    return arrays


def _caruana_selection(
    x_val: np.ndarray, y_val: np.ndarray, model_names: list[str], max_models: int = 10
) -> tuple[np.ndarray, list[str], float]:
    """Run Caruana ensemble selection with replacement and early stopping."""

    selected_indices: list[int] = []
    prediction_sum = np.zeros_like(y_val, dtype=float)
    current_mae = math.inf

    for _ in range(max_models):
        best_index: int | None = None
        best_mae = math.inf
        for model_index in range(x_val.shape[1]):
            candidate = (prediction_sum + x_val[:, model_index]) / (
                len(selected_indices) + 1
            )
            candidate_mae = float(np.mean(np.abs(y_val - candidate)))
            if candidate_mae < best_mae:
                best_mae = candidate_mae
                best_index = model_index

        if best_index is None or (selected_indices and best_mae >= current_mae - 1e-12):
            break
        selected_indices.append(best_index)
        prediction_sum += x_val[:, best_index]
        current_mae = best_mae

    if not selected_indices:
        raise RuntimeError("Caruana selection did not select a model.")

    weights = np.zeros(x_val.shape[1], dtype=float)
    for index in selected_indices:
        weights[index] += 1.0 / len(selected_indices)
    return weights, [model_names[index] for index in selected_indices], current_mae


def _quantile_higher(values: np.ndarray, level: float) -> float:
    """Return an observed residual quantile without interpolation."""

    return float(np.quantile(values, level, method="higher"))


def _minimum_threshold_for_coverage(
    absolute_residuals: np.ndarray, target_coverage: float
) -> tuple[float, float]:
    """Find the narrowest symmetric half-width reaching target test coverage."""

    ordered = np.sort(absolute_residuals)
    index = max(0, min(len(ordered) - 1, math.ceil(target_coverage * len(ordered)) - 1))
    threshold = float(ordered[index])
    achieved = float(np.mean(absolute_residuals <= threshold))
    return threshold, achieved


def _fmt_number(value: float | None) -> str:
    return "N/A" if value is None or not np.isfinite(value) else f"{value:.4f}"


def _fmt_percent(value: float | None) -> str:
    return "N/A" if value is None or not np.isfinite(value) else f"{100.0 * value:.2f}%"


def _metrics_row(label: str, metrics: PointMetrics, coverage: float | None) -> str:
    return (
        f"| {label} | {_fmt_number(metrics.mae)} | {_fmt_number(metrics.rmse)} | "
        f"{_fmt_number(metrics.peak_mae)} | {_fmt_percent(coverage)} |"
    )


def _top_weights(model_names: list[str], weights: np.ndarray, limit: int = 10) -> str:
    pairs = sorted(
        ((name, float(weight)) for name, weight in zip(model_names, weights) if weight > 1e-10),
        key=lambda item: item[1],
        reverse=True,
    )
    return ", ".join(f"{name}={weight:.4f}" for name, weight in pairs[:limit]) or "none"


def _weighted_prediction(matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Apply only nonzero weights so catastrophic unused columns cannot overflow."""

    active = weights > 1e-10
    if not active.any():
        raise ValueError("Weighted prediction requires at least one nonzero weight.")
    return matrix[:, active] @ weights[active]


def _comparison_verdict(
    baseline: PointMetrics, candidates: dict[str, PointMetrics], metric: str
) -> str:
    baseline_value = getattr(baseline, metric)
    best_name, best_metrics = min(candidates.items(), key=lambda item: getattr(item[1], metric))
    best_value = getattr(best_metrics, metric)
    delta = baseline_value - best_value
    if delta > 0:
        return f"Yes: {best_name} improves {metric} by {delta:.4f}."
    return f"No: best candidate {best_name} is worse by {-delta:.4f}."


def main() -> None:
    """Compute all requested analyses and print a structured Markdown report."""

    logging.basicConfig(level=logging.WARNING, format="WARNING: %(message)s")
    data = _load_predictions(PREDICTIONS_DIR)
    backtest_meta, backtest_models = _load_backtest(BACKTEST_PATH)
    single_best, summary_row = _single_best_name(
        SUMMARY_PATH, set(data.model_predictions)
    )
    if single_best not in backtest_models:
        raise ValueError(f"Single-best model {single_best} is absent from backtest.json.")

    model_names = sorted(data.model_predictions)
    x_val = np.column_stack([data.model_predictions[name]["val"] for name in model_names])
    x_test = np.column_stack([data.model_predictions[name]["test"] for name in model_names])
    peak_mask, peak_threshold = _peak_mask(data.y_test)
    baseline_val_pred = data.model_predictions[single_best]["val"]
    baseline_test_pred = data.model_predictions[single_best]["test"]
    baseline_metrics = _point_metrics(data.y_test, baseline_test_pred, peak_mask)

    # Analysis 1: VAL-calibrated ensembles.
    nnls_weights, nnls_residual_norm = nnls(x_val, data.y_val, maxiter=10_000)
    nnls_val_pred = _weighted_prediction(x_val, nnls_weights)
    nnls_test_pred = _weighted_prediction(x_test, nnls_weights)
    nnls_metrics = _point_metrics(data.y_test, nnls_test_pred, peak_mask)
    nnls_val_mae = float(np.mean(np.abs(data.y_val - nnls_val_pred)))

    caruana_weights, caruana_selected, caruana_val_mae = _caruana_selection(
        x_val, data.y_val, model_names
    )
    caruana_test_pred = _weighted_prediction(x_test, caruana_weights)
    caruana_metrics = _point_metrics(data.y_test, caruana_test_pred, peak_mask)

    baseline_entry = backtest_models[single_best]
    baseline_test_json = _point_arrays(baseline_entry, "test_points")
    required_pi_fields = {"actual", "predicted", "lower95", "upper95"}
    if not required_pi_fields.issubset(baseline_test_json):
        missing = sorted(required_pi_fields.difference(baseline_test_json))
        raise ValueError(f"Single-best test_points cannot confirm original PI: missing {missing}.")
    original_coverage = _coverage(
        baseline_test_json["actual"],
        baseline_test_json["lower95"],
        baseline_test_json["upper95"],
    )
    original_interval_width = float(
        np.mean(baseline_test_json["upper95"] - baseline_test_json["lower95"])
    )
    baseline_requested_wis = _requested_interval_score(
        baseline_test_json["actual"],
        baseline_test_json["predicted"],
        baseline_test_json["lower95"],
        baseline_test_json["upper95"],
    )
    dashboard_wis = float(baseline_entry.get("metrics", {}).get("wis", np.nan))

    # Analysis 2: VAL-to-TEST bias correction.
    linear = LinearRegression().fit(baseline_val_pred.reshape(-1, 1), data.y_val)
    linear_test_pred = linear.predict(baseline_test_pred.reshape(-1, 1))
    linear_metrics = _point_metrics(data.y_test, linear_test_pred, peak_mask)
    additive_bias = float(np.mean(data.y_val - baseline_val_pred))
    additive_test_pred = baseline_test_pred + additive_bias
    additive_metrics = _point_metrics(data.y_test, additive_test_pred, peak_mask)

    # Analysis 3: conformal PI recalibration.
    val_residuals = np.abs(data.y_val - baseline_val_pred)
    test_residuals = np.abs(data.y_test - baseline_test_pred)
    val_q95 = _quantile_higher(val_residuals, 0.95)
    conformal_coverage = float(np.mean(test_residuals <= val_q95))
    required_threshold, required_coverage = _minimum_threshold_for_coverage(
        test_residuals, 0.90
    )
    required_width_multiplier = (2.0 * required_threshold) / original_interval_width

    top10_names = [name for name in backtest_models if name in data.model_predictions]
    pooled_val_residuals = np.concatenate(
        [
            np.abs(data.y_val - data.model_predictions[name]["val"])
            for name in top10_names
        ]
    )
    pooled_q95 = _quantile_higher(pooled_val_residuals, 0.95)
    pooled_coverage = float(np.mean(test_residuals <= pooled_q95))
    tradeoff = [
        (
            level,
            _quantile_higher(val_residuals, level),
            float(np.mean(test_residuals <= _quantile_higher(val_residuals, level))),
        )
        for level in QUANTILE_LEVELS
    ]

    # Analysis 4: observed-lag mechanistic trend blend on rising weeks.
    rising_mask = np.zeros(len(data.y_test), dtype=bool)
    for index in range(2, len(data.y_test)):
        if (
            data.y_test[index - 2] != 0
            and data.y_test[index] > data.y_test[index - 1]
            and data.y_test[index] > 1.2 * data.y_test[index - 2]
        ):
            rising_mask[index] = True
    if not rising_mask.any():
        raise RuntimeError("No rising weeks satisfy the requested definition.")
    rising_indices = np.flatnonzero(rising_mask)
    seir_trend = (
        data.y_test[rising_indices - 1]
        * data.y_test[rising_indices - 1]
        / data.y_test[rising_indices - 2]
    )
    oracle_blend_pred = 0.4 * seir_trend + 0.6 * baseline_test_pred[rising_indices]
    rising_baseline_mae = float(
        np.mean(np.abs(data.y_test[rising_indices] - baseline_test_pred[rising_indices]))
    )
    rising_blend_mae = float(
        np.mean(np.abs(data.y_test[rising_indices] - oracle_blend_pred))
    )
    rising_gain = rising_baseline_mae - rising_blend_mae
    rising_gain_pct = rising_gain / rising_baseline_mae if rising_baseline_mae else np.nan

    ensemble_candidates = {"NNLS": nnls_metrics, "Caruana": caruana_metrics}
    correction_candidates = {"linear": linear_metrics, "additive": additive_metrics}
    deployable_candidates = {
        "NNLS": nnls_metrics,
        "Caruana": caruana_metrics,
        "bias-linear": linear_metrics,
        "bias-additive": additive_metrics,
    }
    all_round = [
        name
        for name, metrics in deployable_candidates.items()
        if metrics.mae < baseline_metrics.mae
        and metrics.rmse < baseline_metrics.rmse
        and metrics.peak_mae < baseline_metrics.peak_mae
    ]
    best_overall_name, best_overall_metrics = min(
        deployable_candidates.items(), key=lambda item: item[1].mae
    )
    best_peak_name, best_peak_metrics = min(
        deployable_candidates.items(), key=lambda item: item[1].peak_mae
    )

    val_range = backtest_meta.get("val_date_range", {})
    test_range = backtest_meta.get("test_date_range", {})
    lines = [
        "# Seoul ILI Web-Side Post-Processing Analysis",
        "",
        (
            f"- Inputs: {len(data.model_predictions)} valid prediction files; "
            f"VAL n={len(data.y_val)} ({val_range.get('start', 'N/A')} to "
            f"{val_range.get('end', 'N/A')}); TEST n={len(data.y_test)} "
            f"({test_range.get('start', 'N/A')} to {test_range.get('end', 'N/A')})."
        ),
        (
            f"- Single-best resolved from summary_metrics.csv: **{single_best}** "
            f"(summary TEST MAE={float(summary_row['test_mae']):.4f}, "
            f"RMSE={float(summary_row['test_rmse']):.4f})."
        ),
        (
            f"- Peak weeks: y_true >= TEST 75th percentile ({peak_threshold:.4f}); "
            f"n={int(peak_mask.sum())}."
        ),
        f"- Skipped malformed prediction files: {len(data.warnings)}.",
        "",
        "## 1. Summary",
        "",
        "| Method | Overall-MAE | Overall-RMSE | Peak-MAE | Coverage |",
        "|---|---:|---:|---:|---:|",
        _metrics_row("Single-best-baseline", baseline_metrics, original_coverage),
        _metrics_row("NNLS-ensemble", nnls_metrics, None),
        _metrics_row("Caruana-ensemble", caruana_metrics, None),
        _metrics_row("Bias-corrected-linear", linear_metrics, None),
        _metrics_row("Bias-corrected-additive", additive_metrics, None),
        _metrics_row("Conformal-recalibrated", baseline_metrics, conformal_coverage),
        "",
        "## 2. Analysis 1 - VAL-Calibrated Ensemble Weights",
        "",
        (
            f"- Single-best {single_best}: TEST MAE={baseline_metrics.mae:.4f}, "
            f"RMSE={baseline_metrics.rmse:.4f}, peak MAE={baseline_metrics.peak_mae:.4f}; "
            f"dashboard multi-interval WIS={dashboard_wis:.4f}; requested 95%-interval "
            f"score={baseline_requested_wis:.4f}."
        ),
        (
            f"- NNLS: VAL MAE={nnls_val_mae:.4f}, residual norm={nnls_residual_norm:.4f}, "
            f"weight sum={nnls_weights.sum():.4f}, nonzero weights="
            f"{int(np.sum(nnls_weights > 1e-10))}; TEST MAE={nnls_metrics.mae:.4f}, "
            f"RMSE={nnls_metrics.rmse:.4f}, point-only WIS={nnls_metrics.mae:.4f}, "
            f"peak MAE={nnls_metrics.peak_mae:.4f}."
        ),
        f"- NNLS top weights: {_top_weights(model_names, nnls_weights)}.",
        (
            f"- Caruana: VAL MAE={caruana_val_mae:.4f}, selections="
            f"{len(caruana_selected)}; TEST MAE={caruana_metrics.mae:.4f}, "
            f"RMSE={caruana_metrics.rmse:.4f}, point-only WIS={caruana_metrics.mae:.4f}, "
            f"peak MAE={caruana_metrics.peak_mae:.4f}."
        ),
        f"- Caruana selection sequence: {', '.join(caruana_selected)}.",
        (
            "- Verdict: "
            + _comparison_verdict(baseline_metrics, ensemble_candidates, "peak_mae")
            + " VAL is summer-only, so this is direct evidence about winter-peak transfer."
        ),
        "",
        "## 3. Analysis 2 - Post-Hoc Bias Correction",
        "",
        (
            f"- Before correction: TEST MAE={baseline_metrics.mae:.4f}, "
            f"RMSE={baseline_metrics.rmse:.4f}, peak MAE={baseline_metrics.peak_mae:.4f}."
        ),
        (
            f"- Linear correction: y={float(linear.intercept_):.4f} + "
            f"{float(linear.coef_[0]):.4f}*prediction; TEST MAE={linear_metrics.mae:.4f}, "
            f"RMSE={linear_metrics.rmse:.4f}, peak MAE={linear_metrics.peak_mae:.4f}."
        ),
        (
            f"- Additive correction: add {additive_bias:.4f}; TEST MAE="
            f"{additive_metrics.mae:.4f}, RMSE={additive_metrics.rmse:.4f}, "
            f"peak MAE={additive_metrics.peak_mae:.4f}."
        ),
        (
            "- Verdict: "
            + _comparison_verdict(baseline_metrics, correction_candidates, "peak_mae")
            + " This tests summer-VAL correction against winter peaks."
        ),
        "",
        "## 4. Analysis 3 - Conformal PI Recalibration",
        "",
        (
            f"- Original dashboard 95% PI: confirmed coverage={original_coverage:.4f} "
            f"({int(round(original_coverage * len(data.y_test)))}/{len(data.y_test)}), "
            f"mean full width={original_interval_width:.4f}."
        ),
        (
            f"- Single-best VAL residual q95={val_q95:.4f}: symmetric full width="
            f"{2.0 * val_q95:.4f}, TEST coverage={conformal_coverage:.4f}."
        ),
        (
            f"- Narrowest TEST-aware threshold reaching >=90%: half-width="
            f"{required_threshold:.4f}, full width={2.0 * required_threshold:.4f}, "
            f"achieved coverage={required_coverage:.4f}, width multiplier vs original="
            f"{required_width_multiplier:.2f}x. This is an oracle diagnostic, not a "
            f"deployable VAL-only calibration."
        ),
        (
            f"- Top-10 pooled VAL residual q95 (n={len(pooled_val_residuals)}): "
            f"half-width={pooled_q95:.4f}, full width={2.0 * pooled_q95:.4f}, "
            f"TEST coverage around {single_best}={pooled_coverage:.4f}; width vs "
            f"single-best VAL q95={(pooled_q95 / val_q95):.2f}x."
        ),
        "",
        "| VAL residual quantile | Half-width | Full width | TEST coverage |",
        "|---:|---:|---:|---:|",
    ]
    lines.extend(
        f"| {level:.2f} | {threshold:.4f} | {2.0 * threshold:.4f} | {coverage:.4f} |"
        for level, threshold, coverage in tradeoff
    )

    if pooled_coverage > conformal_coverage:
        pooling_verdict = (
            f"Pooling raises coverage by {pooled_coverage - conformal_coverage:.4f}, "
            f"with a {(pooled_q95 / val_q95):.2f}x half-width."
        )
    elif pooled_coverage < conformal_coverage:
        pooling_verdict = (
            f"Pooling lowers coverage by {conformal_coverage - pooled_coverage:.4f}, "
            f"with a {(pooled_q95 / val_q95):.2f}x half-width."
        )
    else:
        pooling_verdict = (
            f"Pooling leaves coverage unchanged, with a {(pooled_q95 / val_q95):.2f}x "
            f"half-width."
        )
    lines.extend(
        [
            "",
            (
                f"- Verdict: VAL-q95 changes coverage from {original_coverage:.4f} to "
                f"{conformal_coverage:.4f}. {pooling_verdict} Reaching 90% on this TEST "
                f"set requires the oracle width reported above."
            ),
            "",
            "## 5. Analysis 4 - ABM/SEIR Blend Upper-Bound Estimate",
            "",
            (
                f"- Rising weeks matching the requested rule: n={len(rising_indices)}. "
                f"Single-best MAE={rising_baseline_mae:.4f}."
            ),
            (
                f"- 40% observed-lag mechanistic trend + 60% ML oracle blend: "
                f"rising-week MAE={rising_blend_mae:.4f}."
            ),
            (
                f"- Implied ceiling gain: {rising_gain:.4f} MAE "
                f"({_fmt_percent(rising_gain_pct)} relative)."
            ),
            (
                "- Verdict: "
                + (
                    "The oracle blend improves rising-week error, but the gain is an "
                    "upper-bound diagnostic using realized lagged outcomes."
                    if rising_gain > 0
                    else "Even the oracle blend does not improve rising-week error."
                )
            ),
            "",
            "## 6. Web-Side Verdict",
            "",
        ]
    )

    if all_round:
        lines.append(
            "- Techniques improving overall MAE, RMSE, and peak MAE together: "
            + ", ".join(all_round)
            + "."
        )
    else:
        lines.append(
            "- No deployable VAL-learned ensemble or bias correction improves overall "
            "MAE, RMSE, and peak MAE together versus the single-best baseline."
        )
    lines.extend(
        [
            (
                f"- Best deployable overall MAE is {best_overall_name}="
                f"{best_overall_metrics.mae:.4f} versus baseline={baseline_metrics.mae:.4f}; "
                f"best deployable peak MAE is {best_peak_name}={best_peak_metrics.peak_mae:.4f} "
                f"versus baseline={baseline_metrics.peak_mae:.4f}."
            ),
            (
                f"- VAL-only conformal recalibration yields {conformal_coverage:.4f} "
                f"coverage at full width {2.0 * val_q95:.4f}; >=90% coverage requires "
                f"{required_width_multiplier:.2f}x the original dashboard PI width in "
                f"this TEST-aware diagnostic."
            ),
            (
                f"- Without retraining, the observed-lag ABM/SEIR blend ceiling on rising "
                f"weeks is a {rising_gain:.4f} MAE gain ({_fmt_percent(rising_gain_pct)})."
            ),
        ]
    )

    print("\n".join(lines))


if __name__ == "__main__":
    main()
