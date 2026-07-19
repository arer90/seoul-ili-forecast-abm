"""Unified, honest metric table for the forecasting models.

Consolidates the SSOT per-model metrics (phase-14 ``evaluate_predictions_full``
output) with an **extrapolation-stability flag** so that no single — possibly
cherry-picked — number stands alone. The flag catches models like ElasticNet
that blow up out of range on a held-out window (predicting ~3.5x the observed
maximum), which posts a catastrophic R^2 on one window while looking fine on
another. That instability — not a metric-computation bug — is the source of the
OOF-vs-test inconsistency, so the unified table makes it explicit.

The metric values themselves are NOT recomputed here (the SSOT is
``phase_evaluator.evaluate_predictions_full``); this module only joins them with
the stability diagnostic and labels the evaluation basis.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

# A point prediction beyond this multiple of the observed maximum is flagged as
# extrapolation-unstable: the model leaves the support of the training data.
STABILITY_RATIO_CEILING = 1.5


def compute_stability(predictions_csv: str | Path) -> dict[str, dict]:
    """Per-model extrapolation-stability diagnostic from point predictions.

    Args:
        predictions_csv: a ``predictions_per_model.csv`` with columns
            ``model``, ``y_true``, ``y_pred``.

    Returns:
        ``{model: {"max_pred", "max_obs", "extrapolation_ratio", "stable"}}``.
        ``stable`` is False when ``max_pred > STABILITY_RATIO_CEILING * max_obs``.

    Side effects: none.
    """
    rows = list(csv.DictReader(open(predictions_csv, encoding="utf-8")))
    by_model: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    out: dict[str, dict] = {}
    for model, rs in by_model.items():
        preds = [float(r["y_pred"]) for r in rs]
        obs = [float(r["y_true"]) for r in rs]
        max_pred = max(preds) if preds else 0.0
        max_obs = max(obs) if obs else 0.0
        ratio = (max_pred / max_obs) if max_obs > 0 else float("inf")
        out[model] = {
            "max_pred": round(max_pred, 3),
            "max_obs": round(max_obs, 3),
            "extrapolation_ratio": round(ratio, 3),
            "stable": bool(ratio <= STABILITY_RATIO_CEILING),
        }
    return out


def _fnum(row: dict, key: str) -> Optional[float]:
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return None


def build_unified_table(
    metrics_csv: str | Path,
    predictions_csv: str | Path,
    *,
    eval_basis: str = "post-optimization test hold-out (phase 14, n=68 weeks)",
) -> dict:
    """Join SSOT per-model metrics with the stability flag into one honest table.

    Args:
        metrics_csv: phase-14 ``per_model_metrics.csv`` (SSOT 134-metric output).
        predictions_csv: matching ``predictions_per_model.csv`` for stability.
        eval_basis: human-readable label of what these metrics are measured on,
            carried into the output so the numbers are never quoted context-free.

    Returns:
        ``{"eval_basis", "rows", "n_models", "n_unstable"}`` where ``rows`` is a
        WIS-sorted list of ``{model, wis, r2, mape, pi95_coverage, rank_wis,
        extrapolation_ratio, stable}`` dicts. Unstable models keep their numbers
        but are flagged so they are not silently ranked against stable ones.
    """
    metrics = list(csv.DictReader(open(metrics_csv, encoding="utf-8")))
    stability = compute_stability(predictions_csv)
    rows: list[dict] = []
    for m in metrics:
        model = m.get("model", "?")
        st = stability.get(model, {})
        rows.append({
            "model": model,
            "wis": _fnum(m, "wis"),
            "r2": _fnum(m, "r2"),
            "mape": _fnum(m, "mape"),
            "pi95_coverage": _fnum(m, "pi95_coverage"),
            "rank_wis": _fnum(m, "rank_wis"),
            "extrapolation_ratio": st.get("extrapolation_ratio"),
            "stable": st.get("stable"),
        })
    rows.sort(key=lambda r: (r["wis"] if r["wis"] is not None else float("inf")))
    return {
        "eval_basis": eval_basis,
        "rows": rows,
        "n_models": len(rows),
        "n_unstable": sum(1 for r in rows if r["stable"] is False),
        "pi95_note": (
            "PI95 coverage is conformal-calibrated to ~nominal 95%, so most "
            "models land at ~66/68=0.971; it does not discriminate model quality."
        ),
    }


def write_unified_table(table: dict, output_path: str | Path) -> None:
    """Write the unified table to JSON (sorted keys, utf-8)."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(table, fh, indent=2, sort_keys=True, ensure_ascii=False)
