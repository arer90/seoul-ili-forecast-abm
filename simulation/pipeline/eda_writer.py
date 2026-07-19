"""Phase-level EDA sidecar writer (2026-05-26, Sprint 3).

Non-fatal helper that every phase calls at the end of its run. Writes a
standardized triplet to ``simulation/results/eda/phase{NN}_<tag>/``:

  predictions_per_model.csv  — (week_idx, model, y_true, y_pred, residual)
  metrics_summary.json       — point metrics + per-model status
  issues.md                  — auto-flagged anomalies (NaN, low R², high MAPE,
                                residual ACF, outliers)

Design (D-4 deep module):
  Single public function ``write_phase_eda`` that wraps the entire IO + issue
  detection pipeline. Caller passes prediction arrays + names; helper handles
  shape validation, metric computation, atomic write, and issue detection.

Safety (Codex § 3.3):
  - Wrapped in try/except — never raises, returns ``False`` on failure.
  - Atomic write via ``tmp`` file + ``Path.replace()`` — partial files cannot
    corrupt downstream readers if the process is killed mid-write.
  - No-op when ``MPH_DISABLE_EDA_SIDECAR=1`` (escape hatch for tight loops).

Usage:
    from simulation.pipeline.eda_writer import write_phase_eda
    write_phase_eda(
        phase_id=2,
        phase_tag="baseline",
        y_true=y_val,
        predictions={"DNN": y_pred_dnn, "LightGBM": y_pred_lgb, ...},
        save_dir=Path("simulation/results/eda"),
    )

Performance: O(n_models × n_weeks) per call. Typical per-model-eval /
comprehensive (R10/R12) call = ~50ms for 53 models × 68 weeks.
Side effects: creates ``save_dir/phase{NN}_<tag>/`` + 3 files inside.
Caller responsibility: ``predictions`` dict values must all have ``len() == len(y_true)``.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from simulation.config_global import GLOBAL  # SSOT (2026-05-28)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Issue detection rules
# ─────────────────────────────────────────────────────────────────────

DEFAULT_ISSUE_RULES = {
    "nan_threshold":     0.05,   # > 5% NaN → flag
    "r2_floor":          0.50,   # R² < 0.5  → flag (catastrophic if < 0)
    "mape_ceiling":      30.0,   # MAPE > 30% → flag
    "residual_acf_max":  0.30,   # |ACF lag-1| > 0.3 → autocorrelation flag
    "outlier_z":         3.0,    # |residual / σ| > 3 → outlier flag
}


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def write_phase_eda(
    phase_id: int,
    phase_tag: str,
    y_true: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    save_dir: Path | str | None = None,
    issue_rules: Optional[dict] = None,
    extra_meta: Optional[dict] = None,
) -> bool:
    """Non-fatal phase-level EDA sidecar.

    Args:
        phase_id: 0-15 phase number (used in directory name as zero-padded).
        phase_tag: short label (e.g. "baseline", "wfcv", "per_model_eval").
        y_true: shape (n_weeks,) ground truth.
        predictions: dict ``{model_name: y_pred_array}``. All arrays must have
            same length as ``y_true``.
        save_dir: parent EDA directory. Subdir is ``phase{NN:02d}_{tag}/``.
        issue_rules: override DEFAULT_ISSUE_RULES; missing keys fall back.
        extra_meta: extra dict merged into ``metrics_summary.json`` top-level.

    Returns:
        True on success, False if anything went wrong (including disabled).
        Never raises — long-running training paths can call freely.

    Performance: O(n_models × n_weeks). Atomic via tmp + Path.replace.
    Side effects: writes 3 files under ``save_dir/phase{NN}_<tag>/``.
    Caller responsibility: prediction arrays length-match y_true.
    """
    if GLOBAL.ops.disable_eda_sidecar:
        return False

    if save_dir is None:  # SSOT MPH_OUTPUT_ROOT (2026-05-29) — default routed via get_results_dir
        from simulation.utils.paths import get_results_dir
        save_dir = get_results_dir() / "eda"

    try:
        return _write_phase_eda_impl(
            phase_id=phase_id, phase_tag=phase_tag,
            y_true=y_true, predictions=predictions,
            save_dir=Path(save_dir),
            issue_rules={**DEFAULT_ISSUE_RULES, **(issue_rules or {})},
            extra_meta=extra_meta or {},
        )
    except Exception as e:
        log.warning(f"  [eda_writer] phase {phase_id}/{phase_tag} write failed: "
                    f"{type(e).__name__}: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────
# Implementation (inside try/except wrapper)
# ─────────────────────────────────────────────────────────────────────

def _write_phase_eda_impl(
    phase_id: int, phase_tag: str,
    y_true: np.ndarray, predictions: Mapping[str, np.ndarray],
    save_dir: Path, issue_rules: dict, extra_meta: dict,
) -> bool:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    n = len(y_true)
    if n == 0 or not predictions:
        log.debug(f"  [eda_writer] phase {phase_id}: nothing to write")
        return False

    out_dir = save_dir / f"phase{phase_id:02d}_{phase_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── per-model metrics ──
    per_model_metrics = {}
    for mname, y_pred in predictions.items():
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
        if len(y_pred) != n:
            per_model_metrics[mname] = {"error": f"length_mismatch ({len(y_pred)} vs {n})"}
            continue
        per_model_metrics[mname] = _compute_metrics(y_true, y_pred)

    # ── predictions CSV ──
    _write_predictions_csv(out_dir / "predictions_per_model.csv",
                           y_true=y_true, predictions=predictions)

    # ── metrics JSON ──
    metrics_doc = {
        "phase_id": phase_id,
        "phase_tag": phase_tag,
        "n_weeks": n,
        "n_models": len(predictions),
        "per_model": per_model_metrics,
        **extra_meta,
    }
    _atomic_write_json(out_dir / "metrics_summary.json", metrics_doc)

    # ── issues MD ──
    issues_md = _detect_issues(per_model_metrics, y_true, predictions, issue_rules)
    _atomic_write_text(out_dir / "issues.md", issues_md)

    log.info(f"  [eda_writer] phase {phase_id:02d}_{phase_tag}: wrote "
             f"{len(predictions)} models × {n} weeks → {out_dir}/")
    return True


# ─────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────

def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Point metrics + NaN counts.  Pure function."""
    finite = np.isfinite(y_pred)
    n_nan = int((~finite).sum())
    nan_frac = float(n_nan) / max(len(y_pred), 1)

    if not finite.any():
        return {"r2": float("nan"), "mae": float("inf"), "rmse": float("inf"),
                "mape_pct": float("inf"), "n_nan": n_nan, "nan_frac": nan_frac}

    yt = y_true[finite]
    yp = y_pred[finite]
    if len(yt) < 2:
        return {"r2": float("nan"), "mae": float("inf"), "rmse": float("inf"),
                "mape_pct": float("inf"), "n_nan": n_nan, "nan_frac": nan_frac}

    mae  = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    nz = yt != 0
    mape = (
        100.0 * float(np.mean(np.abs((yt[nz] - yp[nz]) / yt[nz])))
        if nz.any() else float("inf")
    )

    # residual lag-1 ACF
    resid = yt - yp
    if len(resid) >= 3 and float(np.std(resid)) > 1e-12:
        acf1 = float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
    else:
        acf1 = float("nan")

    return {
        "r2": r2, "mae": mae, "rmse": rmse, "mape_pct": mape,
        "n_nan": n_nan, "nan_frac": nan_frac, "resid_acf_lag1": acf1,
    }


# ─────────────────────────────────────────────────────────────────────
# Issue detection → markdown
# ─────────────────────────────────────────────────────────────────────

def _detect_issues(per_model_metrics: dict, y_true: np.ndarray,
                    predictions: Mapping[str, np.ndarray],
                    rules: dict) -> str:
    """Auto-flag anomalies per model.  Returns Markdown body."""
    lines = ["# Phase EDA issues", ""]
    any_issue = False

    for mname, m in per_model_metrics.items():
        flags = []
        if "error" in m:
            flags.append(f"**error**: `{m['error']}`")
        else:
            if m.get("nan_frac", 0) > rules["nan_threshold"]:
                flags.append(f"NaN: {m['nan_frac']:.1%} (> {rules['nan_threshold']:.0%})")
            r2 = m.get("r2", float("nan"))
            if np.isfinite(r2) and r2 < rules["r2_floor"]:
                marker = " catastrophic" if r2 < 0 else ""
                flags.append(f"R² = {r2:.3f} (< {rules['r2_floor']}){marker}")
            mape = m.get("mape_pct", float("inf"))
            if np.isfinite(mape) and mape > rules["mape_ceiling"]:
                flags.append(f"MAPE = {mape:.1f}% (> {rules['mape_ceiling']:.0f}%)")
            acf1 = m.get("resid_acf_lag1", float("nan"))
            if np.isfinite(acf1) and abs(acf1) > rules["residual_acf_max"]:
                flags.append(f"|ACF lag-1| = {abs(acf1):.2f} (> {rules['residual_acf_max']})")

            # outliers — count residuals beyond z*σ
            y_pred = np.asarray(predictions[mname], dtype=np.float64).ravel()
            if len(y_pred) == len(y_true):
                resid = y_true - y_pred
                resid_f = resid[np.isfinite(resid)]
                if len(resid_f) >= 3:
                    sig = float(np.std(resid_f))
                    if sig > 1e-12:
                        n_out = int((np.abs(resid_f) > rules["outlier_z"] * sig).sum())
                        if n_out > 0:
                            lines.append("")  # placeholder; outliers reported per-model only when found
                            flags.append(f"outliers: {n_out} (|z| > {rules['outlier_z']})")

        if flags:
            any_issue = True
            lines.append(f"- **{mname}**: {'; '.join(flags)}")

    if not any_issue:
        lines.append("_No issues detected (all models pass the auto rules)._")
    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────
# IO helpers (atomic tmp → rename)
# ─────────────────────────────────────────────────────────────────────

def _atomic_write_text(path: Path, content: str) -> None:
    """Write text atomically: write to tmp in same dir, then Path.replace.

    Same-dir guarantees the rename is atomic on POSIX. Caller must ensure
    ``path.parent`` exists.
    """
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.tmp.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        Path(tmp).replace(path)
    except Exception:
        try: Path(tmp).unlink()
        except FileNotFoundError: pass
        raise


def _atomic_write_json(path: Path, obj: dict) -> None:
    _atomic_write_text(path, json.dumps(obj, indent=2, default=str, ensure_ascii=False))


def _write_predictions_csv(path: Path, y_true: np.ndarray,
                            predictions: Mapping[str, np.ndarray]) -> None:
    """One CSV row per (week_idx, model).  Atomic write."""
    n = len(y_true)
    lines = ["week_idx,model,y_true,y_pred,residual"]
    for mname, y_pred in predictions.items():
        y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
        if len(y_pred) != n:
            continue
        for i in range(n):
            yt = y_true[i]
            yp = y_pred[i]
            r = yt - yp if np.isfinite(yp) else float("nan")
            lines.append(f"{i},{mname},{yt:.6f},{yp:.6f},{r:.6f}")
    _atomic_write_text(path, "\n".join(lines) + "\n")


__all__ = ["write_phase_eda", "DEFAULT_ISSUE_RULES"]
