"""
P2: Inference-on-new-data using saved champion `.pt` artifacts.
======================================================================

After R9 promotes a champion `ChampionArtifact` (model + fitted
scaler + transform state + feature_indices) into ``models/<name>.pt``,
this phase loads the bundle and predicts on NEW real-world data — the
operational deployment flow.

Use cases:
  1. Weekly KDCA data lands → re-collect via ``simulation collect`` →
     predict next-week ILI rate with all champions:
         simulation predict-real --weeks-ahead 4
  2. Hold-out validation: re-load champions, predict on a date window
     models never saw, evaluate against actuals (computes the full
     forecasting metric stack — WIS / MAE / RMSE / R²).
  3. Cross-config comparison: load champions from different
     (covid_mode, weather_mode) runs and compare on the same window.

Artifact protocol
-----------------
A champion `.pt` produced by R9 is a pickle of
``simulation.utils.model_artifact.ChampionArtifact``:

    artifact.predict(X_full) →
        X_sub  = artifact.apply_features(X_full)         # by feature_indices
        X_scl  = artifact.apply_scaler(X_sub)            # fitted scaler.transform
        y_t    = artifact.predict_raw(X_scl)             # model.predict
        y_pred = artifact.inverse_transform_target(y_t)  # boxcox λ / PT.inverse / log1p / id

The pipeline therefore replays *exactly* the training-time chain on
inference X, with no re-fitting (which would leak inference statistics).

Legacy bare-model `.pt` files are still loadable via
``model_artifact.load_artifact`` — they get wrapped in an identity-
transform artifact and a WARN is logged.

This module is loaded by:
  • ``simulation/__main__.py predict-real`` subcommand for ad-hoc inference
  • ``simulation/pipeline/runner.py`` P2 wire-in (when target dates
    extend past the existing test slab)

Output:

    simulation/results/inference/<run_id>/
      ├── predictions.csv            ← date × model × {pred, actual?}
      ├── inference_metrics.json     ← if actuals available
      ├── champions_used.json        ← {model: {version, config, transform,
      │                                          scaler, n_features, ...}}
      └── REPORT.md
"""
from __future__ import annotations
from simulation.config_global import GLOBAL  # SSOT (2026-05-28)

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger(__name__)


def predict_with_champion(
    model_name: str,
    artifact,                     # ChampionArtifact
    X: np.ndarray,
) -> Optional[np.ndarray]:
    """End-to-end inference using a ChampionArtifact bundle.

    The artifact replays the exact training-time chain — feature subset,
    fitted scaler.transform, model.predict, inverse target transform —
    so callers just hand the full feature matrix in.
    """
    if artifact is None or X is None or len(X) == 0:
        return None
    try:
        return np.asarray(artifact.predict(X), dtype=np.float64)
    except Exception as e:
        log.error(f"  [phase17] {model_name}.predict failed: {e}")
        return None


def run_inference(
    X_inference: np.ndarray,
    inference_dates: Optional[np.ndarray] = None,
    actuals: Optional[np.ndarray] = None,
    *,
    model_names: Optional[list[str]] = None,
    models_dir: Path = Path("models"),
    log_path: Path = Path("models/champion_log.json"),
    out_dir: Optional[Path] = None,
) -> dict:
    """Predict on NEW data using champion `.pt` artifacts.

    Args:
      X_inference: (N, n_features) full feature matrix for the inference
        window. Must be in the same column order as training (the artifact
        applies its own feature_indices subset).
      inference_dates: optional (N,) datetime array for output column.
      actuals: optional (N,) ground truth — if provided, computes metrics.
      model_names: which champions to use (default: all in champion_log.json).
      models_dir: ``models/`` directory (default: 'models').
      log_path: ``champion_log.json`` path.
      out_dir: where to write predictions.csv / report.md.

    Returns: {predictions, metrics_per_model (if actuals),
              champions_used, report_path, out_dir, elapsed}.
    """
    t0 = time.time()
    from simulation.utils.champion_log import ChampionLog
    from simulation.utils.model_artifact import load_artifact

    cl = ChampionLog(models_dir=models_dir, log_path=log_path)
    available = cl.summary()

    if model_names is None:
        model_names = sorted(available.keys())
    elif not model_names:
        log.warning("  [phase17] no model_names provided")
        return {"skipped": True, "reason": "no model_names",
                "elapsed": time.time() - t0}

    # Filter to those with current champion
    model_names = [n for n in model_names if n in available]
    if not model_names:
        log.warning("  [phase17] no champions match the requested model_names")
        return {"skipped": True, "reason": "no champions available",
                "elapsed": time.time() - t0}

    log.info(f"  [phase17] inference using {len(model_names)} champions: "
             f"{sorted(model_names)}")

    # ── Load and predict ──
    predictions: dict[str, np.ndarray] = {}
    champions_used: dict[str, dict] = {}
    for name in model_names:
        cur = available[name]
        # Q5 / G-276: 운영 inference 는 전체-데이터 재학습 _deploy.pt 우선 (없으면 eval .pt).
        #   _deploy = train+val+test+real fit (최신 관측 반영), eval .pt = hold-out metric 용.
        _deploy_path = Path(models_dir) / f"{name}_deploy.pt"
        pt_path = _deploy_path if _deploy_path.exists() else Path(models_dir) / f"{name}.pt"
        artifact = load_artifact(pt_path)
        if artifact is None:
            log.warning(f"  [phase17] {name}: champion file unloadable, skip")
            continue
        if pt_path is _deploy_path:
            log.info(f"  [phase17] {name}: using deploy artifact (full-data refit)")
        yp = predict_with_champion(name, artifact, X_inference)
        if yp is None or len(yp) != len(X_inference):
            log.warning(f"  [phase17] {name} prediction length mismatch (skip)")
            continue
        predictions[name] = yp
        # Roll forward useful metadata for the report
        art_summary = artifact.summary()
        champions_used[name] = {
            "version": cur.get("current_version"),
            "test_wis_at_promotion": cur.get("current_test_wis"),
            "test_mae_at_promotion": cur.get("current_test_mae"),
            "transform":          art_summary.get("transform_name"),
            "scaler":             art_summary.get("scaler_class"),
            "n_features_used":    art_summary.get("n_features_used"),
            "config":             art_summary.get("config"),
            "promoted_at":        cur.get("promoted_at"),
            "is_legacy":          bool(art_summary.get("config", {}).get("legacy", False)),
        }

    if not predictions:
        log.warning("  [phase17] every champion failed to produce predictions")
        return {"skipped": True, "reason": "all predictions failed",
                "elapsed": time.time() - t0}

    # ── Optional metrics if actuals provided ──
    # Reports BOTH aggregate (over all horizons) and PER-HORIZON metrics.
    # Per-horizon (h=1, h=2, …) is the operationally meaningful view —
    # h=1 (next-week) is what KDCA / public-health uses for weekly alerts;
    # later horizons naturally degrade due to compounding uncertainty.
    metrics_per_model: dict[str, dict] = {}
    if actuals is not None and len(actuals) == len(X_inference):
        from simulation.analytics.diagnostics import weighted_interval_score
        from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
        for name, yp in predictions.items():
            err = yp - actuals
            mask = np.isfinite(actuals) & np.isfinite(yp)
            if not mask.any():
                continue
            ae = np.abs(err[mask])
            sigma = float(np.std(err[mask])) or 1.0
            sse = float(np.sum(err[mask] ** 2))
            sst = float(np.sum((actuals[mask] - actuals[mask].mean()) ** 2))
            try:
                wis_agg = float(np.mean(weighted_interval_score(
                    actuals[mask], yp[mask], sigma, alphas=FLUSIGHT_ALPHAS)))
            except Exception:
                wis_agg = float("nan")

            # Per-horizon (h=1..N) breakdown
            n_horizons = len(actuals)
            per_h: dict[str, dict] = {}
            for h in range(n_horizons):
                if not (np.isfinite(actuals[h]) and np.isfinite(yp[h])):
                    continue
                err_h = float(yp[h] - actuals[h])
                ae_h = float(abs(err_h))
                # WIS for a single point: use a naive σ (in-sample residual std)
                try:
                    wis_h = float(weighted_interval_score(
                        np.array([actuals[h]]), np.array([yp[h]]),
                        sigma, alphas=FLUSIGHT_ALPHAS).item())
                except Exception:
                    wis_h = float("nan")
                per_h[f"h{h+1}"] = {
                    "actual": float(actuals[h]),
                    "pred":   float(yp[h]),
                    "ae":     ae_h,
                    "ape":    (ae_h / float(actuals[h]) * 100.0
                                if actuals[h] != 0 else float("nan")),
                    "wis":    wis_h,
                }

            metrics_per_model[name] = {
                "n_horizons": int(mask.sum()),
                "aggregate": {
                    "mae":  float(np.mean(ae)),
                    "rmse": float(np.sqrt(np.mean((err[mask]) ** 2))),
                    "r2":   (1.0 - sse / sst) if sst > 0 else float("nan"),
                    "wis":  wis_agg,
                },
                "per_horizon": per_h,
                # Back-compat: keep flat keys for older readers
                "n":    int(mask.sum()),
                "mae":  float(np.mean(ae)),
                "rmse": float(np.sqrt(np.mean((err[mask]) ** 2))),
                "r2":   (1.0 - sse / sst) if sst > 0 else float("nan"),
                "wis":  wis_agg,
            }
            # 2026-05-28 사용자 명시 R3: R8 evaluator (134 metric) on P2 predictions
            try:
                from simulation.pipeline.phase_evaluator import evaluate_predictions_full
                _full_r8 = evaluate_predictions_full(
                    y_test=actuals[mask], y_pred=yp[mask],
                    residuals=actuals[mask] - yp[mask],
                    sigma=sigma, y_train_pool=None,
                    threshold=GLOBAL.filter.alert_threshold, phase_id=f"Pinf_{name}",
                )
                # Merge into metrics_per_model[name] — preserve existing keys
                for _k, _v in _full_r8.items():
                    if _k not in metrics_per_model[name] and not _k.startswith("_"):
                        metrics_per_model[name][_k] = _v
            except Exception as _r8_err:
                log.warning(f"  [phase17] R8 skip {name}: {_r8_err}")

    # ── Persist ──
    if out_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
        out_dir = get_results_dir() / "inference" / ts
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV: date, actual?, model1, model2, ...
    import csv
    cols = ["date"] + (["actual"] if actuals is not None else []) + sorted(predictions.keys())
    with (out_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(len(X_inference)):
            row = [str(inference_dates[i]) if inference_dates is not None else i]
            if actuals is not None:
                row.append(float(actuals[i]) if np.isfinite(actuals[i]) else "")
            for nm in sorted(predictions.keys()):
                v = predictions[nm][i]
                row.append(float(v) if np.isfinite(v) else "")
            w.writerow(row)

    (out_dir / "champions_used.json").write_text(
        json.dumps(champions_used, indent=2, default=str))

    if metrics_per_model:
        (out_dir / "inference_metrics.json").write_text(
            json.dumps(metrics_per_model, indent=2, default=str))

    # ── Markdown report ──
    md = [
        "# P2 — Inference on new data using champion `.pt` artifacts",
        "",
        f"- Models used: {len(predictions)}",
        f"- Inference window: n = {len(X_inference)} weeks",
    ]
    if inference_dates is not None and len(inference_dates) > 0:
        md.append(f"- Date range: {inference_dates[0]} → {inference_dates[-1]}")
    md.append("")
    md.append("Each champion ships its **fitted scaler** and **transform state** "
              "(boxcox λ / PowerTransformer / log1p), so inference replays the "
              "exact training-time pipeline without re-fitting on inference X.")
    md.append("")
    if metrics_per_model:
        md += ["## Aggregate inference metrics (all horizons combined)",
               "",
               "| model | n | WIS | MAE | RMSE | R² |",
               "|---|---|---|---|---|---|"]
        for nm in sorted(metrics_per_model, key=lambda k: metrics_per_model[k]["wis"]):
            m = metrics_per_model[nm]
            md.append(
                f"| {nm} | {m['n']} | {m['wis']:.3f} | {m['mae']:.3f} | "
                f"{m['rmse']:.3f} | {m['r2']:.3f} |"
            )
        md.append("")

        # Per-horizon table — h=1 is the operationally meaningful one
        md += ["## Per-horizon breakdown (h=1 = next-week, primary KPI)",
               "",
               "Note: forecasting accuracy *naturally* degrades over horizons "
               "(compounding uncertainty). h=1 is what KDCA weekly alerts use; "
               "later horizons are scenario-planning only."]
        md.append("")
        # Build column header from the first model's per_horizon keys
        first_per_h = next(iter(metrics_per_model.values())).get("per_horizon", {})
        if first_per_h:
            horizons = sorted(first_per_h.keys(),
                                key=lambda k: int(k.replace("h", "")))
            # AE table
            md.append("### Absolute error per horizon (lower = better)")
            md.append("")
            md.append(f"| model | {' | '.join(horizons)} |")
            md.append("|---|" + "---|" * len(horizons))
            for nm in sorted(metrics_per_model.keys()):
                m = metrics_per_model[nm]
                ph = m.get("per_horizon", {})
                row_vals = []
                for h in horizons:
                    ae = ph.get(h, {}).get("ae", float("nan"))
                    row_vals.append(f"{ae:.2f}" if np.isfinite(ae) else "?")
                md.append(f"| {nm} | {' | '.join(row_vals)} |")
            md.append("")
            # Pred vs actual
            md.append("### Prediction vs actual per horizon")
            md.append("")
            md.append(f"| model | metric | {' | '.join(horizons)} |")
            md.append("|---|---|" + "---|" * len(horizons))
            actual_row = [str(first_per_h.get(h, {}).get("actual", "?"))
                          for h in horizons]
            md.append(f"| _ground truth_ | actual | {' | '.join(actual_row)} |")
            for nm in sorted(metrics_per_model.keys()):
                m = metrics_per_model[nm]
                ph = m.get("per_horizon", {})
                preds = []
                for h in horizons:
                    p = ph.get(h, {}).get("pred", float("nan"))
                    preds.append(f"{p:.2f}" if np.isfinite(p) else "?")
                md.append(f"| {nm} | pred | {' | '.join(preds)} |")
            md.append("")
    md += ["## Champions used",
           "",
           "| model | version | test_WIS@promotion | promoted_at | "
           "transform | scaler | n_features |",
           "|---|---|---|---|---|---|---|"]
    for nm in sorted(champions_used.keys()):
        c = champions_used[nm]
        legacy = " (legacy)" if c.get("is_legacy") else ""
        md.append(
            f"| {nm}{legacy} | v{c['version']} | "
            f"{c.get('test_wis_at_promotion', '?')} | "
            f"{c.get('promoted_at', '?')} | "
            f"{c.get('transform') or '?'} | "
            f"{c.get('scaler') or 'none'} | "
            f"{c.get('n_features_used') or '?'} |"
        )
    md.append("")
    md.append("> **Legacy** flag: champion was a bare-model pickle (pre-artifact). "
              "Inference falls back to identity transform + no scaler — these "
              "predictions may differ from training-time pipeline. Re-run "
              "R9 to upgrade to a `ChampionArtifact`.")
    (out_dir / "REPORT.md").write_text("\n".join(md))

    log.info(f"  [phase17] wrote: {out_dir}")
    return {
        "predictions": {k: v.tolist() for k, v in predictions.items()},
        "metrics_per_model": metrics_per_model,
        "champions_used": champions_used,
        "out_dir": str(out_dir),
        "report_path": str(out_dir / "REPORT.md"),
        "elapsed": time.time() - t0,
    }


# back-compat aliases (2026-06-02 semantic rename — 옛 run_phaseN)
run_phase17 = run_inference
