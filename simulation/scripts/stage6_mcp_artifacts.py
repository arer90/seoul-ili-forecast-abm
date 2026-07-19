"""Stage 6a — Compile MCP artifact files so the 3 stubbed tools start returning
real data instead of ``status: not_available``.

Wires:
  - ``epi.forecast``       via ``stage3_forecasts.json``
  - ``epi.model_compare``  via ``stage4_dm_results.json``
  - ``epi.shap_features``  via ``stage3_shap/summary.json``

Data sources:
  - ``simulation/results/csv/predictions_*.csv``      (66 models × test-split)
  - ``simulation/results/post_E/pi_samples_wide.csv`` (per-horizon PI bands)
  - ``simulation/results/post_E_eval.json``           (WIS + ranking)
  - ``simulation/r_verification/results/03_dm_canonical.csv`` (pairwise DM)
  - ``simulation/results/checkpoints/checkpoint_phase8.json`` (R11 SHAP/XAI: MI + top-10 features)

Output dir:  ``simulation/results/``  (EpiMCPServer auto-finds)
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("stage6.artifacts")

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "simulation" / "results"

# Input-source dir (mutable via --src-dir). Reads fall back to RES if not
# overridden; outputs always go to RES so EpiMCPServer finds them.
SRC: Path = RES
SRC_CHECKPOINTS: Optional[Path] = None  # optional separate path for phase8 ckpt


def _src_path(rel: str) -> Path:
    """Resolve a relative input path against SRC with fallback to RES."""
    cand = SRC / rel
    if cand.exists():
        return cand
    return RES / rel


# ── stage3_forecasts.json ───────────────────────────────────────────────
def build_forecast_manifest():
    """Per-(gu, horizon, model_id) → forecast series. Seoul-aggregate only
    at this stage (model was trained on city-level ILI). Per-gu breakdowns
    will land when Stage 3b runs the panel forecaster."""
    eval_ = json.load(_src_path("post_E_eval.json").open(encoding="utf-8"))
    ranking = eval_["summary"]["ranking_by_probabilistic_score"]
    details = {d["model"]: d for d in eval_["details"]}

    pi_wide = pd.read_csv(_src_path("post_E/pi_samples_wide.csv"))
    pi_wide["week_start"] = pd.to_datetime(pi_wide["week_start"])

    out = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "schema_version": "0.1",
        "note": (
            "City-wide Seoul ILI forecasts from the Round-3 test split "
            "(69 weeks, 2024-12→2026-04). Per-gu per-horizon panel forecasts "
            "land in Stage 3b."
        ),
        "default_gu": "seoul_city",
        "horizons_available": [1],  # test-split is 1-step WF-CV
        "models_available": sorted(details.keys()),
        "top_20_by_wis": ranking,
        "forecasts": {},
    }

    # For each model in ranking, pull its PI row and produce a series record.
    CSV_DIR = _src_path("csv")
    for model in sorted(details.keys()):
        csv_path = CSV_DIR / f"predictions_{model}.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        test_df = df[df["split"] == "test"].copy()
        if test_df.empty:
            continue
        # Merge PI if available
        pi_model = pi_wide[pi_wide["model"] == model].copy()
        series = []
        for i, row in enumerate(test_df.itertuples(index=False)):
            rec = {
                "week_idx": int(row.idx) if hasattr(row, "idx") else i,
                "y_true": float(row.y_true),
                "y_pred": float(row.y_pred),
            }
            if not pi_model.empty and i < len(pi_model):
                prow = pi_model.iloc[i]
                rec.update({
                    "week_start": str(prow["week_start"].date()),
                    "pi_lo_95": float(prow["q025"]),
                    "pi_hi_95": float(prow["q975"]),
                    "pi_lo_90": float(prow["q050"]),
                    "pi_hi_90": float(prow["q950"]),
                })
            series.append(rec)
        det = details[model]
        out["forecasts"][model] = {
            "model_id": model,
            "gu": "seoul_city",
            "horizon": 1,
            "n_points": len(series),
            "wis": det.get("wis"),
            "wis_source": det.get("wis_source"),
            "crps_gaussian": det.get("crps_gaussian"),
            "pi_coverage_95": det.get("pi_coverage_95"),
            "series": series,
        }

    out_path = RES / "stage3_forecasts.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s (%d models, %d forecasts)", out_path,
             len(details), len(out["forecasts"]))


# ── stage4_dm_results.json ──────────────────────────────────────────────
def build_dm_manifest():
    """Pairwise Diebold-Mariano from R canonical. Regime-split."""
    dm = pd.read_csv(RES.parent / "r_verification" / "results" / "03_dm_canonical.csv")
    out = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "schema_version": "0.1",
        "source": "R forecast::dm.test (canonical Diebold-Mariano)",
        "regimes": sorted(dm["regime"].dropna().unique().tolist()),
        "default_metric": "mae",   # R's default: |e1| - |e2|
        "n_pairs": len(dm),
        "significance_threshold": 0.05,
        "pairs": [],
    }
    for _, r in dm.iterrows():
        out["pairs"].append({
            "model_a": r["model_a"],
            "model_b": r["model_b"],
            "regime": r["regime"],
            "n": int(r["n"]),
            "dm_stat": float(r["dm_stat"]) if pd.notna(r["dm_stat"]) else None,
            "dm_p": float(r["dm_p"]) if pd.notna(r["dm_p"]) else None,
            "better_model": r["better_model"] if pd.notna(r["better_model"]) else None,
            "significant": bool(pd.notna(r["dm_p"]) and float(r["dm_p"]) < 0.05),
        })

    # Per-model aggregate wins/losses across regimes
    wins: dict[str, int] = {}
    losses: dict[str, int] = {}
    for p in out["pairs"]:
        if not p["significant"] or p["better_model"] is None:
            continue
        loser = p["model_b"] if p["better_model"] == p["model_a"] else p["model_a"]
        wins[p["better_model"]] = wins.get(p["better_model"], 0) + 1
        losses[loser] = losses.get(loser, 0) + 1
    out["per_model_summary"] = {
        m: {"wins": wins.get(m, 0), "losses": losses.get(m, 0)}
        for m in sorted(set(wins) | set(losses))
    }

    out_path = RES / "stage4_dm_results.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s (%d pairs, %d models summarized)",
             out_path, len(out["pairs"]), len(out["per_model_summary"]))


# ── stage3_shap/summary.json ────────────────────────────────────────────
def build_shap_manifest():
    """Compile R11 SHAP/XAI MI ranking + per-model top-10 as a SHAP-proxy summary.

    Real SHAP was only computed for tree models (XGB, RF, GB) in the R11
    SHAP/XAI stage; for other models we expose MI-based ranking as the fallback.
    """
    # R11 SHAP/XAI checkpoint can live in a different backup than post_E artifacts
    ckpt_path = (
        SRC_CHECKPOINTS / "checkpoint_phase8.json"
        if SRC_CHECKPOINTS is not None
        else _src_path("checkpoints/checkpoint_phase8.json")
    )
    cp8 = json.load(ckpt_path.open(encoding="utf-8"))
    fi = cp8["data"]["feature_importance"]
    mi_ranking = fi.get("mi_ranking", [])
    per_model = fi.get("per_model_recommended", {})

    # Normalize mi_ranking to top-50 dicts
    mi_out = []
    for i, entry in enumerate(mi_ranking):
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            name, score = entry
            mi_out.append({"rank": i + 1, "feature": str(name), "mi_score": float(score)})
        elif isinstance(entry, dict):
            mi_out.append({"rank": i + 1, **entry})
        else:
            mi_out.append({"rank": i + 1, "feature": str(entry)})

    out_dir = RES / "stage3_shap"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "summary.json"

    # If a real-SHAP run has already populated this file, keep its per-model
    # block (source='TreeExplainer.shap_values on test split'). We only
    # refresh the global MI ranking and metadata.
    existing = {}
    if out_path.exists():
        try:
            existing = json.load(out_path.open(encoding="utf-8")) or {}
        except Exception:
            existing = {}
    has_real_shap = existing.get("shap_source", "").startswith("TreeExplainer")

    out = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "schema_version": "0.1",
        "note": (
            "Mutual-information global ranking (R11 SHAP/XAI) + real SHAP "
            "TreeExplainer values for 5 tree ensembles (if computed). "
            "Non-tree models surface MI order as a proxy."
            if has_real_shap else
            "Mutual-information global ranking (R11 SHAP/XAI). Per-model block "
            "currently surfaces MI as a proxy for all models — run "
            "post_E_real_shap.py to populate real SHAP for tree ensembles."
        ),
        "default_model": "XGBoost",
        "global_ranking_mi": mi_out,
    }
    if has_real_shap:
        # Preserve real-SHAP per-model rankings & metadata verbatim.
        for k in ("per_model_top_features", "per_model_full_rankings",
                  "models_with_real_shap", "models_with_mi_fallback",
                  "shap_source"):
            if k in existing:
                out[k] = existing[k]
    else:
        out["per_model_top_features"] = {
            model: [{"rank": i + 1, "feature": f} for i, f in enumerate(feats)]
            for model, feats in per_model.items()
        }
        out["models_with_real_shap"] = []
        out["models_with_mi_fallback"] = "all_models"

    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("wrote %s (mi_top=%d, per_model=%d)",
             out_path, len(mi_out), len(per_model))


# ── Entrypoint ──────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--src-dir",
        type=Path,
        default=None,
        help="Directory to read inputs from (post_E_eval.json, post_E/, csv/, "
             "checkpoints/). Defaults to simulation/results/ (live). Use a "
             "backup_*/ path while training is still writing to live.",
    )
    ap.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=None,
        help="Override path to the checkpoints directory (for phase8 MI "
             "ranking). Useful when post_E lives in one backup and "
             "checkpoints in another.",
    )
    args = ap.parse_args()

    global SRC, SRC_CHECKPOINTS
    if args.src_dir is not None:
        SRC = args.src_dir
        log.info("input source = %s (override)", SRC)
    if args.checkpoints_dir is not None:
        SRC_CHECKPOINTS = args.checkpoints_dir
        log.info("checkpoints source = %s (override)", SRC_CHECKPOINTS)

    build_forecast_manifest()
    build_dm_manifest()
    build_shap_manifest()
    log.info("Stage 6a MCP artifacts complete")


if __name__ == "__main__":
    main()
