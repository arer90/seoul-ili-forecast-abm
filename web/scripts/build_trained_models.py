#!/usr/bin/env python3
"""Generate web/public/aggregates/trained-models.json from the LIVE pipeline output.

M3 (2026-06-06): replaces the v22.6-FROZEN hand-made file (which had NO generator
— the web's model list/ranking was stuck at an April vintage, unrelated to the
trained champions) with a fresh build from
``simulation/results/per_model_eval/per_model_metrics.csv``.

Run after training (wired into the db→web orchestration, M4) so the web always
reflects the latest run. Schema matches the existing consumers
(ForecastModelPicker.tsx / TrainedModelsCard.tsx): {version, timestamp,
total_models, source, metric_hint, top:[{rank,name,r2,rmse,wis,crps,cov95,mape,family}]}.
"""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repo root
DEFAULT_CSV = ROOT / "simulation" / "results" / "per_model_eval" / "per_model_metrics.csv"
DEFAULT_OUT = ROOT / "web" / "public" / "aggregates" / "trained-models.json"


def _family(name: str) -> str:
    """Model family/category (best-effort from the registry; 'unknown' on failure)."""
    try:
        from simulation.models.registry import CATEGORY_MODELS  # lazy
        for cat, models in CATEGORY_MODELS.items():
            if name in models:
                return cat
    except Exception:
        pass
    return "unknown"


def build(metrics_csv: str | Path = DEFAULT_CSV,
          out: str | Path = DEFAULT_OUT,
          timestamp: str | None = None) -> tuple[Path, int]:
    """Build trained-models.json from per_model_metrics.csv.

    Args:
        metrics_csv: live per_model_eval metrics CSV (model × metric).
        out: output JSON path (web aggregates).
        timestamp: ISO date; default = today (UTC).

    Returns:
        (out_path, n_models). Raises FileNotFoundError if metrics_csv missing.
    """
    metrics_csv, out = Path(metrics_csv), Path(out)
    with metrics_csv.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    def _f(r: dict, k: str):
        try:
            return round(float(r.get(k)), 4)
        except (TypeError, ValueError):
            return None

    ranked = sorted(
        rows,
        key=lambda r: (_f(r, "wis") if _f(r, "wis") is not None else 1e18),
    )
    top = [{
        "rank": i + 1,
        "name": r.get("model"),
        "r2": _f(r, "r2"), "rmse": _f(r, "rmse"), "wis": _f(r, "wis"),
        "crps": _f(r, "crps_gaussian"), "cov95": _f(r, "pi95_coverage"),
        "mape": _f(r, "mape"), "family": _family(r.get("model", "")),
    } for i, r in enumerate(ranked)]

    payload = {
        "version": "live",
        "timestamp": timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "total_models": len(top),
        "source": "per_model_eval/per_model_metrics.csv",
        "metric_hint": "sorted by WIS ascending (lower=better)",
        "top": top,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out, len(top)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build web trained-models.json from live metrics")
    ap.add_argument("--metrics-csv", default=str(DEFAULT_CSV))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    try:
        out, n = build(args.metrics_csv, args.out)
    except FileNotFoundError:
        print(f"[trained-models] metrics CSV not found: {args.metrics_csv} "
              f"(run training first)")
        return 1
    print(f"[trained-models] wrote {n} models → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
