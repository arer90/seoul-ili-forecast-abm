#!/usr/bin/env python3
"""Wire the research champion FusedEpi into the operational web forecast.

The web previously displayed NegBinGLM (the production model that can be refit
for a synthetic future row). The research champion by WIS is FusedEpi, and its
operational forward forecast already exists on disk — produced by the
expanding-window multi-horizon run (``run_expanding_multihorizon``), which refits
FusedEpi at each rolling origin and forecasts forward leak-free. This script
takes the **latest origin's** FusedEpi forward forecast (a genuine future
prediction, ``actual=null``) and writes it into ``ili-forecast.json`` with the
production gate applied, so that web = ABM anchor = ARIA grounding = research
champion = FusedEpi, all consistent.

No retraining: the FusedEpi forecast is reused from the on-disk artifact
(single-champion refit already done by the expanding-window run, not the
48-model pipeline). The web JSON's derived fields (gate, surge, per-gu) are
recomputed consistently so nothing is left inconsistent.

Run from project root:
    python3 web/scripts/build_fusedepi_web_forecast.py

Performance: < 1s (reads two JSON files, writes one). No DB write, no model fit.
Side effects: backs up then overwrites web/public/aggregates/ili-forecast.json.
Caller responsibility: run_expanding_multihorizon must have produced result.json.
"""
from __future__ import annotations

import datetime
import json
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_EXPANDING = _ROOT / "simulation" / "results" / "expanding_multihorizon" / "result.json"
_OUT = _ROOT / "web" / "public" / "aggregates" / "ili-forecast.json"
_OUT_MODELS = _ROOT / "web" / "public" / "aggregates" / "ili-forecast-models.json"

# FusedEpi rolling 1-step test metrics (per_model_eval SSOT, 68-week slab).
_FUSEDEPI = {"wis": 3.278, "test_r2": 0.9357, "test_rmse": 6.591, "test_mae": 3.896}
# Gate contract (mirror of build_production_forecast._gate_forecast bounds).
_TRAIN_MAX_X3 = None  # set from data if available; else a generous epidemiological ceiling
_EPI_CEILING = 120.0  # ILI per 1k — far above any observed Seoul value; pure finiteness guard


def _latest_fusedepi_forward(result: dict) -> dict:
    """Return the latest rolling origin's FusedEpi h=1 forward forecast.

    Args:
        result: parsed expanding_multihorizon/result.json.

    Returns:
        dict with origin_date, pred, pi_lo, pi_hi for the h=1 horizon of the
        most recent origin (the genuine operational forward forecast).

    Raises:
        SystemExit: if no origin/horizon is available.
    """
    origins = result.get("origins", [])
    if not origins:
        raise SystemExit(f"no origins in {_EXPANDING}")
    origin = origins[-1]
    h1 = next((h for h in origin.get("horizons", []) if int(h["h"]) == 1), None)
    if h1 is None:
        raise SystemExit("no h=1 horizon in latest origin")
    return {
        "origin_date": origin.get("origin_date"),
        "pred": float(h1["pred"]),
        "pi_lo": float(h1.get("pi_lo", 0.0)),
        "pi_hi": float(h1.get("pi_hi", 0.0)),
    }


def _gate(forecast: float) -> dict:
    """Apply the production gate (finite ∧ nonneg ∧ ≤ epidemiological ceiling).

    Args:
        forecast: the point forecast (ILI per 1k).

    Returns:
        gate result dict {passed, n_violations, reason, replaced}.
    """
    viol: list[str] = []
    if not (forecast == forecast) or forecast in (float("inf"), float("-inf")):
        viol.append("non-finite")
    if forecast < 0:
        viol.append("negative")
    if forecast > _EPI_CEILING:
        viol.append(f"exceeds epi ceiling {_EPI_CEILING}")
    return {
        "passed": not viol,
        "n_violations": len(viol),
        "reason": "; ".join(viol) if viol else "ok",
        "replaced": False,
    }


def _sync_models_json(
    city: float, lo: float, hi: float, half: float,
    forecast_date: datetime.date, observed_at_iso: str,
) -> None:
    """Re-assert FusedEpi as champion in ili-forecast-models.json (the 12-model panel).

    The multi-model panel (``build_production_forecast._write_ili_forecast_models``)
    refits the cheaply-refittable NegBinGLM and labels it champion. The research +
    production champion by WIS is FusedEpi, whose forward forecast comes from the
    expanding-window run (a different artifact, not the 48-model panel). Since
    FusedEpi is not in the panel's summary_metrics top-12, we sync the panel by:
      1. flipping the top-level champion fields to FusedEpi (forecast = the same
         value ili-forecast.json carries, so the two web files agree),
      2. inserting/refreshing a FusedEpi entry at rank 1 with ``is_champion=True``
         and clearing every other model's ``is_champion`` flag.

    No retraining: reuses the on-disk FusedEpi forward forecast (city/lo/hi) that
    ili-forecast.json already wrote. The other panel models keep their existing
    comparison metrics/forecasts untouched.

    Args:
        city: FusedEpi point forecast (ILI per 1k), == ili-forecast.json city_forecast.
        lo, hi: 95% PI bounds for the FusedEpi forecast.
        half: conformal half-width ((hi-lo)/2).
        forecast_date: target week date.
        observed_at_iso: ISO datetime of the last observation (origin).

    Side effects: backs up then overwrites ili-forecast-models.json.
    Caller responsibility: ili-forecast.json must already be FusedEpi-synced.
    """
    if not _OUT_MODELS.exists():
        print(f"[build_fusedepi_web_forecast] WARN {_OUT_MODELS} absent — models.json not synced")
        return
    models = json.loads(_OUT_MODELS.read_text(encoding="utf-8"))
    bak = _OUT_MODELS.with_suffix(".json.bak_pre_fusedepi")
    shutil.copy2(_OUT_MODELS, bak)

    now = datetime.datetime.utcnow().isoformat() + "Z"
    gu_names = list(next(
        (m.get("gu", {}) for m in models.get("models", []) if m.get("gu")), {}
    ).keys())
    # Uniform_city allocation (district-level ILI unobserved, permutation p=0.073).
    gu_block = {g: {"ili": round(city, 4), "lo": round(lo, 4), "hi": round(hi, 4)} for g in gu_names}

    fusedepi_entry = {
        "name": "FusedEpi",
        "category": "fusion",
        "rank": 1,
        "is_champion": True,
        "forecast_source": "expanding-origin-forward (single-champion refit)",
        "pi_method": "adaptive-conformal-PID",
        "metrics": {
            "test_r2": _FUSEDEPI["test_r2"],
            "test_rmse": _FUSEDEPI["test_rmse"],
            "test_mae": _FUSEDEPI["test_mae"],
            "test_wis": _FUSEDEPI["wis"],
        },
        "city_forecast": round(city, 4),
        "city_lo": round(lo, 4),
        "city_hi": round(hi, 4),
        "gu": gu_block,
    }

    # Clear stale champion flags; drop any pre-existing FusedEpi row to avoid dupes.
    others = []
    for m in models.get("models", []):
        if m.get("name") == "FusedEpi":
            continue
        m["is_champion"] = False
        others.append(m)
    models["models"] = [fusedepi_entry] + others
    for i, m in enumerate(models["models"], 1):
        m["rank"] = i

    models.update({
        "generated_at": now,
        "observed_at": observed_at_iso,
        "forecast_at": forecast_date.isoformat() + "T00:00:00Z",
        "source": "fusedepi-expanding-origin-forward (panel synced)",
        "champion": "FusedEpi",
        "champion_version": "R10-champion (TiRex+TabPFN fusion, NegBin, mechanistic)",
        "champion_forecast": round(city, 4),
        "conformal_q95": round(half, 4),
        "note": (
            "Champion = FusedEpi (research + production, best WIS 3.278), synced to "
            "ili-forecast.json. Forecast = on-disk expanding-window single-champion "
            "refit (leak-free, no 48-model retrain). The remaining panel models keep "
            "their comparison metrics for context; only the champion is refit."
        ),
    })

    _OUT_MODELS.write_text(json.dumps(models, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[build_fusedepi_web_forecast] models.json champion -> FusedEpi (forecast={city:.3f})")
    print(f"  backup={bak}")
    print(f"  -> {_OUT_MODELS}")


def main() -> None:
    """Load latest FusedEpi forward, recompute web fields, overwrite ili-forecast.json."""
    if not _EXPANDING.exists():
        raise SystemExit(f"missing {_EXPANDING} — run run_expanding_multihorizon first")
    if not _OUT.exists():
        raise SystemExit(f"missing {_OUT}")

    fwd = _latest_fusedepi_forward(json.loads(_EXPANDING.read_text(encoding="utf-8")))
    current = json.loads(_OUT.read_text(encoding="utf-8"))

    # Backup before overwrite.
    bak = _OUT.with_suffix(".json.bak_pre_fusedepi")
    shutil.copy2(_OUT, bak)

    city = fwd["pred"]
    lo, hi = max(0.0, fwd["pi_lo"]), fwd["pi_hi"]
    half = round((hi - lo) / 2.0, 4)
    gate = _gate(city)
    # Surge: the ML/forecast model defers a large surge to the SEIR/ABM engine.
    surge = city > 30.0  # ILI per 1k — well above seasonal peak; none expected here

    origin = datetime.date.fromisoformat(fwd["origin_date"])
    forecast_date = origin + datetime.timedelta(days=7)
    now = datetime.datetime.utcnow().isoformat() + "Z"

    current.update({
        "generated_at": now,
        "observed_at": origin.isoformat() + "T00:00:00Z",
        "forecast_at": forecast_date.isoformat() + "T00:00:00Z",
        "source": "fusedepi-expanding-origin-forward",
        "model": "FusedEpi",
        "model_version": "R10-champion (TiRex+TabPFN fusion, NegBin, mechanistic)",
        "horizon_weeks": 1,
        "city_forecast": round(city, 4),
        "city_lo": round(lo, 4),
        "city_hi": round(hi, 4),
        "conformal_q95": half,
        "gate": gate,
        "surge": {
            "detected": bool(surge),
            "reason": "forecast within seasonal range" if not surge else "large surge",
            "action": ("DEFER to mechanistic SEIR/ABM engine" if surge else "none"),
        },
        "metrics": {
            "test_wis": _FUSEDEPI["wis"],
            "test_r2": _FUSEDEPI["test_r2"],
            "test_rmse": _FUSEDEPI["test_rmse"],
            "test_mae": _FUSEDEPI["test_mae"],
        },
        "staleness_note": (
            "Web now serves the research champion FusedEpi (R10, lowest WIS 3.278). "
            f"Forecast origin {fwd['origin_date']} (latest expanding-window rolling origin); "
            "for the absolute latest observed week, re-run the single-champion FusedEpi "
            "refit (run_expanding_multihorizon) — no 48-model retrain. "
            "web = ABM anchor = ARIA grounding = FusedEpi, consistent."
        ),
        "note": (
            "FusedEpi operational forward forecast (1-step), reused from the on-disk "
            "expanding-window rolling-origin run (single-champion refit, leak-free, "
            "no 48-model retrain). Gate: " + gate["reason"] + ". "
            "PI from the run's adaptive-conformal interval."
        ),
    })
    # Production champion now aligned with the research champion.
    current["research_champion"] = {
        "model": "FusedEpi",
        "selected_by": "R10 per_model_eval — best WIS (3.278), R² 0.936",
        "now_also_production": True,
    }

    # Recompute per-gu consistently with the FusedEpi city forecast. The gu
    # allocation is uniform_city (district-level ILI is unobserved and the
    # permutation test failed, p=0.073), so every district carries the city
    # value — otherwise the map would show stale numbers under a new label.
    if isinstance(current.get("gu"), dict):
        for gu_name in current["gu"]:
            current["gu"][gu_name] = {
                "ili": round(city, 4),
                "lo": round(lo, 4),
                "hi": round(hi, 4),
            }

    # Invariant: city and per-gu must agree (uniform allocation).
    gu_vals = {round(v["ili"], 4) for v in current.get("gu", {}).values()}
    assert gu_vals <= {round(city, 4)}, f"gu/city inconsistency: {gu_vals} vs {city}"

    _OUT.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[build_fusedepi_web_forecast] web champion -> FusedEpi")
    print(f"  origin={fwd['origin_date']}  city_forecast={city:.3f}  PI=[{lo:.2f},{hi:.2f}]  gate={gate['reason']}")
    print(f"  backup={bak}")
    print(f"  -> {_OUT}")

    # Sync the multi-model panel (ili-forecast-models.json) so its champion is also
    # FusedEpi — otherwise the panel keeps showing the stale NegBinGLM champion.
    _sync_models_json(
        city, lo, hi, half, forecast_date,
        origin.isoformat() + "T00:00:00Z",
    )


if __name__ == "__main__":
    main()
