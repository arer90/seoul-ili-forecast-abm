"""Champion consistency guard — keeps the FusedEpi champion coherent across layers.

This session surfaced two champion-coherence failures that a guard would have
caught immediately:
  1. the web dashboard showed a "FusedEpi" label while its per-district map still
     served stale NegBinGLM numbers (city 4.70 vs gu 8.98) — an internal
     city/gu inconsistency (found by an external cross-check, fixed in G-381);
  2. the research champion (FusedEpi, lowest WIS) drifted from the deployed
     web/production model (NegBinGLM) without that being reconciled.

These tests assert the invariants that must always hold once the champion is
wired end-to-end, so the same inconsistencies fail loudly here instead of
shipping silently. They read on-disk artifacts only (no training, no DB writes).

Run (per-file, macOS): .venv/bin/python -m pytest tests/test_champion_consistency.py -q
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_EVAL_CSV = _ROOT / "simulation" / "results" / "per_model_eval" / "per_model_metrics.csv"
_WEB = _ROOT / "web" / "public" / "aggregates" / "ili-forecast.json"
_WEB_MODELS = _ROOT / "web" / "public" / "aggregates" / "ili-forecast-models.json"

_CHAMPION = "FusedEpi"


def _eval_rows() -> dict[str, dict]:
    if not _EVAL_CSV.exists():
        pytest.skip(f"{_EVAL_CSV} absent (run the eval pipeline first)")
    return {r["model"]: r for r in csv.DictReader(_EVAL_CSV.open(encoding="utf-8"))}


def _web() -> dict:
    if not _WEB.exists():
        pytest.skip(f"{_WEB} absent")
    return json.loads(_WEB.read_text(encoding="utf-8"))


def _web_models() -> dict:
    if not _WEB_MODELS.exists():
        pytest.skip(f"{_WEB_MODELS} absent")
    return json.loads(_WEB_MODELS.read_text(encoding="utf-8"))


def test_champion_is_lowest_wis_in_eval() -> None:
    """The G-339 champion (FusedEpi) must hold the lowest WIS among scored models.

    If a future run makes another model lower-WIS, this fails so the champion
    label is re-confirmed deliberately rather than drifting.
    """
    rows = _eval_rows()
    wis = {}
    for m, r in rows.items():
        try:
            v = float(r.get("wis", ""))
        except (TypeError, ValueError):
            continue
        if not math.isnan(v):
            wis[m] = v
    assert _CHAMPION in wis, f"{_CHAMPION} has no scored WIS in eval CSV"
    best = min(wis, key=wis.get)
    assert best == _CHAMPION, (
        f"lowest-WIS model is {best} ({wis[best]:.3f}), not {_CHAMPION} "
        f"({wis[_CHAMPION]:.3f}) — champion drift; re-confirm via G-339 selection"
    )


def test_web_serves_the_champion() -> None:
    """The deployed web forecast must be labelled with the champion model."""
    web = _web()
    assert web.get("model") == _CHAMPION, (
        f"web model={web.get('model')!r} != champion {_CHAMPION!r} — "
        f"research/production champion drift (reconcile or document deliberately)"
    )
    rc = web.get("research_champion", {})
    assert rc.get("model") == _CHAMPION, f"research_champion={rc.get('model')!r} != {_CHAMPION}"


def test_web_city_and_gu_consistent() -> None:
    """Per-district map values must equal the city forecast (uniform allocation).

    This is the exact invariant that the G-381 bug violated: city was updated to
    FusedEpi while the gu block kept stale numbers. With uniform_city allocation
    every district carries the city value, so any mismatch is a wiring error.
    """
    web = _web()
    city = round(float(web["city_forecast"]), 4)
    gu = web.get("gu", {})
    assert gu, "web forecast has no gu block"
    mismatches = {
        name: round(float(v["ili"]), 4)
        for name, v in gu.items()
        if round(float(v["ili"]), 4) != city
    }
    assert not mismatches, (
        f"city_forecast={city} but {len(mismatches)} districts differ "
        f"(e.g. {dict(list(mismatches.items())[:3])}) — stale gu numbers under a "
        f"new model label (the G-381 failure mode)"
    )


def test_models_panel_serves_the_champion() -> None:
    """The multi-model panel (ili-forecast-models.json) must declare FusedEpi champion.

    This is the exact divergence a Codex SCI review caught: ili-forecast.json was
    already FusedEpi while ili-forecast-models.json still labelled NegBinGLM the
    champion (champion_forecast=8.9774). The panel and the single-forecast file
    must agree, so a future stale panel fails here instead of shipping.
    """
    panel = _web_models()
    assert panel.get("champion") == _CHAMPION, (
        f"models.json champion={panel.get('champion')!r} != {_CHAMPION!r} — "
        f"the panel drifted from the FusedEpi champion (the Codex-caught divergence)"
    )
    models = panel.get("models", [])
    champs = [m["name"] for m in models if m.get("is_champion")]
    assert champs == [_CHAMPION], (
        f"is_champion flags = {champs}, expected exactly [{_CHAMPION!r}] — "
        f"stale or multiple champion flags in the panel"
    )


def test_models_panel_agrees_with_single_forecast() -> None:
    """models.json champion_forecast must equal ili-forecast.json city_forecast.

    Both web files serve the same FusedEpi forward forecast; if they disagree the
    dashboard shows two different "champion" numbers (8.98 vs 4.70) — the exact
    inconsistency under review.
    """
    panel = _web_models()
    web = _web()
    panel_fc = round(float(panel["champion_forecast"]), 4)
    city_fc = round(float(web["city_forecast"]), 4)
    assert panel_fc == city_fc, (
        f"models.json champion_forecast={panel_fc} != ili-forecast.json "
        f"city_forecast={city_fc} — the two web files serve different champion numbers"
    )
