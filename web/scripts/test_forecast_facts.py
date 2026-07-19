#!/usr/bin/env python3
"""Forecast / accuracy regression tests (TDD) — all assertions use REAL data as-is.

Locks in the honest findings from the 2026-06 accuracy investigation (codex + gemini +
model-advisor + the SEIR-blend falsifier) so they can't silently regress:

  - relative-conformal PI lifted coverage 38%→~66% (and improved WIS)
  - the winter peak under-shoot is REAL and structural
  - naive ensembling does NOT beat single-best at peaks (don't re-add it)
  - the project's seasonal SEIR produces NO winter wave (reT<1.15) → the ABM blend
    lever is genuinely exhausted (don't wire a degenerate-SEIR blend)
  - production forecast stays future + gated

Real sources only: web/public/aggregates/backtest.json, ili-forecast-models.json,
web_prototype/data.json, and the deterministic SEIR in _blend_validate.

Run:  .venv/bin/python web/scripts/test_forecast_facts.py
  or: .venv/bin/python -m pytest web/scripts/test_forecast_facts.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGG = ROOT / "web" / "public" / "aggregates"
sys.path.insert(0, str(ROOT / "web" / "scripts"))


def _backtest() -> dict:
    return json.loads((AGG / "backtest.json").read_text(encoding="utf-8"))


def _champion() -> dict:
    return _backtest()["models"][0]


# ── relative-conformal PI (38%→~66%) ───────────────────────────────────────
def test_relative_conformal_lifted_coverage():
    """PI95 coverage must be materially above the broken additive 38% (now ~66%)."""
    cov = _champion()["metrics"]["pi95_coverage"]
    assert 0.55 <= cov <= 0.85, f"PI95 coverage {cov} not in the relative-conformal band (was 0.38 additive)"


def test_wis_improved_vs_additive():
    """Better-calibrated PI lowered WIS below the additive-era 4.50."""
    wis = _champion()["metrics"]["wis"]
    assert wis < 4.5, f"WIS {wis} did not improve over additive-era 4.50"


def test_pi_bands_present_on_test_points():
    tp = _champion()["test_points"][0]
    for k in ("lower95", "upper95", "wis"):
        assert k in tp, f"test_point missing {k}"


# ── the winter peak under-shoot is REAL and structural ──────────────────────
def test_peak_undershoot_is_real():
    """At the worst rising-edge week the model under-shoots by a large margin (real)."""
    pts = _champion()["test_points"]
    worst = min(((p["predicted"] - p["actual"]) / p["actual"], p)
                for p in pts if p["actual"] > 40)[1]
    rel = (worst["predicted"] - worst["actual"]) / worst["actual"]
    assert rel < -0.25, f"expected a real >25% under-shoot at peaks; worst was {rel:.0%} on {worst['date']}"


# ── naive ensemble does NOT beat single-best at peaks (locked negative) ──────
def test_naive_ensemble_not_better_at_peaks():
    """Mean ensemble must NOT beat single-best at peaks — guards against re-adding it."""
    models = _backtest()["models"]
    actual = [p["actual"] for p in models[0]["test_points"]]
    preds = [[p["predicted"] for p in m["test_points"]] for m in models]
    ens = [sum(preds[j][i] for j in range(len(preds))) / len(preds) for i in range(len(actual))]
    thr = sorted(actual, reverse=True)[max(1, len(actual) // 4)]
    hi = [i for i, a in enumerate(actual) if a >= thr]
    mae_single = sum(abs(preds[0][i] - actual[i]) for i in hi) / len(hi)
    mae_ens = sum(abs(ens[i] - actual[i]) for i in hi) / len(hi)
    assert mae_ens >= mae_single - 1e-6, (
        f"naive ensemble peak MAE {mae_ens:.2f} beat single-best {mae_single:.2f} — "
        "re-investigate, but the 2026-06 finding was ensemble HURTS peaks")


# ── the project's seasonal SEIR produces NO winter wave (blend lever exhausted) ──
def test_seir_has_no_winter_wave():
    """Deterministic seasonal SEIR reT must stay <1.15 (no wave) — the falsifier's root cause.

    If this ever passes (reT>1.15), the SEIR was re-calibrated and the ABM-blend lever
    should be RE-VALIDATED via _blend_validate.py before claiming it works.
    """
    from _blend_validate import seir_seasonal
    max_reT = max(r for _, r in seir_seasonal(0.45))
    assert max_reT < 1.15, (
        f"SEIR max reT {max_reT:.3f} ≥ 1.15 — it now produces a wave; "
        "re-run _blend_validate.py to (re)assess the mechanistic blend")


# ── production forecast stays future + gated ────────────────────────────────
def test_production_forecast_future_and_gated():
    fc = json.loads((AGG / "ili-forecast-models.json").read_text(encoding="utf-8"))
    assert fc["forecast_at"] > fc["observed_at"], "forecast not beyond last observation"
    v = fc["models"][0]["city_forecast"]
    assert 0.0 <= v < 300.0, f"forecast {v} outside gate"


# ── rising-edge warning fires only in the elevated regime (honest, real-threshold) ──
def test_rising_edge_warning_threshold():
    """The web warning triggers at city_forecast ≥ 10/1k; the current (summer) forecast
    is low so it must NOT fire now (no false alarm)."""
    fc = json.loads((AGG / "ili-forecast-models.json").read_text(encoding="utf-8"))
    v = fc["models"][0]["city_forecast"]
    # current forecast is summer-low → warning should be OFF (v < 10)
    assert v < 10.0, f"current forecast {v} unexpectedly ≥10 (warning would fire) — verify regime"


if __name__ == "__main__":
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    p = f = 0
    for fn in funcs:
        try:
            fn()
            print(f"  ✓ PASS  {fn.__name__}")
            p += 1
        except Exception as e:
            print(f"  ✗ FAIL  {fn.__name__}: {e}")
            f += 1
    print(f"\n  {p} PASS / {f} FAIL")
    sys.exit(1 if f else 0)
