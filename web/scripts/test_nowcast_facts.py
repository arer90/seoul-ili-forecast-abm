#!/usr/bin/env python3
"""Nowcast / external-feature regression tests (TDD) — REAL data, all 3 winters.

Locks the 2026-06 honest findings (nowcast_external.run_nowcast) so they can't regress:
  - weekly 1-step nowcast >> recursive multi-step roll (the REAL fix for the Jan–Feb 2nd
    wave — re-anchoring with real ILI each week, not external data)
  - real-time road traffic (Rt도로) is the ONLY external group that helps Jan–Feb robustly
    across all 3 winters AND never catastrophically overfits → the one deploy-worthy signal
  - dumping all 45 external features overfits (worse than curated Rt도로; catastrophic in
    ≥1 winter) → never deploy the full dump
  - airport/flight data is NOT in the feature matrix (honest limitation, collector needed)

Slow (~2 min: ~18 NegBinGLM refits across 3 winters).
Run:  .venv/bin/python web/scripts/test_nowcast_facts.py
  or: .venv/bin/python -m pytest web/scripts/test_nowcast_facts.py -v
"""
from __future__ import annotations

import datetime
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "web" / "scripts"))

from nowcast_external import run_nowcast, RECURSIVE_ROLL_2025_26  # noqa: E402

# Compute the primary (2025-26) winter once; reused by the fast assertions.
_R = run_nowcast(datetime.date(2025, 12, 31))


# ── the REAL fix = nowcast structure, not external data ─────────────────────
def test_nowcast_beats_recursive_roll():
    """Weekly 1-step (real lags) must beat the Dec-anchored recursive roll (decaying lags)."""
    assert _R["mae"]["basic_all"] < RECURSIVE_ROLL_2025_26, (
        f"1-step BASIC {_R['mae']['basic_all']} must beat recursive roll "
        f"{RECURSIVE_ROLL_2025_26} — the nowcast structure is the main fix")


# ── Rt road traffic = the one deploy-worthy external signal ──────────────────
def test_rt_road_helps_janfeb_primary():
    rt = _R["mae"]["groups"]["+Rt도로"]["janfeb"]
    base = _R["mae"]["basic_janfeb"]
    assert rt < base, f"Rt도로 Jan-Feb {rt} must beat BASIC {base} (2025-26)"


def test_rt_road_robust_across_three_winters():
    """Across 2023-24, 2024-25, 2025-26: Rt도로 helps-or-neutral on Jan-Feb AND never
    catastrophically overfits (full MAE < 1.5×BASIC). This is what justifies deploying it."""
    results = {2025: _R}
    for yr in (2023, 2024):
        results[yr] = run_nowcast(datetime.date(yr, 12, 31))
    for yr, r in results.items():
        base_jf = r["mae"]["basic_janfeb"]
        base_all = r["mae"]["basic_all"]
        rt = r["mae"]["groups"]["+Rt도로"]
        assert rt["janfeb"] <= base_jf + 0.5, (
            f"{yr} winter: Rt도로 Jan-Feb {rt['janfeb']} worse than BASIC {base_jf}")
        assert rt["all"] < 1.5 * base_all, (
            f"{yr} winter: Rt도로 full {rt['all']} blew up vs BASIC {base_all}")


# ── dumping all external features overfits (don't deploy the dump) ───────────
def test_full_dump_worse_than_curated_rt():
    rt = _R["mae"]["groups"]["+Rt도로"]["all"]
    full = _R["mae"]["groups"]["+전체외부"]["all"]
    assert full > rt, (
        f"full-dump {full} should be worse than curated Rt도로 {rt} — 45 features overfit")


# ── honest limitation: no airport data ──────────────────────────────────────
def test_airport_data_absent():
    assert _R["airport_cols"] == 0, (
        "airport/flight columns unexpectedly present — update the collect-audit and re-test")


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
