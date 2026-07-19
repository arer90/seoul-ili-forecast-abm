"""Guard: an all-zero attribution must never be reported as an explanation.

`_permutation_importance` builds `imps = np.zeros(p)` and returns one entry per
feature regardless of what happened, so a run that measured nothing still hands
back a full list of `(feature, 0.0)` pairs. `shap_analysis` decided "this model
was explained" by testing that list for truthiness — which is True for the
all-zero case. Two things followed:

  `_summary.json` counted 34 models as having permutation importance and 34 as
  having native SHAP. Recomputed from the attributions actually on disk: 18 and
  27.

  `status["top_native"]` took the first entries of a list sorted by an all-zero
  key. Python's sort is stable, so it returned the original column order, and
  `REPORT.md` printed `temp_avg, temp_min, humidity, wind_speed` — columns 0-3 —
  as the top drivers for seven models whose `shap_values.npy` was entirely zero.

The distinction the flag is supposed to carry is "measured, and these features
matter" versus "could not be measured". An all-zero result is the second, and
reporting it as the first is worse than reporting nothing.

Run standalone — macOS needs per-file pytest runs (LightGBM/OpenMP):
    .venv/bin/python -m pytest simulation/tests/test_shap_zero_is_not_measured.py -q
"""

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from simulation.pipeline.shap_analysis import _measured, _measured_array

ROOT = Path(__file__).resolve().parents[2]
SHAP = ROOT / "simulation" / "results" / "shap"


# ── the rule ─────────────────────────────────────────────────────────────────
def test_all_zero_ranking_is_not_measured():
    """The exact shape _permutation_importance returns when nothing registered."""
    assert _measured([("a", 0.0), ("b", 0.0), ("c", 0.0)]) is False


def test_a_single_nonzero_makes_it_measured():
    assert _measured([("a", 0.0), ("b", 0.0), ("c", 1e-9)]) is True


def test_empty_and_nan_rankings_are_not_measured():
    assert _measured([]) is False
    assert _measured([("a", float("nan")), ("b", float("nan"))]) is False


def test_truthiness_would_have_said_the_opposite():
    """Shows the guard is not vacuous — the old test passed on this input."""
    all_zero = [("a", 0.0), ("b", 0.0)]
    assert bool(all_zero) is True          # what the code used to check
    assert _measured(all_zero) is False    # what it checks now


def test_all_zero_matrix_is_not_measured():
    assert _measured_array(np.zeros((30, 397))) is False
    assert _measured_array(None) is False
    m = np.zeros((30, 397))
    m[3, 7] = -0.02
    assert _measured_array(m) is True


# ── the shipped artifacts must obey it ───────────────────────────────────────
def _shipped():
    if not SHAP.is_dir():
        pytest.skip("no shipped SHAP tree")
    out = {}
    for d in sorted(x for x in SHAP.iterdir() if x.is_dir()):
        f = d / "importance.csv"
        if not f.exists():
            continue
        rows = list(csv.DictReader(f.open(encoding="utf-8")))

        def col(k):
            vals = []
            for r in rows:
                try:
                    vals.append((r["feature"], float(r[k])))
                except (TypeError, ValueError, KeyError):
                    pass
            return vals

        out[d.name] = (col("permutation_importance"), col("native_shap_importance"), d)
    return out


def test_no_shipped_shap_values_file_is_all_zero():
    """A zero matrix on disk is indistinguishable from a real explanation."""
    offenders = []
    for name, (_, _, d) in _shipped().items():
        f = d / "shap_values.npy"
        if f.exists() and not _measured_array(np.load(f)):
            offenders.append(name)
    assert not offenders, (
        f"all-zero shap_values.npy shipped for {offenders} — anyone loading these "
        f"gets a correctly shaped matrix of zeros with no way to tell"
    )


def test_summary_counts_match_the_attributions_on_disk():
    p = SHAP / "_summary.json"
    if not p.exists():
        pytest.skip("no _summary.json")
    j = json.loads(p.read_text(encoding="utf-8"))
    shipped = _shipped()
    n_perm = sum(1 for perm, _, _ in shipped.values() if _measured(perm))
    n_nat = sum(1 for _, nat, _ in shipped.values() if _measured(nat))
    assert j.get("n_with_permutation") == n_perm, (
        f"summary claims {j.get('n_with_permutation')} models with permutation "
        f"importance; {n_perm} have a non-zero attribution"
    )
    assert j.get("n_with_native_shap") == n_nat, (
        f"summary claims {j.get('n_with_native_shap')} models with native SHAP; "
        f"{n_nat} have a non-zero attribution"
    )


def test_report_does_not_tick_an_unmeasured_model():
    p = SHAP / "REPORT.md"
    if not p.exists():
        pytest.skip("no REPORT.md")
    shipped = _shipped()
    bad = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or line.startswith("| model"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 4:
            continue
        name, _, perm_cell, nat_cell = cells[0], cells[1], cells[2], cells[3]
        if name not in shipped:
            continue
        perm, nat, _ = shipped[name]
        if perm_cell == "✓" and not _measured(perm):
            bad.append(f"{name}: permutation ✓ but all-zero")
        if nat_cell == "✓" and not _measured(nat):
            bad.append(f"{name}: native SHAP ✓ but all-zero")
    assert not bad, "REPORT.md ticks models nothing was measured on:\n  " + "\n  ".join(bad)
