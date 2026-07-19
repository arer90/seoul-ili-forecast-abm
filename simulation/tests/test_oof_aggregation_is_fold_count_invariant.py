"""Guard: the OOF scalar used to pick the champion does not depend on fold counts.

This pins a fact that was got backwards on 2026-07-19, in the repository's own
comments and then in an audit that repeated them.

The claim that was wrong: that `oof_wis` (regime-balanced, 0.5·quiet + 0.5·elevated)
is fold-COUNT sensitive, and that FusedEpi — the only model with 4 folds rather
than 5 — therefore "escapes the outbreak penalty" and holds the championship
through a structural advantage. Substituting the plain-mean scalar
(`_selection_oof_wis`) does move the champion to GAM-Spline, which looked like
confirmation.

It is not. A weighted average of two group means,

    0.5 · mean(quiet folds) + 0.5 · mean(elevated folds)

does not contain the group sizes. It is invariant to fold count by construction.
FusedEpi pays "no penalty" relative to the plain mean only because at a 2-2 split
the 50/50 weighting *coincides* with the plain mean — arithmetic, not advantage.

The scalar that IS composition-sensitive is the plain mean: it weights outbreak
folds by however many of them a model happens to have (40% at 3 quiet + 2
elevated, 50% at 2-2). Swapping it in does not remove a bias, it introduces one.
And substantively it reverses the epidemiological intent: FusedEpi beats
GAM-Spline on the elevated folds (mean 2.297 vs 2.599) and loses on the quiet
ones, so a score that weights outbreaks at 50% should — and does — prefer it.

Run standalone — macOS needs per-file pytest runs (LightGBM/OpenMP):
    .venv/bin/python -m pytest simulation/tests/test_oof_aggregation_is_fold_count_invariant.py -q
"""

import ast
import csv
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "simulation" / "results" / "per_model_eval" / "per_model_metrics.csv"

PENALTY_COEF = 0.05          # MPH_OOF_WIS_VAR_PENALTY default


def _regime(folds, k_quiet):
    q, e = np.asarray(folds[:k_quiet], float), np.asarray(folds[k_quiet:], float)
    return 0.5 * q.mean() + 0.5 * e.mean()


def _penalize(score, folds):
    v = np.asarray(folds, float)
    return score * (1.0 + PENALTY_COEF * v.std(ddof=1) / max(abs(v.mean()), 1e-9))


# ── the property, on synthetic data ──────────────────────────────────────────
def test_regime_mean_ignores_how_many_folds_are_in_each_regime():
    """Add a duplicate quiet fold: the regime mean must not move."""
    quiet, elevated = [0.30, 0.35], [2.50, 2.90]
    base = _regime(quiet + elevated, 2)
    grown = _regime(quiet + [0.32] + elevated, 3)          # 3 quiet + 2 elevated
    assert grown == pytest.approx(base, rel=0.05), (
        "the regime-balanced mean moved when a quiet fold was added; it is "
        "supposed to be a 50/50 average of the two group means"
    )


def test_plain_mean_is_the_one_that_moves():
    """Shows the test above is not vacuous — plain mean is composition-sensitive."""
    quiet, elevated = [0.30, 0.35], [2.50, 2.90]
    base = float(np.mean(quiet + elevated))
    grown = float(np.mean(quiet + [0.32] + elevated))
    assert grown < base - 0.15, (
        "adding a quiet fold should drag the plain mean down noticeably; if it "
        "does not, this fixture no longer demonstrates the asymmetry"
    )


def test_a_four_fold_model_is_not_advantaged_by_the_regime_scalar():
    """The specific claim about FusedEpi, on controlled numbers.

    Two models with identical per-regime performance, one with 4 folds and one
    with 5, must score identically under the regime mean.
    """
    four = [0.40, 0.60, 2.00, 2.40]                        # quiet .5, elevated 2.2
    five = [0.40, 0.50, 0.60, 2.00, 2.40]                  # quiet .5, elevated 2.2
    assert _regime(four, 2) == pytest.approx(_regime(five, 3), abs=1e-12)


# ── the property, on the shipped table ───────────────────────────────────────
def _shipped():
    if not CSV.exists():
        pytest.skip("per_model_metrics.csv not present")
    out = []
    for r in csv.DictReader(CSV.open(encoding="utf-8")):
        try:
            folds = ast.literal_eval(r["oof_wis_folds"]) if r.get("oof_wis_folds") else None
            stored = float(r["oof_wis"])
        except (ValueError, SyntaxError, TypeError, KeyError):
            continue
        if folds and len(folds) >= 2 and np.isfinite(stored):
            out.append((r["model"], list(folds), stored))
    return out


def test_no_shipped_model_is_scored_on_a_plain_mean():
    """Every stored oof_wis is a regime aggregate, so the field is comparable.

    Only models where the two aggregations actually differ can distinguish them.
    A 2-2 split makes the regime mean and the plain mean identical — FusedEpi is
    that case, and matching both is a coincidence of arithmetic, not evidence of
    a different scoring path.
    """
    checked = 0
    offenders = []
    for model, folds, stored in _shipped():
        plain = _penalize(float(np.mean(folds)), folds)
        # Best guess at the split; only used to see whether the two can differ.
        k = len(folds) // 2 if len(folds) % 2 == 0 else len(folds) // 2 + 1
        regime = _penalize(_regime(sorted(folds), k), folds)
        if abs(regime - plain) < 1e-6:
            continue                       # indistinguishable — tells us nothing
        checked += 1
        if abs(plain - stored) < 1e-6:
            offenders.append(model)
    assert checked >= 10, f"only {checked} models could distinguish the two aggregations"
    assert not offenders, (
        f"these models' stored oof_wis equals a plain-mean aggregate while their "
        f"regime aggregate differs, so they are not on the same footing: {offenders}"
    )


def test_champion_wins_on_the_elevated_folds():
    """The substantive reason the champion is the champion.

    If this ever fails, the championship is resting on quiet-period performance
    and the regime weighting is not doing what it is there to do.
    """
    rows = {m: folds for m, folds, _ in _shipped()}
    if "FusedEpi" not in rows or "GAM-Spline" not in rows:
        pytest.skip("expected models absent")
    champ_elev = np.sort(rows["FusedEpi"])[len(rows["FusedEpi"]) // 2:].mean()
    rival = np.sort(rows["GAM-Spline"])[len(rows["GAM-Spline"]) // 2 + 1:].mean()
    assert champ_elev < rival, (
        f"champion elevated-fold WIS {champ_elev:.4f} is not better than the "
        f"plain-mean-preferred rival's {rival:.4f}"
    )
