"""A2 (M7 SCI-grade): one WIS definition across the test-slab evaluation.

The champion is selected on empirical residual-quantile WIS; the pairwise
tournament, bootstrap WIS-CI, and persistence skill-denominator now use the SAME
empirical WIS — previously they used the Gaussian-σ closed form (self-DEPRECATED
for ILI), so the relative-WIS tournament ranked models on an assumption the
champion ranking rejects. A referee recomputing the table would catch that.
"""
import re
from pathlib import Path

import numpy as np

from simulation.analytics.diagnostics import (
    weighted_interval_score,
    weighted_interval_score_empirical,
)


def test_empirical_and_gaussian_wis_differ_on_skewed_residuals():
    """On right-skewed residuals the two definitions diverge → which one the
    tournament uses materially changes the ranking (motivates the migration)."""
    rng = np.random.default_rng(0)
    y = rng.gamma(2.0, 2.0, size=60)
    pred = y + rng.normal(0, 1, size=60)
    resid = rng.gamma(2.0, 2.0, size=200) - 4.0  # right-skewed
    emp = float(np.mean(weighted_interval_score_empirical(y, pred, resid)))
    gau = float(np.mean(weighted_interval_score(y, pred, float(np.std(resid)))))
    assert np.isfinite(emp) and np.isfinite(gau)
    assert abs(emp - gau) > 1e-6, "empirical and Gaussian WIS must be distinct scales"


def test_per_model_eval_has_no_gaussian_wis_call():
    """Guard: every WIS call in per_model_eval is the empirical variant."""
    src = Path("simulation/pipeline/per_model_eval.py").read_text(encoding="utf-8")
    bare = re.findall(r"weighted_interval_score\(", src)  # excludes *_empirical(
    assert bare == [], f"{len(bare)} Gaussian WIS call(s) still in per_model_eval"
