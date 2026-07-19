"""Step-2: split-conformal vs CV+ A/B summary (phase11_scoring)."""
from __future__ import annotations

import math
import numpy as np


def test_ab_summary_recommends_cv_plus_when_tighter_and_covers():
    from simulation.pipeline.scoring import compute_pi_ab_summary

    pi = {
        "M1": {
            "conformal": {"width": 20.0, "coverage": 0.92, "source": "holdout"},
            "cv_plus":   {"width": 18.0, "coverage": 0.94, "n_folds": 5, "n_cal": 40},
        },
        "M2": {
            "conformal": {"width": 22.0, "coverage": 0.95, "source": "holdout"},
            "cv_plus":   {"width": 21.0, "coverage": 0.95, "n_folds": 5, "n_cal": 40},
        },
        "M3": {
            "conformal": {"width": 18.0, "coverage": 0.90, "source": "holdout"},
            "cv_plus":   {"width": 17.0, "coverage": 0.93, "n_folds": 5, "n_cal": 40},
        },
        "_meta": {"x": 1},
    }
    out = compute_pi_ab_summary(pi)
    agg = out["aggregate"]

    assert agg["n_with_both"] == 3
    assert agg["n_with_split_only"] == 0
    assert agg["recommendation"] == "cv_plus"
    assert agg["width_ratio_median"] < 1.0
    # CV+ should be tighter in all three
    assert math.isclose(agg["cv_plus_tighter_pct"], 1.0)


def test_ab_summary_prefers_split_when_cv_plus_wider():
    from simulation.pipeline.scoring import compute_pi_ab_summary

    # CV+ is uniformly wider; both still cover → pick the tighter (split).
    pi = {
        f"M{i}": {
            "conformal": {"width": 20.0, "coverage": 0.95, "source": "holdout"},
            "cv_plus":   {"width": 25.0, "coverage": 0.96, "n_folds": 5, "n_cal": 40},
        }
        for i in range(5)
    }
    out = compute_pi_ab_summary(pi)
    assert out["aggregate"]["recommendation"] == "split"
    assert out["aggregate"]["width_ratio_median"] > 1.05


def test_ab_summary_tie_when_ratio_near_unity():
    from simulation.pipeline.scoring import compute_pi_ab_summary

    pi = {
        f"M{i}": {
            "conformal": {"width": 20.0, "coverage": 0.94, "source": "holdout"},
            "cv_plus":   {"width": 20.0 * (1.00 + 0.01 * (i - 2)),
                           "coverage": 0.95, "n_folds": 5, "n_cal": 40},
        }
        for i in range(5)
    }
    out = compute_pi_ab_summary(pi)
    agg = out["aggregate"]
    assert agg["recommendation"] == "tie"
    assert 0.95 <= agg["width_ratio_median"] <= 1.05


def test_ab_summary_insufficient_when_no_cv_plus_entries():
    from simulation.pipeline.scoring import compute_pi_ab_summary

    pi = {
        "M1": {"conformal": {"width": 20.0, "coverage": 0.9, "source": "holdout"},
               "cv_plus": {}},
        "M2": {"conformal": {"width": 19.0, "coverage": 0.91, "source": "holdout"},
               "cv_plus": {}},
    }
    out = compute_pi_ab_summary(pi)
    agg = out["aggregate"]
    assert agg["n_with_both"] == 0
    assert agg["n_with_split_only"] == 2
    assert agg["recommendation"] == "insufficient_data"


def test_ab_summary_neither_covers_flagged():
    from simulation.pipeline.scoring import compute_pi_ab_summary

    pi = {
        f"M{i}": {
            "conformal": {"width": 20.0, "coverage": 0.70, "source": "holdout"},
            "cv_plus":   {"width": 19.0, "coverage": 0.75, "n_folds": 5, "n_cal": 40},
        }
        for i in range(4)
    }
    out = compute_pi_ab_summary(pi)
    assert out["aggregate"]["recommendation"] == "neither_covers"


def test_ab_summary_rejects_meta_keys_and_non_dict_entries():
    from simulation.pipeline.scoring import compute_pi_ab_summary

    pi = {
        "_diagnostics": {"should": "be_skipped"},
        "_meta": {"x": 1},
        "bad_entry": "not-a-dict",
        "M1": {"conformal": {"width": 20.0, "coverage": 0.95},
               "cv_plus": {"width": 18.0, "coverage": 0.96, "n_folds": 3, "n_cal": 20}},
    }
    out = compute_pi_ab_summary(pi)
    assert out["aggregate"]["n_total"] == 1
    assert out["per_model"][0]["model"] == "M1"


def test_ab_summary_nan_safe_with_missing_fields():
    from simulation.pipeline.scoring import compute_pi_ab_summary

    pi = {
        "M1": {"conformal": {"width": 20.0, "coverage": 0.95}, "cv_plus": {}},
        "M2": {"conformal": {}, "cv_plus": {"width": 18.0, "coverage": 0.95, "n_folds": 5, "n_cal": 40}},
    }
    out = compute_pi_ab_summary(pi)
    # Neither model has both → recommendation falls through to insufficient_data.
    agg = out["aggregate"]
    assert agg["n_with_both"] == 0
    assert agg["n_with_split_only"] == 1
    assert agg["n_with_cv_plus_only"] == 1
    assert agg["recommendation"] == "insufficient_data"
    # Width medians still computed from whichever side has data.
    assert np.isfinite(agg["split_width_median"])
    assert np.isfinite(agg["cv_plus_width_median"])
