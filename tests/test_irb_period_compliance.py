"""IRB period-compliance guard — the *modeling* data must lie within the approved window.

The KUIRB research plan (연구계획서_세부내역_KUIRB_v5 §2.1) approves the public,
de-identified KDCA sentinel ILI data for the **2019-2025 influenza seasons**
(341 raw weeks, ending 2026-03-15). The IRB constraint binds the data the study
*analyzes* — i.e. the train / validation / test split the model is fit on and
evaluated against.

The **forward window is intentionally NOT bounded here**. The forward forecast is
a model *output* (a prediction of weeks the model never trained on), not analyzed
study data, so whether its target dates fall after the IRB window is irrelevant
to the approval. Only train/val/test must stay inside it.

By construction the modeling boundary is fixed by ``paper_cutoff_week = 337``
(config.py §3: "341 raw 에서 결측 ~4 제거 → 337"), which lands at 2026-02-15 —
inside the IRB window regardless of how much later data the collectors gather.
This test fails if that boundary ever drifts past the IRB cutoff (e.g. someone
raises paper_cutoff_week, or the early-week dates shift), so post-IRB weeks can
never silently enter training/selection/evaluation.

Run (per-file, macOS): .venv/bin/python -m pytest tests/test_irb_period_compliance.py -q
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path

import polars as pl
import pytest

_ROOT = Path(__file__).resolve().parents[1]

# IRB-approved data window (KUIRB v5 §2.1): 2019-2025 seasons = 341 weeks.
IRB_N_WEEKS = 341
IRB_FIRST_WEEK = _dt.date(2019, 9, 8)        # feature-cache week 0
IRB_CUTOFF = _dt.date(2026, 3, 15)           # week 341 — end of the approved window
# The modeling split ends at the in-sample / test boundary (paper_cutoff_week).
PAPER_CUTOFF_WEEK = 337                       # config.py §3 (341 raw − ~4 missing)

_CACHE = _ROOT / "simulation" / "cache" / "feature_cache.parquet"


def test_irb_window_is_341_weeks() -> None:
    """The documented IRB window must be exactly 341 weeks ending 2026-03-15."""
    weeks = (IRB_CUTOFF - IRB_FIRST_WEEK).days // 7 + 1
    assert weeks == IRB_N_WEEKS, (
        f"IRB window {IRB_FIRST_WEEK}..{IRB_CUTOFF} spans {weeks} weeks, "
        f"expected {IRB_N_WEEKS} — constants drifted from the KUIRB plan"
    )


def test_modeling_data_within_irb() -> None:
    """train/val/test (everything ≤ paper_cutoff_week) must end inside the IRB window.

    The forward window (rows past paper_cutoff_week) is a forecast and is
    deliberately exempt — this asserts ONLY that the modeling/evaluation data
    stays within the IRB-approved 2019-2025 period.
    """
    if not _CACHE.exists():
        pytest.skip(f"{_CACHE} absent (build the feature cache first)")
    dates = pl.read_parquet(_CACHE, columns=["week_start"])["week_start"].to_list()
    assert len(dates) >= PAPER_CUTOFF_WEEK, (
        f"feature cache has {len(dates)} weeks (< paper_cutoff {PAPER_CUTOFF_WEEK})"
    )
    boundary = _dt.date.fromisoformat(str(dates[PAPER_CUTOFF_WEEK - 1])[:10])
    assert boundary <= IRB_CUTOFF, (
        f"in-sample/test boundary (week {PAPER_CUTOFF_WEEK}) = {boundary} > IRB "
        f"cutoff {IRB_CUTOFF} — modeling data has drifted past the IRB window; "
        f"the forward forecast may run later, but train/val/test must not"
    )
