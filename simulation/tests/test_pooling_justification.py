"""A8 (M7): single-series ILI forecasting is data-forced (no per-gu ILI).

Guards the §Methods claim (docs/POOLING_JUSTIFICATION_A8_20260606.md): the ILI
forecasting target (sentinel_influenza) is NOT per-gu, so pooling to one Seoul
series is a data constraint — the 25-gu resolution is the SEIR simulation layer.
"""
import sqlite3
from pathlib import Path

import pytest

DB = Path("simulation/data/db/epi_real_seoul.db")


@pytest.mark.skipif(not DB.exists(), reason="real DB not present")
def test_sentinel_ili_is_not_per_gu():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        cols = [d[1] for d in con.execute("PRAGMA table_info(sentinel_influenza)").fetchall()]
    finally:
        con.close()
    assert cols, "sentinel_influenza missing"
    gu_cols = [c for c in cols if "gu" in c.lower()]
    assert gu_cols == [], (
        f"sentinel_influenza unexpectedly has gu columns {gu_cols} — the pooling "
        "justification (A8) assumes the ILI target is city-level, not per-gu"
    )
    assert "ili_rate" in cols and "age_group" in cols  # city-level, by age


def test_justification_doc_exists():
    assert Path("docs/POOLING_JUSTIFICATION_A8_20260606.md").exists()
