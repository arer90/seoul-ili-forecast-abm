"""⑤B: ABM age-stratification validation — does the contact structure reproduce
the real age-ILI ordering (school-age children highest)?"""
import sqlite3

import numpy as np

from simulation.abm.age_validation import (
    model_age_risk,
    validate_age_ili_pattern,
)

_BANDS = ["0세", "1-6세", "7-12세", "13-18세", "19-49세", "50-64세", "65세 이상"]


def _make_db(tmp_path, rates):
    db = tmp_path / "t.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE sentinel_influenza (age_group TEXT, ili_rate REAL, "
                "season_start INT, week_seq INT)")
    for band, rate in zip(_BANDS, rates):
        # two weeks per band so AVG is exercised
        con.execute("INSERT INTO sentinel_influenza VALUES (?,?,?,?)", (band, rate, 2024, 1))
        con.execute("INSERT INTO sentinel_influenza VALUES (?,?,?,?)", (band, rate, 2024, 2))
    con.commit(); con.close()
    return str(db)


def test_model_age_risk_is_valid_distribution():
    r = model_age_risk()
    assert r.shape == (7,)
    assert np.all(r >= 0) and abs(r.sum() - 1.0) < 1e-9


def test_matches_real_child_highest_pattern(tmp_path):
    # real influenza ordering: school-age children highest, elderly lowest
    db = _make_db(tmp_path, [7.5, 15.8, 27.9, 23.4, 15.0, 7.3, 4.3])
    res = validate_age_ili_pattern(db)
    assert res["child_highest_both"] is True
    assert res["spearman"] >= 0.5
    assert res["match"] is True, res["verdict"]


def test_flags_reversed_elderly_highest_pattern(tmp_path):
    # an inverted pattern (elderly highest) must NOT match the model
    db = _make_db(tmp_path, [4.3, 7.3, 8.0, 9.0, 15.0, 23.0, 30.0])
    res = validate_age_ili_pattern(db)
    assert res["match"] is False, res["verdict"]


def test_incomplete_sentinel_returns_error(tmp_path):
    db = _make_db(tmp_path, [7.5, 15.8, 27.9, 23.4, 15.0, 7.3, 4.3])
    # drop a band → incomplete
    con = sqlite3.connect(db)
    con.execute("DELETE FROM sentinel_influenza WHERE age_group = '65세 이상'")
    con.commit(); con.close()
    assert "error" in validate_age_ili_pattern(db)
