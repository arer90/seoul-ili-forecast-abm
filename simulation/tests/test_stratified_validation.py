"""C (M7): stratified ABM validation — mobility-daytime matching guard."""
import sqlite3

from simulation.abm.stratified_validation import validate_mobility_daytime


def _make_db(tmp_path, real_pops):
    db = tmp_path / "t.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE commuter_matrix (origin_gu TEXT, dest_gu TEXT, "
                "coupling REAL, commuters INT, night_population REAL)")
    con.execute("CREATE TABLE daily_population_gu_hourly (gu_nm TEXT, hour INT, tot_pop REAL)")
    # row-stochastic coupling; model daytime: A=1800, B=700, C=500 (night pop 1000 each)
    rows = [("A", "A", 0.8, 0, 1000), ("A", "B", 0.1, 0, 1000), ("A", "C", 0.1, 0, 1000),
            ("B", "A", 0.4, 0, 1000), ("B", "B", 0.6, 0, 1000), ("B", "C", 0.0, 0, 1000),
            ("C", "A", 0.6, 0, 1000), ("C", "B", 0.0, 0, 1000), ("C", "C", 0.4, 0, 1000)]
    con.executemany("INSERT INTO commuter_matrix VALUES (?,?,?,?,?)", rows)
    for gu, pop in real_pops.items():
        for h in range(10, 18):
            con.execute("INSERT INTO daily_population_gu_hourly VALUES (?,?,?)", (gu, h, pop))
    con.commit(); con.close()
    return str(db)


def test_mobility_matches_when_rankings_align(tmp_path):
    db = _make_db(tmp_path, {"A": 1800, "B": 700, "C": 500})  # = model ranking
    res = validate_mobility_daytime(db)
    assert res["n_gu"] == 3
    assert res["spearman"] == 1.0  # perfect rank match
    assert "STATIC-OK" in res["verdict"]


def test_mobility_mismatch_flags_branch_a(tmp_path):
    db = _make_db(tmp_path, {"A": 400, "B": 800, "C": 1900})  # reversed → mismatch
    res = validate_mobility_daytime(db)
    assert res["spearman"] < 0.8
    assert "branch A" in res["verdict"]


def test_uses_coupling_not_empty_commuters_column(tmp_path):
    # commuters=0 everywhere (real DB reality); must use `coupling` or it's empty
    db = _make_db(tmp_path, {"A": 1800, "B": 700, "C": 500})
    res = validate_mobility_daytime(db)
    assert "error" not in res and res["n_gu"] == 3


def test_missing_table_returns_error_not_raise(tmp_path):
    db = tmp_path / "empty.db"
    sqlite3.connect(db).close()
    res = validate_mobility_daytime(str(db))
    assert "error" in res
