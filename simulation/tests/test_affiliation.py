"""⑬-소속: school-affiliation graph (real-school-count grounded)."""
import sqlite3
import numpy as np
from simulation.abm.affiliation import (
    assign_school_clusters, load_schools_per_gu, validate_school_affiliation)


def _db(tmp_path, schools):
    db = tmp_path / "t.db"; con = sqlite3.connect(db)
    con.execute("CREATE TABLE school_info (gu_nm TEXT, school_kind TEXT)")
    for gu, n in schools.items():
        con.executemany("INSERT INTO school_info VALUES (?,?)", [(gu, "초등학교")] * n)
    con.commit(); con.close(); return str(db)


def test_load_schools_per_gu(tmp_path):
    db = _db(tmp_path, {"A": 10, "B": 3, "C": 1})
    assert list(load_schools_per_gu(db, ["A", "B", "C"])) == [10, 3, 1]


def test_only_students_get_clusters(tmp_path):
    gu = np.array([0, 0, 0, 1, 1]); bands = np.array([1, 1, 3, 1, 5])  # band1=student
    ids = assign_school_clusters(gu, bands, np.array([5, 2]))
    assert ids[2] == -1 and ids[4] == -1  # non-students
    assert ids[0] >= 0 and ids[3] >= 0


def test_validate_tracks_real_school_counts(tmp_path):
    db = _db(tmp_path, {"A": 20, "B": 10, "C": 2})
    r = validate_school_affiliation(db, ["A", "B", "C"], n_students_per_gu=3000)
    assert r["match"] is True and r["rank_corr"] >= 0.8
