"""⑧A: per-agent location foundation — working-age daytime commute signature."""
import sqlite3

_AGE_COLS = ["pop_0_9", "pop_10_19", "pop_20_29", "pop_30_39",
             "pop_40_49", "pop_50_59", "pop_60_69", "pop_70plus"]
_WORKING_IDX = [2, 3, 4, 5]  # 20-59


def _make_db(tmp_path, rows):
    """rows: list of (gu, hour, working_pop, nonworking_pop)."""
    db = tmp_path / "t.db"
    con = sqlite3.connect(db)
    cols = ", ".join(f"{c} REAL" for c in _AGE_COLS)
    con.execute(f"CREATE TABLE daily_population_gu_hourly (gu_nm TEXT, hour INT, {cols})")
    ph = ",".join("?" * (2 + len(_AGE_COLS)))
    for gu, hour, work, nonwork in rows:
        vals = [work if i in _WORKING_IDX else nonwork for i in range(len(_AGE_COLS))]
        con.execute(f"INSERT INTO daily_population_gu_hourly VALUES ({ph})", (gu, hour, *vals))
    con.commit(); con.close()
    return str(db)


def _gu(gu, day_work, day_non, night_work, night_non):
    rows = []
    for h in range(0, 7):
        rows.append((gu, h, night_work, night_non))
    for h in range(10, 18):
        rows.append((gu, h, day_work, day_non))
    return rows


def test_load_shares_reads_commute(tmp_path):
    from simulation.abm.agent_mobility import load_age_hub_shares
    # 중구 = hub: workers commute in by day; 은평구 residential
    rows = _gu("중구", 800, 150, 100, 100) + _gu("은평구", 200, 900, 1000, 1000)
    sh = load_age_hub_shares(_make_db(tmp_path, rows))
    assert sh["pop_30_39"]["shift"] > sh["pop_0_9"]["shift"]  # workers shift more
    assert sh["pop_30_39"]["shift"] > 0


def test_validate_working_age_commute_match(tmp_path):
    from simulation.abm.agent_mobility import validate_working_age_commute
    rows = _gu("중구", 800, 150, 100, 100) + _gu("은평구", 200, 900, 1000, 1000)
    res = validate_working_age_commute(_make_db(tmp_path, rows))
    assert res["working_positive"] is True
    assert res["working_gt_nonworking"] is True
    assert res["match"] is True, res["verdict"]


def test_validate_flags_no_commute_gradient(tmp_path):
    from simulation.abm.agent_mobility import validate_working_age_commute
    # everyone identical day/night → no shift for anyone
    rows = _gu("중구", 100, 100, 100, 100) + _gu("은평구", 100, 100, 100, 100)
    res = validate_working_age_commute(_make_db(tmp_path, rows))
    assert res["match"] is False


def test_incomplete_bands_error(tmp_path):
    from simulation.abm.agent_mobility import validate_working_age_commute
    db = tmp_path / "e.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE daily_population_gu_hourly (gu_nm TEXT, hour INT, pop_0_9 REAL)")
    con.commit(); con.close()
    assert "error" in validate_working_age_commute(str(db))


def test_daytime_location_sampler(tmp_path):
    import numpy as np
    from simulation.abm.agent_mobility import assign_daytime_location, load_daytime_location_dist
    # 2 gu, band 2 always in gu0, band 6 always in gu1
    dist = np.array([[0.5, 0.5]] * 7)
    dist[2] = [1.0, 0.0]; dist[6] = [0.0, 1.0]
    bands = np.array([2, 2, 6, 6])
    loc = assign_daytime_location(bands, dist)
    assert (loc[bands == 2] == 0).all() and (loc[bands == 6] == 1).all()
