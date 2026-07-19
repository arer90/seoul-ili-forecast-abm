"""⑤A: time-resolved mobility — validated daytime-amplification foundation."""
import sqlite3


def _make_db(tmp_path, rows):
    """rows: list of (gu, hour, tot_pop)."""
    db = tmp_path / "t.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE daily_population_gu_hourly (gu_nm TEXT, hour INT, tot_pop REAL)")
    con.executemany("INSERT INTO daily_population_gu_hourly VALUES (?,?,?)", rows)
    con.commit(); con.close()
    return str(db)


def _gu_rows(gu, day_pop, night_pop):
    out = []
    for h in range(0, 7):
        out.append((gu, h, night_pop))
    for h in range(10, 18):
        out.append((gu, h, day_pop))
    return out


def test_amplification_ratio_reads_swing(tmp_path):
    from simulation.abm.mobility_timeresolved import load_daytime_amplification
    rows = _gu_rows("중구", 400.0, 200.0) + _gu_rows("은평구", 80.0, 100.0)
    amp = load_daytime_amplification(_make_db(tmp_path, rows))
    assert abs(amp["중구"] - 2.0) < 1e-9      # business hub gains
    assert amp["은평구"] < 1.0                # residential empties


def test_validate_detects_located_swing(tmp_path):
    from simulation.abm.mobility_timeresolved import validate_temporal_mobility_swing
    rows = (_gu_rows("중구", 400.0, 200.0) + _gu_rows("종로구", 350.0, 200.0)
            + _gu_rows("강남구", 480.0, 330.0) + _gu_rows("서초구", 300.0, 240.0)
            + _gu_rows("영등포구", 300.0, 230.0) + _gu_rows("은평구", 80.0, 100.0)
            + _gu_rows("관악구", 85.0, 110.0))
    res = validate_temporal_mobility_swing(_make_db(tmp_path, rows))
    assert res["hubs_amplify"] is True
    assert res["swing_ratio"] >= 1.5
    assert res["detected"] is True, res["verdict"]


def test_validate_flags_flat_no_swing(tmp_path):
    from simulation.abm.mobility_timeresolved import validate_temporal_mobility_swing
    # every district flat (day == night) → no swing
    rows = []
    for gu in ("중구", "종로구", "강남구", "서초구", "영등포구", "은평구"):
        rows += _gu_rows(gu, 100.0, 100.0)
    res = validate_temporal_mobility_swing(_make_db(tmp_path, rows))
    assert res["detected"] is False


def test_too_few_districts_returns_error(tmp_path):
    from simulation.abm.mobility_timeresolved import validate_temporal_mobility_swing
    res = validate_temporal_mobility_swing(_make_db(tmp_path, _gu_rows("중구", 400.0, 200.0)))
    assert "error" in res
