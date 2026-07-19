"""⑬-환경: weather→flu environmental coupling (real-data validation)."""
import sqlite3
from simulation.abm.environment import load_weather_monthly, validate_weather_flu_coupling


def _db(tmp_path):
    db = tmp_path / "t.db"; con = sqlite3.connect(db)
    con.execute("CREATE TABLE weather_historical (obs_date TEXT, ta_avg REAL)")
    con.execute("CREATE TABLE google_search_trends (period TEXT, geo TEXT, keyword TEXT, interest REAL)")
    for yr in (2021, 2022, 2023):
        for mo in range(1, 13):
            temp = -5.0 if mo in (12, 1, 2) else (28.0 if mo in (6, 7, 8) else 15.0)
            flu = 80.0 if mo in (12, 1, 2) else (10.0 if mo in (6, 7, 8) else 30.0)
            con.executemany("INSERT INTO weather_historical VALUES (?,?)",
                            [(f"{yr}{mo:02d}{d:02d}", temp) for d in (1, 15)])
            con.execute("INSERT INTO google_search_trends VALUES (?,?,?,?)",
                        (f"{yr}-{mo:02d}-01", "KR", "독감", flu))
    con.commit(); con.close(); return str(db)


def test_weather_loads_monthly(tmp_path):
    w = load_weather_monthly(_db(tmp_path))
    assert w["2022-01"] == -5.0 and w["2022-07"] == 28.0


def test_cold_drives_flu(tmp_path):
    r = validate_weather_flu_coupling(_db(tmp_path))
    assert r["temp_flu_corr"] < -0.3 and r["match"] is True
    assert r["winter_summer_ratio"] > 1.0
