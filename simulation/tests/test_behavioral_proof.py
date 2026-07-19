"""Pillar-1 proof: observed behavioral signal = risk + fatigue decomposition.

Guards the empirical finding that the ABM's two mechanisms (risk perception α,
fatigue F) are BOTH present in the independent transit-mobility data — and that
the naive pooled dose-response is hysteresis-confounded (the trap).
"""
import csv
import sqlite3

import numpy as np

from simulation.abm.behavioral_proof import (
    behavioral_decomposition,
    build_behavior_panel,
    confounding_check,
    load_covid_kr_monthly,
    load_vax_kr_monthly,
)


def _synthetic_panel(b_risk=-0.5, b_fatigue=0.3):
    # cases NON-collinear with time so the 2-var OLS can separate the effects
    cases = [1000, 100, 5000, 200, 8000, 500, 12000, 300]
    panel = []
    for i, c in enumerate(cases):
        dev = b_risk * np.log1p(c) * 0.01 + b_fatigue * i * 0.01
        panel.append({"ym": f"2020-{i + 3:02d}", "regime": "R1", "covid": c,
                      "mobility_dev": dev})
    return panel


def test_decomposition_recovers_coefficients_but_flags_confounding():
    # on synthetic NON-collinear data the 2-var OLS still recovers the signs,
    # but the verdict/caveat must NOT over-claim "fatigue" — it is confounded.
    d = behavioral_decomposition(_synthetic_panel(b_risk=-0.5, b_fatigue=0.3))
    assert d["beta_risk"] < 0, "risk response not recovered"
    assert d["beta_fatigue"] > 0, "fatigue not recovered"
    assert "both observed" not in d["verdict"], "must not over-claim"
    assert "CONFOUNDED" in d["verdict"] and "caveat" in d
    assert "vaccination" in d["caveat"]


def test_confounding_check_flags_collinear_time_and_vaccination(tmp_path):
    # vaccination rises monotonically with time (the real Korea reality) → the
    # time-trend ('fatigue') is NOT separable from vaccination.
    panel = [{"ym": f"2021-{m:02d}", "regime": "R1", "covid": 1000 * (m % 3 + 1),
              "mobility_dev": -0.3 + 0.02 * m} for m in range(1, 13)]
    vax = tmp_path / "vax.csv"
    with vax.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["location", "date", "people_fully_vaccinated_per_hundred"])
        for m in range(1, 13):  # monotone with time
            w.writerow(["South Korea", f"2021-{m:02d}-15", str(m * 7.0)])
    res = confounding_check(panel, str(vax))
    assert res["time_vax_corr"] > 0.9, "should detect collinearity"
    assert res["separable"] is False
    assert "NOT separable" in res["verdict"]


def test_vax_loader_monthly_mean(tmp_path):
    p = tmp_path / "v.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["location", "date", "people_fully_vaccinated_per_hundred"])
        w.writerow(["South Korea", "2021-06-01", "4.0"])
        w.writerow(["South Korea", "2021-06-15", "6.0"])
    m = load_vax_kr_monthly(str(p))
    assert abs(m["2021-06"] - 5.0) < 1e-9


def test_fatigue_dominated_panel_flips_naive_slope():
    # strong fatigue + modest risk → naive pooled slope can come out POSITIVE
    # (the hysteresis trap) while the decomposition still finds β_risk<0.
    d = behavioral_decomposition(_synthetic_panel(b_risk=-0.3, b_fatigue=0.9))
    assert d["beta_risk"] < 0 and d["beta_fatigue"] > 0


def test_too_few_rows_returns_error():
    assert "error" in behavioral_decomposition([{"regime": "R1", "covid": 1,
                                                 "mobility_dev": 0.0}])


def test_covid_loader_aggregates_weekly_to_monthly(tmp_path):
    p = tmp_path / "covid.csv"
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Date_reported", "Country_code", "New_cases"])
        w.writerow(["2020-03-01", "KR", "100"])
        w.writerow(["2020-03-08", "KR", "200"])
        w.writerow(["2020-04-01", "KR", "50"])
    m = load_covid_kr_monthly(str(p))
    assert m["2020-03"] == 300 and m["2020-04"] == 50


def test_panel_deseasonalises_against_r0_baseline(tmp_path):
    db = tmp_path / "t.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE monthly_subway_hourly (use_ym TEXT, ride_cnt REAL)")
    # R0 baseline for March = 1000; a later March at 600 → dev = -0.4
    con.executemany("INSERT INTO monthly_subway_hourly VALUES (?,?)",
                    [("201903", 1000.0), ("202003", 600.0), ("201904", 1000.0)])
    con.commit(); con.close()
    cov = tmp_path / "c.csv"
    cov.write_text("Date_reported,Country_code,New_cases\n2020-03-01,KR,500\n", encoding="utf-8")
    panel = build_behavior_panel(str(db), str(cov))
    mar20 = next((r for r in panel if r["ym"] == "2020-03"), None)
    assert mar20 is not None and abs(mar20["mobility_dev"] - (-0.4)) < 1e-9
    assert mar20["regime"] == "R1"
