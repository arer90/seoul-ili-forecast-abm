import re
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from simulation.abm import synthetic_population as sp


N_LARGE = 50_000
SEED = 20260604
EXPECTED_KEYS = {
    "home_gu",
    "age_band",
    "sex",
    "occupation",
    "severity",
    "work_gu",
}


def _connect() -> sqlite3.Connection:
    uri = f"file:{Path(sp.DB_PATH).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _age_band(label: str) -> int:
    numbers = [int(x) for x in re.findall(r"\d+", label)]
    if not numbers:
        raise ValueError(f"Cannot parse age group: {label!r}")
    lower = numbers[0]
    if lower >= 60:
        return 6
    return lower // 10


def _normalize(values: np.ndarray) -> np.ndarray:
    total = float(values.sum())
    if total <= 0:
        return np.full_like(values, 1.0 / values.size, dtype=np.float64)
    return values / total


def _db_home_probs() -> np.ndarray:
    values = np.zeros(len(sp.GU_NAMES), dtype=np.float64)
    index = {name: i for i, name in enumerate(sp.GU_NAMES)}
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT origin_gu, AVG(night_population) AS night_pop
            FROM commuter_matrix
            GROUP BY origin_gu
            """
        ).fetchall()
    for row in rows:
        values[index[row["origin_gu"]]] = float(row["night_pop"] or 0.0)
    return _normalize(values)


def _db_age_probs_by_gu() -> np.ndarray:
    values = np.zeros((len(sp.GU_NAMES), 7), dtype=np.float64)
    index = {name: i for i, name in enumerate(sp.GU_NAMES)}
    with _connect() as conn:
        latest_year = conn.execute(
            "SELECT MAX(CAST(prd_de AS INT)) AS y FROM kosis_age_district"
        ).fetchone()["y"]
        rows = conn.execute(
            """
            SELECT gu_nm, age_group, population
            FROM kosis_age_district
            WHERE CAST(prd_de AS INT) = ?
            """,
            (latest_year,),
        ).fetchall()
    for row in rows:
        values[index[row["gu_nm"]], _age_band(row["age_group"])] += float(
            row["population"] or 0.0
        )
    return np.vstack([_normalize(row) for row in values])


def _db_occupation_probs_by_gu() -> np.ndarray:
    values = np.zeros((len(sp.GU_NAMES), len(sp.INDUSTRY_NAMES)), dtype=np.float64)
    gu_index = {name: i for i, name in enumerate(sp.GU_NAMES)}
    industry_index = {name: i for i, name in enumerate(sp.INDUSTRY_NAMES)}
    with _connect() as conn:
        latest_period = conn.execute(
            "SELECT MAX(CAST(prd_de AS INT)) AS p FROM employment_workplace"
        ).fetchone()["p"]
        rows = conn.execute(
            """
            SELECT c1_nm, c2_nm, SUM(dt) AS dt
            FROM employment_workplace
            WHERE CAST(prd_de AS INT) = ?
              AND c2_nm != '계'
            GROUP BY c1_nm, c2_nm
            """,
            (latest_period,),
        ).fetchall()
    for row in rows:
        parts = str(row["c1_nm"]).split()
        gu_name = parts[-1] if parts and parts[0] == "서울" else str(row["c1_nm"])
        if gu_name in gu_index and row["c2_nm"] in industry_index:
            values[gu_index[gu_name], industry_index[row["c2_nm"]]] += float(
                row["dt"] or 0.0
            )
    city = _normalize(values.sum(axis=0))
    out = np.empty_like(values)
    for i, row in enumerate(values):
        out[i] = _normalize(row) if row.sum() > 0 else city
    return out


@pytest.fixture(scope="module")
def pop_50k() -> dict[str, np.ndarray]:
    return sp.generate_population(N_LARGE, seed=SEED)


def _prob(values: np.ndarray, bins: int) -> np.ndarray:
    return np.bincount(values, minlength=bins).astype(np.float64) / values.size


def test_gu_marginal_matches_db(pop_50k: dict[str, np.ndarray]) -> None:
    observed = _prob(pop_50k["home_gu"], len(sp.GU_NAMES))
    expected = _db_home_probs()
    assert np.max(np.abs(observed - expected)) < 0.03


def test_age_marginal_matches_db(pop_50k: dict[str, np.ndarray]) -> None:
    observed = _prob(pop_50k["age_band"], 7)
    expected = _db_home_probs() @ _db_age_probs_by_gu()
    assert np.max(np.abs(observed - expected)) < 0.03


def test_age_conditional_differs_by_gu(pop_50k: dict[str, np.ndarray]) -> None:
    db_age = _db_age_probs_by_gu()
    distances = np.abs(db_age[:, None, :] - db_age[None, :, :]).sum(axis=2)
    i, j = np.unravel_index(np.argmax(distances), distances.shape)

    home = pop_50k["home_gu"]
    age = pop_50k["age_band"]
    observed_i = _prob(age[home == i], 7)
    observed_j = _prob(age[home == j], 7)
    assert np.abs(observed_i - observed_j).sum() > 0.05


def test_sex_marginal(pop_50k: dict[str, np.ndarray]) -> None:
    female_fraction = float(pop_50k["sex"].mean())
    assert 0.45 <= female_fraction <= 0.55


def test_occupation_marginal(pop_50k: dict[str, np.ndarray]) -> None:
    observed = _prob(pop_50k["occupation"], len(sp.INDUSTRY_NAMES))
    expected = _db_home_probs() @ _db_occupation_probs_by_gu()
    assert np.max(np.abs(observed - expected)) < 0.05


def test_severity_valid(pop_50k: dict[str, np.ndarray]) -> None:
    severity = pop_50k["severity"]
    age = pop_50k["age_band"]
    assert set(np.unique(severity)).issubset({0, 1})

    rates = np.array([severity[age == band].mean() for band in range(7)])
    assert rates[-1] > rates[0] + 0.03
    assert np.corrcoef(np.arange(7), rates)[0, 1] > 0.80
    assert np.count_nonzero(np.diff(rates) < -0.015) <= 1


def test_work_gu_conditional(pop_50k: dict[str, np.ndarray]) -> None:
    home = pop_50k["home_gu"]
    work = pop_50k["work_gu"]
    dist_0 = _prob(work[home == 0], len(sp.GU_NAMES))
    dist_12 = _prob(work[home == 12], len(sp.GU_NAMES))
    assert np.abs(dist_0 - dist_12).sum() > 0.01


def test_determinism() -> None:
    first = sp.generate_population(1000, seed=42)
    second = sp.generate_population(1000, seed=42)
    for key in EXPECTED_KEYS:
        np.testing.assert_array_equal(first[key], second[key])


def test_different_seeds() -> None:
    first = sp.generate_population(1000, seed=42)
    second = sp.generate_population(1000, seed=99)
    assert not np.array_equal(first["home_gu"], second["home_gu"])


def test_no_invalid(pop_50k: dict[str, np.ndarray]) -> None:
    assert pop_50k["home_gu"].min() >= 0
    assert pop_50k["home_gu"].max() <= 24
    assert pop_50k["age_band"].min() >= 0
    assert pop_50k["age_band"].max() <= 6
    assert set(np.unique(pop_50k["sex"])).issubset({0, 1})
    assert pop_50k["occupation"].min() >= 0
    assert set(np.unique(pop_50k["severity"])).issubset({0, 1})
    assert pop_50k["work_gu"].min() >= 0
    assert pop_50k["work_gu"].max() <= 24
    for array in pop_50k.values():
        assert np.isfinite(array).all()


def test_small_n() -> None:
    pop = sp.generate_population(1000, seed=7)
    assert set(pop) == EXPECTED_KEYS
    assert all(array.shape == (1000,) for array in pop.values())
    assert pop["home_gu"].dtype == np.int8
    assert pop["age_band"].dtype == np.int8
    assert pop["sex"].dtype == np.int8
    assert pop["occupation"].dtype == np.int16
    assert pop["severity"].dtype == np.int8
    assert pop["work_gu"].dtype == np.int8
    with pytest.raises(ValueError, match="N must be >= 1"):
        sp.generate_population(0, seed=7)
