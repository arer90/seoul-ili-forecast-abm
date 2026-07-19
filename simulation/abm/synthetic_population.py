"""DB-grounded synthetic population generator for the ABM stage 3a."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
from sqlite3 import Connection, Row

import numpy as np


DB_PATH = Path(__file__).parent.parent / "data" / "db" / "epi_real_seoul.db"

_AGE_BANDS = 7
_SEOUL_GU_COUNT = 25
_FALLBACK_SEVERITY = np.array(
    [0.02, 0.03, 0.04, 0.05, 0.08, 0.12, 0.20], dtype=np.float64
)


@dataclass(frozen=True)
class _PopulationInputs:
    gu_names: tuple[str, ...]
    industry_names: tuple[str, ...]
    home_probs: np.ndarray
    age_probs_by_gu: np.ndarray
    sex_probs_by_gu: np.ndarray
    occupation_probs_by_gu: np.ndarray
    severity_probs_by_sex_age: np.ndarray
    work_probs_by_home: np.ndarray


def _connect() -> Connection:
    # G-116/G-117: project SSOT connection (PRAGMA quick_check + WAL hygiene), not a
    # raw DB-API connect. Read-only by usage; DB integrity verified at pipeline start.
    from simulation.database import safe_connect
    conn = safe_connect(str(DB_PATH), verify=False)
    conn.row_factory = Row
    return conn


def _columns(conn: Connection, table: str) -> set[str]:
    return {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}


def _table_exists(conn: Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _normalize_gu_name(value: object) -> str:
    text = str(value or "").strip()
    if text.startswith("서울특별시 "):
        text = text.removeprefix("서울특별시 ").strip()
    if text.startswith("서울 "):
        text = text.removeprefix("서울 ").strip()
    return text


def _normalize_prob(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    total = float(arr.sum())
    if total <= 0.0 or not np.isfinite(total):
        return np.full(arr.shape, 1.0 / arr.size, dtype=np.float64)
    return arr / total


def _normalize_rows(values: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).copy()
    if arr.ndim != 2:
        raise ValueError("row probabilities must be a 2D matrix")
    if fallback is None:
        fallback = np.full(arr.shape[1], 1.0 / arr.shape[1], dtype=np.float64)
    else:
        fallback = _normalize_prob(fallback)

    for i in range(arr.shape[0]):
        total = float(arr[i].sum())
        if total <= 0.0 or not np.isfinite(total):
            arr[i] = fallback
        else:
            arr[i] /= total
    return arr


def _age_band_from_label(label: object) -> int | None:
    numbers = [int(value) for value in re.findall(r"\d+", str(label or ""))]
    if not numbers:
        return None
    lower = numbers[0]
    if lower >= 60:
        return 6
    band = lower // 10
    if 0 <= band < 6:
        return band
    return None


def _sex_index(label: object) -> int | None:
    text = str(label or "").strip().lower()
    if text in {"0", "m", "male", "남", "남자"}:
        return 0
    if text in {"1", "f", "female", "여", "여자"}:
        return 1
    return None


def _period_text(
    conn: Connection,
    table: str,
    column: str,
    *,
    year: int | None,
    annual: bool,
) -> str | None:
    year_expr = f"CAST({column} AS INT)" if annual else f"CAST(SUBSTR({column}, 1, 4) AS INT)"
    if year is not None:
        row = conn.execute(
            f"""
            SELECT {column} AS period
            FROM {table}
            WHERE {year_expr} <= ?
            ORDER BY {year_expr} DESC, CAST({column} AS INT) DESC
            LIMIT 1
            """,
            (year,),
        ).fetchone()
        if row is not None:
            return str(row["period"])

    row = conn.execute(
        f"""
        SELECT {column} AS period
        FROM {table}
        ORDER BY {year_expr} DESC, CAST({column} AS INT) DESC
        LIMIT 1
        """
    ).fetchone()
    return None if row is None else str(row["period"])


def _period_int(conn: Connection, table: str, column: str, year: int | None) -> int | None:
    if year is not None:
        row = conn.execute(
            f"""
            SELECT {column} AS period
            FROM {table}
            WHERE CAST({column} AS INT) <= ?
            ORDER BY CAST({column} AS INT) DESC
            LIMIT 1
            """,
            (year,),
        ).fetchone()
        if row is not None:
            return int(row["period"])

    row = conn.execute(
        f"""
        SELECT {column} AS period
        FROM {table}
        ORDER BY CAST({column} AS INT) DESC
        LIMIT 1
        """
    ).fetchone()
    return None if row is None else int(row["period"])


def _load_gu_names(conn: Connection) -> tuple[str, ...]:
    names: set[str] = set()
    if _table_exists(conn, "commuter_matrix"):
        cols = _columns(conn, "commuter_matrix")
        if {"origin_gu", "dest_gu"}.issubset(cols):
            rows = conn.execute(
                """
                SELECT origin_gu AS gu FROM commuter_matrix
                UNION
                SELECT dest_gu AS gu FROM commuter_matrix
                """
            ).fetchall()
            names = {_normalize_gu_name(row["gu"]) for row in rows if row["gu"]}

    if len(names) != _SEOUL_GU_COUNT and _table_exists(conn, "kosis_age_district"):
        cols = _columns(conn, "kosis_age_district")
        if "gu_nm" in cols:
            rows = conn.execute(
                "SELECT DISTINCT gu_nm AS gu FROM kosis_age_district WHERE gu_nm IS NOT NULL"
            ).fetchall()
            names = {_normalize_gu_name(row["gu"]) for row in rows if row["gu"]}

    gu_names = tuple(sorted(names))
    if len(gu_names) != _SEOUL_GU_COUNT:
        raise RuntimeError(
            f"expected 25 Seoul gu names from DB, found {len(gu_names)}"
        )
    return gu_names


def _load_industry_names(conn: Connection) -> tuple[str, ...]:
    if not _table_exists(conn, "employment_workplace"):
        return ("unavailable",)
    cols = _columns(conn, "employment_workplace")
    if not {"c2", "c2_nm"}.issubset(cols):
        return ("unavailable",)

    rows = conn.execute(
        """
        SELECT c2_nm, MIN(c2) AS c2_order
        FROM employment_workplace
        WHERE c2_nm IS NOT NULL
          AND c2_nm != '계'
        GROUP BY c2_nm
        ORDER BY c2_order
        """
    ).fetchall()
    names = tuple(str(row["c2_nm"]) for row in rows if row["c2_nm"])
    return names if names else ("unavailable",)


def _load_home_probs(
    conn: Connection, gu_names: tuple[str, ...], year: int | None
) -> np.ndarray:
    gu_index = {name: i for i, name in enumerate(gu_names)}
    values = np.zeros(len(gu_names), dtype=np.float64)

    if _table_exists(conn, "commuter_matrix"):
        cols = _columns(conn, "commuter_matrix")
        if {"origin_gu", "night_population"}.issubset(cols):
            where = ""
            if "collected_at" in cols:
                where = """
                WHERE collected_at = (
                    SELECT MAX(collected_at) FROM commuter_matrix
                    WHERE night_population IS NOT NULL
                )
                """
            rows = conn.execute(
                f"""
                SELECT origin_gu, AVG(night_population) AS weight
                FROM commuter_matrix
                {where}
                GROUP BY origin_gu
                """
            ).fetchall()
            for row in rows:
                gu = _normalize_gu_name(row["origin_gu"])
                if gu in gu_index:
                    values[gu_index[gu]] = float(row["weight"] or 0.0)
            if values.sum() > 0:
                return _normalize_prob(values)

    values = _load_kosis_gu_totals(conn, gu_names, year=year)
    return _normalize_prob(values)


def _load_kosis_gu_totals(
    conn: Connection, gu_names: tuple[str, ...], year: int | None
) -> np.ndarray:
    values = np.zeros(len(gu_names), dtype=np.float64)
    if not _table_exists(conn, "kosis_age_district"):
        return values
    cols = _columns(conn, "kosis_age_district")
    if not {"prd_de", "gu_nm", "population"}.issubset(cols):
        return values

    period = _period_text(conn, "kosis_age_district", "prd_de", year=year, annual=True)
    if period is None:
        return values

    gu_index = {name: i for i, name in enumerate(gu_names)}
    rows = conn.execute(
        """
        SELECT gu_nm, SUM(population) AS population
        FROM kosis_age_district
        WHERE CAST(prd_de AS INT) = CAST(? AS INT)
        GROUP BY gu_nm
        """,
        (period,),
    ).fetchall()
    for row in rows:
        gu = _normalize_gu_name(row["gu_nm"])
        if gu in gu_index:
            values[gu_index[gu]] = float(row["population"] or 0.0)
    return values


def _load_age_probs(
    conn: Connection, gu_names: tuple[str, ...], year: int | None
) -> np.ndarray:
    values = np.zeros((len(gu_names), _AGE_BANDS), dtype=np.float64)
    if not _table_exists(conn, "kosis_age_district"):
        return _normalize_rows(values)
    cols = _columns(conn, "kosis_age_district")
    if not {"prd_de", "gu_nm", "age_group", "population"}.issubset(cols):
        return _normalize_rows(values)

    period = _period_text(conn, "kosis_age_district", "prd_de", year=year, annual=True)
    if period is None:
        return _normalize_rows(values)

    gu_index = {name: i for i, name in enumerate(gu_names)}
    rows = conn.execute(
        """
        SELECT gu_nm, age_group, population
        FROM kosis_age_district
        WHERE CAST(prd_de AS INT) = CAST(? AS INT)
        """,
        (period,),
    ).fetchall()
    for row in rows:
        gu = _normalize_gu_name(row["gu_nm"])
        band = _age_band_from_label(row["age_group"])
        if gu in gu_index and band is not None:
            values[gu_index[gu], band] += float(row["population"] or 0.0)

    city = values.sum(axis=0)
    return _normalize_rows(values, fallback=city)


def _load_sex_probs(
    conn: Connection, gu_names: tuple[str, ...], year: int | None
) -> np.ndarray:
    values = np.zeros((len(gu_names), 2), dtype=np.float64)
    gu_index = {name: i for i, name in enumerate(gu_names)}

    if _table_exists(conn, "daily_population_gu_hourly"):
        cols = _columns(conn, "daily_population_gu_hourly")
        if {"stdr_de", "gu_nm", "male_pop", "female_pop"}.issubset(cols):
            period = _period_text(
                conn,
                "daily_population_gu_hourly",
                "stdr_de",
                year=year,
                annual=False,
            )
            if period is not None:
                rows = conn.execute(
                    """
                    SELECT gu_nm, SUM(male_pop) AS male_pop, SUM(female_pop) AS female_pop
                    FROM daily_population_gu_hourly
                    WHERE stdr_de = ?
                    GROUP BY gu_nm
                    """,
                    (period,),
                ).fetchall()
                for row in rows:
                    gu = _normalize_gu_name(row["gu_nm"])
                    if gu in gu_index:
                        values[gu_index[gu], 0] = float(row["male_pop"] or 0.0)
                        values[gu_index[gu], 1] = float(row["female_pop"] or 0.0)
                if values.sum() > 0:
                    return _normalize_rows(values, fallback=values.sum(axis=0))

    if _table_exists(conn, "kosis_age_district"):
        cols = _columns(conn, "kosis_age_district")
        if {"sex", "population", "gu_nm"}.issubset(cols):
            rows = conn.execute(
                """
                SELECT gu_nm, sex, SUM(population) AS population
                FROM kosis_age_district
                GROUP BY gu_nm, sex
                """
            ).fetchall()
            for row in rows:
                gu = _normalize_gu_name(row["gu_nm"])
                sex = _sex_index(row["sex"])
                if gu in gu_index and sex is not None:
                    values[gu_index[gu], sex] += float(row["population"] or 0.0)
            if values.sum() > 0:
                return _normalize_rows(values, fallback=values.sum(axis=0))

    values[:, :] = 1.0
    return _normalize_rows(values)


def _load_occupation_probs(
    conn: Connection,
    gu_names: tuple[str, ...],
    industry_names: tuple[str, ...],
    year: int | None,
) -> np.ndarray:
    values = np.zeros((len(gu_names), len(industry_names)), dtype=np.float64)
    if industry_names == ("unavailable",) or not _table_exists(conn, "employment_workplace"):
        values[:, 0] = 1.0
        return values

    cols = _columns(conn, "employment_workplace")
    if not {"prd_de", "c1_nm", "c2_nm", "dt"}.issubset(cols):
        values[:, 0] = 1.0
        return values

    period = _period_text(conn, "employment_workplace", "prd_de", year=year, annual=False)
    if period is None:
        values[:, 0] = 1.0
        return values

    gu_index = {name: i for i, name in enumerate(gu_names)}
    industry_index = {name: i for i, name in enumerate(industry_names)}
    rows = conn.execute(
        """
        SELECT c1_nm, c2_nm, SUM(dt) AS dt
        FROM employment_workplace
        WHERE CAST(prd_de AS INT) = CAST(? AS INT)
          AND c2_nm != '계'
        GROUP BY c1_nm, c2_nm
        """,
        (period,),
    ).fetchall()
    for row in rows:
        gu = _normalize_gu_name(row["c1_nm"])
        industry = str(row["c2_nm"] or "")
        if gu in gu_index and industry in industry_index:
            values[gu_index[gu], industry_index[industry]] += float(row["dt"] or 0.0)

    city = values.sum(axis=0)
    return _normalize_rows(values, fallback=city)


def _load_work_probs(conn: Connection, gu_names: tuple[str, ...]) -> np.ndarray:
    values = np.zeros((len(gu_names), len(gu_names)), dtype=np.float64)
    if not _table_exists(conn, "commuter_matrix"):
        np.fill_diagonal(values, 1.0)
        return values

    cols = _columns(conn, "commuter_matrix")
    if not {"origin_gu", "dest_gu"}.issubset(cols):
        np.fill_diagonal(values, 1.0)
        return values

    weight_col = "coupling" if "coupling" in cols else "commuters" if "commuters" in cols else None
    if weight_col is None:
        np.fill_diagonal(values, 1.0)
        return values

    where = ""
    if "collected_at" in cols:
        where = """
        WHERE collected_at = (
            SELECT MAX(collected_at) FROM commuter_matrix
        )
        """

    gu_index = {name: i for i, name in enumerate(gu_names)}
    rows = conn.execute(
        f"""
        SELECT origin_gu, dest_gu, SUM({weight_col}) AS weight
        FROM commuter_matrix
        {where}
        GROUP BY origin_gu, dest_gu
        """
    ).fetchall()
    for row in rows:
        origin = _normalize_gu_name(row["origin_gu"])
        dest = _normalize_gu_name(row["dest_gu"])
        if origin in gu_index and dest in gu_index:
            values[gu_index[origin], gu_index[dest]] += float(row["weight"] or 0.0)

    identity = np.eye(len(gu_names), dtype=np.float64)
    for i in range(values.shape[0]):
        if values[i].sum() <= 0:
            values[i] = identity[i]
    return _normalize_rows(values)


def _load_direct_severity(
    conn: Connection, ref_year: int | None
) -> np.ndarray | None:
    cols = _columns(conn, "hira_inpat_opat")
    if not {"kcd_code", "sex", "age_group", "inpat_opat", "patient_count"}.issubset(cols):
        return None

    values = np.zeros((2, _AGE_BANDS, 2), dtype=np.float64)
    rows = conn.execute(
        """
        SELECT sex, age_group, inpat_opat, SUM(patient_count) AS patient_count
        FROM hira_inpat_opat
        WHERE (kcd_code LIKE 'J10%' OR kcd_code LIKE 'J11%')
          AND ref_year = ?
        GROUP BY sex, age_group, inpat_opat
        """,
        (ref_year,),
    ).fetchall()
    for row in rows:
        sex = _sex_index(row["sex"])
        band = _age_band_from_label(row["age_group"])
        if sex is None or band is None:
            continue
        slot = 1 if row["inpat_opat"] == "입원" else 0
        values[sex, band, slot] += float(row["patient_count"] or 0.0)

    denom = values.sum(axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = np.divide(values[:, :, 1], denom, out=np.zeros((2, _AGE_BANDS)), where=denom > 0)
    if probs.sum() > 0:
        return np.clip(probs, 0.0, 0.95)
    return None


def _load_severity_probs(conn: Connection, year: int | None) -> np.ndarray:
    fallback = np.vstack([_FALLBACK_SEVERITY, _FALLBACK_SEVERITY]).astype(np.float64)
    if not _table_exists(conn, "hira_inpat_opat"):
        return fallback
    cols = _columns(conn, "hira_inpat_opat")
    if not {"kcd_code", "ref_year", "sex", "inpat_opat", "patient_count"}.issubset(cols):
        return fallback

    ref_year = _period_int(conn, "hira_inpat_opat", "ref_year", year)
    if ref_year is None:
        return fallback

    direct = _load_direct_severity(conn, ref_year)
    if direct is not None:
        return direct

    sex_targets = np.zeros(2, dtype=np.float64)
    rows = conn.execute(
        """
        SELECT sex,
               SUM(CASE WHEN inpat_opat = '입원' THEN patient_count ELSE 0 END) AS inpat,
               SUM(patient_count) AS total
        FROM hira_inpat_opat
        WHERE (kcd_code LIKE 'J10%' OR kcd_code LIKE 'J11%')
          AND ref_year = ?
        GROUP BY sex
        """,
        (ref_year,),
    ).fetchall()
    for row in rows:
        sex = _sex_index(row["sex"])
        total = float(row["total"] or 0.0)
        if sex is not None and total > 0:
            sex_targets[sex] = float(row["inpat"] or 0.0) / total

    if sex_targets.sum() <= 0:
        return fallback

    age_weights = np.zeros((2, _AGE_BANDS), dtype=np.float64)
    if _table_exists(conn, "hira_gender_age"):
        gender_cols = _columns(conn, "hira_gender_age")
        if {"kcd_code", "ref_year", "sex", "age_group", "patient_count"}.issubset(
            gender_cols
        ):
            gender_ref_year = _period_int(conn, "hira_gender_age", "ref_year", year)
            if gender_ref_year is not None:
                rows = conn.execute(
                    """
                    SELECT sex, age_group, SUM(patient_count) AS patient_count
                    FROM hira_gender_age
                    WHERE (kcd_code LIKE 'J10%' OR kcd_code LIKE 'J11%')
                      AND ref_year = ?
                    GROUP BY sex, age_group
                    """,
                    (gender_ref_year,),
                ).fetchall()
                for row in rows:
                    sex = _sex_index(row["sex"])
                    band = _age_band_from_label(row["age_group"])
                    if sex is not None and band is not None:
                        age_weights[sex, band] += float(row["patient_count"] or 0.0)

    # Actual schema has inpatient/outpatient by sex only and age distribution separately.
    # Scale the monotone age profile to each sex-specific influenza inpatient fraction.
    probs = np.empty((2, _AGE_BANDS), dtype=np.float64)
    for sex in range(2):
        target = sex_targets[sex] if sex_targets[sex] > 0 else float(sex_targets.mean())
        weights = _normalize_prob(age_weights[sex]) if age_weights[sex].sum() > 0 else None
        profile_mean = (
            float(np.dot(_FALLBACK_SEVERITY, weights))
            if weights is not None
            else float(_FALLBACK_SEVERITY.mean())
        )
        scale = target / profile_mean if profile_mean > 0 else 1.0
        probs[sex] = np.clip(_FALLBACK_SEVERITY * scale, 0.0, 0.95)
    return probs


@lru_cache(maxsize=8)
def _load_population_inputs(year: int | None) -> _PopulationInputs:
    with _connect() as conn:
        gu_names = _load_gu_names(conn)
        industry_names = _load_industry_names(conn)
        home_probs = _load_home_probs(conn, gu_names, year)
        age_probs_by_gu = _load_age_probs(conn, gu_names, year)
        sex_probs_by_gu = _load_sex_probs(conn, gu_names, year)
        occupation_probs_by_gu = _load_occupation_probs(
            conn, gu_names, industry_names, year
        )
        severity_probs_by_sex_age = _load_severity_probs(conn, year)
        work_probs_by_home = _load_work_probs(conn, gu_names)

    return _PopulationInputs(
        gu_names=gu_names,
        industry_names=industry_names,
        home_probs=home_probs,
        age_probs_by_gu=age_probs_by_gu,
        sex_probs_by_gu=sex_probs_by_gu,
        occupation_probs_by_gu=occupation_probs_by_gu,
        severity_probs_by_sex_age=severity_probs_by_sex_age,
        work_probs_by_home=work_probs_by_home,
    )


def _sample_categorical(rng: np.random.Generator, probs: np.ndarray, size: int) -> np.ndarray:
    cdf = np.cumsum(probs, dtype=np.float64)
    cdf[-1] = 1.0
    return np.searchsorted(cdf, rng.random(size), side="right")


def _sample_categorical_by_row(
    rng: np.random.Generator, probs_by_row: np.ndarray, rows: np.ndarray
) -> np.ndarray:
    cdf = np.cumsum(probs_by_row, axis=1, dtype=np.float64)
    cdf[:, -1] = 1.0
    out = np.empty(rows.shape[0], dtype=np.int64)
    for row in np.unique(rows):
        mask = rows == row
        out[mask] = np.searchsorted(cdf[int(row)], rng.random(int(mask.sum())), side="right")
    return out


def generate_population(N: int, *, seed: int, year: int | None = None) -> dict[str, np.ndarray]:
    """Generate a DB-grounded synthetic Seoul population in structure-of-arrays form.

    Args:
        N: Number of agents to generate. Must be at least 1.
        seed: Seed passed to ``numpy.random.default_rng`` for reproducible sampling.
        year: Optional reference year. When provided, annual and dated DB inputs use
            the latest available period not after that year; commuter coupling is static.

    Returns:
        Dictionary of one-dimensional numpy arrays with shape ``(N,)`` and keys
        ``home_gu``, ``age_band``, ``sex``, ``occupation``, ``severity``, and
        ``work_gu``. Gu indices are sorted Seoul gu names ``0..24``. Age bands are
        KDCA-style decade bands ``0..6``. Sex is ``0=male, 1=female``. Severity is
        ``0=low, 1=high``.

    Raises:
        ValueError: If ``N < 1``.
        RuntimeError: If the DB cannot provide the 25 Seoul gu names needed for
            stable indexing.

    Performance:
        O(N) time and O(N) memory for generated arrays; DB probability matrices are
        loaded once per ``year`` value and cached with ``functools.lru_cache``.

    Side effects:
        Opens ``simulation/data/db/epi_real_seoul.db`` in SQLite read-only mode.
        Does not write to the database or filesystem. Caches DB-derived matrices
        in process memory.

    Caller responsibility:
        Treat returned categorical integer codes as indices into ``GU_NAMES`` and
        ``INDUSTRY_NAMES`` from this module.
    """
    if N < 1:
        raise ValueError("N must be >= 1")

    inputs = _load_population_inputs(year)
    rng = np.random.default_rng(seed)

    home_gu = _sample_categorical(rng, inputs.home_probs, N).astype(np.int8)
    age_band = _sample_categorical_by_row(rng, inputs.age_probs_by_gu, home_gu).astype(
        np.int8
    )
    sex = _sample_categorical_by_row(rng, inputs.sex_probs_by_gu, home_gu).astype(np.int8)
    occupation = _sample_categorical_by_row(
        rng, inputs.occupation_probs_by_gu, home_gu
    ).astype(np.int16)
    severity_probs = inputs.severity_probs_by_sex_age[sex, age_band]
    severity = (rng.random(N) < severity_probs).astype(np.int8)
    work_gu = _sample_categorical_by_row(rng, inputs.work_probs_by_home, home_gu).astype(
        np.int8
    )

    return {
        "home_gu": home_gu,
        "age_band": age_band,
        "sex": sex,
        "occupation": occupation,
        "severity": severity,
        "work_gu": work_gu,
    }


_DEFAULT_INPUTS = _load_population_inputs(None)
GU_NAMES = list(_DEFAULT_INPUTS.gu_names)
INDUSTRY_NAMES = list(_DEFAULT_INPUTS.industry_names)
