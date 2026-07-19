"""
simulation.collectors.import_external
======================================
Import external data files into ``epi_real_seoul.db``.

Handled sources (all under ``simulation/data/external/`` or ``simulation/data/pdf/``):

- ``VIW_FNT.csv``                   -> ``who_flunet``
- ``VIW_FLU_METADATA.csv``          -> ``who_flunet_metadata``
- ``commuter_matrix.json``          -> ``commuter_matrix`` (coupling + night pop)
- ``kosis_infectious_disease_sources.txt``                     -> ``kosis_source_registry`` (URL references)
- ``201_DT_201004_O110011_02_*.csv``-> ``kosis_disease_gender``  (Seoul gu x disease x gender)
- ``201_DT_201004_O110011_02_*.txt``-> ``kosis_source_registry`` (metadata header)

All importers are idempotent: each call ``DELETE``s the target table (by
source tag for kosis_source_registry) before re-inserting, so running
``--import-all`` multiple times is safe.

Usage::

    python -m simulation import-external --scan
    python -m simulation import-external --import-all
    python -m simulation.collectors.import_external --flunet
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

log = logging.getLogger(__name__)

_PKG_ROOT = Path(__file__).resolve().parent.parent  # simulation/
EXTERNAL_DIR = _PKG_ROOT / "data" / "external"
PDF_DIR = _PKG_ROOT / "data" / "pdf"


# ──────────────────────────────────────────────────────────────────────────
#  shared helpers
# ──────────────────────────────────────────────────────────────────────────
def _connect(db_path: str) -> sqlite3.Connection:
    """Open DB with safe defaults — delegates to simulation.database.safe_connect
 (: quick_check + WAL + tuning 자동).
 """
    from simulation.database import safe_connect
    conn = safe_connect(db_path, timeout=60, isolation_level=None)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 60000")
    return conn


def _safe_int(v, default: int = 0) -> int:
    if v is None:
        return default
    s = str(v).strip()
    if not s or s in ("-", "..", "X", "x"):
        return default
    try:
        return int(float(s.replace(",", "")))
    except (ValueError, TypeError):
        return default


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s in ("-", "..", "X", "x"):
        return None
    try:
        return float(s.replace(",", ""))
    except (ValueError, TypeError):
        return None


# ──────────────────────────────────────────────────────────────────────────
#  WHO FluNet — VIW_FNT.csv
#  ------------------------------------------------------------------
#  xMart source: https://xmart-api-public.who.int/FLUMART/VIW_FNT
#  Local cache : simulation/data/external/VIW_FNT.csv (53 cols)
#
#  fix (2026-04-17) — field-name audit vs WHO FluMart 4.0:
#     OLD name (never existed in VIW_FNT)   → CORRECT VIW_FNT field
#     -----------------------------------------------------------------
#     INF_A_H1N1                            → AH1N12009  (pdm09 subtype)
#     INF_A_H3N2                            → AH3
#     INF_A_NOTSUBTYPED                     → ANOTSUBTYPED
#     INF_B_YAMAGATA                        → BYAM
#     INF_B_VICTORIA                        → BVIC_2DEL + BVIC_3DEL
#                                              + BVIC_DELUNK + BVIC_NODEL
#     ILI_RATE (category 1..6 misnomer)     → ILI_ACTIVITY  (INTEGER enum)
#     SARI_RATE (does not exist in VIW_FNT) → column dropped (lives in
#                                              VIW_FID = WHO FluID, a
#                                              separate dataset)
#
#  Result before fix: 183,081 rows / 5 subtype columns all zero.
#  Result after  fix: subtype counts populated; ili_activity is an INT
#                     enum (1..6) instead of a REAL mislabelled "rate".
# ──────────────────────────────────────────────────────────────────────────
def import_flunet(db_path: str, csv_path: Optional[Path] = None) -> int:
    """Import WHO FluNet global influenza surveillance data (schema)."""
    if csv_path is None:
        csv_path = EXTERNAL_DIR / "VIW_FNT.csv"
    if not csv_path.exists():
        log.warning("VIW_FNT.csv not found at %s", csv_path)
        return 0

    conn = _connect(db_path)
    cur = conn.cursor()

    # : rebuild who_flunet schema from scratch so legacy columns
    # (ili_rate misnomer, sari_rate ghost) are removed cleanly.
    cur.execute("DROP TABLE IF EXISTS who_flunet")
    cur.execute("""
        CREATE TABLE who_flunet (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collected_at TEXT,
            -- geography
            whoregion TEXT,               -- AFR/AMR/EMR/EUR/SEAR/WPR
            hemisphere TEXT,              -- NH/SH
            itz TEXT,                     -- Influenza Transmission Zone
            country TEXT,
            iso3 TEXT,
            iso2 TEXT,
            -- ISO8601 week
            year INTEGER,
            week_no INTEGER,
            sdate TEXT,                   -- ISO_WEEKSTARTDATE
            edate TEXT,                   -- computed (sdate + 6d) if absent
            origin_source TEXT,           -- SENTINEL/NONSENTINEL/NOTDEFINED
            -- specimen volume
            spec_received INTEGER DEFAULT 0,
            spec_processed INTEGER DEFAULT 0,
            -- influenza A subtypes (fixed field names)
            inf_a_h1n1pdm09 INTEGER DEFAULT 0,  -- AH1N12009
            inf_a_h1 INTEGER DEFAULT 0,         -- AH1 (non-pdm)
            inf_a_h3 INTEGER DEFAULT 0,         -- AH3
            inf_a_h5 INTEGER DEFAULT 0,         -- AH5 (avian)
            inf_a_h7n9 INTEGER DEFAULT 0,       -- AH7N9
            inf_a_other INTEGER DEFAULT 0,      -- AOTHER_SUBTYPE
            inf_a_notsubtyped INTEGER DEFAULT 0, -- ANOTSUBTYPED
            inf_a_notsubtypable INTEGER DEFAULT 0, -- ANOTSUBTYPABLE
            inf_a INTEGER DEFAULT 0,            -- INF_A (all A subtypes)
            -- influenza B lineages
            inf_b_victoria INTEGER DEFAULT 0,   -- BVIC_2DEL+3DEL+DELUNK+NODEL
            inf_b_yamagata INTEGER DEFAULT 0,   -- BYAM
            inf_b_notdetermined INTEGER DEFAULT 0, -- BNOTDETERMINED
            inf_b INTEGER DEFAULT 0,            -- INF_B (all B lineages)
            -- totals
            inf_total INTEGER DEFAULT 0,        -- INF_ALL
            inf_negative INTEGER DEFAULT 0,     -- INF_NEGATIVE
            -- ILI activity (categorical 1..6, NOT a rate)
            ili_activity INTEGER,               -- 1=No report .. 6=Widespread
            -- respiratory co-detections (same specimens)
            adeno INTEGER DEFAULT 0,
            rsv INTEGER DEFAULT 0,
            rsv_processed INTEGER DEFAULT 0,
            metapneumo INTEGER DEFAULT 0,
            parainfluenza INTEGER DEFAULT 0,
            rhino INTEGER DEFAULT 0,
            human_corona INTEGER DEFAULT 0,     -- non-SARS-CoV-2
            other_respvirus INTEGER DEFAULT 0,
            UNIQUE(iso3, year, week_no, origin_source)
        )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_who_flunet_country_year "
        "ON who_flunet(iso3, year, week_no)"
    )

    # BVIC_* sum — helper so the INSERT stays readable.
    def _bvic_sum(r: dict) -> int:
        return (
            _safe_int(r.get("BVIC_2DEL"))
            + _safe_int(r.get("BVIC_3DEL"))
            + _safe_int(r.get("BVIC_DELUNK"))
            + _safe_int(r.get("BVIC_NODEL"))
        )

    def _ili_activity(v) -> Optional[int]:
        """ILI_ACTIVITY is a 1..6 enum (No report .. Widespread).
        Return None when empty/non-numeric so downstream models can
        treat it as missing instead of 0."""
        if v is None:
            return None
        s = str(v).strip()
        if not s or s in ("-", "..", "X", "x"):
            return None
        try:
            val = int(float(s))
            return val if 1 <= val <= 6 else None
        except (ValueError, TypeError):
            return None

    now = datetime.now().isoformat()
    rows: list[tuple] = []

    # WHO xMart VIW_FNT.csv occasionally contains embedded NUL bytes
    # inside string fields (observed in LAB_RESULT_COMMENT). csv.reader
    # raises _csv.Error: line contains NUL. Strip them before parsing.
    def _nul_free_lines(path: Path):
        with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
            for ln in fh:
                yield ln.replace("\x00", "")

    reader = csv.DictReader(_nul_free_lines(csv_path))
    for row in reader:
        ah1n1pdm09 = _safe_int(row.get("AH1N12009"))
        ah1 = _safe_int(row.get("AH1"))
        ah3 = _safe_int(row.get("AH3"))
        ah5 = _safe_int(row.get("AH5"))
        ah7n9 = _safe_int(row.get("AH7N9"))
        a_other = _safe_int(row.get("AOTHER_SUBTYPE"))
        a_notsub = _safe_int(row.get("ANOTSUBTYPED"))
        a_nonsubtypable = _safe_int(row.get("ANOTSUBTYPABLE"))

        bvic = _bvic_sum(row)
        byam = _safe_int(row.get("BYAM"))
        bnotd = _safe_int(row.get("BNOTDETERMINED"))

        rows.append((
            now,
            # geography
            (row.get("WHOREGION") or "").strip(),
            (row.get("HEMISPHERE") or "").strip(),
            (row.get("ITZ") or "").strip(),
            (row.get("COUNTRY_AREA_TERRITORY") or "").strip(),
            (row.get("COUNTRY_CODE") or "").strip(),
            (row.get("ISO2") or "").strip(),
            # week
            _safe_int(row.get("ISO_YEAR")),
            _safe_int(row.get("ISO_WEEK")),
            (row.get("ISO_WEEKSTARTDATE") or "").strip(),
            "",  # edate: WHO xMart doesn't ship EDATE; leave empty
            (row.get("ORIGIN_SOURCE") or "").strip(),
            # specimens
            _safe_int(row.get("SPEC_RECEIVED_NB")),
            _safe_int(row.get("SPEC_PROCESSED_NB")),
            # A subtypes
            ah1n1pdm09, ah1, ah3, ah5, ah7n9, a_other,
            a_notsub, a_nonsubtypable,
            _safe_int(row.get("INF_A")),
            # B lineages
            bvic, byam, bnotd,
            _safe_int(row.get("INF_B")),
            # totals
            _safe_int(row.get("INF_ALL")),
            _safe_int(row.get("INF_NEGATIVE")),
            # ILI activity (enum)
            _ili_activity(row.get("ILI_ACTIVITY")),
            # co-detections
            _safe_int(row.get("ADENO")),
            _safe_int(row.get("RSV")),
            _safe_int(row.get("RSV_PROCESSED")),
            _safe_int(row.get("METAPNEUMO")),
            _safe_int(row.get("PARAINFLUENZA")),
            _safe_int(row.get("RHINO")),
            _safe_int(row.get("HUMAN_CORONA")),
            _safe_int(row.get("OTHERRESPVIRUS")),
        ))

    if rows:
        cur.execute("BEGIN IMMEDIATE")
        cur.executemany(
            "INSERT OR REPLACE INTO who_flunet ("
            "collected_at, whoregion, hemisphere, itz, country, iso3, iso2, "
            "year, week_no, sdate, edate, origin_source, "
            "spec_received, spec_processed, "
            "inf_a_h1n1pdm09, inf_a_h1, inf_a_h3, inf_a_h5, inf_a_h7n9, "
            "inf_a_other, inf_a_notsubtyped, inf_a_notsubtypable, inf_a, "
            "inf_b_victoria, inf_b_yamagata, inf_b_notdetermined, inf_b, "
            "inf_total, inf_negative, ili_activity, "
            "adeno, rsv, rsv_processed, metapneumo, parainfluenza, rhino, "
            "human_corona, other_respvirus) "
            "VALUES (" + ",".join(["?"] * 38) + ")",
            rows,
        )
        cur.execute("COMMIT")
        log.info("Imported %d FluNet rows (schema)", len(rows))

    conn.close()
    return len(rows)


# ──────────────────────────────────────────────────────────────────────────
#  WHO FluNet metadata — VIW_FLU_METADATA.csv
# ──────────────────────────────────────────────────────────────────────────
def import_flunet_metadata(db_path: str, csv_path: Optional[Path] = None) -> int:
    """Import WHO FluNet metadata (reporting lab info)."""
    if csv_path is None:
        csv_path = EXTERNAL_DIR / "VIW_FLU_METADATA.csv"
    if not csv_path.exists():
        log.warning("VIW_FLU_METADATA.csv not found at %s", csv_path)
        return 0

    conn = _connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS who_flunet_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collected_at TEXT,
            country TEXT,
            iso3 TEXT,
            data_type TEXT,
            reporting_labs TEXT,
            description TEXT
        )
    """)

    now = datetime.now().isoformat()
    rows: list[tuple] = []

    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((
                now,
                (row.get("COUNTRY_AREA_TERRITORY") or "").strip(),
                (row.get("ISO3") or row.get("Country_code") or "").strip(),
                (row.get("HEMISPHERE") or "").strip(),
                (row.get("REPORTING_LABS") or "").strip(),
                (row.get("DESCRIPTION") or row.get("OTHERSOURCES") or "").strip(),
            ))

    if rows:
        cur.execute("BEGIN IMMEDIATE")
        cur.execute("DELETE FROM who_flunet_metadata")
        cur.executemany(
            "INSERT INTO who_flunet_metadata "
            "(collected_at, country, iso3, data_type, reporting_labs, description) "
            "VALUES (?,?,?,?,?,?)",
            rows,
        )
        cur.execute("COMMIT")
        log.info("Imported %d FluNet metadata rows", len(rows))

    conn.close()
    return len(rows)


# ──────────────────────────────────────────────────────────────────────────
#  Commuter matrix — commuter_matrix.json (districts + coupling_matrix)
# ──────────────────────────────────────────────────────────────────────────
def import_commuter_matrix(db_path: str, json_path: Optional[Path] = None) -> int:
    """Import Seoul commuter coupling matrix (25x25) for metapopulation model.

    Expected schema::

        {
            "districts":       ["강남구", ..., "중랑구"],      # 25 items
            "coupling_matrix": [[float, ...], ...],            # 25 x 25
            "night_population": {"강남구": float, ...}          # optional
        }
    """
    if json_path is None:
        json_path = EXTERNAL_DIR / "commuter_matrix.json"
    if not json_path.exists():
        log.warning("commuter_matrix.json not found at %s", json_path)
        return 0

    with open(json_path, "r", encoding="utf-8") as f:
        matrix = json.load(f)

    conn = _connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS commuter_matrix (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collected_at TEXT,
            origin_gu TEXT,
            dest_gu TEXT,
            coupling REAL DEFAULT 0,
            night_population REAL,
            source TEXT DEFAULT 'KOSIS'
        )
    """)
    # Backwards-compat: add columns if old schema (commuters INTEGER) is present.
    cur.execute("PRAGMA table_info(commuter_matrix)")
    existing_cols = {r[1] for r in cur.fetchall()}
    for col, ddl in [
        ("coupling", "coupling REAL DEFAULT 0"),
        ("night_population", "night_population REAL"),
    ]:
        if col not in existing_cols:
            try:
                cur.execute(f"ALTER TABLE commuter_matrix ADD COLUMN {ddl}")
            except sqlite3.OperationalError:
                pass
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_commuter_matrix_origin "
        "ON commuter_matrix(origin_gu, dest_gu)"
    )

    now = datetime.now().isoformat()
    rows: list[tuple] = []

    if isinstance(matrix, dict) and "districts" in matrix and "coupling_matrix" in matrix:
        districts = matrix["districts"]
        cm = matrix["coupling_matrix"]
        night = matrix.get("night_population", {}) or {}
        if len(cm) != len(districts) or any(len(r) != len(districts) for r in cm):
            log.error("coupling_matrix shape mismatch: %dx%d vs %d districts",
                      len(cm), len(cm[0]) if cm else 0, len(districts))
            conn.close()
            return 0
        for i, origin in enumerate(districts):
            for j, dest in enumerate(districts):
                coupling = float(cm[i][j])
                np_val = _safe_float(night.get(origin))
                rows.append((now, origin, dest, coupling, np_val, "KOSIS"))
    elif isinstance(matrix, dict):
        # Legacy fallback: {origin: {dest: value}}
        for origin, dests in matrix.items():
            if isinstance(dests, dict):
                for dest, count in dests.items():
                    val = _safe_float(count)
                    if val is not None:
                        rows.append((now, origin, dest, val, None, "KOSIS"))

    if rows:
        cur.execute("BEGIN IMMEDIATE")
        cur.execute("DELETE FROM commuter_matrix")
        cur.executemany(
            "INSERT INTO commuter_matrix "
            "(collected_at, origin_gu, dest_gu, coupling, night_population, source) "
            "VALUES (?,?,?,?,?,?)",
            rows,
        )
        cur.execute("COMMIT")
        log.info("Imported %d commuter matrix entries", len(rows))

    conn.close()
    return len(rows)


# ──────────────────────────────────────────────────────────────────────────
#  KOSIS Seoul gu x disease x gender — 201_DT_*.csv
# ──────────────────────────────────────────────────────────────────────────
_KOSIS_PATTERNS = ("201_DT_201004_O110011_02",)


def _find_kosis_gender_csv() -> Optional[Path]:
    if not EXTERNAL_DIR.exists():
        return None
    for p in EXTERNAL_DIR.iterdir():
        if p.suffix.lower() == ".csv" and any(k in p.name for k in _KOSIS_PATTERNS):
            return p
    return None


def _find_kosis_gender_txt() -> Optional[Path]:
    if not EXTERNAL_DIR.exists():
        return None
    for p in EXTERNAL_DIR.iterdir():
        if p.suffix.lower() == ".txt" and any(k in p.name for k in _KOSIS_PATTERNS):
            return p
    return None


def import_kosis_disease_gender(db_path: str, csv_path: Optional[Path] = None) -> int:
    """Import KOSIS Seoul gu x disease x gender x year (발생/사망) table.

    Source CSV header::

        [A]자치구별, 자치구별, [B]감염병별, 감염병별,
        [C]성별, 성별, [Item]항목, 항목, 단위,
        2020 년, 2021 년, 2022 년, 2023 년, 2024 년

    ``성별`` column contains one of: ``발생`` (total cases), ``사망`` (total deaths),
    ``남`` (male), ``여`` (female). We unpivot years into long form and split
    into ``metric`` (``발생`` or ``사망``) and ``gender`` (``계``/``남``/``여``).

    Rows where all year values are empty or ``-`` are skipped.
    """
    if csv_path is None:
        csv_path = _find_kosis_gender_csv()
    if csv_path is None or not csv_path.exists():
        log.warning("KOSIS 201_DT_*.csv not found in %s", EXTERNAL_DIR)
        return 0

    conn = _connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS kosis_disease_gender (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collected_at TEXT,
            source TEXT DEFAULT 'KOSIS_201_DT_201004_O110011_02',
            gu_code TEXT,
            gu_nm TEXT,
            disease_code TEXT,
            disease_nm TEXT,
            metric TEXT,
            gender TEXT,
            year INTEGER,
            cases INTEGER,
            unit TEXT DEFAULT '건'
        )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_kdg_lookup "
        "ON kosis_disease_gender(gu_nm, disease_nm, year, metric, gender)"
    )

    now = datetime.now().isoformat()
    rows: list[tuple] = []

    # item[5] is the sub-group label.  Grouping works like this:
    #   발생 / 사망   -> metric row (gender='계')
    #   남 / 여       -> inherit last metric, gender from label
    # Rows are stored in sequence so we can carry state forward.
    current_metric = "발생"  # sensible default
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            conn.close()
            return 0
        # Header columns 9..13 are 2020..2024
        year_cols = header[9:14]
        years = [int(re.search(r"\d{4}", y).group()) for y in year_cols if re.search(r"\d{4}", y)]
        unit_col = header[8] if len(header) > 8 else "건"

        for r in reader:
            if len(r) < 14:
                continue
            gu_code = (r[0] or "").strip()
            gu_nm = (r[1] or "").strip()
            dis_code = (r[2] or "").strip()
            dis_nm = (r[3] or "").strip()
            label = (r[5] or "").strip()
            if not label:
                continue
            if label in ("발생", "사망"):
                current_metric = label
                gender = "계"
            elif label in ("남", "여"):
                gender = label
            else:
                # unknown label, skip
                continue

            # Skip rows where every year value is empty/-
            vals = [_safe_int(r[9 + i], default=-1) for i in range(len(years))]
            if all(v == -1 for v in vals):
                continue

            for yr, raw_v, parsed in zip(years, r[9:9 + len(years)], vals):
                s = (raw_v or "").strip()
                if not s or s == "-":
                    continue  # preserve NULL semantics for missing
                rows.append((
                    now,
                    "KOSIS_201_DT_201004_O110011_02",
                    gu_code, gu_nm, dis_code, dis_nm,
                    current_metric, gender, yr, parsed, unit_col,
                ))

    if rows:
        cur.execute("BEGIN IMMEDIATE")
        cur.execute("DELETE FROM kosis_disease_gender")
        cur.executemany(
            "INSERT INTO kosis_disease_gender "
            "(collected_at, source, gu_code, gu_nm, disease_code, disease_nm, "
            "metric, gender, year, cases, unit) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        cur.execute("COMMIT")
        log.info("Imported %d KOSIS disease gender rows", len(rows))

    conn.close()
    return len(rows)


# ──────────────────────────────────────────────────────────────────────────
#  KOSIS source registry — kosis_infectious_disease_sources.txt + 201_DT_*.txt metadata
# ──────────────────────────────────────────────────────────────────────────
def import_kosis_source_registry(db_path: str) -> int:
    """Store KOSIS source-URL references and table metadata as rows.

    This turns free-form .txt reference files into a queryable table rather
    than silently ignoring them.  Each non-URL line becomes a ``note``; each
    URL line becomes a ``url`` entry with its preceding label as ``title``.
    """
    conn = _connect(db_path)
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS kosis_source_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collected_at TEXT,
            source_file TEXT,
            kind TEXT,          -- 'url' | 'note' | 'meta'
            title TEXT,
            url TEXT,
            note TEXT
        )
    """)
    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_kosis_registry_kind "
        "ON kosis_source_registry(kind, source_file)"
    )

    now = datetime.now().isoformat()
    rows: list[tuple] = []

    # ── kosis_infectious_disease_sources.txt: alternating title + URL lines ──
    txt1 = EXTERNAL_DIR / "kosis_infectious_disease_sources.txt"
    if txt1.exists():
        content = txt1.read_text(encoding="utf-8").splitlines()
        pending_title: Optional[str] = None
        for line in content:
            s = line.strip()
            if not s:
                continue
            if s.startswith("http://") or s.startswith("https://"):
                rows.append((now, txt1.name, "url", pending_title or "", s, ""))
                pending_title = None
            else:
                pending_title = s

    # ── 201_DT_*.txt: statistics table metadata ──
    txt2 = _find_kosis_gender_txt()
    if txt2 is not None and txt2.exists():
        content = txt2.read_text(encoding="utf-8")
        # Extract key fields via regex on the ○ bullets
        def _extract(key: str) -> str:
            m = re.search(rf"○\s*{re.escape(key)}\s*:\s*(.+)", content)
            return m.group(1).strip() if m else ""
        title = _extract("통계표명")
        tbl_id = _extract("통계표ID")
        source = _extract("출처")
        url_m = re.search(r"https://stat\.eseoul\.go\.kr/\S+", content)
        url = url_m.group(0) if url_m else ""
        note = content.strip()
        rows.append((now, txt2.name, "meta", f"{tbl_id} — {title}", url, note))
        rows.append((now, txt2.name, "note", title, "", source))

    if rows:
        cur.execute("BEGIN IMMEDIATE")
        cur.execute("DELETE FROM kosis_source_registry")
        cur.executemany(
            "INSERT INTO kosis_source_registry "
            "(collected_at, source_file, kind, title, url, note) "
            "VALUES (?,?,?,?,?,?)",
            rows,
        )
        cur.execute("COMMIT")
        log.info("Imported %d KOSIS source registry entries", len(rows))

    conn.close()
    return len(rows)


# ──────────────────────────────────────────────────────────────────────────
#  Scan + import-all
# ──────────────────────────────────────────────────────────────────────────
def scan_available(db_path: str) -> None:
    print("\n=== External Data Scan ===")
    files = {
        "VIW_FNT.csv": ("WHO FluNet global flu surveillance (~31MB)", "who_flunet"),
        "VIW_FLU_METADATA.csv": ("WHO FluNet metadata", "who_flunet_metadata"),
        "commuter_matrix.json": ("Seoul commuter coupling matrix (25x25)", "commuter_matrix"),
        "kosis_infectious_disease_sources.txt": ("KOSIS open-API URL references", "kosis_source_registry"),
    }
    for fname, (desc, _table) in files.items():
        path = EXTERNAL_DIR / fname
        icon = "OK" if path.exists() else "MISSING"
        print(f"  [{icon}] {fname}: {desc}")

    kosis_csv = _find_kosis_gender_csv()
    if kosis_csv is not None:
        print(f"  [OK] {kosis_csv.name}: KOSIS Seoul gu x disease x gender (2020-2024)")
    else:
        print(f"  [MISSING] 201_DT_201004_O110011_02_*.csv")

    pdf_files = list(PDF_DIR.glob("*.pdf")) if PDF_DIR.exists() else []
    for pf in pdf_files:
        print(f"  [OK] pdf/{pf.name}: Annual report PDF")
    if not pdf_files:
        print(f"  [MISSING] pdf/*.pdf: No annual report PDFs")

    conn = _connect(db_path)
    cur = conn.cursor()
    print("\n=== DB Import Status ===")
    target_tables = [
        "who_flunet", "who_flunet_metadata", "commuter_matrix",
        "kosis_disease_gender", "kosis_source_registry",
        "seoul_annual_report_district", "disease_master",
    ]
    for table in target_tables:
        try:
            cur.execute(f"SELECT COUNT(*) FROM [{table}]")
            cnt = cur.fetchone()[0]
            icon = "OK" if cnt > 0 else "EMPTY"
            print(f"  [{icon}] {table}: {cnt:,} rows")
        except sqlite3.OperationalError:
            print(f"  [MISSING] {table}: table not created yet")
    conn.close()


def import_all(db_path: str) -> int:
    total = 0
    total += import_flunet(db_path)
    total += import_flunet_metadata(db_path)
    total += import_commuter_matrix(db_path)
    total += import_kosis_disease_gender(db_path)
    total += import_kosis_source_registry(db_path)
    log.info("Total imported: %d rows", total)
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Import external data files")
    parser.add_argument("--scan", action="store_true",
                        help="Scan available files and DB status")
    parser.add_argument("--import-all", action="store_true",
                        help="Import all available external data")
    parser.add_argument("--flunet", action="store_true", help="Import WHO FluNet only")
    parser.add_argument("--commuter", action="store_true", help="Import commuter matrix only")
    parser.add_argument("--kosis-gender", action="store_true",
                        help="Import KOSIS gu x disease x gender CSV only")
    parser.add_argument("--kosis-registry", action="store_true",
                        help="Import KOSIS source URL registry only")
    parser.add_argument("--db", type=str, default=None, help="DB path override")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.db:
        db_path = args.db
    else:
        from simulation.database.config import DB_PATH
        db_path = str(DB_PATH)

    if args.scan:
        scan_available(db_path)
    elif args.import_all:
        import_all(db_path)
    elif args.flunet:
        import_flunet(db_path)
    elif args.commuter:
        import_commuter_matrix(db_path)
    elif args.kosis_gender:
        import_kosis_disease_gender(db_path)
    elif args.kosis_registry:
        import_kosis_source_registry(db_path)
    else:
        scan_available(db_path)
        print("\nUse --import-all to import everything, or --scan to check status.")


if __name__ == "__main__":
    main()
