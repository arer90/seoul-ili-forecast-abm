"""
simulation.collectors.extract_pdf
=================================
Extract Seoul Annual Infectious Disease Surveillance Report (감염병감시연보) PDF
into the ``seoul_annual_report_*`` tables in ``epi_real_seoul.db``.

Two table layouts are supported (the dominant patterns in the 2015-2024 report):

1. **District tables** — "자치구별 연도별 신고현황 2015-2024 (계속)"

 Layout (after pdfplumber flattening)::

 row[0]: ['구\n분', '감염병명', <gu_nm>, None, None, ..., None]
 row[1]: [None, None, '2015', '2016', ..., '2024']
 row[k]: [None, <disease_nm>, , ..., v24] # k >= 2

2. **Monthly tables** — "서울특별시 <disease> 월별 신고현황 2015-2024"

 Layout::

 row[0]: ['구분', <disease_nm>, None, None, ..., None]
 row[1]: [None, '1월', '2월', ..., '12월']
 row[k]: [<year>, m1, m2, ..., m12] # year may appear
 # in row[k][0] or be
 # inferred sequentially

The parser deliberately targets only these two high-signal layouts. Age and
gender tables use a multi-level header with 전국/서울 × 발생수/발생률 columns that
are messy to split; gender breakdowns for Seoul are already available via
``kosis_disease_gender`` from the KOSIS CSV importer.

The extraction is **idempotent**: every run ``DELETE``s rows whose ``source``
column equals the computed tag for this PDF before inserting again. The
default source tag is ``'PDF_2024_감염병감시연보'`` (stable across re-runs,
independent of the actual filename) so the previous production ingestion
(tagged ``'2024_감염병감시연보'``) is preserved untouched.

By default, if the target tables already contain ``>= MIN_EXISTING_ROWS``
rows for the default source tag **OR** for the legacy tag, the extractor
refuses to run and reports the row count. Use ``--force`` to override.

Usage::

 python -m simulation extract-pdf
 python -m simulation extract-pdf --force
 python -m simulation.collectors.extract_pdf --pdf path/to/report.pdf
"""
from __future__ import annotations

import argparse
import logging
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

_PKG_ROOT = Path(__file__).resolve().parent.parent  # simulation/
PDF_DIR = _PKG_ROOT / "data" / "pdf"

DEFAULT_SOURCE_TAG = "PDF_2024_감염병감시연보"
LEGACY_SOURCE_TAG = "2024_감염병감시연보"  # previous production tag
MIN_EXISTING_ROWS = 1000

DISTRICTS_25 = (
    "종로구 중구 용산구 성동구 광진구 동대문구 중랑구 성북구 강북구 도봉구 "
    "노원구 은평구 서대문구 마포구 양천구 강서구 구로구 금천구 영등포구 "
    "동작구 관악구 서초구 강남구 송파구 강동구"
).split()
_DISTRICT_SET = frozenset(DISTRICTS_25)

_MONTH_LABELS = [f"{i}월" for i in range(1, 13)]


# ──────────────────────────────────────────────────────────────────────────
#  PDF discovery
# ──────────────────────────────────────────────────────────────────────────
def find_pdf(pdf_path: Optional[str] = None) -> Path:
    import unicodedata

    if pdf_path:
        p = Path(pdf_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if PDF_DIR.exists():
        # macOS stores filenames in NFD (decomposed) form but Python string
        # literals are NFC (composed). Normalize both sides before `in`
        # match, otherwise "감염병감시연보" ∉ "감염병감시연보" on macOS APFS.
        target_nfc = unicodedata.normalize("NFC", "감염병감시연보")
        for f in PDF_DIR.glob("*.pdf"):
            name_nfc = unicodedata.normalize("NFC", f.name)
            if target_nfc in name_nfc or "annual_report" in name_nfc.lower():
                return f

    # Task C : dropped `_past/data/` fallback — ENGINEERING_PRINCIPLES.md requires
    # that simulation/ never reach outside simulation/data/. If the PDF
    # is missing, caller must copy it into simulation/data/pdf/ explicitly.
    raise FileNotFoundError(
        f"No annual report PDF found. Place it in: {PDF_DIR}/\n"
        f"Expected filename containing '감염병감시연보'."
    )


# ──────────────────────────────────────────────────────────────────────────
#  Value parsing
# ──────────────────────────────────────────────────────────────────────────
def _clean_int(val) -> Optional[int]:
    if val is None:
        return None
    s = str(val).strip()
    if not s or s in ("-", "·", "..", "x", "X"):
        return None
    s = s.replace(",", "").replace(" ", "")
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def _first_year(text: str) -> Optional[int]:
    if text is None:
        return None
    m = re.search(r"20\d{2}", str(text))
    return int(m.group()) if m else None


# ──────────────────────────────────────────────────────────────────────────
#  Table classification + parsing
# ──────────────────────────────────────────────────────────────────────────
def _parse_district_table(table, page_text: str) -> list[dict]:
    """Parse one 'district × year grid' table.

    Returns a list of dicts ready for ``seoul_annual_report_district``.
    """
    if not table or len(table) < 3:
        return []

    # Header sanity: row0 should contain district name somewhere in cols 2+
    row0 = [str(c).strip() if c else "" for c in table[0]]
    row1 = [str(c).strip() if c else "" for c in table[1]]

    # Find the gu_nm from row0 (first matching district label)
    gu_nm = None
    for cell in row0:
        stripped = cell.replace("\n", "").strip()
        if stripped in _DISTRICT_SET:
            gu_nm = stripped
            break
    if gu_nm is None:
        return []

    # Find year columns from row1
    year_cols: list[tuple[int, int]] = []
    for idx, cell in enumerate(row1):
        yr = _first_year(cell)
        if yr is not None:
            year_cols.append((idx, yr))
    if not year_cols:
        return []

    results: list[dict] = []
    for row in table[2:]:
        if not row or len(row) < 3:
            continue
        # disease name is typically in col 1 (col 0 is the rotated '구분')
        disease_cell = row[1] if len(row) > 1 else None
        disease_nm = str(disease_cell).strip() if disease_cell else ""
        if not disease_nm or disease_nm.startswith("("):
            continue
        # Skip if disease cell is actually a number (misaligned row)
        if re.fullmatch(r"[\d,\-\.]+", disease_nm):
            continue
        for col_idx, yr in year_cols:
            if col_idx >= len(row):
                continue
            cases = _clean_int(row[col_idx])
            if cases is None:
                continue
            results.append({
                "gu_nm": gu_nm,
                "disease_nm": disease_nm,
                "year": yr,
                "cases": cases,
            })
    return results


def _parse_monthly_table(table, page_text: str) -> list[dict]:
    """Parse one 'year × month grid' monthly table for a single disease."""
    if not table or len(table) < 3:
        return []

    row0 = [str(c).strip() if c else "" for c in table[0]]
    row1 = [str(c).strip() if c else "" for c in table[1]]

    # Disease name should appear in row0 (usually col 1)
    disease_nm = ""
    for cell in row0:
        s = cell.replace("\n", "").strip()
        if not s or s in ("구분",):
            continue
        # Skip if it looks like a month label
        if s in _MONTH_LABELS:
            continue
        disease_nm = s
        break
    if not disease_nm:
        return []

    # Find month columns in row1
    month_cols: list[tuple[int, int]] = []
    for idx, cell in enumerate(row1):
        if cell in _MONTH_LABELS:
            month_cols.append((idx, _MONTH_LABELS.index(cell) + 1))
    # Need at least 6 month columns for this to be a valid monthly table
    if len(month_cols) < 6:
        return []

    # Data rows: row[0] may be the year label; if absent, infer sequential 2015+
    results: list[dict] = []
    inferred_year = 2015
    for row in table[2:]:
        if not row:
            continue
        year = _first_year(row[0] if len(row) > 0 else None)
        if year is None:
            year = inferred_year
        inferred_year = year + 1
        if not (2000 <= year <= 2099):
            continue
        for col_idx, month in month_cols:
            if col_idx >= len(row):
                continue
            cases = _clean_int(row[col_idx])
            if cases is None:
                continue
            results.append({
                "disease_nm": disease_nm,
                "year": year,
                "month": month,
                "cases": cases,
            })
    return results


# ──────────────────────────────────────────────────────────────────────────
#  Main extraction
# ──────────────────────────────────────────────────────────────────────────
def _ensure_schema(cur: sqlite3.Cursor) -> None:
    """Create target tables if missing (for fresh DBs)."""
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS seoul_annual_report_district (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collected_at TEXT,
            source TEXT,
            gu_nm TEXT,
            disease_nm TEXT,
            grade TEXT,
            year INTEGER,
            cases INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_sard_lookup
            ON seoul_annual_report_district(gu_nm, disease_nm, year);

        CREATE TABLE IF NOT EXISTS seoul_annual_report_monthly (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            collected_at TEXT,
            source TEXT,
            disease_nm TEXT,
            year INTEGER,
            month INTEGER,
            cases INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_sarm_lookup
            ON seoul_annual_report_monthly(disease_nm, year, month);
    """)


def _count_existing(cur: sqlite3.Cursor, table: str, tags: tuple[str, ...]) -> int:
    placeholders = ",".join("?" for _ in tags)
    try:
        cur.execute(
            f"SELECT COUNT(*) FROM {table} WHERE source IN ({placeholders})", tags
        )
        return int(cur.fetchone()[0])
    except sqlite3.OperationalError:
        return 0


def extract_pdf(
    pdf_path: Path,
    db_path: str,
    *,
    source_tag: str = DEFAULT_SOURCE_TAG,
    force: bool = False,
) -> dict:
    """Extract the PDF and load district + monthly tables into the DB.

    Returns a dict with ``district`` and ``monthly`` row counts.  If the DB
    already has data for this PDF (``source_tag`` or ``LEGACY_SOURCE_TAG``)
    and ``force`` is False, the extraction is a no-op and the returned counts
    are zero.
    """
    try:
        import pdfplumber
    except ImportError as e:
        log.error("pdfplumber not installed. Run: uv pip install pdfplumber")
        raise RuntimeError(
            "pdfplumber not installed; install with `uv pip install pdfplumber`"
        ) from e

    from simulation.database import safe_connect  # : quick_check + WAL
    conn = safe_connect(db_path, timeout=60)
    cur = conn.cursor()
    _ensure_schema(cur)

    existing_district = _count_existing(
        cur, "seoul_annual_report_district", (source_tag, LEGACY_SOURCE_TAG)
    )
    existing_monthly = _count_existing(
        cur, "seoul_annual_report_monthly", (source_tag, LEGACY_SOURCE_TAG)
    )
    if not force and (
        existing_district >= MIN_EXISTING_ROWS
        or existing_monthly >= MIN_EXISTING_ROWS
    ):
        log.info(
            "PDF data already present in DB "
            "(district=%d, monthly=%d rows for source in {%r, %r}). "
            "Skipping extraction. Use --force to re-extract.",
            existing_district, existing_monthly, source_tag, LEGACY_SOURCE_TAG,
        )
        conn.close()
        return {"district": 0, "monthly": 0, "skipped": True}

    now = datetime.now().isoformat()

    log.info("Opening PDF: %s", pdf_path.name)
    district_rows: list[dict] = []
    monthly_rows: list[dict] = []

    with pdfplumber.open(str(pdf_path)) as pdf:
        n_pages = len(pdf.pages)
        log.info("Total pages: %d", n_pages)

        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables() or []

            # Heuristic: decide which parser to try based on page text
            is_district_page = "자치구별" in text and "연도별" in text
            is_monthly_page = "월별 신고현황" in text or "월별신고현황" in text

            for table in tables:
                if is_district_page:
                    parsed = _parse_district_table(table, text)
                    for row in parsed:
                        district_rows.append({
                            "collected_at": now,
                            "source": source_tag,
                            "gu_nm": row["gu_nm"],
                            "disease_nm": row["disease_nm"],
                            "grade": "",
                            "year": row["year"],
                            "cases": row["cases"],
                        })
                elif is_monthly_page:
                    parsed = _parse_monthly_table(table, text)
                    for row in parsed:
                        monthly_rows.append({
                            "collected_at": now,
                            "source": source_tag,
                            "disease_nm": row["disease_nm"],
                            "year": row["year"],
                            "month": row["month"],
                            "cases": row["cases"],
                        })

            if (i + 1) % 20 == 0:
                log.info("  progress: page %d/%d (district=%d, monthly=%d)",
                         i + 1, n_pages, len(district_rows), len(monthly_rows))

    log.info("Parsed: district=%d, monthly=%d", len(district_rows), len(monthly_rows))

    cur.execute("BEGIN IMMEDIATE")
    if district_rows:
        cur.execute(
            "DELETE FROM seoul_annual_report_district WHERE source = ?",
            (source_tag,),
        )
        cur.executemany(
            "INSERT INTO seoul_annual_report_district "
            "(collected_at, source, gu_nm, disease_nm, grade, year, cases) "
            "VALUES (:collected_at, :source, :gu_nm, :disease_nm, :grade, :year, :cases)",
            district_rows,
        )
        log.info("Inserted %d district rows (source=%s)", len(district_rows), source_tag)

    if monthly_rows:
        cur.execute(
            "DELETE FROM seoul_annual_report_monthly WHERE source = ?",
            (source_tag,),
        )
        cur.executemany(
            "INSERT INTO seoul_annual_report_monthly "
            "(collected_at, source, disease_nm, year, month, cases) "
            "VALUES (:collected_at, :source, :disease_nm, :year, :month, :cases)",
            monthly_rows,
        )
        log.info("Inserted %d monthly rows (source=%s)", len(monthly_rows), source_tag)
    cur.execute("COMMIT")
    conn.close()

    if not district_rows and not monthly_rows:
        log.warning(
            "PDF parser produced 0 rows.  The PDF layout may have changed; "
            "inspect simulation/collectors/extract_pdf.py::_parse_*_table."
        )

    return {
        "district": len(district_rows),
        "monthly": len(monthly_rows),
        "skipped": False,
    }


# Legacy name retained for backwards compatibility with any caller that used it.
extract_district_table = extract_pdf


# ──────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Extract annual report PDF into DB")
    parser.add_argument("--pdf", type=str, default=None,
                        help="Path to PDF file (auto-detect if omitted)")
    parser.add_argument("--db", type=str, default=None,
                        help="Path to SQLite DB (default: simulation/data/db/epi_real_seoul.db)")
    parser.add_argument("--force", action="store_true",
                        help="Re-extract even if data already exists in DB")
    parser.add_argument("--source-tag", type=str, default=DEFAULT_SOURCE_TAG,
                        help=f"Source tag for DB rows (default: {DEFAULT_SOURCE_TAG})")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    pdf_path = find_pdf(args.pdf)
    log.info("PDF: %s", pdf_path)

    if args.db:
        db_path = args.db
    else:
        from simulation.database.config import DB_PATH
        db_path = str(DB_PATH)
    log.info("DB: %s", db_path)

    result = extract_pdf(
        pdf_path, db_path, source_tag=args.source_tag, force=args.force,
    )
    if result.get("skipped"):
        log.info("Done (skipped — data already present).")
    else:
        log.info(
            "Done. district=%d, monthly=%d rows extracted.",
            result["district"], result["monthly"],
        )


if __name__ == "__main__":
    main()
