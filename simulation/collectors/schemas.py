"""
simulation.collectors.schemas — Pydantic input validation (Task D).

Purpose
-------
ENGINEERING_PRINCIPLES.md note:
 "개선 여지가 있다면 ORM 대신 **pydantic 기반 입력 검증** 을 collector
 단에 얇게 덧대는 게 다음 단계다."

This module lands the thin validation layer the note describes. Schemas
mirror the DB column shape (not the raw XML/JSON field names from the
upstream API) so callers build a plain Python dict in the collector's
shape-coercion step, pass it through `validate_batch(...)`, and only
then feed the validated dict into `executemany` / `insert_rows`.

Non-fatal by design
-------------------
`validate_batch` never raises. It partitions the input into
`(validated, invalid)` so the collector can log + skip bad rows but
keep making progress on the rest. This matches the project's broader
"best-effort collection" philosophy — a single malformed XML item
should not abort a 5-year year-by-year sweep.

Integration plan
----------------
1. Collector builds `dict` row in DB-column shape (not the raw source
 shape).
2. Collector calls `validate_batch(<RowModel>, rows, label="<endpoint>")`.
3. Collector iterates `validated` to build the `executemany` tuples.
4. `invalid` is discarded after being logged by `validate_batch`.

As of this module is complete and covered by unit tests, but
wiring into the live HIRA / NEIS collectors is intentionally left as
a follow-up — the existing tuple-based `executemany` paths would need
to be rewritten dict-first. Landing the schema surface first lets us
iterate on field constraints without blocking the collector rewrite.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────
# Base model — shared config for all collector rows.
# ──────────────────────────────────────────────────────────────────────────
class _CollectorRow(BaseModel):
    """Shared base: ignore extra keys, strip whitespace from strings."""
    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
        # pydantic : allow int-coercion from strings like "123" but
        # reject genuinely non-numeric garbage.
        str_to_lower=False,
    )
    # `collected_at` is stamped by the collector's `self.now_iso()` call;
    # kept optional here because some endpoints attach it after the fact.
    collected_at: Optional[str] = None


# ──────────────────────────────────────────────────────────────────────────
# HIRA (Group H) — 4 endpoints, mirror the DB columns in group_h_hira.py.
# ──────────────────────────────────────────────────────────────────────────
class HiraBaseRow(_CollectorRow):
    """Fields shared across all 4 HIRA endpoints (inpat_opat, gender_age,
    facility, region). `kcd_code` is the KCD-7 disease code; `ref_year`
    is the 4-digit calendar year the stats cover."""
    kcd_code: str = Field(min_length=1, max_length=16)
    kcd_name: Optional[str] = Field(default=None, max_length=200)
    ref_year: int = Field(ge=2000, le=2100)


class HiraInpatOpatRow(HiraBaseRow):
    sex: Optional[str] = Field(default=None, max_length=8)
    inpat_opat: Optional[str] = Field(default=None, max_length=16)
    patient_count: Optional[int] = Field(default=None, ge=0)
    spec_count: Optional[int] = Field(default=None, ge=0)
    visit_days: Optional[int] = Field(default=None, ge=0)
    # 부담금 / 청구금은 drug recalls 등으로 음수가 나올 수도 있어 ge 제약 없음.
    insup_brdn_amt: Optional[int] = None
    rpe_tamt_amt: Optional[int] = None


class HiraGenderAgeRow(HiraBaseRow):
    sex: Optional[str] = Field(default=None, max_length=8)
    age_group: Optional[str] = Field(default=None, max_length=16)
    patient_count: Optional[int] = Field(default=None, ge=0)
    spec_count: Optional[int] = Field(default=None, ge=0)
    visit_days: Optional[int] = Field(default=None, ge=0)
    insup_brdn_amt: Optional[int] = None
    rpe_tamt_amt: Optional[int] = None


class HiraFacilityRow(HiraBaseRow):
    facility_type: Optional[str] = Field(default=None, max_length=40)
    patient_count: Optional[int] = Field(default=None, ge=0)
    spec_count: Optional[int] = Field(default=None, ge=0)
    visit_days: Optional[int] = Field(default=None, ge=0)
    insup_brdn_amt: Optional[int] = None
    rpe_tamt_amt: Optional[int] = None


class HiraRegionRow(HiraBaseRow):
    region: Optional[str] = Field(default=None, max_length=32)
    patient_count: Optional[int] = Field(default=None, ge=0)
    spec_count: Optional[int] = Field(default=None, ge=0)
    visit_days: Optional[int] = Field(default=None, ge=0)
    insup_brdn_amt: Optional[int] = None
    rpe_tamt_amt: Optional[int] = None


# ──────────────────────────────────────────────────────────────────────────
# NEIS (Group R) — 2 endpoints, mirror school_info_seoul / school_closure_seoul.
# ──────────────────────────────────────────────────────────────────────────
class SchoolInfoRow(_CollectorRow):
    school_code: str = Field(min_length=1, max_length=20)
    school_name: Optional[str] = Field(default=None, max_length=200)
    school_type: Optional[str] = Field(default=None, max_length=40)
    gu_name: Optional[str] = Field(default=None, max_length=40)
    address: Optional[str] = Field(default=None, max_length=200)
    # FOND_YMD is YYYYMMDD but occasionally blank for historic schools.
    found_date: Optional[str] = Field(default=None, max_length=10)

    @field_validator("found_date")
    @classmethod
    def _check_found_date(cls, v: Optional[str]) -> Optional[str]:
        if v is None or v == "":
            return None
        if not v.isdigit() or len(v) != 8:
            raise ValueError(f"found_date must be YYYYMMDD or blank, got {v!r}")
        return v


class SchoolScheduleRow(_CollectorRow):
    # AA_YMD is YYYYMMDD, always required.
    date: str = Field(min_length=8, max_length=8)
    school_name: Optional[str] = Field(default=None, max_length=200)
    school_type: Optional[str] = Field(default=None, max_length=40)
    event_name: Optional[str] = Field(default=None, max_length=200)
    is_closure: int = Field(default=0, ge=0, le=1)
    event_content: Optional[str] = Field(default=None, max_length=500)

    @field_validator("date")
    @classmethod
    def _check_date(cls, v: str) -> str:
        if not v.isdigit() or len(v) != 8:
            raise ValueError(f"date must be 8-digit YYYYMMDD, got {v!r}")
        return v


# ──────────────────────────────────────────────────────────────────────────
# Public helpers
# ──────────────────────────────────────────────────────────────────────────
def validate_batch(
    schema_cls: type[BaseModel],
    rows: Iterable[dict],
    *,
    label: str = "",
) -> tuple[list[dict], list[tuple[dict, str]]]:
    """Partition `rows` into (validated, invalid).

    Parameters
    ----------
    schema_cls : type[BaseModel]
        Pydantic class to validate against (e.g. `HiraGenderAgeRow`).
    rows : Iterable[dict]
        Pre-shaped dicts already in the DB-column shape.
    label : str, optional
        Free-form label for log messages (e.g. "H3 gender_age kcd=B019").

    Returns
    -------
    (validated, invalid) : (list[dict], list[tuple[dict, str]])
        `validated` contains dicts ready for `executemany`.
        `invalid` contains `(original_row, error_message)` pairs so the
        caller can decide whether to persist them for offline review.

    Never raises. If `schema_cls` fails at all, the offending row is
    logged at WARN and collected into `invalid` — the collector keeps
    iterating over the rest of the batch.
    """
    validated: list[dict] = []
    invalid: list[tuple[dict, str]] = []
    tag = label or schema_cls.__name__

    for r in rows:
        try:
            m = schema_cls.model_validate(r)
            validated.append(m.model_dump())
        except Exception as e:
            invalid.append((r, str(e)))

    if invalid:
        log.warning(
            "[schemas] %s: %d valid, %d invalid (first error: %s)",
            tag, len(validated), len(invalid), invalid[0][1][:160],
        )
    elif validated:
        log.debug("[schemas] %s: %d valid", tag, len(validated))
    return validated, invalid


# Convenience alias to make "which DB table does this schema back?"
# discoverable at a glance.
SCHEMA_BY_TABLE: dict[str, type[BaseModel]] = {
    "hira_inpat_opat": HiraInpatOpatRow,
    "hira_gender_age": HiraGenderAgeRow,
    "hira_facility": HiraFacilityRow,
    "hira_region": HiraRegionRow,
    "school_info_seoul": SchoolInfoRow,
    "school_closure_seoul": SchoolScheduleRow,
}


def validate_for_table(
    table: str, rows: Iterable[dict], *, label: str = "",
) -> tuple[list[dict], list[tuple[dict, str]]]:
    """Dispatch to the right schema based on DB table name.

    Raises `KeyError` if the table has no registered schema — the caller
    should treat that as a developer error (schemas.py is out of sync).
    """
    schema_cls = SCHEMA_BY_TABLE[table]
    return validate_batch(schema_cls, rows, label=label or table)


__all__ = [
    "HiraBaseRow",
    "HiraInpatOpatRow",
    "HiraGenderAgeRow",
    "HiraFacilityRow",
    "HiraRegionRow",
    "SchoolInfoRow",
    "SchoolScheduleRow",
    "SCHEMA_BY_TABLE",
    "validate_batch",
    "validate_for_table",
]
