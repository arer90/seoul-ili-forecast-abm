"""Smoke tests for the pydantic collector schemas (Task D).

Covers:
 - HIRA gender_age / facility / region / inpat_opat: accept valid rows,
 reject bad years / negative counts / missing kcd_code.
 - NEIS school_info / school_schedule: 8-digit date guards.
 - validate_batch partitions valid/invalid without raising.
 - SCHEMA_BY_TABLE round-trip via validate_for_table.
"""
from __future__ import annotations

import pytest


# ── HIRA basics ─────────────────────────────────────────────────────────
def test_hira_gender_age_accepts_valid_row():
    from simulation.collectors.schemas import HiraGenderAgeRow

    row = {
        "kcd_code": "B019",
        "kcd_name": "수두",
        "ref_year": 2023,
        "sex": "M",
        "age_group": "15-19",
        "patient_count": 123,
        "spec_count": 456,
        "visit_days": 789,
        "insup_brdn_amt": 1000,
        "rpe_tamt_amt": 2000,
        "collected_at": "2026-04-17T10:00:00",
    }
    m = HiraGenderAgeRow.model_validate(row)
    d = m.model_dump()
    assert d["kcd_code"] == "B019"
    assert d["patient_count"] == 123


def test_hira_base_rejects_future_year():
    from simulation.collectors.schemas import HiraGenderAgeRow
    with pytest.raises(Exception):
        HiraGenderAgeRow.model_validate({
            "kcd_code": "B019", "ref_year": 2200,
        })


def test_hira_base_rejects_missing_kcd_code():
    from simulation.collectors.schemas import HiraInpatOpatRow
    with pytest.raises(Exception):
        HiraInpatOpatRow.model_validate({
            # no kcd_code
            "ref_year": 2023,
        })


def test_hira_rejects_negative_patient_count():
    from simulation.collectors.schemas import HiraFacilityRow
    with pytest.raises(Exception):
        HiraFacilityRow.model_validate({
            "kcd_code": "B019", "ref_year": 2023,
            "patient_count": -1,
        })


def test_hira_allows_negative_amount_fields():
    """Amount fields can legitimately go negative (claim reversals,
    recalls). Guard against over-tightening."""
    from simulation.collectors.schemas import HiraRegionRow
    m = HiraRegionRow.model_validate({
        "kcd_code": "B019", "ref_year": 2023,
        "region": "11",
        "insup_brdn_amt": -500,
        "rpe_tamt_amt": -1000,
    })
    assert m.insup_brdn_amt == -500


def test_hira_ignores_unknown_fields():
    from simulation.collectors.schemas import HiraGenderAgeRow
    m = HiraGenderAgeRow.model_validate({
        "kcd_code": "B019", "ref_year": 2023,
        "extra_garbage": "should_be_dropped",
        "raw_ptnt_cnt": 99,   # raw source field name, should drop
    })
    d = m.model_dump()
    assert "extra_garbage" not in d
    assert "raw_ptnt_cnt" not in d


# ── NEIS basics ─────────────────────────────────────────────────────────
def test_school_schedule_requires_8digit_date():
    from simulation.collectors.schemas import SchoolScheduleRow
    # OK
    m = SchoolScheduleRow.model_validate({
        "date": "20260301", "event_name": "개학",
    })
    assert m.date == "20260301"

    # Bad: wrong length
    with pytest.raises(Exception):
        SchoolScheduleRow.model_validate({"date": "2026-03-01"})
    # Bad: non-numeric
    with pytest.raises(Exception):
        SchoolScheduleRow.model_validate({"date": "2026Mar01"})


def test_school_info_found_date_blank_is_ok():
    from simulation.collectors.schemas import SchoolInfoRow
    m = SchoolInfoRow.model_validate({
        "school_code": "7010057",
        "school_name": "서울초등학교",
        "found_date": "",
    })
    assert m.found_date is None


def test_school_info_rejects_malformed_found_date():
    from simulation.collectors.schemas import SchoolInfoRow
    with pytest.raises(Exception):
        SchoolInfoRow.model_validate({
            "school_code": "7010057",
            "found_date": "not-a-date",
        })


def test_school_schedule_is_closure_bounded_0_1():
    from simulation.collectors.schemas import SchoolScheduleRow
    with pytest.raises(Exception):
        SchoolScheduleRow.model_validate({
            "date": "20260301", "is_closure": 2,
        })
    # default is 0 (open)
    m = SchoolScheduleRow.model_validate({"date": "20260301"})
    assert m.is_closure == 0


# ── validate_batch / validate_for_table ─────────────────────────────────
def test_validate_batch_partitions_valid_and_invalid():
    from simulation.collectors.schemas import (
        HiraGenderAgeRow, validate_batch,
    )
    good = {"kcd_code": "B019", "ref_year": 2023, "patient_count": 10}
    bad = {"kcd_code": "B019", "ref_year": 2023, "patient_count": -1}
    rows = [good, bad, good, bad, bad]
    validated, invalid = validate_batch(
        HiraGenderAgeRow, rows, label="unit-test",
    )
    assert len(validated) == 2
    assert len(invalid) == 3
    # invalid should preserve the original dict
    assert all(r["patient_count"] == -1 for (r, _err) in invalid)
    # error strings should mention the validation failure
    assert all(isinstance(err, str) and err for (_r, err) in invalid)


def test_validate_batch_never_raises_on_mixed_garbage():
    from simulation.collectors.schemas import (
        SchoolScheduleRow, validate_batch,
    )
    rows = [
        {"date": "20260301"},
        {"date": "bad"},
        {},                       # no date at all
        {"date": 20260302},       # int, not str — pydantic will coerce/complain
        "not-a-dict-at-all",      # type error path
    ]
    # Must not raise; worst case everything ends up in `invalid`.
    validated, invalid = validate_batch(
        SchoolScheduleRow, rows, label="stress",
    )
    assert len(validated) + len(invalid) == len(rows)


def test_validate_for_table_dispatches_correctly():
    from simulation.collectors.schemas import validate_for_table, SCHEMA_BY_TABLE

    # Round-trip: every registered table dispatches to the right class.
    assert set(SCHEMA_BY_TABLE) == {
        "hira_inpat_opat", "hira_gender_age",
        "hira_facility", "hira_region",
        "school_info_seoul", "school_closure_seoul",
    }

    validated, invalid = validate_for_table(
        "school_closure_seoul",
        [{"date": "20260101"}, {"date": "bad"}],
    )
    assert len(validated) == 1
    assert len(invalid) == 1


def test_validate_for_table_raises_on_unknown_table():
    from simulation.collectors.schemas import validate_for_table
    with pytest.raises(KeyError):
        validate_for_table("does_not_exist", [{"x": 1}])
