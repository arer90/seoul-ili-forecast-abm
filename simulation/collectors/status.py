"""Collection status vocabulary — single source of truth.

Before this module existed, collector modules used raw string literals
("OK", "SKIP", "FAIL", "EMPTY") scattered across 22 files, and audit
scripts would treat any status != "OK" as "FAIL". That caused
SPOP_LOCAL_RESD_* to appear broken when they were actually skipping
because the data was already fresh.

Now:
    * Every collector writes via the constants below (or `log_outcome()`).
    * Every audit/monitoring script classifies via the helper sets.
    * Adding a new status (e.g. "RATE_LIMITED") means adding it here ONCE
      and the audit tooling recognizes it automatically.

Usage in a collector:
    from simulation.collectors.status import Status, log_outcome
    log_outcome(group="C", api="SPOP_DAILYSUM_JACHI",
                status=Status.OK if n > 0 else Status.SKIP,
                rows=n, elapsed=time.time() - t0)

Usage in an audit script:
    from simulation.collectors.status import Status, is_failure, is_healthy
    if is_failure(row["status"]): ...
"""
from __future__ import annotations

from typing import Final


class Status:
    """Enumerated string constants for `collection_log.status`.

    Keep as a class of string constants rather than a Python Enum so that
    the values round-trip through SQLite text columns with zero coercion.
    """
    # Successful collection: ≥1 new row persisted to DB.
    OK: Final[str] = "OK"

    # Intentional skip — data for this target already fresh in DB,
    # or the source API reports "not yet published" (e.g. Seoul INFO-200).
    # NOT a problem.
    SKIP: Final[str] = "SKIP"

    # API returned 200/OK with zero rows. Typically upstream has nothing
    # new yet (e.g. today's FluNet not yet released). NOT a code bug.
    EMPTY: Final[str] = "EMPTY"

    # HTTP error, timeout, parse failure, or exception after retry
    # exhaustion. Indicates a real problem — investigate.
    FAIL: Final[str] = "FAIL"

    # Uncaught Python exception above the collector's own try/except.
    # Rarer than FAIL; generally means a bug in our code, not the API.
    ERROR: Final[str] = "ERROR"

    # Specialized failures — keep as first-class values so audit scripts
    # don't collapse them into generic FAIL.
    FAIL_IMPORT: Final[str] = "FAIL_IMPORT"            # module-level import blocked
    FAIL_NO_SCHOOLS: Final[str] = "FAIL_NO_SCHOOLS"    # school_schedule upstream empty list
    FAIL_NO_KEY: Final[str] = "FAIL_NO_KEY"            # required API key missing in KEYS
    RATE_LIMITED: Final[str] = "RATE_LIMITED"          # 429; caller should back off


# ─── Classification helpers (use these in audit / monitoring code) ───

#: All status strings that count as "real failure needing investigation".
FAILURE_STATUSES: frozenset[str] = frozenset({
    Status.FAIL,
    Status.ERROR,
    Status.FAIL_IMPORT,
    Status.FAIL_NO_SCHOOLS,
    Status.FAIL_NO_KEY,
})

#: All status strings that indicate nothing was persisted this call but
#: that's NOT a bug (data fresh, upstream empty, deliberate rate-limit).
BENIGN_NO_PROGRESS: frozenset[str] = frozenset({
    Status.SKIP,
    Status.EMPTY,
    Status.RATE_LIMITED,
})

#: All known status values. If a value NOT in this set appears in the DB,
#: it's an un-documented status from some collector that bypassed the
#: enum — the classifier returns False from all helpers, surfacing it.
KNOWN_STATUSES: frozenset[str] = frozenset({
    Status.OK, Status.SKIP, Status.EMPTY, Status.FAIL, Status.ERROR,
    Status.FAIL_IMPORT, Status.FAIL_NO_SCHOOLS, Status.FAIL_NO_KEY,
    Status.RATE_LIMITED,
})


def is_success(status: str) -> bool:
    """True when the call persisted at least one new row."""
    return status == Status.OK


def is_failure(status: str) -> bool:
    """True only for real failures; SKIP/EMPTY do NOT count."""
    return status in FAILURE_STATUSES


def is_benign_no_progress(status: str) -> bool:
    """True when this call made no progress but the system is healthy."""
    return status in BENIGN_NO_PROGRESS


def is_known(status: str) -> bool:
    """False means the status string is not in our vocabulary —
    probably a raw literal in a collector that should be refactored."""
    return status in KNOWN_STATUSES
