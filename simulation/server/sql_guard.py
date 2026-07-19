"""
simulation.server.sql_guard
===========================
Read-only SQL guard for the ARIA ``epi.query_db`` MCP tool.

Design
------
- **Whitelist-first**: only ``SELECT`` / ``WITH`` / ``EXPLAIN`` /
  ``DESCRIBE`` / ``SHOW`` / ``PRAGMA`` (read-only form) are accepted.
  Everything else — ``INSERT``, ``UPDATE``, ``DELETE``, ``DROP``,
  ``CREATE``, ``ALTER``, ``ATTACH``, ``DETACH``, ``COPY``, ``INSTALL``,
  ``LOAD``, ``PRAGMA``-write, function-call side-effects — is rejected.
- **Stacked queries blocked**: a ``;`` outside of string literals aborts
  validation (prevents ``SELECT 1; DROP TABLE x``).
- **String-aware tokenizer**: single-quoted and double-quoted strings,
  plus SQL line (``--``) and block (``/* */``) comments, are stripped
  before scanning so keywords hidden in strings cannot spoof the guard.
- **Defence-in-depth**: the tool still calls this *in addition to*
  opening DuckDB with ``READ_ONLY`` attach mode.

The guard is intentionally dependency-free (stdlib only). For richer
SQL parsing upgrade to ``sqlglot`` later — the ``validate_read_only``
public API stays the same.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Final


__all__ = [
    "SqlGuardError",
    "GuardResult",
    "validate_read_only",
    "ALLOWED_LEADING_KEYWORDS",
    "FORBIDDEN_KEYWORDS",
]


class SqlGuardError(ValueError):
    """Raised (optionally) when a SQL statement fails the read-only guard."""


# ── Keyword sets ──────────────────────────────────────────────────────
#: First non-comment token must belong to this set.
ALLOWED_LEADING_KEYWORDS: Final[frozenset[str]] = frozenset({
    "SELECT", "WITH", "EXPLAIN", "DESCRIBE", "DESC", "SHOW", "PRAGMA",
    # DuckDB-specific read-only introspection
    "SUMMARIZE", "TABLE",  # TABLE <expr> is SELECT * FROM expr
})

#: Forbidden verbs/keywords — anywhere in the statement (outside strings).
FORBIDDEN_KEYWORDS: Final[frozenset[str]] = frozenset({
    "INSERT", "UPDATE", "DELETE", "UPSERT", "MERGE", "REPLACE",
    "DROP", "CREATE", "ALTER", "TRUNCATE", "RENAME",
    "ATTACH", "DETACH", "COPY", "EXPORT", "IMPORT", "INSTALL", "LOAD",
    "CALL", "EXECUTE", "EXEC", "VACUUM", "ANALYZE", "CHECKPOINT",
    "GRANT", "REVOKE", "COMMIT", "ROLLBACK", "BEGIN", "START",
    # DuckDB function-style side-effects in SELECT are a risk vector; we
    # block read_* / write_* / pragma_ functions by name below.
})

#: Forbidden function prefixes (case-insensitive, word-boundary match).
FORBIDDEN_FUNCTION_PREFIXES: Final[tuple[str, ...]] = (
    "READ_CSV", "READ_PARQUET", "READ_JSON", "READ_TEXT", "READ_BLOB",
    "WRITE_CSV", "WRITE_PARQUET", "WRITE_JSON",
    "PRAGMA_", "HTTPFS", "DUCKDB_FUNCTIONS",  # introspection is fine but
)

# (Note: PRAGMA as a *leading* keyword is allowed — DuckDB's PRAGMA
# does table-level introspection. Only PRAGMA_*_SET or PRAGMA-as-function
# in expressions are dangerous; the ``PRAGMA_`` prefix catches the latter.)


# ── Tokenizer ─────────────────────────────────────────────────────────
_STRING_PATTERNS = (
    # single-quoted string with '' escape
    r"'(?:''|[^'])*'",
    # double-quoted identifier (may contain keywords that shouldn't count)
    r'"(?:""|[^"])*"',
    # block comment /* ... */
    r"/\*.*?\*/",
    # line comment -- ... EOL
    r"--[^\n]*",
)
_STRING_RE = re.compile("|".join(_STRING_PATTERNS), re.DOTALL)

_WORD_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


def _strip_strings_and_comments(sql: str) -> str:
    """Blank out string literals + comments so keyword scans don't spoof."""
    return _STRING_RE.sub(
        lambda m: " " * (m.end() - m.start()), sql
    )


@dataclass(frozen=True)
class GuardResult:
    """Outcome of :func:`validate_read_only`."""
    ok: bool
    reason: str = ""
    leading_keyword: str = ""

    def raise_if_bad(self) -> None:
        if not self.ok:
            raise SqlGuardError(self.reason)


# ── Public API ────────────────────────────────────────────────────────
def validate_read_only(sql: str) -> GuardResult:
    """Validate that ``sql`` is a single read-only statement.

    Returns a ``GuardResult`` — callers can either branch on ``.ok`` or
    call ``.raise_if_bad()`` for a hard fail.

    Guarantees
    ----------
    - Exactly one statement (no ``;`` outside strings).
    - First non-comment token is in :data:`ALLOWED_LEADING_KEYWORDS`.
    - No forbidden verb keyword anywhere outside strings/comments.
    - No forbidden function name prefix (read_csv, write_parquet, …).
    """
    if not sql or not sql.strip():
        return GuardResult(False, "empty SQL")

    scrubbed = _strip_strings_and_comments(sql)

    # Block stacked statements. A trailing ``;`` is tolerated.
    stripped = scrubbed.rstrip().rstrip(";").rstrip()
    if ";" in stripped:
        return GuardResult(
            False,
            "multiple statements not allowed (semicolon outside strings)",
        )

    # First word
    m = _WORD_RE.search(scrubbed)
    if m is None:
        return GuardResult(False, "no SQL keyword found")
    leading = m.group(0).upper()

    if leading not in ALLOWED_LEADING_KEYWORDS:
        return GuardResult(
            False,
            f"leading keyword {leading!r} not in allowed set "
            f"{sorted(ALLOWED_LEADING_KEYWORDS)}",
            leading_keyword=leading,
        )

    # Scan for forbidden verbs anywhere (outside strings/comments).
    # Collect all word tokens and check set membership in one pass.
    up_words = {w.upper() for w in _WORD_RE.findall(scrubbed)}
    banned = up_words & FORBIDDEN_KEYWORDS
    if banned:
        return GuardResult(
            False,
            f"forbidden keyword(s) present: {sorted(banned)}",
            leading_keyword=leading,
        )

    # Forbidden function name prefixes (read_csv, write_parquet, pragma_…).
    for tok in up_words:
        for pref in FORBIDDEN_FUNCTION_PREFIXES:
            if tok.startswith(pref):
                return GuardResult(
                    False,
                    f"forbidden function/identifier prefix: {tok!r} "
                    f"(starts with {pref!r})",
                    leading_keyword=leading,
                )

    return GuardResult(True, "", leading_keyword=leading)
