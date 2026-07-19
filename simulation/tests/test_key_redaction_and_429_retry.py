"""Regression guards for the three key-handling fixes (2026-07-19).

The audited defects, each reproduced here before being fixed:

1. ``legacy/base.py`` logged the full request URL on failure, and Korean
   government APIs carry the key *in* that URL — so every 4xx/5xx wrote a
   plaintext credential into ``simulation/logs/collect_*.log``.
2. The same wrapper treated every 4xx as non-retryable. HTTP 429 is a 4xx, so
   quota exhaustion aborted collection immediately and produced partial data
   with nothing to distinguish it from a genuinely sparse API response.
3. (web, covered by ``test_public_demo_fails_closed``-style review rather than
   pytest) the chat route skipped rate limiting entirely when Upstash was
   absent.

Run standalone — macOS needs per-file pytest runs (LightGBM/OpenMP segfault):
    .venv/bin/python -m pytest simulation/tests/test_key_redaction_and_429_retry.py -q
"""

import logging
import re
from pathlib import Path

import pytest
import requests

from simulation.database import config as cfg
from simulation.collectors.legacy import base as legacy_base


SECRET = "AbCdEf1234567890SeoulKeyValue=="
SHORT = "xyz"


@pytest.fixture
def keyed(monkeypatch):
    """Install known key values so redaction has something to mask."""
    monkeypatch.setitem(cfg.KEYS, "seoul_general", SECRET)
    monkeypatch.setitem(cfg.KEYS, "tiny", SHORT)
    return SECRET


# ── 1. redact_secrets ────────────────────────────────────────────────────────

def test_masks_key_in_url_path(keyed):
    """Seoul keys sit in the path, not a query parameter."""
    url = f"http://openapi.seoul.go.kr:8088/{SECRET}/json/SPOP/1/5/"
    out = cfg.redact_secrets(url)
    assert SECRET not in out
    assert "***REDACTED***" in out
    assert "openapi.seoul.go.kr" in out, "surrounding URL must stay readable"


def test_masks_key_in_query_string(keyed):
    """data.go.kr takes the key as serviceKey=."""
    out = cfg.redact_secrets(f"https://apis.data.go.kr/x?serviceKey={SECRET}&n=1")
    assert SECRET not in out
    assert "serviceKey=***REDACTED***" in out


def test_masks_percent_encoded_form(keyed):
    """requests percent-encodes '+' and '=' — the encoded form must mask too."""
    from urllib.parse import quote

    out = cfg.redact_secrets(f"https://x/?k={quote(SECRET, safe='')}")
    assert quote(SECRET, safe="") not in out
    assert "***REDACTED***" in out


def test_masks_key_inside_params_dict(keyed):
    """group_s_sentinel logs params= directly."""
    out = cfg.redact_secrets({"serviceKey": SECRET, "gubun": "ILI"})
    assert SECRET not in out
    assert "gubun" in out


def test_short_values_are_left_alone(keyed):
    """A 3-char value would match unrelated substrings and mangle the line."""
    assert cfg.redact_secrets("the xyz axis") == "the xyz axis"


def test_empty_and_missing_keys_are_safe(monkeypatch):
    monkeypatch.setattr(cfg, "KEYS", {"a": "", "b": None})
    assert cfg.redact_secrets("http://example.com/ok") == "http://example.com/ok"


def test_non_string_input_is_coerced(keyed):
    assert cfg.redact_secrets(None) == "None"
    assert cfg.redact_secrets(42) == "42"


# ── 2. Retry-After parsing ───────────────────────────────────────────────────

class _Resp:
    def __init__(self, headers=None, status_code=429, text=""):
        self.headers = headers or {}
        self.status_code = status_code
        self.text = text


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("30", 30.0),
        ("0", 0.0),
        (None, 0.0),
        ("not-a-number", 0.0),          # HTTP-date form -> fall back to backoff
        ("86400", legacy_base.RETRY_AFTER_CAP),  # a full day must be capped
        ("-5", 0.0),                    # never sleep a negative duration
    ],
)
def test_retry_after_seconds(raw, expected):
    headers = {} if raw is None else {"Retry-After": raw}
    assert legacy_base._retry_after_seconds(_Resp(headers)) == expected


# ── 3. 429 is retried; other 4xx are not ─────────────────────────────────────

def _http_error(status):
    resp = _Resp(status_code=status, text=f"body {status}")
    err = requests.exceptions.HTTPError(response=resp)
    return err


class _FakeSession:
    """Raises `status` for the first `fail_times` calls, then succeeds."""

    def __init__(self, status, fail_times):
        self.status = status
        self.fail_times = fail_times
        self.calls = 0

    def get(self, url, params=None, timeout=None):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise _http_error(self.status)

        class _Ok:
            @staticmethod
            def raise_for_status():
                return None

            @staticmethod
            def json():
                return {"ok": True}

        return _Ok()


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch):
    monkeypatch.setattr(legacy_base.time, "sleep", lambda *_: None)


def test_429_is_retried_and_can_succeed():
    c = legacy_base.BaseCollector()
    c.session = _FakeSession(429, fail_times=1)
    assert c.get("http://x/") == {"ok": True}
    assert c.session.calls == 2, "429 must be retried, not abandoned"


def test_404_is_not_retried():
    c = legacy_base.BaseCollector()
    c.session = _FakeSession(404, fail_times=99)
    assert c.get("http://x/") is None
    assert c.session.calls == 1, "ordinary 4xx must still fail fast"


def test_500_is_still_retried():
    c = legacy_base.BaseCollector()
    c.session = _FakeSession(500, fail_times=1)
    assert c.get("http://x/") == {"ok": True}
    assert c.session.calls == 2


def test_exhausted_429_emits_greppable_quota_marker(caplog):
    c = legacy_base.BaseCollector()
    c.session = _FakeSession(429, fail_times=99)
    with caplog.at_level(logging.ERROR):
        assert c.get("http://x/") is None
    assert "QUOTA-EXHAUSTED" in caplog.text, (
        "silent partial data was the original harm — the marker must be greppable"
    )


# ── 4. the failure log itself must not leak ──────────────────────────────────

def test_failure_log_does_not_contain_the_key(keyed, caplog):
    c = legacy_base.BaseCollector()
    c.session = _FakeSession(500, fail_times=99)
    url = f"http://openapi.seoul.go.kr:8088/{SECRET}/json/SPOP/1/5/"
    with caplog.at_level(logging.WARNING):
        assert c.get(url) is None
    assert SECRET not in caplog.text, "the key must never reach a log record"
    assert "***REDACTED***" in caplog.text


# ── 5. static guard: no un-redacted URL logging creeps back in ───────────────

def test_no_raw_url_logging_in_legacy_collectors():
    """A log line interpolating a bare {url} is how defect 1 happened."""
    root = Path(__file__).resolve().parents[1] / "collectors" / "legacy"
    pattern = re.compile(r"log\.\w+\(\s*f?\"[^\"]*\{(?:url|params)\}")
    offenders = []
    for path in root.rglob("*.py"):
        for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pattern.search(line) and "redact_secrets" not in line:
                offenders.append(f"{path.name}:{i}")
    assert not offenders, f"un-redacted URL/params logging: {offenders}"
