"""Smoke tests for simulation.utils.http SSOT download wrapper (TDD, 2026-05-28).

네트워크 없이 검증: session 구성/retry config/singleton/http_get 위임(monkeypatch).
"""
from __future__ import annotations

import requests

from simulation.utils import http as H


def test_make_session_mounts_adapter_both_schemes():
    s = H.make_session()
    assert isinstance(s, requests.Session)
    assert s.get_adapter("https://x") is s.get_adapter("http://x")  # same adapter object


def test_retry_config():
    s = H.make_session(retries=5, backoff=0.25)
    retry = s.get_adapter("https://x").max_retries
    assert retry.total == 5
    assert retry.backoff_factor == 0.25
    assert 503 in retry.status_forcelist and 429 in retry.status_forcelist
    # raw requests.get 동작 보존 — 최종 status 에서 예외 raise 안 함
    assert retry.raise_on_status is False


def test_session_singleton():
    assert H.session() is H.session()


def test_defaults():
    assert H.DEFAULT_TIMEOUT_S == 20
    assert H.DEFAULT_RETRIES == 3
    assert set(H.RETRY_STATUS) >= {429, 500, 502, 503, 504}


def test_http_get_delegates_to_shared_session(monkeypatch):
    captured = {}

    class _FakeResp:
        status_code = 200

    def _fake_get(url, params=None, headers=None, timeout=None, **kw):
        captured.update(url=url, timeout=timeout, params=params)
        return _FakeResp()

    monkeypatch.setattr(H.session(), "get", _fake_get)
    r = H.http_get("https://example.test/data", params={"a": 1})
    assert r.status_code == 200
    assert captured["url"] == "https://example.test/data"
    assert captured["timeout"] == H.DEFAULT_TIMEOUT_S  # default applied
    assert captured["params"] == {"a": 1}
