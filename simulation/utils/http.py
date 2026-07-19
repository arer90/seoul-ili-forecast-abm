"""HTTP download SSOT — shared retry-equipped requests.Session (2026-05-28).

콜렉터에 산재한 raw ``requests.get`` (retry/timeout 비일관, base.py 의 retry Session
우회) 통일. ``requests`` 는 이미 hard dependency (urllib3 번들) — **신규 의존성 0**.
urllib3 ``Retry`` + ``HTTPAdapter`` 로 선언적 retry (수동 loop 보다 단순).

설계 (D-4 deep module): 작은 인터페이스 (`http_get` / `session` / `make_session`),
풍부한 구현 (connection-pool 재사용 + 지수 backoff + 5xx/429 retry).

용도: 데이터 수집 collectors 의 GET. **다운로드는 network-bound 라 속도 이득은 없고,
일관된 timeout/retry/robustness 가 목적.** raise_for_status 는 caller 책임 (기존 동작 보존).
"""
from __future__ import annotations

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

#: 기본 timeout(초) — 모든 collector GET 의 단일 기준.
DEFAULT_TIMEOUT_S: int = 20
#: transient 실패 시 총 재시도 횟수.
DEFAULT_RETRIES: int = 3
#: 재시도 대상 HTTP status (transient).
RETRY_STATUS: tuple[int, ...] = (429, 500, 502, 503, 504)


def make_session(*, retries: int = DEFAULT_RETRIES, backoff: float = 0.5) -> requests.Session:
    """retry+지수 backoff 가 https/http 양쪽에 mount 된 새 Session 반환.

    Args:
        retries: 총 재시도 횟수 (total). 0 = 재시도 없음.
        backoff: backoff_factor — n번째 재시도 전 대기 = backoff × (2**(n-1)) 초.

    Returns:
        connection-pool 재사용 + RETRY_STATUS 자동 재시도 Session.
        raise_on_status=False 라 4xx/최종-5xx 는 예외 없이 Response 반환 (caller 가
        raise_for_status 판단 — raw requests.get 동작 보존).
    """
    retry = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=RETRY_STATUS,
        allowed_methods=frozenset(["GET", "POST", "HEAD"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s = requests.Session()
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


_SESSION: requests.Session | None = None


def session() -> requests.Session:
    """프로세스 공유 singleton Session (연결 재사용 — lazy 초기화)."""
    global _SESSION
    if _SESSION is None:
        _SESSION = make_session()
    return _SESSION


def http_get(
    url: str,
    *,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = DEFAULT_TIMEOUT_S,
    **kwargs,
) -> requests.Response:
    """공유 retry-session 으로 GET. `requests.get` drop-in (raise_for_status 미호출).

    Args:
        url: 대상 URL.
        params/headers: requests 와 동일.
        timeout: 초 (default DEFAULT_TIMEOUT_S).
        **kwargs: stream/verify/auth 등 requests.get 패스스루.

    Returns:
        requests.Response (status 검사는 caller 책임).
    """
    return session().get(url, params=params, headers=headers, timeout=timeout, **kwargs)
