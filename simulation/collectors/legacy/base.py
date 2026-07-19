"""
pipeline/collectors/base.py
============================
모든 Collector가 상속하는 BaseCollector
- requests 세션 재사용, retry 로직, 타임아웃 공통 처리
"""

import time
import logging
import requests
from datetime import datetime
from ..config import TIMEOUT, MAX_RETRY, RETRY_WAIT
from simulation.database.config import redact_secrets

log = logging.getLogger(__name__)

# 429 재시도 대기 상한. 서버가 Retry-After 로 하루치(86400s)를 요구해도
# 수집 run 전체가 멈추지 않도록 자른다.
RETRY_AFTER_CAP = 120.0


def _retry_after_seconds(response, cap: float = RETRY_AFTER_CAP) -> float:
    """429 응답의 ``Retry-After`` 헤더를 초 단위로 해석한다.

    Args:
        response: ``requests`` 응답 객체 (headers 매핑 보유).
        cap: 상한 초. 헤더가 이보다 크면 cap 으로 자른다.

    Returns:
        0.0 이상 ``cap`` 이하의 초. 헤더가 없거나 delta-seconds 형식이
        아니면 0.0 (호출자의 점진적 backoff 가 그대로 쓰인다).

    Performance: O(1). Side effects: 없음.
    Caller responsibility: 반환값과 자체 backoff 중 큰 쪽을 쓸 것.
    """
    raw = getattr(response, "headers", {}).get("Retry-After")
    if not raw:
        return 0.0
    try:
        return max(0.0, min(float(str(raw).strip()), cap))
    except (TypeError, ValueError):
        # HTTP-date 형식은 드물고 파싱 가치가 낮다 — backoff 로 넘긴다.
        return 0.0


class BaseCollector:
    """공통 HTTP 요청 + retry 래퍼"""

    def __init__(self):
        self.session = requests.Session()

    def get(self, url: str, params: dict = None,
            expect_json: bool = True, timeout: int = None) -> dict | str | None:
        """
        GET 요청 → JSON dict 또는 텍스트 반환
        실패 시 MAX_RETRY 재시도 후 None 반환
        """
        t0 = time.time()
        last_err = None
        for attempt in range(1, MAX_RETRY + 1):
            try:
                r = self.session.get(
                    url, params=params,
                    timeout=timeout or TIMEOUT
                )
                r.raise_for_status()
                if expect_json:
                    return r.json()
                return r.text
            except requests.exceptions.Timeout as e:
                last_err = f"Timeout (attempt {attempt})"
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code
                last_err = redact_secrets(
                    f"HTTP {status_code}: {e.response.text[:200]}"
                )
                # 500/502/503/504 서버 오류 + 429 rate-limit 은 재시도 가치 있음.
                # 429 는 4xx 지만 일시적 — 재시도 없이 포기하면 quota 소진 시
                # 남은 항목이 조용히 누락된 채 run 이 "성공" 으로 끝난다.
                if (status_code >= 500 or status_code == 429) and attempt < MAX_RETRY:
                    wait = RETRY_WAIT * attempt  # 점진적 대기
                    if status_code == 429:
                        wait = max(wait, _retry_after_seconds(e.response))
                    log.warning(f"  Retry {attempt}/{MAX_RETRY} ({wait:.0f}s): {last_err}")
                    time.sleep(wait)
                    continue
                if status_code == 429:
                    # 재시도를 다 쓰고도 429 = quota 소진. 부분 데이터로 끝나므로
                    # grep 가능한 표식을 남긴다 (조용한 누락 방지).
                    log.error(
                        f"  QUOTA-EXHAUSTED after {MAX_RETRY} retries: "
                        f"{redact_secrets(url)} — 이후 항목이 누락됩니다"
                    )
                break  # 그 밖의 4xx 클라이언트 오류는 재시도 불필요
            except requests.exceptions.ConnectionError as e:
                last_err = f"ConnError (attempt {attempt}): {str(e)[:100]}"
            except Exception as e:
                last_err = f"Error: {str(e)[:200]}"
                break

            if attempt < MAX_RETRY:
                log.warning(f"  Retry {attempt}/{MAX_RETRY}: {last_err}")
                time.sleep(RETRY_WAIT)

        elapsed = round(time.time() - t0, 2)
        log.error(f"  FAIL after {elapsed}s: {redact_secrets(url)} → {last_err}")
        return None

    @staticmethod
    def now_iso() -> str:
        """현재 시각 ISO 8601 문자열 (초 단위)"""
        return datetime.now().isoformat(timespec="seconds")

    @staticmethod
    def safe_float(v) -> float | None:
        """값을 float로 변환, 실패하면 None"""
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def safe_int(v) -> int | None:
        try:
            return int(v)
        except (TypeError, ValueError):
            return None
