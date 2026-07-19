"""Deterministic multi-encoding text reader — SSOT (2026-05-28).

흩어진 ``utf-8-sig`` / ``cp949`` read 패턴 통일. chardet 류 **확률적** 인코딩
감지 대신 **결정적 ordered fallback** 사용 — 재현성(#5) 보장 (동일 입력 → 동일
결과, chardet 버전/짧은 입력 오탐 위험 없음).

용도: 외부(KDCA·overseas) CSV/텍스트처럼 인코딩이 utf-8 / utf-8-sig(BOM) /
cp949 / euc-kr 로 갈릴 수 있는 ingestion read. 내부에서 우리가 쓴 파일은
인코딩이 확실하므로 ``encoding="utf-8"`` explicit read 가 더 적합 (이 helper 불필요).
"""
from __future__ import annotations

from pathlib import Path

#: 시도 순서. utf-8-sig 가 BOM 유무 utf-8 을 모두 커버 → 최우선.
#: cp949 ⊇ euc-kr (Windows Korean) 라 한국어 바이트는 cp949 단계에서 대개 해소.
DEFAULT_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "utf-8", "cp949", "euc-kr")


def decode_bytes_safe(
    raw: bytes,
    *,
    encodings: tuple[str, ...] = DEFAULT_ENCODINGS,
    fallback_errors: str | None = None,
) -> str:
    """bytes 를 결정적 ordered-encoding fallback 으로 decode 하여 str 반환 (SSOT core).

    Args:
        raw: 디코드할 바이트열.
        encodings: strict-decode 시도 순서. 첫 성공 즉시 반환.
        fallback_errors: None 이면 전부 실패 시 마지막 예외 raise. ``"replace"`` /
            ``"ignore"`` 등을 주면 전부 실패해도 ``encodings[0]`` + 해당 errors 로
            **절대 raise 하지 않고** lossy 디코드 (display string 용).

    Returns:
        디코드된 str (utf-8-sig 성공 시 BOM 제거).

    Raises:
        UnicodeDecodeError: 모든 인코딩 실패 AND fallback_errors is None.

    재현성: 확률적 감지 미사용 — 동일 입력 → 동일 출력.
    """
    last_exc: Exception | None = None
    for enc in encodings:
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError) as exc:
            last_exc = exc
    if fallback_errors is not None:
        return raw.decode(encodings[0], errors=fallback_errors)
    assert last_exc is not None
    raise last_exc


def read_text_safe(path, *, encodings: tuple[str, ...] = DEFAULT_ENCODINGS) -> str:
    """파일을 결정적 ordered-encoding fallback 으로 읽어 전체 텍스트(str) 반환.

    Args:
        path: 파일 경로 (str | os.PathLike). 한 번만 bytes 로 읽고 순차 decode.
        encodings: strict-decode 시도 순서 (default DEFAULT_ENCODINGS).
            첫 성공 인코딩의 결과를 즉시 반환.

    Returns:
        디코드된 전체 텍스트. utf-8-sig 성공 시 BOM 은 자동 제거.

    Raises:
        FileNotFoundError: 경로가 없을 때 (bytes read 단계).
        UnicodeDecodeError: 모든 인코딩이 strict-decode 실패 시 (마지막 시도 예외).

    Performance: O(n) — 파일 전체를 1회 bytes 로드 후 in-memory decode (재-open X).
        매우 큰 파일은 streaming 이 아니므로 caller 가 메모리 고려.
    Side effects: 디스크 read only (write/mutation 없음).
    재현성: 확률적 감지 미사용 — 동일 bytes + 동일 encodings → 항상 동일 출력.
    """
    raw = Path(path).read_bytes()  # 경로 부재 시 여기서 FileNotFoundError
    return decode_bytes_safe(raw, encodings=encodings)
