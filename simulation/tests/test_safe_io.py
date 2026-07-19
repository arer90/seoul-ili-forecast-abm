"""Smoke tests for simulation.utils.safe_io.read_text_safe (TDD, 2026-05-28).

결정적 multi-encoding fallback 검증 — chardet 류 확률적 감지 대체.
8 case: ascii / utf8-korean / utf8-sig(BOM) / cp949 / euc-kr / empty /
        missing / all-fail.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from simulation.utils.safe_io import read_text_safe, decode_bytes_safe, DEFAULT_ENCODINGS

_KO = "서울 인플루엔자 의사환자분율 8.6"


def _w(tmp_path: Path, name: str, data: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_plain_ascii(tmp_path):
    p = _w(tmp_path, "a.txt", b"hello world\n123")
    assert read_text_safe(p) == "hello world\n123"


def test_utf8_korean(tmp_path):
    p = _w(tmp_path, "ko.txt", _KO.encode("utf-8"))
    assert read_text_safe(p) == _KO


def test_utf8_sig_bom_stripped(tmp_path):
    # utf-8-sig 가 첫 시도 → BOM 제거되어야 (BOM 문자가 본문에 남지 않음)
    p = _w(tmp_path, "bom.txt", _KO.encode("utf-8-sig"))
    out = read_text_safe(p)
    assert out == _KO
    assert not out.startswith("﻿")


def test_cp949_korean(tmp_path):
    # utf-8 디코드 실패 → cp949 단계에서 해소
    p = _w(tmp_path, "cp949.csv", _KO.encode("cp949"))
    assert read_text_safe(p) == _KO


def test_euc_kr_korean(tmp_path):
    p = _w(tmp_path, "euckr.csv", _KO.encode("euc-kr"))
    assert read_text_safe(p) == _KO


def test_empty_file(tmp_path):
    p = _w(tmp_path, "empty.txt", b"")
    assert read_text_safe(p) == ""


def test_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_text_safe(tmp_path / "nope.txt")


def test_all_fail_raises(tmp_path):
    # 어떤 인코딩으로도 strict-decode 안 되는 바이트 → UnicodeDecodeError
    p = _w(tmp_path, "bad.bin", b"\xff\xfe\x00\x81\x80")
    with pytest.raises(UnicodeDecodeError):
        read_text_safe(p, encodings=("utf-8", "ascii"))


def test_deterministic_same_input_same_output(tmp_path):
    # 재현성: 동일 입력 반복 read → 동일 결과 (확률적 감지 아님)
    p = _w(tmp_path, "det.csv", _KO.encode("cp949"))
    assert read_text_safe(p) == read_text_safe(p) == _KO


def test_default_encoding_order():
    # 계약: utf-8-sig 가 최우선 (BOM+plain 모두 커버)
    assert DEFAULT_ENCODINGS[0] == "utf-8-sig"
    assert set(DEFAULT_ENCODINGS) >= {"utf-8-sig", "cp949", "euc-kr"}


def test_decode_bytes_safe_cp949():
    assert decode_bytes_safe(_KO.encode("cp949")) == _KO


def test_decode_bytes_safe_all_fail_raises():
    with pytest.raises(UnicodeDecodeError):
        decode_bytes_safe(b"\xff\xfe\x81", encodings=("utf-8", "ascii"))


def test_decode_bytes_safe_fallback_never_raises():
    # fallback_errors="replace" → 전부 실패해도 raise 안 함 (display string 계약)
    out = decode_bytes_safe(b"\xff\xfe\x81", encodings=("utf-8",), fallback_errors="replace")
    assert isinstance(out, str)
