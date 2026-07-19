#!/usr/bin/env python3
"""Bridge data-collection API keys (simulation/data/api_key.txt) → web/.env.local.

`simulation/data/api_key.txt` is the **Python backend** key store
(KOSIS / 기상청 / 서울 열린데이터 / KDCA …). The **Next.js** live-overlays
read `process.env` from `web/.env.local`. By design the two files are kept
separate (API_KEYS_LAYOUT.md), but the web's real-time overlays
(kma-weather / seoul-air / metro / nedis-er) need a handful of the same
data-source keys. This copies *only those* keys across.

SECURITY (hard rule):
  - NEVER prints a key value — only the matched label + character length.
  - Reads api_key.txt at runtime; the script itself contains no secrets.
  - Both api_key.txt and web/.env.local stay gitignored. Do not commit either.

Idempotent: skips any env var already present in web/.env.local.

Usage:
    .venv/bin/python scripts/bridge_api_keys_to_web.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "simulation" / "data" / "api_key.txt"
DST = ROOT / "web" / ".env.local"

# web env var → ordered list of label substrings in api_key.txt (first match wins).
# Order encodes specificity: the most-specific label is tried first so e.g.
# "일반인증키(대기)" beats the generic "일반인증키" for the air-quality key.
MAPPING: dict[str, list[str]] = {
    "KMA_API_KEY": ["기상청 api허브", "기상청 인증키"],   # apihub.kma.go.kr authKey (kma-weather.ts)
    "SEOUL_OPENAPI_KEY": ["일반인증키(대기)", "일반인증키"],  # 서울 RealtimeCityAir (seoul-air.ts)
    "METRO_API_KEY": ["지하철실시간", "지하철인증키"],     # 서울 지하철 실시간 (mobility)
    "NEDIS_API_KEY": ["공공데이터포털 서비스키"],          # apis.data.go.kr B552657 응급의료 (nedis-er.ts)
    "NEXT_PUBLIC_VWORLD_KEY": ["Vword", "디지털트윈"],      # VWorld 1m map tiles (optional)
}

_TOKEN_RE = re.compile(r"^[A-Za-z0-9_\-+/=.%]+$")
_LINE_RE = re.compile(r"^(.*?)\s*[:：]\s*(.+)$")  # split on first half/full-width colon


def parse_keys(text: str) -> dict[str, str]:
    """Parse 'label : value' lines into {label: token}.

    Only keeps lines whose value's first whitespace token looks like a real
    key (>=10 chars, key-safe charset) — this filters out headers, prose, and
    the tab-delimited 고용노동부 sub-table.
    """
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        label, val = m.group(1).strip(), m.group(2).strip()
        tok = val.split()[0] if val else ""
        if len(tok) >= 10 and _TOKEN_RE.match(tok):
            out[label] = tok
    return out


def resolve(label_kws: list[str], keys: dict[str, str]) -> tuple[str | None, str | None]:
    """Return (value, label) for the first keyword that substring-matches a label."""
    for kw in label_kws:
        for label, val in keys.items():
            if kw in label:
                return val, label
    return None, None


def main() -> int:
    if not SRC.exists():
        print(f"✗ {SRC} 없음")
        return 1
    keys = parse_keys(SRC.read_text(encoding="utf-8"))
    existing = DST.read_text(encoding="utf-8") if DST.exists() else ""
    existing_vars = set(re.findall(r"^([A-Z0-9_]+)=", existing, re.M))

    print(f"api_key.txt 파싱: {len(keys)}개 라벨")
    to_add: list[str] = []
    for env_var, kws in MAPPING.items():
        val, label = resolve(kws, keys)
        if val is None:
            print(f"  - {env_var}: 매칭 라벨 없음 (skip)")
            continue
        if env_var in existing_vars:
            print(f"  = {env_var}: 이미 web/.env.local에 존재 (유지)")
            continue
        to_add.append(f"{env_var}={val}")
        # NOTE: only label + length printed — value is NEVER echoed.
        print(f"  + {env_var} ← '{label}' (설정됨, len={len(val)})")

    if to_add:
        block = (
            "\n# --- bridged from simulation/data/api_key.txt "
            "(real-time overlays; gitignored) ---\n" + "\n".join(to_add) + "\n"
        )
        with DST.open("a", encoding="utf-8") as f:
            f.write(block)
        print(f"✓ {len(to_add)}개 키를 web/.env.local에 추가 (값 미출력)")
    else:
        print("추가할 키 없음 (전부 이미 존재 또는 매칭 실패)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
