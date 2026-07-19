#!/usr/bin/env python3
"""Verify the NEDIS hpid→gu table against the REAL e-gen.or.kr API.

The realtime ER endpoint (getEmrrmRltmUsefulSckbdInfoInqire) lacks dutyAddr,
so web/lib/live-overlays/nedis-er.ts uses a static HPID_TO_GU table. This
script rebuilds that table from the AUTHORITATIVE address sources so no entry
is guessed:

  1. realtime feed              → the exact hpids we must cover
  2. getEgytListInfoInqire      → hpid → dutyAddr (bulk, Seoul)
  3. getEgytBassInfoInqire      → hpid → dutyAddr (per-hpid, for any missing)

Output: /tmp/hpid_to_gu_verified.json (+ a TS snippet) and a coverage report.
SECURITY: reads NEDIS_API_KEY from web/.env.local; NEVER prints the key.
"""
from __future__ import annotations

import json
import re
import sys
import time
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENV = ROOT / "web" / ".env.local"
BASE = "https://apis.data.go.kr/B552657/ErmctInfoInqireService/"
_TMP = Path(tempfile.gettempdir())
OUT_JSON = _TMP / "hpid_to_gu_verified.json"
OUT_TS = _TMP / "hpid_to_gu_verified.ts"


def get_key() -> str | None:
    if not ENV.exists():
        return None
    for line in ENV.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("NEDIS_API_KEY="):
            return line.split("=", 1)[1].strip()
    return None


KEY = get_key()
if not KEY:
    print("✗ NEDIS_API_KEY 없음 (web/.env.local)")
    sys.exit(1)


def fetch(endpoint: str, params: dict) -> str:
    """Mirror the web's `?ServiceKey=${encodeURIComponent(key)}` exactly."""
    qs = "ServiceKey=" + urllib.parse.quote(KEY, safe="")
    for k, v in params.items():
        qs += f"&{k}=" + urllib.parse.quote(str(v), safe="")
    url = BASE + endpoint + "?" + qs
    req = urllib.request.Request(url, headers={"User-Agent": "mph-verify"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return r.read().decode("utf-8", "replace")


def items(xml: str) -> list[str]:
    return re.findall(r"<item>([\s\S]*?)</item>", xml)


def field(block: str, tag: str) -> str:
    m = re.search(
        rf"<{tag}[^>]*>(?:<!\[CDATA\[)?([\s\S]*?)(?:\]\]>)?</{tag}>", block, re.I
    )
    return (m.group(1).strip() if m else "")


def gu_of(addr: str) -> str | None:
    m = re.search(r"([가-힣]+구)", addr or "")
    return m.group(1) if m else None


SEOUL_GU = {
    "종로구", "중구", "용산구", "성동구", "광진구", "동대문구", "중랑구", "성북구",
    "강북구", "도봉구", "노원구", "은평구", "서대문구", "마포구", "양천구", "강서구",
    "구로구", "금천구", "영등포구", "동작구", "관악구", "서초구", "강남구", "송파구", "강동구",
}


def main() -> int:
    # 1. realtime feed → hpids we must cover (+ their dutyName for comments)
    print("1) 실시간 피드 hpid 수집…")
    rt = fetch("getEmrrmRltmUsefulSckbdInfoInqire",
               {"STAGE1": "서울특별시", "pageNo": 1, "numOfRows": 200})
    feed: dict[str, str] = {}
    for it in items(rt):
        hp = field(it, "hpid")
        if hp:
            feed[hp] = field(it, "dutyName")
    print(f"   실시간 피드 병원 수: {len(feed)}")
    if not feed:
        print("   ✗ 피드 비어있음 (키/엔드포인트 확인)"); return 1

    # 2. bulk list endpoint → hpid → dutyAddr
    print("2) 병원목록(getEgytListInfoInqire) 주소 수집…")
    addr: dict[str, str] = {}
    name: dict[str, str] = {}
    for page in range(1, 4):  # up to 3 pages × 100
        try:
            lx = fetch("getEgytListInfoInqire",
                       {"STAGE1": "서울특별시", "pageNo": page, "numOfRows": 100})
        except Exception as e:
            print(f"   page{page} 실패: {type(e).__name__}"); break
        got = 0
        for it in items(lx):
            hp = field(it, "hpid")
            if not hp:
                continue
            addr[hp] = field(it, "dutyAddr")
            name[hp] = field(it, "dutyName")
            got += 1
        print(f"   page{page}: {got}건")
        if got < 100:
            break
    print(f"   목록 주소 확보: {len(addr)}개")

    # 3. per-hpid basinfo for feed hpids still missing an address
    missing = [hp for hp in feed if hp not in addr or not gu_of(addr.get(hp, ""))]
    print(f"3) 목록서 누락된 피드 hpid {len(missing)}개 → getEgytBassInfoInqire 개별 조회…")
    for i, hp in enumerate(missing):
        try:
            bx = fetch("getEgytBassInfoInqire", {"HPID": hp})
            it = (items(bx) or [""])[0]
            a = field(it, "dutyAddr")
            if a:
                addr[hp] = a
                if not name.get(hp):
                    name[hp] = field(it, "dutyName")
        except Exception:
            pass
        if (i + 1) % 10 == 0:
            time.sleep(0.3)  # be polite to the API

    # 4. build verified hpid → gu, only for feed hpids
    verified: dict[str, dict] = {}
    unresolved: list[str] = []
    for hp in feed:
        g = gu_of(addr.get(hp, ""))
        nm = name.get(hp) or feed.get(hp) or ""
        if g and g in SEOUL_GU:
            verified[hp] = {"gu": g, "name": nm}
        else:
            unresolved.append(hp)

    OUT_JSON.write_text(json.dumps(
        {"verified": verified, "unresolved": unresolved,
         "feed_count": len(feed)}, ensure_ascii=False, indent=2), encoding="utf-8")

    # 5. TS snippet (ready to paste) — sorted by gu then hpid for readability
    lines = ["const HPID_TO_GU: Record<string, string> = {"]
    for hp in sorted(verified, key=lambda h: (verified[h]["gu"], h)):
        nm = verified[hp]["name"].replace("*/", "")[:40]
        lines.append(f'  {hp}: "{verified[hp]["gu"]}", // {nm}')
    lines.append("};")
    OUT_TS.write_text("\n".join(lines), encoding="utf-8")

    # 6. report
    print("\n=== 검증 결과 ===")
    print(f"  실시간 피드 병원: {len(feed)}")
    print(f"  실주소로 gu 확정: {len(verified)}")
    print(f"  미해결(주소 없음): {len(unresolved)} → {unresolved[:8]}")
    gus = sorted({v["gu"] for v in verified.values()})
    print(f"  커버 자치구: {len(gus)}/25 → {gus}")
    miss_gu = sorted(SEOUL_GU - set(gus))
    print(f"  병원 없는 구: {miss_gu}")
    print(f"\n  → JSON: {OUT_JSON}")
    print(f"  → TS:   {OUT_TS}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
