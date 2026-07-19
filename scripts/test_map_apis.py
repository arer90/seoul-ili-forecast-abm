#!/usr/bin/env python3
"""지도 API endpoint 실제 호출 테스트.

각 API 가 정상 작동하는지 실제 HTTP 요청으로 확인:
- key 불필요 (OSM/Esri/CartoDB/NASA GIBS) — 단순 ping
- key 필요 (VWorld/Kakao/Naver/NASA Earthdata) — 인증 검증

값은 노출 X (key 길이만 표시).
"""
from __future__ import annotations

import os
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def load_env_local() -> dict:
    """web/.env.local 파싱."""
    env_path = Path(__file__).resolve().parent.parent / "web" / ".env.local"
    if not env_path.exists():
        return {}
    result = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, _, v = line.partition("=")
            result[k.strip()] = v.strip()
    return result


def http_test(url: str, headers: dict = None, timeout: int = 10,
               method: str = "GET") -> tuple[int, str, int]:
    """HTTP 호출 → (status_code, content_type, content_length)."""
    req = Request(url, method=method, headers=headers or {})
    try:
        with urlopen(req, timeout=timeout) as r:
            data = r.read(2048)    # max 2KB read
            return r.status, r.headers.get("Content-Type", ""), len(data)
    except HTTPError as e:
        return e.code, str(e.reason), 0
    except URLError as e:
        return 0, str(e.reason), 0
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}", 0


def fmt(name: str, ok: bool, detail: str, key_status: str = "") -> str:
    icon = "✅" if ok else "❌"
    return f"  {icon} {name:<30s} {detail:<35s}  {key_status}"


# ══════════════════════════════════════════════════════════
# Tests
# ══════════════════════════════════════════════════════════

def test_osm():
    """OpenStreetMap (key 불필요)."""
    code, ct, sz = http_test(
        "https://a.tile.openstreetmap.org/15/27947/12713.png",
        headers={"User-Agent": "MPH-test/1.0"},
    )
    return code == 200 and "image" in ct, f"HTTP {code}, {sz}B"


def test_esri_satellite():
    """Esri World Imagery (key 불필요)."""
    code, ct, sz = http_test(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/15/12713/27947",
        headers={"User-Agent": "MPH-test/1.0"},
    )
    return code == 200 and ("image" in ct.lower() or sz > 0), f"HTTP {code}, {sz}B"


def test_cartodb_dark():
    """CartoDB Dark (key 불필요)."""
    code, ct, sz = http_test(
        "https://a.basemaps.cartocdn.com/dark_all/15/27947/12713.png",
        headers={"User-Agent": "MPH-test/1.0"},
    )
    return code == 200 and "image" in ct, f"HTTP {code}, {sz}B"


def test_nasa_gibs_clouds():
    """NASA GIBS clouds (key 불필요)."""
    yest = time.strftime("%Y-%m-%d", time.gmtime(time.time() - 86400))
    url = (
        f"https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
        f"MODIS_Aqua_CorrectedReflectance_TrueColor/default/{yest}/"
        f"GoogleMapsCompatible_Level9/5/12/22.jpg"
    )
    code, ct, sz = http_test(url, headers={"User-Agent": "MPH-test/1.0"})
    return code == 200, f"HTTP {code}, {sz}B"


def test_vworld(key: str):
    """VWorld 위성 (key 필요)."""
    if not key:
        return False, "(key 없음)"
    url = (
        f"https://api.vworld.kr/req/wmts/1.0.0/{key}/"
        f"Satellite/15/27947/12713.jpeg"
    )
    code, ct, sz = http_test(url, headers={"User-Agent": "MPH-test/1.0"})
    if code == 200:
        return True, f"HTTP 200, {sz}B"
    return False, f"HTTP {code} (도메인 미등록 or 키 오류)"


def test_kakao_rest(rest_key: str):
    """카카오 REST API (key 필요).
    JS key 와 다르지만 동일한 32-char 형식이라 간단 검증."""
    if not rest_key:
        return False, "(key 없음)"
    # local 검색 endpoint (REST key 로만 호출 가능, JS key 는 401)
    url = "https://dapi.kakao.com/v2/local/search/address.json?query=서울"
    code, ct, sz = http_test(
        url,
        headers={
            "Authorization": f"KakaoAK {rest_key}",
            "User-Agent": "MPH-test/1.0",
        },
    )
    if code == 200:
        return True, f"HTTP 200, {sz}B"
    elif code == 401:
        return False, "HTTP 401 (JS key 라 REST 호출 안됨, 정상)"
    return False, f"HTTP {code}"


def test_naver_geocoding(client_id: str, client_secret: str):
    """네이버 Geocoding — NCP Maps 공식 endpoint.

    2026-04-28 fix: naveropenapi (deprecated) → maps.apigw.ntruss.com
    공식 문서: https://api.ncloud-docs.com/docs/application-maps-overview
    """
    if not client_id or not client_secret:
        return False, "(key 부족)"
    # 정확한 NCP Maps endpoint
    url = (
        "https://maps.apigw.ntruss.com/map-geocode/v2/geocode"
        f"?query={quote('서울특별시 강남구')}"
    )
    code, ct, sz = http_test(
        url,
        headers={
            "X-NCP-APIGW-API-KEY-ID": client_id,
            "X-NCP-APIGW-API-KEY": client_secret,
            "Accept": "application/json",
            "User-Agent": "MPH-test/1.0",
        },
    )
    if code == 200:
        return True, f"HTTP 200, {sz}B"
    return False, f"HTTP {code} (인증 실패 or API 미신청)"


def test_nasa_earthdata(token: str):
    """NASA Earthdata Token (CMR API)."""
    if not token:
        return False, "(token 없음)"
    # CMR collections (token 검증)
    url = "https://cmr.earthdata.nasa.gov/search/collections.json?page_size=1"
    code, ct, sz = http_test(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "User-Agent": "MPH-test/1.0",
        },
    )
    if code == 200:
        return True, f"HTTP 200, {sz}B"
    return False, f"HTTP {code}"


def test_cartodb_light():
    """CartoDB Light (key 불필요)."""
    code, ct, sz = http_test(
        "https://a.basemaps.cartocdn.com/light_all/15/27947/12713.png",
        headers={"User-Agent": "MPH-test/1.0"},
    )
    return code == 200 and "image" in ct, f"HTTP {code}, {sz}B"


def test_esri_topo():
    """Esri Topography (key 불필요)."""
    code, ct, sz = http_test(
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Topo_Map/MapServer/tile/15/12713/27947",
        headers={"User-Agent": "MPH-test/1.0"},
    )
    return code == 200, f"HTTP {code}, {sz}B"


# ══════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  🛰️  지도 API endpoint 실제 호출 테스트")
    print("=" * 70)
    print()

    env = load_env_local()
    n_pass = 0
    n_fail = 0
    n_total = 0

    print("─── No-key Tests (4 base + 1 overlay) ────────────────────────────")
    tests_nokey = [
        ("OSM", test_osm),
        ("Esri Satellite", test_esri_satellite),
        ("Esri Topography", test_esri_topo),
        ("CartoDB Dark", test_cartodb_dark),
        ("CartoDB Light", test_cartodb_light),
        ("NASA GIBS Clouds", test_nasa_gibs_clouds),
    ]
    for name, fn in tests_nokey:
        n_total += 1
        try:
            ok, detail = fn()
            print(fmt(name, ok, detail))
            if ok: n_pass += 1
            else: n_fail += 1
        except Exception as e:
            print(fmt(name, False, f"Error: {e}"))
            n_fail += 1
    print()

    print("─── Key-required Tests (4) ───────────────────────────────────────")

    # VWorld
    vworld = env.get("NEXT_PUBLIC_VWORLD_KEY", "")
    n_total += 1
    ok, detail = test_vworld(vworld)
    print(fmt("VWorld", ok, detail, f"key {len(vworld)}c"))
    if ok: n_pass += 1
    else: n_fail += 1

    # 카카오 (JS key 로 시도, REST 호출 → 401 예상)
    kakao_js = env.get("NEXT_PUBLIC_KAKAO_MAP_KEY", "")
    n_total += 1
    ok, detail = test_kakao_rest(kakao_js)
    print(fmt("Kakao (JS key, REST 시도)", ok, detail, f"key {len(kakao_js)}c"))
    if ok: n_pass += 1
    else:
        # JS key 라 REST 401 은 정상 — 표시
        if "JS key" in detail:
            print(fmt("  └ Note", True, "JS key — 브라우저에서만 작동, 정상", ""))
        n_fail += 1

    # 네이버
    naver_id = env.get("NEXT_PUBLIC_NAVER_MAP_CLIENT_ID", "")
    naver_secret = env.get("NAVER_MAP_CLIENT_SECRET", "")
    n_total += 1
    ok, detail = test_naver_geocoding(naver_id, naver_secret)
    print(fmt("Naver Geocoding",
               ok, detail,
               f"id {len(naver_id)}c + secret {len(naver_secret)}c"))
    if ok: n_pass += 1
    else: n_fail += 1

    # NASA Earthdata token
    nasa_token = env.get("NASA_EARTHDATA_TOKEN", "")
    n_total += 1
    ok, detail = test_nasa_earthdata(nasa_token)
    print(fmt("NASA Earthdata", ok, detail, f"token {len(nasa_token)}c"))
    if ok: n_pass += 1
    else: n_fail += 1

    print()
    print("=" * 70)
    print(f"  Summary: {n_pass}/{n_total} passed, {n_fail} failed")
    print("=" * 70)

    if n_fail > 0:
        print()
        print("📌 실패한 API 트러블슈팅:")
        print("  • VWorld 도메인 미등록 → https://www.vworld.kr 에서 도메인 추가")
        print("  • Naver 'API 미신청' → Geocoding API 신청 (Maps 콘솔에서)")
        print("  • Kakao JS key 401 = 정상 (브라우저에서만 작동, REST key 로는 X)")
        print("  • NASA Earthdata 401 → token 만료 or invalid")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
