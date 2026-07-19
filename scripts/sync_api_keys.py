#!/usr/bin/env python3
"""api_key.txt → web/.env.local 자동 sync.

사용자가 simulation/data/api_key.txt 에 모든 key 를 한 곳에 모음.
Next.js (web/) 는 .env.local 만 자동 읽어서 — 매핑 sync 필요.

이 스크립트는:
1. simulation/data/api_key.txt 파싱 (한국어 라벨)
2. 매핑된 키만 web/.env.local 에 추가/갱신
3. 기존 .env.local 의 다른 변수는 보존

실행:
    python scripts/sync_api_keys.py

자동 실행 (권장):
    pre-commit hook 또는 npm postinstall
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# 매핑 — api_key.txt 의 한국어 라벨 → web/.env.local 의 환경변수 이름
# (라벨은 substring match — 부분일치 OK)
MAPPING = {
    "Vword": "NEXT_PUBLIC_VWORLD_KEY",        # VWorld 한국 정밀
    "Vworld": "NEXT_PUBLIC_VWORLD_KEY",        # 오타 보호
    "카카오 Javascript": "NEXT_PUBLIC_KAKAO_MAP_KEY",
    "Nasa earth data": "NASA_EARTHDATA_TOKEN",
    "NASA earth": "NASA_EARTHDATA_TOKEN",
    # 2026-05-07: Claude API (sole LLM provider). Multiple aliases so the
    # user can write the label in whatever form feels natural.
    "Anthropic Claude API": "ANTHROPIC_API_KEY",
    "Anthropic": "ANTHROPIC_API_KEY",
    "Claude API": "ANTHROPIC_API_KEY",
    "Claude": "ANTHROPIC_API_KEY",
}


def parse_api_key_txt(path: Path) -> dict[str, str]:
    """api_key.txt 파싱 → {label: value}."""
    if not path.exists():
        return {}

    result = {}
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        label, _, value = line.partition(":")
        label = label.strip()
        value = value.strip()
        if value:
            result[label] = value

    return result


def parse_naver_keys(api_keys: dict) -> tuple[str | None, str | None]:
    """'네이버 지도: {client:..., client secret:...}' 파싱."""
    naver_line = None
    for label, value in api_keys.items():
        if "네이버" in label and "지도" in label:
            naver_line = value
            break
    if not naver_line:
        return None, None

    # {client:xxx, client secret:yyy} 또는 {client:xxx, client_secret:yyy}
    cid_match = re.search(r"client\s*:?\s*([\w-]+)", naver_line, re.IGNORECASE)
    csec_match = re.search(
        r"client[\s_]*secret\s*:?\s*([\w-]+)", naver_line, re.IGNORECASE,
    )
    cid = cid_match.group(1) if cid_match else None
    csec = csec_match.group(1) if csec_match else None
    return cid, csec


def parse_env_local(path: Path) -> dict[str, str]:
    """기존 .env.local 파싱 → {key: value}."""
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            result[key.strip()] = value
    return result


def write_env_local(path: Path, env: dict[str, str], header: str = ""):
    """env 를 .env.local 로 저장. ANTHROPIC_API_KEY 가 비어있을 때
    사용자가 어디에 키를 넣어야 할지 안내 주석을 자동 포함."""
    lines = []
    if header:
        lines.append(f"# {header}")
        lines.append("")

    # ANTHROPIC_API_KEY 는 가장 중요 (ARIA 채팅 동작 조건)
    # → 비어있으면 가이드 주석 + 위쪽 배치, 채워져 있으면 그대로
    anth_key = "ANTHROPIC_API_KEY"
    anth_val = env.get(anth_key, "")

    lines.append("# ── 🤖 Claude API key — REQUIRED for ARIA chat ──")
    lines.append("# 발급: https://console.anthropic.com/settings/keys")
    lines.append("# = 뒤에 sk-ant-api03-... 값 붙여넣기 (공백 / 따옴표 / 줄바꿈 X)")
    lines.append("# 저장 후 `npm run dev` (web/) 재시작하면 반영")
    lines.append(f"{anth_key}={anth_val}")
    lines.append("")

    # 나머지 변수 (정렬)
    for key in sorted(env.keys()):
        if key == anth_key:
            continue  # already written above
        lines.append(f"{key}={env[key]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    project_root = Path(__file__).resolve().parent.parent
    api_path = project_root / "simulation" / "data" / "api_key.txt"
    env_path = project_root / "web" / ".env.local"

    print(f"📖 source : {api_path}")
    print(f"📝 target : {env_path}")
    print()

    if not api_path.exists():
        print(f"❌ {api_path} 없음")
        sys.exit(1)

    # 1. 기존 .env.local 보존 (다른 변수: DEMO_TOKEN, MCP_BRIDGE_URL 등)
    existing = parse_env_local(env_path)

    # 2. api_key.txt 파싱
    keys = parse_api_key_txt(api_path)
    print(f"📖 api_key.txt 에서 {len(keys)} 항목 발견")

    # 3. 매핑 적용
    updated = {}
    for label, env_name in MAPPING.items():
        for k, v in keys.items():
            if label.lower() in k.lower():
                updated[env_name] = v
                print(f"  ✓ '{label}' → {env_name} ({len(v)} chars)")
                break

    # 4. 네이버 (특수 형식)
    naver_id, naver_secret = parse_naver_keys(keys)
    if naver_id:
        updated["NEXT_PUBLIC_NAVER_MAP_CLIENT_ID"] = naver_id
        print(f"  ✓ 네이버 client ID → NEXT_PUBLIC_NAVER_MAP_CLIENT_ID ({len(naver_id)} chars)")
    if naver_secret:
        updated["NAVER_MAP_CLIENT_SECRET"] = naver_secret
        print(f"  ✓ 네이버 client secret → NAVER_MAP_CLIENT_SECRET ({len(naver_secret)} chars)")

    # 5. 기존 변수 + 새 변수 merge (새 변수 우선)
    final = {**existing, **updated}

    # 6. 저장
    write_env_local(
        env_path, final,
        header=("Auto-generated from simulation/data/api_key.txt by "
                "scripts/sync_api_keys.py — DO NOT commit"),
    )

    print()
    print(f"✅ {env_path} 갱신 완료 ({len(final)} 변수)")
    print()
    print("📋 변경 내역:")
    new = set(updated.keys()) - set(existing.keys())
    if new:
        for k in sorted(new):
            print(f"  + {k}")
    same = set(updated.keys()) & set(existing.keys())
    if same:
        for k in sorted(same):
            if updated[k] != existing.get(k, ""):
                print(f"  ~ {k} (변경됨)")


if __name__ == "__main__":
    main()
