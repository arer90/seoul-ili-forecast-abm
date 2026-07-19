#!/usr/bin/env python3
"""L0 외부 위험신호 수집 → external-risk.json (평상시/팬데믹 mode 게이트 입력).

설계 SSOT: docs/PANDEMIC_MODE_DESIGN_20260610.md. 큰 사건(팬데믹/대규모 outbreak)은 로컬 ILI
trajectory 가 아니라 외부 선행/권위 신호로 감지한다. 채택 소스(워크플로 적대검증):
  - KDCA 위기경보 4단계(관심1/주의2/경계3/심각4) = 유일 hard PANDEMIC 트리거(FP≈0). 공개 API
    없음 → 운영자 수동 ordinal(web/public/aggregates/kdca-alert.json), 미존재 시 0(평시).
  - WHO DON REST API(무인증) = 국제 확증(lagging, 단독 트리거 아님). 호흡기/novel 키워드 매칭.
  - GDELT DOC 2.0(무키·15분·한국어) = 뉴스볼륨 spike → WATCH 보조 선행신호.
  - (KDCA 표본감시 ILI = 계절 anchor; 본 수집기 범위 밖, 별도 ETL.)

graceful degradation: 소스별 try/except 독립, 실패=null+error 마킹(crash 금지). 스코어링은 순수
함수(_score_don_items/_gdelt_z)로 분리해 TDD. 이 신호들은 mode 게이트(resolve_mode)의 입력일
뿐 forecast feature 가 아니다(누수 없음).

Read-only(write external-risk.json). Run: .venv/bin/python web/scripts/build_external_risk.py
"""
from __future__ import annotations

import datetime
import json
import ssl
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGG = ROOT / "web" / "public" / "aggregates"

# 호흡기 팬데믹 관련 키워드 (DON Title/Summary 매칭) — 소문자
RESP_KEYWORDS = ("influenza", "avian", "h5n", "h7n", "h9n", "respiratory", "sars",
                 "mers", "coronavirus", "covid", "pneumonia")
NOVEL_KEYWORDS = ("novel", "pandemic", "new variant", "new strain", "unknown", "outbreak",
                  "pheic", "public health emergency")
_CTX = ssl.create_default_context()
_CTX.check_hostname = False
_CTX.verify_mode = ssl.CERT_NONE


def _get(url: str, timeout: int = 15) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "MPH-ILI-monitor/1.0"})
    return urllib.request.urlopen(req, timeout=timeout, context=_CTX).read()


# ── 순수 스코어링 (TDD 대상) ──────────────────────────────────────────────────
def _score_don_items(items: list[dict]) -> dict:
    """WHO DON 항목 목록 → 호흡기/novel 스코어.

    Args:
        items: [{title: str, date: str}] (DON value 배열).

    Returns:
        {respiratory_count, novel_count, respiratory_novel_confirmed(bool), top}.
        respiratory_novel_confirmed = 호흡기 키워드 AND novel 키워드 동시 매칭 항목 존재.

    Side effects: none.
    """
    resp = nov = 0
    confirmed = False
    top = []
    for it in items:
        t = str(it.get("title", "")).lower()
        is_resp = any(k in t for k in RESP_KEYWORDS)
        is_nov = any(k in t for k in NOVEL_KEYWORDS)
        if is_resp:
            resp += 1
        if is_nov:
            nov += 1
        if is_resp and is_nov:
            confirmed = True
            top.append(it.get("title", "")[:90])
    return {"respiratory_count": resp, "novel_count": nov,
            "respiratory_novel_confirmed": bool(confirmed), "top": top[:5]}


def _gdelt_z(points: list[dict]) -> dict:
    """GDELT timelinevol 포인트 → 최근값 z-score + spike 여부.

    Args:
        points: [{date: str, value: float}] (timeline data).

    Returns:
        {latest, mean, std, z, spike(bool)}.  spike = z >= 2.0 (뉴스볼륨 급증).
        포인트 < 4 면 z=0, spike=False (불충분).

    Side effects: none.
    """
    vals = [float(p.get("value", 0) or 0) for p in points if p.get("value") is not None]
    vals = [v for v in vals if v == v]                       # drop NaN
    if len(vals) < 4:
        return {"latest": (vals[-1] if vals else 0.0), "mean": 0.0, "std": 0.0,
                "z": 0.0, "spike": False}
    latest = vals[-1]
    hist = vals[:-1]
    mean = sum(hist) / len(hist)
    var = sum((v - mean) ** 2 for v in hist) / len(hist)
    std = var ** 0.5
    if std > 1e-9:
        z = (latest - mean) / std
    else:
        # 평탄한 baseline(std≈0)에서 급등 = 명백한 spike 이나 z 가 0/0 → 상대배율로 판정.
        z = 99.0 if latest > max(3.0 * mean, 1e-6) else 0.0
    return {"latest": round(latest, 5), "mean": round(mean, 5), "std": round(std, 5),
            "z": round(z, 2), "spike": bool(z >= 2.0)}


# ── 네트워크 fetch (graceful) ─────────────────────────────────────────────────
def fetch_don() -> dict:
    try:
        url = ("https://www.who.int/api/news/diseaseoutbreaknews?$top=20"
               "&$orderby=PublicationDateAndTime%20desc"
               "&$select=Title,PublicationDateAndTime")
        raw = _get(url)
        d = json.loads(raw)
        arr = d.get("value", d if isinstance(d, list) else [])
        items = [{"title": x.get("Title", ""), "date": x.get("PublicationDateAndTime", "")}
                 for x in arr]
        sc = _score_don_items(items)
        sc["n_items"] = len(items)
        sc["error"] = None
        return sc
    except Exception as e:
        return {"respiratory_count": 0, "novel_count": 0, "respiratory_novel_confirmed": False,
                "top": [], "n_items": 0, "error": f"{type(e).__name__}: {str(e)[:80]}"}


def fetch_gdelt() -> dict:
    try:
        q = urllib.parse.quote("(influenza OR pandemic OR 독감 OR 신종) sourcelang:korean")
        url = (f"https://api.gdeltproject.org/api/v2/doc/doc?query={q}"
               "&mode=timelinevol&format=json&timespan=8weeks")
        raw = _get(url)
        d = json.loads(raw)
        tl = d.get("timeline", [])
        pts = tl[0].get("data", []) if tl else []
        z = _gdelt_z(pts)
        z["error"] = None
        return z
    except Exception as e:
        return {"latest": 0.0, "mean": 0.0, "std": 0.0, "z": 0.0, "spike": False,
                "error": f"{type(e).__name__}: {str(e)[:80]}"}


KDCA_STALE_DAYS = 21          # 수동 갱신이 이 일수보다 오래되면 stale → WATCH fail-safe


def _age_days(iso: str) -> float | None:
    """updated_at(ISO) 으로부터 경과 일수. 파싱 불가 시 None."""
    try:
        d = datetime.datetime.fromisoformat(str(iso).replace("Z", "+00:00"))
        if d.tzinfo is None:
            d = d.replace(tzinfo=datetime.timezone.utc)
        return (datetime.datetime.now(datetime.timezone.utc) - d).total_seconds() / 86400.0
    except Exception:
        return None


def read_kdca_alert() -> dict:
    """운영자 수동 KDCA 위기경보 ordinal + stale 감지(fail-safe).

    안전(codex/gemini): 파일이 존재하나 갱신이 KDCA_STALE_DAYS 보다 오래되면 stale=True →
    resolve_mode 가 최소 WATCH 로 fail-safe(미탐=가장 비싼 실패). 파일 미존재=초기 미설정(평시 0).
    """
    f = AGG / "kdca-alert.json"
    if f.is_file():
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            lvl = max(0, min(4, int(d.get("level", 0))))
            age = _age_days(d.get("updated_at", ""))
            stale = bool(age is not None and age > KDCA_STALE_DAYS)
            return {"level": lvl, "label": d.get("label", ""),
                    "updated_at": d.get("updated_at", ""), "age_days": round(age, 1) if age is not None else None,
                    "stale": stale, "source": "manual"}
        except Exception as e:
            # 파싱 오류 = 신뢰 불가 → stale 취급(fail-safe)
            return {"level": 0, "label": "평시(파싱오류)", "error": str(e)[:60], "stale": True, "source": "manual"}
    return {"level": 0, "label": "평시(미설정)", "stale": False, "source": "default",
            "note": "KDCA 위기경보는 공개 API 없음 → kdca-alert.json 수동 갱신. 미존재=초기 평시."}


def read_arrivals() -> dict:
    """arrivals-monthly.json(KOSIS 입국 인원) → 유입압 context. mode 게이트 입력(누수-안전).

    Returns:
        {z(탈계절 anomaly), latest, pressure_high(bool z≥2.5), n_months, error}.
        pressure_high = 입국 인원이 같은-달 과거 대비 +2.5σ↑ (관광 회복 등 고분산이라 보수적 컷).
        역할 = 팬데믹 유입 context(신종에만 유효; 정상 계절 ILI 와는 COVID 통제 후 r≈0) — 독립 트리거 아님.
    """
    try:
        p = AGG / "arrivals-monthly.json"
        if not p.is_file():
            return {"z": 0.0, "latest": None, "pressure_high": False, "n_months": 0, "error": "미수집"}
        d = json.loads(p.read_text(encoding="utf-8"))
        imp = d.get("importation_pressure", {})
        z = float(imp.get("z", 0.0) or 0.0)
        return {"z": z, "latest": d.get("latest"), "pressure_high": bool(z >= 2.5),
                "n_months": int(d.get("n_months", 0)), "error": None}
    except Exception as e:
        return {"z": 0.0, "latest": None, "pressure_high": False, "n_months": 0, "error": f"{type(e).__name__}"}


def main() -> int:
    don = fetch_don()
    gdelt = fetch_gdelt()
    kdca = read_kdca_alert()
    arrivals = read_arrivals()
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # 실시간(동적)

    payload = {
        "generated_at": now,
        "kdca_alert_level": kdca["level"],
        "kdca_alert": kdca,
        "don": don,
        "gdelt": gdelt,
        "arrivals": arrivals,
        "summary": {
            "respiratory_novel_confirmed": don.get("respiratory_novel_confirmed", False),
            "news_spike": gdelt.get("spike", False),
            "kdca_stale": bool(kdca.get("stale", False)),
            "arrivals_pressure_high": bool(arrivals.get("pressure_high", False)),
            "any_source_error": bool(don.get("error") or gdelt.get("error")),
        },
        "note": "mode 게이트(resolve_mode) 입력 전용. KDCA 경계(3)↑=hard PANDEMIC, DON novel·GDELT "
                "spike=WATCH 보조. 입국 유입압(arrivals)=팬데믹 seeding context(독립 트리거 아님 — "
                "정상시 계절 ILI 와 COVID 통제 후 r≈0). forecast feature 아님(누수 없음). 상세 docs/PANDEMIC_MODE_DESIGN.",
    }
    AGG.mkdir(parents=True, exist_ok=True)
    (AGG / "external-risk.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                            encoding="utf-8")
    print(f"=== external-risk.json ===")
    print(f"  KDCA 위기경보: {kdca['level']} ({kdca.get('label','')})  [{kdca['source']}]")
    print(f"  WHO DON: 호흡기 {don['respiratory_count']}/novel {don['novel_count']}/총 {don.get('n_items',0)}"
          f" · novel확증={don['respiratory_novel_confirmed']}" + (f" · ERR {don['error']}" if don.get('error') else ""))
    print(f"  GDELT: latest={gdelt['latest']} z={gdelt['z']} spike={gdelt['spike']}"
          + (f" · ERR {gdelt['error']}" if gdelt.get('error') else ""))
    print(f"  입국 유입압: z={arrivals['z']} (최신 {arrivals.get('latest')}) "
          f"pressure_high={arrivals['pressure_high']}" + (f" · ERR {arrivals['error']}" if arrivals.get('error') else ""))
    print(f"  → mode 입력: novel확증={payload['summary']['respiratory_novel_confirmed']} "
          f"spike={payload['summary']['news_spike']} arrivals압={payload['summary']['arrivals_pressure_high']} "
          f"(KDCA≥3 시 hard PANDEMIC; arrivals=seeding context, 독립 트리거 아님)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
