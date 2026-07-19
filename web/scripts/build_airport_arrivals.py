#!/usr/bin/env python3
"""국제선 입국 '인원수' 수집기 — KOSIS 외래객 입국(월별, 명) → 유입압 신호.

정정(2026-06-10, 사용자 지적 + 3-렌즈 조사): 이전 버전은 data.go.kr StatusOfPassengerFlightsDSOdp
(운항편 게이트·편명·시각) 였는데 **승객 머릿수가 없어 역학적으로 무용**. 감염병/ILI 에 영향을 주는 건
**입국 인원·규모**다. 그 데이터는 KOSIS 한국관광공사(orgId=314) 외래객 입국이 월별·다년 historical 로
존재(우리 KOSIS 키 작동) → flight-status 폐기, 이걸로 교체.

조사 실측(overlap 2019-07~2025-12, n=77): 입국 인원 vs 계절 ILI 의 raw 상관 +0.41, 탈계절 +0.52 처럼
보이나 **COVID 동반붕괴 artifact** — COVID dummy 통제 시 부분상관 r=+0.009(p=0.94)로 붕괴, 정상-여행
구간 r=+0.14(n.s.). 즉 **정상시 입국 인원은 계절 ILI 의 유의미한 예측인자가 아님**(여행=가을 정점,
ILI=겨울 정점, 정반대 계절성·confounded). 따라서:
  - 계절 ILI forecaster feature 로 **절대 넣지 않음**(spurious COVID 계수·오해 SHAP 방지).
  - 진짜 역할 = **팬데믹 유입(importation) 선행압** — 신종 병원체는 유입 없이는 국내 유행 시작 불가
    (2009 H1N1·COVID 문헌). external-risk.json 의 누수-안전 mode 게이트 입력으로만 사용(WATCH 보조,
    KDCA≥3 hard 트리거 금지). 신종에는 정반대-계절성 confound 가 없어 정당.

출력 = arrivals-monthly.json (월별 인원 + 유입압 z-score anomaly). DB 미적재(분석 산출은 cached csv).
Run: .venv/bin/python web/scripts/build_airport_arrivals.py
"""
from __future__ import annotations

import csv
import datetime
import json
import ssl
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AGG = ROOT / "web" / "public" / "aggregates"
CACHE = ROOT / "simulation" / "data" / "external" / "kto_foreign_arrivals_monthly.csv"
KOSIS_BASE = "https://kosis.kr/openapi/Param/statisticsParameterData.do"
# 검증된 recipe (조사 라이브 확인): orgId=314 한국관광공사 외래객 입국, 월별, 단위=명. 두 테이블 타일.
ORG = "314"
_TBL_NEW = {"tblId": "DT_TRD_TGT_ENT_AGG_MONTH", "itmId": "13103314422T01", "objL1": "13102314422A.1"}  # 2022~
_CTX = ssl.create_default_context(); _CTX.check_hostname = False; _CTX.verify_mode = ssl.CERT_NONE


def _load_cached() -> dict:
    """검증된 cached 시리즈(2015~2025) → {yyyymm:int → persons:int}. 네트워크 무관 기준."""
    out = {}
    if CACHE.is_file():
        with CACHE.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    out[int(row["yyyymm"])] = int(float(row["arrivals_persons"]))
                except (ValueError, KeyError, TypeError):
                    continue
    return out


def fetch_kosis_year(key: str, year: int) -> dict:
    """KOSIS 외래객 입국 한 해치(월별) fetch → {yyyymm:int → persons}. 실패 시 {} (graceful)."""
    try:
        q = {"method": "getList", "apiKey": key, "format": "json", "jsonVD": "Y",
             "prdSe": "M", "startPrdDe": f"{year}01", "endPrdDe": f"{year}12",
             "orgId": ORG, **_TBL_NEW}
        raw = urllib.request.urlopen(KOSIS_BASE + "?" + urllib.parse.urlencode(q),
                                     timeout=20, context=_CTX).read().decode("utf-8", "replace")
        data = json.loads(raw)
        if not isinstance(data, list):
            return {}
        out = {}
        for d in data:
            try:
                out[int(d["PRD_DE"])] = int(float(d["DT"]))
            except (ValueError, KeyError, TypeError):
                continue
        return out
    except Exception:
        return {}


def _importation_z(series: dict, asof: int) -> dict:
    """유입압 = 최신월 입국 인원의 **탈계절 z-score**(같은 calendar-month 과거 분포 대비).

    Args:
        series: {yyyymm:int → persons:int}.
        asof: 기준 yyyymm (이 달의 anomaly 를 계산).

    Returns:
        {asof, persons, month_mean, month_std, z, n_history}. 같은 달 과거표본 <3 이면 z=0.0(판단보류).
        raw 볼륨이 아니라 anomaly 인 이유: "여름 입국 많음"은 그냥 관광이라 단독 트리거 금지(조사 결론).

    Side effects: none.
    """
    if asof not in series:
        return {"asof": asof, "persons": None, "z": 0.0, "n_history": 0}
    cal_m = asof % 100
    hist = [v for ym, v in series.items() if ym % 100 == cal_m and ym < asof]
    persons = series[asof]
    if len(hist) < 3:
        return {"asof": asof, "persons": persons, "month_mean": None, "month_std": None,
                "z": 0.0, "n_history": len(hist)}
    mean = sum(hist) / len(hist)
    var = sum((x - mean) ** 2 for x in hist) / len(hist)
    std = var ** 0.5
    z = 0.0 if std < 1e-9 else (persons - mean) / std
    return {"asof": asof, "persons": persons, "month_mean": round(mean), "month_std": round(std),
            "z": round(z, 3), "n_history": len(hist)}


def main() -> int:
    try:
        from simulation.collectors.config import KEYS
        key = KEYS.get("kosis", "")
    except Exception:
        key = ""
    now = datetime.datetime.now(datetime.timezone.utc)

    series = _load_cached()
    base_n = len(series)
    # 현재+직전 해 라이브 refresh (graceful — 실패해도 cached 사용)
    refreshed = 0
    if key:
        for yr in (now.year, now.year - 1):
            got = fetch_kosis_year(key, yr)
            for ym, v in got.items():
                if ym not in series:
                    refreshed += 1
                series[ym] = v

    asof = max(series) if series else int(now.strftime("%Y%m"))
    imp = _importation_z(series, asof)

    payload = {
        "generated_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "unit": "persons(명)", "freq": "monthly",
        "source": "KOSIS 한국관광공사(orgId=314) 외래객 입국 — 국제선 입국 인원(여객수 아님, 운항편수 아님)",
        "n_months": len(series), "n_cached": base_n, "n_refreshed_live": refreshed,
        "latest": {"yyyymm": asof, "persons": series.get(asof)},
        "importation_pressure": imp,
        "role": "팬데믹 유입(importation) mode 게이트 입력 전용 — 계절 ILI forecaster feature 아님(누수 없음). "
                "정상시 계절 ILI 와 상관은 COVID 통제 후 r≈0(정반대 계절성 confounded). 신종 병원체 유입압에만 유효.",
        "months": {str(ym): series[ym] for ym in sorted(series)},
    }
    AGG.mkdir(parents=True, exist_ok=True)
    (AGG / "arrivals-monthly.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    # 옛 flight-status stub 정리
    old = AGG / "airport-arrivals.json"
    if old.is_file():
        old.unlink()

    print("=== 국제선 입국 인원(KOSIS 외래객 입국) ===")
    print(f"  {len(series)}개월 (cached {base_n} + live refresh {refreshed}), 최신 {asof} = {series.get(asof):,}명")
    print(f"  유입압 z-score(탈계절, {asof}) = {imp['z']} (같은달 과거 {imp['n_history']}표본)")
    print("  역할: 팬데믹 유입 WATCH 신호(계절 forecaster 아님 — COVID 통제 후 계절상관 ≈0).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
