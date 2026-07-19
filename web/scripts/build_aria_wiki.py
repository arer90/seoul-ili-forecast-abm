#!/usr/bin/env python3
"""Build the ARIA grounding wiki — Korean infectious-disease law + KDCA data.

ARIA's LLM consultants (hermes.ts SYSTEM_GROUNDING, applied to every provider /
mode) cite a tiered evidence policy. This generator produces the
``[법령·KDCA wiki]`` tier: the real 「감염병의 예방 및 관리에 관한 법률」(감염병
예방법) framework + KDCA surveillance facts, so each LLM answers from actual
Korean law and 질병관리청 data instead of hallucinating.

The LAW provisions are authoritative reference text (article numbers + official
sources; exact 종수 vary by amendment → verify at law.go.kr). The KDCA DATA
section is generated live from epi_real_seoul.db (disease_master legal grades,
kosis_kdca_notifiable counts, sentinel_influenza ILI) so it tracks the DB.

Output ``web/lib/aria-wiki.json`` is imported by hermes.ts at build time
(Edge runtime has no fs). Reproducible (DB read-only, no key).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB = ROOT / "simulation" / "data" / "db" / "epi_real_seoul.db"
OUT = ROOT / "web" / "lib" / "aria-wiki.json"

#: Authoritative 감염병예방법 framework (verified facts; cite law.go.kr / KDCA).
LAW = {
    "name": "「감염병의 예방 및 관리에 관한 법률」(약칭 감염병예방법)",
    "authority": "질병관리청(KDCA) — 시·도지사 / 시장·군수·구청장 협조",
    "classification": (
        "2020 개정 후 제1~4급 체계. 인플루엔자 = 제4급감염병 → 표본감시(sentinel) 대상."
    ),
    "provisions": [
        "제2조(정의): 제1급=생물테러·치명·음압격리·즉시신고 / 제2급=전파력 높음·격리·24시간내 신고 / "
        "제3급=발생 모니터링·24시간내 신고 / 제4급=유행여부 조사 위한 표본감시(7일 이내 신고).",
        "인플루엔자: 제4급 → 전국 의원급 표본감시기관이 인플루엔자 의사환자분율(ILI, /1,000명)을 "
        "주간 신고(KDCA 절기 표본감시). 본 프로젝트의 예측 표적이 곧 이 ILI 표본감시 지표.",
        "제11조(의사 등의 신고): 의사·치과의사·한의사·의료기관의 장 신고 의무. "
        "제3·4급 미신고/거짓신고 시 300만원 이하 벌금.",
        "제18조(역학조사): 질병관리청장·시·도지사·시장군수구청장이 실시(조사 거부·방해 처벌).",
        "제45조(업무종사 일시 제한), 제47조·제49조(감염병 유행 방역조치): 집합 제한·금지, 마스크 착용 "
        "명령, 흥행·집회·제례·휴업·휴교·휴원 제한, 감염병의심자 격리·입원 — 자치단체장 권한.",
        "제24조(필수예방접종)·제25조(임시예방접종): 인플루엔자는 65세 이상 어르신·어린이 등 "
        "국가예방접종지원(NIP) 대상.",
    ],
    "sources": [
        "law.go.kr 「감염병의 예방 및 관리에 관한 법률」",
        "질병관리청 dportal.kdca.go.kr — 법정감염병",
        "찾기쉬운 생활법령정보 easylaw.go.kr — 감염병 신고",
    ],
    "caveat": "조문 번호·정의는 요지. 정확한 현행 조문과 감염병 종수는 개정으로 변동 → law.go.kr 확인.",
}


def _kdca_data(con) -> dict:
    out: dict = {}
    # project DB coverage by legal grade (NOT the full legal species count)
    try:
        rows = con.execute(
            "SELECT disease_group, COUNT(*) FROM disease_catalog "
            "WHERE disease_group LIKE '제%급' GROUP BY disease_group").fetchall()
        out["db_coverage_by_grade"] = {g: n for g, n in rows}
    except Exception:
        pass
    # influenza legal/clinical row from disease_master
    try:
        r = con.execute(
            "SELECT legal_grade, transmission_route, vaccine_available, flags "
            "FROM disease_master WHERE disease_nm = '인플루엔자'").fetchone()
        if r:
            out["influenza"] = {
                "legal_grade": r[0], "transmission": r[1],
                "vaccine_available": bool(r[2]), "flags": r[3],
            }
    except Exception:
        pass
    # most recent KDCA notifiable national report count, by grade
    try:
        yr = con.execute(
            "SELECT MAX(year) FROM kosis_kdca_notifiable WHERE sido='계'").fetchone()[0]
        rows = con.execute(
            "SELECT disease, SUM(cases) FROM kosis_kdca_notifiable "
            "WHERE sido='계' AND year=? AND disease LIKE '제%급' GROUP BY disease",
            (yr,)).fetchall()
        out["notifiable_national"] = {"year": yr, "by_grade": {d: int(c or 0) for d, c in rows}}
    except Exception:
        pass
    # latest sentinel ILI (전체 연령 if present, else any)
    try:
        r = con.execute(
            "SELECT season_start, week_label, age_group, ili_rate FROM sentinel_influenza "
            "ORDER BY season_start DESC, week_seq DESC LIMIT 1").fetchone()
        if r:
            out["sentinel_ili_latest"] = {
                "season": r[0], "week": r[1], "age_group": r[2], "ili_rate": r[3],
                "unit": "의사환자 /1,000명 (KDCA 표본감시)",
            }
    except Exception:
        pass
    return out


def build() -> dict:
    from simulation.database import read_only_connect
    con = read_only_connect(str(DB))
    try:
        kdca = _kdca_data(con)
    finally:
        con.close()
    return {"law": LAW, "kdca_data": kdca}


def main() -> int:
    gj = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(gj, ensure_ascii=False, indent=2), encoding="utf-8")
    k = gj["kdca_data"]
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  influenza: {k.get('influenza')}")
    print(f"  notifiable: {k.get('notifiable_national')}")
    print(f"  sentinel ILI: {k.get('sentinel_ili_latest')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
