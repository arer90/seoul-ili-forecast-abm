"""
pipeline/collectors/group_s_sentinel.py
========================================
질병관리청 감염병포털 표본감시감염병 수집기

[수집 대상]  BASE_URL = https://dportal.kdca.go.kr
─────────────────────────────────────────────────────
 S1  sentinel_influenza    인플루엔자 ILI 의사환자율 (시즌별·연령별·주별)
 S2  sentinel_ari          급성호흡기감염증 병원체별 (RSV, 리노, 아데노 등) 주별
 S3  sentinel_sari         중증급성호흡기감염증 주별
 S4  sentinel_hfmd         수족구병 주별
 S5  sentinel_enterovirus  엔테로바이러스 주별
 S6  sentinel_intestinal   장관감염증 병원체별 주별
 S7  sentinel_ophlgc       안과감염병 (유행성각결막염·급성출혈결막염) 주별 의사환자율
 S8  sentinel_hfmdc        합병증동반수족구병 연도별 신고수

[실행]
  uv run python -m pipeline.run_pipeline --group S
  uv run python -m pipeline.run_pipeline --group S --s-year 2020 2026
"""

import logging
import time
from datetime import datetime
from typing import Optional

import requests

from ..config import DB_PATH, TIMEOUT
from simulation.database.config import redact_secrets
from ..storage import get_conn as get_connection

log = logging.getLogger(__name__)

BASE_URL = "https://dportal.kdca.go.kr"
HEADERS = {
    "Content-Type": "application/x-www-form-urlencoded",
    "Referer": "https://dportal.kdca.go.kr/pot/is/st/influ.do",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
}
# 요청 간격 (서버 부하 방지)
REQUEST_DELAY = 0.8

# ── ARI 병원체 컬럼 매핑 (captionList 순서 기준) ──────────────────────────────
ARI_PATHOGENS = [
    ("계", "계"),
    ("세균", "마이코플라즈마균"),
    ("세균", "클라미디아균"),
    ("바이러스", "아데노바이러스"),
    ("바이러스", "사람 보카바이러스"),
    ("바이러스", "파라인플루엔자바이러스"),
    ("바이러스", "호흡기세포융합바이러스(RSV)"),
    ("바이러스", "리노바이러스"),
    ("바이러스", "사람 메타뉴모바이러스"),
    ("바이러스", "사람 코로나바이러스(계절)"),
    ("인플루엔자", "인플루엔자 바이러스"),
    ("코로나19", "코로나19 바이러스"),
]

# ── 장관감염증 병원체 컬럼 매핑 ──────────────────────────────────────────────
INTESTINAL_PATHOGENS = [
    ("계", "계"),
    ("세균", "살모넬라균"),
    ("세균", "장염비브리오균"),
    ("세균", "장독소성대장균(ETEC)"),
    ("세균", "장침습성대장균(EIEC)"),
    ("세균", "장병원성대장균(EPEC)"),
    ("세균", "캄필로박터균"),
    ("세균", "클로스트리듐 퍼프린젠스"),
    ("세균", "황색포도알균"),
    ("세균", "바실루스 세레우스균"),
    ("세균", "예르시니아 엔테로콜리티카"),
    ("세균", "리스테리아 모노사이토제네스"),
    ("바이러스", "그룹 A형 로타바이러스"),
    ("바이러스", "아스트로바이러스"),
    ("바이러스", "장내 아데노바이러스"),
    ("바이러스", "노로바이러스"),
    ("바이러스", "사포바이러스"),
    ("원충", "이질아메바"),
    ("원충", "람블편모충"),
    ("원충", "작은와포자충"),
    ("원충", "원포자충"),
]


# ─────────────────────────────────────────────────────────────────────────────
# 공통 유틸
# ─────────────────────────────────────────────────────────────────────────────
def _post(endpoint: str, params: dict, retries: int = 3) -> Optional[dict]:
    """POST 요청 + 재시도"""
    url = BASE_URL + endpoint
    for attempt in range(retries):
        try:
            resp = requests.post(url, data=params, headers=HEADERS, timeout=TIMEOUT)
            resp.raise_for_status()
            j = resp.json()
            if j.get("result"):
                return j.get("value", {})
            log.warning(
                f"  API result=false: {redact_secrets(url)} "
                f"params={redact_secrets(params)}"
            )
            return None
        except Exception as e:
            log.warning(f"  요청 실패 ({attempt+1}/{retries}): {redact_secrets(e)}")
            if attempt < retries - 1:
                time.sleep(2)
    return None


def _col(row: dict, idx: int) -> Optional[float]:
    """COLUMN{idx} 값을 float으로 변환 (빈 값 → None)"""
    v = row.get(f"COLUMN{idx}", "")
    if v is None or str(v).strip() in ("", "-", "--"):
        return None
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return None


def _now() -> str:
    return datetime.now().isoformat()


# ─────────────────────────────────────────────────────────────────────────────
# S1  인플루엔자 (ILI 의사환자율, 시즌별·연령별)
# ─────────────────────────────────────────────────────────────────────────────
def collect_influenza(start_season: int = 2019, end_season: int = 2025) -> int:
    """
    인플루엔자 ILI 표본감시 수집.
    한 시즌 = startYear 의 36주 ~ endYear 의 35주.
    start_season=2019 → 2019-2020 시즌부터 수집.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sentinel_influenza (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            collected_at TEXT,
            season_start INTEGER,   -- 시즌 시작 연도 (2019 = 2019-2020 시즌)
            week_seq     INTEGER,   -- 1~53 (captionList 순서)
            week_label   TEXT,      -- "36주", "37주", ..., "35주"
            age_group    TEXT,      -- "0세", "1-6세", ..., "65세 이상"
            ili_rate     REAL,      -- 의사환자분율 (외래환자 1000명당)
            UNIQUE(season_start, week_seq, age_group)
        )
    """)
    conn.commit()

    total_inserted = 0
    now = _now()

    for sy in range(start_season, end_season + 1):
        ey = sy + 1
        log.info(f"  [S1 인플루엔자] {sy}-{ey} 시즌 수집 중...")
        val = _post("/pot/is/st/influListAjax.do", {
            "startYear": str(sy), "endYear": str(ey),
            "age": "", "intoDivi": "1"
        })
        if not val:
            continue

        captions = val.get("captionList", [])  # ["36주","37주",...]
        rows = val.get("data", [])

        inserted = 0
        for row in rows:
            age_group = row.get("TITLE", "").strip()
            if not age_group:
                continue
            for seq, week_label in enumerate(captions, start=1):
                rate = _col(row, seq)
                if rate is None:
                    continue
                cur.execute("""
                    INSERT OR REPLACE INTO sentinel_influenza
                    (collected_at, season_start, week_seq, week_label, age_group, ili_rate)
                    VALUES (?,?,?,?,?,?)
                """, (now, sy, seq, week_label, age_group, rate))
                inserted += 1

        conn.commit()
        log.info(f"    → {inserted}건 저장")
        total_inserted += inserted
        time.sleep(REQUEST_DELAY)

    conn.close()
    return total_inserted


# ─────────────────────────────────────────────────────────────────────────────
# S2  급성호흡기감염증 (RSV 포함, 주별·병원체별)
# ─────────────────────────────────────────────────────────────────────────────
def collect_ari(start_year: int = 2020, end_year: int = 2026) -> int:
    """
    급성호흡기감염증 병원체 표본감시.
    RSV(호흡기세포융합바이러스), 리노, 아데노, 메타뉴모,
    인플루엔자, 코로나19 등 12개 병원체 주별 수집.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sentinel_ari (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            collected_at    TEXT,
            year            INTEGER,
            week_no         INTEGER,
            pathogen_group  TEXT,   -- 세균/바이러스/인플루엔자/코로나19/계
            pathogen_nm     TEXT,   -- RSV, 리노바이러스 등
            count           INTEGER,
            UNIQUE(year, week_no, pathogen_nm)
        )
    """)
    conn.commit()

    total_inserted = 0
    now = _now()

    # 연도를 청크로 나눠서 요청 (서버 부하 줄이기)
    chunk_size = 2
    for yr in range(start_year, end_year + 1, chunk_size):
        ey = min(yr + chunk_size - 1, end_year)
        log.info(f"  [S2 급성호흡기] {yr}-{ey}년 수집 중...")
        val = _post("/pot/is/st/ariListAjax.do", {
            "startYear": str(yr), "startWeek": "01",
            "endYear": str(ey), "endWeek": "52",
            "dayCheck": "1", "infectiousGubun": "",
            "subInfectious": "", "age": ""
        })
        if not val:
            continue

        rows = val.get("data", [])
        inserted = 0
        for row in rows:
            year_str = str(row.get("TITLE", "")).strip()
            week_str = str(row.get("SUBTITLE", "")).strip()
            if not year_str.isdigit() or not week_str.isdigit():
                continue
            year_int = int(year_str)
            week_int = int(week_str)

            for col_idx, (p_group, p_nm) in enumerate(ARI_PATHOGENS, start=1):
                cnt = _col(row, col_idx)
                if cnt is None:
                    continue
                cur.execute("""
                    INSERT OR REPLACE INTO sentinel_ari
                    (collected_at, year, week_no, pathogen_group, pathogen_nm, count)
                    VALUES (?,?,?,?,?,?)
                """, (now, year_int, week_int, p_group, p_nm, int(cnt)))
                inserted += 1

        conn.commit()
        log.info(f"    → {inserted}건 저장 ({yr}-{ey})")
        total_inserted += inserted
        time.sleep(REQUEST_DELAY)

    conn.close()
    return total_inserted


# ─────────────────────────────────────────────────────────────────────────────
# S3  중증급성호흡기감염증 (SARI)
# ─────────────────────────────────────────────────────────────────────────────
def collect_sari(start_year: int = 2020, end_year: int = 2026) -> int:
    """중증급성호흡기감염증 주별 입원 환자 수"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sentinel_sari (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            collected_at TEXT,
            year         INTEGER,
            week_no      INTEGER,
            week_label   TEXT,
            count        INTEGER,
            UNIQUE(year, week_no)
        )
    """)
    conn.commit()

    total_inserted = 0
    now = _now()

    log.info(f"  [S3 중증호흡기] {start_year}-{end_year}년 수집 중...")
    val = _post("/pot/is/st/sariListAjax.do", {
        "startYear": str(start_year), "startWeek": "01",
        "endYear": str(end_year), "endWeek": "52",
        "dayCheck": "1", "age": ""
    })
    if val:
        captions = val.get("captionList", [])   # ["2020 01", "2020 02", ...]
        rows = val.get("data", [])
        # SARI는 전체 1행, COLUMN1~N 이 각 주 데이터
        for row in rows:
            for col_idx, cap in enumerate(captions, start=1):
                parts = cap.strip().split()
                if len(parts) < 2:
                    continue
                year_str, week_str = parts[0], parts[1]
                if not year_str.isdigit() or not week_str.isdigit():
                    continue
                cnt = _col(row, col_idx)
                if cnt is None:
                    continue
                cur.execute("""
                    INSERT OR REPLACE INTO sentinel_sari
                    (collected_at, year, week_no, week_label, count)
                    VALUES (?,?,?,?,?)
                """, (now, int(year_str), int(week_str), cap, int(cnt)))
                total_inserted += 1

        conn.commit()
        log.info(f"    → {total_inserted}건 저장")

    conn.close()
    return total_inserted


# ─────────────────────────────────────────────────────────────────────────────
# S4  수족구병 (HFMD)
# ─────────────────────────────────────────────────────────────────────────────
def collect_hfmd(start_year: int = 2020, end_year: int = 2026) -> int:
    """수족구병 주별 의사환자율"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sentinel_hfmd (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            collected_at TEXT,
            year         INTEGER,
            week_no      INTEGER,
            week_label   TEXT,
            rate         REAL,      -- 외래환자 1000명당 의사환자율
            UNIQUE(year, week_no)
        )
    """)
    conn.commit()

    total_inserted = 0
    now = _now()

    log.info(f"  [S4 수족구병] {start_year}-{end_year}년 수집 중...")
    val = _post("/pot/is/st/hfmdListAjax.do", {
        "startYear": str(start_year), "startWeek": "01",
        "endYear": str(end_year), "endWeek": "52",
        "dayCheck": "1", "age": ""
    })
    if val:
        captions = val.get("captionList", [])   # ["01주","02주",...,"52주"]
        rows = val.get("data", [])
        for row in rows:
            title = row.get("TITLE", "")
            # TITLE = "2024년" 형태
            year_str = str(title).replace("년", "").strip()
            if not year_str.isdigit():
                continue
            year_int = int(year_str)
            for col_idx, week_label in enumerate(captions, start=1):
                rate = _col(row, col_idx)
                if rate is None:
                    continue
                week_no = col_idx
                cur.execute("""
                    INSERT OR REPLACE INTO sentinel_hfmd
                    (collected_at, year, week_no, week_label, rate)
                    VALUES (?,?,?,?,?)
                """, (now, year_int, week_no, week_label, rate))
                total_inserted += 1

        conn.commit()
        log.info(f"    → {total_inserted}건 저장")

    conn.close()
    return total_inserted


# ─────────────────────────────────────────────────────────────────────────────
# S5  엔테로바이러스
# ─────────────────────────────────────────────────────────────────────────────
def collect_enterovirus(start_year: int = 2020, end_year: int = 2026) -> int:
    """엔테로바이러스 주별 환자 수"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sentinel_enterovirus (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            collected_at TEXT,
            year         INTEGER,
            week_no      INTEGER,
            count        INTEGER,
            UNIQUE(year, week_no)
        )
    """)
    conn.commit()

    total_inserted = 0
    now = _now()

    log.info(f"  [S5 엔테로바이러스] {start_year}-{end_year}년 수집 중...")
    val = _post("/pot/is/st/etrvnftnListAjax.do", {
        "startYear": str(start_year), "startWeek": "01",
        "endYear": str(end_year), "endWeek": "52",
        "dayCheck": "1", "age": ""
    })
    if val:
        rows = val.get("data", [])
        # TITLE=year, SUBTITLE=week_no, COLUMN1=총계
        for row in rows:
            year_str = str(row.get("TITLE", "")).strip()
            week_str = str(row.get("SUBTITLE", "")).strip()
            if not year_str.isdigit() or not week_str.isdigit():
                continue
            cnt = _col(row, 1)
            if cnt is None:
                continue
            cur.execute("""
                INSERT OR REPLACE INTO sentinel_enterovirus
                (collected_at, year, week_no, count)
                VALUES (?,?,?,?)
            """, (now, int(year_str), int(week_str), int(cnt)))
            total_inserted += 1

        conn.commit()
        log.info(f"    → {total_inserted}건 저장")

    conn.close()
    return total_inserted


# ─────────────────────────────────────────────────────────────────────────────
# S6  장관감염증 (노로바이러스, 로타바이러스, 살모넬라 등)
# ─────────────────────────────────────────────────────────────────────────────
def collect_intestinal(start_year: int = 2020, end_year: int = 2026) -> int:
    """장관감염증 병원체별 주별 수집"""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sentinel_intestinal (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            collected_at    TEXT,
            year            INTEGER,
            week_no         INTEGER,
            pathogen_group  TEXT,
            pathogen_nm     TEXT,
            count           INTEGER,
            UNIQUE(year, week_no, pathogen_nm)
        )
    """)
    conn.commit()

    total_inserted = 0
    now = _now()

    chunk_size = 3
    for yr in range(start_year, end_year + 1, chunk_size):
        ey = min(yr + chunk_size - 1, end_year)
        log.info(f"  [S6 장관감염증] {yr}-{ey}년 수집 중...")
        val = _post("/pot/is/st/gstrnftnListAjax.do", {
            "startYear": str(yr), "startWeek": "01",
            "endYear": str(ey), "endWeek": "52",
            "dayCheck": "1", "age": "",
            "infectiousGubun": "", "subInfectious": ""
        })
        if not val:
            continue

        rows = val.get("data", [])
        inserted = 0
        for row in rows:
            year_str = str(row.get("TITLE", "")).strip()
            week_str = str(row.get("SUBTITLE", "")).strip()
            if not year_str.isdigit() or not week_str.isdigit():
                continue
            year_int = int(year_str)
            week_int = int(week_str)

            for col_idx, (p_group, p_nm) in enumerate(INTESTINAL_PATHOGENS, start=1):
                cnt = _col(row, col_idx)
                if cnt is None:
                    continue
                cur.execute("""
                    INSERT OR REPLACE INTO sentinel_intestinal
                    (collected_at, year, week_no, pathogen_group, pathogen_nm, count)
                    VALUES (?,?,?,?,?,?)
                """, (now, year_int, week_int, p_group, p_nm, int(cnt)))
                inserted += 1

        conn.commit()
        log.info(f"    → {inserted}건 저장 ({yr}-{ey})")
        total_inserted += inserted
        time.sleep(REQUEST_DELAY)

    conn.close()
    return total_inserted


# ─────────────────────────────────────────────────────────────────────────────
# S7  안과감염병 (유행성각결막염·급성출혈결막염)
# ─────────────────────────────────────────────────────────────────────────────

# disName 값 → 질병명 매핑 (0=전체 합산, 1=유행성각결막염, 2=급성출혈결막염)
OPHLGC_DISEASES = [
    ("0", "전체"),
    ("1", "유행성각결막염"),
    ("2", "급성출혈결막염"),
]


def collect_ophlgc(start_year: int = 2013, end_year: int = 2026) -> int:
    """
    안과감염병 주별 의사환자율 수집.
    여름철 대유행하는 유행성각결막염(EKC)·급성출혈결막염(AHC) 포함.
    단위: 외래환자 1000명당 의사환자 수.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sentinel_ophlgc (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            collected_at TEXT,
            year         INTEGER,
            week_no      INTEGER,
            disease_nm   TEXT,   -- 전체 / 유행성각결막염 / 급성출혈결막염
            rate         REAL,   -- 외래환자 1000명당 의사환자율
            UNIQUE(year, week_no, disease_nm)
        )
    """)
    conn.commit()

    total_inserted = 0
    now = _now()

    for dis_code, dis_nm in OPHLGC_DISEASES:
        log.info(f"  [S7 안과감염병] {dis_nm} {start_year}-{end_year}년 수집 중...")
        val = _post("/pot/is/st/ophlgcListAjax.do", {
            "startYear": str(start_year), "startWeek": "01",
            "endYear": str(end_year), "endWeek": "53",
            "age": "", "disName": dis_code, "dayCheck": "1",
        })
        if not val:
            continue

        captions = val.get("captionList", [])   # ["01주","02주",...,"52주"]
        rows = val.get("data", [])

        inserted = 0
        for row in rows:
            # TITLE = "2024년", SUBTITLE = 질병명 or 연령대
            title = str(row.get("TITLE", "")).replace("년", "").strip()
            if not title.isdigit():
                continue
            year_int = int(title)
            for col_idx, week_label in enumerate(captions, start=1):
                rate = _col(row, col_idx)
                if rate is None:
                    continue
                # week_label = "01주" → 1
                week_no = col_idx
                cur.execute("""
                    INSERT OR REPLACE INTO sentinel_ophlgc
                    (collected_at, year, week_no, disease_nm, rate)
                    VALUES (?,?,?,?,?)
                """, (now, year_int, week_no, dis_nm, rate))
                inserted += 1

        conn.commit()
        log.info(f"    → {inserted}건 저장 ({dis_nm})")
        total_inserted += inserted
        time.sleep(REQUEST_DELAY)

    conn.close()
    return total_inserted


# ─────────────────────────────────────────────────────────────────────────────
# S8  합병증동반수족구병 (중증 수족구, 연도별)
# ─────────────────────────────────────────────────────────────────────────────
def collect_hfmdc(start_year: int = 2009, end_year: int = 2026) -> int:
    """
    합병증동반수족구병 연도별 신고 수 수집.
    수족구병 중 신경계 합병증 동반 중증 사례 (사망자 포함).
    수족구병 유행 강도의 심각도 지표로 활용.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS sentinel_hfmdc (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            collected_at TEXT,
            year         INTEGER,
            count        INTEGER,   -- 연간 신고 건수
            UNIQUE(year)
        )
    """)
    conn.commit()

    total_inserted = 0
    now = _now()

    log.info(f"  [S8 합병증수족구] {start_year}-{end_year}년 수집 중...")
    val = _post("/pot/is/st/hfmdcListAjax.do", {
        "startYear": str(start_year),
        "endYear": str(end_year),
        "dayCheck": "3",   # 연도별
    })
    if val:
        rows = val.get("data", [])
        # captionList = ["계"] → COLUMN1 = 연간 건수
        for row in rows:
            title = str(row.get("TITLE", "")).strip()
            # TITLE = "2024" 형태 (연도만)
            if not title.isdigit():
                continue
            cnt = _col(row, 1)
            if cnt is None:
                continue
            cur.execute("""
                INSERT OR REPLACE INTO sentinel_hfmdc
                (collected_at, year, count)
                VALUES (?,?,?)
            """, (now, int(title), int(cnt)))
            total_inserted += 1

        conn.commit()
        log.info(f"    → {total_inserted}건 저장")

    conn.close()
    return total_inserted


# ─────────────────────────────────────────────────────────────────────────────
# 전체 실행
# ─────────────────────────────────────────────────────────────────────────────
def collect_all_sentinel(start_year: int = 2019, end_year: int = 2026) -> dict:
    """표본감시감염병 전체 수집"""
    results = {}
    flu_start = max(2019, start_year - 1)  # 시즌은 전년도 36주부터

    log.info("=" * 55)
    log.info("  표본감시감염병 수집 시작")
    log.info(f"  기간: {start_year}–{end_year}년")
    log.info("=" * 55)

    try:
        results["S1_influenza"]    = collect_influenza(flu_start, end_year)
    except Exception as e:
        log.error(f"S1 오류: {e}"); results["S1_influenza"] = -1
    time.sleep(REQUEST_DELAY)

    try:
        results["S2_ari"]          = collect_ari(start_year, end_year)
    except Exception as e:
        log.error(f"S2 오류: {e}"); results["S2_ari"] = -1
    time.sleep(REQUEST_DELAY)

    try:
        results["S3_sari"]         = collect_sari(start_year, end_year)
    except Exception as e:
        log.error(f"S3 오류: {e}"); results["S3_sari"] = -1
    time.sleep(REQUEST_DELAY)

    try:
        results["S4_hfmd"]         = collect_hfmd(start_year, end_year)
    except Exception as e:
        log.error(f"S4 오류: {e}"); results["S4_hfmd"] = -1
    time.sleep(REQUEST_DELAY)

    try:
        results["S5_enterovirus"]  = collect_enterovirus(start_year, end_year)
    except Exception as e:
        log.error(f"S5 오류: {e}"); results["S5_enterovirus"] = -1
    time.sleep(REQUEST_DELAY)

    try:
        results["S6_intestinal"]   = collect_intestinal(start_year, end_year)
    except Exception as e:
        log.error(f"S6 오류: {e}"); results["S6_intestinal"] = -1
    time.sleep(REQUEST_DELAY)

    try:
        ophlgc_start = max(2013, start_year)
        results["S7_ophlgc"]       = collect_ophlgc(ophlgc_start, end_year)
    except Exception as e:
        log.error(f"S7 오류: {e}"); results["S7_ophlgc"] = -1
    time.sleep(REQUEST_DELAY)

    try:
        hfmdc_start = max(2009, start_year)
        results["S8_hfmdc"]        = collect_hfmdc(hfmdc_start, end_year)
    except Exception as e:
        log.error(f"S8 오류: {e}"); results["S8_hfmdc"] = -1

    total = sum(v for v in results.values() if v >= 0)
    log.info("=" * 55)
    log.info(f"  표본감시감염병 수집 완료 -- 총 {total:,}건")
    for k, v in results.items():
        status = f"{v:,}건" if v >= 0 else "오류"
        log.info(f"    {k}: {status}")
    log.info("=" * 55)
    return results
