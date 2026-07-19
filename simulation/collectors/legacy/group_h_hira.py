"""
pipeline/collectors/group_h_hira.py
=====================================
Group H - HIRA 건강보험심사평가원_질병정보서비스 (data.go.kr #15119055)

[목적]
  전수감시(1~3급) 분모 보정을 위한 환자인구 기반 데이터 수집.
  CIR = 신고건수/주민등록인구 (신고율) → HIRA 보정 CIR = 신고건수/HIRA 진료실인원

[데이터 출처]
  data.go.kr REST API (건강보험심사평가원_질병정보서비스)
  End Point: https://apis.data.go.kr/B551182/diseaseInfoService1

[5개 API 엔드포인트]
  1. getDissNameCodeList1      -- 질병 명칭/코드 조회
  2. getDissByHsptlzFrgnStats1 -- 질병 입원외래별 통계
  3. getDissByGenderAgeStats1  -- 질병 성별연령별 통계
  4. getDissByClassesStats1    -- 질병 의료기관종별 통계
  5. getDissByAreaStats1       -- 질병 의료기관 지역별 통계

[공통 파라미터] (엔드포인트 2~5)
  serviceKey : 인증키 (data.go.kr 발급)
  numOfRows  : 한 페이지 결과 수 (기본 10, 최대 1000)
  pageNo     : 페이지 번호
  year       : 조회 연도 (예: 2022)
  sickCd     : KCD 상병코드 (예: B019)
  sickType   : 상병 구분 (1: 3단 상병, 2: 4단 상병)
  medTp      : 양방/한방 (1: 양방, 2: 한방)

[데이터 포맷]
  XML only (JSON 미지원)

[DB 테이블]
  hira_inpat_opat   : 입원외래별 (엔드포인트 2)
  hira_gender_age   : 성별연령별 (엔드포인트 3) ← 가장 중요
  hira_facility     : 의료기관종별 (엔드포인트 4)
  hira_region       : 지역별 (엔드포인트 5)

[사용법]
  from pipeline.collectors.group_h_hira import GroupHCollector
  c = GroupHCollector()
  c.run()                          # 전체 수집 (기본 KCD 코드, 2020-2024)
  c.run(kcd_codes=["B019","A37"])  # 특정 질환만
"""

import time
import logging
import xml.etree.ElementTree as ET
from datetime import datetime

from .base import BaseCollector
from ..config import KEYS, DB_PATH, TIMEOUT
from ..storage import log_collection
# : sqlite3.connect → safe_connect (PRAGMA tuning + quick_check).
from simulation.database import safe_connect

log = logging.getLogger(__name__)

# ── HIRA API 설정 ────────────────────────────────────────────────────────────
HIRA_BASE = "https://apis.data.go.kr/B551182/diseaseInfoService1"

# 주요 감염병 KCD 코드 (4단 상병 sickType=2 기준)
# 3단 코드(B01)로 조회 시 sickType=1, 4단(B019)으로 조회 시 sickType=2
INFECTIOUS_KCD = {
    # ── 호흡기/비말 전파 (14 codes) ──
    "B019":  "수두(합병증없음)",
    "A379":  "백일해(상세불명)",
    "A38":   "성홍열",
    "B059":  "홍역(합병증없음)",
    "B069":  "풍진(합병증없음)",
    "B269":  "유행성이하선염(합병증없음)",
    "J09":   "인플루엔자(신종)",
    "J100":  "인플루엔자(확인,폐렴동반)",
    "J101":  "인플루엔자(확인,기타호흡기)",
    "J108":  "인플루엔자(확인,기타)",
    "J110":  "인플루엔자(미확인,폐렴동반)",
    "J111":  "인플루엔자(미확인,기타호흡기)",
    "J118":  "인플루엔자(미확인,기타)",
    "A36":   "디프테리아",
    "A481":  "레지오넬라증",              # 추가
    "A403":  "폐렴구균패혈증",            # 추가
    "J13":   "폐렴구균폐렴",              # 추가
    "A39":   "수막구균감염",              # 추가
    # ── 경구/식품매개 (8 codes) ──
    "A00":   "콜레라",
    "A010":  "장티푸스",
    "B150":  "A형간염(간성혼수동반)",
    "B159":  "A형간염(간성혼수미동반)",
    "B172":  "E형간염",                   # 추가
    "A03":   "세균성이질(시겔라)",         # 추가
    "A040":  "장병원성대장균감염",         # 추가
    "A050":  "비브리오식중독",            # 추가
    # ── 혈액/체액 (5 codes) ──
    "B171":  "급성C형간염",
    "B182":  "만성C형간염",
    "B160":  "급성B형간염(D형동반)",       # 추가
    "B169":  "급성B형간염(D형미동반)",     # 추가
    "A51":   "조기매독",                  # 추가
    # ── 벡터매개 (8 codes) ──
    "A753":  "쯔쯔가무시병",
    "A938":  "기타모기매개바이러스열",  # SFTS proxy
    "B51":   "삼일열말라리아",            # 추가
    "B54":   "상세불명말라리아",          # 추가
    "A90":   "뎅기열",                    # 추가
    "A830":  "일본뇌염",                  # 추가
    "A279":  "렙토스피라증(상세불명)",     # 추가
    "A78":   "큐열",                      # 추가
    # ── 동물/환경 (3 codes) ──
    "A985":  "출혈열신증후군(HFRS)",       # 추가
    "A35":   "기타파상풍",                # 추가
    "A23":   "브루셀라증",                # 추가
}

# KCD 코드의 sickType 결정: 3자리=1(3단), 4자리+=2(4단)
def _sick_type(kcd: str) -> str:
    """KCD 코드 길이에 따라 sickType 반환."""
    # A38=3단, B019=4단, B17.1→B171=4단
    clean = kcd.replace(".", "")
    return "1" if len(clean) <= 3 else "2"


class GroupHCollector(BaseCollector):
    """HIRA 질병정보서비스 수집기 (data.go.kr #15119055).

    API 키: pipeline/config.py KEYS['data_go_kr']
    (기존 KDCA API와 동일한 data.go.kr 인증키 사용)
    """

    def __init__(self):
        super().__init__()
        self._tables_created = False

    # ── DB 테이블 생성 ────────────────────────────────────────────────────

    def _ensure_tables(self):
        """4개 HIRA 통계 테이블 생성."""
        if self._tables_created:
            return
        conn = safe_connect(str(DB_PATH))

        # 입원외래별
        conn.execute("""CREATE TABLE IF NOT EXISTS hira_inpat_opat (
            kcd_code TEXT NOT NULL,
            kcd_name TEXT,
            ref_year INTEGER NOT NULL,
            sex TEXT,
            inpat_opat TEXT,
            patient_count INTEGER,
            spec_count INTEGER,
            visit_days INTEGER,
            insup_brdn_amt INTEGER,
            rpe_tamt_amt INTEGER,
            collected_at TEXT,
            UNIQUE(kcd_code, ref_year, sex, inpat_opat)
        )""")

        # 성별연령별
        conn.execute("""CREATE TABLE IF NOT EXISTS hira_gender_age (
            kcd_code TEXT NOT NULL,
            kcd_name TEXT,
            ref_year INTEGER NOT NULL,
            sex TEXT,
            age_group TEXT,
            patient_count INTEGER,
            spec_count INTEGER,
            visit_days INTEGER,
            insup_brdn_amt INTEGER,
            rpe_tamt_amt INTEGER,
            collected_at TEXT,
            UNIQUE(kcd_code, ref_year, sex, age_group)
        )""")

        # 의료기관종별
        conn.execute("""CREATE TABLE IF NOT EXISTS hira_facility (
            kcd_code TEXT NOT NULL,
            kcd_name TEXT,
            ref_year INTEGER NOT NULL,
            facility_type TEXT,
            patient_count INTEGER,
            spec_count INTEGER,
            visit_days INTEGER,
            insup_brdn_amt INTEGER,
            rpe_tamt_amt INTEGER,
            collected_at TEXT,
            UNIQUE(kcd_code, ref_year, facility_type)
        )""")

        # 지역별
        conn.execute("""CREATE TABLE IF NOT EXISTS hira_region (
            kcd_code TEXT NOT NULL,
            kcd_name TEXT,
            ref_year INTEGER NOT NULL,
            region TEXT,
            patient_count INTEGER,
            spec_count INTEGER,
            visit_days INTEGER,
            insup_brdn_amt INTEGER,
            rpe_tamt_amt INTEGER,
            collected_at TEXT,
            UNIQUE(kcd_code, ref_year, region)
        )""")

        conn.commit()
        conn.close()
        self._tables_created = True

    # ── XML 파싱 ──────────────────────────────────────────────────────────

    def _get_xml(self, endpoint: str, params: dict) -> list[dict]:
        """HIRA API GET → XML 파싱 → list of item dicts."""
        url = f"{HIRA_BASE}/{endpoint}"
        resp = self.get(url, params=params, expect_json=False, timeout=TIMEOUT)
        if not resp:
            return []

        try:
            root = ET.fromstring(resp)
        except ET.ParseError as e:
            log.error(f"  [H] XML parse error: {e}")
            return []

        # 에러 체크
        code = root.findtext(".//resultCode")
        msg = root.findtext(".//resultMsg")
        if code != "00":
            log.warning(f"  [H] API error: {code} {msg}")
            return []

        total = int(root.findtext(".//totalCount") or "0")
        if total == 0:
            return []

        items = []
        for item_el in root.findall(".//item"):
            d = {}
            for child in item_el:
                d[child.tag] = child.text
            items.append(d)

        return items

    def _common_params(self, year: int, kcd: str, key: str) -> dict:
        """통계 엔드포인트 공통 파라미터."""
        return {
            "serviceKey": key,
            "numOfRows": "1000",
            "pageNo": "1",
            "year": str(year),
            "sickCd": kcd,
            "sickType": _sick_type(kcd),
            "medTp": "1",  # 양방
        }

    # ── 엔드포인트별 수집 ──────────────────────────────────────────────────

    def _collect_inpat_opat(self, kcd: str, kcd_name: str,
                             years: range, key: str) -> int:
        """H2: 입원외래별 통계."""
        rows = []
        now = self.now_iso()
        for year in years:
            params = self._common_params(year, kcd, key)
            items = self._get_xml("getDissByHsptlzFrgnStats1", params)
            for it in items:
                rows.append((
                    kcd, kcd_name, year,
                    it.get("sex"), it.get("inpatOpat"),
                    self.safe_int(it.get("ptntCnt")),
                    self.safe_int(it.get("specCnt")),
                    self.safe_int(it.get("vstDdcnt")),
                    self.safe_int(it.get("rvdInsupBrdnAmt")),
                    self.safe_int(it.get("rvdRpeTamtAmt")),
                    now,
                ))
            time.sleep(0.3)

        if not rows:
            return 0
        conn = safe_connect(str(DB_PATH))
        conn.executemany(
            """INSERT OR REPLACE INTO hira_inpat_opat
               (kcd_code,kcd_name,ref_year,sex,inpat_opat,
                patient_count,spec_count,visit_days,
                insup_brdn_amt,rpe_tamt_amt,collected_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            rows)
        conn.commit()
        conn.close()
        return len(rows)

    def _collect_gender_age(self, kcd: str, kcd_name: str,
                             years: range, key: str) -> int:
        """H3: 성별연령별 통계 (가장 중요 -- 연령별 발생률 산출용)."""
        rows = []
        now = self.now_iso()
        for year in years:
            params = self._common_params(year, kcd, key)
            items = self._get_xml("getDissByGenderAgeStats1", params)
            for it in items:
                rows.append((
                    kcd, kcd_name, year,
                    it.get("sex"), it.get("age"),
                    self.safe_int(it.get("ptntCnt")),
                    self.safe_int(it.get("specCnt")),
                    self.safe_int(it.get("vstDdcnt")),
                    self.safe_int(it.get("rvdInsupBrdnAmt")),
                    self.safe_int(it.get("rvdRpeTamtAmt")),
                    now,
                ))
            time.sleep(0.3)

        if not rows:
            return 0
        conn = safe_connect(str(DB_PATH))
        conn.executemany(
            """INSERT OR REPLACE INTO hira_gender_age
               (kcd_code,kcd_name,ref_year,sex,age_group,
                patient_count,spec_count,visit_days,
                insup_brdn_amt,rpe_tamt_amt,collected_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            rows)
        conn.commit()
        conn.close()
        return len(rows)

    def _collect_facility(self, kcd: str, kcd_name: str,
                           years: range, key: str) -> int:
        """H4: 의료기관종별 통계."""
        rows = []
        now = self.now_iso()
        for year in years:
            params = self._common_params(year, kcd, key)
            items = self._get_xml("getDissByClassesStats1", params)
            for it in items:
                ftype = it.get("grade", "")
                rows.append((
                    kcd, kcd_name, year, ftype,
                    self.safe_int(it.get("ptntCnt")),
                    self.safe_int(it.get("specCnt")),
                    self.safe_int(it.get("vstDdcnt")),
                    self.safe_int(it.get("rvdInsupBrdnAmt")),
                    self.safe_int(it.get("rvdRpeTamtAmt")),
                    now,
                ))
            time.sleep(0.3)

        if not rows:
            return 0
        conn = safe_connect(str(DB_PATH))
        conn.executemany(
            """INSERT OR REPLACE INTO hira_facility
               (kcd_code,kcd_name,ref_year,facility_type,
                patient_count,spec_count,visit_days,
                insup_brdn_amt,rpe_tamt_amt,collected_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            rows)
        conn.commit()
        conn.close()
        return len(rows)

    def _collect_region(self, kcd: str, kcd_name: str,
                         years: range, key: str) -> int:
        """H5: 의료기관지역별 통계 (시도별)."""
        rows = []
        now = self.now_iso()
        for year in years:
            params = self._common_params(year, kcd, key)
            items = self._get_xml("getDissByAreaStats1", params)
            for it in items:
                region = it.get("lcName", "")
                rows.append((
                    kcd, kcd_name, year, region,
                    self.safe_int(it.get("ptntCnt")),
                    self.safe_int(it.get("specCnt")),
                    self.safe_int(it.get("vstDdcnt")),
                    self.safe_int(it.get("rvdInsupBrdnAmt")),
                    self.safe_int(it.get("rvdRpeTamtAmt")),
                    now,
                ))
            time.sleep(0.3)

        if not rows:
            return 0
        conn = safe_connect(str(DB_PATH))
        conn.executemany(
            """INSERT OR REPLACE INTO hira_region
               (kcd_code,kcd_name,ref_year,region,
                patient_count,spec_count,visit_days,
                insup_brdn_amt,rpe_tamt_amt,collected_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            rows)
        conn.commit()
        conn.close()
        return len(rows)

    # ── 메인 실행 ──────────────────────────────────────────────────────────

    def run(self, kcd_codes: list = None, skip_apis: list = None,
            start_year: int = 2020, end_year: int = 2024) -> dict:
        """Group H 전체 실행.

        Parameters:
            kcd_codes: 수집할 KCD 코드 리스트 (기본: INFECTIOUS_KCD 전체)
            skip_apis: 건너뛸 서브태스크 리스트 (예: ["H2","H4"])
            start_year: 시작 연도
            end_year: 종료 연도

        Returns:
            dict: 테이블별 적재 행 수
        """
        skip_apis = skip_apis or []
        log.info(">> Group H - HIRA 질병정보서비스 (data.go.kr)")

        # API 키 확인 (data.go.kr 공용 키 사용)
        key = KEYS.get("data_go_kr", "")
        if not key:
            log.warning("  [H] data.go.kr API key not configured in KEYS['data_go_kr']")
            log_collection("H", "HIRA_all", "SKIP", note="API key not configured")
            return {"hira_inpat_opat": 0, "hira_gender_age": 0,
                    "hira_facility": 0, "hira_region": 0}

        t0 = time.time()
        self._ensure_tables()

        codes = kcd_codes or list(INFECTIOUS_KCD.keys())
        years = range(start_year, end_year + 1)
        r2 = r3 = r4 = r5 = 0

        for kcd in codes:
            kcd_name = INFECTIOUS_KCD.get(kcd, kcd)
            log.info(f"  [H] {kcd} ({kcd_name}) ...")

            if "H2" not in skip_apis:
                try:
                    n = self._collect_inpat_opat(kcd, kcd_name, years, key)
                    r2 += n
                    log.info(f"    H2 입원외래별: {n} rows")
                except Exception as e:
                    log.error(f"    H2 error: {e}")

            if "H3" not in skip_apis:
                try:
                    n = self._collect_gender_age(kcd, kcd_name, years, key)
                    r3 += n
                    log.info(f"    H3 성별연령별: {n} rows")
                except Exception as e:
                    log.error(f"    H3 error: {e}")

            if "H4" not in skip_apis:
                try:
                    n = self._collect_facility(kcd, kcd_name, years, key)
                    r4 += n
                    log.info(f"    H4 기관종별: {n} rows")
                except Exception as e:
                    log.error(f"    H4 error: {e}")

            if "H5" not in skip_apis:
                try:
                    n = self._collect_region(kcd, kcd_name, years, key)
                    r5 += n
                    log.info(f"    H5 지역별: {n} rows")
                except Exception as e:
                    log.error(f"    H5 error: {e}")

        total = r2 + r3 + r4 + r5
        elapsed = round(time.time() - t0, 1)
        log_collection("H", "HIRA_all",
                       "OK" if total > 0 else "FAIL",
                       total, elapsed=elapsed)
        log.info(f"  [H] DONE: {total} rows total ({elapsed}s)")
        log.info(f"    H2={r2}, H3={r3}, H4={r4}, H5={r5}")

        return {
            "hira_inpat_opat": r2,
            "hira_gender_age": r3,
            "hira_facility": r4,
            "hira_region": r5,
        }
