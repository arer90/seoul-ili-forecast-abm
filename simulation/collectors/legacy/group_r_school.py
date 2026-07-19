"""
pipeline/collectors/group_r_school.py
======================================
Group R -- 학교 학사일정 데이터 (NEIS)
  R1. NEIS 학교기본정보 (서울 학교 목록)  → school_info_seoul
  R2. NEIS 학사일정 (휴업/방학 일정)      → school_closure_seoul

NEIS 키: KEYS["neis"]
서울 교육청 코드: B10

NEIS API 특성:
  - JSON 응답에서 에러 시 {"RESULT": {"CODE": "...", "MESSAGE": "..."}} 반환
  - 정상 시 {"데이터셋명": [{"head": [...]}, {"row": [...]}]}
  - SchoolSchedule은 SD_SCHUL_CODE 또는 날짜 범위로 조회 가능
"""

import logging
import time
from datetime import datetime
from .base import BaseCollector
from ..config import KEYS
from ..storage import insert_rows, save_csv, log_collection

log = logging.getLogger(__name__)

SEOUL_ATPT_CODE = "B10"  # 서울특별시교육청


class GroupRCollector(BaseCollector):
    """학교 학사일정 데이터"""

    NEIS_KEY = KEYS.get("neis", "")
    NEIS_BASE = "https://open.neis.go.kr/hub"

    def _check_neis_error(self, data: dict, code: str) -> str | None:
        """NEIS API 에러 응답 체크. 에러면 메시지 반환, 정상이면 None."""
        if "RESULT" in data:
            result = data["RESULT"]
            err_code = result.get("CODE", "")
            err_msg = result.get("MESSAGE", "")
            if err_code == "INFO-200":
                log.info(f"  [{code}] 해당 조건의 데이터 없음")
            elif err_code == "ERROR-300":
                log.error(f"  [{code}] NEIS 인증 실패: {err_msg}\n"
                          "  → open.neis.go.kr에서 API 키 발급 확인")
            elif err_code == "ERROR-337":
                log.error(f"  [{code}] 필수 파라미터 누락: {err_msg}")
            else:
                log.warning(f"  [{code}] NEIS 응답: {err_code} - {err_msg}")
            return err_code
        return None

    # ── R1: 서울 학교 기본정보 ───────────────────────────────────────────────
    def collect_school_info(self) -> int:
        """
        NEIS 학교기본정보 → school_info_seoul

        endpoint: /schoolInfo
        서울 모든 초/중/고 학교 목록 + 학교코드 수집
        (R2 학사일정 조회에 학교코드가 필요하므로 먼저 수집)
        """
        t0 = time.time()
        rows = []
        now = self.now_iso()

        url = f"{self.NEIS_BASE}/schoolInfo"

        page = 1
        total_fetched = 0
        while True:
            params = {
                "KEY": self.NEIS_KEY,
                "Type": "json",
                "pIndex": page,
                "pSize": 1000,
                "ATPT_OFCDC_SC_CODE": SEOUL_ATPT_CODE,
            }

            data = self.get(url, params=params, timeout=30)
            if data is None:
                break

            err = self._check_neis_error(data, "R1")
            if err:
                break

            school_info = data.get("schoolInfo", [])
            found_rows = False
            for block in school_info:
                if "row" in block:
                    found_rows = True
                    for item in block["row"]:
                        row = {
                            "collected_at": now,
                            "school_code": item.get("SD_SCHUL_CODE", ""),
                            "school_name": item.get("SCHUL_NM", ""),
                            "school_type": item.get("SCHUL_KND_SC_NM", ""),
                            "gu_name": item.get("ORG_RDNMA", "")[:20],
                            "address": item.get("ORG_RDNMA", ""),
                            "found_date": item.get("FOND_YMD", ""),
                        }
                        rows.append(row)

            if not found_rows:
                break

            total_fetched += len([b for b in school_info if "row" in b])
            page += 1
            time.sleep(0.3)

            # 안전 제한 (서울 학교 ~2000개)
            if page > 5:
                break

        n = insert_rows("school_info_seoul", rows)
        save_csv("school_info_seoul", rows)
        log_collection("R", "school_info", "OK", n,
                       elapsed=time.time() - t0,
                       error=(None if rows else "no rows returned"))
        log.info(f"  [R1] school_info_seoul: {n}건")
        return n

    # ── R2: 학사일정 (휴업/방학) ─────────────────────────────────────────────
    def _load_school_codes(self, sample_per_type: int = 10) -> list[dict]:
        """school_info_seoul에서 학교 유형별 샘플 로드 (SD_SCHUL_CODE 필수)"""
        # : sqlite3.connect → safe_connect (PRAGMA tuning + quick_check).
        from simulation.database import safe_connect
        from ..config import DB_PATH
        try:
            conn = safe_connect(str(DB_PATH))
            cur = conn.cursor()
            # 학교 유형별로 샘플링 (초/중/고 각 10개)
            cur.execute("""
                SELECT school_code, school_name, school_type
                FROM school_info_seoul
                WHERE school_code != ''
                ORDER BY school_type, school_name
            """)
            all_schools = cur.fetchall()
            conn.close()
        except Exception as e:
            log.error(f"  [R2] school_info_seoul 로드 실패: {e}")
            return []

        # 유형별 샘플링
        from collections import defaultdict
        by_type = defaultdict(list)
        for code, name, stype in all_schools:
            by_type[stype].append({"code": code, "name": name, "type": stype})

        sampled = []
        for stype, schools in by_type.items():
            sampled.extend(schools[:sample_per_type])
            log.info(f"  [R2] {stype}: {len(schools)}개 중 {min(len(schools), sample_per_type)}개 샘플")

        return sampled

    def collect_school_schedule(self, year: int = None) -> int:
        """
        NEIS 학사일정 → school_closure_seoul

        endpoint: /SchoolSchedule
        필수 파라미터:
          - ATPT_OFCDC_SC_CODE: B10 (서울)
          - SD_SCHUL_CODE: 행정표준코드 (필수!)
        선택 파라미터:
          - AA_FROM_YMD / AA_TO_YMD: YYYYMMDD
        학교 유형별 샘플(10개씩)로 대표 학사일정 수집.
        """
        t0 = time.time()
        rows = []
        now = self.now_iso()

        if year is None:
            year = datetime.now().year

        url = f"{self.NEIS_BASE}/SchoolSchedule"

        # 학교코드 로드 (유형별 10개 샘플)
        schools = self._load_school_codes(sample_per_type=10)
        if not schools:
            log.error("  [R2] 학교코드 없음 — R1 먼저 실행 필요")
            log_collection("R", "school_schedule", "FAIL_NO_SCHOOLS",
                            0, elapsed=time.time() - t0)
            return 0

        from_ymd = f"{year}0101"
        to_ymd = f"{year}1231"

        for si, school in enumerate(schools):
            school_code = school["code"]
            page = 1
            school_rows = 0

            while True:
                params = {
                    "KEY": self.NEIS_KEY,
                    "Type": "json",
                    "pIndex": page,
                    "pSize": 1000,
                    "ATPT_OFCDC_SC_CODE": SEOUL_ATPT_CODE,
                    "SD_SCHUL_CODE": school_code,
                    "AA_FROM_YMD": from_ymd,
                    "AA_TO_YMD": to_ymd,
                }

                data = self.get(url, params=params, timeout=30)
                if data is None:
                    break

                err = self._check_neis_error(data, "R2")
                if err:
                    break

                schedule = data.get("SchoolSchedule", [])
                found_rows = False
                for block in schedule:
                    if "row" in block:
                        found_rows = True
                        for item in block["row"]:
                            event_name = item.get("EVENT_NM", "")
                            is_closure = any(k in event_name for k in
                                             ["휴업", "휴교", "방학", "재량",
                                              "감염병", "개교기념"])
                            row = {
                                "collected_at": now,
                                "date": item.get("AA_YMD", ""),
                                "school_name": item.get("SCHUL_NM", ""),
                                "school_type": item.get("SCHUL_KND_SC_NM", ""),
                                "event_name": event_name[:200],
                                "is_closure": 1 if is_closure else 0,
                                "event_content": item.get("EVENT_CNTNT", "")[:500],
                            }
                            rows.append(row)
                            school_rows += 1

                if not found_rows:
                    break
                page += 1
                time.sleep(0.2)

                if page > 10:
                    break

            if school_rows > 0 and si < 3:  # 처음 3개만 로그
                log.info(f"  [R2] {school['name']}: {school_rows}건")
            time.sleep(0.2)

        n = insert_rows("school_closure_seoul", rows)
        save_csv("school_closure_seoul", rows)
        log_collection("R", "school_schedule", "OK", n,
                       elapsed=time.time() - t0,
                       error=(None if rows else "no rows returned"))
        log.info(f"  [R2] school_closure_seoul: {n}건 ({year}, {len(schools)}개 학교 샘플)")
        return n

    def run(self, skip_apis: list = None) -> dict:
        skip_apis = skip_apis or []
        log.info("Group R -- 학교 학사일정 수집")
        results = {}

        for key, method, code in [
            ("school_info", self.collect_school_info, "R1"),
            ("school_schedule", self.collect_school_schedule, "R2"),
        ]:
            if code in skip_apis:
                log.info(f"  [{code}] -- skip")
                results[key] = 0
                continue
            try:
                n = method()
                results[key] = n
            except Exception as e:
                log.error(f"  [{code}] {key} failed: {e}")
                results[key] = 0

        total = sum(results.values())
        log.info(f"Group R 완료 -- total {total}: {results}")
        return results
