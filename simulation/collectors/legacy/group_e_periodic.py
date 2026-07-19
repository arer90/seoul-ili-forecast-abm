"""
pipeline/collectors/group_e_periodic.py
=========================================
Group E -- 월간 / 반기 (저빈도 배치)
  E1. KOSIS DT_1ES3C07S  → employment_workplace  (근무지기준, 반기)
  E2. KOSIS DT_1ES3A31S  → employment_residence  (거주지기준, 반기)
  E3. KOSIS DT_1DA7002S  → employment_monthly    (경활인구, 월간)
  E4. NEIS schoolInfo    → school_info            (연간)
  E5. KOSIS DT_1B040A3   → kosis_age_district    (연령별 자치구 인구, 연간)
  E6. HIRA getMedBasisList → hospitals            (의료기관 기본정보)
"""

import re
import time
import logging
from datetime import datetime
from .base import BaseCollector
from ..config import KEYS, SEOUL_BASE
from ..storage import get_conn, insert_rows, save_csv, log_collection

log = logging.getLogger(__name__)

KOSIS_BASE = "https://kosis.kr/openapi/Param/statisticsParameterData.do"


class GroupECollector(BaseCollector):

    # ── 공통: KOSIS 반기 수집 ────────────────────────────────────────────────
    def _collect_kosis_employment(self, tbl_id: str, table_name: str,
                                   start_prd: str, end_prd: str) -> int:
        """
        KOSIS 지역별고용조사 (거주지/근무지) 공통 수집기
        tbl_id: 'DT_1ES3C07S' (근무지) or 'DT_1ES3A31S' (거주지)
        start_prd / end_prd: YYYYHH 형식 (예: '202301', '202502')
        """
        t0 = time.time()
        params = {
            "method":     "getList",
            "apiKey":     KEYS["kosis"],
            "itmId":      "T1",       # 취업자
            "objL1":      "ALL",      # 행정구역 전체
            "objL2":      "ALL",      # 직업 전체
            "format":     "json",
            "jsonVD":     "Y",
            "prdSe":      "H",        # 반기
            "startPrdDe": start_prd,
            "endPrdDe":   end_prd,
            "orgId":      "101",
            "tblId":      tbl_id,
        }
        data = self.get(KOSIS_BASE, params=params)
        if not data:
            log_collection("E", tbl_id, "FAIL", elapsed=time.time()-t0)
            return 0

        rows = []
        try:
            item_list = data if isinstance(data, list) else data.get("items", [])
            for it in item_list:
                # KOSIS 응답: err 필드가 있으면 오류
                if "err" in it:
                    log.error(f"  [E] KOSIS err: {it}")
                    break
                rows.append({
                    "prd_de": it.get("PRD_DE", ""),
                    "c1":     it.get("C1", ""),
                    "c1_nm":  it.get("C1_NM", ""),
                    "c2":     it.get("C2", ""),
                    "c2_nm":  it.get("C2_NM", ""),
                    "dt":     self.safe_float(it.get("DT")),
                })
        except Exception as e:
            log.error(f"  [E] {tbl_id} parse error: {e}")

        n = insert_rows(table_name, rows)
        save_csv(table_name, rows,
                 date_str=f"{start_prd}_{end_prd}", overwrite=True)
        log_collection("E", tbl_id, "OK", n,
                       elapsed=time.time()-t0,
                       error=(None if rows else "no rows returned"))
        log.info(f"  [E] {table_name} ({tbl_id}): {n}건 저장")
        return n

    # ── E1: 근무지기준 지역별고용조사 ─────────────────────────────────────────
    def collect_employment_workplace(self,
                                      start_prd: str = "202301",
                                      end_prd: str = "202502") -> int:
        """DT_1ES3C07S -- ABM 주간 에이전트 배치 핵심"""
        return self._collect_kosis_employment(
            "DT_1ES3C07S", "employment_workplace", start_prd, end_prd)

    # ── E2: 거주지기준 지역별고용조사 ─────────────────────────────────────────
    def collect_employment_residence(self,
                                      start_prd: str = "202301",
                                      end_prd: str = "202502") -> int:
        """DT_1ES3A31S -- ABM 야간 에이전트 배치"""
        return self._collect_kosis_employment(
            "DT_1ES3A31S", "employment_residence", start_prd, end_prd)

    # ── E3: 경제활동인구조사 (월간) ─────────────────────────────────────────
    def collect_employment_monthly(self,
                                    start_prd: str = None,
                                    end_prd: str = None) -> int:
        """DT_1DA7002S -- 직업별 취업자 월간"""
        t0 = time.time()
        now = datetime.now()
        # 기본: 최근 24개월
        sp = start_prd or f"{now.year - 2}{now.month:02d}"
        ep = end_prd   or f"{now.year}{now.month:02d}"

        params = {
            "method":     "getList",
            "apiKey":     KEYS["kosis"],
            "itmId":      "T10+",
            "objL1":      "ALL",
            "objL2":      "",
            "format":     "json",
            "jsonVD":     "Y",
            "prdSe":      "M",
            "startPrdDe": sp,
            "endPrdDe":   ep,
            "orgId":      "101",
            "tblId":      "DT_1DA7002S",
        }
        data = self.get(KOSIS_BASE, params=params)
        if not data:
            log_collection("E", "DT_1DA7002S", "FAIL", elapsed=time.time()-t0)
            return 0

        rows = []
        try:
            item_list = data if isinstance(data, list) else []
            for it in item_list:
                if "err" in it:
                    log.error(f"  [E3] KOSIS err: {it}")
                    break
                rows.append({
                    "prd_de": it.get("PRD_DE", ""),
                    "c1":     it.get("C1", ""),
                    "c1_nm":  it.get("C1_NM", ""),
                    "itm_id": it.get("ITM_ID", ""),
                    "itm_nm": it.get("ITM_NM", ""),
                    "dt":     self.safe_float(it.get("DT")),
                })
        except Exception as e:
            log.error(f"  [E3] parse error: {e}")

        n = insert_rows("employment_monthly", rows)
        save_csv("employment_monthly", rows, date_str=f"{sp}_{ep}",
                 overwrite=True)
        log_collection("E", "DT_1DA7002S", "OK", n,
                       elapsed=time.time() - t0,
                       error=(None if rows else "no rows returned"))
        log.info(f"  [E3] employment_monthly: {n}건 저장")
        return n

    # ── E4: NEIS 학교기본정보 ─────────────────────────────────────────────────
    def collect_schools(self, max_rows: int = 2000) -> int:
        """
        NEIS schoolInfo (서울 전체 학교) → school_info

        API 문서: https://open.neis.go.kr/portal/data/service/selectServicePage.do
                  ?infId=OPEN17020190531110010104913&infSeq=2

        [주요 응답 필드]
          ATPT_OFCDC_SC_CODE  시도교육청코드
          SD_SCHUL_CODE       표준학교코드
          SCHUL_NM            학교명
          SCHUL_KND_SC_NM     학교종류명 (초등학교/중학교/고등학교/특수학교)
          LCTN_SC_NM          소재지명 (시도)
          JU_ORG_NM           관할조직명 (교육지원청)
          FOND_SC_NM          설립명 (공립/사립)
          ORG_RDNZC           도로명우편번호
          ORG_RDNMA           도로명주소
          ORG_RDNDA           도로명상세주소
          ORG_TELNO           전화번호
          HMPG_ADRES          홈페이지주소
          COEDU_SC_NM         남녀공학구분명
          DGHT_SC_NM          주야구분명
          FOND_YMD            설립일자
          LOAD_DTM            수정일자

        ⚠️ schoolInfo에는 학생수/학급수/위경도가 없음
           (별도 서비스: 학교학과정보, 학급수, 학생수 → 필요시 추가 수집)
        """
        t0 = time.time()
        all_rows = []
        page = 1
        page_size = 100
        max_retries_500 = 2  # HTTP 500 시 추가 재시도

        while len(all_rows) < max_rows:
            params = {
                "Key":              KEYS["neis"],   # 명세서 기준: 'Key' (대소문자 주의)
                "Type":             "json",
                "pIndex":           str(page),
                "pSize":            str(page_size),
                "ATPT_OFCDC_SC_CODE": "B10",  # 서울특별시교육청 코드
            }

            # NEIS는 간헐적 HTTP 500 반환 → 재시도 (최대 3회, 지수 백오프)
            data = None
            for retry in range(max_retries_500 + 1):
                data = self.get("https://open.neis.go.kr/hub/schoolInfo",
                                 params=params)
                if data is not None:
                    break
                wait = 2 ** retry  # 1s, 2s, 4s
                if retry < max_retries_500:
                    log.info(f"  [E4] NEIS 재시도 {retry+1}/{max_retries_500} "
                             f"(pIndex={page}) → {wait}s 대기")
                    time.sleep(wait)

            if not data:
                log.error(f"  [E4] NEIS schoolInfo 최종 실패 (pIndex={page})"
                           " -- 인증키 확인: open.neis.go.kr → 활용신청 → API KEY")
                break

            try:
                # NEIS 오류 응답 처리 (HTTP 200이지만 RESULT 코드로 오류 반환)
                if "RESULT" in data:
                    neis_code = data["RESULT"].get("CODE", "?")
                    neis_msg  = data["RESULT"].get("MESSAGE", "")
                    log.warning(f"  [E4] NEIS 오류 응답 CODE={neis_code}: {neis_msg}")
                    # INFO-200: 데이터 없음,  INFO-300: 서비스 제한
                    if neis_code in ("INFO-200", "INFO-300"):
                        break
                    # 기타 에러는 진행 중단
                    if neis_code.startswith("ERROR"):
                        break
                    # INFO-000 이면 정상이지만 RESULT만 반환된 경우
                    break

                result = data.get("schoolInfo", [])
                if not result or len(result) < 2:
                    log.warning(f"  [E4] schoolInfo 응답 구조 이상: "
                                f"{list(data.keys())[:5]}")
                    break

                head = result[0].get("head", [{}])
                total = int(head[0].get("list_total_count", 0))
                items = result[1].get("row", [])
                if not items:
                    break

                # 첫 페이지에서 실제 필드명 로그
                if page == 1 and items:
                    log.info(f"  [E4] NEIS 응답 필드: {list(items[0].keys())[:10]}...")
                    log.info(f"  [E4] 총 {total}건 수집 예정")

                now_iso = self.now_iso()
                for it in items:
                    # gu 추출: ORG_RDNMA(도로명주소)에서 '구' 부분 파싱
                    address = it.get("ORG_RDNMA", "")
                    gu_nm = ""
                    if address:
                        # "서울특별시 강남구 ..." → "강남구"
                        parts = address.split()
                        for p in parts:
                            if p.endswith("구"):
                                gu_nm = p
                                break

                    all_rows.append({
                        "collected_at": now_iso,
                        "school_nm":    it.get("SCHUL_NM", ""),
                        "school_code":  it.get("SD_SCHUL_CODE", ""),
                        "school_kind":  it.get("SCHUL_KND_SC_NM", ""),
                        "sido_nm":      it.get("LCTN_SC_NM", ""),
                        "gu_nm":        gu_nm,
                        "address":      address,
                        "fond_sc":      it.get("FOND_SC_NM", ""),
                        "coedu_sc":     it.get("COEDU_SC_NM", ""),
                        "dght_sc":      it.get("DGHT_SC_NM", ""),
                        "fond_ymd":     it.get("FOND_YMD", ""),
                        "tel_no":       it.get("ORG_TELNO", ""),
                        "homepage":     it.get("HMPG_ADRES", ""),
                    })

                if len(all_rows) >= total:
                    break
                page += 1
                time.sleep(0.3)

            except Exception as e:
                log.error(f"  [E4] NEIS parse error: {e}")
                break

        n = insert_rows("school_info", all_rows, replace=True)
        save_csv("school_info", all_rows, overwrite=True)
        log_collection("E", "NEIS_schoolInfo", "OK", n,
                       elapsed=time.time() - t0,
                       error=(None if all_rows else "no rows returned"))
        log.info(f"  [E4] school_info: {n}건 저장")
        return n

    # ── E5: KOSIS 연령별 자치구 인구 (DT_1B040A3) ───────────────────────────
    def collect_age_district(self, prd_de: str = None) -> int:
        """
        연령별 자치구 인구 수집
        1순위: KOSIS DT_1B040A3 (prdSe=Y, 4자리 연도 -- err=21 수정 버전)
        2순위: 서울 열린데이터 SPOP_LOCAL_RESD_JACHI (최신 연도만 가능, max_chunks 제한)
        """
        t0 = time.time()
        now = datetime.now()
        target_year = str(prd_de or now.year)[:4]

        n = self._collect_age_district_kosis(target_year)
        if n == 0:
            log.info(f"  [E5] KOSIS 0건 → 서울 열린데이터 폴백 (연도: {target_year})")
            n = self._collect_age_district_seoul_fallback(target_year)

        log_collection("E", "DT_1B040A3", "OK", n,
                       elapsed=time.time() - t0,
                       error=(None if n else "no rows returned"))
        log.info(f"  [E5] kosis_age_district: {n}건 저장 (기준연도: {target_year})")
        return n

    # ── E5 주: KOSIS DT_1B04005N (연간 prdSe=Y) ──────────────────────────────
    def _collect_age_district_kosis(self, target_year: str) -> int:
        """
        KOSIS DT_1B04005N -- 행정구역(읍면동)별/5세별 주민등록인구(2011년~)

        DT_1B040A3는 err:21 반환 (폐기 테이블).
        DT_1B04005N은 40,000셀 초과 방지를 위해 구별(objL1=5자리코드) 개별 호출.
        """
        gu_code_map = {
            "11110": "종로구", "11140": "중구", "11170": "용산구", "11200": "성동구",
            "11215": "광진구", "11230": "동대문구", "11260": "중랑구", "11290": "성북구",
            "11305": "강북구", "11320": "도봉구", "11350": "노원구", "11380": "은평구",
            "11410": "서대문구", "11440": "마포구", "11470": "양천구", "11500": "강서구",
            "11530": "구로구", "11545": "금천구", "11560": "영등포구", "11590": "동작구",
            "11620": "관악구", "11650": "서초구", "11680": "강남구", "11710": "송파구",
            "11740": "강동구",
        }

        # KOSIS 5세별 연령 → 표준 14구간 매핑
        age_map_5yr = {
            "0 - 4세": "0-9", "5 - 9세": "0-9",
            "10 - 14세": "10-14", "15 - 19세": "15-19",
            "20 - 24세": "20-24", "25 - 29세": "25-29",
            "30 - 34세": "30-34", "35 - 39세": "35-39",
            "40 - 44세": "40-44", "45 - 49세": "45-49",
            "50 - 54세": "50-54", "55 - 59세": "55-59",
            "60 - 64세": "60-64", "65 - 69세": "65-69",
            "70 - 74세": "70-74", "75 - 79세": "70-74",
            "80 - 84세": "70-74", "85 - 89세": "70-74",
            "90 - 94세": "70-74", "95 - 99세": "70-74",
            "100+": "70-74",
        }

        accum: dict = {}  # (gu_code, age_group) -> population
        gu_ok = 0

        for gu_code, gu_nm in gu_code_map.items():
            params = {
                "method":     "getList",
                "apiKey":     KEYS["kosis"],
                "itmId":      "ALL",
                "objL1":      gu_code,    # ← 구별 개별 호출
                "objL2":      "ALL",
                "format":     "json",
                "jsonVD":     "Y",
                "prdSe":      "Y",
                "startPrdDe": target_year,
                "endPrdDe":   target_year,
                "orgId":      "101",
                "tblId":      "DT_1B04005N",
            }
            data = self.get(KOSIS_BASE, params=params)
            if not data:
                continue
            item_list = data if isinstance(data, list) else []
            if not item_list or "err" in item_list[0]:
                log.warning(f"  [E5-KOSIS] {gu_nm} err: {item_list[0] if item_list else 'empty'}")
                continue

            gu_ok += 1
            for it in item_list:
                if "err" in it:
                    break
                itm_nm = it.get("ITM_NM", "")
                c2_nm = it.get("C2_NM", "").strip()
                if "총인구" not in itm_nm:
                    continue
                if c2_nm == "계":
                    continue
                age_group = age_map_5yr.get(c2_nm)
                if not age_group:
                    continue
                dt = self.safe_float(it.get("DT"))
                if dt is None:
                    continue
                key = (gu_code, age_group)
                accum[key] = accum.get(key, 0.0) + dt

            time.sleep(0.5)  # rate limit

        if not accum:
            log.warning(f"  [E5-KOSIS] {target_year}년 DT_1B04005N 파싱 0건 (gu_ok={gu_ok})")
            return 0

        # 자치구명 → 코드 역매핑
        gu_code_map = {
            "종로구": "11110", "중구": "11140", "용산구": "11170", "성동구": "11200",
            "광진구": "11215", "동대문구": "11230", "중랑구": "11260", "성북구": "11290",
            "강북구": "11305", "도봉구": "11320", "노원구": "11350", "은평구": "11380",
            "서대문구": "11410", "마포구": "11440", "양천구": "11470", "강서구": "11500",
            "구로구": "11530", "금천구": "11545", "영등포구": "11560", "동작구": "11590",
            "관악구": "11620", "서초구": "11650", "강남구": "11680", "송파구": "11710",
            "강동구": "11740",
        }

        # KOSIS 개별·범위 연령 표기 → 표준 구간 매핑
        def _age_to_group(age_nm: str) -> str | None:
            nm = age_nm.replace(" ", "")
            m = re.match(r"(\d+)[~\-](\d+)", nm)
            if m:
                lo, hi = int(m.group(1)), int(m.group(2))
            else:
                m2 = re.match(r"(\d+)", nm)
                if not m2:
                    return None
                lo = hi = int(m2.group(1))
            bands = [
                (0, 9, "0-9"), (10, 14, "10-14"), (15, 19, "15-19"),
                (20, 24, "20-24"), (25, 29, "25-29"), (30, 34, "30-34"),
                (35, 39, "35-39"), (40, 44, "40-44"), (45, 49, "45-49"),
                (50, 54, "50-54"), (55, 59, "55-59"), (60, 64, "60-64"),
                (65, 69, "65-69"), (70, 74, "70-74"),
            ]
            for g_lo, g_hi, label in bands:
                if lo <= g_hi and hi >= g_lo:
                    return label
            return None

        # 자치구별·연령구간별 총인구 집계
        accum: dict = {}
        for it in item_list:
            if "err" in it:
                break
            itm_nm = it.get("ITM_NM", "")
            c1_nm  = it.get("C1_NM", "").strip()
            c2_nm  = it.get("C2_NM", "").strip()
            prd    = it.get("PRD_DE", target_year)[:4]
            dt     = self.safe_float(it.get("DT"))

            # 서울 자치구만
            if not c1_nm.endswith("구") or c1_nm not in gu_code_map:
                continue
            # 총인구 항목만 (남자/여자 행 제외, 중복 방지)
            if "총인구" not in itm_nm and "합계" not in itm_nm:
                continue
            if dt is None:
                continue

            age_group = _age_to_group(c2_nm)
            if not age_group:
                continue

            key = (prd, c1_nm, age_group)
            accum[key] = accum.get(key, 0.0) + dt

        if not accum:
            log.warning(f"  [E5-KOSIS] {target_year}년 서울 자치구 데이터 파싱 결과 0건")
            return 0

        rows = []
        for (gu_code, age_group), pop in sorted(accum.items()):
            rows.append({
                "collected_at": self.now_iso(),
                "prd_de":       target_year,
                "gu_code":      gu_code,
                "gu_nm":        gu_code_map.get(gu_code, ""),
                "age_group":    age_group,
                "population":   int(round(pop)),
            })

        with get_conn() as conn:
            conn.execute("DELETE FROM kosis_age_district WHERE prd_de = ?",
                         (target_year,))

        n = insert_rows("kosis_age_district", rows)
        save_csv("kosis_age_district", rows, date_str=target_year, overwrite=True)
        log.info(f"  [E5-KOSIS] DT_1B04005N {target_year}년: {n}건 저장 ({gu_ok}/25 gu)")
        return n

    # ── E5 폴백: 서울 열린데이터 SPOP_LOCAL_RESD_JACHI ─────────────────────────
    def _collect_age_district_seoul_fallback(self, target_year: str) -> int:
        """
        SPOP_LOCAL_RESD_JACHI는 연령대별 인구를 wide 컬럼으로 제공한다.
        최신 기준일의 24시간 값을 자치구별로 평균 내어 연령대 총인구로 변환한다.

        ⚠️ 이 API는 최신 데이터(2025~2026)만 보유.
           2020~2024 수집 시도 시 전체 데이터셋을 역순 스캔하게 되므로
           max_chunks 제한 및 조기 종료 로직이 반드시 필요.
        """
        gu_map = {
            "11110": "종로구", "11140": "중구", "11170": "용산구", "11200": "성동구",
            "11215": "광진구", "11230": "동대문구", "11260": "중랑구", "11290": "성북구",
            "11305": "강북구", "11320": "도봉구", "11350": "노원구", "11380": "은평구",
            "11410": "서대문구", "11440": "마포구", "11470": "양천구", "11500": "강서구",
            "11530": "구로구", "11545": "금천구", "11560": "영등포구", "11590": "동작구",
            "11620": "관악구", "11650": "서초구", "11680": "강남구", "11710": "송파구",
            "11740": "강동구",
        }
        age_columns = [
            ("F0T9", "0-9"), ("F10T14", "10-14"), ("F15T19", "15-19"),
            ("F20T24", "20-24"), ("F25T29", "25-29"), ("F30T34", "30-34"),
            ("F35T39", "35-39"), ("F40T44", "40-44"), ("F45T49", "45-49"),
            ("F50T54", "50-54"), ("F55T59", "55-59"), ("F60T64", "60-64"),
            ("F65T69", "65-69"), ("F70T74", "70-74"),
        ]
        api_key = KEYS.get("seoul_general", "")
        if not api_key:
            return 0

        meta = self.get(f"{SEOUL_BASE}/{api_key}/json/SPOP_LOCAL_RESD_JACHI/1/1/")
        total_count = (
            meta.get("SPOP_LOCAL_RESD_JACHI", {}).get("list_total_count", 0)
            if isinstance(meta, dict) else 0
        )
        if not total_count:
            log.warning("  [E5 폴백] SPOP_LOCAL_RESD_JACHI 총건수 조회 실패")
            return 0

        target_year = str(target_year)[:4]
        chunk_size = 1000
        max_chunks = 100          # 최대 100청크(100,000건) 역순 스캔 후 중단
        end_idx = total_count
        latest_date = None
        candidate_rows = []
        chunk_count = 0

        log.info(f"  [E5 폴백] 총 {total_count}건 역순 스캔 시작 (최대 {max_chunks}청크)")

        while end_idx > 0 and chunk_count < max_chunks:
            start_idx = max(1, end_idx - chunk_size + 1)
            raw = self.get(
                f"{SEOUL_BASE}/{api_key}/json/SPOP_LOCAL_RESD_JACHI/{start_idx}/{end_idx}/"
            )
            items = (raw.get("SPOP_LOCAL_RESD_JACHI", {}).get("row", [])
                     if isinstance(raw, dict) else [])
            chunk_count += 1

            if not items:
                log.warning(f"  [E5 폴백] chunk {chunk_count}: 빈 응답 -- 스캔 중단")
                break

            # 조기 종료: 현재 청크의 최고 기준일이 목표 연도보다 훨씬 이전이면 중단
            oldest_date = min(str(it.get("STDR_DE_ID", "99991231")) for it in items)
            if oldest_date[:4] < str(int(target_year) - 1):
                log.warning(
                    f"  [E5 폴백] 조기 중단: 최고 기준일 {oldest_date} < {int(target_year)-1}년 "
                    f"({chunk_count}청크/{chunk_size}건 스캔)"
                )
                break

            year_items = [
                it for it in items
                if str(it.get("STDR_DE_ID", "")).startswith(target_year)
            ]
            if year_items:
                latest_date = max(str(it.get("STDR_DE_ID", "")) for it in year_items)
                candidate_rows = [
                    it for it in year_items
                    if str(it.get("STDR_DE_ID", "")) == latest_date
                ]
                break

            end_idx = start_idx - 1
            time.sleep(0.2)

        if not candidate_rows:
            if chunk_count >= max_chunks:
                log.warning(
                    f"  [E5 폴백] {target_year}년 데이터 없음 "
                    f"(최대 {max_chunks}청크={max_chunks*chunk_size:,}건 스캔 완료)"
                )
            else:
                log.warning(f"  [E5 폴백] {target_year}년 데이터 없음")
            return 0

        log.info(f"  [E5 폴백] 기준일 {latest_date}, 행수 {len(candidate_rows)}")

        accum = {}
        counts = {}
        for it in candidate_rows:
            gu_code = str(it.get("ADSTRD_CODE_SE", "") or "")[:5]
            if gu_code not in gu_map:
                continue
            for suffix, age_group in age_columns:
                male   = self.safe_float(it.get(f"MALE_{suffix}_LVPOP_CO")) or 0.0
                female = self.safe_float(it.get(f"FEMALE_{suffix}_LVPOP_CO")) or 0.0
                key = (gu_code, age_group)
                accum[key]  = accum.get(key, 0.0) + male + female
                counts[key] = counts.get(key, 0) + 1

        rows = []
        for (gu_code, age_group), total_val in sorted(accum.items()):
            avg_val = total_val / max(counts.get((gu_code, age_group), 1), 1)
            rows.append({
                "collected_at": self.now_iso(),
                "prd_de":       latest_date[:4],
                "gu_code":      gu_code,
                "gu_nm":        gu_map.get(gu_code, ""),
                "age_group":    age_group,
                "population":   int(round(avg_val)),
            })

        if not rows:
            log.warning("  [E5 폴백] 연령 파싱 후 0건")
            return 0

        with get_conn() as conn:
            conn.execute("DELETE FROM kosis_age_district WHERE prd_de = ?",
                         (latest_date[:4],))

        n = insert_rows("kosis_age_district", rows)
        save_csv("kosis_age_district", rows, date_str=latest_date, overwrite=True)
        return n

    # ── E6: HIRA 의료기관 기본정보 ────────────────────────────────────────────
    def collect_hospitals(self, max_pages: int = 20) -> int:
        """
        HIRA hospInfoServicev2/getHospBasisList → hospitals

        서울 의료기관(병원·의원·상급종합) 위치·병상수·의사수.
        SIR 모델 의료 수용력(병상수) 및 격리·치료 파라미터 보정에 사용.

        API: https://apis.data.go.kr/B551182/hospInfoServicev2/getHospBasisList
        serviceKey: data_go_kr
        """
        t0 = time.time()
        # ── HIRA hospInfoServicev2/getHospBasisList ───────────────────────────
        # bedCnt 문제: API 응답에서 drTotCnt는 정상 반환되지만 bedCnt는 NULL.
        # 원인 미확인 -- E6 재실행 시 로그의 "API fields" 줄에서 실제 필드명 확인 필요.
        # sidoCd: HIRA 코드표 기준 서울=110000 (구버전 API에서는 11 또는 1100000 사용 사례)
        HIRA_BASE = "https://apis.data.go.kr/B551182/hospInfoServicev2/getHospBasisList"
        hospital_class_codes = ["01", "11", "21", "28", "29"]
        all_rows = []
        seen_ykiho = set()
        completed_classes = set()

        for cl_cd in hospital_class_codes:
            page = 1
            while page <= max_pages:
                params = {
                    "serviceKey": KEYS["data_go_kr"],
                    "pageNo":     str(page),
                    "numOfRows":  "100",
                    "_type":      "json",
                    "sidoCd":     "110000",
                    "clCd":       cl_cd,
                }
                data = self.get(HIRA_BASE, params=params)
                if not data:
                    log.warning(f"  [E6] HIRA clCd={cl_cd} page {page} ?ㅽ뙣")
                    break

                try:
                    body = data.get("response", {}).get("body", {})
                    items = body.get("items", {})
                    item_list = (items.get("item", [])
                                 if isinstance(items, dict) else items or [])
                    if isinstance(item_list, dict):
                        item_list = [item_list]

                    if not item_list:
                        completed_classes.add(cl_cd)
                        break

                    total_cnt = int(body.get("totalCount", 0) or 0)
                    if page == 1:
                        cl_nm = item_list[0].get("clCdNm", cl_cd)
                        log.info(f"  [E6] {cl_nm}({cl_cd}) total={total_cnt}")
                        # ── 진단: API 실제 반환 필드명 1회 로그 ──────────────
                        # bed_cnt가 모두 NULL일 경우 아래 로그에서 정확한 필드명 확인
                        # 확인 후 아래 bed_cnt fallback 목록에 추가할 것
                        log.info(f"  [E6] API fields({cl_cd}): "
                                 f"{sorted(item_list[0].keys())}")

                    for it in item_list:
                        ykiho = it.get("ykiho", "")
                        if not ykiho or ykiho in seen_ykiho:
                            continue
                        seen_ykiho.add(ykiho)

                        addr = it.get("addr", "")
                        gu_nm = ""
                        for p in addr.split():
                            if p.endswith(chr(0xAD6C)):
                                gu_nm = p
                                break

                        # ── 병상수 필드명 다중 fallback ──────────────────────
                        # HIRA getHospBasisList 필드명이 버전마다 다름.
                        # API fields 로그 확인 후 실제 필드명 추가 가능.
                        bed_cnt = self.safe_int(
                            it.get("bedCnt")         or   # 표준 필드
                            it.get("bedCount")       or   # 일부 버전
                            it.get("totBedCnt")      or   # 합계 병상
                            it.get("gnrlSckbdCnt")   or   # 일반병상
                            it.get("sickBedCount")   or   # 영문 변형
                            it.get("ICUBedCnt")      or   # 중환자실
                            it.get("hopBedCnt")           # 병원 병상
                        )

                        all_rows.append({
                            "collected_at": self.now_iso(),
                            "ykiho":        ykiho,
                            "inst_nm":      it.get("yadmNm", ""),
                            "addr":         addr,
                            "gu_nm":        gu_nm,
                            "clcd_nm":      it.get("clCdNm", ""),
                            "bed_cnt":      bed_cnt,
                            "dr_cnt":       self.safe_int(it.get("drTotCnt")),
                            "tel":          it.get("telno", ""),
                            "lat":          self.safe_float(it.get("YPos")),
                            "lng":          self.safe_float(it.get("XPos")),
                        })

                    if page * 100 >= total_cnt:
                        completed_classes.add(cl_cd)
                        break
                    page += 1
                    time.sleep(0.2)

                except Exception as e:
                    log.error(f"  [E6] HIRA clCd={cl_cd} page {page} parse error: {e}")
                    break

        # ── bedCnt 보완: HIRA API가 bedCnt 미반환 → 종별 표준 병상수 추정값 대입 ──
        # 근거: 건강보험심사평가원 2023년 의료기관 현황 (서울 평균)
        #   상급종합(01): 서울 14개 평균 ~850병상
        #   종합병원(11) : 서울 45개 평균 ~330병상
        #   병원(21)     : 서울 233개 평균 ~100병상
        #   요양병원(28) : 서울 102개 평균 ~190병상
        #   정신병원(29) : 서울 13개 평균 ~300병상
        # 개별 기관의 실제 병상수가 필요한 경우 KOSIS DT_MEDI 또는
        #   건강보험심사평가원 홈페이지 "요양기관 현황" CSV 직접 다운로드 필요.
        AVG_BEDS: dict[str, int] = {
            "상급종합": 850, "종합병원": 330,
            "병원": 100,    "요양병원": 190, "정신병원": 300,
        }
        filled = 0
        for row in all_rows:
            if row.get("bed_cnt") is None:
                est = AVG_BEDS.get(row.get("clcd_nm", ""), None)
                if est:
                    row["bed_cnt"] = est
                    filled += 1
        if filled:
            log.info(f"  [E6] bed_cnt 추정값 대입: {filled}개 기관 "
                     f"(HIRA API bedCnt 미반환 -- 종별 평균값 사용)")

        if all_rows and len(completed_classes) == len(hospital_class_codes):
            with get_conn() as conn:
                conn.execute("DELETE FROM hospitals")

        n = insert_rows("hospitals", all_rows, replace=True)
        save_csv("hospitals", all_rows, overwrite=True)
        log_collection("E", "HIRA_hospInfoServicev2", "OK", n,
                       elapsed=time.time() - t0,
                       error=(None if all_rows else "no rows returned"))
        log.info(f"  [E6] hospitals: {n} rows saved")
        return n


    def run(self, start_prd: str = "202301", end_prd: str = "202502",
            e5_years: list = None,
            skip_apis: list = None) -> dict:
        """
        Group E 전체 실행
        start_prd / end_prd: KOSIS 반기 범위 (기본 2023~2025 전체)
        e5_years: E5 자치구 연령인구 수집 연도 목록 (예: ['2020','2021','2022','2023','2024'])
                  미지정 시 현재 연도만 수집
        skip_apis: 스킵할 API 코드 목록 (예: ['E4', 'E3'])
        """
        skip_apis = skip_apis or []
        log.info("▶ Group E -- 반기/월간 고용 수집 시작")
        r1 = r2 = r3 = r4 = r5 = r6 = 0

        if "E1" not in skip_apis:
            try:
                r1 = self.collect_employment_workplace(start_prd, end_prd)
            except Exception as e:
                log.error(f"  [E1] employment_workplace 예외 (스킵): {e}")
        else:
            log.info("  [E1] DT_1ES3C07S -- 스킵 (--skip E1)")
        time.sleep(1)

        if "E2" not in skip_apis:
            try:
                r2 = self.collect_employment_residence(start_prd, end_prd)
            except Exception as e:
                log.error(f"  [E2] employment_residence 예외 (스킵): {e}")
        else:
            log.info("  [E2] DT_1ES3A31S -- 스킵 (--skip E2)")
        time.sleep(1)

        if "E3" not in skip_apis:
            try:
                r3 = self.collect_employment_monthly()
            except Exception as e:
                log.error(f"  [E3] employment_monthly 예외 (스킵): {e}")
        else:
            log.info("  [E3] DT_1DA7002S -- 스킵 (--skip E3)")
        time.sleep(1)

        # ── E4: NEIS schoolInfo ───────────────────────────────────────────────
        # 2026-03 NEIS 포털 상태 OK 확인 → 재활성화
        if "E4" not in skip_apis:
            try:
                r4 = self.collect_schools()
            except Exception as e:
                log.error(f"  [E4] NEIS schoolInfo 예외 (스킵): {e}")
        else:
            log.info("  [E4] NEIS_schoolInfo -- 스킵 (--skip E4)")
        time.sleep(1)

        if "E5" not in skip_apis:
            try:
                years_to_collect = e5_years or [str(datetime.now().year)]
                log.info(f"  [E5] 수집 연도 목록: {years_to_collect}")
                for yr in years_to_collect:
                    n_yr = self.collect_age_district(prd_de=str(yr))
                    r5 += n_yr
                    if n_yr == 0:
                        log.warning(f"  [E5] {yr}년 데이터 0건 (Seoul API에 해당 연도 없음)")
                    time.sleep(1)
            except Exception as e:
                log.error(f"  [E5] KOSIS DT_1B040A3 예외 (스킵): {e}")
        else:
            log.info("  [E5] KOSIS DT_1B040A3 -- 스킵 (--skip E5)")
        time.sleep(1)

        if "E6" not in skip_apis:
            try:
                r6 = self.collect_hospitals()
            except Exception as e:
                log.error(f"  [E6] HIRA getMedBasisList 예외 (스킵): {e}")
        else:
            log.info("  [E6] HIRA getMedBasisList -- 스킵 (--skip E6)")

        return {
            "employment_workplace": r1,
            "employment_residence": r2,
            "employment_monthly":   r3,
            "school_info":          r4,
            "kosis_age_district":   r5,
            "hospitals":            r6,
        }
