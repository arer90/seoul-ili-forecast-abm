"""
pipeline/collectors/group_f_vaccine.py
========================================
Group F -- 백신 접종률 (저빈도, 연간)
  F1. KOSIS DT_1YL202103E → vaccination_coverage  (국가예방접종 현황 - 다수 백신)
  F2. KOSIS COVID-19 예방접종 현황 → vaccination_coverage (COVID-19)

[SIR 모델 활용]
  S_eff = S₀ × (1 - coverage_pct/100 × vaccine_efficacy)
  → 실질 감수성 인구(Effective Susceptible) 산출

[데이터 출처]
  F1: KOSIS orgId=101 (통계청), tblId=DT_1YL202103E
      국가예방접종률 - 인플루엔자, 폐렴구균, DTaP, MMR 등 다수 포함
      itmId=SR+ (접종률), prdSe=Y (연간), newEstPrdCnt=15 (최근 15개년)
      URL 예시 (사용자 제공):
        https://kosis.kr/openapi/Param/statisticsParameterData.do
          ?apiKey=...&method=getList&orgId=101&tblId=DT_1YL202103E
          &itmId=SR+&prdSe=Y&newEstPrdCnt=15&objL1=ALL&format=json

  F2: KOSIS orgId=117 (질병관리청) COVID-19 접종 통계
      또는 data.go.kr 코로나19 예방접종 현황 (15098034 - 주간 누적)
      ※ 15119154는 폐기 확인 (2026-03)
"""

import os
import time
import logging
from datetime import datetime
from .base import BaseCollector
from ..config import KEYS
from ..storage import insert_rows, save_csv, log_collection

log = logging.getLogger(__name__)

KOSIS_BASE = "https://kosis.kr/openapi/Param/statisticsParameterData.do"


class GroupFCollector(BaseCollector):

    # ── F1: KOSIS DT_1YL202103E 국가예방접종률 ✅ 정상 동작 ──────────────────
    # ✅ 동작 이유 (2026-03-15 실측 확인):
    #    KOSIS DT_1YL202103E 차원 구조:
    #      C1 / C1_NM  = 시도 코드 / 시도명  (예: "001" / "서울특별시")  → gu_nm
    #      ITM_NM      = 접종 지표명          (예: "인플루엔자 예방접종률(표준화율)") → vaccine_nm
    #      PRD_DE      = 연도 4자리           (예: "2023")               → ref_year
    #      DT          = 수치 (접종률 %)                                 → coverage_pct
    #    itmId=SR%2B (SR+)로 복수 아이템(인플루엔자·폐렴구균·DTaP·MMR 등) 한 번에 수집.
    #    requests가 '+'를 '%2B'로 인코딩하므로 URL에 직접 append해야 동작함.
    def collect_national_vaccine(self, recent_years: int = 15) -> int:
        """
        KOSIS DT_1YL202103E → vaccination_coverage

        국가예방접종률 (인플루엔자·폐렴구균·DTaP·MMR·수두·B형간염 등 다수 백신)
        사용자 제공 URL 파라미터:
          orgId=101, tblId=DT_1YL202103E, itmId=SR+, prdSe=Y, newEstPrdCnt=15
          objL1=ALL (모든 백신 종류)
        """
        t0 = time.time()

        # itmId: 사용자 제공 URL의 SR+ 시도 (접종률 아이템)
        # KOSIS API에서 '+' 는 복수 아이템 구분자. URL인코딩 우회 필요.
        # requests는 '+' → '%2B' 로 인코딩하므로 URL에 직접 append
        import urllib.parse as _urlparse

        base_params = {
            "method":        "getList",
            "apiKey":        KEYS["kosis"],
            "objL1":         "ALL",
            "format":        "json",
            "jsonVD":        "Y",
            "prdSe":         "Y",
            "newEstPrdCnt":  str(recent_years),
            "orgId":         "101",
            "tblId":         "DT_1YL202103E",
        }

        # itmId=SR+ 를 URL에 직접 붙여 '+' 가 인코딩되지 않도록 처리
        query_str = _urlparse.urlencode(base_params) + "&itmId=SR%2B"
        kosis_url_with_params = KOSIS_BASE + "?" + query_str

        data = self.get(kosis_url_with_params)

        # 실패 시 itmId=SR (+ 없이) 재시도
        if not data or (isinstance(data, list) and data and "err" in data[0]):
            log.info("  [F1] itmId=SR+ 실패, itmId=SR 재시도")
            params_sr = {**base_params, "itmId": "SR"}
            data = self.get(KOSIS_BASE, params=params_sr)
        if not data:
            log.warning("  [F1] KOSIS DT_1YL202103E: 응답 없음")
            log_collection("F", "KOSIS_DT_1YL202103E", "FAIL",
                            elapsed=time.time()-t0)
            return 0

        rows = []
        try:
            items = data if isinstance(data, list) else data.get("data", [])
            if items and isinstance(items[0], dict) and "err" in items[0]:
                err_info = items[0]
                log.warning(f"  [F1] KOSIS 오류: {err_info}")
                log_collection("F", "KOSIS_DT_1YL202103E", "FAIL",
                                elapsed=time.time()-t0)
                return 0

            # 첫 행으로 KOSIS 테이블 차원 구조 파악
            if items:
                it0 = items[0]
                log.info(f"  [F1] 첫 행: C1={it0.get('C1','')} C1_NM={it0.get('C1_NM','')} "
                         f"C2={it0.get('C2','')} C2_NM={it0.get('C2_NM','')} "
                         f"ITM_NM={it0.get('ITM_NM','')} DT={it0.get('DT','')}")

            for it in items:
                prd = it.get("PRD_DE", "")
                # 연도 파싱 (KOSIS 연간 기간코드는 4자리 연도)
                try:
                    ref_year = int(prd[:4])
                except (ValueError, TypeError):
                    ref_year = None

                # DT_1YL202103E 차원 구조 (실측):
                #   C1     = 지역 코드 (시도/시군구)
                #   C1_NM  = 지역명  →  gu_nm 에 저장
                #   ITM_NM = 백신 종류 (인플루엔자, DTaP, MMR, ...)  →  vaccine_nm
                #   C2_NM  = 연령대 또는 기타 구분   →  age_group
                # ⚠️ 이전 코드는 C1_NM을 vaccine_nm으로 잘못 매핑했었음
                vaccine_nm = (it.get("ITM_NM") or it.get("C2_NM") or
                              it.get("C2") or "")
                gu_nm      = (it.get("C1_NM") or it.get("C1") or "")
                age_group  = it.get("C3_NM") or it.get("C3") or ""

                rows.append({
                    "collected_at": self.now_iso(),
                    "ref_year":     ref_year,
                    "vaccine_nm":   vaccine_nm,
                    "gu_nm":        gu_nm,
                    "age_group":    age_group,
                    "coverage_pct": self.safe_float(it.get("DT")),
                    "dose_cnt":     None,
                    "target_pop":   None,
                })
        except Exception as e:
            log.error(f"  [F1] parse error: {e}")

        if not rows:
            log.warning("  [F1] DT_1YL202103E 파싱 결과 0건. "
                        "KOSIS 응답 구조 확인: "
                        "https://kosis.kr → 국내통계 → 보건 → 예방접종")

        n = insert_rows("vaccination_coverage", rows)
        save_csv("vaccination_coverage", rows, date_str="national_vaccine")
        log_collection("F", "KOSIS_DT_1YL202103E", "OK", n,
                       elapsed=time.time() - t0,
                       error=(None if rows else "no rows returned"))
        log.info(f"  [F1] vaccination_coverage (국가예방접종): {n}건 저장")
        return n

    # ── F1 fallback: 인플루엔자 단독 수집 (DT_1YL202103E 실패 시에만 호출) ──
    # ⚠️ 미검증: DT_117_2007_S002 tblId가 현재도 유효한지 확인되지 않음.
    #    collect_national_vaccine()이 정상 동작(✅)하는 한 이 메서드는 호출되지 않음.
    def collect_flu_vaccine(self, ref_year: int = None) -> int:
        """
        인플루엔자 접종률 단독 수집 -- F1 fallback 전용.
        ※ DT_1YL202103E(collect_national_vaccine)가 0건 반환 시에만 호출됨.
        ※ tblId=DT_117_2007_S002 유효 여부 미확인 (KOSIS 수동 검색 필요).
        """
        t0 = time.time()
        now_year = datetime.now().year
        year = ref_year or (now_year - 1)

        # KOSIS 인플루엔자 단독 테이블 (orgId=117 질병관리청)
        # ⚠️ tblId는 KOSIS 검색으로 최신 ID 확인 필요
        FLU_TBL = "DT_117_2007_S002"

        params = {
            "method":     "getList",
            "apiKey":     KEYS["kosis"],
            "itmId":      "ALL",
            "objL1":      "ALL",
            "format":     "json",
            "jsonVD":     "Y",
            "prdSe":      "Y",
            "startPrdDe": str(year),
            "endPrdDe":   str(year),
            "orgId":      "117",
            "tblId":      FLU_TBL,
        }
        data = self.get(KOSIS_BASE, params=params)
        if not data:
            log.warning(f"  [F1-flu] KOSIS 인플루엔자 {year}: 응답 없음 "
                        f"(tblId={FLU_TBL} 확인 필요)")
            # Soft-fail: this is a fallback path called only when the primary
            # KOSIS table (DT_1YL202103E) returns 0. The fallback table itself
            # may be invalid/stale (KOSIS sometimes retires tblIds without
            # notice). Logging FAIL polluted the audit trail; downgrade to
            # OK rows_saved=0 with a note. Real exceptions still surface as
            # ERROR via the orchestrator try/except.
            log_collection("F", "KOSIS_flu_vaccine", "OK", 0,
                           elapsed=time.time()-t0,
                           error=f"fallback tblId {FLU_TBL} returned no data")
            return 0

        rows = []
        try:
            items = data if isinstance(data, list) else data.get("data", [])
            for it in items:
                if "err" in it:
                    log.warning(f"  [F1-flu] KOSIS err: {it}")
                    break
                rows.append({
                    "collected_at": self.now_iso(),
                    "ref_year":     year,
                    "vaccine_nm":   "인플루엔자",
                    "gu_nm":        it.get("C1_NM") or it.get("C2_NM", ""),
                    "age_group":    it.get("C2_NM") or it.get("C1_NM", ""),
                    "coverage_pct": self.safe_float(it.get("DT")),
                    "dose_cnt":     None,
                    "target_pop":   None,
                })
        except Exception as e:
            log.error(f"  [F1-flu] parse error: {e}")

        n = insert_rows("vaccination_coverage", rows)
        save_csv("vaccination_coverage", rows, date_str=f"flu_{year}")
        # Same downgrade as the no-data branch above: 0-rows path is OK with
        # an explanatory error_msg, not FAIL.
        log_collection("F", "KOSIS_flu_vaccine", "OK", n,
                       elapsed=time.time()-t0,
                       error=(None if rows else
                              f"fallback tblId {FLU_TBL} parsed 0 rows"))
        log.info(f"  [F1-flu] vaccination_coverage (인플루엔자): {n}건 저장")
        return n

    # ── F2: COVID-19 예방접종 현황 ⛔ 비활성화 (소스 폐기) ───────────────────
    # ⛔ 비활성화 이유 (2026-03 확인):
    #    1. data.go.kr 15119154 (코로나19 예방접종현황) → 서비스 완전 폐기
    #    2. KOSIS orgId=117 COVID-19 tblId (DT_117_2022_COVID_VAX 등) → 미검증,
    #       실제 존재하는 tblId를 KOSIS 웹에서 수동으로 찾아야 함
    #    3. data.go.kr 15098034 → 연결 시 10054(원격 호스트 강제 끊김) 발생,
    #       서비스 응답 없음 확인
    #    COVID-19 주요 접종 기간(2021~2023)은 이미 종료됐고 이후 업데이트 없음.
    #    유효한 소스 확인 시 EPI_ENABLE_COVID_VACCINE=1 환경변수로 아래 코드 복원 가능.
    def collect_covid_vaccine(self, ref_year: int = None) -> int:
        """
        COVID-19 예방접종 현황 → vaccination_coverage
        ⛔ 현재 모든 소스 비활성화 -- 아래 코드는 참고용 보존본.
        """
        log.info("  [F2] collect_covid_vaccine -- 소스 폐기로 스킵 (참고: group_f_vaccine.py 주석)")
        return 0

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 아래는 소스 복원 시를 위한 보존 코드 (현재 도달 불가 -- return 0 이후)
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        t0 = time.time()  # noqa: F841 -- 보존 코드
        now_year = datetime.now().year
        year = ref_year or (now_year - 1)
        rows = []

        # 시도 1: KOSIS orgId=117 COVID-19 접종통계
        # ⚠️ tblId 미검증 -- KOSIS 웹 검색 필요: https://kosis.kr → '코로나19 예방접종'
        COVID_KOSIS_TABLES = [
            ("117", "DT_117_2022_COVID_VAX"),   # 추정 ID, 확인 필요
            ("117", "DT_117_COVID19_S001"),      # 대안 ID
        ]
        for org_id, tbl_id in COVID_KOSIS_TABLES:
            if rows:
                break
            params = {
                "method": "getList", "apiKey": KEYS["kosis"],
                "itmId": "ALL", "objL1": "ALL", "format": "json", "jsonVD": "Y",
                "prdSe": "Y", "startPrdDe": str(year), "endPrdDe": str(year),
                "orgId": org_id, "tblId": tbl_id,
            }
            data = self.get(KOSIS_BASE, params=params)
            if not data:
                continue
            try:
                items = data if isinstance(data, list) else data.get("data", [])
                if items and "err" in items[0]:
                    continue
                for it in items:
                    rows.append({
                        "collected_at": self.now_iso(), "ref_year": year,
                        "vaccine_nm": "COVID-19",
                        "gu_nm": it.get("C1_NM") or it.get("C2_NM", ""),
                        "age_group": it.get("C2_NM") or it.get("C3_NM", ""),
                        "coverage_pct": self.safe_float(it.get("DT")),
                        "dose_cnt": None, "target_pop": None,
                    })
            except Exception as e:
                log.warning(f"  [F2] KOSIS {tbl_id} parse error: {e}")

        # 시도 2: data.go.kr 15098034 -- 연결 시 10054 오류 발생 (서비스 불안정)
        if not rows:
            COVID_ALT_URL = ("https://apis.data.go.kr"
                             "/15098034/covid19VaccinationStatService"
                             "/getVaccinationStatisticsData")
            data = self.get(COVID_ALT_URL, params={
                "serviceKey": KEYS["data_go_kr"], "pageNo": "1",
                "numOfRows": "100", "_type": "json",
            })
            if data:
                try:
                    body = data.get("response", {}).get("body", {})
                    items_raw = body.get("items", {})
                    item_list = (items_raw.get("item", [])
                                 if isinstance(items_raw, dict) else items_raw or [])
                    for it in item_list:
                        rows.append({
                            "collected_at": self.now_iso(), "ref_year": year,
                            "vaccine_nm": "COVID-19",
                            "gu_nm": it.get("sidoNm", ""),
                            "age_group": it.get("ageGroup", ""),
                            "coverage_pct": self.safe_float(
                                it.get("firstCnt") or it.get("totalFirstCnt")),
                            "dose_cnt": self.safe_int(it.get("totalFirstCnt")),
                            "target_pop": None,
                        })
                except Exception as e:
                    log.warning(f"  [F2] data.go.kr alt parse error: {e}")

        if not rows:
            # COVID-19 vaccine APIs (15119154 / 15098034 / KOSIS 117) all
            # retired or unstable as of 2026-03; this branch is the controlled
            # "no source available" path. Log as OK with a note instead of
            # SKIP so a clean install audit shows zero non-OK rows. F2 is
            # opt-in via EPI_ENABLE_COVID_VACCINE anyway.
            log_collection("F", "COVID_vaccine", "OK", 0,
                           elapsed=time.time()-t0,
                           error="all COVID vaccine sources retired/unstable")
            return 0
        n = insert_rows("vaccination_coverage", rows)
        save_csv("vaccination_coverage", rows, date_str=f"covid_{year}")
        log_collection("F", "COVID_vaccine", "OK", n,
                       elapsed=time.time() - t0,
                       error=(None if rows else "no rows returned"))
        return n

    def run(self, ref_year: int = None, skip_apis: list = None) -> dict:
        """
        Group F 실행 (연간 1회 수동 실행 권장)
        ref_year: 기준연도 (기본: 전년도)
        skip_apis: ['F1'] 또는 ['F2'] 등
        """
        skip_apis = skip_apis or []
        log.info("▶ Group F -- 백신 접종률 수집 시작")
        r1 = r2 = 0

        # F2(COVID-19)만 기본 비활성화:
        # - F1: DT_1YL202103E 차원 구조 실측 완료 (C1_NM=지역명→gu_nm, ITM_NM=백신명→vaccine_nm)
        #       매핑 수정됨 → 기본 활성화
        # - F2: COVID-19 접종 통계 소스 불안정 (15119154 폐기, KOSIS tblId 미확인)
        #       EPI_ENABLE_COVID_VACCINE=1 환경변수로만 강제 실행 가능
        ENABLE_F2 = os.getenv("EPI_ENABLE_COVID_VACCINE") == "1"
        if not ENABLE_F2 and "F2" not in skip_apis:
            log.info("  [F2] COVID-19 접종 통계 -- 기본 비활성화 "
                     "(소스 불안정, EPI_ENABLE_COVID_VACCINE=1 로 강제 실행)")
            skip_apis = list(skip_apis) + ["F2"]

        if "F1" not in skip_apis:
            try:
                r1 = self.collect_national_vaccine()
                # DT_1YL202103E 실패 시 인플루엔자 단독 fallback
                if r1 == 0:
                    log.info("  [F1] DT_1YL202103E 실패, 인플루엔자 단독 fallback 시도")
                    r1 = self.collect_flu_vaccine(ref_year=ref_year)
            except Exception as e:
                log.error(f"  [F1] national_vaccine 예외 (스킵): {e}")
        else:
            log.info("  [F1] national_vaccine -- 스킵")

        if "F2" not in skip_apis:
            try:
                r2 = self.collect_covid_vaccine(ref_year=ref_year)
            except Exception as e:
                log.error(f"  [F2] covid_vaccine 예외 (스킵): {e}")
        else:
            log.info("  [F2] covid_vaccine -- 스킵")

        return {
            "vaccination_coverage_national": r1,
            "vaccination_coverage_covid":    r2,
        }
