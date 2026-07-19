"""
pipeline/collectors/group_d_weekly.py
========================================
Group D -- 주간 배치 (매주 목~금 발표)
  D1. KDCA EIDAPIService/PeriodBasic → weekly_disease  (주별 전국)
  D2. KDCA EIDAPIService/Region      → weekly_disease  (서울 지역별 연간)
  D3. KDCA EIDAPIService/Age         → disease_age     (연령별 발생)
  D4. KDCA EIDAPIService/Gender      → disease_gender  (성별 발생)
  D5. KDCA EIDAPIService/death       → disease_death   (감염병별 사망)
  D6. KDCA EIDAPIService/PeriodBasic → weekly_disease  (월별 전국)
"""

import re
import time
import logging
from datetime import datetime
from .base import BaseCollector
from ..config import KEYS
from ..storage import (
    get_conn,
    insert_rows,
    save_csv,
    log_collection,
    refresh_disease_catalog,
)

log = logging.getLogger(__name__)

KDCA_BASE = "https://apis.data.go.kr/1790387/EIDAPIService"


class GroupDCollector(BaseCollector):

    @staticmethod
    def _parse_year_text(value: str | None) -> int | None:
        if not value:
            return None
        match = re.search(r"(\d{4})", str(value))
        return int(match.group(1)) if match else None

    @staticmethod
    def _parse_period_text(value: str | None) -> tuple[int | None, int | None, int | None]:
        if not value:
            return None, None, None
        text = str(value)
        year_match = re.search(r"(\d{4})", text)
        week_match = re.search(r"(\d{1,2})\s*주", text)
        month_match = re.search(r"(\d{1,2})\s*월", text)
        year = int(year_match.group(1)) if year_match else None
        week_no = int(week_match.group(1)) if week_match else None
        month_no = int(month_match.group(1)) if month_match else None
        return year, week_no, month_no

    @staticmethod
    def _normalize_gender(value: str | None) -> str:
        mapping = {
            "계": "T",
            "총계": "T",
            "남": "M",
            "남자": "M",
            "남성": "M",
            "여": "F",
            "여자": "F",
            "여성": "F",
            "M": "M",
            "F": "F",
            "T": "T",
        }
        raw = str(value or "").strip()
        return mapping.get(raw, raw)

    @staticmethod
    def _normalize_disease_group(value: str | None) -> str:
        raw = str(value or "").strip().replace(" ", "")
        match = re.fullmatch(r"(?:제)?([1-4])급", raw)
        if match:
            return f"제{match.group(1)}급"
        return raw

    @staticmethod
    def _normalize_disease_name(value: str | None) -> str:
        raw = str(value or "").strip()
        if raw.startswith("@"):
            raw = raw[1:]
        return raw

    @classmethod
    def _disease_fields(cls, item: dict) -> tuple[str, str, str]:
        disease_nm = cls._normalize_disease_name(item.get("icdNm"))
        group_nm = cls._normalize_disease_group(item.get("icdGroupNm"))
        disease_cd = f"{group_nm}:{disease_nm}" if group_nm and disease_nm else disease_nm or group_nm
        return disease_cd, group_nm, disease_nm

    def _kdca_items(self, endpoint: str, params: dict) -> tuple[list[dict], int]:
        data = self.get(f"{KDCA_BASE}/{endpoint}", params=params)
        if not data:
            return [], 0

        body = data.get("response", {}).get("body", {})
        items = body.get("items", {})
        item_list = items.get("item", []) if isinstance(items, dict) else items or []
        if isinstance(item_list, dict):
            item_list = [item_list]
        total_count = self.safe_int(body.get("totalCount")) or len(item_list)
        return item_list, total_count

    @staticmethod
    def _purge_rows(table: str, where_sql: str, params: tuple):
        with get_conn() as conn:
            conn.execute(f"DELETE FROM {table} WHERE {where_sql}", params)

    # ── D1: 기간별 전국 발생현황 ─────────────────────────────────────────────
    def collect_national_disease(self, start_year: int = None,
                                  end_year: int = None,
                                  period_type: int = 3) -> int:
        """
        KDCA PeriodBasic → weekly_disease
        period_type: 1=연도별, 2=월별, 3=주별 (기본: 주별)
        """
        t0 = time.time()
        now_year = datetime.now().year
        sy = start_year or (now_year - 2)
        ey = end_year   or now_year

        source_type_map = {
            1: "yearly_national",
            2: "monthly_national",
            3: "weekly_national",
        }
        prefix_map = {
            1: "D1Y",
            2: "D6",
            3: "D1",
        }
        collection_map = {
            1: "KDCA_PeriodBasic_Yearly",
            2: "KDCA_PeriodBasic_Monthly",
            3: "KDCA_PeriodBasic_Weekly",
        }
        csv_suffix_map = {
            1: f"{sy}_{ey}_yearly",
            2: f"{sy}_{ey}_monthly",
            3: f"{sy}_{ey}_weekly",
        }

        all_rows = []
        page_size = 100
        source_type = source_type_map.get(period_type, "weekly_national")
        log_prefix = prefix_map.get(period_type, "D1")

        for year in range(sy, ey + 1):
            page = 1
            while True:
                params = {
                    "serviceKey":       KEYS["data_go_kr"],
                    "resType":          "2",
                    "searchPeriodType": str(period_type),
                    "searchStartYear":  str(year),
                    "searchEndYear":    str(year),
                    "pageNo":           str(page),
                    "numOfRows":        str(page_size),
                }
                item_list, total_cnt = self._kdca_items("PeriodBasic", params)
                if not item_list:
                    if page == 1:
                        log.warning(f"  [{log_prefix}] {year}: 응답 없음")
                    break

                try:
                    if item_list and page == 1 and year == sy:
                        it0 = item_list[0]
                        log.info(f"  [{log_prefix}] API 응답 키: {list(it0.keys())}")
                        log.info(f"  [{log_prefix}] 첫 행 전체: {it0}")

                    for it in item_list:
                        yr, wk, mn = self._parse_period_text(it.get("period"))
                        dis_cd, dis_group, dis_nm = self._disease_fields(it)
                        if not dis_nm:
                            continue
                        all_rows.append({
                            "collected_at": self.now_iso(),
                            "year":         yr or year,
                            "week_no":      wk if period_type == 3 else None,
                            "month_no":     mn if period_type == 2 else None,
                            "source_type":  source_type,
                            "disease_group": dis_group,
                            "disease_cd":   dis_cd,
                            "disease_nm":   dis_nm,
                            "cases":        self.safe_int(it.get("resultVal")),
                            "deaths":       None,
                            "sido_cd":      "00",
                            "sido_nm":      "전국",
                        })

                    if page * page_size >= int(total_cnt or 0):
                        break
                    page += 1
                    time.sleep(0.4)
                except Exception as e:
                    log.error(f"  [{log_prefix}] parse error ({year}, page {page}): {e}")
                    break

            time.sleep(0.3)

        if all_rows:
            self._purge_rows(
                "weekly_disease",
                "year BETWEEN ? AND ? AND source_type = ?",
                (sy, ey, source_type),
            )

        n = insert_rows("weekly_disease", all_rows)
        save_csv("weekly_disease", all_rows, date_str=csv_suffix_map[period_type], overwrite=True)
        log_collection("D", collection_map[period_type], "OK", n,
                       elapsed=time.time()-t0,
                       error=(None if all_rows else "no rows returned"))
        if period_type == 3:
            log.info(f"  [D1] weekly_disease (전국): {n}건 저장")
        elif period_type == 2:
            log.info(f"  [D6] disease_monthly (월별): {n}건 저장 ({sy}~{ey})")
        else:
            log.info(f"  [D1Y] disease_yearly (연별): {n}건 저장 ({sy}~{ey})")
        return n

    # ── D2: 서울 지역별 발생현황 ─────────────────────────────────────────────
    def collect_seoul_disease(self, start_year: int = None,
                                end_year: int = None) -> int:
        """KDCA Region (서울 01) → weekly_disease (연간 지역 집계)"""
        t0 = time.time()
        now_year = datetime.now().year
        sy = start_year or (now_year - 2)
        ey = end_year   or now_year

        all_rows = []
        for year in range(sy, ey + 1):
            params = {
                "serviceKey":   KEYS["data_go_kr"],
                "resType":      "2",
                "searchType":   "1",
                "searchYear":   str(year),
                "searchSidoCd": "01",
                "pageNo":       "1",
                "numOfRows":    "200",
            }
            item_list, _ = self._kdca_items("Region", params)
            if not item_list:
                continue

            try:
                if item_list:
                    log.info(f"  [D2] {year} API 응답 키: {list(item_list[0].keys())}")

                for it in item_list:
                    if str(it.get("sidoCd", "")) != "01":
                        continue
                    dis_cd, dis_group, dis_nm = self._disease_fields(it)
                    if not dis_nm:
                        continue
                    all_rows.append({
                        "collected_at": self.now_iso(),
                        "year":         self._parse_year_text(it.get("year")) or year,
                        "week_no":      None,
                        "month_no":     None,
                        "source_type":  "yearly_seoul",
                        "disease_group": dis_group,
                        "disease_cd":   dis_cd,
                        "disease_nm":   dis_nm,
                        "cases":        self.safe_int(it.get("resultVal")),
                        "deaths":       None,
                        "sido_cd":      "01",
                        "sido_nm":      str(it.get("sidoNm", "") or "서울"),
                    })
            except Exception as e:
                log.error(f"  [D2] {year} parse error: {e}")
            time.sleep(0.5)

        if all_rows:
            self._purge_rows(
                "weekly_disease",
                "year BETWEEN ? AND ? AND source_type = ?",
                (sy, ey, "yearly_seoul"),
            )

        n = insert_rows("weekly_disease", all_rows)
        save_csv("weekly_disease", all_rows, date_str=f"{sy}_{ey}_seoul",
                 overwrite=True)
        log_collection("D", "KDCA_Region_Seoul", "OK", n,
                       elapsed=time.time() - t0,
                       error=(None if all_rows else "no rows returned"))
        log.info(f"  [D2] weekly_disease (서울): {n}건 저장")
        return n

    # ── D3: 연령별 감염병 발생 ───────────────────────────────────────────────
    def collect_disease_age(self, start_year: int = None,
                              end_year: int = None) -> int:
        """
        KDCA Age → disease_age
        searchType=5: 5세 단위 연령대
        """
        t0 = time.time()
        now_year = datetime.now().year
        sy = start_year or (now_year - 2)
        ey = end_year   or now_year

        all_rows = []
        for year in range(sy, ey + 1):
            page = 1
            while True:
                items, total_cnt = self._kdca_items("Age", {
                    "serviceKey": KEYS["data_go_kr"],
                    "resType":    "2",
                    "searchType": "5",
                    "searchYear": str(year),
                    "pageNo":     str(page),
                    "numOfRows":  "200",
                })
                if not items:
                    if page == 1:
                        log.warning(f"  [D3] {year}: 응답 없음")
                    break

                for it in items:
                    dis_cd, dis_group, dis_nm = self._disease_fields(it)
                    if not dis_nm:
                        continue
                    all_rows.append({
                        "collected_at": self.now_iso(),
                        "year":         self._parse_year_text(it.get("year")) or year,
                        "disease_group": dis_group,
                        "disease_cd":   dis_cd,
                        "disease_nm":   dis_nm,
                        "age_group":    str(it.get("ageRange", "") or "").strip(),
                        "cases":        self.safe_int(it.get("resultVal")),
                    })

                if page * 200 >= total_cnt:
                    break
                page += 1
                time.sleep(0.3)
            time.sleep(0.4)

        if all_rows:
            self._purge_rows("disease_age", "year BETWEEN ? AND ?", (sy, ey))

        n = insert_rows("disease_age", all_rows)
        save_csv("disease_age", all_rows, date_str=f"{sy}_{ey}",
                 overwrite=True)
        status = "OK" if all_rows else "FAIL"
        log_collection("D", "KDCA_Age", status, n, elapsed=time.time()-t0)
        log.info(f"  [D3] disease_age: {n}건 저장 ({sy}~{ey})")
        return n

    # ── D4: 성별 감염병 발생 ─────────────────────────────────────────────────
    def collect_disease_gender(self, start_year: int = None,
                                 end_year: int = None) -> int:
        """
        KDCA Gender → disease_gender
        """
        t0 = time.time()
        now_year = datetime.now().year
        sy = start_year or (now_year - 2)
        ey = end_year   or now_year

        all_rows = []
        for year in range(sy, ey + 1):
            page = 1
            while True:
                items, total_cnt = self._kdca_items("Gender", {
                    "serviceKey": KEYS["data_go_kr"],
                    "resType":    "2",
                    "searchType": "1",
                    "searchYear": str(year),
                    "pageNo":     str(page),
                    "numOfRows":  "200",
                })
                if not items:
                    if page == 1:
                        log.warning(f"  [D4] {year}: 응답 없음")
                    break

                for it in items:
                    dis_cd, dis_group, dis_nm = self._disease_fields(it)
                    if not dis_nm:
                        continue
                    all_rows.append({
                        "collected_at": self.now_iso(),
                        "year":         self._parse_year_text(it.get("year")) or year,
                        "disease_group": dis_group,
                        "disease_cd":   dis_cd,
                        "disease_nm":   dis_nm,
                        "gender":       self._normalize_gender(it.get("sex")),
                        "cases":        self.safe_int(it.get("resultVal")),
                    })

                if page * 200 >= total_cnt:
                    break
                page += 1
                time.sleep(0.3)
            time.sleep(0.4)

        if all_rows:
            self._purge_rows("disease_gender", "year BETWEEN ? AND ?", (sy, ey))

        n = insert_rows("disease_gender", all_rows)
        save_csv("disease_gender", all_rows, date_str=f"{sy}_{ey}",
                 overwrite=True)
        status = "OK" if all_rows else "FAIL"
        log_collection("D", "KDCA_Gender", status, n, elapsed=time.time()-t0)
        log.info(f"  [D4] disease_gender: {n}건 저장 ({sy}~{ey})")
        return n

    # ── D5: 감염병별 사망자 ──────────────────────────────────────────────────
    def collect_disease_death(self, start_year: int = None,
                                end_year: int = None) -> int:
        """
        KDCA death → disease_death
        """
        t0 = time.time()
        now_year = datetime.now().year
        sy = start_year or (now_year - 2)
        ey = end_year   or now_year

        all_rows = []
        for year in range(sy, ey + 1):
            page = 1
            while True:
                items, total_cnt = self._kdca_items("death", {
                    "serviceKey":      KEYS["data_go_kr"],
                    "resType":         "2",
                    "searchStartYear": str(year),
                    "searchEndYear":   str(year),
                    "pageNo":          str(page),
                    "numOfRows":       "200",
                })
                if not items:
                    if page == 1:
                        log.warning(f"  [D5] {year}: 응답 없음")
                    break

                for it in items:
                    dis_cd, dis_group, dis_nm = self._disease_fields(it)
                    if not dis_nm:
                        continue
                    all_rows.append({
                        "collected_at": self.now_iso(),
                        "year":         self._parse_year_text(it.get("year")) or year,
                        "disease_group": dis_group,
                        "disease_cd":   dis_cd,
                        "disease_nm":   dis_nm,
                        "deaths":       self.safe_int(it.get("resultVal")),
                    })

                if page * 200 >= total_cnt:
                    break
                page += 1
                time.sleep(0.3)
            time.sleep(0.4)

        if all_rows:
            self._purge_rows("disease_death", "year BETWEEN ? AND ?", (sy, ey))

        n = insert_rows("disease_death", all_rows)
        save_csv("disease_death", all_rows, date_str=f"{sy}_{ey}",
                 overwrite=True)
        status = "OK" if all_rows else "FAIL"
        log_collection("D", "KDCA_death", status, n, elapsed=time.time()-t0)
        log.info(f"  [D5] disease_death: {n}건 저장 ({sy}~{ey})")
        return n

    # ── D6: 월별 감염병 신고현황 (2020~) ─────────────────────────────────────
    def collect_disease_monthly(self, start_year: int = None,
                                  end_year: int = None) -> int:
        """
        KDCA PeriodBasic(searchPeriodType=2) → weekly_disease (month_no 기준)
        """
        return self.collect_national_disease(
            start_year=start_year,
            end_year=end_year,
            period_type=2,
        )

    def run(self, start_year: int = None, end_year: int = None,
            skip_apis: list = None) -> dict:
        """Group D 실행. 연도 범위 지정 가능 (기본: 최근 3년)"""
        skip_apis = skip_apis or []
        log.info("▶ Group D -- 주간 감염병 수집 시작")
        r1 = r2 = r3 = r4 = r5 = r6 = 0

        if "D1" not in skip_apis:
            try:
                r1 = self.collect_national_disease(start_year=start_year,
                                                   end_year=end_year)
            except Exception as e:
                log.error(f"  [D1] collect_national_disease 예외 (스킵): {e}")

        if "D2" not in skip_apis:
            try:
                r2 = self.collect_seoul_disease(start_year=start_year,
                                                end_year=end_year)
            except Exception as e:
                log.error(f"  [D2] collect_seoul_disease 예외 (스킵): {e}")

        if "D3" not in skip_apis:
            try:
                r3 = self.collect_disease_age(start_year=start_year,
                                              end_year=end_year)
            except Exception as e:
                log.error(f"  [D3] collect_disease_age 예외 (스킵): {e}")

        if "D4" not in skip_apis:
            try:
                r4 = self.collect_disease_gender(start_year=start_year,
                                                 end_year=end_year)
            except Exception as e:
                log.error(f"  [D4] collect_disease_gender 예외 (스킵): {e}")

        if "D5" not in skip_apis:
            try:
                r5 = self.collect_disease_death(start_year=start_year,
                                                end_year=end_year)
            except Exception as e:
                log.error(f"  [D5] collect_disease_death 예외 (스킵): {e}")

        if "D6" not in skip_apis:
            try:
                r6 = self.collect_disease_monthly(start_year=start_year,
                                                  end_year=end_year)
            except Exception as e:
                log.error(f"  [D6] collect_disease_monthly 예외 (스킵): {e}")

        try:
            n_catalog = refresh_disease_catalog()
            log.info(f"  [D-CATALOG] disease_catalog: {n_catalog}건 갱신")
        except Exception as e:
            log.warning(f"  [D-CATALOG] catalog refresh 실패: {e}")

        return {
            "weekly_disease_national": r1,
            "weekly_disease_seoul":    r2,
            "disease_age":             r3,
            "disease_gender":          r4,
            "disease_death":           r5,
            "disease_monthly":         r6,
        }
