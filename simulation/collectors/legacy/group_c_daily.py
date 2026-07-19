"""
pipeline/collectors/group_c_daily.py
======================================
Group C -- 일별 배치
  C1. citydata_ppltn (OA-21778) → daily_population_hotspot  (활성)
      SeoulRtd 공식 등록 115개 POI 실시간 인구 일별 스냅샷 (자치구 매핑 포함)
  C2. CardSubwayStatsNew              → daily_subway              (활성, D-3)
  C3. CardBusStatisticsServiceNew     → daily_bus                 (활성, D-3)
      서울 열린데이터광장 OA-12912, 노선×정류장별 일별 승하차
  C4. SPOP_DAILYSUM_JACHI → daily_population_district  (활성, 2026-03-22 재활성화)
      자치구 일별 생활인구 집계 (주간/야간/유입/이동)
  C5. SPOP_LOCAL_RESD_JACHI → daily_population_gu_hourly  (활성, 2026-03-22 신규)
      자치구 시간대별(0~23) 생활인구 (성별/연령 포함)
  C6. SPOP_LOCAL_RESD_DONG → daily_population_dong  (활성, 2026-03-22 신규)
      행정동별 시간대별 생활인구 → 일평균 집계 후 저장
"""

import json
import time
import logging
import urllib.parse
from datetime import datetime, timedelta
from .base import BaseCollector
from ..config import KEYS, SEOUL_BASE
from ..storage import insert_rows, save_csv, log_collection, get_conn

log = logging.getLogger(__name__)

# ── citydata_ppltn 공식 POI 목록 (SeoulRtd 등록 115개) ───────────────────────
# 서울 도시데이터 서비스 (https://data.seoul.go.kr/dataList/OA-21778)
# 형식: (AREA_NM, GU_CODE_4자리, GU_NM)
# ⚠️ AREA_NM은 SeoulRtd /api/hotspot-category에서 가져온 공식 등록명과 정확히 일치
# ⚠️ 임의로 추가/수정 금지 -- API 미등록 POI는 ERROR-500 반환
CITYDATA_POIS: list[tuple[str, str, str]] = [
    # ── 종로구 ──────────────────────────────────────────────────────────────
    ("경복궁",                            "1101", "종로구"),
    ("광장(전통)시장",                    "1101", "종로구"),
    ("광화문·덕수궁",                     "1101", "종로구"),
    ("광화문광장",                        "1101", "종로구"),
    ("보신각",                            "1101", "종로구"),
    ("북촌한옥마을",                      "1101", "종로구"),
    ("송현녹지광장",                      "1101", "종로구"),
    ("시의회 앞",                         "1101", "종로구"),
    ("익선동",                            "1101", "종로구"),
    ("인사동",                            "1101", "종로구"),
    ("종로·청계 관광특구",                "1101", "종로구"),
    ("창덕궁·종묘",                       "1101", "종로구"),
    ("혜화역",                            "1101", "종로구"),

    # ── 중구 ────────────────────────────────────────────────────────────────
    ("DDP(동대문디자인플라자)",            "1102", "중구"),
    ("남대문시장",                        "1102", "중구"),
    ("남산공원",                          "1102", "중구"),
    ("덕수궁길·정동길",                   "1102", "중구"),
    ("동대문 관광특구",                   "1102", "중구"),
    ("동대문역",                          "1102", "중구"),
    ("명동 관광특구",                     "1102", "중구"),
    ("북창동 먹자골목",                   "1102", "중구"),
    ("서울광장",                          "1102", "중구"),
    ("서울역",                            "1102", "중구"),
    ("충정로역",                          "1102", "중구"),

    # ── 용산구 ──────────────────────────────────────────────────────────────
    ("삼각지역",                          "1103", "용산구"),
    ("용리단길",                          "1103", "용산구"),
    ("용산역",                            "1103", "용산구"),
    ("이촌한강공원",                      "1103", "용산구"),
    ("이태원 관광특구",                   "1103", "용산구"),
    ("이태원 앤틱가구거리",               "1103", "용산구"),
    ("이태원역",                          "1103", "용산구"),
    ("해방촌·경리단길",                   "1103", "용산구"),

    # ── 성동구 ──────────────────────────────────────────────────────────────
    ("뚝섬역",                            "1104", "성동구"),
    ("서울숲공원",                        "1104", "성동구"),
    ("성수카페거리",                      "1104", "성동구"),
    ("왕십리역",                          "1104", "성동구"),
    ("응봉산",                            "1104", "성동구"),

    # ── 광진구 ──────────────────────────────────────────────────────────────
    ("건대입구역",                        "1105", "광진구"),
    ("광나루한강공원",                    "1105", "광진구"),
    ("군자역",                            "1105", "광진구"),
    ("뚝섬한강공원",                      "1105", "광진구"),
    ("아차산",                            "1105", "광진구"),
    ("어린이대공원",                      "1105", "광진구"),

    # ── 동대문구 ────────────────────────────────────────────────────────────
    ("장한평역",                          "1106", "동대문구"),
    ("청량리 제기동 일대 전통시장",        "1106", "동대문구"),
    ("회기역",                            "1106", "동대문구"),

    # ── 성북구 ──────────────────────────────────────────────────────────────
    ("성신여대입구역",                    "1108", "성북구"),

    # ── 강북구 ──────────────────────────────────────────────────────────────
    ("미아사거리역",                      "1109", "강북구"),
    ("북서울꿈의숲",                      "1109", "강북구"),
    ("수유역",                            "1109", "강북구"),

    # ── 도봉구 ──────────────────────────────────────────────────────────────
    ("쌍문역",                            "1110", "도봉구"),
    ("창동 신경제 중심지",                "1110", "도봉구"),

    # ── 은평구 ──────────────────────────────────────────────────────────────
    ("연신내역",                          "1112", "은평구"),

    # ── 서대문구 ────────────────────────────────────────────────────────────
    ("서대문독립공원",                    "1113", "서대문구"),
    ("신촌 스타광장",                     "1113", "서대문구"),
    ("신촌·이대역",                       "1113", "서대문구"),

    # ── 마포구 ──────────────────────────────────────────────────────────────
    ("DMC(디지털미디어시티)",             "1114", "마포구"),
    ("난지한강공원",                      "1114", "마포구"),
    ("망원한강공원",                      "1114", "마포구"),
    ("양화한강공원",                      "1114", "마포구"),
    ("연남동",                            "1114", "마포구"),
    ("월드컵공원",                        "1114", "마포구"),
    ("합정역",                            "1114", "마포구"),
    ("홍대 관광특구",                     "1114", "마포구"),
    ("홍대입구역(2호선)",                 "1114", "마포구"),

    # ── 양천구 ──────────────────────────────────────────────────────────────
    ("신정네거리역",                      "1115", "양천구"),
    ("안양천",                            "1115", "양천구"),
    ("오목교역·목동운동장",               "1115", "양천구"),

    # ── 강서구 ──────────────────────────────────────────────────────────────
    ("강서한강공원",                      "1116", "강서구"),
    ("김포공항",                          "1116", "강서구"),
    ("발산역",                            "1116", "강서구"),
    ("서울식물원·마곡나루역",             "1116", "강서구"),

    # ── 구로구 ──────────────────────────────────────────────────────────────
    ("고척돔",                            "1117", "구로구"),
    ("구로디지털단지역",                  "1117", "구로구"),
    ("구로역",                            "1117", "구로구"),
    ("대림역",                            "1117", "구로구"),
    ("신도림역",                          "1117", "구로구"),

    # ── 금천구 ──────────────────────────────────────────────────────────────
    ("가산디지털단지역",                  "1118", "금천구"),

    # ── 영등포구 ────────────────────────────────────────────────────────────
    ("여의도",                            "1119", "영등포구"),
    ("여의도한강공원",                    "1119", "영등포구"),
    ("영등포 타임스퀘어",                 "1119", "영등포구"),

    # ── 동작구 ──────────────────────────────────────────────────────────────
    ("노들섬",                            "1120", "동작구"),
    ("노량진",                            "1120", "동작구"),
    ("보라매공원",                        "1120", "동작구"),
    ("사당역",                            "1120", "동작구"),
    ("총신대입구(이수)역",                "1120", "동작구"),

    # ── 관악구 ──────────────────────────────────────────────────────────────
    ("서울대입구역",                      "1121", "관악구"),
    ("신림역",                            "1121", "관악구"),

    # ── 서초구 ──────────────────────────────────────────────────────────────
    ("고속터미널역",                      "1122", "서초구"),
    ("교대역",                            "1122", "서초구"),
    ("반포한강공원",                      "1122", "서초구"),
    ("서리풀공원·몽마르뜨공원",           "1122", "서초구"),
    ("양재역",                            "1122", "서초구"),
    ("잠원한강공원",                      "1122", "서초구"),
    ("청계산",                            "1122", "서초구"),

    # ── 강남구 ──────────────────────────────────────────────────────────────
    ("가로수길",                          "1123", "강남구"),
    ("강남 MICE 관광특구",                "1123", "강남구"),
    ("강남역",                            "1123", "강남구"),
    ("선릉역",                            "1123", "강남구"),
    ("신논현역·논현역",                   "1123", "강남구"),
    ("압구정로데오거리",                  "1123", "강남구"),
    ("역삼역",                            "1123", "강남구"),
    ("청담동 명품거리",                   "1123", "강남구"),

    # ── 송파구 ──────────────────────────────────────────────────────────────
    ("가락시장",                          "1124", "송파구"),
    ("송리단길·호수단길",                 "1124", "송파구"),
    ("올림픽공원",                        "1124", "송파구"),
    ("잠실 관광특구",                     "1124", "송파구"),
    ("잠실롯데타워 일대",                 "1124", "송파구"),
    ("잠실새내역",                        "1124", "송파구"),
    ("잠실역",                            "1124", "송파구"),
    ("잠실종합운동장",                    "1124", "송파구"),
    ("잠실한강공원",                      "1124", "송파구"),
    ("장지역",                            "1124", "송파구"),

    # ── 강동구 ──────────────────────────────────────────────────────────────
    ("고덕역",                            "1125", "강동구"),
    ("천호역",                            "1125", "강동구"),
]

# GU_NM 빠른 조회용 역방향 맵
_GU_CODE_MAP = {nm: (cd, nm) for _, cd, nm in CITYDATA_POIS}

# 자치구 5자리 코드 → 구 이름 매핑 (SPOP API용)
_GU_CODE_5_MAP = {
    "11110": "종로구",   "11140": "중구",     "11170": "용산구",
    "11200": "성동구",   "11215": "광진구",   "11230": "동대문구",
    "11260": "중랑구",   "11290": "성북구",   "11305": "강북구",
    "11320": "도봉구",   "11350": "노원구",   "11380": "은평구",
    "11410": "서대문구", "11440": "마포구",   "11470": "양천구",
    "11500": "강서구",   "11530": "구로구",   "11545": "금천구",
    "11560": "영등포구", "11590": "동작구",   "11620": "관악구",
    "11650": "서초구",   "11680": "강남구",   "11710": "송파구",
    "11740": "강동구",
}


class GroupCCollector(BaseCollector):

    # ── C1: citydata_ppltn 일별 스냅샷 (SPOP 대체) ──────────────────────────
    def collect_hotspot_population(self) -> int:
        """
        서울 citydata_ppltn (OA-21778) → daily_population_hotspot

        SPOP_DAILYSUM_JACHI 서비스 중단(2018 데이터 고착) 대체.
        SeoulRtd 공식 등록 115개 POI의 실시간 인구를 수집해 일별 스냅샷으로 저장.
        자치구 코드/명칭은 CITYDATA_POIS 매핑 테이블 기준으로 부여.

        API 형식:
          GET {SEOUL_BASE}/{KEY}/json/citydata_ppltn/1/5/{AREA_NM(URL인코딩)}
        응답 최상위 키: SeoulRtd.citydata_ppltn  (리스트, 보통 1개)
        """
        t0 = time.time()
        today = datetime.now().strftime("%Y%m%d")
        rows = []
        fail_cnt = 0

        failed_pois = []

        for area_nm, gu_code, gu_nm in CITYDATA_POIS:
            encoded = urllib.parse.quote(area_nm)
            url = (f"{SEOUL_BASE}/{KEYS['seoul_general']}"
                   f"/json/citydata_ppltn/1/5/{encoded}")
            data = self.get(url)
            if not data:
                fail_cnt += 1
                failed_pois.append(area_nm)
                time.sleep(0.2)
                continue

            try:
                items = data.get("SeoulRtd.citydata_ppltn", [])
                if not items:
                    fail_cnt += 1
                    failed_pois.append(f"{area_nm}(빈결과)")
                    time.sleep(0.2)
                    continue

                item = items[0]
                rows.append({
                    "stdr_de":              today,
                    "area_cd":              item.get("AREA_CD", ""),
                    "area_nm":              item.get("AREA_NM", area_nm),
                    "gu_code":              gu_code,
                    "gu_nm":               gu_nm,
                    "congestion":          item.get("AREA_CONGEST_LVL", ""),
                    "ppltn_min":           self.safe_int(item.get("AREA_PPLTN_MIN")),
                    "ppltn_max":           self.safe_int(item.get("AREA_PPLTN_MAX")),
                    "ppltn_rate_0":        self.safe_float(item.get("PPLTN_RATE_0")),
                    "ppltn_rate_10":       self.safe_float(item.get("PPLTN_RATE_10")),
                    "ppltn_rate_20":       self.safe_float(item.get("PPLTN_RATE_20")),
                    "ppltn_rate_30":       self.safe_float(item.get("PPLTN_RATE_30")),
                    "ppltn_rate_40":       self.safe_float(item.get("PPLTN_RATE_40")),
                    "ppltn_rate_50":       self.safe_float(item.get("PPLTN_RATE_50")),
                    "ppltn_rate_60":       self.safe_float(item.get("PPLTN_RATE_60")),
                    "ppltn_rate_70":       self.safe_float(item.get("PPLTN_RATE_70")),
                    "male_ppltn_rate":     self.safe_float(item.get("MALE_PPLTN_RATE")),
                    "female_ppltn_rate":   self.safe_float(item.get("FEMALE_PPLTN_RATE")),
                    "resnt_ppltn_rate":    self.safe_float(item.get("RESNT_PPLTN_RATE")),
                    "non_resnt_ppltn_rate":self.safe_float(item.get("NON_RESNT_PPLTN_RATE")),
                    "raw_json":            json.dumps(item, ensure_ascii=False)[:800],
                })
            except Exception as e:
                log.warning(f"  [C1] parse error ({area_nm}): {e}")
                fail_cnt += 1
                failed_pois.append(f"{area_nm}(파싱오류)")

            time.sleep(0.3)   # 과도한 요청 방지

        n = insert_rows("daily_population_hotspot", rows)
        save_csv("daily_population_hotspot", rows, date_str=today)
        status = "OK" if n > 0 else "FAIL"
        log.info(f"  [C1] daily_population_hotspot: {n}건 저장 "
                 f"(성공 {len(rows)}, 실패 {fail_cnt}, 기준일 {today})")
        if failed_pois:
            log.info(f"  [C1] 실패 POI 목록 ({len(failed_pois)}개): "
                     f"{', '.join(failed_pois)}")
            log.info("  [C1] 실패 POI는 citydata_ppltn 서비스에 등록되지 않은 "
                     "지역명입니다. 다음 실행 시 로그 확인 후 POI 목록 수정 권장.")
        log_collection("C", "citydata_ppltn_daily", status,
                        n, elapsed=time.time()-t0)
        return n

    # ── C4: SPOP_DAILYSUM_JACHI → daily_population_district ──────────────
    def collect_district_population(self, target_date: str = None,
                                     days_back: int = 30,
                                     force: bool = False) -> int:
        """
        SPOP_DAILYSUM_JACHI → daily_population_district
        자치구 일별 생활인구 집계 (주간/야간/유입/이동).

        API 필드명 (2026-03 검증 완료):
          STDR_DE_ID        기준일 YYYYMMDD
          SIGNGU_CODE_SE    시군구코드 5자리 (11110~11740)
          SIGNGU_NM         자치구명
          TOT_LVPOP_CO      총 생활인구
          DAY_LVPOP_CO      주간(09~18) 생활인구
          NIGHT_LVPOP_CO    야간(18~09) 생활인구
          SU_ELSE_INFLOW_LVPOP_CO  타시도 유입 생활인구
          SIGNGU_MVMN_LVPOP_CO     자치구 간 이동 인구
        """
        t0 = time.time()
        total = 0
        base = datetime.now()

        if target_date:
            dates = [target_date]
        else:
            dates = [(base - timedelta(days=i)).strftime("%Y%m%d")
                     for i in range(1, days_back + 1)]

        for dt in dates:
            # 기존재 스킵 (force=True 면 우회 — 기존 행 덮어쓰지는 않으나
            # 신규로 추가될 row 가 있다면 insert_rows INSERT OR IGNORE 로 처리).
            if not force:
                conn = get_conn()
                existing = conn.execute(
                    "SELECT COUNT(*) FROM daily_population_district WHERE stdr_de=?", (dt,)
                ).fetchone()[0]
                conn.close()
                if existing >= 25:
                    continue

            url = (f"{SEOUL_BASE}/{KEYS['seoul_general2']}"
                   f"/json/SPOP_DAILYSUM_JACHI/1/100/{dt}")
            data = self.get(url)
            if not data:
                continue

            items = (data.get("SPOP_DAILYSUM_JACHI", {}).get("row")
                     or [])
            if not items:
                result_code = data.get("RESULT", {}).get("CODE", "")
                if result_code == "INFO-200":
                    continue  # 미게재
                log.warning(f"  [C4] {dt}: 빈 응답 (keys={list(data.keys())[:5]})")
                continue

            rows = []
            for r in items:
                signgu_code = r.get("SIGNGU_CODE_SE", "")
                if not signgu_code or not signgu_code.startswith("11"):
                    continue
                rows.append({
                    "stdr_de":       r.get("STDR_DE_ID", dt),
                    "signgu_code":   signgu_code,
                    "signgu_nm":     r.get("SIGNGU_NM", ""),
                    "tot_livpop":    self.safe_float(r.get("TOT_LVPOP_CO")),
                    "day_livpop":    self.safe_float(r.get("DAY_LVPOP_CO")),
                    "night_livpop":  self.safe_float(r.get("NIGHT_LVPOP_CO")),
                    "inflow_livpop": self.safe_float(r.get("SU_ELSE_INFLOW_LVPOP_CO")),
                    "move_livpop":   self.safe_float(r.get("SIGNGU_MVMN_LVPOP_CO")),
                })

            n = insert_rows("daily_population_district", rows)
            total += n
            if n > 0:
                log.info(f"  [C4] district {dt}: {n}건")
            time.sleep(0.3)

        # Status semantics (2026-04-25): API call succeeded → OK regardless
        # of n_rows. `rows_saved=0` is the "no new data in window" signal,
        # not failure. Previously SKIP polluted the audit because Seoul Open
        # API publishes today's data tomorrow → backfill_days=1 always 0.
        log_collection("C", "SPOP_DAILYSUM_JACHI", "OK", total,
                       elapsed=time.time()-t0,
                       error=(None if total > 0 else "no new data in window"))
        return total

    # ── C5: SPOP_LOCAL_RESD_JACHI → daily_population_gu_hourly ─────────
    def collect_gu_hourly_population(self, target_date: str = None,
                                      days_back: int = 30,
                                      force: bool = False) -> int:
        """
        SPOP_LOCAL_RESD_JACHI → daily_population_gu_hourly
        자치구 시간대별(0~23) 생활인구 (성별/연령 포함).

        API 필드명 (2026-03 검증 완료):
          STDR_DE_ID         기준일 YYYYMMDD
          TMZON_PD_SE        시간대 0~23
          ADSTRD_CODE_SE     자치구코드 5자리 (11110~11740)  ⚠️ NOT SIGNGU_CODE_SE
          MALE_F0T4_LVPOP_CO ~ MALE_F70T74_LVPOP_CO   남성 연령별
          FEMALE_F0T4_LVPOP_CO ~ FEMALE_F70T74_LVPOP_CO 여성 연령별
          ⚠️ 70+ 필드명은 F70T74 (NOT F70OVER)
        """
        t0 = time.time()
        total = 0
        base = datetime.now()

        if target_date:
            dates = [target_date]
        else:
            dates = [(base - timedelta(days=i)).strftime("%Y%m%d")
                     for i in range(1, days_back + 1)]

        for dt in dates:
            # 기존재 스킵 (25구 × 24시간 = 600행). force=True 면 우회.
            if not force:
                conn = get_conn()
                existing = conn.execute(
                    "SELECT COUNT(*) FROM daily_population_gu_hourly WHERE stdr_de=?", (dt,)
                ).fetchone()[0]
                conn.close()
                if existing >= 600:
                    continue

            # 1일치 = 25구 × 24시간 = 600행, 페이지네이션 필요
            all_items = []
            page_start = 1
            page_size = 1000
            while True:
                url = (f"{SEOUL_BASE}/{KEYS['seoul_general2']}"
                       f"/json/SPOP_LOCAL_RESD_JACHI/{page_start}/{page_start + page_size - 1}/{dt}")
                data = self.get(url)
                if not data:
                    break

                sect = data.get("SPOP_LOCAL_RESD_JACHI", {})
                items = sect.get("row") or []
                if not items:
                    result_code = data.get("RESULT", {}).get("CODE", "")
                    if result_code == "INFO-200":
                        break  # 미게재
                    break

                all_items.extend(items)
                total_count = self.safe_int(sect.get("list_total_count")) or 0
                if page_start + page_size - 1 >= total_count:
                    break
                page_start += page_size
                time.sleep(0.2)

            if not all_items:
                continue

            rows = []
            for r in all_items:
                gu_code = r.get("ADSTRD_CODE_SE", "")
                if not gu_code or not gu_code.startswith("11"):
                    continue
                hour = self.safe_int(r.get("TMZON_PD_SE"))
                if hour is None:
                    continue

                gu_nm = _GU_CODE_5_MAP.get(gu_code, "")

                def s(prefix_m, prefix_f, age_suffix):
                    m = self.safe_float(r.get(f"{prefix_m}{age_suffix}_LVPOP_CO")) or 0
                    f = self.safe_float(r.get(f"{prefix_f}{age_suffix}_LVPOP_CO")) or 0
                    return m, f, m + f

                m_0_4, f_0_4, _ = s("MALE_F", "FEMALE_F", "0T4")
                m_5_9, f_5_9, _ = s("MALE_F", "FEMALE_F", "5T9")
                m_10_14, f_10_14, _ = s("MALE_F", "FEMALE_F", "10T14")
                m_15_19, f_15_19, _ = s("MALE_F", "FEMALE_F", "15T19")
                m_20_24, f_20_24, _ = s("MALE_F", "FEMALE_F", "20T24")
                m_25_29, f_25_29, _ = s("MALE_F", "FEMALE_F", "25T29")
                m_30_34, f_30_34, _ = s("MALE_F", "FEMALE_F", "30T34")
                m_35_39, f_35_39, _ = s("MALE_F", "FEMALE_F", "35T39")
                m_40_44, f_40_44, _ = s("MALE_F", "FEMALE_F", "40T44")
                m_45_49, f_45_49, _ = s("MALE_F", "FEMALE_F", "45T49")
                m_50_54, f_50_54, _ = s("MALE_F", "FEMALE_F", "50T54")
                m_55_59, f_55_59, _ = s("MALE_F", "FEMALE_F", "55T59")
                m_60_64, f_60_64, _ = s("MALE_F", "FEMALE_F", "60T64")
                m_65_69, f_65_69, _ = s("MALE_F", "FEMALE_F", "65T69")
                m_70plus, f_70plus, _ = s("MALE_F", "FEMALE_F", "70T74")

                pop_0_9 = m_0_4 + f_0_4 + m_5_9 + f_5_9
                pop_10_19 = m_10_14 + f_10_14 + m_15_19 + f_15_19
                pop_20_29 = m_20_24 + f_20_24 + m_25_29 + f_25_29
                pop_30_39 = m_30_34 + f_30_34 + m_35_39 + f_35_39
                pop_40_49 = m_40_44 + f_40_44 + m_45_49 + f_45_49
                pop_50_59 = m_50_54 + f_50_54 + m_55_59 + f_55_59
                pop_60_69 = m_60_64 + f_60_64 + m_65_69 + f_65_69
                pop_70plus_total = m_70plus + f_70plus

                male_total = (m_0_4 + m_5_9 + m_10_14 + m_15_19 +
                              m_20_24 + m_25_29 + m_30_34 + m_35_39 +
                              m_40_44 + m_45_49 + m_50_54 + m_55_59 +
                              m_60_64 + m_65_69 + m_70plus)
                female_total = (f_0_4 + f_5_9 + f_10_14 + f_15_19 +
                                f_20_24 + f_25_29 + f_30_34 + f_35_39 +
                                f_40_44 + f_45_49 + f_50_54 + f_55_59 +
                                f_60_64 + f_65_69 + f_70plus)
                tot_pop = male_total + female_total

                rows.append({
                    "stdr_de":    r.get("STDR_DE_ID", dt),
                    "gu_code":    gu_code,
                    "gu_nm":      gu_nm,
                    "hour":       hour,
                    "tot_pop":    round(tot_pop, 2),
                    "male_pop":   round(male_total, 2),
                    "female_pop": round(female_total, 2),
                    "pop_0_9":    round(pop_0_9, 2),
                    "pop_10_19":  round(pop_10_19, 2),
                    "pop_20_29":  round(pop_20_29, 2),
                    "pop_30_39":  round(pop_30_39, 2),
                    "pop_40_49":  round(pop_40_49, 2),
                    "pop_50_59":  round(pop_50_59, 2),
                    "pop_60_69":  round(pop_60_69, 2),
                    "pop_70plus": round(pop_70plus_total, 2),
                })

            n = insert_rows("daily_population_gu_hourly", rows)
            total += n
            if n > 0:
                log.info(f"  [C5] gu_hourly {dt}: {n}건")
            time.sleep(0.3)

        log_collection("C", "SPOP_LOCAL_RESD_JACHI", "OK", total,
                       elapsed=time.time()-t0,
                       error=(None if total > 0 else "no new data in window"))
        return total

    # ── C6: SPOP_LOCAL_RESD_DONG → daily_population_dong ───────────────
    def collect_dong_population(self, target_date: str = None,
                                 days_back: int = 30,
                                 force: bool = False) -> int:
        """
        SPOP_LOCAL_RESD_DONG → daily_population_dong
        행정동별 시간대별 생활인구 → 일평균(24시간대 평균) 집계 후 저장.

        API 필드명 (2026-03 검증 완료):
          STDR_DE_ID         기준일 YYYYMMDD
          TMZON_PD_SE        시간대 0~23
          ADSTRD_CODE_SE     행정동코드 8자리  (앞 5자리 = 자치구코드)
          MALE_F0T4_LVPOP_CO ~ MALE_F70T74_LVPOP_CO
          FEMALE_F0T4_LVPOP_CO ~ FEMALE_F70T74_LVPOP_CO
        """
        t0 = time.time()
        total = 0
        base = datetime.now()

        if target_date:
            dates = [target_date]
        else:
            dates = [(base - timedelta(days=i)).strftime("%Y%m%d")
                     for i in range(1, days_back + 1)]

        for dt in dates:
            # 기존재 스킵 (~460 동). force=True 면 우회.
            if not force:
                conn = get_conn()
                existing = conn.execute(
                    "SELECT COUNT(*) FROM daily_population_dong WHERE stdr_de=?", (dt,)
                ).fetchone()[0]
                conn.close()
                if existing >= 400:
                    continue

            # 수집: 1일치 = ~460동 × 24시간 ≈ 11,000행, 페이지네이션 필수
            all_items = []
            page_start = 1
            page_size = 1000
            while True:
                url = (f"{SEOUL_BASE}/{KEYS['seoul_general2']}"
                       f"/json/SPOP_LOCAL_RESD_DONG/{page_start}/{page_start + page_size - 1}/{dt}")
                data = self.get(url)
                if not data:
                    break

                sect = data.get("SPOP_LOCAL_RESD_DONG", {})
                items = sect.get("row") or []
                if not items:
                    result_code = data.get("RESULT", {}).get("CODE", "")
                    if result_code == "INFO-200":
                        break
                    break

                all_items.extend(items)
                total_count = self.safe_int(sect.get("list_total_count")) or 0
                if page_start + page_size - 1 >= total_count:
                    break
                page_start += page_size
                time.sleep(0.2)

            if not all_items:
                continue

            # 시간대별 → 일평균 집계 (동별로 24시간대 평균)
            from collections import defaultdict
            dong_accum = defaultdict(lambda: {
                "count": 0, "tot": 0, "male": 0, "female": 0,
                "p0_9": 0, "p10_19": 0, "p20_29": 0, "p30_39": 0,
                "p40_49": 0, "p50_59": 0, "p60_69": 0, "p70plus": 0,
            })

            for r in all_items:
                dong_code = r.get("ADSTRD_CODE_SE", "")
                if not dong_code or len(dong_code) < 5:
                    continue

                acc = dong_accum[dong_code]
                acc["count"] += 1

                def sf(key):
                    return self.safe_float(r.get(key)) or 0

                # 연령 5세 → 10세 집계
                m_0_9 = sf("MALE_F0T4_LVPOP_CO") + sf("MALE_F5T9_LVPOP_CO")
                f_0_9 = sf("FEMALE_F0T4_LVPOP_CO") + sf("FEMALE_F5T9_LVPOP_CO")
                m_10_19 = sf("MALE_F10T14_LVPOP_CO") + sf("MALE_F15T19_LVPOP_CO")
                f_10_19 = sf("FEMALE_F10T14_LVPOP_CO") + sf("FEMALE_F15T19_LVPOP_CO")
                m_20_29 = sf("MALE_F20T24_LVPOP_CO") + sf("MALE_F25T29_LVPOP_CO")
                f_20_29 = sf("FEMALE_F20T24_LVPOP_CO") + sf("FEMALE_F25T29_LVPOP_CO")
                m_30_39 = sf("MALE_F30T34_LVPOP_CO") + sf("MALE_F35T39_LVPOP_CO")
                f_30_39 = sf("FEMALE_F30T34_LVPOP_CO") + sf("FEMALE_F35T39_LVPOP_CO")
                m_40_49 = sf("MALE_F40T44_LVPOP_CO") + sf("MALE_F45T49_LVPOP_CO")
                f_40_49 = sf("FEMALE_F40T44_LVPOP_CO") + sf("FEMALE_F45T49_LVPOP_CO")
                m_50_59 = sf("MALE_F50T54_LVPOP_CO") + sf("MALE_F55T59_LVPOP_CO")
                f_50_59 = sf("FEMALE_F50T54_LVPOP_CO") + sf("FEMALE_F55T59_LVPOP_CO")
                m_60_69 = sf("MALE_F60T64_LVPOP_CO") + sf("MALE_F65T69_LVPOP_CO")
                f_60_69 = sf("FEMALE_F60T64_LVPOP_CO") + sf("FEMALE_F65T69_LVPOP_CO")
                m_70plus = sf("MALE_F70T74_LVPOP_CO")
                f_70plus = sf("FEMALE_F70T74_LVPOP_CO")

                male_total = (m_0_9 + m_10_19 + m_20_29 + m_30_39 +
                              m_40_49 + m_50_59 + m_60_69 + m_70plus)
                female_total = (f_0_9 + f_10_19 + f_20_29 + f_30_39 +
                                f_40_49 + f_50_59 + f_60_69 + f_70plus)

                acc["tot"] += male_total + female_total
                acc["male"] += male_total
                acc["female"] += female_total
                acc["p0_9"] += m_0_9 + f_0_9
                acc["p10_19"] += m_10_19 + f_10_19
                acc["p20_29"] += m_20_29 + f_20_29
                acc["p30_39"] += m_30_39 + f_30_39
                acc["p40_49"] += m_40_49 + f_40_49
                acc["p50_59"] += m_50_59 + f_50_59
                acc["p60_69"] += m_60_69 + f_60_69
                acc["p70plus"] += m_70plus + f_70plus

            rows = []
            for dong_code, acc in dong_accum.items():
                cnt = acc["count"] or 1
                gu_code = dong_code[:5]
                gu_nm = _GU_CODE_5_MAP.get(gu_code, "")
                rows.append({
                    "stdr_de":    dt,
                    "dong_code":  dong_code,
                    "gu_code":    gu_code,
                    "gu_nm":      gu_nm,
                    "tot_pop":    round(acc["tot"] / cnt, 2),
                    "male_pop":   round(acc["male"] / cnt, 2),
                    "female_pop": round(acc["female"] / cnt, 2),
                    "pop_0_9":    round(acc["p0_9"] / cnt, 2),
                    "pop_10_19":  round(acc["p10_19"] / cnt, 2),
                    "pop_20_29":  round(acc["p20_29"] / cnt, 2),
                    "pop_30_39":  round(acc["p30_39"] / cnt, 2),
                    "pop_40_49":  round(acc["p40_49"] / cnt, 2),
                    "pop_50_59":  round(acc["p50_59"] / cnt, 2),
                    "pop_60_69":  round(acc["p60_69"] / cnt, 2),
                    "pop_70plus": round(acc["p70plus"] / cnt, 2),
                })

            n = insert_rows("daily_population_dong", rows)
            total += n
            if n > 0:
                log.info(f"  [C6] dong {dt}: {n}건 ({len(dong_accum)}동)")
            time.sleep(0.3)

        log_collection("C", "SPOP_LOCAL_RESD_DONG", "OK", total,
                       elapsed=time.time()-t0,
                       error=(None if total > 0 else "no new data in window"))
        return total

    # ── C2: 지하철 승하차 ────────────────────────────────────────────────────
    def collect_subway(self, target_date: str = None,
                       days_back: int = 1, max_rows: int = 1000) -> int:
        """
        CardSubwayStatsNew → daily_subway

        target_date: YYYYMMDD (지정 없으면 D-3 자동 계산)
        days_back: 연속 수집 일수
        max_rows: 한 번 요청에 가져올 역 수 (전체 ~600+)

        ⚠️ 서울 열린데이터 API 응답 키/필드명 변형 대응:
           CardSubwayStatsNew / CardSubwayStatsSVC 둘 다 시도
           필드명도 LINE_NUM vs SBWY_ROUT_LN_NM 등 변형 대응
        """
        t0 = time.time()
        total = 0

        if target_date:
            dates = [target_date]
        else:
            # D-3 ~ D-14: 나와 있는 날짜만 자동 수집, 미게재면 다음 실행에서 재시도
            base = datetime.now()
            dates = [(base - timedelta(days=i)).strftime("%Y%m%d")
                     for i in range(3, 15)]

        field_logged = False  # 첫 응답에서 필드명 1회만 로그

        for dt in dates:
            # 이미 적재된 날짜 스킵
            conn = get_conn()
            existing = conn.execute(
                "SELECT COUNT(*) FROM daily_subway WHERE use_dt = ?", (dt,)
            ).fetchone()[0]
            conn.close()
            if existing > 0:
                log.info(f"  [C2] {dt}: {existing:,}건 기존재 → 스킵")
                continue

            url = (f"{SEOUL_BASE}/{KEYS['seoul_subway']}"
                   f"/json/CardSubwayStatsNew/1/{max_rows}/{dt}")
            data = self.get(url)
            if not data:
                log.warning(f"  [C2] {dt}: 응답 없음")
                continue

            rows = []
            try:
                raw = data
                result_code = raw.get("RESULT", {}).get("CODE", "")
                if result_code == "INFO-200":
                    log.info(f"  [C2] {dt}: INFO-200 (미게재) -- 다음 실행 시 재시도")
                    continue
                items = (raw.get("CardSubwayStatsNew", {}).get("row")
                         or raw.get("CardSubwayStatsSVC", {}).get("row")
                         or [])
                if not items:
                    log.warning(f"  [C2] {dt} 응답 최상위 키: {list(raw.keys())[:5]}")
                    continue

                if not field_logged and items:
                    sample = items[0]
                    log.info(f"  [C2] 응답 필드명: {list(sample.keys())}")
                    field_logged = True

                for it in items:
                    line = (it.get("LINE_NUM")
                            or it.get("SBWY_ROUT_LN_NM")
                            or it.get("LINE_NM")
                            or "")
                    station = (it.get("SUB_STA_NM")
                               or it.get("SBWY_STNS_NM")
                               or it.get("STATN_NM")
                               or it.get("STATION_NM")
                               or "")
                    ride = (it.get("RIDE_PASGR_NUM")
                            or it.get("GTON_TNOPE")
                            or 0)
                    alight = (it.get("ALIGHT_PASGR_NUM")
                              or it.get("GTOFF_TNOPE")
                              or 0)

                    if not station:
                        continue

                    rows.append({
                        "use_dt":       it.get("USE_DT", dt),
                        "line_num":     line,
                        "station_nm":   station,
                        "ride_pasgr":   self.safe_int(ride),
                        "alight_pasgr": self.safe_int(alight),
                    })
            except Exception as e:
                log.error(f"  [C2] {dt} parse error: {e}")

            n = insert_rows("daily_subway", rows)
            save_csv("daily_subway", rows, date_str=dt)
            total += n
            log.info(f"  [C2] subway {dt}: {n}건 저장 (파싱 {len(rows)}건)")
            time.sleep(0.5)

        log_collection("C", "CardSubwayStatsNew",
                        "OK" if total > 0 else "FAIL",
                        total, elapsed=time.time()-t0)
        return total

    # ── C3: 버스 정류소별 승하차 ─────────────────────────────────────────────
    def collect_bus(self, target_date: str = None,
                    days_back: int = 1, max_rows: int = 1000) -> int:
        """
        CardBusStatisticsServiceNew → daily_bus

        서울 버스 일별 노선×정류장별 승하차 인원.
        데이터 갱신: 매일 D-3 기준 (3일 전 데이터까지 공개)

        API: 서울 열린데이터광장 OA-12912
             {SEOUL_BASE}/{key}/json/CardBusStatisticsServiceNew/{start}/{end}/{USE_YMD}
             (USE_YMD: YYYYMMDD, RTE_NO 선택)

        반환 필드:
          USE_YMD       사용일자 YYYYMMDD
          RTE_ID        노선ID
          RTE_NO        노선번호 (예: 472, N26)
          RTE_NM        노선명
          STOPS_ID      표준버스정류장ID
          STOPS_ARS_NO  버스정류장ARS번호
          SBWY_STNS_NM  역명(정류장명)
          GTON_TNOPE    승차총승객수
          GTOFF_TNOPE   하차총승객수
          REG_YMD       등록일자
        """
        ENDPOINT = "CardBusStatisticsServiceNew"
        t0 = time.time()
        total = 0

        if target_date:
            dates = [target_date]
        else:
            # D-3 ~ D-14: 나와 있는 날짜만 자동 수집, 미게재면 다음 실행에서 재시도
            base = datetime.now()
            dates = [(base - timedelta(days=i)).strftime("%Y%m%d")
                     for i in range(3, 15)]

        field_logged = False

        for dt in dates:
            # 이미 적재된 날짜 스킵
            conn = get_conn()
            existing = conn.execute(
                "SELECT COUNT(*) FROM daily_bus WHERE use_dt = ?", (dt,)
            ).fetchone()[0]
            conn.close()
            if existing > 0:
                log.info(f"  [C3] {dt}: {existing:,}건 기존재 → 스킵")
                continue

            data = None
            used_key = None

            # ── API 키 순서대로 시도 ──────────────────────────────────────
            for key_name in ("seoul_subway", "seoul_general", "seoul_general2"):
                url = (f"{SEOUL_BASE}/{KEYS[key_name]}"
                       f"/json/{ENDPOINT}/1/{max_rows}/{dt}")
                resp = self.get(url)
                if not resp:
                    continue
                result_code = resp.get("RESULT", {}).get("CODE", "")
                if result_code == "INFO-200":
                    log.debug(f"  [C3] {dt}/{key_name}: INFO-200 (미게재)")
                    continue
                if ENDPOINT in resp:
                    data = resp
                    used_key = key_name
                    break
                log.warning(f"  [C3] {dt}/{key_name}: 응답 이상 -- "
                            f"RESULT={resp.get('RESULT')}, "
                            f"top-level keys={list(resp.keys())[:6]}")

            if not data:
                # 모든 키가 INFO-200(미게재) 또는 응답 없음(네트워크 차단 등)
                log.info(f"  [C3] {dt}: 스킵 -- 게재 전 or 네트워크 차단, 다음 실행 시 재시도")
                continue

            rows = []
            try:
                raw_sect = data[ENDPOINT]
                items = raw_sect.get("row") or []
                if not items:
                    log.warning(f"  [C3] {dt}: row 없음 "
                                f"(RESULT={raw_sect.get('RESULT')})")
                    continue

                if not field_logged:
                    log.info(f"  [C3] API fields: {list(items[0].keys())}")
                    field_logged = True

                # CardBusStatisticsServiceNew: 이미 일별 집계값 → 그대로 저장
                for it in items:
                    station_nm = it.get("SBWY_STNS_NM", "").strip()
                    if not station_nm:
                        continue
                    rows.append({
                        "use_dt":     it.get("USE_YMD", dt),
                        "route_id":   it.get("RTE_ID", ""),
                        "route_no":   it.get("RTE_NO", ""),
                        "station_id": it.get("STOPS_ID", ""),
                        "station_nm": station_nm,
                        "ride_cnt":   self.safe_int(it.get("GTON_TNOPE")) or 0,
                        "alight_cnt": self.safe_int(it.get("GTOFF_TNOPE")) or 0,
                    })

            except Exception as e:
                log.error(f"  [C3] {dt} parse error: {e}")
                continue

            if rows:
                n = insert_rows("daily_bus", rows)
                save_csv("daily_bus", rows, date_str=dt)
                total += n
                log.info(f"  [C3] bus {dt}: {n}건 저장 [{used_key}]")

            time.sleep(0.5)

        log_collection("C", "CardBusStatisticsServiceNew",
                        "OK" if total > 0 else "FAIL",
                        total, elapsed=time.time()-t0)
        return total

    # ── C7: 응급의료 실시간 가용병상 ────────────────────────────────────────
    def collect_emergency_rooms(self) -> int:
        """
        국립중앙의료원 응급의료 실시간 가용병상 정보 → emergency_room_availability

        API: apis.data.go.kr/B552657/ErmctInfoInqireService/
             getEmrrmRltmUsefulSckbdInfoInqire

        수집 데이터: 서울시 응급실 가용병상, 중환자실, 음압격리, 구급차 가용 여부
        활용: SEIR-V-D 모델의 의료체계 부담 지표 (hospitalization capacity)
              → 가용병상 감소 시 case fatality rate 상승 proxy
        """
        from ..config import KEYS
        t0 = time.time()
        url = "https://apis.data.go.kr/B552657/ErmctInfoInqireService/getEmrrmRltmUsefulSckbdInfoInqire"

        rows = []
        page = 1
        total_count = 999  # 초기값
        now_iso = datetime.now().isoformat(timespec="seconds")

        # 서비스 접근 가능 여부 먼저 확인 (403 조기 탈출)
        test_params = {
            "serviceKey": KEYS["data_go_kr"],
            "STAGE1": "서울특별시",
            "pageNo": "1",
            "numOfRows": "1",
        }
        test_data = self.get(url, params=test_params, expect_json=False)
        if test_data is None:
            log.error(
                "  [C7] 응급의료 API 접근 실패 (403 가능성).\n"
                "  → data.go.kr에서 서비스 활용 신청 필요:\n"
                "    https://www.data.go.kr/data/15000563/openapi.do\n"
                "    '국립중앙의료원_응급의료기관 실시간 가용병상정보 조회 서비스'\n"
                "  → 승인 후 기존 키(data_go_kr)로 자동 접근 가능"
            )
            log_collection("C", "EmergencyRoomAvailability", "FAIL_AUTH", 0,
                           elapsed=time.time() - t0)
            return 0

        while (page - 1) * 100 < total_count:
            params = {
                "serviceKey": KEYS["data_go_kr"],
                "STAGE1":     "서울특별시",
                "pageNo":     str(page),
                "numOfRows":  "100",
            }
            data = self.get(url, params=params, expect_json=False)
            if not data:
                log.warning(f"  [C7] 응급의료 page {page}: 응답 없음")
                break

            try:
                import xml.etree.ElementTree as ET
                root = ET.fromstring(data)

                # totalCount 추출
                tc_el = root.find(".//totalCount")
                if tc_el is not None:
                    total_count = int(tc_el.text)

                items = root.findall(".//item")
                for it in items:
                    def _txt(tag: str) -> str:
                        el = it.find(tag)
                        return el.text.strip() if el is not None and el.text else ""

                    def _int(tag: str):
                        v = _txt(tag)
                        try:
                            return int(v)
                        except (ValueError, TypeError):
                            return None

                    hp_nm = _txt("dutyName")
                    if not hp_nm:
                        continue

                    rows.append({
                        "collected_at": now_iso,
                        "hp_id":        _txt("hpid"),
                        "hp_nm":        hp_nm,
                        "sido_nm":      "서울특별시",
                        "gu_nm":        _txt("dutyAddr").split()[1] if len(_txt("dutyAddr").split()) > 1 else "",
                        "hp_tel":       _txt("dutyTel3"),
                        "latitude":     self.safe_float(_txt("wgs84Lat")),
                        "longitude":    self.safe_float(_txt("wgs84Lon")),
                        "hvec":         _int("hvec"),
                        "hvoc":         _int("hvoc"),
                        "hvcc":         _int("hvcc"),
                        "hvncc":        _int("hvncc"),
                        "hvicc":        _int("hvicc"),
                        "hvgc":         _int("hvgc"),
                        "hv2":          _int("hv2"),
                        "hv3":          _int("hv3"),
                        "hv6":          _int("hv6"),
                        "hv8":          _int("hv8"),
                        "hv9":          _int("hv9"),
                        "hv10":         _int("hv10"),
                        "hv11":         _int("hv11"),
                        "hvamyn":       _txt("hvamyn"),
                    })
            except Exception as e:
                log.error(f"  [C7] 응급의료 page {page} parse error: {e}")
                break

            page += 1
            time.sleep(0.5)

        n = insert_rows("emergency_room_availability", rows)
        save_csv("emergency_room_availability", rows,
                 date_str=now_iso[:10].replace("-", ""))
        log_collection("C", "EmergencyRoomAvailability",
                        "OK" if n > 0 else "FAIL",
                        n, elapsed=time.time()-t0)
        log.info(f"  [C7] emergency_room_availability: {n}건 저장 "
                 f"(서울 {total_count}개 기관)")
        return n

    def run(self, backfill_days: int = 1, skip_apis: list = None) -> dict:
        """
        Group C 실행
        backfill_days: 과거 몇 일치 소급 수집 (C2·C3 교통, C4·C5·C6 인구에도 적용)
        skip_apis: 스킵할 API 코드 목록 (예: ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'])
        """
        skip_apis = skip_apis or []
        log.info("▶ Group C -- 일별 배치 수집 시작")
        r1, r2, r3, r4, r5, r6, r7 = 0, 0, 0, 0, 0, 0, 0

        # ── C1: citydata_ppltn 일별 스냅샷 ──────────────────────────────────
        if "C1" not in skip_apis:
            try:
                r1 = self.collect_hotspot_population()
            except Exception as e:
                log.error(f"  [C1] citydata_ppltn_daily 예외 (스킵): {e}")
        else:
            log.info("  [C1] citydata_ppltn_daily -- 스킵 (--skip C1)")

        if "C2" not in skip_apis:
            try:
                r2 = self.collect_subway(days_back=backfill_days)
            except Exception as e:
                log.error(f"  [C2] CardSubwayStatsNew 예외 (스킵): {e}")
        else:
            log.info("  [C2] CardSubwayStatsNew -- 스킵 (--skip C2)")

        if "C3" not in skip_apis:
            try:
                r3 = self.collect_bus(days_back=backfill_days)
            except Exception as e:
                log.error(f"  [C3] CardBusStatNew 예외 (스킵): {e}")
        else:
            log.info("  [C3] CardBusStatNew -- 스킵 (--skip C3)")

        # ── C4: 자치구 일별 생활인구 ────────────────────────────────────────
        if "C4" not in skip_apis:
            try:
                r4 = self.collect_district_population(days_back=backfill_days)
            except Exception as e:
                log.error(f"  [C4] SPOP_DAILYSUM_JACHI 예외 (스킵): {e}")
        else:
            log.info("  [C4] SPOP_DAILYSUM_JACHI -- 스킵 (--skip C4)")

        # ── C5: 자치구 시간대별 생활인구 ────────────────────────────────────
        if "C5" not in skip_apis:
            try:
                r5 = self.collect_gu_hourly_population(days_back=backfill_days)
            except Exception as e:
                log.error(f"  [C5] SPOP_LOCAL_RESD_JACHI 예외 (스킵): {e}")
        else:
            log.info("  [C5] SPOP_LOCAL_RESD_JACHI -- 스킵 (--skip C5)")

        # ── C6: 행정동별 생활인구 ───────────────────────────────────────────
        if "C6" not in skip_apis:
            try:
                r6 = self.collect_dong_population(days_back=backfill_days)
            except Exception as e:
                log.error(f"  [C6] SPOP_LOCAL_RESD_DONG 예외 (스킵): {e}")
        else:
            log.info("  [C6] SPOP_LOCAL_RESD_DONG -- 스킵 (--skip C6)")

        # ── C7: 응급의료 가용병상 ────────────────────────────────────────────
        if "C7" not in skip_apis:
            try:
                r7 = self.collect_emergency_rooms()
            except Exception as e:
                log.error(f"  [C7] EmergencyRoom 예외 (스킵): {e}")
        else:
            log.info("  [C7] EmergencyRoom -- 스킵 (--skip C7)")

        results = {
            "hotspot_population": r1,
            "subway": r2,
            "bus": r3,
            "district_population": r4,
            "gu_hourly_population": r5,
            "dong_population": r6,
            "emergency_rooms": r7,
        }
        total = sum(results.values())
        log.info(f"▶ Group C 완료 -- total {total} rows: {results}")
        return results