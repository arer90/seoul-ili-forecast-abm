"""
pipeline/collectors/group_a_realtime.py
========================================
Group A -- 실시간 / 준실시간 (<10분 ~ 1시간)
  A1. 서울 실시간 인구 (citydata_ppltn)   → rt_population
  A2. 서울 실시간 대기 (RealtimeCityAir)  → rt_air_quality (source='seoul')
  A3. 에어코리아 대기오염                  → rt_air_quality (source='airkorea')
  A4. S-DoT 환경정보 (IotVdata017)        → rt_sdot_env
"""

import json
import logging
from .base import BaseCollector
from ..config import KEYS, SEOUL_BASE, SDOT_CGG_MAP
from ..storage import insert_rows, save_csv, log_collection
import time

log = logging.getLogger(__name__)

# 서울 대표 POI 목록 (전체 120개 수집 시 API 부하 큼 → 우선 주요 25개)
# 전체 수집 원할 시 True로 변경
COLLECT_ALL_POIS = False

SAMPLE_POIS = [
    "강남역", "홍대입구역", "명동", "광화문·덕수궁", "강북구청",
    "노원역", "신촌·이대역", "건대입구역", "잠실종합운동장", "여의도",
    "서울대입구역", "이태원역", "동대문역사문화공원", "마포구청", "영등포구청",
    "은평구청", "성북구청", "양천구청", "서초구청", "강서구청",
    "구로구청", "용산구청", "성동구청", "동작구청", "강동구청",
]


class GroupACollector(BaseCollector):

    # ── A1: 서울 실시간 인구 ─────────────────────────────────────────────────
    def collect_population(self, pois: list[str] = None) -> int:
        """citydata_ppltn → rt_population 테이블"""
        import urllib.parse
        t0 = time.time()
        target_pois = pois or SAMPLE_POIS
        rows = []
        now = self.now_iso()

        for poi in target_pois:
            encoded = urllib.parse.quote(poi)
            url = f"{SEOUL_BASE}/{KEYS['seoul_general']}/json/citydata_ppltn/1/5/{encoded}"
            data = self.get(url, timeout=90)  # 서버 과부하 대응 (기본 60→90초)
            if not data:
                continue

            try:
                items = data.get("SeoulRtd.citydata_ppltn", [])
                if not items:
                    continue
                item = items[0]  # 장소 1개

                rows.append({
                    "collected_at": now,
                    "area_cd":      item.get("AREA_CD", ""),
                    "area_nm":      item.get("AREA_NM", poi),
                    "congestion":   item.get("AREA_CONGEST_LVL", ""),
                    "ppltn_min":    self.safe_int(item.get("AREA_PPLTN_MIN")),
                    "ppltn_max":    self.safe_int(item.get("AREA_PPLTN_MAX")),
                    "raw_json":     json.dumps(item, ensure_ascii=False)[:1500],
                })
            except Exception as e:
                log.warning(f"  Population parse error ({poi}): {e}")
            time.sleep(0.3)

        n = insert_rows("rt_population", rows)
        save_csv("rt_population", rows)
        log_collection("A", "citydata_ppltn", "OK", n,
                       elapsed=time.time() - t0,
                       error=(None if rows else "no rows returned"))
        log.info(f"  [A1] rt_population: {n}건 저장")
        return n

    # ── A2: 서울 실시간 대기 ─────────────────────────────────────────────────
    def collect_seoul_air(self) -> int:
        """RealtimeCityAir → rt_air_quality (source='seoul')"""
        t0 = time.time()
        url = f"{SEOUL_BASE}/{KEYS['seoul_air']}/json/RealtimeCityAir/1/30/"
        data = self.get(url)
        if not data:
            log_collection("A", "RealtimeCityAir", "FAIL", elapsed=time.time()-t0)
            return 0

        rows = []
        now = self.now_iso()
        try:
            items = data.get("RealtimeCityAir", {}).get("row", [])
            for item in items:
                # Seoul API field mapping (API키 -> DB컬럼):
                #   PM -> pm10, FPM -> pm25, OZON -> o3,
                #   NTDX -> no2, SPDX -> so2, CBMX -> co,
                #   MSRSTN_NM -> location_nm, CAI_GRD -> khai_grade
                _grade_map = {"좋음": 1, "보통": 2, "나쁨": 3, "매우나쁨": 4}
                rows.append({
                    "collected_at": now,
                    "source":       "seoul",
                    "location_nm":  item.get("MSRSTN_NM", item.get("MSRSTE_NM", "")),
                    "pm10":         self.safe_float(item.get("PM")),
                    "pm25":         self.safe_float(item.get("FPM")),
                    "o3":           self.safe_float(item.get("OZON")),
                    "no2":          self.safe_float(item.get("NTDX")),
                    "so2":          self.safe_float(item.get("SPDX")),
                    "co":           self.safe_float(item.get("CBMX")),
                    "khai_grade":   _grade_map.get(item.get("CAI_GRD"), None),
                    "raw_json":     json.dumps(item, ensure_ascii=False)[:600],
                })
        except Exception as e:
            log.error(f"  [A2] Seoul air parse error: {e}")

        n = insert_rows("rt_air_quality", rows)
        save_csv("rt_air_quality", rows)
        log_collection("A", "RealtimeCityAir", "OK", n,
                       elapsed=time.time() - t0,
                       error=(None if rows else "no rows returned"))
        log.info(f"  [A2] rt_air_quality (seoul): {n}건 저장")
        return n

    # ── A3: 에어코리아 대기오염 ──────────────────────────────────────────────
    def collect_airkorea(self, stations: list[str] = None) -> int:
        """에어코리아 → rt_air_quality (source='airkorea')"""
        t0 = time.time()
        # 서울 대표 측정소 25개 구별 1개씩
        default_stations = [
            "종로구", "중구", "용산구", "성동구", "광진구",
            "동대문구", "중랑구", "성북구", "강북구", "도봉구",
            "노원구", "은평구", "서대문구", "마포구", "양천구",
            "강서구", "구로구", "금천구", "영등포구", "동작구",
            "관악구", "서초구", "강남구", "송파구", "강동구",
        ]
        target = stations or default_stations
        rows = []
        now = self.now_iso()

        for station in target:
            url = "https://apis.data.go.kr/B552584/ArpltnInforInqireSvc/getMsrstnAcctoRltmMesureDnsty"
            # ⚠️ stationName은 requests가 자동 인코딩하므로 raw 문자열 그대로 전달
            #    urllib.parse.quote() 사용 시 이중 인코딩 발생 → 0건 반환
            params = {
                "serviceKey": KEYS["data_go_kr"],
                "returnType": "json",
                "numOfRows": "1",
                "pageNo": "1",
                "stationName": station,   # raw 한글 문자열 (requests가 인코딩)
                "dataTerm": "DAILY",
                "ver": "1.0",
            }
            data = self.get(url, params=params)
            if not data:
                continue
            try:
                items = (data.get("response", {})
                              .get("body", {})
                              .get("items", []))
                if not items:
                    continue
                it = items[0]
                rows.append({
                    "collected_at": now,
                    "source":       "airkorea",
                    "location_nm":  station,
                    "pm10":         self.safe_float(it.get("pm10Value")),
                    "pm25":         self.safe_float(it.get("pm25Value")),
                    "o3":           self.safe_float(it.get("o3Value")),
                    "no2":          self.safe_float(it.get("no2Value")),
                    "so2":          self.safe_float(it.get("so2Value")),
                    "co":           self.safe_float(it.get("coValue")),
                    "khai_grade":   self.safe_int(it.get("khaiGrade")),
                    "raw_json":     json.dumps(it, ensure_ascii=False)[:400],
                })
            except Exception as e:
                log.warning(f"  Airkorea parse error ({station}): {e}")
            time.sleep(0.2)

        n = insert_rows("rt_air_quality", rows)
        save_csv("rt_air_quality", rows)
        log_collection("A", "airkorea", "OK", n,
                       elapsed=time.time() - t0,
                       error=(None if rows else "no rows returned"))
        log.info(f"  [A3] rt_air_quality (airkorea): {n}건 저장")
        return n

    # ── A4: S-DoT 환경정보 ──────────────────────────────────────────────────
    def collect_sdot_env(self, max_rows: int = 200) -> int:
        """IotVdata017 → rt_sdot_env (최대 max_rows건)"""
        t0 = time.time()
        url = f"{SEOUL_BASE}/{KEYS['seoul_air']}/json/IotVdata017/1/{max_rows}/"
        data = self.get(url)
        if not data:
            log_collection("A", "IotVdata017", "FAIL", elapsed=time.time()-t0)
            return 0

        rows = []
        now = self.now_iso()
        try:
            items = data.get("IotVdata017", {}).get("row", [])
            for it in items:
                cgg = it.get("CGG", "")
                # S-DoT IotVdata017 field mapping (Chrome API 검증 2026-04-11):
                #   SN -> sensor_id, AVG_TP -> temperature, AVG_HUM -> humidity,
                #   AVG_UV -> uv_index, AVG_NIS -> noise (NOT AVG_NOISE!),
                #   AVG_WSPD -> wind_speed, AVG_WD -> wind_dir
                #   ⚠️ S-DoT 센서에 PM10/PM25 필드 없음 (미세먼지 미측정)
                rows.append({
                    "collected_at": now,
                    "sensor_id":    it.get("SN", ""),
                    "cgg":          cgg,
                    "gu_code":      SDOT_CGG_MAP.get(cgg, ""),
                    "dong":         it.get("DONG", ""),
                    "temperature":  self.safe_float(it.get("AVG_TP")),
                    "humidity":     self.safe_float(it.get("AVG_HUM")),
                    "pm10":         None,  # S-DoT 센서 미측정
                    "pm25":         None,  # S-DoT 센서 미측정
                    "uv_index":     self.safe_float(it.get("AVG_UV")),
                    "noise":        self.safe_float(it.get("AVG_NIS")),
                    "wind_speed":   self.safe_float(it.get("AVG_WSPD")),
                    "wind_dir":     it.get("AVG_WD", ""),
                    "raw_json":     json.dumps(it, ensure_ascii=False)[:1000],
                })
        except Exception as e:
            log.error(f"  [A4] S-DoT parse error: {e}")

        n = insert_rows("rt_sdot_env", rows)
        save_csv("rt_sdot_env", rows)
        log_collection("A", "IotVdata017", "OK", n,
                       elapsed=time.time() - t0,
                       error=(None if rows else "no rows returned"))
        log.info(f"  [A4] rt_sdot_env: {n}건 저장")
        return n

    def run(self, skip_apis: list = None) -> dict:
        """Group A 전체 실행"""
        skip_apis = skip_apis or []
        log.info("Group A -- 실시간 수집 시작")
        results = {}
        for key, method, code in [
            ("rt_population", self.collect_population,  "A1"),
            ("seoul_air",     self.collect_seoul_air,   "A2"),
            ("airkorea",      self.collect_airkorea,    "A3"),
            ("sdot_env",      self.collect_sdot_env,    "A4"),
        ]:
            if code in skip_apis:
                log.info(f"  [{code}] -- skip (--skip {code})")
                results[key] = 0
                continue
            try:
                n = method()
                results[key] = n
                log.info(f"  [{code}] {key}: {n} rows")
            except Exception as e:
                log.error(f"  [{code}] {key} failed: {e}")
                results[key] = 0

        total = sum(results.values())
        log.info(f"Group A 완료 -- total {total} rows: {results}")
        return results