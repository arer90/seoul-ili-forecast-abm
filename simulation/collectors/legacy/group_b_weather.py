"""
pipeline/collectors/group_b_weather.py
========================================
Group B -- 기상 데이터
 B1. VilageFcstInfoService_2.0/getVilageFcst → weather_forecast (서울 nx=60, ny=127)
 B2. getWthrSituation (텍스트 예보 -- 참고용)
 B3. KMA ASOS kma_sfcdd3.php → weather_historical (서울 일별 과거 관측)

[변경 이력]
 : nph-dfs_shrt_grd (격자 텍스트 API) -- 응답 포맷 불일치로 0건 지속
 : getVilageFcst (JSON API)로 전환 -- 서울 좌표 직접 지정, JSON 파싱
 : B3 ASOS 일별 관측 추가 (2026-03)
"""

import time
import logging
from datetime import datetime, timedelta
from .base import BaseCollector
from ..config import KEYS, TIMEOUT
from ..storage import insert_rows, save_csv, log_collection

log = logging.getLogger(__name__)

# 서울 격자 좌표 (동네예보 기준, 서울 중심)
SEOUL_NX = 60
SEOUL_NY = 127

# KMA 동네예보 발표 시각 (KST, 3시간 간격)
# base_time: 0200, 0500, 0800, 1100, 1400, 1700, 2000, 2300
FORECAST_BASE_TIMES = ["0200", "0500", "0800", "1100", "1400", "1700", "2000", "2300"]

# 수집할 기상변수 (카테고리)
# TMP: 1시간 기온, POP: 강수확률, PCP: 1시간 강수량, REH: 습도
# WSD: 풍속, SKY: 하늘상태, PTY: 강수형태
WEATHER_CATEGORIES = {"TMP", "POP", "PCP", "REH", "WSD", "SKY", "PTY"}


class GroupBCollector(BaseCollector):

    @staticmethod
    def _looks_like_asos_header(cols: list[str]) -> bool:
        """
        ASOS 텍스트 응답의 "컬럼명" 줄만 식별한다.

        help=0 응답에는 보통 컬럼명 줄과 단위 줄이 모두 `#` 로 시작한다.
        과거에는 단위 줄(M/S, MM, HPA ...)을 헤더로 잘못 잡아 전 필드가 NULL 이 되거나
        위치 기반 fallback 으로 509°C 같은 비정상 값이 저장됐다.
        """
        upper = {c.upper() for c in cols}
        has_core = {"TM", "STN"}.issubset(upper)
        has_temp = any(k in upper for k in ("TA", "TA_MAX", "TA_MIN", "TX", "TN"))
        has_weather = any(k in upper for k in ("HM", "WS", "RN", "RN_DAY", "PS", "SS", "SS_DAY"))
        return has_core and has_temp and has_weather

    @staticmethod
    def _clean_asos_value(value):
        """ASOS 결측 sentinels(-9, -9.0, -99 등)을 None으로 정규화한다."""
        try:
            val = float(value)
        except (TypeError, ValueError):
            return None
        if val in (-9.0, -99.0, -999.0):
            return None
        return val

    # ── B1: KMA 동네예보 (getVilageFcst -- JSON) ────────────────────────────────
    def _latest_base_time(self) -> tuple[str, str]:
        """
        현재 시각 기준 가장 최근 발표 base_date, base_time 반환.
        KMA 발표 데이터는 발표 시각 + ~10분 후부터 유효.
        반환: (base_date='YYYYMMDD', base_time='HHMM')
        """
        now = datetime.now()
        # 발표 시각 + API 반영 지연 고려 (10분)
        adjusted = now - timedelta(minutes=10)
        hour_str = f"{adjusted.hour:02d}00"

        # adjusted 시각보다 이전인 발표 시각 중 가장 최근
        valid_times = [t for t in FORECAST_BASE_TIMES if t <= hour_str]
        if valid_times:
            base_time = valid_times[-1]
            base_date = adjusted.strftime("%Y%m%d")
        else:
            # 새벽 0~2시: 전날 23시 발표
            base_time = FORECAST_BASE_TIMES[-1]  # "2300"
            base_date = (adjusted - timedelta(days=1)).strftime("%Y%m%d")

        return base_date, base_time

    def collect_forecast(self, base_date: str = None,
                          base_time: str = None,
                          quiet: bool = False) -> int:
        """
        KMA VilageFcstInfoService_2.0/getVilageFcst → weather_forecast 테이블

        base_date: YYYYMMDD (기본: 최근 발표 자동 계산)
        base_time: HHMM (기본: 자동 계산)
        quiet: when True (called from historical loop), skip per-iteration
               log_collection — the loop will log a single aggregate row.
               Prevents polluting the audit with N×8 FAIL rows for dates
               beyond KMA's 2-3 day retention window.
        """
        t0 = time.time()
        # Helper: only emit log_collection when not in quiet mode
        def _emit(status, n=0, error=None):
            if quiet:
                return
            log_collection("B", "getVilageFcst", status, n,
                           elapsed=time.time()-t0, error=error)

        if not base_date or not base_time:
            base_date, base_time = self._latest_base_time()

        url = ("https://apihub.kma.go.kr/api/typ02/openApi/"
               "VilageFcstInfoService_2.0/getVilageFcst")
        params = {
            "authKey":    KEYS["kma_hub"],
            "pageNo":     "1",
            "numOfRows":  "1000",    # 충분히 큰 값 (7변수 × ~여러 시간)
            "dataType":   "JSON",
            "base_date":  base_date,
            "base_time":  base_time,
            "nx":         str(SEOUL_NX),
            "ny":         str(SEOUL_NY),
        }

        data = self.get(url, params=params, timeout=TIMEOUT)
        if not data:
            _emit("FAIL", error="응답 없음")
            return 0

        # 응답 구조 검증
        rows = []
        now_iso = self.now_iso()
        try:
            # JSON 응답 구조: response.header.resultCode, response.body.items.item[]
            header = data.get("response", {}).get("header", {})
            result_code = header.get("resultCode", "?")
            result_msg  = header.get("resultMsg", "")

            if result_code != "00":
                log.error(f"  [B1] getVilageFcst 오류: resultCode={result_code}, "
                          f"msg={result_msg}")
                _emit("FAIL", error=f"resultCode={result_code}: {result_msg}")
                return 0

            items = (data.get("response", {})
                         .get("body", {})
                         .get("items", {})
                         .get("item", []))

            if not items:
                log.warning(f"  [B1] items 빈 배열 (base={base_date}/{base_time})")
                _emit("FAIL", error="empty items")
                return 0

            # 우리가 필요한 카테고리만 필터링
            for item in items:
                category = item.get("category", "")
                if category not in WEATHER_CATEGORIES:
                    continue

                fcst_date = item.get("fcstDate", "")
                fcst_time = item.get("fcstTime", "")
                fcst_value = item.get("fcstValue", "")

                val = self.safe_float(fcst_value)
                if val is None:
                    # "강수없음" 등 텍스트 값 → 0.0으로 변환
                    if fcst_value in ("강수없음", "적설없음"):
                        val = 0.0
                    else:
                        continue

                # issued_at: YYYYMMDDHH 형태로 저장 (호환성)
                issued_at = base_date + base_time[:2]
                # valid_at: 예보 적용 시각
                valid_at = fcst_date + fcst_time[:2]

                rows.append({
                    "collected_at": now_iso,
                    "issued_at":    issued_at,
                    "valid_at":     valid_at,
                    "variable":     category,
                    "nx":           SEOUL_NX,
                    "ny":           SEOUL_NY,
                    "value":        val,
                })

        except Exception as e:
            log.error(f"  [B1] getVilageFcst parse error: {e}")

        # 중복 제거: 같은 (issued_at, valid_at, variable)은 한 번만
        seen = set()
        unique_rows = []
        for r in rows:
            key = (r["issued_at"], r["valid_at"], r["variable"])
            if key not in seen:
                seen.add(key)
                unique_rows.append(r)

        n = insert_rows("weather_forecast", unique_rows)
        save_csv("weather_forecast", unique_rows)
        _emit("OK", n=n, error=(None if unique_rows else "no rows returned"))
        log.info(f"  [B1] weather_forecast: {n}건 저장 "
                 f"(base={base_date}/{base_time}, categories={len(WEATHER_CATEGORIES)})")
        return n

    def collect_historical_weather(self, days_back: int = 2) -> int:
        """
        과거 days_back일치 동네예보 일괄 수집 (DB 초기 적재 시 사용)
        하루 8회(3시간 발표) × days_back일

        ⚠️ KMA getVilageFcst는 미래 예보 전용 API.
           과거 base_date는 최근 ~2일 이내만 데이터 반환.
           2일 초과 요청 시 빈 응답 → 기본값 2.

        Per-iteration log_collection is suppressed (quiet=True) because KMA's
        2-3 day retention window guarantees most historical iterations FAIL.
        We emit a single aggregate row at the end of the loop so the audit
        shows one OK entry for the whole sweep instead of N×8 FAILs.
        """
        t0 = time.time()
        total = 0
        now = datetime.now()
        log.info(f"  [B1] 과거 {days_back}일치 동네예보 수집 시작")

        for d in range(days_back, 0, -1):
            base = now - timedelta(days=d)
            base_date = base.strftime("%Y%m%d")
            for bt in FORECAST_BASE_TIMES:
                n = self.collect_forecast(base_date=base_date, base_time=bt,
                                           quiet=True)
                total += n
                time.sleep(1.0)  # API 부하 방지

        log_collection("B", "getVilageFcst", "OK", total,
                       elapsed=time.time()-t0,
                       error=(None if total else
                              f"no rows in last {days_back}d "
                              f"(KMA retains ~2-3 days)"))
        log.info(f"  [B1] 과거 수집 완료: 총 {total}건")
        return total

    # ── B2: 기상 개황 텍스트 (참고용) ───────────────────────────────────────
    def collect_weather_situation(self) -> dict | None:
        """getWthrSituation -- 텍스트 예보 반환 (DB 저장 안 함, 로그용)"""
        url = ("https://apihub.kma.go.kr/api/typ02/openApi/"
               "VilageFcstMsgService/getWthrSituation")
        data = self.get(url, params={
            "authKey":    KEYS["kma_hub"],
            "numOfRows":  "1",
            "pageNo":     "1",
            "dataType":   "JSON",
            "stnId":      "108",   # 서울
        })
        if not data:
            return None
        try:
            items = (data.get("response", {})
                         .get("body", {})
                         .get("items", {})
                         .get("item", []))
            if items:
                log.info(f"  [B2] 기상개황: {items[0].get('wfSv','')[:80]}...")
                return items[0]
        except Exception as e:
            log.warning(f"  [B2] parse error: {e}")
        return None

    # ── B3: ASOS 일별 과거 관측 ─────────────────────────────────────────────
    def collect_asos_daily(self, start_date: str = None,
                            end_date: str = None,
                            stn: int = 108) -> int:
        """
        기상청 ASOS kma_sfcdd3.php → weather_historical

        서울(108) 관측소 일별 기상 관측값 (기온, 강수, 습도 등).
        감염병 계절성 분석 및 SIR β 보정(기온/습도 효과)에 사용.

        API: https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd3.php
        파라미터:
          tm1, tm2 : 기간 (YYYYMMDD)
          stn      : 관측소 코드 (서울=108, 강화=400, 양평=401)
          help     : 0
          authKey  : kma_hub

        ⚠️ 활용신청 필요: https://apihub.kma.go.kr → 서비스 → 1.4 일자료 기간조회
           동일 kma_hub 키 사용 (별도 승인 필요)
        """
        t0 = time.time()
        from datetime import datetime, timedelta

        now = datetime.now()
        # 기본: 최근 2년치 (미지정 시)
        end_dt   = end_date   or (now - timedelta(days=1)).strftime("%Y%m%d")
        start_dt = start_date or (now - timedelta(days=730)).strftime("%Y%m%d")

        ASOS_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfcdd3.php"

        params = {
            "tm1":     start_dt,
            "tm2":     end_dt,
            "stn":     str(stn),
            "help":    "0",
            "authKey": KEYS["kma_hub"],
        }
        # ⚠️ ASOS API는 텍스트(CSV) 형식 반환 → expect_json=False
        data = self.get(ASOS_URL, params=params, expect_json=False)
        if not data:
            log.warning(f"  [B3] ASOS {start_dt}~{end_dt}: 응답 없음. "
                        "활용신청(1.4 일자료 기간조회) 여부를 확인하세요.")
            log_collection("B", "ASOS_kma_sfcdd3", "FAIL",
                            elapsed=time.time()-t0)
            return 0

        rows = []
        try:
            # ── ASOS 응답 파싱 ────────────────────────────────────────────────
            # expect_json=False → data는 str
            # 응답 형식 1 (텍스트/CSV):
            #   # TM            STN  TA     TA_MAX   TA_MIN ...
            #   20200101 108  -3.4    2.1   -7.2 ...
            # 응답 형식 2 (JSON, 일부 API 버전):
            #   {"info":{...}, "data":[{"tm":"20200101","stn":"108","ta":"-3.4",...}]}
            raw_text = data if isinstance(data, str) else ""

            if not raw_text.strip():
                log.warning(f"  [B3] ASOS 빈 응답 ({start_dt}~{end_dt}). "
                            "활용신청(1.4 일자료 기간조회) 여부를 확인하세요.")
                log_collection("B", "ASOS_kma_sfcdd3", "FAIL",
                                elapsed=time.time()-t0)
                return 0

            # JSON 파싱 시도 먼저
            import json as _json
            try:
                data_json = _json.loads(raw_text)
                if isinstance(data_json, dict):
                    items = data_json.get("data", [])
                    for it in items:
                        rows.append({
                            "obs_date": str(it.get("tm", ""))[:8],
                            "stn_id":   self.safe_int(it.get("stn")) or stn,
                            "stn_nm":   it.get("stnNm", "서울"),
                            "ta_avg":   self.safe_float(it.get("ta")),
                            "ta_max":   self.safe_float(it.get("taMax")),
                            "ta_min":   self.safe_float(it.get("taMin")),
                            "hm_avg":   self.safe_float(it.get("hm")),
                            "ws_avg":   self.safe_float(it.get("ws")),
                            "rn_day":   self.safe_float(it.get("rn")),
                            "ps_avg":   self.safe_float(it.get("ps")),
                            "ss_day":   self.safe_float(it.get("ss")),
                        })
            except (_json.JSONDecodeError, ValueError):
                # ── typ01 fixed-order text parser (현재 kma_sfcdd3.php 실응답) ─────
                if raw_text.startswith("#START7777"):
                    # ── KMA ASOS kma_sfcdd3 일자료 컬럼 순서 (0-indexed) ────────
                    # ⚠️ 2026-04-10 수정: help=1 실응답 기반 정확한 매핑
                    # ── 기존 매핑은 kma_sfctm3(시간자료) 순서였으며 완전히 틀렸음 ──
                    #  0: TM           관측일 (KST)
                    #  1: STN          국내 지점번호
                    #  2: WS_AVG       일 평균 풍속 (m/s)
                    #  3: WR_DAY       일 풍정 (m)
                    #  4: WD_MAX       최대풍향
                    #  5: WS_MAX       최대풍속 (m/s)
                    #  6: WS_MAX_TM    최대풍속 시각 (시분)
                    #  7: WD_INS       최대순간풍향
                    #  8: WS_INS       최대순간풍속 (m/s)
                    #  9: WS_INS_TM    최대순간풍속 시각 (시분)
                    # 10: TA_AVG       일 평균기온 (°C)
                    # 11: TA_MAX       최고기온 (°C)
                    # 12: TA_MAX_TM    최고기온 시각 (시분) ← 구파서가 TA로 착각!
                    # 13: TA_MIN       최저기온 (°C)
                    # 14: TA_MIN_TM    최저기온 시각 (시분) ← 구파서가 HM으로 착각!
                    # 15: TD_AVG       일 평균 이슬점온도 (°C)
                    # 16: TS_AVG       일 평균 지면온도 (°C)
                    # 17: TG_MIN       일 최저 초상온도 (°C)
                    # 18: HM_AVG       일 평균 상대습도 (%)
                    # 19: HM_MIN       최저습도 (%)
                    # 20: HM_MIN_TM    최저습도 시각 (시분)
                    # 21: PV_AVG       일 평균 수증기압 (hPa)
                    # 22: EV_S         소형 증발량 (mm)
                    # 23: EV_L         대형 증발량 (mm)
                    # 24: FG_DUR       안개계속시간 (hr)
                    # 25: PA_AVG       일 평균 현지기압 (hPa)
                    # 26: PS_AVG       일 평균 해면기압 (hPa)
                    # 27: PS_MAX       최고 해면기압 (hPa)
                    # 28: PS_MAX_TM    최고 해면기압 시각 (시분)
                    # 29: PS_MIN       최저 해면기압 (hPa)
                    # 30: PS_MIN_TM    최저 해면기압 시각 (시분)
                    # 31: CA_TOT       일 평균 전운량 (1/10)
                    # 32: SS_DAY       일조합 (hr)
                    # 33: SS_DUR       가조시간 (hr)
                    # 34: SS_CMB       캄벨 일조 (hr)
                    # 35: SI_DAY       일사합 (MJ/m²)
                    # 36: SI_60M_MAX   최대 1시간일사 (MJ/m²)
                    # 37: SI_60M_MAX_TM 최대 1시간일사 시각 (시분)
                    # 38: RN_DAY       일 강수량 (mm)
                    # 39~55: RN_D99, RN_DUR, 적설, 지중온도 등
                    for line in raw_text.splitlines():
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue

                        parts = line.split()
                        if len(parts) < 39:
                            continue
                        if not parts[0][:8].isdigit():
                            continue

                        rows.append({
                            "obs_date": parts[0][:8],
                            "stn_id":   self.safe_int(parts[1]) or stn,
                            "stn_nm":   "서울",
                            "ta_avg":   self._clean_asos_value(parts[10]),   # TA_AVG  일 평균기온 (°C)
                            "ta_max":   self._clean_asos_value(parts[11]),   # TA_MAX  최고기온 (°C)
                            "ta_min":   self._clean_asos_value(parts[13]),   # TA_MIN  최저기온 (°C)
                            "hm_avg":   self._clean_asos_value(parts[18]),   # HM_AVG  일 평균 상대습도 (%)
                            "ws_avg":   self._clean_asos_value(parts[2]),    # WS_AVG  일 평균 풍속 (m/s)
                            "rn_day":   self._clean_asos_value(parts[38]),   # RN_DAY  일 강수량 (mm)
                            "ps_avg":   self._clean_asos_value(parts[25]),   # PA_AVG  현지기압 (hPa)
                            "ss_day":   self._clean_asos_value(parts[32]),   # SS_DAY  일조합 (hr)
                        })

                # ── 텍스트(CSV) 헤더 기반 parser (구 포맷 호환) ───────────────────
                if not rows:
                # ── 텍스트(CSV) 파싱 ─────────────────────────────────────────
                # KMA ASOS sfcdd3 텍스트 컬럼 순서 (help=0):
                # TM  STN  WD  WS  GST  GSD  GST_TR  GSD_TR  WD_MAX  WS_MAX
                # WD_10M  WS_10M  TA  TD  HM  PV  RN  RN_DAY  SD_DAY  SD_TOT
                # SD_3  TS  TD05  TD10  TD20  TD30  TE05  TE10  TE20  TE30
                # ⚠️ 위치 기반(pos 2=WD)이 아닌 헤더 기반 매핑 사용해야 함
                    header_cols = []
                    header_index = {}
                    for line in raw_text.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        if line.startswith("#"):
                            # 헤더라인 파싱 (# 제거 후 컬럼명 추출)
                            cols = line.lstrip("# ").split()
                            if cols and not cols[0].isdigit() and self._looks_like_asos_header(cols):
                                header_cols = [c.upper() for c in cols]
                                header_index = {name: idx for idx, name in enumerate(header_cols)}
                            continue

                        parts = line.split()
                        if len(parts) < 3:
                            continue
                        # 날짜 형식 검증 (YYYYMMDD 또는 YYYY-MM-DD)
                        date_str = parts[0].replace("-", "")
                        if not date_str[:8].isdigit():
                            continue

                        # 헤더 기반 컬럼 인덱스 매핑 (헤더 있는 경우)
                        # KMA sfcdd3 표준 컬럼명: TA, TA_MAX, TA_MIN, HM, WS, RN, PS, SS
                        def _hcol(col_name, default=None):
                            """헤더 컬럼명으로 값 조회 (대소문자 무관)"""
                            if header_index:
                                for alias in (col_name, col_name.replace("_", "")):
                                    idx = header_index.get(alias)
                                    if idx is not None:
                                        return self.safe_float(parts[idx]) if len(parts) > idx else default
                            return default

                        # 첫 데이터 행에서 헤더 진단 로그
                        if not rows and header_cols:
                            log.info(f"  [B3] ASOS 텍스트 헤더: {header_cols[:15]}")
                        elif not rows and not header_cols:
                            log.warning(
                                "  [B3] ASOS 컬럼명 헤더를 찾지 못해 위치 기반 fallback 을 사용하지 않습니다. "
                                f"첫 데이터 행: {parts[:10]}"
                            )

                        # 헤더 기반 파싱
                        if header_cols:
                            row = {
                                "obs_date": date_str[:8],
                                "stn_id":   stn,
                                "stn_nm":   "서울",
                                "ta_avg":   _hcol("TA"),
                                "ta_max":   _hcol("TA_MAX") or _hcol("TAMAX") or _hcol("TX"),
                                "ta_min":   _hcol("TA_MIN") or _hcol("TAMIN") or _hcol("TN"),
                                "hm_avg":   _hcol("HM"),
                                "ws_avg":   _hcol("WS"),
                                "rn_day":   _hcol("RN_DAY") or _hcol("RN"),
                                "ps_avg":   _hcol("PA") or _hcol("PS"),
                                "ss_day":   _hcol("SS") or _hcol("SS_DAY"),
                            }
                        else:
                            # 위치 기반 fallback 은 비활성화한다.
                            # 이유: 2020-2025 백업에서 ta_max=509.0, ws_avg=904.0 같은
                            # 비정상 값이 저장됐고, 원인은 API 버전별 컬럼 위치 차이였다.
                            continue

                        measurements = (
                            row["ta_avg"], row["ta_max"], row["ta_min"],
                            row["hm_avg"], row["ws_avg"], row["rn_day"],
                            row["ps_avg"], row["ss_day"],
                        )
                        if any(v is not None for v in measurements):
                            rows.append(row)

            if rows:
                log.info(f"  [B3] ASOS 파싱 {len(rows)}건 ({rows[0]['obs_date']}~{rows[-1]['obs_date']})")
            else:
                log.warning(f"  [B3] ASOS 파싱 0건. 응답 샘플: {raw_text[:200]}")

        except Exception as e:
            log.error(f"  [B3] ASOS parse error: {e}")

        n = insert_rows("weather_historical", rows)
        save_csv("weather_historical", rows, date_str=f"{start_dt}_{end_dt}")
        log_collection("B", "ASOS_kma_sfcdd3", "OK", n,
                       elapsed=time.time() - t0,
                       error=(None if rows else "no rows returned"))
        log.info(f"  [B3] weather_historical: {n}건 저장 ({start_dt}~{end_dt}, stn={stn})")
        return n

    def run(self, historical_days: int = 0, skip_apis: list = None,
            asos_start: str = None, asos_end: str = None) -> dict:
        """
        Group B 실행.
        historical_days > 0 이면 B1 과거 일괄 수집.
        asos_start/asos_end: B3 ASOS 수집 기간 (YYYYMMDD, 기본: 최근 2년).
        """
        skip_apis = skip_apis or []
        log.info("▶ Group B -- 기상 수집 시작")
        n_fcst = 0
        n_asos = 0

        if "B1" not in skip_apis:
            try:
                if historical_days > 0:
                    n_fcst = self.collect_historical_weather(days_back=historical_days)
                else:
                    n_fcst = self.collect_forecast()
            except Exception as e:
                log.error(f"  [B1] getVilageFcst 예외 (스킵): {e}")
        else:
            log.info("  [B1] getVilageFcst -- 스킵 (--skip B1)")

        if "B2" not in skip_apis:
            try:
                self.collect_weather_situation()
            except Exception as e:
                log.error(f"  [B2] getWthrSituation 예외 (스킵): {e}")

        if "B3" not in skip_apis:
            try:
                n_asos = self.collect_asos_daily(start_date=asos_start,
                                                  end_date=asos_end)
            except Exception as e:
                log.error(f"  [B3] ASOS_kma_sfcdd3 예외 (스킵): {e}")
        else:
            log.info("  [B3] ASOS_kma_sfcdd3 -- 스킵 (--skip B3)")

        return {
            "weather_forecast":    n_fcst,
            "weather_historical":  n_asos,
        }
