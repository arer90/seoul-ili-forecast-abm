"""
pipeline/collectors/group_c_monthly.py
========================================
Group CM -- 월별 교통카드 시간대별 통계
  C4. CardSubwayTime   (OA-12252) → monthly_subway_hourly
      지하철 호선별 역별 시간대별 승하차 (매월 5일 전월 갱신)
  C5. CardBusTimeNew   (OA-12913) → monthly_bus_hourly
      버스 노선별 정류장별 시간대별 승하차 (매월 5일 전월 갱신)

갱신 주기: 매월 5일 전월 데이터 확정
스케줄 예시:
  0 9 6 * *  python -m pipeline.run_pipeline --group CM

API 형식 (XML 전용 -- JSON 미지원):
  CardSubwayTime  : {SEOUL_BASE}/{key}/xml/CardSubwayTime/1/{max}/{YYYYMM}/
  CardBusTimeNew  : {SEOUL_BASE}/{key}/xml/CardBusTimeNew/1/{max}/{YYYYMM}/
  ※ /json/ 요청 시 빈 응답 반환 → /xml/ 형식만 정상 동작 확인

필드 패턴 (공통):
  - 시간대 승차: HR_{h}_GET_ON_NOPE  또는 HR_{h}_GET_ON_TNOPE  (h = 0..23)
  - 시간대 하차: HR_{h}_GET_OFF_NOPE 또는 HR_{h}_GET_OFF_TNOPE
  - 두 suffix 모두 런타임에 탐색하여 처리 (API 버전에 따라 혼재)
"""

import time
import logging
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from .base import BaseCollector
from ..config import KEYS, SEOUL_BASE
from ..storage import insert_rows, save_csv, log_collection, get_conn

log = logging.getLogger(__name__)


def _xml_to_dict(row_el: ET.Element) -> dict:
    """XML <row> 요소를 {tag: text} 딕셔너리로 변환."""
    return {child.tag: (child.text or "").strip() for child in row_el}


def _extract_hourly(item: dict) -> dict[int, tuple[int, int]]:
    """
    item 딕셔너리에서 시간대별 승하차 수를 추출.
    HR_{h}_GET_ON_NOPE / HR_{h}_GET_ON_TNOPE 두 가지 suffix 모두 허용.
    반환: {hour(0-23): (ride_cnt, alight_cnt)}
    """
    result: dict[int, tuple[int, int]] = {}
    for h in range(24):
        ride = alight = None
        for sfx in ("NOPE", "TNOPE"):
            on_key  = f"HR_{h}_GET_ON_{sfx}"
            off_key = f"HR_{h}_GET_OFF_{sfx}"
            if on_key in item:
                try:
                    ride   = int(item[on_key]  or 0)
                    alight = int(item.get(off_key) or 0)
                except (ValueError, TypeError):
                    ride, alight = 0, 0
                break
        if ride is not None:
            result[h] = (ride, alight)
    return result


class GroupCMonthlyCollector(BaseCollector):
    """매월 5일 전월 교통카드 시간대별 데이터 수집."""

    # ── 수집 대상 YYYYMM 목록 ────────────────────────────────────────────────
    def _target_months(self, n: int = 3) -> list[str]:
        """
        매월 5일 이후 전월 데이터가 확정되므로:
          - day >= 5 : 전월(offset 1)부터 n개월
          - day <  5 : 전전월(offset 2)부터 n개월
        반환: ['202502', '202501', '202412', ...] (최신순)
        """
        now = datetime.now()
        start_offset = 1 if now.day >= 5 else 2
        result = []
        base = now.replace(day=1)
        for i in range(n):
            m = base
            for _ in range(start_offset + i):
                m = (m - timedelta(days=1)).replace(day=1)
            ym = m.strftime("%Y%m")
            if ym not in result:
                result.append(ym)
        return result

    # ── C4: 지하철 호선별 역별 시간대별 ─────────────────────────────────────
    def collect_subway_hourly(self, target_ym: str = None,
                               months_back: int = 3) -> int:
        """
        CardSubwayTime → monthly_subway_hourly

        API 제한: 한 번에 최대 1000건 (ERROR-336)
        지하철 총 ~609역/월 → 단일 요청(1~1000)으로 전수 수집 가능
        """
        ENDPOINT = "CardSubwayTime"
        PAGE_SIZE = 1000   # API 최대 허용 건수
        t0 = time.time()
        total = 0
        skipped = 0  # months already in DB

        yms = [target_ym] if target_ym else self._target_months(months_back)

        for ym in yms:
            # ── 기존재 월 스킵 ──────────────────────────────────────────────
            conn = get_conn()
            existing = conn.execute(
                "SELECT COUNT(*) FROM monthly_subway_hourly WHERE use_ym = ?", (ym,)
            ).fetchone()[0]
            conn.close()
            if existing > 0:
                log.info(f"  [C4] {ym}: {existing:,}건 기존재 → 스킵")
                skipped += 1
                continue

            # CardSubwayTime은 /xml/ 형식만 정상 응답, 최대 1000건/요청
            url = (f"{SEOUL_BASE}/{KEYS['seoul_subway']}"
                   f"/xml/{ENDPOINT}/1/{PAGE_SIZE}/{ym}/")
            text = self.get(url, expect_json=False)
            if not text:
                log.warning(f"  [C4] {ym}: 응답 없음 -- 네트워크 차단 또는 게재 전")
                continue

            try:
                root = ET.fromstring(text)
            except ET.ParseError as e:
                log.error(f"  [C4] {ym}: XML 파싱 오류 -- {e}")
                continue

            result_code = root.findtext("RESULT/CODE", "")
            if result_code == "INFO-200":
                log.info(f"  [C4] {ym}: INFO-200 (미게재) -- 다음 실행 시 재시도")
                continue

            row_elements = root.findall("row")
            if not row_elements:
                log.warning(
                    f"  [C4] {ym}: row 없음 -- RESULT={result_code!r} "
                    f"root.tag={root.tag} children={[c.tag for c in root][:8]}"
                )
                continue

            total_cnt = int(root.findtext("list_total_count", "0") or 0)
            if total_cnt > PAGE_SIZE:
                log.warning(f"  [C4] {ym}: 총 {total_cnt}건 > {PAGE_SIZE}건 -- 페이지 초과")

            rows = []
            for item in [_xml_to_dict(el) for el in row_elements]:
                line_nm    = item.get("SBWY_ROUT_LN_NM", "").strip()
                station_nm = item.get("STTN", "").strip()
                use_ym_val = item.get("USE_MM", ym)
                if not station_nm:
                    continue
                for hour, (ride, alight) in _extract_hourly(item).items():
                    rows.append({
                        "use_ym":     use_ym_val,
                        "line_nm":    line_nm,
                        "station_nm": station_nm,
                        "hour":       hour,
                        "ride_cnt":   ride,
                        "alight_cnt": alight,
                    })

            n = insert_rows("monthly_subway_hourly", rows)
            save_csv("monthly_subway_hourly", rows, date_str=ym)
            total += n
            log.info(
                f"  [C4] subway_hourly {ym}: {n:,}건 저장 "
                f"(역 {len(row_elements)}개 × 최대 24시간)"
            )
            time.sleep(0.5)

        # OK if any rows inserted OR all months already in DB (skipped = normal)
        status = "OK" if (total > 0 or skipped == len(yms)) else "FAIL"
        log_collection("C", "CardSubwayTime", status, total, elapsed=time.time() - t0)
        return total

    # ── C5: 버스 노선별 정류장별 시간대별 ────────────────────────────────────
    def collect_bus_hourly(self, target_ym: str = None,
                            months_back: int = 3) -> int:
        """
        CardBusTimeNew → monthly_bus_hourly

        API 제한: 한 번에 최대 1000건 (ERROR-336)
        버스 총 ~38,000 노선×정류장/월 → 1000건씩 페이징 (~38 페이지/월)
        각 페이지×시간대(0~23) long-format 행으로 저장
        """
        ENDPOINT = "CardBusTimeNew"
        PAGE_SIZE = 1000   # API 최대 허용 건수
        t0 = time.time()
        total = 0
        skipped = 0  # months already in DB

        yms = [target_ym] if target_ym else self._target_months(months_back)

        for ym in yms:
            # ── 기존재 월 스킵 ──────────────────────────────────────────────
            conn = get_conn()
            existing = conn.execute(
                "SELECT COUNT(*) FROM monthly_bus_hourly WHERE use_ym = ?", (ym,)
            ).fetchone()[0]
            conn.close()
            if existing > 0:
                log.info(f"  [C5] {ym}: {existing:,}건 기존재 → 스킵")
                skipped += 1
                continue

            # ── 1페이지로 전체 건수 파악 ──────────────────────────────────
            used_key = None
            total_cnt = 0

            for key_name in ("seoul_subway", "seoul_general", "seoul_general2"):
                url = (f"{SEOUL_BASE}/{KEYS[key_name]}"
                       f"/xml/{ENDPOINT}/1/{PAGE_SIZE}/{ym}/")
                text = self.get(url, expect_json=False)
                if not text:
                    continue
                try:
                    r = ET.fromstring(text)
                except ET.ParseError:
                    continue
                rc = r.findtext("RESULT/CODE", "")
                if rc == "INFO-200":
                    log.info(f"  [C5] {ym}/{key_name}: INFO-200 (미게재)")
                    continue
                if r.findall("row"):
                    used_key = key_name
                    total_cnt = int(r.findtext("list_total_count", "0") or 0)
                    break
                log.warning(
                    f"  [C5] {ym}/{key_name}: row 없음 -- RESULT={rc!r} "
                    f"root.tag={r.tag} children={[c.tag for c in r][:6]}"
                )

            if not used_key:
                log.info(f"  [C5] {ym}: 스킵 -- 게재 전 or 네트워크 차단, 다음 실행 시 재시도")
                continue

            # ── 페이징으로 전체 수집 ───────────────────────────────────────
            n_pages = max(1, -(-total_cnt // PAGE_SIZE))  # ceil division
            log.info(f"  [C5] {ym}: 총 {total_cnt:,}건, {n_pages}페이지 수집 시작 [{used_key}]")

            all_rows = []
            for page in range(n_pages):
                start = page * PAGE_SIZE + 1
                end   = start + PAGE_SIZE - 1
                url = (f"{SEOUL_BASE}/{KEYS[used_key]}"
                       f"/xml/{ENDPOINT}/{start}/{end}/{ym}/")
                text = self.get(url, expect_json=False)
                if not text:
                    log.warning(f"  [C5] {ym} page {page+1}/{n_pages}: 응답 없음")
                    continue
                try:
                    root = ET.fromstring(text)
                except ET.ParseError:
                    log.warning(f"  [C5] {ym} page {page+1}/{n_pages}: XML 파싱 오류")
                    continue

                for item in [_xml_to_dict(el) for el in root.findall("row")]:
                    station_nm = item.get("SBWY_STNS_NM", "").strip()
                    if not station_nm:
                        continue
                    for hour, (ride, alight) in _extract_hourly(item).items():
                        all_rows.append({
                            "use_ym":     item.get("USE_YM", ym),
                            "route_no":   item.get("RTE_NO", ""),
                            "route_nm":   item.get("RTE_NM", ""),
                            "station_id": item.get("STOPS_ID", ""),
                            "station_nm": station_nm,
                            "hour":       hour,
                            "ride_cnt":   ride,
                            "alight_cnt": alight,
                        })
                time.sleep(0.3)  # 페이지당 rate-limit 방지

            n = insert_rows("monthly_bus_hourly", all_rows)
            save_csv("monthly_bus_hourly", all_rows, date_str=ym)
            total += n
            log.info(
                f"  [C5] bus_hourly {ym}: {n:,}건 저장 [{used_key}] "
                f"(총 {total_cnt:,} 노선×정류장, {n_pages}페이지)"
            )

        # OK if any rows inserted OR all months already in DB (skipped = normal)
        status = "OK" if (total > 0 or skipped == len(yms)) else "FAIL"
        log_collection("C", "CardBusTimeNew", status, total, elapsed=time.time() - t0)
        return total

    def run(self, months_back: int = 3, skip_apis: list = None) -> dict:
        """Group CM 실행."""
        skip_apis = skip_apis or []
        log.info("▶ Group CM -- 월별 교통 시간대별 수집 시작")
        r4, r5 = 0, 0

        if "C4" not in skip_apis:
            try:
                r4 = self.collect_subway_hourly(months_back=months_back)
            except Exception as e:
                log.error(f"  [C4] CardSubwayTime 예외 (스킵): {e}")
        else:
            log.info("  [C4] CardSubwayTime -- 스킵 (--skip C4)")

        if "C5" not in skip_apis:
            try:
                r5 = self.collect_bus_hourly(months_back=months_back)
            except Exception as e:
                log.error(f"  [C5] CardBusTimeNew 예외 (스킵): {e}")
        else:
            log.info("  [C5] CardBusTimeNew -- 스킵 (--skip C5)")

        return {
            "monthly_subway_hourly": r4,
            "monthly_bus_hourly":    r5,
        }
