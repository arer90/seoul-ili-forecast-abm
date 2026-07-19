"""
pipeline/collectors/group_q_emergency.py
=========================================
Group Q -- 응급실 병상현황
  Q2. 응급실 실시간 병상현황  → ed_visits_symptom

기존 키 사용: KEYS["data_go_kr"]

참고:
  Q2: B552657/ErmctInfoInqireService (C7과 동일 서비스군)
  Q1(119구급출동): 제거됨 — ILI 비특이적 (전체 구급 집계, 호흡기 분리 불가 → 노이즈)
"""

import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from .base import BaseCollector
from ..config import KEYS
from ..storage import insert_rows, save_csv, log_collection

log = logging.getLogger(__name__)


class GroupQCollector(BaseCollector):
    """응급실 병상 데이터"""

    SERVICE_KEY = KEYS.get("data_go_kr", "")

    # ── Q2: 응급실 실시간 가용 병상 ──────────────────────────────────────────
    def collect_ed_visits(self) -> int:
        """
        응급의료정보 실시간 가용 병상 → ed_visits_symptom

        API: B552657/ErmctInfoInqireService/getEmrrmRltmUsefulSckbdInfoInqire
        기존 C7과 동일 서비스 (group_c_daily.py에도 있음)
        """
        t0 = time.time()
        rows = []
        now = self.now_iso()

        base_url = ("https://apis.data.go.kr/B552657/"
                     "ErmctInfoInqireService/getEmrrmRltmUsefulSckbdInfoInqire")

        params = {
            "serviceKey": self.SERVICE_KEY,
            "STAGE1": "서울특별시",
            "pageNo": "1",
            "numOfRows": "100",
        }

        data = self.get(base_url, params=params, expect_json=False, timeout=30)
        if data is None:
            log.error("  [Q2] 응급실 API 접근 실패.\n"
                      "  → data.go.kr에서 서비스 활용 신청 확인:\n"
                      "    https://www.data.go.kr/data/15000563/openapi.do")
            log_collection("Q", "ed_visits", "FAIL_AUTH", 0, elapsed=time.time() - t0)
            return 0

        try:
            root = ET.fromstring(data)
            result_code = root.findtext(".//resultCode")
            if result_code and result_code != "00":
                result_msg = root.findtext(".//resultMsg", "Unknown")
                log.error(f"  [Q2] API 오류: {result_code} - {result_msg}")
                log_collection("Q", "ed_visits", "FAIL", 0, elapsed=time.time() - t0)
                return 0

            items = root.findall(".//item")
            for item in items:
                row = {
                    "collected_at": now,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "hospital_name": _get_text(item, "dutyName"),
                    "hospital_addr": _get_text(item, "dutyAddr"),
                    "bed_total": _get_int(item, "hvec"),
                    "bed_icu": _get_int(item, "hvoc"),
                    "bed_operate": _get_int(item, "hvgc"),
                    "bed_neuro": _get_int(item, "hvcc"),
                    "bed_neonatal": _get_int(item, "hvncc"),
                    "bed_general": _get_int(item, "hv1"),
                    "bed_internal": _get_int(item, "hv2"),
                    "ct_avail": _get_text(item, "hvctayn"),
                    "mri_avail": _get_text(item, "hvmriayn"),
                }
                rows.append(row)
        except ET.ParseError:
            log.warning("  [Q2] XML 파싱 실패")
        except Exception as e:
            log.warning(f"  [Q2] 처리 오류: {e}")

        n = insert_rows("ed_visits_symptom", rows)
        save_csv("ed_visits_symptom", rows)
        log_collection("Q", "ed_visits", "OK", n,
                       elapsed=time.time() - t0,
                       error=(None if rows else "no rows returned"))
        log.info(f"  [Q2] ed_visits_symptom: {n}건")
        return n

    def run(self, skip_apis: list = None) -> dict:
        skip_apis = skip_apis or []
        log.info("Group Q -- 응급실 데이터 수집")
        results = {}

        if "Q2" not in skip_apis:
            try:
                results["ed_visits"] = self.collect_ed_visits()
            except Exception as e:
                log.error(f"  [Q2] ed_visits failed: {e}")
                results["ed_visits"] = 0
        else:
            log.info("  [Q2] -- skip")
            results["ed_visits"] = 0

        log.info(f"Group Q 완료 -- total {sum(results.values())}: {results}")
        return results


def _get_text(item, tag, default=""):
    el = item.find(tag)
    return el.text.strip() if el is not None and el.text else default


def _get_int(item, tag, default=None):
    el = item.find(tag)
    if el is not None and el.text:
        try:
            return int(float(el.text.strip()))
        except ValueError:
            pass
    return default
