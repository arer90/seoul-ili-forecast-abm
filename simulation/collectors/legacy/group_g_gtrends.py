"""
pipeline/collectors/group_g_gtrends.py
=======================================
Group G -- Google Trends 검색 트렌드
  G1. 인플루엔자/ILI 관련 한국 검색 트렌드 → google_search_trends

pytrends 라이브러리 사용 (비공식 API, 키 불필요)
- 설치: pip install pytrends
- 주의: rate limit 존재 (429 Too Many Requests)
- 지역: KR (대한민국), KR-11 (서울)
"""

import logging
import time
from datetime import datetime, timedelta
from .base import BaseCollector
from ..storage import insert_rows, save_csv, log_collection

log = logging.getLogger(__name__)

# 검색어 그룹 (한국어 + 영어 혼합, 최대 5개/요청)
KEYWORD_GROUPS = [
    # 그룹 1: 핵심 ILI 키워드
    ["독감", "인플루엔자", "감기", "발열", "기침"],
    # 그룹 2: 약물/치료 + 의료기관
    ["타미플루", "소아과", "이비인후과", "해열제", "응급실"],
    # 그룹 3: 증상 상세
    ["콧물", "인후통", "몸살", "오한", "두통"],
]


class GroupGCollector(BaseCollector):
    """Google Trends 검색 트렌드 수집기"""

    def collect_search_trends(
        self,
        timeframe: str = "2016-01-01 {end}",
        geo: str = "KR",
    ) -> int:
        """
        Google Trends → google_search_trends 테이블

        Parameters
        ----------
        timeframe : "YYYY-MM-DD YYYY-MM-DD" 형식
        geo : 지역코드 (KR=한국, KR-11=서울)
        """
        t0 = time.time()

        try:
            from pytrends.request import TrendReq
        except ImportError:
            log.error("  [G1] pytrends 미설치! → pip install pytrends")
            log_collection("G", "google_trends", "FAIL_IMPORT", 0, elapsed=time.time() - t0)
            return 0

        end_date = datetime.now().strftime("%Y-%m-%d")
        tf = timeframe.format(end=end_date)

        pytrends = TrendReq(hl="ko", tz=540, timeout=(10, 30))
        all_rows = []
        now = self.now_iso()

        for gi, keywords in enumerate(KEYWORD_GROUPS):
            log.info(f"  [G1] 그룹 {gi+1}/{len(KEYWORD_GROUPS)}: {keywords}")
            try:
                pytrends.build_payload(keywords, cat=0, timeframe=tf, geo=geo)
                df = pytrends.interest_over_time()

                if df is None or df.empty:
                    log.warning(f"  [G1] 그룹 {gi+1}: 데이터 없음")
                    continue

                # 'isPartial' 컬럼 제거
                if "isPartial" in df.columns:
                    df = df.drop(columns=["isPartial"])

                for date_idx, row in df.iterrows():
                    date_str = date_idx.strftime("%Y-%m-%d")
                    for kw in keywords:
                        if kw in row:
                            all_rows.append({
                                "collected_at": now,
                                "period": date_str,
                                "geo": geo,
                                "keyword": kw,
                                "interest": int(row[kw]),
                                "group_idx": gi,
                            })

            except Exception as e:
                err_str = str(e)
                if "429" in err_str:
                    log.warning(f"  [G1] Rate limit! 60초 대기 후 재시도...")
                    time.sleep(60)
                    # 재시도 1회
                    try:
                        pytrends.build_payload(keywords, cat=0, timeframe=tf, geo=geo)
                        df = pytrends.interest_over_time()
                        if df is not None and not df.empty:
                            if "isPartial" in df.columns:
                                df = df.drop(columns=["isPartial"])
                            for date_idx, row in df.iterrows():
                                date_str = date_idx.strftime("%Y-%m-%d")
                                for kw in keywords:
                                    if kw in row:
                                        all_rows.append({
                                            "collected_at": now,
                                            "period": date_str,
                                            "geo": geo,
                                            "keyword": kw,
                                            "interest": int(row[kw]),
                                            "group_idx": gi,
                                        })
                    except Exception as e2:
                        log.error(f"  [G1] 그룹 {gi+1} 재시도 실패: {e2}")
                else:
                    log.error(f"  [G1] 그룹 {gi+1} 실패: {e}")

            # Rate limit 방지
            time.sleep(2)

        n = insert_rows("google_search_trends", all_rows)
        save_csv("google_search_trends", all_rows)
        log_collection("G", "google_trends", "OK", n,
                       elapsed=time.time() - t0,
                       error=(None if all_rows else "no rows returned"))
        log.info(f"  [G1] google_search_trends: {n}건 ({len(set(r['keyword'] for r in all_rows))} keywords)")
        return n

    def collect_seoul_trends(self) -> int:
        """서울 지역 한정 검색 트렌드 (KR-11)"""
        return self.collect_search_trends(geo="KR-11")

    def run(self, skip_apis: list = None) -> dict:
        """Group G 전체 실행"""
        skip_apis = skip_apis or []
        log.info("Group G -- Google Trends 수집 시작")
        results = {}

        if "G1" not in skip_apis:
            try:
                n = self.collect_search_trends(geo="KR")
                results["google_trends_kr"] = n
            except Exception as e:
                log.error(f"  [G1] google_trends failed: {e}")
                results["google_trends_kr"] = 0
        else:
            results["google_trends_kr"] = 0

        total = sum(results.values())
        log.info(f"Group G 완료 -- total {total}: {results}")
        return results
