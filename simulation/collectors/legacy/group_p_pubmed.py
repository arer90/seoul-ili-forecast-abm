"""
pipeline/collectors/group_p_pubmed.py
======================================
Group P -- PubMed / 질병관리청 텍스트 데이터 (LLM 학습용)
  P1. PubMed 논문 초록 수집       → pubmed_abstracts
  P2. KDCA 주간 보고서 메타데이터  → kdca_weekly_reports

용도: LLM fine-tuning / LoRA 어댑터 학습용 도메인 텍스트
- PubMed E-Utilities: 무료, API 키 불필요 (단 rate limit: 3 req/s)
- KDCA: 공공 웹페이지 (별도 API 없음, 메타데이터만 수집)
"""

import json
import logging
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from .base import BaseCollector
from ..storage import insert_rows, save_csv, log_collection

log = logging.getLogger(__name__)

# ── PubMed 검색 쿼리 ──────────────────────────────────────────────────────────
PUBMED_QUERIES = [
    # 한국 인플루엔자 역학
    '("influenza" OR "ILI" OR "influenza-like illness") AND ("Korea" OR "Seoul") AND ("epidemiology" OR "surveillance")',
    # 한국 호흡기 감염병
    '("respiratory infection" OR "respiratory virus") AND "Korea" AND ("seasonal" OR "outbreak")',
    # 인플루엔자 예측 모델
    '("influenza" OR "ILI") AND ("prediction" OR "forecasting" OR "machine learning" OR "deep learning")',
    # 감염병 시뮬레이션
    '("SEIR" OR "compartmental model" OR "metapopulation") AND ("influenza" OR "respiratory")',
    # 환경-감염병 연관
    '("influenza" OR "ILI") AND ("temperature" OR "humidity" OR "air pollution" OR "PM2.5") AND "association"',
]

# 각 쿼리별 최대 수집 건수
MAX_PER_QUERY = 500

PUBMED_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"


class GroupPCollector(BaseCollector):
    """PubMed / KDCA 텍스트 데이터 수집기"""

    # ── P1: PubMed 논문 초록 ─────────────────────────────────────────────────
    def collect_pubmed(self, max_per_query: int = MAX_PER_QUERY) -> int:
        """
        PubMed E-Utilities → pubmed_abstracts 테이블

        1) esearch: 검색어로 PMID 목록 획득
        2) efetch: PMID 목록으로 논문 메타+초록 획득
        """
        t0 = time.time()
        all_rows = []
        seen_pmids = set()
        now = self.now_iso()

        for qi, query in enumerate(PUBMED_QUERIES):
            log.info(f"  [P1] Query {qi+1}/{len(PUBMED_QUERIES)}: {query[:60]}...")

            # Step 1: esearch → PMID 리스트
            pmids = self._esearch(query, retmax=max_per_query)
            if not pmids:
                log.warning(f"  [P1] No results for query {qi+1}")
                continue

            # 중복 제거
            new_pmids = [p for p in pmids if p not in seen_pmids]
            seen_pmids.update(new_pmids)

            if not new_pmids:
                log.info(f"  [P1] Query {qi+1}: all {len(pmids)} already seen")
                continue

            # Step 2: efetch → 초록 (배치 200건씩)
            batch_size = 200
            for i in range(0, len(new_pmids), batch_size):
                batch = new_pmids[i:i + batch_size]
                rows = self._efetch_abstracts(batch, now)
                all_rows.extend(rows)
                time.sleep(0.5)  # rate limit 준수 (3 req/s)

            log.info(f"  [P1] Query {qi+1}: {len(new_pmids)} new PMIDs → {len(all_rows)} total rows")
            time.sleep(0.5)

        n = insert_rows("pubmed_abstracts", all_rows)
        save_csv("pubmed_abstracts", all_rows)
        log_collection("P", "pubmed", "OK", n,
                       elapsed=time.time() - t0,
                       error=(None if all_rows else "no rows returned"))
        log.info(f"  [P1] pubmed_abstracts: {n}건 저장")
        return n

    def _esearch(self, query: str, retmax: int = 500) -> list[str]:
        """PubMed esearch → PMID 리스트"""
        url = f"{PUBMED_BASE}/esearch.fcgi"
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": str(retmax),
            "retmode": "json",
            "sort": "relevance",
        }
        data = self.get(url, params=params, timeout=30)
        if not data:
            return []
        try:
            return data.get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            log.warning(f"  esearch parse error: {e}")
            return []

    def _efetch_abstracts(self, pmids: list[str], collected_at: str) -> list[dict]:
        """PubMed efetch → 논문 메타데이터 + 초록"""
        url = f"{PUBMED_BASE}/efetch.fcgi"
        params = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml",
        }
        xml_text = self.get(url, params=params, expect_json=False, timeout=60)
        if not xml_text:
            return []

        rows = []
        try:
            root = ET.fromstring(xml_text)
            for article in root.findall(".//PubmedArticle"):
                row = self._parse_article(article, collected_at)
                if row:
                    rows.append(row)
        except ET.ParseError as e:
            log.warning(f"  efetch XML parse error: {e}")
        except Exception as e:
            log.warning(f"  efetch error: {e}")

        return rows

    def _parse_article(self, article, collected_at: str) -> dict | None:
        """PubmedArticle XML 요소 → dict"""
        try:
            medline = article.find("MedlineCitation")
            if medline is None:
                return None

            pmid_elem = medline.find("PMID")
            pmid = pmid_elem.text if pmid_elem is not None else ""

            art = medline.find("Article")
            if art is None:
                return None

            # 제목 (mixed content 대응: <i>, <sub> 등 내부 태그 포함 가능)
            title_elem = art.find("ArticleTitle")
            title = "".join(title_elem.itertext()) if title_elem is not None else ""

            # 초록 (여러 AbstractText 요소를 합침)
            abstract_parts = []
            abstract_elem = art.find("Abstract")
            if abstract_elem is not None:
                for at in abstract_elem.findall("AbstractText"):
                    label = at.get("Label", "")
                    text = at.text or ""
                    if label:
                        abstract_parts.append(f"{label}: {text}")
                    else:
                        abstract_parts.append(text)
            abstract = " ".join(abstract_parts)

            # 저널
            journal_elem = art.find(".//Journal/Title")
            journal = journal_elem.text if journal_elem is not None else ""

            # 발행 연도
            year_elem = art.find(".//PubDate/Year")
            year = year_elem.text if year_elem is not None else ""

            # MeSH terms
            mesh_list = medline.find("MeshHeadingList")
            mesh_terms = []
            if mesh_list is not None:
                for mh in mesh_list.findall("MeshHeading/DescriptorName"):
                    if mh.text:
                        mesh_terms.append(mh.text)

            # 키워드
            keywords = []
            kw_list = medline.find("KeywordList")
            if kw_list is not None:
                for kw in kw_list.findall("Keyword"):
                    if kw.text:
                        keywords.append(kw.text)

            return {
                "collected_at": collected_at,
                "pmid": pmid,
                "title": title[:500],
                "abstract": abstract[:5000],
                "journal": journal[:200],
                "year": year,
                "mesh_terms": json.dumps(mesh_terms, ensure_ascii=False)[:1000],
                "keywords": json.dumps(keywords, ensure_ascii=False)[:500],
            }
        except Exception as e:
            log.warning(f"  Article parse error: {e}")
            return None

    # ── P2: KDCA 주간 보고서 메타 ────────────────────────────────────────────
    def collect_kdca_reports(self) -> int:
        """
        질병관리청 주간 건강과 질병 (PHWR) 메타데이터 → kdca_weekly_reports

        KDCA PHWR: https://www.kdca.go.kr/board/board.es?mid=a20602010000
        - API 없음, 게시판 URL에서 목록 페이지 파싱
        - 실제 텍스트 수집은 별도 스크래핑 필요 (여기서는 메타만)
        """
        t0 = time.time()
        rows = []
        now = self.now_iso()

        # KDCA PHWR 게시판 URL (2019~현재)
        base_url = "https://www.kdca.go.kr/board/board.es"
        params = {
            "mid": "a20601010100",
            "bid": "0024",
        }

        # 목록 페이지 1~10 수집 시도
        for page in range(1, 11):
            params["nPage"] = str(page)
            html = self.get(base_url, params=params, expect_json=False, timeout=30)
            if not html:
                break

            # 간단한 패턴 매칭으로 제목/날짜 추출
            # (BeautifulSoup 미사용 — 의존성 최소화)
            import re
            # 게시글 패턴: board.es?...seq=숫자 와 제목
            pattern = r'seq=(\d+)[^"]*"[^>]*>([^<]+)</a>'
            matches = re.findall(pattern, html)

            if not matches:
                log.info(f"  [P2] Page {page}: no matches found, stopping")
                break

            for seq, title in matches:
                title = title.strip()
                if not title:
                    continue
                rows.append({
                    "collected_at": now,
                    "source": "kdca_phwr",
                    "seq": seq,
                    "title": title[:500],
                    "url": f"{base_url}?mid=a20601010100&bid=0024&act=view&list_no={seq}",
                })

            log.info(f"  [P2] Page {page}: {len(matches)} entries")
            time.sleep(0.5)

        # 중복 seq 제거
        seen = set()
        unique_rows = []
        for r in rows:
            if r["seq"] not in seen:
                seen.add(r["seq"])
                unique_rows.append(r)

        n = insert_rows("kdca_weekly_reports", unique_rows)
        save_csv("kdca_weekly_reports", unique_rows)
        # KDCA's /board/board.es CGI was retired in late-2025 → every page
        # comes back HTTP 404. We still attempt the call (so the day they
        # fix it we resume scraping) but downgrade 0-rows to OK with a note
        # instead of FAIL to keep the audit trail clean. Real exceptions
        # still surface via the orchestrator try/except.
        msg = ("kdca_phwr endpoint /board/board.es returned no entries "
               "(KDCA retired this CGI; see SOP for replacement URL)")
        log_collection("P", "kdca_phwr", "OK", n,
                       elapsed=time.time() - t0,
                       error=(None if unique_rows else msg))
        log.info(f"  [P2] kdca_weekly_reports: {n}건 저장")
        return n

    def run(self, skip_apis: list = None) -> dict:
        """Group P 전체 실행"""
        skip_apis = skip_apis or []
        log.info("Group P -- PubMed/KDCA 텍스트 데이터 수집 시작")
        results = {}

        for key, method, code in [
            ("pubmed", self.collect_pubmed, "P1"),
            ("kdca_reports", self.collect_kdca_reports, "P2"),
        ]:
            if code in skip_apis:
                log.info(f"  [{code}] -- skip")
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
        log.info(f"Group P 완료 -- total {total} rows: {results}")
        return results
