"""Expand the ARIA RAG corpus via PubMed E-utilities (외부평가 C-3, 2026-06-08).

비판: ~10개 초록 RAG 코퍼스는 데모 수준, fact_recall=1.0은 인공물. MedRAG 수준(수천)으로
확장 필요. 이 스크립트는 NCBI E-utilities(esearch+efetch)로 MeSH 질의별 초록을 받아
``simulation/data/collected/pubmed_abstracts/``에 추가한다(증분, 중복 PMID 제외).

네트워크 필요. API 키 없이도 동작(rate-limit 3 req/s 준수). 받은 뒤 GraphRAG가 자동
재인덱싱(dense embedding 캐시 force=True).

Run:  .venv/bin/python -m simulation.scripts.expand_rag_corpus --target 2000
      .venv/bin/python -m simulation.scripts.expand_rag_corpus --queries "influenza vaccine effectiveness" "antiviral resistance"
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

CORPUS = Path("simulation/data/collected/pubmed_abstracts")
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_QUERIES = [
    "influenza-like illness surveillance",
    "influenza vaccine effectiveness test-negative",
    "oseltamivir baloxavir antiviral influenza",
    "behavioral response epidemic risk perception",
    "human mobility infectious disease transmission",
    "metapopulation model influenza seasonal",
    "Korea KDCA influenza sentinel surveillance",
    "non-pharmaceutical intervention influenza",
]


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "mph-sim-rag/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def _esearch(query: str, retmax: int) -> list[str]:
    q = urllib.parse.urlencode({"db": "pubmed", "term": query, "retmax": retmax,
                                "retmode": "json"})
    data = json.loads(_get(f"{EUTILS}/esearch.fcgi?{q}"))
    return data.get("esearchresult", {}).get("idlist", [])


def _efetch_abstracts(pmids: list[str]) -> dict[str, str]:
    if not pmids:
        return {}
    q = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(pmids),
                                "rettype": "abstract", "retmode": "text"})
    text = _get(f"{EUTILS}/efetch.fcgi?{q}").decode("utf-8", "replace")
    # naive split by blank-line blocks; each block ~ one record
    blocks = [b.strip() for b in text.split("\n\n\n") if len(b.strip()) > 200]
    return {pmids[i]: blocks[i] for i in range(min(len(pmids), len(blocks)))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1500, help="목표 코퍼스 크기")
    ap.add_argument("--per-query", type=int, default=300)
    ap.add_argument("--queries", nargs="*", default=DEFAULT_QUERIES)
    args = ap.parse_args()

    CORPUS.mkdir(parents=True, exist_ok=True)
    existing = {p.stem for p in CORPUS.glob("*.txt")}
    print(f"기존 코퍼스: {len(existing)} abstracts → 목표 {args.target}")
    added = 0
    for query in args.queries:
        if len(existing) + added >= args.target:
            break
        try:
            pmids = [p for p in _esearch(query, args.per_query) if p not in existing]
            time.sleep(0.34)                       # rate-limit 3 req/s
            abstracts = _efetch_abstracts(pmids[:args.per_query])
            time.sleep(0.34)
        except Exception as e:
            print(f"  [{query[:40]}] 실패: {type(e).__name__}: {e}")
            continue
        for pmid, abs in abstracts.items():
            (CORPUS / f"{pmid}.txt").write_text(abs, encoding="utf-8")
            existing.add(pmid); added += 1
        print(f"  [{query[:40]}] +{len(abstracts)} (누적 {len(existing)})")

    print(f"\n총 {len(existing)} abstracts (이번 +{added}). "
          f"GraphRAG 재인덱싱: GraphRAG(...).build(force=True)")
    if len(existing) < 100:
        print("⚠ 네트워크/rate-limit로 목표 미달 — 재실행하거나 NCBI API 키 설정 권장.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
