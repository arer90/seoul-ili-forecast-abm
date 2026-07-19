"""Build a SCI-grade grounded-epidemiology LoRA dataset from the PubMed corpus.

The 35-example demo dataset (build_aria_lora_data.py) overfits; this builds a
larger, paper-grounded dataset from the already-collected PubMed abstracts
(simulation/data/collected/pubmed_abstracts/*.csv — ~30k rows) filtered to
influenza / ILI / vaccine / respiratory-surveillance relevance. Each example
teaches the ARIA *style*: a concise grounded answer that cites the real source
([기존 문헌: title, journal year, PMID]). Deterministic, offline, no network.

This is a STYLE / format adaptation (grounded, cited epidemiology answers), not a
knowledge-injection (that is what RAG is for). The LoRA result is measured
honestly against the base model.

Run:  .venv/bin/python -m simulation.scripts.build_aria_lora_scidata --max 600
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import re
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results" / "aria_lora_sci" / "data"
PUBMED = Path(__file__).resolve().parents[1] / "data" / "collected" / "pubmed_abstracts"

# Relevance filter: keep abstracts about influenza / ILI / vaccine / surveillance.
_RELEVANT = re.compile(
    r"influenza|\bflu\b|ILI|influenza-like|respiratory|vaccine|vaccination|"
    r"epidemic|outbreak|surveillance|forecast|H1N1|H3N2|SARI|seasonal",
    re.IGNORECASE,
)


def _first_sentences(text: str, n: int = 2, cap: int = 400) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    sents = re.split(r"(?<=[.!?])\s+", text)
    out = " ".join(sents[:n]).strip()
    return out[:cap]


def _question(title: str, mesh: str) -> str:
    title = (title or "").strip().rstrip(".")
    topic = title if title else "this influenza topic"
    forms = [
        f"What does the literature report about: {topic}?",
        f"Summarize the evidence on {topic}.",
        f"From an influenza-surveillance standpoint, what is known about {topic}?",
    ]
    # deterministic choice by title hash (no Math.random)
    return forms[sum(ord(c) for c in title) % len(forms)]


def _iter_rows():
    for f in sorted(glob.glob(str(PUBMED / "*.csv"))):
        with open(f, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                yield {k.lstrip("﻿"): v for k, v in row.items()}


def main(max_n: int) -> int:
    cand = []
    for r in _iter_rows():
        title, abs = r.get("title", ""), r.get("abstract", "")
        if not title or not abs or len(abs) < 120:
            continue
        blob = f"{title} {abs} {r.get('mesh_terms','')} {r.get('keywords','')}"
        if not _RELEVANT.search(blob):
            continue
        ans = _first_sentences(abs, 2)
        if len(ans) < 60:
            continue
        cite = (f"[기존 문헌: {title[:70].strip()}, {r.get('journal','')[:40].strip()} "
                f"{r.get('year','')}, PMID {r.get('pmid','')}]")
        cand.append((_question(title, r.get("mesh_terms", "")), f"{ans} {cite}"))

    rng = random.Random(42)
    rng.shuffle(cand)
    pairs = cand[:max_n]
    n = len(pairs)
    if n < 20:
        print(f"WARNING: only {n} relevant pairs found")
    n_val = max(4, n // 12)
    n_test = max(4, n // 12)
    valid, test, train = pairs[:n_val], pairs[n_val:n_val + n_test], pairs[n_val + n_test:]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("valid", valid), ("test", test)):
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for q, a in rows:
                fh.write(json.dumps({"messages": [
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ]}, ensure_ascii=False) + "\n")
    print(f"relevant_candidates={len(cand)}  used={n}  "
          f"train={len(train)} valid={len(valid)} test={len(test)}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--max", type=int, default=600, help="max examples to use")
    args = ap.parse_args()
    raise SystemExit(main(args.max))
