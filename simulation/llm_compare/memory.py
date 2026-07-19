"""simulation.llm_compare.memory
==================================
Persistent cross-run **verified-answer memory** for the ARIA layer (S3).

This is the "learn from history" surface: after an advisory answer passes the
delivery gate, it is appended (with its tool receipts and verdict) to a JSONL log
that survives across runs. Later runs retrieve the most relevant PAST VERIFIED
answers (BM25 over query + answer text) and inject them as prior reasoning context.

Read-only safety (the load-bearing property): memory stores answer TEXT + receipts
only and is retrieved as CONTEXT. It NEVER contributes to the verifier's gold pool
— the current answer is always gated against the CURRENT blackboard receipts — so a
remembered exemplar can add reasoning context but can NOT introduce an unreceipted
number. Only gate-passed (grounded) answers are ever admitted, so the log itself
never contains an ungrounded number.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

__all__ = ["VerifiedMemory"]


class VerifiedMemory:
    """Append-only JSONL of gate-passed advisory answers + BM25 retrieval.

    Args:
        path: JSONL location (default ``<results>/aria_memory/verified.jsonl``).

    Side effects: :meth:`remember` appends one line to disk. No DB.
    """

    def __init__(self, path: Optional[str | Path] = None):
        if path is not None:
            self.path = Path(path)
        else:
            from simulation.utils.paths import get_results_dir
            self.path = Path(get_results_dir()) / "aria_memory" / "verified.jsonl"

    def remember(self, query: str, final_answer: str, *, tool_receipts,
                 verification: dict) -> bool:
        """Persist an answer IFF it passed the gate (grounded).

        Args:
            query: the advisory question.
            final_answer: the delivered answer text.
            tool_receipts: the provenance receipts backing the answer's numbers.
            verification: the gate verdict; the answer is admitted only when
                ``verification["grounded"]`` is truthy.

        Returns:
            True if the record was written (grounded); False otherwise (rejected —
            an ungrounded answer never enters memory).

        Side effects: appends one JSONL line to :attr:`path` when admitted.
        """
        if not (verification and verification.get("grounded")):
            return False
        rec = {"query": query, "final_answer": final_answer,
               "tool_receipts": tool_receipts, "verification": verification,
               "ts": time.time()}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return True

    def all(self) -> list[dict]:
        """All stored verified records in write order (empty if none)."""
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """Top-``k`` most relevant PAST verified answers (BM25 over query+answer).

        Returns them as prior CONTEXT only — the caller must still gate the current
        answer against current tool receipts; memory adds no gold numbers.
        """
        recs = self.all()
        if not recs:
            return []
        from simulation.server.rag.rust_index import BM25Index
        docs = [f"{r.get('query', '')} {r.get('final_answer', '')}" for r in recs]
        idx = BM25Index(docs)
        hits = idx.search(query, k)
        return [recs[i] for i, _ in hits]

    def __len__(self) -> int:
        return len(self.all())
