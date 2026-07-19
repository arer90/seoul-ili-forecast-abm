"""
simulation.llm_compare.runner
=============================
Comparison harness: runs every enabled backend against every golden-set
item, scores the responses with the seven-pillar rubric, and produces a
thesis-grade comparison report with per-backend ranking, per-item
inter-backend disagreement, and a Hermes-style append-only audit log.

CLI entry::

    python -m simulation.llm_compare.runner --out-dir simulation/results/llm_compare

Default backend pool = ``discover_backends()``; override with
``--api/--ollama/--mock/--local-path`` flags.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from .backends import (
    LLMBackend,
    discover_backends,
    env_status,
)
from .comparison import compare_backends
from .golden_set import GoldenItem, load_golden_set
from .judge import SEVEN_PILLARS, ScoredResponse, score_response

log = logging.getLogger(__name__)

__all__ = [
    "ComparisonReport",
    "run_comparison",
    "verify_audit_chain",
]

DEFAULT_SYSTEM_PROMPT = (
    "You are the ARIA LLM consultation layer for Seoul's district-level "
    "influenza-like-illness forecasting and simulation stack . "
    "You never diagnose; you interpret forecasting and simulation outputs "
    "for a trained epidemiologist. Cite the thesis section (e.g. §4.13) when "
    "you use a value from it. Prefer hedged, specific, short answers with "
    "numbered steps where actionable. Korean input -> Korean answer; "
    "English input -> English answer."
)


def _ground_today(prompt: str) -> str:
    """Append today's date so the LLM grounds epidemiological context as-of-now.

    Sprint 2026-05-06 (#15): 사용자 critique — 질문에서 오늘 날짜 기준으로
    influenza season status / antiviral resistance updates / vaccine effectiveness
    를 추론하도록 명시.
    """
    today = datetime.now().strftime("%Y-%m-%d")
    return (
        f"{prompt}\n\nToday's date is {today}. "
        "Ground all epidemiological reasoning (influenza season phase, "
        "antiviral resistance reports, vaccine effectiveness) as of this date."
    )


@dataclass
class ComparisonReport:
    """Top-level container. Serialisable to JSON and Markdown."""
    generated_at: str
    env: dict
    backends: list[dict]
    items: list[dict]
    golden_set: list[dict]
    per_backend_mean: dict[str, float]
    per_pillar_mean: dict[str, dict[str, float]]
    ranking: list[dict]
    inter_backend_disagreement: list[dict]
    audit_chain: list[dict]
    # P5 (2026-06-06): SCI statistical comparison — pairwise Wilcoxon+Holm,
    # bootstrap-CI ranking, inter-LLM Fleiss κ. Empty for a single backend.
    statistical_comparison: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), indent=2, ensure_ascii=False)

    def to_markdown(self) -> str:
        lines: list[str] = []
        lines.append("# LLM comparison report")
        lines.append(f"Generated at: {self.generated_at}")
        lines.append("")
        lines.append(f"Environment: API keys present = "
                     f"{sorted(self.env.get('api_keys_present', {}).keys()) or 'none'}; "
                     f"Ollama models = {self.env.get('ollama_installed_models') or 'none'}")
        lines.append("")
        lines.append(f"Backends evaluated: {len(self.backends)} "
                     f"({', '.join(b['backend_id'] for b in self.backends)})")
        lines.append(f"Items evaluated: {len(self.items)}")
        lines.append("")
        lines.append("## Ranking")
        lines.append("| rank | backend_id | total | tier | mean latency ms |")
        lines.append("|---|---|---|---|---|")
        for i, row in enumerate(self.ranking, start=1):
            lines.append(
                f"| {i} | {row['backend_id']} | {row['total']:.4f} | "
                f"{row['tier']} | {row['mean_latency_ms']:.0f} |"
            )
        lines.append("")
        lines.append("## Per-pillar mean by backend")
        lines.append("| backend_id | " + " | ".join(SEVEN_PILLARS) + " |")
        lines.append("|---" + "|---" * len(SEVEN_PILLARS) + "|")
        for bid, pill in self.per_pillar_mean.items():
            lines.append(
                f"| {bid} | " + " | ".join(f"{pill.get(p, 0.0):.3f}" for p in SEVEN_PILLARS) + " |"
            )
        lines.append("")
        lines.append("## Inter-backend disagreement per item (std of total)")
        lines.append("| item_id | std_total | mean_total | best | worst |")
        lines.append("|---|---|---|---|---|")
        for row in self.inter_backend_disagreement:
            lines.append(
                f"| {row['item_id']} | {row['std_total']:.3f} | "
                f"{row['mean_total']:.3f} | {row['best_backend']} | {row['worst_backend']} |"
            )
        lines.append("")
        return "\n".join(lines) + "\n"


def _append_audit(chain: list[dict], entry: dict) -> None:
    """Hermes-style append-only SHA-256 hash chain. Each entry records its
    own hash plus the previous entry's hash so tampering is detectable."""
    prev = chain[-1]["hash"] if chain else "genesis"
    payload = json.dumps(entry, sort_keys=True, default=str).encode("utf-8")
    h = hashlib.sha256(prev.encode("utf-8") + payload).hexdigest()
    entry_out = dict(entry)
    entry_out["prev_hash"] = prev
    entry_out["hash"] = h
    chain.append(entry_out)


def verify_audit_chain(chain: list[dict]) -> dict:
    """Verify a Hermes SHA-256 audit chain is intact (tamper-evident read side).

    Recomputes each entry's hash from its payload (all keys except ``prev_hash``
    and ``hash``) chained to the previous entry's hash, and checks the
    ``prev_hash`` linkage. Any insertion, deletion, reordering, or field edit
    breaks the recomputed hash or the linkage and is reported at the first
    offending index. This is the read-side complement to :func:`_append_audit`
    and is what lets the audit ledger serve as tamper-evidence in the paper's
    reproducibility/governance section.

    Args:
        chain: the ``audit_chain`` list produced by :func:`_append_audit`.

    Returns:
        ``{"intact": bool, "n_entries": int, "first_bad_index": int | None,
        "reason": str}``.

    Performance: O(n) over the chain; no I/O. Side effects: none.
    Caller responsibility: pass the chain exactly as stored (do not mutate it).
    """
    prev = "genesis"
    for i, entry in enumerate(chain):
        payload_obj = {k: v for k, v in entry.items() if k not in ("prev_hash", "hash")}
        payload = json.dumps(payload_obj, sort_keys=True, default=str).encode("utf-8")
        expected = hashlib.sha256(prev.encode("utf-8") + payload).hexdigest()
        if entry.get("prev_hash") != prev:
            return {
                "intact": False, "n_entries": len(chain),
                "first_bad_index": i,
                "reason": f"prev_hash linkage broken at index {i}",
            }
        if entry.get("hash") != expected:
            return {
                "intact": False, "n_entries": len(chain),
                "first_bad_index": i,
                "reason": f"hash mismatch at index {i} (entry was modified)",
            }
        prev = entry["hash"]
    return {
        "intact": True, "n_entries": len(chain),
        "first_bad_index": None, "reason": "all entries verified",
    }


def run_comparison(
    backends: Optional[Iterable[LLMBackend]] = None,
    items: Optional[Iterable[GoldenItem]] = None,
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    temperature: float = 0.2,
    max_tokens: int = 512,
    verbose: bool = True,
) -> ComparisonReport:
    """Run the full harness and return a :class:`ComparisonReport`.

    Parameters
    ----------
    backends
        Iterable of pre-instantiated ``LLMBackend`` objects. Defaults to
        :func:`discover_backends` with standard settings.
    items
        Iterable of golden-set items. Defaults to the 20-item catalogue.
    system_prompt
        Shared system prompt prepended to every invocation. The default is
        the ARIA scope-limited instruction set.
    temperature, max_tokens
        Generation controls applied uniformly across backends.
    """
    backends = list(backends) if backends is not None else discover_backends()
    items = list(items) if items is not None else list(load_golden_set())
    if not backends:
        raise RuntimeError("no enabled backends discovered; provide --mock at minimum")
    if not items:
        raise RuntimeError("empty golden set; check simulation.llm_compare.golden_set")

    # Sprint 2026-05-06 (#15): today-grounded reasoning
    system_prompt = _ground_today(system_prompt)

    env = env_status()
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    audit: list[dict] = []
    _append_audit(audit, {
        "event": "harness.start",
        "time": generated_at,
        "n_backends": len(backends),
        "n_items": len(items),
        "env": env,
    })

    scored: list[ScoredResponse] = []
    for b in backends:
        for it in items:
            if verbose:
                log.info("backend=%s item=%s lang=%s", b.backend_id, it.id, it.lang)
            resp = b.generate(
                it.prompt, system=system_prompt,
                temperature=temperature, max_tokens=max_tokens,
            )
            sr = score_response(it, resp)
            scored.append(sr)
            _append_audit(audit, {
                "event": "llm.call",
                "backend_id": resp.backend_id,
                "model": resp.model,
                "item_id": it.id,
                "latency_ms": resp.latency_ms,
                "error": resp.error,
                "total": sr.total,
            })

    # Aggregate ----------------------------------------------------------------
    backend_ids = [b.backend_id for b in backends]
    per_backend: dict[str, list[ScoredResponse]] = {bid: [] for bid in backend_ids}
    for sr in scored:
        per_backend[sr.backend_id].append(sr)

    per_backend_mean = {
        bid: (sum(s.total for s in lst) / max(len(lst), 1)) if lst else 0.0
        for bid, lst in per_backend.items()
    }
    per_pillar_mean: dict[str, dict[str, float]] = {}
    for bid, lst in per_backend.items():
        acc = {p: 0.0 for p in SEVEN_PILLARS}
        if not lst:
            per_pillar_mean[bid] = acc
            continue
        for s in lst:
            for p in SEVEN_PILLARS:
                acc[p] += s.scores.get(p, 0.0)
        per_pillar_mean[bid] = {p: round(v / len(lst), 4) for p, v in acc.items()}

    backend_meta = [
        {
            "backend_id": b.backend_id,
            "model": b.model,
            "tier": b.tier,
            "provider": b.provider,
        }
        for b in backends
    ]
    ranking = sorted([
        {
            "backend_id": bid,
            "total": per_backend_mean[bid],
            "tier": next((b["tier"] for b in backend_meta if b["backend_id"] == bid), ""),
            "mean_latency_ms": (
                sum(s.latency_ms for s in per_backend[bid]) / max(len(per_backend[bid]), 1)
                if per_backend[bid] else 0.0
            ),
        }
        for bid in backend_ids
    ], key=lambda r: r["total"], reverse=True)

    # Inter-backend disagreement per item (std of total score)
    items_by_id = {it.id: it for it in items}
    disagree: list[dict] = []
    for it_id in items_by_id:
        totals = [s.total for s in scored if s.item_id == it_id]
        if not totals:
            continue
        mean = sum(totals) / len(totals)
        var = sum((t - mean) ** 2 for t in totals) / max(len(totals) - 1, 1) if len(totals) > 1 else 0.0
        std = var ** 0.5
        backend_totals = [(s.backend_id, s.total) for s in scored if s.item_id == it_id]
        backend_totals.sort(key=lambda t: t[1], reverse=True)
        disagree.append({
            "item_id": it_id,
            "n_backends": len(totals),
            "std_total": round(std, 4),
            "mean_total": round(mean, 4),
            "best_backend": backend_totals[0][0] if backend_totals else "",
            "worst_backend": backend_totals[-1][0] if backend_totals else "",
        })
    disagree.sort(key=lambda r: r["std_total"], reverse=True)

    _append_audit(audit, {
        "event": "harness.end",
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_scored": len(scored),
    })

    return ComparisonReport(
        generated_at=generated_at,
        env=env,
        backends=backend_meta,
        items=[
            s.to_dict() for s in scored
        ],
        golden_set=[dataclasses.asdict(it) for it in items],
        per_backend_mean={k: round(v, 4) for k, v in per_backend_mean.items()},
        per_pillar_mean=per_pillar_mean,
        ranking=ranking,
        inter_backend_disagreement=disagree,
        audit_chain=audit,
        statistical_comparison=(
            compare_backends(scored) if len(backend_ids) >= 2 else {}
        ),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )


def main(argv: Optional[list[str]] = None) -> int:
    _configure_logging()
    ap = argparse.ArgumentParser()
    from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
    ap.add_argument("--out-dir", default=str(get_results_dir() / "llm_compare"))
    ap.add_argument("--no-api", action="store_true", help="skip API-tier backends")
    ap.add_argument("--no-cli", action="store_true",
                    help="skip CLI-tier backends (claude/codex/gemini)")
    ap.add_argument("--no-ollama", action="store_true", help="skip Ollama tier")
    ap.add_argument("--no-mock", action="store_true", help="skip mock profiles")
    ap.add_argument("--local-path", action="append", default=[],
                    help="local model path (GGUF file or HF dir); may be repeated")
    ap.add_argument("--max-ollama", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--items", default=None,
                    help="comma-separated item ids to run (default: full set)")
    args = ap.parse_args(argv)

    backends = discover_backends(
        include_api=not args.no_api,
        include_cli=not args.no_cli,
        include_ollama=not args.no_ollama,
        include_mock=not args.no_mock,
        include_local_paths=args.local_path,
        max_ollama=args.max_ollama,
    )
    log.info("enabled backends (%d): %s", len(backends),
             [b.backend_id for b in backends])

    items = list(load_golden_set())
    if args.items:
        keep = set(args.items.split(","))
        items = [it for it in items if it.id in keep]
        log.info("restricted to %d items: %s", len(items), [it.id for it in items])

    report = run_comparison(
        backends, items,
        temperature=args.temperature, max_tokens=args.max_tokens,
    )

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(report.to_json(), encoding="utf-8")
    (out / "report.md").write_text(report.to_markdown(), encoding="utf-8")
    log.info("wrote %s and %s", out / "report.json", out / "report.md")

    # Print compact table to stdout for quick review
    print("\nRanking (mean total across all items):")
    for i, r in enumerate(report.ranking, start=1):
        print(f"  {i}. {r['backend_id']:32s}  total={r['total']:.4f}  "
              f"tier={r['tier']:6s}  lat={r['mean_latency_ms']:.0f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
