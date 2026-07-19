"""
simulation.llm_compare.factual_bench
=====================================
Factual-accuracy benchmark runner — the second comparison track.

``runner.py`` scores the 25 advisory golden-set scenarios with the 7-pillar
rubric (advisory *quality*). This module scores **factual accuracy** over the
Korean epidemiology/law QA (`kr_epi_bench`, n=40, official-source anchored) +
the external KorMedMCQA (n up to 3,009). Both tracks share the same backends
(`backends.py`) and the same SCI statistics (`comparison.compare_backends`) so
results are directly comparable; only the scorer differs.

★ CLI-tier confound control (user 2026-06-07)
---------------------------------------------
Cloud frontier models are compared via **subscription CLI** (claude/codex/gemini
`-p`/`exec`), NOT the paid API. Those CLIs are *agents* (system prompt + optional
tools), so this is a **product / as-deployed** comparison, not a raw-model one.
To keep the measurement about model KNOWLEDGE (not agentic web-search → which
would measure retrieval and risk contamination), every prompt carries a
``NO_TOOLS_PREAMBLE`` instructing single-shot, no-tool, knowledge-only answers,
and the manifest records each backend's CLI version + the exact config so the
confound is transparent. Documented in ``docs/LLM_BENCHMARK_STACK.md``.

CLI entry::

    python -m simulation.llm_compare.factual_bench --no-api --mock        # smoke
    python -m simulation.llm_compare.factual_bench --no-api --n-kormedmcqa 200 --repetitions 3
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from .backends import LLMBackend, LLMResponse, discover_backends, env_status
from .comparison import compare_backends, repetition_variance, repro_manifest
from .judge import ScoredResponse
from .kr_epi_bench import (
    format_mcqa,
    load_kormedmcqa,
    load_kr_epi_law,
    score_mcqa,
)

log = logging.getLogger(__name__)

__all__ = [
    "NO_TOOLS_PREAMBLE",
    "iter_factual_items",
    "factual_prompt",
    "score_factual",
    "extract_mcqa_letter",
    "run_factual_benchmark",
]

# Knowledge-only, single-shot, no-tool instruction (confound control).
NO_TOOLS_PREAMBLE = (
    "다음 질문에 당신의 내부 지식만으로 답하세요. 웹 검색·외부 도구·파일 접근을 사용하지 "
    "마세요(단발 응답). 한국 감염병 역학·법령(감염병예방법·KDCA·WHO) 기준으로 간결히 답하세요."
)


# ---------------------------------------------------------------------------
# Item iteration + prompt construction
# ---------------------------------------------------------------------------
def iter_factual_items(n_kormedmcqa: int = 0, *, kormedmcqa_subset: str = "doctor"):
    """Yield ``(kind, item)`` for the factual benchmark.

    Args:
        n_kormedmcqa: how many KorMedMCQA items to include (0 = epi/law only —
            no network dependency). Capped by the dataset / availability.
        kormedmcqa_subset: doctor / nurse / pharm / dentist.
    Yields:
        ``("kr_epi", KrEpiItem)`` then ``("kormedmcqa", dict)`` tuples.
    """
    for it in load_kr_epi_law():
        yield ("kr_epi", it)
    if n_kormedmcqa > 0:
        for it in load_kormedmcqa(kormedmcqa_subset, "test", n=n_kormedmcqa):
            yield ("kormedmcqa", it)


def factual_prompt(kind: str, item) -> str:
    """Build the knowledge-only prompt for an item (preamble + question)."""
    if kind == "kormedmcqa":
        return f"{NO_TOOLS_PREAMBLE}\n\n{format_mcqa(item)}"
    # kr_epi
    return (f"{NO_TOOLS_PREAMBLE}\n\n질문: {item.question}\n"
            "핵심 사실과 근거(조문/기관)를 한두 문장으로 답하세요.")


def _item_id(kind: str, item) -> str:
    if kind == "kormedmcqa":
        # stable per-(subset,question) id without exposing the answer
        import hashlib
        h = hashlib.sha256(item["question"].encode()).hexdigest()[:8]
        return f"KMQ_{item.get('subset', '')}_{h}"
    return item.id


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
_LETTER_RE = re.compile(r"\b([A-Ea-e])\b")


def extract_mcqa_letter(text: str) -> str:
    """Extract the model's chosen option letter (A-E) from a free-text answer.

    Handles '정답: C', '(C)', 'C.', '답은 C', or a lone letter. Returns '' if
    none found (scored as wrong, never raises).
    """
    if not text:
        return ""
    # Prefer an explicit '정답'/'답' marker, else the first standalone A-E.
    m = re.search(r"(?:정답|답)\s*[:：)\.]?\s*\(?([A-Ea-e])\)?", text)
    if m:
        return m.group(1).upper()
    m = _LETTER_RE.search(text)
    return m.group(1).upper() if m else ""


def score_factual(kind: str, item, resp: LLMResponse) -> ScoredResponse:
    """Score one response for factual accuracy → ``ScoredResponse``.

    kr_epi: correctness = fraction of ``must_contain`` present; a ``must_avoid``
        hit halves the score (factual error penalty). MCQA: exact option match
        (1.0 / 0.0). ``total`` is the accuracy in [0,1] so ``compare_backends``
        ranks backends and runs Wilcoxon/Holm + Fleiss κ on it.

    Never raises (errored responses → total 0.0, recorded via raw_response.error).
    """
    text = resp.text or ""
    iid = _item_id(kind, item)
    if resp.error:
        return ScoredResponse(iid, resp.backend_id, resp.model, {"correctness": 0.0},
                              0.0, [], [], [], resp, resp.latency_ms)

    if kind == "kormedmcqa":
        ok = score_mcqa(item, extract_mcqa_letter(text))
        total = 1.0 if ok else 0.0
        return ScoredResponse(iid, resp.backend_id, resp.model,
                              {"correctness": total}, total, [], [], [], resp,
                              resp.latency_ms)

    # kr_epi: must_contain / must_avoid factual check.
    # ⚠ Lexical proxy (Gemini M3 — DOCUMENTED limitation, not silently re-scored):
    # bare substring matching is a SOFT correctness signal, not verified accuracy.
    # Two known failure modes the headline 0.80–0.87 numbers should be read against:
    #   (1) short numeric tokens ('38','7일') can match incidentally;
    #   (2) a CONTRAST answer ("1급은 즉시, 4급은 표본감시") mentions a must_avoid term
    #       while being correct, and is halved.
    # By design (test_score_factual_kr_epi_grading) a must_avoid hit IS a hard
    # factual-error penalty (a contradiction in the answer), so the behavior is
    # kept; report the accuracies as a lexical proxy. A word-boundary + answer-span
    # rescore (or a human-κ subset) is the rigorous upgrade.
    mc = list(item.must_contain)
    hits = [m for m in mc if m in text]
    missing = [m for m in mc if m not in text]
    violations = [a for a in item.must_avoid if a in text]
    base = (len(hits) / len(mc)) if mc else 0.0
    total = base * (0.5 if violations else 1.0)
    return ScoredResponse(iid, resp.backend_id, resp.model,
                          {"correctness": round(total, 4)}, round(total, 4),
                          missing, violations, [], resp, resp.latency_ms)


# ---------------------------------------------------------------------------
# CLI version probe (repro manifest transparency)
# ---------------------------------------------------------------------------
def _cli_version(cmd: str) -> str:
    """Best-effort `<cmd> --version` capture for the manifest (empty on failure)."""
    try:
        r = subprocess.run([cmd, "--version"], capture_output=True, text=True,
                           timeout=15, stdin=subprocess.DEVNULL)
        return (r.stdout or r.stderr or "").strip().splitlines()[0][:80] if (r.stdout or r.stderr) else ""
    except Exception:  # noqa: BLE001
        return ""


def _backend_versions(backends) -> dict:
    """Map backend_id → CLI version (only for cli: tier; '' otherwise)."""
    out = {}
    for b in backends:
        if getattr(b, "tier", "") == "cli":
            cmd = getattr(b, "cli_cmd", ("",))[0]
            out[b.backend_id] = _cli_version(cmd) if cmd else ""
    return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_factual_benchmark(
    backends: Optional[Iterable[LLMBackend]] = None,
    *,
    n_kormedmcqa: int = 0,
    kormedmcqa_subset: str = "doctor",
    max_items: int = 0,
    repetitions: int = 1,
    temperature: float = 0.0,
    max_tokens: int = 256,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """Drive every backend over the factual benchmark and return a report dict.

    Args:
        backends: LLMBackend list (default = ``discover_backends()``).
        n_kormedmcqa: external KorMedMCQA items to add (0 = epi/law only).
        repetitions: repeat each (backend,item) R times → ``repetition_variance``
            captures CLI/model non-determinism (LLMs aren't seed-reproducible).
        temperature: 0.0 for accuracy (CLIs may ignore; recorded regardless).
        seed: recorded in the manifest (effective only for SDK backends).
    Returns:
        ``{generated_at, env, backends, n_items, repetitions, ranking,
        statistical_comparison, repetition_variance, repro_manifest}``.
        ``ranking`` is sorted by mean accuracy (best first). Never raises on a
        single backend failure (errors recorded as accuracy 0).
    """
    backends = list(backends) if backends is not None else discover_backends()
    if not backends:
        raise RuntimeError("no enabled backends; pass --mock at minimum")
    pairs = list(iter_factual_items(n_kormedmcqa, kormedmcqa_subset=kormedmcqa_subset))
    if max_items and max_items > 0:
        pairs = pairs[:max_items]            # cost/time cap for slow/metered CLI tier
    if not pairs:
        raise RuntimeError("empty factual benchmark (kr_epi_bench returned nothing)")

    scored: list[ScoredResponse] = []
    # per-(backend,item) rep scores — variance is measured WITHIN an item across
    # reps (LLM non-determinism), NOT pooled across items (which would conflate
    # item difficulty with non-determinism).
    rep_by_item: dict[str, dict[str, list[float]]] = {b.backend_id: {} for b in backends}
    for b in backends:
        for kind, item in pairs:
            prompt = factual_prompt(kind, item)
            for r in range(max(1, repetitions)):
                resp = b.generate(prompt, temperature=temperature, max_tokens=max_tokens)
                sr = score_factual(kind, item, resp)
                rep_by_item[b.backend_id].setdefault(sr.item_id, []).append(sr.total)
                if r == 0:                       # one ScoredResponse per item (rep 0) for stats
                    scored.append(sr)
                if verbose:
                    log.info("backend=%s item=%s rep=%d acc=%.2f%s",
                             b.backend_id, sr.item_id, r, sr.total,
                             f" ERR={resp.error[:40]}" if resp.error else "")

    # Per-backend accuracy — EXCLUDE errored responses (auth/timeout) from the
    # mean, and drop all-errored backends to an "unavailable" list so an
    # auth-failed CLI is NOT reported as 0.000 (a false "worst model" conclusion).
    ok: dict[str, list[float]] = {b.backend_id: [] for b in backends}
    errs: dict[str, int] = {b.backend_id: 0 for b in backends}
    for sr in scored:
        if sr.raw_response.error:
            errs[sr.backend_id] += 1
        else:
            ok[sr.backend_id].append(sr.total)
    ranking, unavailable = [], []
    for b in backends:
        bid = b.backend_id
        if ok[bid]:
            ranking.append({"backend_id": bid,
                            "accuracy": round(sum(ok[bid]) / len(ok[bid]), 4),
                            "n_items": len(ok[bid]), "n_errors": errs[bid], "tier": b.tier})
        else:
            unavailable.append({"backend_id": bid, "tier": b.tier, "n_errors": errs[bid],
                                "reason": "all responses errored (auth/timeout) — excluded from ranking"})
    ranking.sort(key=lambda d: d["accuracy"], reverse=True)
    scored_ok = [s for s in scored if not s.raw_response.error]

    rep_var: dict = {}
    if repetitions > 1:
        for bid, items_reps in rep_by_item.items():
            sds = [repetition_variance(reps)["sd"]
                   for reps in items_reps.values() if len(reps) >= 2]
            if sds:
                rep_var[bid] = {
                    "mean_item_sd": round(sum(sds) / len(sds), 4),
                    "max_item_sd": round(max(sds), 4),
                    "reps_per_item": repetitions,
                    "n_items": len(sds),
                    "frac_unstable": round(sum(s > 0 for s in sds) / len(sds), 4),
                }

    n_epi = sum(1 for k, _ in pairs if k == "kr_epi")
    n_mcq = sum(1 for k, _ in pairs if k == "kormedmcqa")
    prompts_blob = "\n".join(factual_prompt(k, it) for k, it in pairs)
    import hashlib
    manifest = repro_manifest(
        model="multi-backend",
        temperature=temperature, top_p=1.0,
        prompts_sha256=hashlib.sha256(prompts_blob.encode()).hexdigest()[:32],
        golden_n=len(pairs), golden_freeze_date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        law_version="감염병예방법(국가법령정보센터, as-fetched)", seed=seed,
        n_repetitions=repetitions,
        benchmark=f"kr_epi={n_epi}+KorMedMCQA={n_mcq}",
        backends=[b.backend_id for b in backends],
        cli_versions=_backend_versions(backends),
        confound_control="NO_TOOLS_PREAMBLE single-shot; CLI=agent not raw model",
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "env": env_status(),
        "backends": [{"backend_id": b.backend_id, "model": b.model, "tier": b.tier} for b in backends],
        "n_items": len(pairs), "n_kr_epi": n_epi, "n_kormedmcqa": n_mcq,
        "repetitions": repetitions,
        "ranking": ranking,
        "unavailable": unavailable,          # auth/timeout-failed backends (NOT ranked as 0)
        "statistical_comparison": (
            compare_backends(scored_ok)
            if len({s.backend_id for s in scored_ok}) >= 2 else {}),
        "repetition_variance": rep_var,
        "repro_manifest": manifest,
        "per_item": [s.to_dict() for s in scored],
    }


def _report_md(rep: dict) -> str:
    lines = ["# Factual-accuracy LLM comparison (kr_epi + KorMedMCQA)",
             f"Generated: {rep['generated_at']}",
             f"Items: {rep['n_items']} (kr_epi {rep['n_kr_epi']} + KorMedMCQA {rep['n_kormedmcqa']}), "
             f"repetitions {rep['repetitions']}",
             f"Config SHA: {rep['repro_manifest'].get('config_sha256')}", "",
             "## Accuracy ranking", "| rank | backend | accuracy | n | tier |",
             "|---|---|---|---|---|"]
    for i, r in enumerate(rep["ranking"], 1):
        lines.append(f"| {i} | {r['backend_id']} | {r['accuracy']:.4f} | {r['n_items']} | {r['tier']} |")
    if rep.get("unavailable"):
        lines += ["", "## Unavailable (auth/timeout — EXCLUDED, not scored 0)"]
        for u in rep["unavailable"]:
            lines.append(f"- **{u['backend_id']}** ({u['tier']}): {u['reason']} [{u['n_errors']} errs]")
    sc = rep.get("statistical_comparison", {})
    if sc.get("agreement"):
        lines += ["", f"Inter-LLM Fleiss κ: {sc['agreement'].get('kappa')} "
                  f"({sc['agreement'].get('interpretation')})"]
    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Factual-accuracy LLM comparison")
    from simulation.utils.paths import get_results_dir
    ap.add_argument("--out-dir", default=str(get_results_dir() / "llm_compare_factual"))
    ap.add_argument("--no-api", action="store_true")
    ap.add_argument("--no-cli", action="store_true", help="skip claude/codex/gemini CLI")
    ap.add_argument("--no-ollama", action="store_true")
    ap.add_argument("--openai-compat", action="append", default=[], metavar="MODEL@BASE_URL",
                    help="OpenAI-compatible server (vLLM/MLX/SGLang/LiteLLM) in place of "
                         "Ollama; repeatable. e.g. Qwen/Qwen2.5-7B-Instruct@http://localhost:8000/v1")
    ap.add_argument("--mock", action="store_true", help="include mock control profiles")
    ap.add_argument("--n-kormedmcqa", type=int, default=0)
    ap.add_argument("--kormedmcqa-subset", default="doctor")
    ap.add_argument("--max-items", type=int, default=0,
                    help="limit benchmark items (0=all; for slow/metered CLI tier)")
    ap.add_argument("--repetitions", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args(argv)

    specs = []
    for s in args.openai_compat:
        model, _, base = s.rpartition("@")
        if model and base:
            specs.append({"model": model, "base_url": base})
        else:
            log.warning("ignoring malformed --openai-compat %r (need MODEL@BASE_URL)", s)
    backends = discover_backends(
        include_api=not args.no_api, include_cli=not args.no_cli,
        include_ollama=not args.no_ollama, include_mock=args.mock,
        include_openai_compat=specs or None,
    )
    log.info("backends (%d): %s", len(backends), [b.backend_id for b in backends])
    rep = run_factual_benchmark(
        backends, n_kormedmcqa=args.n_kormedmcqa,
        kormedmcqa_subset=args.kormedmcqa_subset, max_items=args.max_items,
        repetitions=args.repetitions, temperature=args.temperature,
    )
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "factual_report.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "factual_report.md").write_text(_report_md(rep), encoding="utf-8")
    print("\nFactual accuracy ranking:")
    for i, r in enumerate(rep["ranking"], 1):
        print(f"  {i}. {r['backend_id']:30s} acc={r['accuracy']:.4f} (n={r['n_items']}, {r['tier']})")
    print(f"\nwrote {out/'factual_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
