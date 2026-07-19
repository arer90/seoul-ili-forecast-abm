"""LLM-judge RAGAS-equivalent metrics for the ARIA grounding harness.

Why this exists (not the `ragas` package)
------------------------------------------
The real ``ragas`` package (Es et al. 2024) is INCOMPATIBLE with this project's
langchain stack: ragas 0.4.x imports ``langchain_community.chat_models.vertexai``,
which the installed langchain 1.3.x relocated, so ``import ragas`` raises
ModuleNotFoundError; and downgrading langchain would break the live ARIA
LangChain integration (env-protection rule — never break the offline stack).
This module therefore implements the RAGAS *methodology* — faithfulness, context
precision, context recall, and answer relevancy — directly via an LLM judge over
the project's existing offline backends (Ollama by default), with NO new
dependency and no environment risk.

Design discipline (ENGINEERING_PRINCIPLES.md)
-----------------------------
- OFF by default: gated on ``MPH_RAGAS_LLM=1``. When disabled, when no judge
  backend is available, or when a judge call fails, every metric DELEGATES to the
  deterministic token-overlap proxy in :mod:`comparison`, so the default
  reproducible pipeline is unchanged and CI never needs a live model.
- The LLM-judge metrics are non-deterministic (temperature 0 reduces but does not
  remove this); they are a *complementary, externally-recognised* secondary
  signal — NOT a replacement for the deterministic numeric-grounding gate that is
  ARIA's real quality check. Report them as a secondary column.
"""
from __future__ import annotations

import os
import re
from typing import Iterable, Optional

from . import comparison

__all__ = [
    "llm_judge_enabled",
    "ragas_real_enabled",
    "ragas_real_eval",
    "faithfulness_llm",
    "context_precision_llm",
    "context_recall_llm",
    "answer_relevancy_llm",
    "ragas_eval",
]


def llm_judge_enabled() -> bool:
    """True iff MPH_RAGAS_LLM is set to a truthy value (default OFF)."""
    return os.environ.get("MPH_RAGAS_LLM", "0") not in ("", "0", "false", "False")


def _pick_judge(backend=None):
    """Resolve a judge backend: explicit arg, else the first available local one.

    Returns an ``LLMBackend`` or None. Prefers a local Ollama judge (offline,
    no API cost) via ``discover_backends``; never raises.
    """
    if backend is not None:
        return backend
    try:
        from .backends import discover_backends
        backends = discover_backends()
        # prefer ollama/local tier (offline); fall back to any available
        for tier in ("ollama", "local", "api"):
            for b in backends:
                if getattr(b, "tier", "") == tier and b.is_available():
                    return b
        for b in backends:
            if b.is_available():
                return b
    except Exception:
        pass
    return None


def _parse_score(text: str, key: str) -> Optional[float]:
    """Extract a 0-1 score from a judge reply.

    Looks for ``KEY = <float>`` first, then any standalone float clamped to
    [0, 1]. Returns None if nothing parseable (caller falls back to the proxy).
    """
    if not text:
        return None
    m = re.search(rf"{key}\s*[=:]\s*([01](?:\.\d+)?|0?\.\d+)", text, re.IGNORECASE)
    if not m:
        m = re.search(r"\b(0?\.\d+|1\.0+|[01])\b", text)
    if not m:
        return None
    try:
        return max(0.0, min(1.0, float(m.group(1))))
    except ValueError:
        return None


def _judge(backend, prompt: str, key: str) -> Optional[float]:
    """One judge call → parsed score, or None on any failure (proxy fallback)."""
    try:
        resp = backend.generate(prompt, temperature=0.0, max_tokens=256)
        if getattr(resp, "error", ""):
            return None
        return _parse_score(getattr(resp, "text", "") or "", key)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# The four RAGAS-methodology metrics (LLM-judge with deterministic proxy fallback)
# --------------------------------------------------------------------------- #
def faithfulness_llm(answer: str, contexts: Iterable[str], *, backend=None) -> dict:
    """RAGAS faithfulness via LLM judge: fraction of answer claims grounded in context.

    Args:
        answer: the model answer to score.
        contexts: retrieved context strings the answer should be grounded in.
        backend: optional explicit judge backend; else auto-resolved (local first).

    Returns:
        ``{faithfulness in [0,1], method: "llm"|"proxy", n_claims?, ...}``. When the
        judge is disabled/unavailable/unparseable, delegates to
        :func:`comparison.faithfulness` and marks ``method="proxy"``.

    Side effects: one LLM judge call when enabled (temperature 0); none otherwise.
    """
    ctx = list(contexts)
    if llm_judge_enabled():
        b = _pick_judge(backend)
        if b is not None:
            prompt = (
                "You are a strict factuality judge. Given CONTEXT and ANSWER, "
                "decompose the ANSWER into atomic factual claims and decide, for "
                "each, whether it is directly supported by the CONTEXT. Then output "
                "exactly one line: FAITHFULNESS=<supported_claims/total_claims as a "
                "decimal between 0 and 1>.\n\nCONTEXT:\n" + "\n".join(ctx)
                + "\n\nANSWER:\n" + (answer or "")
            )
            s = _judge(b, prompt, "FAITHFULNESS")
            if s is not None:
                return {"faithfulness": round(s, 3), "method": "llm",
                        "backend": getattr(b, "backend_id", "?")}
    proxy = comparison.faithfulness(answer, ctx)
    proxy["method"] = "proxy"
    return proxy


def context_precision_llm(question: str, contexts: Iterable[str], *, backend=None) -> dict:
    """RAGAS context precision via LLM judge: fraction of retrieved contexts relevant.

    Returns ``{context_precision in [0,1], method, ...}`` with proxy fallback. The
    deterministic proxy (:func:`comparison.context_precision`) needs a relevance
    label set; when falling back without one, precision is reported as None-safe.
    """
    ctx = list(contexts)
    if llm_judge_enabled():
        b = _pick_judge(backend)
        if b is not None:
            joined = "\n".join(f"[{i}] {c}" for i, c in enumerate(ctx))
            prompt = (
                "Given a QUESTION and a numbered list of retrieved CONTEXT passages, "
                "decide for each passage whether it is relevant to answering the "
                "question. Output exactly one line: PRECISION=<relevant_passages/"
                "total_passages as a decimal 0-1>.\n\nQUESTION:\n" + (question or "")
                + "\n\nCONTEXT:\n" + joined
            )
            s = _judge(b, prompt, "PRECISION")
            if s is not None:
                return {"context_precision": round(s, 3), "method": "llm",
                        "n_contexts": len(ctx), "backend": getattr(b, "backend_id", "?")}
    # proxy: needs relevant labels; without them, fall back to a self-consistent
    # all-retrieved-relevant baseline of None to avoid a misleading number.
    return {"context_precision": None, "method": "proxy",
            "note": "proxy context_precision needs a labelled relevant set; "
                    "enable MPH_RAGAS_LLM for the LLM-judge estimate", "n_contexts": len(ctx)}


def context_recall_llm(answer: str, contexts: Iterable[str], *, backend=None) -> dict:
    """RAGAS context recall via LLM judge: fraction of answer claims attributable to context.

    Returns ``{context_recall in [0,1], method, ...}`` with proxy fallback.
    """
    ctx = list(contexts)
    if llm_judge_enabled():
        b = _pick_judge(backend)
        if b is not None:
            prompt = (
                "Given an ANSWER (treated as the ground truth) and retrieved "
                "CONTEXT, decide what fraction of the answer's claims can be "
                "attributed to the context. Output exactly one line: RECALL="
                "<attributable_claims/total_claims as a decimal 0-1>.\n\nANSWER:\n"
                + (answer or "") + "\n\nCONTEXT:\n" + "\n".join(ctx)
            )
            s = _judge(b, prompt, "RECALL")
            if s is not None:
                return {"context_recall": round(s, 3), "method": "llm",
                        "backend": getattr(b, "backend_id", "?")}
    # deterministic proxy: token-coverage of answer claims by the context
    import re as _re
    ctx_tokens = set(_re.findall(r"[가-힣a-z0-9]+", " ".join(ctx).lower()))
    claims = [s.strip() for s in _re.split(r"[.!?。\n]+", answer or "") if len(s.strip()) > 4]
    covered = 0
    for c in claims:
        toks = set(_re.findall(r"[가-힣a-z0-9]+", c.lower()))
        if toks and len(toks & ctx_tokens) / len(toks) >= 0.2:
            covered += 1
    n = len(claims)
    return {"context_recall": round(covered / n, 3) if n else 1.0,
            "method": "proxy", "n_claims": n, "n_covered": covered}


def answer_relevancy_llm(question: str, answer: str, *, backend=None) -> dict:
    """RAGAS answer relevancy via LLM judge: how well the answer addresses the question.

    There is no deterministic proxy for relevancy in :mod:`comparison`; when the
    judge is unavailable this returns a token-overlap proxy between question and
    answer as a coarse, clearly-labelled fallback.
    """
    if llm_judge_enabled():
        b = _pick_judge(backend)
        if b is not None:
            prompt = (
                "Rate how directly and completely the ANSWER addresses the "
                "QUESTION, ignoring factual correctness. Output exactly one line: "
                "RELEVANCY=<decimal 0-1>.\n\nQUESTION:\n" + (question or "")
                + "\n\nANSWER:\n" + (answer or "")
            )
            s = _judge(b, prompt, "RELEVANCY")
            if s is not None:
                return {"answer_relevancy": round(s, 3), "method": "llm",
                        "backend": getattr(b, "backend_id", "?")}
    qt = set(re.findall(r"[가-힣a-z0-9]+", (question or "").lower()))
    at = set(re.findall(r"[가-힣a-z0-9]+", (answer or "").lower()))
    jac = len(qt & at) / max(1, len(qt | at))
    return {"answer_relevancy": round(jac, 3), "method": "proxy",
            "note": "token-overlap proxy; enable MPH_RAGAS_LLM for the LLM-judge score"}


def ragas_real_enabled() -> bool:
    """True iff MPH_RAGAS_REAL is truthy (default OFF -> LLM-judge/proxy path)."""
    return os.environ.get("MPH_RAGAS_REAL", "0") not in ("", "0", "false", "False")


def ragas_real_eval(question: str, answer: str, contexts: Iterable[str], *,
                    ground_truth: Optional[str] = None, timeout: int = 300) -> Optional[dict]:
    """Real RAGAS scores via the isolated .venv_ragas runner (subprocess).

    The reference ragas package is incompatible with the main env's langchain
    (1.3.x), so it runs in .venv_ragas (ragas 0.1.21 + langchain 0.2.x) and is
    invoked here as a subprocess. Offline: Ollama LLM judge (MPH_RAGAS_MODEL,
    default mistral:7b — a 7B model judges faithfulness reliably; a 3B model is
    noisy). Computes the three LLM-only metrics (faithfulness, context_precision,
    context_recall); answer_relevancy needs an offline embedder and is left to the
    LLM-judge path in :func:`ragas_eval`.

    Returns:
        ``{faithfulness, context_precision, context_recall, ragas_version, model,
        method}`` or None on any failure (missing venv, subprocess error, Ollama
        down) so the caller falls back to the LLM-judge/proxy path.

    Side effects: spawns one subprocess; writes + removes a temp JSON file.
    """
    import json as _json
    import subprocess
    import tempfile
    from pathlib import Path
    repo = Path(__file__).resolve().parents[2]
    venv_py = repo / ".venv_ragas" / "bin" / "python"
    runner = repo / "simulation" / "llm_compare" / "ragas_runner.py"
    if not venv_py.exists() or not runner.exists():
        return None
    rec = {"question": question or "", "answer": answer or "",
           "contexts": list(contexts), "ground_truth": ground_truth or answer or ""}
    tmp = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                         encoding="utf-8") as fh:
            _json.dump(rec, fh)
            tmp = fh.name
        proc = subprocess.run([str(venv_py), str(runner), tmp], capture_output=True,
                              text=True, timeout=timeout, cwd=str(repo))
        lines = (proc.stdout or "").strip().splitlines()
        out = _json.loads(lines[-1]) if lines else {}
        return None if out.get("error") else out
    except Exception:
        return None
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except Exception:
                pass


def ragas_eval(question: str, answer: str, contexts: Iterable[str], *,
               backend=None) -> dict:
    """Run all four RAGAS-methodology metrics over one (question, answer, contexts).

    Args:
        question: the user question.
        answer: the model answer.
        contexts: retrieved context passages.
        backend: optional explicit judge backend (else auto-resolved, local first).

    Returns:
        ``{faithfulness, context_precision, context_recall, answer_relevancy,
        method: "llm"|"proxy"|"mixed", llm_judge_enabled: bool}`` — each sub-metric
        carries its own ``method`` tag so a mixed (some-llm, some-proxy) run is
        transparent.

    Side effects: up to four LLM judge calls when enabled; none otherwise.
    """
    ctx = list(contexts)
    # Real ragas package (isolated venv) when MPH_RAGAS_REAL=1; falls through to the
    # LLM-judge/proxy path on any failure. answer_relevancy stays on the LLM judge
    # (the real ragas runner skips it — needs an offline embedder we do not have).
    if ragas_real_enabled():
        real = ragas_real_eval(question, answer, ctx)
        if real and any(real.get(m) is not None for m in
                        ("faithfulness", "context_precision", "context_recall")):
            ar = answer_relevancy_llm(question, answer, backend=backend)
            return {
                "faithfulness": real.get("faithfulness"),
                "context_precision": real.get("context_precision"),
                "context_recall": real.get("context_recall"),
                "answer_relevancy": ar.get("answer_relevancy"),
                "per_metric_method": {
                    "faithfulness": "ragas_package", "context_precision": "ragas_package",
                    "context_recall": "ragas_package", "answer_relevancy": ar["method"]},
                "method": "ragas_package",
                "ragas_version": real.get("ragas_version"),
                "ragas_model": real.get("model"),
                "llm_judge_enabled": llm_judge_enabled(),
            }
    b = _pick_judge(backend) if llm_judge_enabled() else None
    f = faithfulness_llm(answer, ctx, backend=b)
    p = context_precision_llm(question, ctx, backend=b)
    r = context_recall_llm(answer, ctx, backend=b)
    a = answer_relevancy_llm(question, answer, backend=b)
    methods = {f["method"], p["method"], r["method"], a["method"]}
    overall = ("llm" if methods == {"llm"} else "proxy" if methods == {"proxy"}
               else "mixed")
    return {
        "faithfulness": f.get("faithfulness"),
        "context_precision": p.get("context_precision"),
        "context_recall": r.get("context_recall"),
        "answer_relevancy": a.get("answer_relevancy"),
        "per_metric_method": {"faithfulness": f["method"], "context_precision": p["method"],
                              "context_recall": r["method"], "answer_relevancy": a["method"]},
        "method": overall,
        "llm_judge_enabled": llm_judge_enabled(),
    }
