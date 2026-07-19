"""simulation.llm_compare.dspy_refresh
========================================
Refresh the analyst few-shot exemplars from VERIFIED history (S3).

"Learning from history" at the PROMPT level (never weight updates that touch
epidemic numbers): select the best past gate-passed answers as few-shot exemplars
and persist them, so future syntheses can be primed with proven-grounded examples.

Two paths, both offline and read-only over the verified log:
  * **curated** (default, always available): pick the most recent, query-diverse
    verified answers. Deterministic, no dependency.
  * **DSPy** (optional, ``MPH_ARIA_DSPY=1`` and ``dspy`` importable): a best-effort
    BootstrapFewShot pass; on any setup failure it degrades to the curated pick.

Neither path alters tool-receipt numbers — exemplars are answer TEXT only, and the
delivery gate still checks every current answer against current tool receipts.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

__all__ = ["refresh_exemplars", "dspy_enabled"]


def dspy_enabled() -> bool:
    """True iff MPH_ARIA_DSPY is set AND dspy is importable (default OFF)."""
    if os.environ.get("MPH_ARIA_DSPY", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return False
    try:
        import dspy  # noqa: F401
        return True
    except Exception:
        return False


def _curated(recs: list[dict], max_exemplars: int) -> list[dict]:
    """Most-recent, query-diverse verified answers (deterministic selection)."""
    seen: set[str] = set()
    out: list[dict] = []
    for r in reversed(recs):  # most recent first
        key = (r.get("query", "")[:24]).strip().lower()
        if key in seen:
            continue
        seen.add(key)
        out.append({"query": r.get("query", ""), "answer": r.get("final_answer", "")})
        if len(out) >= max_exemplars:
            break
    return out


def refresh_exemplars(memory=None, *, out_path: Optional[str | Path] = None,
                      max_exemplars: int = 5) -> dict:
    """Rebuild the analyst exemplar file from the verified-answer memory.

    Args:
        memory: a :class:`~simulation.llm_compare.memory.VerifiedMemory` (default
            constructs the standard one).
        out_path: exemplar JSON destination (default ``<memory dir>/exemplars.json``).
        max_exemplars: cap on stored exemplars.

    Returns:
        ``{n_verified, n_exemplars, method, path}``. ``method`` is ``"dspy"`` or
        ``"curated"``. If there is no verified history, ``n_exemplars`` is 0 and
        nothing is written.

    Side effects: writes the exemplar JSON when at least one verified record exists.
    """
    from simulation.llm_compare.memory import VerifiedMemory
    memory = memory or VerifiedMemory()
    recs = memory.all()
    out = Path(out_path) if out_path is not None else memory.path.parent / "exemplars.json"

    if not recs:
        return {"n_verified": 0, "n_exemplars": 0, "method": "none", "path": str(out)}

    method = "curated"
    exemplars = _curated(recs, max_exemplars)
    if dspy_enabled():
        try:
            exemplars = _dspy_bootstrap(recs, max_exemplars) or exemplars
            method = "dspy"
        except Exception:
            method = "curated"  # degrade cleanly — never let DSPy break the refresh

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"exemplars": exemplars, "method": method},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    return {"n_verified": len(recs), "n_exemplars": len(exemplars),
            "method": method, "path": str(out)}


def _dspy_bootstrap(recs: list[dict], max_exemplars: int) -> list[dict]:  # pragma: no cover
    """Best-effort DSPy BootstrapFewShot over verified Q→A pairs (optional path).

    Requires a configured DSPy LM (e.g. Ollama); returns the bootstrapped demos as
    ``{query, answer}`` dicts, or raises (caller degrades to the curated pick).
    """
    import dspy  # noqa: F401
    # A configured LM is required; if the environment has not set one up, this
    # raises and the caller falls back to the deterministic curated selection.
    trainset = [dspy.Example(query=r.get("query", ""),
                             answer=r.get("final_answer", "")).with_inputs("query")
                for r in recs if r.get("final_answer")]
    if not trainset:
        raise RuntimeError("no verified examples for DSPy bootstrap")

    class _Advise(dspy.Signature):
        """Answer an epidemiology-advisory query, citing only provided facts."""
        query = dspy.InputField()
        answer = dspy.OutputField()

    def _metric(example, pred, trace=None):
        return bool(getattr(pred, "answer", "").strip())

    opt = dspy.BootstrapFewShot(metric=_metric, max_bootstrapped_demos=max_exemplars)
    compiled = opt.compile(dspy.Predict(_Advise), trainset=trainset[:32])
    demos = getattr(getattr(compiled, "predictor", compiled), "demos", []) or []
    return [{"query": getattr(d, "query", ""), "answer": getattr(d, "answer", "")}
            for d in demos[:max_exemplars]]
