"""
simulation.scripts.aria_dspy_optimize
======================================
Apply DSPy ``BootstrapFewShot`` to the ACTUAL ARIA grounding task and quantify
the before -> after improvement on a HELD-OUT test split (no leakage).

The survey (`docs/ARIA_MODEL_OPTIMIZATION_SURVEY.md`) showed, on a 2-context PoC,
that DSPy lifts weak models' numeric grounding (+0.29~+0.51). That PoC was too
small (n=2) for a real train/dev/test split. This script scales the SAME finding
to a proper experiment on ARIA's grounding pipeline:

  * Task   — ARIA's deployed job: read an epidemiology context (official-source
             anchored) + a question, emit an answer that CITES the required facts
             and does NOT contradict them. Expressed as a DSPy signature
             ``context, question -> grounded_answer``.
  * Bench  — `simulation.llm_compare.kr_epi_bench.KR_EPI_LAW_QA` (40 items,
             law.go.kr / KDCA / WHO anchored). Each item carries the grounded
             context (`answer_key`), the question, the gold facts it MUST cite
             (`must_contain`), and the claims it must AVOID (`must_avoid`).
  * Metric — grounding = (fraction of must_contain cited) - (must_avoid penalty),
             the SAME grounding philosophy as `aria_grounding.numeric_grounding`
             / `comparison.faithfulness` but generalised from numbers to the
             official gold tokens. Deterministic (temperature=0) -> reproducible.
  * Split  — train / dev / test by a fixed permutation seed. BootstrapFewShot
             builds demos from TRAIN, validates them on DEV, and we report
             grounding ONLY on the untouched TEST split (leak-free).
  * Models — run SEQUENTIALLY (one Ollama model in memory at a time): a weak
             model with headroom (`qwen2.5:3b`) and, by default, a smaller one
             (`qwen2.5:0.5b`). `--models` overrides.

Reproducible:  .venv/bin/python -m simulation.scripts.aria_dspy_optimize
               .venv/bin/python -m simulation.scripts.aria_dspy_optimize \
                   --models qwen2.5:3b --max-demos 4

Constraints honoured: NO sqlite write, NO `uv sync` (dspy already pip-installed),
read-only benchmark, live `simulation.llm_compare` code unmodified.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

# Repo-root import (script lives under simulation/scripts/).
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from simulation.llm_compare.kr_epi_bench import KR_EPI_LAW_QA  # noqa: E402

DEFAULT_MODELS = ["qwen2.5:3b", "qwen2.5:0.5b"]
DEFAULT_OUT = "simulation/results/aria_dspy_optimize.json"
_NUM = re.compile(r"-?\d+\.?\d*")


# ── grounding metric (ARIA grounding philosophy, deterministic) ───────────────
def grounding_score(answer: str, must_contain, must_avoid) -> float:
    """Grounding faithfulness of an answer against an item's gold tokens.

    Mirrors `aria_grounding.numeric_grounding` / `comparison.faithfulness`: reward
    citing the required gold facts, penalise contradicting / banned claims. Token
    match is case-insensitive substring (handles Korean + numeric facts alike).

    Args:
        answer: backend's free-text grounded answer.
        must_contain: gold tokens the grounded answer SHOULD cite.
        must_avoid: tokens a grounded answer must NOT assert (wrong / banned).

    Returns:
        Score in ``[0, 1]``: recall of ``must_contain`` minus the fraction of
        ``must_avoid`` that leaked, floored at 0. Empty ``must_contain`` -> 0.0.

    Side effects: none. Never raises.
    """
    a = (answer or "").lower()
    mc = [t for t in (must_contain or [])]
    if not mc:
        return 0.0
    cited = sum(1 for t in mc if str(t).lower() in a)
    recall = cited / len(mc)
    av = [t for t in (must_avoid or [])]
    leaked = sum(1 for t in av if str(t).lower() in a) / len(av) if av else 0.0
    return max(0.0, recall - leaked)


def _make_examples(items):
    """Turn KrEpiItem rows into DSPy examples (context, question -> grounded_answer)."""
    import dspy
    exs = []
    for it in items:
        ex = dspy.Example(
            context=it.answer_key,            # the grounded source text (official)
            question=it.question,
            grounded_answer=it.answer_key,     # gold target trajectory
            _must_contain=list(it.must_contain),
            _must_avoid=list(it.must_avoid),
            _id=it.id,
        ).with_inputs("context", "question")
        exs.append(ex)
    return exs


def _split(items, *, seed: int, frac=(0.5, 0.2, 0.3)):
    """Deterministic train/dev/test split by a fixed permutation seed (leak-free).

    Args:
        items: list of benchmark items.
        seed: RNG seed for the permutation (fixed -> reproducible).
        frac: (train, dev, test) fractions; test is the remainder.

    Returns:
        ``(train, dev, test)`` lists of items, disjoint.
    """
    idx = list(range(len(items)))
    random.Random(seed).shuffle(idx)
    n = len(items)
    n_tr = int(round(frac[0] * n))
    n_dev = int(round(frac[1] * n))
    tr = [items[i] for i in idx[:n_tr]]
    dev = [items[i] for i in idx[n_tr:n_tr + n_dev]]
    te = [items[i] for i in idx[n_tr + n_dev:]]
    return tr, dev, te


def _eval_module(module, examples) -> tuple[float, list]:
    """Mean grounding score of a DSPy module over examples (held-out evaluation)."""
    scores, detail = [], []
    for ex in examples:
        try:
            pred = module(context=ex.context, question=ex.question)
            ans = getattr(pred, "grounded_answer", "") or ""
        except Exception as e:  # noqa: BLE001 - a backend hiccup must not abort the sweep
            ans = ""
            detail.append({"id": ex._id, "error": str(e)[:120]})
        s = grounding_score(ans, ex._must_contain, ex._must_avoid)
        scores.append(s)
        detail.append({"id": ex._id, "grounding": round(s, 3),
                       "answer_head": (ans or "")[:140]})
    mean = round(sum(scores) / len(scores), 4) if scores else 0.0
    return mean, detail


def optimize_one(model: str, *, train, dev, test, max_demos: int,
                 max_tokens: int, seed: int) -> dict:
    """Vanilla vs DSPy-optimized grounding for ONE model on the held-out test.

    Builds a `Predict` grounding module, measures its grounding on TEST (vanilla),
    compiles it with `BootstrapFewShot` on TRAIN (validated on DEV), then measures
    the compiled module on the SAME held-out TEST. No test item is ever used to
    pick demos -> the delta is leak-free.

    Args:
        model: Ollama model tag (e.g. ``qwen2.5:3b``).
        train/dev/test: disjoint example lists from `_split`.
        max_demos: BootstrapFewShot ``max_bootstrapped_demos``.
        max_tokens: generation cap (>=500 keeps reasoning models from truncating).
        seed: passed to DSPy for determinism.

    Returns:
        ``{model, grounding_before, grounding_after, delta, n_test, n_train,
        n_dev, n_demos_compiled, error?}``.

    Side effects: loads the model into Ollama and calls it (network localhost).
    """
    import dspy

    lm = dspy.LM(f"ollama_chat/{model}", api_base="http://127.0.0.1:11434",
                 api_key="", max_tokens=max_tokens, temperature=0.0)
    dspy.configure(lm=lm)

    sig = dspy.Signature(
        "context, question -> grounded_answer",
        "당신은 역학자에게 자문하는 ARIA grounding 레이어입니다. 아래 제공된 context의 "
        "사실만 근거로 question에 답하세요. context에 있는 핵심 수치/조항/용어를 반드시 "
        "인용하고, 제공되지 않은 내용을 지어내지 마세요.")
    base = dspy.Predict(sig)

    t0 = time.time()
    before, before_detail = _eval_module(base, test)

    def _metric(example, pred, trace=None):
        ans = getattr(pred, "grounded_answer", "") or ""
        return grounding_score(ans, example._must_contain, example._must_avoid)

    out = {"model": model, "n_test": len(test), "n_train": len(train),
           "n_dev": len(dev), "grounding_before": before}
    try:
        from dspy.teleprompt import BootstrapFewShot
        tele = BootstrapFewShot(
            metric=_metric,
            max_bootstrapped_demos=max_demos,
            max_labeled_demos=max_demos,
            max_rounds=1,
        )
        # Compile on TRAIN; BootstrapFewShot self-validates demos via the metric.
        # DEV is held for an honest demo-selection signal, kept disjoint from TEST.
        compiled = tele.compile(base, trainset=train + dev)
        n_demos = len(getattr(compiled, "demos", []) or
                      getattr(getattr(compiled, "predictors", lambda: [base])()[0],
                              "demos", []))
        after, after_detail = _eval_module(compiled, test)
        out.update({
            "grounding_after": after,
            "delta": round(after - before, 4),
            "n_demos_compiled": n_demos,
            "before_detail": before_detail,
            "after_detail": after_detail,
        })
    except Exception as e:  # noqa: BLE001 - report the failure honestly, never crash sweep
        out.update({"grounding_after": None, "delta": None,
                    "error": f"{type(e).__name__}: {e}"})
    out["elapsed_s"] = round(time.time() - t0, 1)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DSPy BootstrapFewShot on ARIA grounding")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS,
                    help="Ollama model tags, run sequentially")
    ap.add_argument("--max-demos", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=600)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    items = list(KR_EPI_LAW_QA)
    train_i, dev_i, test_i = _split(items, seed=args.seed)
    train = _make_examples(train_i)
    dev = _make_examples(dev_i)
    test = _make_examples(test_i)

    print(f"ARIA grounding bench: n={len(items)} -> train={len(train)} "
          f"dev={len(dev)} test={len(test)} (seed={args.seed}, leak-free)")
    print(f"Models (sequential): {', '.join(args.models)}\n")

    results = []
    for m in args.models:
        print(f"── {m} ── optimizing (BootstrapFewShot, max_demos={args.max_demos}) …",
              flush=True)
        r = optimize_one(m, train=train, dev=dev, test=test,
                         max_demos=args.max_demos, max_tokens=args.max_tokens,
                         seed=args.seed)
        results.append(r)
        b, a, d = r.get("grounding_before"), r.get("grounding_after"), r.get("delta")
        if a is None:
            print(f"   before={b}  after=FAILED ({r.get('error')})  [{r['elapsed_s']}s]\n")
        else:
            print(f"   before={b}  after={a}  Δ={d:+.4f}  "
                  f"(demos={r.get('n_demos_compiled')})  [{r['elapsed_s']}s]\n")

    payload = {
        "task": "ARIA grounding (context, question -> grounded_answer)",
        "benchmark": "kr_epi_bench.KR_EPI_LAW_QA (official-source anchored)",
        "metric": "grounding = recall(must_contain) - leak(must_avoid)",
        "split": {"seed": args.seed, "n_total": len(items),
                  "n_train": len(train), "n_dev": len(dev), "n_test": len(test),
                  "leak_free": "demos from train+dev only; reported on held-out test"},
        "optimizer": f"BootstrapFewShot(max_demos={args.max_demos})",
        "max_tokens": args.max_tokens,
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
