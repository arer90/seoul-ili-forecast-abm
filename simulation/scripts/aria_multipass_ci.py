"""
simulation.scripts.aria_multipass_ci
======================================
Sub-analysis 1 — ★ MULTI-PASS CI.

Re-runs the EXISTING ARIA grounding eval (numeric grounding + Self-Ask, the
``simulation.llm_compare.aria_grounding`` infra that produced
``aria_grounding_multi_llm.json``) N times per model, varying temperature/seed
each pass, and converts the single-pass POINT estimates into DISTRIBUTIONS with
uncertainty: mean ± 95% CI (+ sd) per metric per model.

Metrics per pass (averaged over the two REAL contexts, exactly as the shipped
``grounding_eval`` / ``self_ask_grounding`` do):
  • numeric_faithfulness   (comparison.faithfulness, topical groundedness)
  • numeric_fact_recall    (numeric_grounding, cites the REAL numbers)
  • grounding_score        (0.5*faithfulness + 0.5*fact_recall — a single
                            grounding headline so the CI table has one column
                            per axis the reviewer asked for)
  • selfask_faithfulness / selfask_fact_recall (Self-Ask axis)

⚠ SEQUENTIAL Ollama: models iterated one at a time; passes inside a model are
also sequential. Per pass we set temperature on a fixed schedule and seed=pass
so every pass is individually reproducible yet varied.

Run:
  .venv/bin/python -m simulation.scripts.aria_multipass_ci --passes 10
  .venv/bin/python -m simulation.scripts.aria_multipass_ci --passes 5 --models mistral:7b,llama3.2:1b
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from simulation.llm_compare.aria_grounding import (
    GROUNDING_PROMPT, SELF_ASK_PROMPT, numeric_grounding, self_ask_answer,
)
from simulation.llm_compare.comparison import faithfulness
from simulation.scripts.aria_sci_common import (
    ARIA_OLLAMA_MODELS, MODEL_LABELS, SeededOllamaBackend, mean_sd_ci95,
    out_dir, real_contexts,
)

# Temperature schedule: spread across the low/mid range typical for grounded
# interpretation. Cycled if --passes exceeds its length.
TEMP_SCHEDULE = [0.0, 0.2, 0.3, 0.5, 0.7, 0.1, 0.4, 0.6, 0.8, 0.9]


def _one_pass(backend, contexts, temperature, seed):
    """One full grounding pass for one backend → per-axis scalar scores.

    Returns dict of axis -> mean-over-contexts score (None if all errored).
    """
    num_faith, num_recall, num_spur = [], [], 0
    sa_faith, sa_recall, sa_spur = [], [], 0
    n_err = 0
    for ctx in contexts:
        # (a) direct numeric grounding
        r = backend.generate(GROUNDING_PROMPT + ctx["context"],
                             max_tokens=200, temperature=temperature, seed=seed)
        if r.error:
            n_err += 1
        else:
            num_faith.append(faithfulness(r.text, [ctx["context"]])["faithfulness"])
            ng = numeric_grounding(r.text, ctx["facts"])
            num_recall.append(ng["fact_recall"])
            num_spur += ng["n_spurious"]
        # (b) Self-Ask decomposition (same context, same seed)
        ref = self_ask_answer(ctx)
        subq_block = "\n".join(f"- {s['sub_q']}" for s in ref["sub_questions"])
        prompt = (SELF_ASK_PROMPT + ctx["context"]
                  + "\n\n하위질문(각각 한 줄로 답하세요):\n" + subq_block)
        r2 = backend.generate(prompt, max_tokens=320, temperature=temperature, seed=seed)
        if r2.error:
            n_err += 1
        else:
            sa_faith.append(faithfulness(r2.text, [ctx["context"]])["faithfulness"])
            ng2 = numeric_grounding(r2.text, ctx["facts"])
            sa_recall.append(ng2["fact_recall"])
            sa_spur += ng2["n_spurious"]

    def _avg(xs):
        return sum(xs) / len(xs) if xs else None

    nf, nr = _avg(num_faith), _avg(num_recall)
    sf, sr = _avg(sa_faith), _avg(sa_recall)
    grounding = None
    if nf is not None and nr is not None:
        grounding = 0.5 * nf + 0.5 * nr
    return {
        "numeric_faithfulness": nf,
        "numeric_fact_recall": nr,
        "numeric_spurious": num_spur,
        "grounding_score": grounding,
        "selfask_faithfulness": sf,
        "selfask_fact_recall": sr,
        "selfask_spurious": sa_spur,
        "n_errors": n_err,
    }


AXES = ["numeric_faithfulness", "numeric_fact_recall", "grounding_score",
        "selfask_faithfulness", "selfask_fact_recall"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ARIA multi-pass grounding CI")
    ap.add_argument("--passes", type=int, default=10, help="N passes per model (>=5)")
    ap.add_argument("--models", default=",".join(ARIA_OLLAMA_MODELS),
                    help="comma-separated Ollama tags")
    ap.add_argument("--out", default="simulation/results/aria_sci/aria_multipass_ci.json")
    args = ap.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    contexts = real_contexts()
    t0 = time.time()

    per_model = {}
    for model in models:  # ── SEQUENTIAL across models (one loaded at a time)
        be = SeededOllamaBackend(model)
        if not be.is_available():
            per_model[model] = {"error": "unavailable"}
            print(f"[skip] {model} unavailable", file=sys.stderr)
            continue
        passes = []
        for p in range(args.passes):  # ── SEQUENTIAL passes
            temp = TEMP_SCHEDULE[p % len(TEMP_SCHEDULE)]
            seed = 1000 + p
            rec = _one_pass(be, contexts, temperature=temp, seed=seed)
            rec.update({"pass": p, "temperature": temp, "seed": seed})
            passes.append(rec)
            g = rec["grounding_score"]
            print(f"  {model:14s} pass {p+1}/{args.passes} (T={temp}) "
                  f"grounding={g:.3f}" if g is not None else
                  f"  {model:14s} pass {p+1}/{args.passes} (T={temp}) grounding=NA")
        ci = {ax: mean_sd_ci95([pp[ax] for pp in passes]) for ax in AXES}
        per_model[model] = {"label": MODEL_LABELS.get(model, model),
                            "n_passes": len(passes), "passes": passes, "ci": ci}
        print(f"[done] {model}: grounding mean={ci['grounding_score']['mean']} "
              f"± {ci['grounding_score']['half_width']} (95% CI)")

    payload = {
        "description": ("Multi-pass CI for ARIA grounding eval on REAL thesis "
                        "outputs (ABM forward + ABM real-wave). Each model run N "
                        "times with varied temperature+seed; point estimates -> "
                        "mean ± 95% CI (t-based) per metric. Uses existing "
                        "aria_grounding infra. SEQUENTIAL Ollama."),
        "n_passes": args.passes,
        "temperature_schedule": TEMP_SCHEDULE[:args.passes],
        "contexts": [{"id": c["id"], "source": c["source"]} for c in contexts],
        "axes": AXES,
        "per_model": per_model,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out = out_dir() / "aria_multipass_ci.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}  (elapsed {payload['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
