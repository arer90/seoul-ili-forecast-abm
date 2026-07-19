"""
simulation.scripts.aria_grounded_vs_ungrounded
================================================
Sub-analysis 2 — ★ GROUNDED vs UNGROUNDED significance.

The core "is ARIA's grounding doing anything?" test. Each model answers the
SAME questions in two conditions:

  • GROUNDED   — the real numeric context (RAG) is included in the prompt
                 (exactly the shipped ``GROUNDING_PROMPT + context``).
  • UNGROUNDED — the SAME question is asked with NO context (the model must
                 answer from parametric memory only — it has never seen these
                 numbers, so a faithful/recalled answer is essentially
                 impossible without retrieval).

Per ITEM (item = model × context × sub-fact for fact-recall; model × context for
faithfulness) we record the grounded and ungrounded score, then run a PAIRED
test across all items pooled over models:

  • Wilcoxon signed-rank (primary, non-parametric, robust to non-normal scores)
  • paired t-test (secondary)
  • effect size: rank-biserial (Wilcoxon) + Cohen's d_z (paired)

H0: grounding does not change faithfulness/fact-recall. A significant positive
shift = ARIA's retrieval grounding genuinely improves factual answers.

⚠ SEQUENTIAL Ollama — models iterated one at a time; within a model the two
conditions are run back-to-back with a fixed seed for reproducibility.

Run:
  .venv/bin/python -m simulation.scripts.aria_grounded_vs_ungrounded
  .venv/bin/python -m simulation.scripts.aria_grounded_vs_ungrounded --models mistral:7b
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time

from simulation.llm_compare.aria_grounding import GROUNDING_PROMPT, numeric_grounding
from simulation.llm_compare.comparison import faithfulness
from simulation.scripts.aria_sci_common import (
    ARIA_OLLAMA_MODELS, MODEL_LABELS, SeededOllamaBackend, out_dir, real_contexts,
)

_NUM = re.compile(r"-?\d+\.?\d*")

# Ungrounded prompt: same advisory framing, same question, but NO numbers given.
UNGROUNDED_PROMPT = (
    "당신은 역학자에게 시뮬레이션 결과를 해석해 주는 자문가입니다. "
    "아래 분석에 대해 2~3문장으로 핵심 수치(파라미터/R²/RMSE 등)를 포함해 해석하세요.\n\n"
)


def _fact_hits(answer: str, facts) -> list[int]:
    """Per-gold-fact binary recall vector (1 if that number appears in answer).

    Returns a list aligned to ``facts`` order — these are the paired ITEMS for
    the fact-recall significance test.
    """
    ans = set(_NUM.findall(answer or ""))
    hits = []
    for f in facts:
        m = _NUM.search(str(f).split("=")[-1])
        hits.append(1 if (m and m.group() in ans) else 0)
    return hits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ARIA grounded vs ungrounded significance")
    ap.add_argument("--models", default=",".join(ARIA_OLLAMA_MODELS))
    ap.add_argument("--seed", type=int, default=20260628)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--out", default="simulation/results/aria_sci/aria_grounded_vs_ungrounded.json")
    args = ap.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    contexts = real_contexts()
    t0 = time.time()

    # paired item arrays (pooled across models)
    faith_g, faith_u = [], []          # one pair per (model, context)
    recall_g, recall_u = [], []        # one pair per (model, context, gold-fact)
    per_model = {}
    items = []

    for model in models:  # ── SEQUENTIAL
        be = SeededOllamaBackend(model)
        if not be.is_available():
            per_model[model] = {"error": "unavailable"}
            print(f"[skip] {model} unavailable", file=sys.stderr)
            continue
        mf_g, mf_u, mr_g, mr_u, n_err = [], [], [], [], 0
        for ctx in contexts:
            # build the question text shared by both conditions
            question = ("핵심 수치를 모두 포함해 이 분석을 해석하세요. "
                        f"(분석 ID: {ctx['id']})")
            # GROUNDED: context provided
            rg = be.generate(GROUNDING_PROMPT + ctx["context"],
                            max_tokens=220, temperature=args.temperature, seed=args.seed)
            # UNGROUNDED: same advisory question, NO numbers
            ru = be.generate(UNGROUNDED_PROMPT + question,
                            max_tokens=220, temperature=args.temperature, seed=args.seed)
            if rg.error or ru.error:
                n_err += 1
                continue
            fg = faithfulness(rg.text, [ctx["context"]])["faithfulness"]
            fu = faithfulness(ru.text, [ctx["context"]])["faithfulness"]
            faith_g.append(fg); faith_u.append(fu)
            mf_g.append(fg); mf_u.append(fu)
            hg = _fact_hits(rg.text, ctx["facts"])
            hu = _fact_hits(ru.text, ctx["facts"])
            for a, b in zip(hg, hu):
                recall_g.append(a); recall_u.append(b)
                mr_g.append(a); mr_u.append(b)
            items.append({"model": model, "context": ctx["id"],
                          "faith_grounded": round(fg, 3), "faith_ungrounded": round(fu, 3),
                          "recall_grounded": round(sum(hg) / len(hg), 3),
                          "recall_ungrounded": round(sum(hu) / len(hu), 3)})
            print(f"  {model:14s} {ctx['id']:18s} "
                  f"faith {fg:.2f}/{fu:.2f}  recall {sum(hg)/len(hg):.2f}/{sum(hu)/len(hu):.2f}")
        per_model[model] = {
            "label": MODEL_LABELS.get(model, model),
            "faith_grounded_mean": round(sum(mf_g) / len(mf_g), 4) if mf_g else None,
            "faith_ungrounded_mean": round(sum(mf_u) / len(mf_u), 4) if mf_u else None,
            "recall_grounded_mean": round(sum(mr_g) / len(mr_g), 4) if mr_g else None,
            "recall_ungrounded_mean": round(sum(mr_u) / len(mr_u), 4) if mr_u else None,
            "n_errors": n_err,
        }

    # ── paired significance tests (pooled items) ─────────────────────────────
    import numpy as np
    from scipy.stats import ttest_rel, wilcoxon

    def paired_block(g, u, label):
        g = np.asarray(g, float); u = np.asarray(u, float)
        d = g - u
        n = int(len(d))
        n_nonzero = int(np.count_nonzero(d))
        block = {"label": label, "n_items": n, "n_nonzero_diff": n_nonzero,
                 "mean_grounded": round(float(g.mean()), 4) if n else None,
                 "mean_ungrounded": round(float(u.mean()), 4) if n else None,
                 "mean_diff": round(float(d.mean()), 4) if n else None}
        # Wilcoxon needs at least one non-zero diff
        if n_nonzero >= 1:
            try:
                w_stat, w_p = wilcoxon(g, u, zero_method="wilcox", alternative="greater")
                # rank-biserial effect size
                rbc = 1.0 - (2.0 * w_stat) / (n_nonzero * (n_nonzero + 1) / 2.0)
                block["wilcoxon_stat"] = round(float(w_stat), 3)
                block["wilcoxon_p_onesided_greater"] = float(w_p)
                block["rank_biserial_effect"] = round(float(rbc), 3)
            except Exception as e:  # noqa: BLE001
                block["wilcoxon_error"] = str(e)
        else:
            block["wilcoxon_note"] = "all paired diffs zero — no rank test"
        # paired t + Cohen's d_z
        if n >= 2 and d.std(ddof=1) > 0:
            t_stat, t_p_two = ttest_rel(g, u)
            t_p_one = t_p_two / 2.0 if t_stat > 0 else 1.0 - t_p_two / 2.0
            dz = float(d.mean() / d.std(ddof=1))
            block["paired_t_stat"] = round(float(t_stat), 3)
            block["paired_t_p_onesided_greater"] = round(float(t_p_one), 6)
            block["cohens_dz"] = round(dz, 3)
        return block

    faith_test = paired_block(faith_g, faith_u, "faithfulness")
    recall_test = paired_block(recall_g, recall_u, "numeric_fact_recall")

    payload = {
        "description": ("Grounded (RAG context) vs Ungrounded (no context, same "
                        "questions) paired comparison across all 5 Ollama models "
                        "on the 2 REAL thesis contexts. Wilcoxon signed-rank "
                        "(primary) + paired t (secondary), one-sided H1: grounding "
                        "improves. Core 'is ARIA's grounding doing anything' test."),
        "seed": args.seed, "temperature": args.temperature,
        "contexts": [{"id": c["id"], "source": c["source"]} for c in contexts],
        "per_model": per_model,
        "paired_tests": {"faithfulness": faith_test, "numeric_fact_recall": recall_test},
        "items": items,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out = out_dir() / "aria_grounded_vs_ungrounded.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[faithfulness]  grounded={faith_test['mean_grounded']} vs "
          f"ungrounded={faith_test['mean_ungrounded']}  "
          f"Wilcoxon p={faith_test.get('wilcoxon_p_onesided_greater')}")
    print(f"[fact_recall]   grounded={recall_test['mean_grounded']} vs "
          f"ungrounded={recall_test['mean_ungrounded']}  "
          f"Wilcoxon p={recall_test.get('wilcoxon_p_onesided_greater')}")
    print(f"wrote {out}  (elapsed {payload['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
