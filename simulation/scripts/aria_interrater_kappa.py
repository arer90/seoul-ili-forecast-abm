"""
simulation.scripts.aria_interrater_kappa
==========================================
Sub-analysis 3 — ★ INTER-RATER kappa.

Shows the LLM-judge used to score ARIA grounding is RELIABLE (not arbitrary):
2-3 different Ollama models act as independent JUDGES rating the SAME set of
ARIA outputs on a binary "is this answer grounded in the provided numbers?"
verdict, and we compute inter-rater agreement.

Design (leak-free, reproducible):
  1. Build a fixed RATING SET of ARIA outputs over the two REAL contexts. To get
     genuine rating variance (a panel that only ever sees perfect answers cannot
     demonstrate discriminating agreement), each item is one of:
        • GROUNDED   — the corpus-free Self-Ask reference answer (carries the
                       real numbers) → judges SHOULD say grounded.
        • UNGROUNDED — a perturbed answer whose numbers are corrupted/removed
                       → judges SHOULD say NOT grounded.
     The TRUE label is known, so we also report each judge's accuracy vs truth.
  2. Each judge model rates every item independently (parse YES/NO). Judges run
     SEQUENTIALLY (one model loaded at a time), each item at temperature 0,
     fixed seed → reproducible verdicts.
  3. Agreement:
        • Cohen's kappa for every judge PAIR (sklearn)
        • Fleiss' kappa across all judges (categorical)
        • Krippendorff's alpha (nominal) if the package is importable; else a
          self-contained nominal-alpha fallback.

Run:
  .venv/bin/python -m simulation.scripts.aria_interrater_kappa
  .venv/bin/python -m simulation.scripts.aria_interrater_kappa --judges mistral:7b,qwen2.5:3b,phi3.5:3.8b
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import time

from simulation.llm_compare.aria_grounding import self_ask_answer
from simulation.scripts.aria_sci_common import (
    MODEL_LABELS, SeededOllamaBackend, out_dir, real_contexts,
)

# Default judge panel — 3 mid-capability models with different families.
DEFAULT_JUDGES = ["mistral:7b", "qwen2.5:3b", "phi3.5:3.8b"]

JUDGE_PROMPT = (
    "당신은 엄격한 사실성 심사관입니다. 아래 [제공된 수치]와 [응답]을 보고, "
    "응답이 *제공된 수치에만* 근거하여 정확한지 판정하세요. "
    "응답이 제공된 수치를 올바르게 인용하면 GROUNDED, "
    "수치를 누락/왜곡/날조했으면 NOT_GROUNDED 입니다. "
    "반드시 첫 줄에 GROUNDED 또는 NOT_GROUNDED 한 단어만 쓰세요.\n\n"
    "[제공된 수치]\n{facts}\n\n[응답]\n{answer}\n\n판정:"
)


def _corrupt_numbers(ref_answer: str, scale: float = 1.73, shift: float = 0.41) -> str:
    """Replace every number with a deterministically wrong one (UNGROUNDED).

    A faithful judge must flag this — the numbers no longer match the gold.
    """
    def repl(m):
        s = m.group()
        try:
            v = float(s)
            return str(round(v * scale + shift, 3))
        except ValueError:
            return s
    return re.sub(r"-?\d+\.?\d*", repl, ref_answer)


def _strip_numbers(ref_answer: str) -> str:
    """Remove the numbers entirely, leaving vague prose (UNGROUNDED — no facts)."""
    return re.sub(r"-?\d+\.?\d*", "X", ref_answer)


def _build_rating_set():
    """Fixed list of {item_id, facts, answer, true_label} rating items.

    Per context we build a BALANCED, larger panel-rating set so inter-rater
    kappa is well-defined (a 4-item set is too small for a stable estimate):

      GROUNDED (label-positive):
        • g0  Self-Ask reference answer (all real numbers)
        • g1  reference answer rephrased with a leading sentence (numbers intact)
      NOT_GROUNDED (label-negative):
        • u0  all numbers corrupted (scale+shift)
        • u1  all numbers corrupted with a different perturbation
        • u2  numbers stripped to 'X' (vague, ungrounded prose)

    → 5 items × 2 contexts = 10 items, label-balanced enough for Cohen/Fleiss/
    Krippendorff. Every item has a known TRUE label (judge accuracy reported).
    """
    items = []
    for ctx in real_contexts():
        ref = self_ask_answer(ctx)
        facts_str = ", ".join(ctx["facts"])
        ans = ref["final_answer"]
        rephrased = "분석 결과를 요약하면 다음과 같습니다. " + ans
        variants = [
            (f"{ctx['id']}::g0", ans, "GROUNDED"),
            (f"{ctx['id']}::g1", rephrased, "GROUNDED"),
            (f"{ctx['id']}::u0", _corrupt_numbers(ans, 1.73, 0.41), "NOT_GROUNDED"),
            (f"{ctx['id']}::u1", _corrupt_numbers(ans, 0.37, 1.9), "NOT_GROUNDED"),
            (f"{ctx['id']}::u2", _strip_numbers(ans), "NOT_GROUNDED"),
        ]
        for item_id, answer, label in variants:
            items.append({"item_id": item_id, "context_id": ctx["id"],
                          "facts": facts_str, "answer": answer, "true_label": label})
    return items


def _parse_verdict(text: str) -> str:
    """Map a judge's free text to GROUNDED / NOT_GROUNDED / UNPARSED."""
    t = (text or "").upper()
    # check NOT_GROUNDED first (it contains 'GROUNDED' as substring)
    if "NOT_GROUNDED" in t or "NOT GROUNDED" in t or "NOTGROUNDED" in t:
        return "NOT_GROUNDED"
    if "GROUNDED" in t:
        return "GROUNDED"
    if "근거 없" in (text or "") or "날조" in (text or ""):
        return "NOT_GROUNDED"
    if "근거" in (text or ""):
        return "GROUNDED"
    return "UNPARSED"


def _fleiss_kappa(ratings_matrix):
    """Fleiss' kappa for N items × categories count matrix (n raters per item).

    ratings_matrix: list of [count_cat0, count_cat1, ...] with equal row sums.
    """
    import numpy as np
    M = np.asarray(ratings_matrix, float)
    n_items, n_cat = M.shape
    n_raters = M.sum(axis=1)
    if not (n_raters == n_raters[0]).all() or n_raters[0] < 2:
        return None
    n = n_raters[0]
    p_j = M.sum(axis=0) / (n_items * n)
    P_i = (np.square(M).sum(axis=1) - n) / (n * (n - 1))
    P_bar = P_i.mean()
    P_e = float(np.square(p_j).sum())
    if abs(1 - P_e) < 1e-12:
        return 1.0
    return float((P_bar - P_e) / (1 - P_e))


def _krippendorff_alpha_nominal(data):
    """Nominal Krippendorff's alpha. data: list of rater rows, each a list of
    labels (or None for missing) aligned by item. Self-contained fallback."""
    import numpy as np
    raters = len(data)
    n_items = len(data[0])
    # build per-item value lists
    cols = []
    for j in range(n_items):
        vals = [data[r][j] for r in range(raters) if data[r][j] is not None]
        if len(vals) >= 2:
            cols.append(vals)
    if not cols:
        return None
    # observed disagreement
    Do_num, Do_den = 0.0, 0.0
    for vals in cols:
        m = len(vals)
        for a, b in itertools.permutations(vals, 2):
            Do_num += (a != b)
        Do_den += (m - 1)
    Do = Do_num / Do_den if Do_den else 0.0
    # expected disagreement
    allv = [v for vals in cols for v in vals]
    N = len(allv)
    from collections import Counter
    cnt = Counter(allv)
    De_num = 0.0
    for a in cnt:
        for b in cnt:
            if a != b:
                De_num += cnt[a] * cnt[b]
    De = De_num / (N * (N - 1)) if N > 1 else 0.0
    if De == 0:
        return 1.0
    return float(1.0 - Do / De)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ARIA judge inter-rater kappa")
    ap.add_argument("--judges", default=",".join(DEFAULT_JUDGES))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="simulation/results/aria_sci/aria_interrater_kappa.json")
    args = ap.parse_args(argv)

    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    rating_set = _build_rating_set()
    t0 = time.time()

    # judge × item verdict matrix
    verdicts = {}     # judge -> [labels aligned to rating_set]
    judge_acc = {}
    available_judges = []
    for jm in judges:  # ── SEQUENTIAL judges
        be = SeededOllamaBackend(jm)
        if not be.is_available():
            print(f"[skip] judge {jm} unavailable", file=sys.stderr)
            continue
        labels, correct, parsed = [], 0, 0
        for it in rating_set:
            prompt = JUDGE_PROMPT.format(facts=it["facts"], answer=it["answer"])
            r = be.generate(prompt, max_tokens=40, temperature=0.0, seed=args.seed)
            v = _parse_verdict(r.text) if not r.error else "UNPARSED"
            labels.append(v)
            if v != "UNPARSED":
                parsed += 1
                if v == it["true_label"]:
                    correct += 1
            print(f"  judge {jm:14s} {it['item_id']:28s} -> {v} (truth {it['true_label']})")
        verdicts[jm] = labels
        available_judges.append(jm)
        n_items = len(rating_set)
        judge_acc[jm] = {"label": MODEL_LABELS.get(jm, jm),
                         "accuracy_vs_truth": round(correct / n_items, 3),
                         "n_parsed": parsed, "n_items": n_items,
                         # parse_rate = how many verdicts the judge actually emitted
                         # (UNPARSED verdicts drag a bare Fleiss kappa down even when
                         # the parseable judges agree perfectly — m2 honesty fix).
                         "parse_rate": round(parsed / n_items, 3)}

    result = {
        "description": ("Inter-rater agreement of LLM judges on ARIA grounding. "
                        "2-3 Ollama judges rate the same set of ARIA outputs "
                        "(grounded vs number-corrupted, known truth) as "
                        "GROUNDED/NOT_GROUNDED. Cohen (pairwise) + Fleiss + "
                        "Krippendorff alpha. SEQUENTIAL Ollama, temp 0, fixed seed."),
        "judges": available_judges,
        "seed": args.seed,
        "rating_set": [{"item_id": it["item_id"], "true_label": it["true_label"]}
                       for it in rating_set],
        "verdicts": verdicts,
        "judge_accuracy_vs_truth": judge_acc,
    }

    if len(available_judges) < 2:
        result["error"] = "need >=2 available judges for inter-rater agreement"
        out = out_dir() / "aria_interrater_kappa.json"
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[error] <2 judges available; wrote {out}")
        return 1

    # ── pairwise Cohen's kappa ───────────────────────────────────────────────
    from sklearn.metrics import cohen_kappa_score
    cats = ["GROUNDED", "NOT_GROUNDED", "UNPARSED"]
    pairwise = {}
    cohen_vals = []
    for a, b in itertools.combinations(available_judges, 2):
        ka = cohen_kappa_score(verdicts[a], verdicts[b], labels=cats)
        ka = None if (ka != ka) else round(float(ka), 4)  # NaN guard
        pairwise[f"{a} | {b}"] = ka
        if ka is not None:
            cohen_vals.append(ka)
    mean_cohen = round(sum(cohen_vals) / len(cohen_vals), 4) if cohen_vals else None

    # ── Fleiss' kappa ────────────────────────────────────────────────────────
    cat_index = {c: i for i, c in enumerate(cats)}
    counts = []
    for j in range(len(rating_set)):
        row = [0] * len(cats)
        for jm in available_judges:
            row[cat_index[verdicts[jm][j]]] += 1
        counts.append(row)
    fleiss = _fleiss_kappa(counts)
    fleiss = round(fleiss, 4) if fleiss is not None else None

    # ── Krippendorff alpha ───────────────────────────────────────────────────
    rater_rows = [verdicts[jm] for jm in available_judges]
    # UNPARSED = unreliable verdict → treated as MISSING for agreement (np.nan
    # for the krippendorff pkg / None for the fallback), so a judge that failed
    # to emit a verdict does not get charged as a spurious third category.
    try:
        import krippendorff  # type: ignore
        import numpy as np
        lut = {"GROUNDED": 0, "NOT_GROUNDED": 1}
        arr = np.array([[lut.get(v, np.nan) for v in row] for row in rater_rows], float)
        kr_alpha = float(krippendorff.alpha(reliability_data=arr,
                                            level_of_measurement="nominal"))
        kr_source = "krippendorff-pkg"
    except Exception:
        rows_missing = [[(None if v == "UNPARSED" else v) for v in row]
                        for row in rater_rows]
        kr_alpha = _krippendorff_alpha_nominal(rows_missing)
        kr_source = "builtin-nominal-fallback"
    kr_alpha = round(kr_alpha, 4) if kr_alpha is not None else None

    def interp(k):
        if k is None:
            return "undefined"
        if k < 0:    return "worse-than-chance"
        if k < 0.20: return "slight"
        if k < 0.40: return "fair"
        if k < 0.60: return "moderate"
        if k < 0.80: return "substantial"
        return "almost-perfect"

    # ── per-judge parse-rate + accuracy reported ALONGSIDE Fleiss (m2 honesty) ──
    # A bare Fleiss kappa is misleading when one judge emits UNPARSED verdicts:
    # the parseable judges may agree perfectly yet the headline kappa is depressed
    # by treating UNPARSED as a third (always-disagreeing) category. Surface the
    # per-judge parse-rate + accuracy so the reader can see WHY a kappa is low.
    n_items = len(rating_set)
    n_fully_parsed = sum(1 for jm in available_judges
                         if judge_acc[jm]["n_parsed"] == n_items)
    any_unparsed = n_fully_parsed < len(available_judges)
    interrater_summary = {
        "per_judge": {jm: {"parse_rate": judge_acc[jm]["parse_rate"],
                           "accuracy_vs_truth": judge_acc[jm]["accuracy_vs_truth"]}
                      for jm in available_judges},
        "n_judges": len(available_judges),
        "n_judges_fully_parsed": n_fully_parsed,
        "any_unparsed_verdicts": any_unparsed,
        "caveat": (
            "Bare Fleiss kappa treats UNPARSED verdicts as a disagreeing third "
            "category; read it together with per-judge parse_rate/accuracy above. "
            if any_unparsed else
            "All judges emitted parseable verdicts on every item; Fleiss is "
            "uncontaminated by UNPARSED.")}

    result.update({
        "cohen_pairwise": pairwise,
        "cohen_mean": mean_cohen,
        "fleiss_kappa": fleiss,
        "krippendorff_alpha": kr_alpha,
        "krippendorff_source": kr_source,
        "headline_kappa": fleiss if fleiss is not None else mean_cohen,
        "interrater_summary": interrater_summary,
        "interpretation": {
            "cohen_mean": interp(mean_cohen),
            "fleiss": interp(fleiss),
            "krippendorff": interp(kr_alpha),
        },
        "elapsed_s": round(time.time() - t0, 1),
    })
    out = out_dir() / "aria_interrater_kappa.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nCohen mean={mean_cohen} ({interp(mean_cohen)})  "
          f"Fleiss={fleiss} ({interp(fleiss)})  "
          f"Krippendorff={kr_alpha} ({interp(kr_alpha)})")
    print("per-judge parse_rate / accuracy_vs_truth:")
    for jm in available_judges:
        ja = judge_acc[jm]
        print(f"  {jm:14s} parse_rate={ja['parse_rate']}  acc={ja['accuracy_vs_truth']}")
    if interrater_summary["any_unparsed_verdicts"]:
        print("  [caveat] " + interrater_summary["caveat"])
    print(f"wrote {out}  (elapsed {result['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
