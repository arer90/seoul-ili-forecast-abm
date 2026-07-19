"""
simulation.llm_compare.aria_grounding
======================================
ARIA grounding faithfulness on REAL thesis outputs + Self-Ask decomposition.

The standalone QA benchmark (`kr_epi_bench`) measures whether a backend KNOWS
Korean epidemiology/law. But ARIA's ACTUAL task is to *interpret the project's
own forecast / simulation outputs for an epidemiologist*. This module connects
`comparison.faithfulness` + `numeric_grounding` to the REAL results (ABM
forward-validation calibration + ABM real-wave fit metrics) — does ARIA's
interpretation ground in the actual numbers, or hallucinate different ones?

This is the §Bedi-3 (factuality/grounding) evaluation on ARIA's real task,
distinct from (and stronger than) the standalone QA bench, because the context
is the project's own validated output, not a self-authored wiki.

Two grounding axes are produced (both leak-free, file-based, read-only):

  • numeric_grounding  — does the answer cite the REAL numbers (fact_recall) and
    avoid inventing wrong ones (n_spurious)?  The precise check token-overlap
    faithfulness misses.
  • Self-Ask (SubQ)    — ``self_ask_decompose`` splits the numeric context into
    atomic sub-questions, answers each FROM THE NUMBERS (corpus-free, so it works
    even though the QA corpus is only ~20 docs; Press et al. 2022, arXiv:2210.03350),
    and recomposes a final grounded answer.  ``self_ask_grounding`` scores a
    backend's recomposition against the sub-answer gold.

CLI:  python -m simulation.llm_compare.aria_grounding --mock
      python -m simulation.llm_compare.aria_grounding --self-ask --mock
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from .backends import discover_backends
from .comparison import faithfulness

__all__ = [
    "load_real_context", "numeric_grounding", "semantic_consistency", "grounding_eval",
    "GROUNDING_PROMPT",
    "self_ask_decompose", "self_ask_answer", "self_ask_grounding", "SELF_ASK_PROMPT",
]

_NUM = re.compile(r"-?\d+\.?\d*")

# Comparator phrases → +1 means "left number > right number" is the CLAIMED relation,
# -1 means "left number < right number" is the CLAIMED relation. Longer/compound
# phrases are listed first so 'lower than' wins over a bare 'than' scan.
_COMPARATORS = (
    # English
    ("greater than", +1), ("higher than", +1), ("larger than", +1),
    ("more than", +1), ("bigger than", +1),
    ("less than", -1), ("lower than", -1), ("smaller than", -1),
    ("fewer than", -1),
    # Korean ("A 는/가 B 보다 낮/높" → the comparator follows the SECOND number)
    ("보다 낮", -1), ("보다 작", -1), ("보다 적", -1),
    ("보다 높", +1), ("보다 크", +1), ("보다 많", +1),
)
# Number immediately followed by a comparator, then another number (English word order).
_CMP_EN = re.compile(
    r"(-?\d+\.?\d*)\D{0,40}?\b(greater than|higher than|larger than|more than|bigger than|"
    r"less than|lower than|smaller than|fewer than)\b\D{0,40}?(-?\d+\.?\d*)",
    re.IGNORECASE)
# Korean order: "<num1> ... 보다 ... <num2> ... 낮/높" → num1 vs num2, comparator AFTER num2.
_CMP_KO = re.compile(
    r"(-?\d+\.?\d*)[^\d]{0,40}?보다[^\d]{0,40}?(-?\d+\.?\d*)[^\d]{0,12}?"
    r"(낮|작|적|높|크|많)")


def semantic_consistency(answer: str, facts) -> dict:
    """Semantic-relation grounding — catches *confidently-wrong comparisons* that
    :func:`numeric_grounding` (presence-only) passes.

    ``numeric_grounding`` rewards an answer that merely *cites* the gold numbers;
    a reversed claim like "ON R²=0.557 is lower than OFF R²=0.0408" still scores
    perfect recall because both numbers are present. For a public-health advisory
    that is the most dangerous failure (a directionally-wrong, confident claim).

    This lightweight pass (regex + arithmetic, no external NLP) finds comparison
    clauses joining two numbers via a comparator ("lower/higher than", "보다 낮/높",
    …), and — when *both* numbers are gold facts — checks the CLAIMED direction
    against the facts' real ordering. A mismatch is a contradiction.

    Args:
        answer: the backend's free-text interpretation.
        facts: iterable of ``"key=value"`` strings (the gold numeric facts). Only
            comparisons whose *both* operands are gold values are judged (so an
            unrelated comparison the model invents is not penalised here — that is
            ``numeric_grounding``'s n_spurious job).

    Returns:
        ``{n_comparisons, n_consistent, n_contradictory, contradictions}`` where
        ``contradictions`` is a list of ``{left, right, claim, truth}`` dicts.
        ``n_comparisons`` counts only gold-vs-gold comparisons that were checked.

    Side effects: none. Never raises.
    """
    gold = set()
    for f in facts:
        m = _NUM.search(str(f).split("=")[-1])
        if m:
            gold.add(m.group())
    text = answer or ""
    n_cmp = n_ok = 0
    contradictions: list[dict] = []
    seen: set[tuple] = set()

    def _judge(left_s: str, right_s: str, claim_dir: int) -> None:
        nonlocal n_cmp, n_ok
        if left_s not in gold or right_s not in gold:
            return  # only judge comparisons between two GOLD numbers
        key = (left_s, right_s, claim_dir)
        if key in seen:
            return
        seen.add(key)
        try:
            lv, rv = float(left_s), float(right_s)
        except ValueError:
            return
        if lv == rv:
            return  # equal numbers carry no direction to contradict
        n_cmp += 1
        true_dir = +1 if lv > rv else -1   # actual ordering of left vs right
        if claim_dir == true_dir:
            n_ok += 1
        else:
            contradictions.append({
                "left": left_s, "right": right_s,
                "claim": "left>right" if claim_dir > 0 else "left<right",
                "truth": "left>right" if true_dir > 0 else "left<right",
            })

    for m in _CMP_EN.finditer(text):
        ldir = +1 if any(w in m.group(2).lower() for w in
                         ("greater", "higher", "larger", "more", "bigger")) else -1
        _judge(m.group(1), m.group(3), ldir)
    for m in _CMP_KO.finditer(text):
        # Korean "A 보다 B (가) 낮/높" = "B is lower/higher than A". A=num1 (the
        # 보다 reference, left), B=num2 (the subject, right). The comparator
        # describes B vs A: 높/크/많 ⇒ B>A (right>left ⇒ left<right ⇒ -1);
        # 낮/작/적 ⇒ B<A (right<left ⇒ left>right ⇒ +1).
        suffix = m.group(3)
        ldir = +1 if suffix in ("낮", "작", "적") else -1
        _judge(m.group(1), m.group(2), ldir)

    return {"n_comparisons": n_cmp, "n_consistent": n_ok,
            "n_contradictory": len(contradictions), "contradictions": contradictions}


def numeric_grounding(answer: str, facts) -> dict:
    """Numerical grounding — the precise check ``faithfulness`` (token-overlap)
    misses: does the answer cite the REAL values, and avoid contradicting them?

    Token-overlap faithfulness rewards topical words even when the *numbers* are
    wrong (R²=0.95 vs truth 0.22 still shares 'R²'). This extracts the gold
    values from ``facts`` ('key=value') and checks how many appear in the answer
    (fact recall) plus how many answer-numbers are NOT in the gold (spurious).

    Args:
        answer: the backend's free-text interpretation.
        facts: iterable of ``"key=value"`` strings (the gold numeric facts).

    Returns:
        ``{fact_recall, n_gold_cited, n_gold, n_spurious}``.

    Side effects: none. Never raises.
    """
    gold = set()
    for f in facts:
        m = _NUM.search(str(f).split("=")[-1])
        if m:
            gold.add(m.group())
    ans = set(_NUM.findall(answer or ""))
    cited = gold & ans
    spurious = [v for v in ans if v not in gold and len(v.lstrip("-")) > 1]  # ignore 1-digit
    return {"fact_recall": round(len(cited) / len(gold), 3) if gold else 0.0,
            "n_gold_cited": len(cited), "n_gold": len(gold), "n_spurious": len(spurious)}

GROUNDING_PROMPT = (
    "당신은 역학자에게 시뮬레이션 결과를 해석해 주는 자문가입니다. 아래 *제공된 수치만* "
    "근거로 2~3문장으로 해석하세요. 제공되지 않은 수치를 지어내지 마세요.\n\n"
)

SELF_ASK_PROMPT = (
    "당신은 역학 자문가입니다. 아래 *제공된 수치만* 근거로 답하세요. "
    "먼저 핵심 하위질문(Self-Ask)에 하나씩 답한 뒤, 마지막에 종합 해석을 쓰세요. "
    "제공되지 않은 수치는 지어내지 마세요.\n\n"
)


def _fmt(x) -> str:
    return f"{x:.3g}" if isinstance(x, (int, float)) else str(x)


def _resolve(base: Path, *candidates: str) -> Path | None:
    """First existing path among ``base/<candidate>``; ``None`` if none exist.

    Args:
        base: results root (e.g. ``simulation/results``).
        candidates: relative paths tried IN ORDER (active first, archive last).

    Returns:
        The first ``Path`` that exists, else ``None``. Never raises.
    """
    for c in candidates:
        p = base / c
        if p.exists():
            return p
    return None


def load_real_context(which: str = "identifiability", *, root: str | None = None) -> dict:
    """Build a grounding context + the gold 'facts' from a REAL thesis output.

    Reads the project's own *active* ABM results (read-only, no DB). Resolves the
    active file first and falls back to the legacy ``_trash`` snapshot only if the
    active artifact is missing, so the score always reflects on-disk truth.

    Args:
        which: ``'identifiability'`` (ABM forward-validation calibration headline:
            calibrated behaviour params + forward R² + anchor correlation) or
            ``'abm'`` (ABM real-wave fit: adaptive/static R², RMSE, DM p-value).
        root: results root (default ``simulation/results``).

    Returns:
        ``{id, context, facts, source}`` — ``context`` is the prompt-ready text,
        ``facts`` the list of ground-truth ``"key=value"`` fact tokens.

    Raises:
        FileNotFoundError: if NEITHER the active nor the legacy artifact exists
            (fail-loud: a grounding context with no real source is meaningless).

    Side effects: reads JSON from disk only. No DB, no writes.
    """
    base = Path(root or "simulation/results")
    if which == "identifiability":
        p = _resolve(base,
                     "abm_forward_validation/result.json",
                     "identifiability_4d_calibrated.json",
                     "../../_trash/results_retrain_partial_20260608/identifiability_4d_calibrated.json")
        if p is None:
            raise FileNotFoundError(
                "no ABM forward-validation / identifiability artifact under "
                f"{base} (looked for abm_forward_validation/result.json)")
        d = json.loads(p.read_text(encoding="utf-8"))
        if "calibrated_behaviour" in d:  # active: abm_forward_validation/result.json
            cb = d.get("calibrated_behaviour", {})
            facts = [f"alpha={_fmt(cb.get('alpha'))}", f"kappa={_fmt(cb.get('kappa'))}",
                     f"tau={_fmt(cb.get('tau'))}", f"theta={_fmt(cb.get('theta'))}",
                     f"forward_r2={_fmt(d.get('forward_r2'))}",
                     f"anchor_corr={_fmt(d.get('anchor_corr_sim_vs_forecast'))}",
                     f"r2_behavior_on={_fmt(d.get('forward_r2_behavior_on'))}",
                     f"r2_behavior_off={_fmt(d.get('forward_r2_behavior_off'))}"]
            context = (
                "ABM 전향(forward) 검증(2026 실데이터). 챔피언 예측 anchor에 결합한 "
                "행동-ON ABM의 보정 행동 파라미터: "
                f"alpha={_fmt(cb.get('alpha'))}, kappa={_fmt(cb.get('kappa'))}, "
                f"tau={_fmt(cb.get('tau'))}, theta={_fmt(cb.get('theta'))}. "
                f"전향 R²={_fmt(d.get('forward_r2'))}, "
                f"anchor 상관={_fmt(d.get('anchor_corr_sim_vs_forecast'))}. "
                f"행동 ON 전향 R²={_fmt(d.get('forward_r2_behavior_on'))} vs "
                f"OFF={_fmt(d.get('forward_r2_behavior_off'))} (행동 ON 우세).")
            return {"id": "P4_identifiability", "context": context, "facts": facts,
                    "source": "abm_forward_validation/result.json"}
        # legacy fallback: identifiability_4d_calibrated.json
        ct = d.get("calibrated_truth", {})
        prof = d.get("profiles", {})
        moved = [k for k, v in prof.items() if v.get("identified_by_mobility")]
        facts = [f"alpha={_fmt(ct.get('alpha'))}", f"kappa={_fmt(ct.get('kappa'))}",
                 f"tau={_fmt(ct.get('tau'))}", f"theta={_fmt(ct.get('theta'))}",
                 f"shape_r2={_fmt(d.get('shape_r2'))}",
                 f"mobility_identifies={','.join(moved) or 'none'}"]
        context = (f"계절 {d.get('season')} 식별성 분석(P4). 보정된 참값: "
                   f"alpha={_fmt(ct.get('alpha'))}, kappa={_fmt(ct.get('kappa'))}, "
                   f"tau={_fmt(ct.get('tau'))}, theta={_fmt(ct.get('theta'))}. "
                   f"형태 적합 R²={_fmt(d.get('shape_r2'))}. "
                   f"mobility로 식별되는 파라미터: {', '.join(moved) or '없음'}.")
        return {"id": "P4_identifiability", "context": context, "facts": facts,
                "source": "identifiability_4d_calibrated.json"}
    # abm
    p = _resolve(base,
                 "abm_real_validation/result.json",
                 "agent_world_134metrics.json",
                 "../../_trash/results_retrain_partial_20260608/agent_world_134metrics.json")
    if p is None:
        raise FileNotFoundError(
            f"no ABM real-validation / agent-world artifact under {base} "
            "(looked for abm_real_validation/result.json)")
    d = json.loads(p.read_text(encoding="utf-8"))
    if "r2_adaptive" in d:  # active: abm_real_validation/result.json
        facts = [f"r2_adaptive={_fmt(d.get('r2_adaptive'))}",
                 f"r2_static={_fmt(d.get('r2_static'))}",
                 f"rmse={_fmt(d.get('rmse'))}",
                 f"dm_p={_fmt(d.get('dm_p_value'))}",
                 f"n_seasons={_fmt(d.get('n_seasons'))}"]
        context = (
            f"ABM 실파동 적합(계절 {d.get('season')}, {d.get('n_seasons')}개 시즌 검증). "
            f"개별행동 SEIR-V-D의 실 ILI 적합: 적응 R²={_fmt(d.get('r2_adaptive'))}, "
            f"정적(행동OFF) R²={_fmt(d.get('r2_static'))}, RMSE={_fmt(d.get('rmse'))}. "
            f"적응 vs 정적 DM p={_fmt(d.get('dm_p_value'))}.")
        return {"id": "ABM_fit", "context": context, "facts": facts,
                "source": "abm_real_validation/result.json"}
    # legacy fallback: agent_world_134metrics.json
    facts = [f"{k}={_fmt(d.get(k))}" for k in ("r2", "mae", "rmse", "mape") if k in d]
    context = ("에이전트 기반 모형(ABM) 적합 메트릭: " + ", ".join(facts) + ".")
    return {"id": "ABM_fit", "context": context, "facts": facts,
            "source": "agent_world_134metrics.json"}


def grounding_eval(backends, contexts, *, max_tokens: int = 200) -> dict:
    """Run each backend on each real context and measure grounding faithfulness.

    Args:
        backends: iterable of ``LLMBackend`` (``.generate``, ``.backend_id``).
        contexts: list of ``load_real_context`` dicts.
        max_tokens: generation cap per call.

    Returns:
        ``{per_backend: {bid: {fact_recall, faithfulness, n_spurious_total,
        n_contexts, n_errors}}, contexts}``. fact_recall = primary grounding
        signal (cites the REAL numbers); faithfulness = secondary (topical).

    Side effects: calls ``backend.generate`` (network for live backends; none for
    Mock). Never raises on a backend error.
    """
    out: dict = {}
    for b in backends:
        faith, recall, spur, errs = [], [], 0, 0
        for ctx in contexts:
            resp = b.generate(GROUNDING_PROMPT + ctx["context"], max_tokens=max_tokens)
            if resp.error:
                errs += 1
                continue
            faith.append(faithfulness(resp.text, [ctx["context"]])["faithfulness"])
            ng = numeric_grounding(resp.text, ctx["facts"])
            recall.append(ng["fact_recall"])
            spur += ng["n_spurious"]
        out[b.backend_id] = {
            "fact_recall": round(sum(recall) / len(recall), 4) if recall else None,
            "faithfulness": round(sum(faith) / len(faith), 4) if faith else None,
            "n_spurious_total": spur,
            "n_contexts": len(recall), "n_errors": errs,
        }
    return {"per_backend": out,
            "contexts": [{"id": c["id"], "source": c["source"]} for c in contexts]}


# ── Self-Ask (SubQ) — corpus-free decomposition over NUMERIC facts ────────────
# Press et al. 2022 (arXiv:2210.03350): elicit atomic sub-questions, answer each,
# then recompose. The standalone QA corpus is only ~20 docs (too small for
# retrieval-based decomposition), so we decompose over the *numbers* in
# load_real_context — fully corpus-independent and deterministic.

_SUBQ_TEMPLATES = {
    "alpha": "행동 반응 강도(alpha)는 얼마인가?",
    "kappa": "기억 감쇠(kappa)는 얼마인가?",
    "tau": "행동 지연(tau, 일)은 얼마인가?",
    "theta": "발현 임계(theta)는 얼마인가?",
    "forward_r2": "전향(forward) R²는 얼마인가?",
    "anchor_corr": "anchor 상관계수는 얼마인가?",
    "r2_behavior_on": "행동 ON 전향 R²는 얼마인가?",
    "r2_behavior_off": "행동 OFF 전향 R²는 얼마인가?",
    "shape_r2": "형태 적합 R²는 얼마인가?",
    "mobility_identifies": "mobility로 식별되는 파라미터는 무엇인가?",
    "r2_adaptive": "적응(행동 ON) R²는 얼마인가?",
    "r2_static": "정적(행동 OFF) R²는 얼마인가?",
    "rmse": "RMSE는 얼마인가?",
    "dm_p": "Diebold-Mariano p-value는 얼마인가?",
    "n_seasons": "검증 시즌 수는 몇 개인가?",
}


def self_ask_decompose(context: dict) -> list[dict]:
    """Decompose a numeric grounding context into atomic Self-Ask sub-questions.

    Each gold ``"key=value"`` fact becomes one sub-question whose gold answer IS
    the value — corpus-free (Press et al. 2022). A generic question is used for
    keys without a template, so the decomposition never drops a fact.

    Args:
        context: a ``load_real_context`` dict (must carry ``facts``).

    Returns:
        ``[{sub_q, key, gold_value}, ...]`` in fact order. Empty list if no facts.

    Side effects: none. Never raises.
    """
    subqs = []
    for f in context.get("facts", []):
        key, _, val = str(f).partition("=")
        key = key.strip()
        q = _SUBQ_TEMPLATES.get(key, f"{key} 값은 얼마인가?")
        subqs.append({"sub_q": q, "key": key, "gold_value": val.strip()})
    return subqs


def self_ask_answer(context: dict) -> dict:
    """Self-Ask reference answer: decompose → answer each sub-Q from the numbers
    → recompose a grounded final answer.

    This is the corpus-free reference trajectory (what a perfectly-grounded ARIA
    would produce). A backend's own recomposition is scored against it by
    ``self_ask_grounding``.

    Args:
        context: a ``load_real_context`` dict.

    Returns:
        ``{sub_questions:[{sub_q, key, gold_value, sub_answer}], final_answer,
        n_subq}``. ``final_answer`` is the recomposed grounded text.

    Side effects: none. Never raises.
    """
    subs = self_ask_decompose(context)
    for s in subs:
        s["sub_answer"] = f"{s['key']} = {s['gold_value']}"
    parts = ", ".join(s["sub_answer"] for s in subs)
    final = (f"제공된 수치 종합({context.get('id')}): {parts}. "
             "모든 해석은 위 실제 산출 수치에만 근거합니다.")
    return {"sub_questions": subs, "final_answer": final, "n_subq": len(subs)}


def self_ask_grounding(backends, contexts, *, max_tokens: int = 320) -> dict:
    """Run each backend through the Self-Ask trajectory and score grounding.

    The backend is prompted with the Self-Ask instruction + the listed
    sub-questions; its recomposed answer is scored by ``numeric_grounding`` (does
    it carry the sub-answer numbers?) and ``faithfulness`` (topical support).

    Args:
        backends: iterable of ``LLMBackend``.
        contexts: list of ``load_real_context`` dicts.
        max_tokens: generation cap.

    Returns:
        ``{per_backend: {bid: {subq_fact_recall, faithfulness, n_spurious_total,
        mean_n_subq, n_contexts, n_errors}}, reference}`` where ``reference`` is
        the corpus-free Self-Ask gold trajectory per context.

    Side effects: calls ``backend.generate``. Never raises on a backend error.
    """
    references = []
    for ctx in contexts:
        ref = self_ask_answer(ctx)
        references.append({"id": ctx["id"], "source": ctx["source"],
                           "sub_questions": [{"sub_q": s["sub_q"], "gold": s["sub_answer"]}
                                             for s in ref["sub_questions"]],
                           "final_answer": ref["final_answer"]})
    out: dict = {}
    for b in backends:
        recall, faith, spur, nsub, errs = [], [], 0, [], 0
        for ctx in contexts:
            ref = self_ask_answer(ctx)
            subq_block = "\n".join(f"- {s['sub_q']}" for s in ref["sub_questions"])
            prompt = (SELF_ASK_PROMPT + ctx["context"] +
                      "\n\n하위질문(각각 한 줄로 답하세요):\n" + subq_block)
            resp = b.generate(prompt, max_tokens=max_tokens)
            if resp.error:
                errs += 1
                continue
            ng = numeric_grounding(resp.text, ctx["facts"])
            recall.append(ng["fact_recall"])
            spur += ng["n_spurious"]
            faith.append(faithfulness(resp.text, [ctx["context"]])["faithfulness"])
            nsub.append(ref["n_subq"])
        out[b.backend_id] = {
            "subq_fact_recall": round(sum(recall) / len(recall), 4) if recall else None,
            "faithfulness": round(sum(faith) / len(faith), 4) if faith else None,
            "n_spurious_total": spur,
            "mean_n_subq": round(sum(nsub) / len(nsub), 2) if nsub else 0.0,
            "n_contexts": len(recall), "n_errors": errs,
        }
    return {"per_backend": out, "reference": references}


# ── producers: scores JSON + per-backend markdown / CSV ───────────────────────
def _write_markdown(rep: dict, sa: dict | None, path: Path, *, backend_note: str) -> None:
    lines = ["# ARIA grounding scores (real thesis outputs)", "",
             f"Backends: {backend_note}", "",
             "## (a) Numeric grounding — direct interpretation", "",
             "| backend | fact_recall | faithfulness | n_spurious | n_ctx | n_err |",
             "|---|---|---|---|---|---|"]
    for bid, s in rep["per_backend"].items():
        lines.append(f"| {bid} | {s['fact_recall']} | {s['faithfulness']} | "
                     f"{s['n_spurious_total']} | {s['n_contexts']} | {s['n_errors']} |")
    lines += ["", f"Contexts: {', '.join(c['source'] for c in rep['contexts'])}", ""]
    if sa is not None:
        lines += ["## (b) Self-Ask (SubQ) — decomposed grounding", "",
                  "| backend | subq_fact_recall | faithfulness | n_spurious | mean_n_subq | n_err |",
                  "|---|---|---|---|---|---|"]
        for bid, s in sa["per_backend"].items():
            lines.append(f"| {bid} | {s['subq_fact_recall']} | {s['faithfulness']} | "
                         f"{s['n_spurious_total']} | {s['mean_n_subq']} | {s['n_errors']} |")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(rep: dict, sa: dict | None, path: Path) -> None:
    import csv
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["axis", "backend", "fact_recall", "faithfulness",
                    "n_spurious", "extra", "n_contexts", "n_errors"])
        for bid, s in rep["per_backend"].items():
            w.writerow(["numeric", bid, s["fact_recall"], s["faithfulness"],
                        s["n_spurious_total"], "", s["n_contexts"], s["n_errors"]])
        if sa is not None:
            for bid, s in sa["per_backend"].items():
                w.writerow(["self_ask", bid, s["subq_fact_recall"], s["faithfulness"],
                            s["n_spurious_total"], f"mean_n_subq={s['mean_n_subq']}",
                            s["n_contexts"], s["n_errors"]])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="ARIA grounding faithfulness on real outputs")
    ap.add_argument("--no-api", action="store_true")
    ap.add_argument("--no-cli", action="store_true")
    ap.add_argument("--no-ollama", action="store_true")
    ap.add_argument("--mock", action="store_true",
                    help="force the deterministic Mock control group (offline)")
    ap.add_argument("--self-ask", action="store_true",
                    help="also run the Self-Ask (SubQ) decomposition axis")
    ap.add_argument("--out", default="simulation/results/aria_grounding.json")
    args = ap.parse_args(argv)
    backends = discover_backends(include_api=not args.no_api, include_cli=not args.no_cli,
                                 include_ollama=not args.no_ollama, include_mock=args.mock)
    is_mock_only = all(getattr(b, "tier", "") == "mock" for b in backends)
    backend_note = ("MOCK only (no live LLM backend — deterministic control group)"
                    if is_mock_only else
                    ", ".join(sorted({getattr(b, 'tier', '?') for b in backends})))
    contexts = [load_real_context("identifiability"), load_real_context("abm")]
    rep = grounding_eval(backends, contexts)
    sa = self_ask_grounding(backends, contexts) if args.self_ask else None

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"mock_only": is_mock_only, "backend_note": backend_note,
               "numeric_grounding": rep}
    if sa is not None:
        payload["self_ask"] = sa
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(rep, sa, out.with_suffix(".md"), backend_note=backend_note)
    _write_csv(rep, sa, out.with_suffix(".csv"))

    print(f"ARIA grounding (real thesis outputs) — backends: {backend_note}")
    print("(a) numeric grounding:")
    for bid, s in rep["per_backend"].items():
        print(f"  {bid.split('@')[0]:30s} fact_recall={s['fact_recall']} "
              f"faithfulness={s['faithfulness']} (spurious={s['n_spurious_total']}, err={s['n_errors']})")
    if sa is not None:
        print("(b) Self-Ask (SubQ):")
        for bid, s in sa["per_backend"].items():
            print(f"  {bid.split('@')[0]:30s} subq_fact_recall={s['subq_fact_recall']} "
                  f"faithfulness={s['faithfulness']} (mean_n_subq={s['mean_n_subq']})")
    print(f"wrote {out}  +  {out.with_suffix('.md').name}  +  {out.with_suffix('.csv').name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
