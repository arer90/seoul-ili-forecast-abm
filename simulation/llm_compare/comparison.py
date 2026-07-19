"""P5 — statistical multi-LLM comparison (PROOF_VALIDATION_PROTOCOL §2).

The existing benchmark *scores* each backend (7 pillars, ``judge.score_response``).
That is necessary but not SCI-sufficient: a claim that LLM A beats LLM B needs a
significance test on the PAIRED per-item scores, not a raw mean gap, plus
agreement metrics. This module consumes the ``ScoredResponse`` list the runner
already produces and adds:

  • pairwise Wilcoxon signed-rank on paired per-item totals + Holm–Bonferroni
    correction across the K·(K-1)/2 comparisons (family-wise error control);
  • bootstrap CIs on each backend's mean composite (effect size, not just p);
  • Fleiss' κ — inter-LLM agreement on per-item pass/fail (benchmark consistency
    vs discrimination);
  • Cohen's κ — judge reliability: the rule-based judge vs a reference rater
    (human or an LLM-judge) on the same pass/fail calls.

Pure statistics over the score matrix → fully testable without live LLMs. Never
raises (degenerate inputs → explicit ``{"error": …}`` or neutral values).
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np

try:  # scipy is a hard dep elsewhere; degrade loudly-but-safely if absent
    from scipy.stats import wilcoxon as _wilcoxon
except Exception:  # pragma: no cover
    _wilcoxon = None


def _pivot(scored: Iterable, value: str = "total") -> tuple[dict, list, list]:
    """``ScoredResponse`` list → ``{item_id: {backend_id: value}}`` plus the
    sorted backend ids and the items answered by EVERY backend (paired set)."""
    table: dict[str, dict[str, float]] = {}
    backends: set[str] = set()
    for s in scored:
        bid = getattr(s, "backend_id", None) if not isinstance(s, dict) else s.get("backend_id")
        iid = getattr(s, "item_id", None) if not isinstance(s, dict) else s.get("item_id")
        val = getattr(s, value, None) if not isinstance(s, dict) else s.get(value)
        if bid is None or iid is None or val is None:
            continue
        table.setdefault(iid, {})[bid] = float(val)
        backends.add(bid)
    blist = sorted(backends)
    paired = sorted(i for i, row in table.items() if all(b in row for b in blist))
    return table, blist, paired


def holm_correction(pvals: list[float]) -> list[float]:
    """Holm–Bonferroni step-down adjusted p-values (preserves order, monotone,
    controls FWER without the conservatism of plain Bonferroni)."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        v = (m - rank) * pvals[idx]
        running = max(running, v)  # enforce monotonicity
        adj[idx] = min(1.0, running)
    return adj


def pairwise_wilcoxon_holm(scored, value: str = "total", alpha: float = 0.05) -> dict:
    """Pairwise Wilcoxon signed-rank on paired per-item totals + Holm correction.

    Returns ``{n_items, backends, comparisons:[{pair, median_diff, p_value,
    p_holm, significant}], ...}``. A pair with all-equal scores → p=1.0 (not a
    crash). Never raises. Caller responsibility: ≥1 shared item; with <6 paired
    items Wilcoxon power is low (reported, not hidden)."""
    table, backends, paired = _pivot(scored, value)
    if len(backends) < 2:
        return {"error": "need ≥2 backends", "backends": backends}
    if len(paired) < 1:
        return {"error": "no item answered by all backends (no paired set)"}
    raw, pairs, meds = [], [], []
    for a, b in combinations(backends, 2):
        da = np.array([table[i][a] for i in paired])
        db = np.array([table[i][b] for i in paired])
        diff = da - db
        if _wilcoxon is None or np.allclose(diff, 0.0):
            p = 1.0
        else:
            try:
                p = float(_wilcoxon(da, db, zero_method="pratt").pvalue)
            except ValueError:
                p = 1.0  # all-zero differences
            if not np.isfinite(p):
                p = 1.0
        raw.append(p); pairs.append((a, b)); meds.append(float(np.median(diff)))
    adj = holm_correction(raw)
    low_power = len(paired) < 6
    # M1 (Gemini): with <6 paired items two-sided Wilcoxon CANNOT reach p<0.05, so
    # "not significant" is structurally guaranteed — a THIRD state (inconclusive),
    # never read as "equivalent" (absence-of-evidence ≠ evidence-of-equivalence;
    # assert equivalence only with a pre-specified TOST margin).
    def _verdict(sig: bool) -> str:
        if sig:
            return "different (Holm-significant)"
        return "inconclusive — underpowered (n<6); NOT equivalence" if low_power \
            else "no difference detected"
    comps = [{"pair": f"{a} vs {b}", "median_diff": round(meds[k], 4),
              "p_value": round(raw[k], 5), "p_holm": round(adj[k], 5),
              "significant": bool(adj[k] < alpha),
              "verdict": _verdict(bool(adj[k] < alpha))}
             for k, (a, b) in enumerate(pairs)]
    return {"n_items": len(paired), "backends": backends, "alpha": alpha,
            "low_power": low_power, "comparisons": comps,
            "power_note": ("n<6: non-significance is inconclusive, not equivalence "
                           "(run TOST with a margin to assert equivalence)"
                           if low_power else "adequately powered")}


def bootstrap_ci(values, n_boot: int = 5000, seed: int = 42, ci: float = 0.95) -> dict:
    """Percentile bootstrap CI for the mean. Returns ``{mean, lo, hi, n}``."""
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=np.float64)
    if v.size == 0:
        return {"error": "no finite values"}
    rng = np.random.default_rng(seed)
    boot = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [100 * (1 - ci) / 2, 100 * (1 + ci) / 2])
    return {"mean": round(float(v.mean()), 4), "lo": round(float(lo), 4),
            "hi": round(float(hi), 4), "n": int(v.size)}


def fleiss_kappa_binary(scored, value: str = "total", threshold: float = 0.7) -> dict:
    """Fleiss' κ for inter-LLM agreement on per-item pass/fail (pass iff
    ``value ≥ threshold``). Backends are the raters, items the subjects, {pass,
    fail} the categories.

    High κ ⇒ the LLMs agree on which items pass (consistent item difficulty);
    low/zero κ ⇒ they disagree (the benchmark discriminates between models).
    Returns ``{kappa, n_items, n_raters, interpretation}``. Never raises."""
    table, backends, paired = _pivot(scored, value)
    n = len(backends)
    if n < 2 or len(paired) < 2:
        return {"error": f"need ≥2 backends and ≥2 paired items (got {n}, {len(paired)})"}
    # n_ij = (#pass, #fail) per item
    counts = np.zeros((len(paired), 2), dtype=np.float64)
    for r, i in enumerate(paired):
        n_pass = sum(1 for b in backends if table[i][b] >= threshold)
        counts[r] = [n_pass, n - n_pass]
    P_i = (np.sum(counts ** 2, axis=1) - n) / (n * (n - 1))
    P_bar = float(P_i.mean())
    p_j = counts.sum(axis=0) / (len(paired) * n)
    P_e = float(np.sum(p_j ** 2))
    kappa = (P_bar - P_e) / (1 - P_e) if (1 - P_e) > 1e-12 else 0.0
    interp = ("almost perfect" if kappa > 0.8 else "substantial" if kappa > 0.6
              else "moderate" if kappa > 0.4 else "fair" if kappa > 0.2
              else "slight/none (models discriminate)")
    return {"kappa": round(float(kappa), 4), "n_items": len(paired),
            "n_raters": n, "threshold": threshold, "interpretation": interp}


def cohen_kappa(rater_a, rater_b) -> dict:
    """Cohen's κ between two binary raters (judge reliability: rule-judge vs a
    human/LLM-judge reference on the same pass/fail calls). Returns
    ``{kappa, agreement, n, interpretation}``. Never raises."""
    a = np.asarray(list(rater_a)); b = np.asarray(list(rater_b))
    if a.shape != b.shape or a.size == 0:
        return {"error": "raters must be equal-length non-empty"}
    a = a.astype(bool); b = b.astype(bool)
    po = float((a == b).mean())
    pa1, pb1 = a.mean(), b.mean()
    pe = pa1 * pb1 + (1 - pa1) * (1 - pb1)
    kappa = (po - pe) / (1 - pe) if (1 - pe) > 1e-12 else 1.0
    interp = ("almost perfect" if kappa > 0.8 else "substantial" if kappa > 0.6
              else "moderate" if kappa > 0.4 else "fair" if kappa > 0.2 else "poor")
    return {"kappa": round(float(kappa), 4), "agreement": round(po, 4),
            "n": int(a.size), "interpretation": interp}


def fleiss_kappa_ratings(ratings_by_rater) -> dict:
    """Fleiss' κ for ≥2 raters giving binary pass/fail verdicts on the same items
    — the **human expert-panel** case (raters=experts, subjects=responses).

    Distinct from ``fleiss_kappa_binary`` (which pivots ``ScoredResponse``
    objects over LLM backends): this consumes raw per-rater verdict lists, so it
    serves the blinded human panel where each expert hand-labels every response.

    Args:
        ratings_by_rater: ``{rater_id: [verdict, ...]}``; each verdict truthy
            (pass) or the strings pass/fail/1/0/true/false/yes/no. All lists
            must be equal length (same items, same order).
    Returns:
        ``{kappa, n_items, n_raters, p_pass, band}`` or ``{error}``. Never raises.
    """
    raters = list(ratings_by_rater)
    if len(raters) < 2:
        return {"error": f"need ≥2 raters (got {len(raters)})"}

    def _b(v):
        return 1 if (v is True or v == 1 or
                     str(v).strip().lower() in ("pass", "1", "true", "yes")) else 0
    cols = [[_b(v) for v in ratings_by_rater[r]] for r in raters]
    n_items = len(cols[0]) if cols else 0
    if n_items < 2 or any(len(c) != n_items for c in cols):
        return {"error": "raters must give equal-length (≥2) verdict lists"}
    n = len(raters)
    counts = np.zeros((n_items, 2), dtype=np.float64)
    for i in range(n_items):
        n_pass = sum(c[i] for c in cols)
        counts[i] = [n_pass, n - n_pass]
    P_i = (np.sum(counts ** 2, axis=1) - n) / (n * (n - 1))
    P_bar = float(P_i.mean())
    p_j = counts.sum(axis=0) / (n_items * n)
    P_e = float(np.sum(p_j ** 2))
    kappa = (P_bar - P_e) / (1 - P_e) if (1 - P_e) > 1e-12 else 1.0
    return {"kappa": round(float(kappa), 4), "n_items": n_items, "n_raters": n,
            "p_pass": round(float(p_j[0]), 4), "band": landis_koch_band(kappa)}


def compare_backends(scored, *, value: str = "total", pass_threshold: float = 0.7,
                     n_boot: int = 5000, seed: int = 42) -> dict:
    """Full SCI comparison report over a ``ScoredResponse`` list: per-backend
    bootstrap-CI ranking, pairwise Wilcoxon+Holm, and inter-LLM Fleiss κ.

    Returns ``{ranking, pairwise, agreement, n_backends}``. The ``ranking`` is
    sorted by mean composite with 95% CIs (non-overlapping CIs + a significant
    Holm-adjusted Wilcoxon ⇒ a defensible 'A > B' claim). Never raises."""
    table, backends, paired = _pivot(scored, value)
    if len(backends) < 2:
        return {"error": "need ≥2 backends for a comparison"}
    ranking = []
    for b in backends:
        vals = [table[i][b] for i in paired if b in table[i]]
        ci = bootstrap_ci(vals, n_boot=n_boot, seed=seed)
        ranking.append({"backend": b, **ci})
    ranking.sort(key=lambda d: d.get("mean", 0.0), reverse=True)
    return {
        "n_backends": len(backends), "n_paired_items": len(paired),
        "ranking": ranking,
        "pairwise": pairwise_wilcoxon_holm(scored, value),
        "agreement": fleiss_kappa_binary(scored, value, pass_threshold),
    }


# ── External-review hardening (2026-06-06): RAGAS groundedness + judge-bias ───
def landis_koch_band(kappa: float) -> str:
    """Landis & Koch (1977) agreement band for a κ value.

    Heuristic convention (NOT a statistically-derived cutoff): <0 poor /
    ≤0.20 slight / ≤0.40 fair / ≤0.60 moderate / ≤0.80 substantial /
    ≤1.0 almost perfect. Use to annotate Fleiss/Cohen κ outputs.
    """
    if kappa < 0:
        return "poor"
    for hi, label in [(0.20, "slight"), (0.40, "fair"), (0.60, "moderate"),
                      (0.80, "substantial"), (1.0001, "almost perfect")]:
        if kappa <= hi:
            return label
    return "almost perfect"


def faithfulness(answer: str, contexts: Iterable[str], *, overlap: float = 0.2) -> dict:
    """RAGAS-style groundedness of an answer against its retrieved context.

    Decomposes the answer into claim sentences; a claim is *supported* if it
    carries an explicit grounding marker ([law:…]/[data:…]/[tool:…]/[기존 문헌])
    OR shares ≥ `overlap` content-token Jaccard with the concatenated context.
    Dependency-free proxy for RAGAS faithfulness (Es et al. 2024) — appropriate
    for ARIA's citation-enforcement / KDCA-법령 grounding.

    Returns ``{faithfulness ∈ [0,1], n_claims, n_supported, per_claim[...]}``.
    """
    import re
    ctx_tokens = set(re.findall(r"[가-힣a-z0-9]+", " ".join(contexts).lower()))
    claims = [s.strip() for s in re.split(r"[.!?。\n]+", answer) if len(s.strip()) > 4]
    per, supp = [], 0
    for c in claims:
        cited = bool(re.search(r"\[(law|data|tool|기존)", c))
        toks = set(re.findall(r"[가-힣a-z0-9]+", c.lower()))
        jac = len(toks & ctx_tokens) / max(1, len(toks))
        ok = cited or jac >= overlap
        supp += int(ok)
        per.append({"claim": c[:70], "cited": cited, "overlap": round(jac, 2), "supported": ok})
    n = len(claims)
    return {"faithfulness": round(supp / n, 3) if n else 1.0,
            "n_claims": n, "n_supported": supp, "per_claim": per}


def judge_position_debias(judgments: Iterable[dict]) -> dict:
    """Quantify/mitigate LLM-judge position bias (Zheng et al. 2023).

    Each pair must be judged in *both* presentation orders. A preference is
    reliable only if the same model wins regardless of order; order-dependent
    flips are the position-bias rate.

    Args:
        judgments: dicts ``{"pair": id, "order": "AB"|"BA", "winner": model_id}``.
    Returns:
        ``{n_pairs, consistent_preferences, position_bias_rate}`` — a high
        position_bias_rate means the judge cannot be trusted naively.
    """
    from collections import defaultdict
    byp: dict = defaultdict(dict)
    for j in judgments:
        byp[j["pair"]][j["order"]] = j["winner"]
    n = flips = consistent = 0
    for d in byp.values():
        if "AB" in d and "BA" in d:
            n += 1
            consistent += int(d["AB"] == d["BA"])
            flips += int(d["AB"] != d["BA"])
    return {"n_pairs": n, "consistent_preferences": consistent,
            "position_bias_rate": round(flips / n, 3) if n else 0.0}


def verbosity_bias(pairs: Iterable[dict], *, flag_at: float = 0.65) -> dict:
    """Detect verbosity bias — LLM judges tend to favor longer answers (Zheng 2023).

    Args:
        pairs: dicts ``{"winner_len": int, "loser_len": int}`` (token/char counts).
    Returns:
        ``{n, longer_won_rate, flag}`` — flag=True if the longer answer wins more
        than `flag_at` of decisive pairs (control response length before judging).
    """
    n = longer = 0
    for p in pairs:
        if p["winner_len"] == p["loser_len"]:
            continue
        n += 1
        longer += int(p["winner_len"] > p["loser_len"])
    rate = round(longer / n, 3) if n else 0.0
    return {"n": n, "longer_won_rate": rate, "flag": bool(n and rate > flag_at)}


# ── P5 standards hardening (2026-06-06): RAGAS retrieval, harm, power, repro ──
def context_precision(retrieved: list, relevant: list) -> float:
    """RAGAS context precision — rank-weighted average precision of retrieval.

    `retrieved` = ordered chunk/source ids returned by the RAG retriever;
    `relevant` = the ground-truth relevant ids (e.g. the correct 감염병예방법
    article / KDCA table). AP rewards placing relevant chunks early. RAGAS
    separates this *retrieval* quality from generation faithfulness (Es 2024).
    """
    rel = set(relevant)
    hits = 0
    ap = 0.0
    for k, c in enumerate(retrieved, 1):
        if c in rel:
            hits += 1
            ap += hits / k
    denom = min(len(rel), len(retrieved)) or 1
    return round(ap / denom, 3)


def context_recall(retrieved: list, relevant: list) -> float:
    """RAGAS context recall — fraction of ground-truth-relevant ids retrieved.
    Did the retriever fetch the right law article / KDCA source at all?"""
    rel = set(relevant)
    return round(len(rel & set(retrieved)) / max(1, len(rel)), 3)


def harm_summary(errors: Iterable[dict], *, levels=("none", "minor", "major", "critical")) -> dict:
    """Clinical-harm classification of advisory errors (DECIDE-AI lens).

    A low average error rate can still hide a single fatal error (wrong disease
    grade, isolation rule, or dose). Each error carries a `severity`; **critical
    errors are zero-tolerance** (a hard gate), separate from the mean harm rate.

    Args:
        errors: dicts ``{"severity": one of levels}``.
    Returns:
        ``{counts, n_critical, critical_gate_pass, harm_rate}``.
    """
    from collections import Counter
    c = Counter(e.get("severity", "none") for e in errors)
    n = sum(c.values())
    n_crit = c.get("critical", 0)
    harmful = n - c.get("none", 0)
    return {"counts": {lv: c.get(lv, 0) for lv in levels},
            "n_critical": n_crit, "critical_gate_pass": n_crit == 0,
            "harm_rate": round(harmful / max(1, n), 3)}


def n_for_power(effect_size: float, *, power: float = 0.8, alpha: float = 0.05) -> int:
    """Paired-design sample size (n queries) for a target power (normal approx).

    `effect_size` is the standardized mean difference (Cohen's d) you want to
    detect — define it from a substantive 'meaningful difference' (an LLM-MCID),
    not post-hoc. Use to justify the number of evaluation queries a priori.
    """
    from scipy.stats import norm
    z_a, z_b = norm.ppf(1 - alpha / 2), norm.ppf(power)
    return int(np.ceil(((z_a + z_b) / max(1e-6, abs(effect_size))) ** 2))


def repetition_variance(runs: Iterable[float]) -> dict:
    """LLM non-determinism: variance of a metric over repeated runs of one prompt.

    Unlike a seeded ODE, an LLM is not reproducible by seed; a single run (n=1)
    cannot show whether '0 hallucinations' was luck. Report mean ± sd + CV over
    ≥3 repetitions at a fixed model snapshot/temperature (FAIR; TRIPOD-LLM repro).
    """
    a = np.asarray(list(runs), float)
    m = float(a.mean()) if a.size else 0.0
    sd = float(a.std(ddof=1)) if a.size > 1 else 0.0
    return {"n_runs": int(a.size), "mean": round(m, 4), "sd": round(sd, 4),
            "cv": round(sd / m, 4) if m else 0.0,
            "min": float(a.min()) if a.size else 0.0,
            "max": float(a.max()) if a.size else 0.0}


# ── P5 priorities 2/3/4/5 (2026-06-07): factuality·abstention·judge-tier·repro ─
def risk_coverage_curve(confidence: Iterable[float], correct: Iterable[bool]) -> dict:
    """Selective-prediction risk–coverage curve (abstention, priority 3).

    Sort by descending confidence; at coverage k report selective accuracy on the
    top-k most-confident answers. AURC (area under risk–coverage, lower=better)
    summarizes 'does knowing-when-to-abstain help' — the calibration property a
    public-health advisor needs (`repetition_variance` only measures run noise).

    Returns ``{coverage[], selective_accuracy[], aurc, full_accuracy}``.
    """
    conf = np.asarray(list(confidence), float)
    cor = np.asarray(list(correct), float)
    if conf.size == 0:
        return {"coverage": [], "selective_accuracy": [], "aurc": 0.0, "full_accuracy": 0.0}
    order = np.argsort(-conf)
    c = cor[order]
    k = np.arange(1, c.size + 1)
    cov = k / c.size
    sel_acc = np.cumsum(c) / k
    _trapz = getattr(np, "trapezoid", np.trapz)  # numpy≥2.0 renamed trapz→trapezoid
    aurc = float(_trapz(1 - sel_acc, cov))
    return {"coverage": cov.round(3).tolist(), "selective_accuracy": sel_acc.round(3).tolist(),
            "aurc": round(aurc, 4), "full_accuracy": round(float(c.mean()), 3)}


def citation_metrics(claims: Iterable[dict]) -> dict:
    """ALCE-style citation precision/recall (factuality, priority 2; Gao 2023).

    Args:
        claims: dicts ``{"has_citation": bool, "citation_supports": bool}`` — per
            answer claim, whether it carries a citation and whether that citation
            actually supports the claim (verified against the law/KDCA source).
    Returns:
        ``{citation_precision (supported / cited), citation_recall (cited / all),
           n_claims}``. Use against EXTERNAL sources — internal-wiki overlap is a
        ceiling artifact, not a quality signal.
    """
    claims = list(claims)
    n = len(claims)
    cited = [c for c in claims if c.get("has_citation")]
    prec = sum(bool(c.get("citation_supports")) for c in cited) / len(cited) if cited else 0.0
    rec = len(cited) / n if n else 0.0
    return {"citation_precision": round(prec, 3), "citation_recall": round(rec, 3), "n_claims": n}


def judge_tier_agreement(human: list, **judges: list) -> dict:
    """Calibrate each judge tier against humans (priority 4 — calibrate, NOT demote).

    The production judge is rule-based (deterministic, reproducible but shallow —
    token presence only). An LLM judge adds reasoning-quality assessment but has
    biases. Rather than blanket-demoting the LLM judge, report each tier's Cohen κ
    vs a human-expert subset: κ ≥ 0.60 (substantial) ⇒ trust that tier to scale;
    below ⇒ route contested items to humans. Conditional trust, not demotion.

    Args:
        human: gold human labels; judges: ``tier_name=tier_labels`` (e.g.
            rule=[...], llm=[...]).
    Returns:
        ``{tier: {kappa, band, scale_ok}}`` per judge tier.
    """
    out: dict = {}
    for tier, labels in judges.items():
        k = cohen_kappa(human, labels)["kappa"]
        out[tier] = {"kappa": round(k, 3), "band": landis_koch_band(k), "scale_ok": k >= 0.60}
    return out


def repro_manifest(*, model: str, temperature: float, top_p: float, prompts_sha256: str,
                   golden_n: int, golden_freeze_date: str, law_version: str,
                   seed: int, n_repetitions: int, **extra) -> dict:
    """Reproducibility manifest (priority 5; FAIR + TRIPOD-LLM repro).

    The exact items a reviewer needs to replay an LLM evaluation: model snapshot,
    decoding params, prompt-template hash, frozen gold-set size + date, the
    감염병예방법 / KDCA corpus version, RNG seed, and repetition count. Returns the
    manifest plus a `config_sha256` over it (matches the project's run-manifest
    discipline). LLMs are not seed-reproducible, so n_repetitions is mandatory.
    """
    import hashlib
    import json as _json
    man = {"model": model, "temperature": temperature, "top_p": top_p,
           "prompts_sha256": prompts_sha256, "golden_n": golden_n,
           "golden_freeze_date": golden_freeze_date, "law_version": law_version,
           "seed": seed, "n_repetitions": n_repetitions, **extra}
    man["config_sha256"] = hashlib.sha256(
        _json.dumps(man, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]
    return man
