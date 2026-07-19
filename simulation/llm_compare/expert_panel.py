"""Blinded expert-panel harness for ARIA LLM-judge calibration (P5 priority #4).

The LLM-judge is **calibrated, not demoted** (user challenge "왜 강등?"). This
module operationalizes the calibration: ≥3 blinded human experts score each LLM
response pass/fail; we then report

  1. **Fleiss κ among experts** — is the human panel itself reliable? (Landis-Koch
     band; <0.40 ⇒ rubric/training problem, not a judge problem)
  2. **per-tier κ vs the expert consensus** — `judge_tier_agreement` Cohen κ for
     each automated tier (rule / LLM-judge). κ≥0.60 ⇒ that tier may auto-scale;
     below ⇒ human adjudication required.

NO human ratings are fabricated. This module GENERATES the blinded rating sheet,
exports a blank template, scores whatever automated tiers are runnable, and
aggregates real human ratings supplied via the filled template. `dry_run()`
exercises the full κ pipeline on **explicitly-synthetic** rater streams (labeled
SYNTHETIC) so the plumbing is verifiable before real experts are recruited —
those numbers are a self-test, never reported as expert findings.

Design (D-4 deep module): small interface (make_sheet / template / load /
report / dry_run), rich blinding + κ-aggregation inside. Portable (csv + numpy
only; no pandas/network). Side effects: only the explicit *_csv / *_json writers.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

import numpy as np

from simulation.llm_compare.comparison import (
    cohen_kappa,
    fleiss_kappa_ratings,
    landis_koch_band,
)


@dataclass(frozen=True)
class PanelResponse:
    """One LLM response to be rated by the panel (the unit of blinding)."""
    item_id: str          # golden/epi item this answers
    backend: str          # which LLM produced it (HIDDEN from raters)
    question: str
    answer: str


@dataclass
class BlindedSheet:
    """Source-blinded, order-shuffled rating sheet + the private unblinding key.

    Raters see only ``rows`` (opaque sheet_id + question + answer); ``key`` maps
    each sheet_id back to (item_id, backend) for post-hoc aggregation. Keeping
    the key separate is what makes the panel blind to model identity + position.
    """
    rows: list[dict] = field(default_factory=list)        # [{sheet_id, question, answer}]
    key: dict[str, dict] = field(default_factory=dict)    # sheet_id -> {item_id, backend}


def _sheet_id(item_id: str, backend: str, salt: str) -> str:
    """Opaque, deterministic per-response id (hides item_id/backend ordering)."""
    return "R" + sha256(f"{salt}:{item_id}:{backend}".encode()).hexdigest()[:10]


def make_blinded_sheet(responses, *, seed: int = 42, salt: str = "aria-panel") -> BlindedSheet:
    """Build a blinded, shuffled rating sheet from LLM responses.

    Args:
        responses: iterable of ``PanelResponse`` (or dicts with the same keys).
        seed: RNG seed for the deterministic shuffle (reproducible blinding).
        salt: string mixed into the opaque sheet ids.
    Returns:
        ``BlindedSheet`` (rows are source-blind; key holds the unblinding map).
    """
    items = [r if isinstance(r, PanelResponse) else PanelResponse(**r) for r in responses]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(items))
    sheet = BlindedSheet()
    for idx in order:
        r = items[int(idx)]
        sid = _sheet_id(r.item_id, r.backend, salt)
        sheet.rows.append({"sheet_id": sid, "question": r.question, "answer": r.answer})
        sheet.key[sid] = {"item_id": r.item_id, "backend": r.backend}
    return sheet


def rating_template_csv(sheet: BlindedSheet, path) -> Path:
    """Write a blank per-rater CSV (sheet_id, question, answer, verdict, notes).

    Each expert fills ``verdict`` with pass/fail (notes optional). The sheet is
    source-blind, so raters cannot infer model identity. Returns the path.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sheet_id", "question", "answer", "verdict(pass/fail)", "notes"])
        for row in sheet.rows:
            w.writerow([row["sheet_id"], row["question"], row["answer"], "", ""])
    return p


def load_ratings_csv(path) -> dict:
    """Load one rater's filled template → ``{sheet_id: verdict}`` (blank skipped)."""
    out: dict = {}
    with Path(path).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            v = (row.get("verdict(pass/fail)") or row.get("verdict") or "").strip()
            if v:
                out[row["sheet_id"]] = v
    return out


def _align(sheet: BlindedSheet, ratings_by_rater: dict) -> tuple[list, dict]:
    """Restrict to sheet_ids every rater scored; return (ordered_ids, per-rater lists)."""
    ids = [r["sheet_id"] for r in sheet.rows]
    common = [sid for sid in ids if all(sid in rr for rr in ratings_by_rater.values())]
    cols = {rid: [rr[sid] for sid in common] for rid, rr in ratings_by_rater.items()}
    return common, cols


def expert_consensus(ordered_ids, cols) -> list:
    """Majority pass/fail per item across raters (ties → fail = conservative)."""
    def _b(v):
        return 1 if str(v).strip().lower() in ("pass", "1", "true", "yes") else 0
    out = []
    n = len(cols)
    for i in range(len(ordered_ids)):
        votes = sum(_b(cols[r][i]) for r in cols)
        out.append("pass" if votes * 2 > n else "fail")
    return out


def panel_report(sheet: BlindedSheet, human_ratings_by_rater: dict, *,
                 rule_verdicts: dict | None = None,
                 llm_verdicts: dict | None = None) -> dict:
    """Aggregate a blinded panel into κ statistics + per-tier calibration verdicts.

    Args:
        sheet: the ``BlindedSheet`` the raters scored.
        human_ratings_by_rater: ``{rater_id: {sheet_id: verdict}}`` (≥2 raters).
        rule_verdicts / llm_verdicts: optional ``{sheet_id: verdict}`` from the
            automated tiers, scored on the SAME sheet (for calibration κ).
    Returns:
        ``{n_items, n_raters, expert_fleiss, expert_band, tier_calibration,
        per_item}`` — ``tier_calibration[tier] = {kappa, band, scale_ok}``
        (scale_ok = κ≥0.60 vs expert consensus ⇒ tier may auto-scale).
        Never raises; returns ``{error}`` if too few raters/items.
    """
    if len(human_ratings_by_rater) < 2:
        return {"error": f"need ≥2 human raters (got {len(human_ratings_by_rater)})"}
    ids, cols = _align(sheet, human_ratings_by_rater)
    if len(ids) < 2:
        return {"error": "need ≥2 commonly-rated items across all raters"}

    fk = fleiss_kappa_ratings(cols)
    consensus = expert_consensus(ids, cols)

    tiers: dict = {}
    for name, verds in (("rule", rule_verdicts), ("llm", llm_verdicts)):
        if not verds:
            continue
        # align tier verdicts to the consensus item order
        tier_seq, cons_seq = [], []
        for sid, c in zip(ids, consensus):
            if sid in verds:
                tier_seq.append(1 if str(verds[sid]).strip().lower()
                                in ("pass", "1", "true", "yes") else 0)
                cons_seq.append(1 if c == "pass" else 0)
        if len(tier_seq) >= 2:
            ck = cohen_kappa(cons_seq, tier_seq)
            k = ck.get("kappa", 0.0)
            tiers[name] = {"kappa": k, "band": landis_koch_band(k),
                           "n": ck.get("n", len(tier_seq)),
                           "scale_ok": k >= 0.60}

    return {
        "n_items": len(ids), "n_raters": len(human_ratings_by_rater),
        "expert_fleiss": fk.get("kappa"), "expert_band": fk.get("band"),
        "expert_reliable": (fk.get("kappa") or 0) >= 0.40,
        "tier_calibration": tiers,
        "consensus_pass_rate": round(sum(c == "pass" for c in consensus) / len(consensus), 3),
        "per_item": [{"sheet_id": s, **sheet.key.get(s, {}), "consensus": c}
                     for s, c in zip(ids, consensus)],
    }


def write_report_json(report: dict, path) -> Path:
    """Persist a panel report to JSON (utf-8). Returns the path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


# ── DRY-RUN (pipeline self-test on SYNTHETIC raters — never reported as real) ──
def dry_run(n_items: int = 30, *, seed: int = 42) -> dict:
    """Exercise the full κ pipeline on explicitly-SYNTHETIC raters.

    Generates ``n_items`` fake responses + a latent ground truth, then three
    synthetic raters (two reliable, one noisy) and two automated tiers (rule:
    high-agreement, llm: moderate). Returns ``panel_report`` output so the
    plumbing (blinding → Fleiss → per-tier Cohen κ → scale_ok) is verifiable.

    THESE NUMBERS ARE A SELF-TEST, NOT EXPERT FINDINGS. The real panel replaces
    the synthetic raters with filled human templates (`load_ratings_csv`).
    """
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 2, n_items)                     # latent pass/fail
    responses = [PanelResponse(f"IT{i:03d}", "modelX",
                               f"q{i}", f"synthetic answer {i}") for i in range(n_items)]
    sheet = make_blinded_sheet(responses, seed=seed)
    ids = [r["sheet_id"] for r in sheet.rows]
    truth_by_id = {sid: int(truth[i]) for i, sid in enumerate(
        _sheet_id(r.item_id, r.backend, "aria-panel") for r in responses)}

    def _rater(flip_p):
        return {sid: ("pass" if (truth_by_id[sid] ^ int(rng.random() < flip_p)) else "fail")
                for sid in ids}
    humans = {"E1": _rater(0.05), "E2": _rater(0.08), "E3": _rater(0.25)}   # 2 reliable + 1 noisy
    rule = _rater(0.06)                                                     # automated tiers
    llm = _rater(0.18)
    return panel_report(sheet, humans, rule_verdicts=rule, llm_verdicts=llm)


__all__ = [
    "PanelResponse", "BlindedSheet", "make_blinded_sheet", "rating_template_csv",
    "load_ratings_csv", "expert_consensus", "panel_report", "write_report_json", "dry_run",
]
