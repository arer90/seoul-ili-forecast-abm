"""Build a small grounded-epidemiology instruction dataset for ARIA LoRA fine-tuning.

Produces train/valid/test JSONL (mlx-lm chat format) under
simulation/results/aria_lora/data/. Each example teaches the ARIA *style*: a
concise, grounded epidemiological answer that carries citation tags
([data:...] / [law:...] / [기존 문헌]) and never invents a figure. Sources:
- the project's static citation catalogue (simulation.server.static_citations);
- a curated set of Seoul-ILI / SEIR-V-D / surveillance domain Q&A.

This is a SMALL demonstration dataset; the LoRA result is measured honestly
against the base model (a biomedical fine-tune may not beat the base — Survey
caution). Deterministic (seeded split), offline, no network.

Run:  .venv/bin/python -m simulation.scripts.build_aria_lora_data
"""
from __future__ import annotations

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "results" / "aria_lora" / "data"

# Curated ARIA-style grounded domain Q&A (concise, citation-tagged, honest).
DOMAIN: list[tuple[str, str]] = [
    ("What does the Korean ILI sentinel system measure?",
     "It measures a notification rate of influenza-like illness from sentinel clinics that elect to report, not a complete incidence count [data: KDCA sentinel]; the reporting denominator can shift between seasons, so under-ascertainment is structural."),
    ("Is ILI the same as laboratory-confirmed influenza?",
     "No. ILI is a syndromic surveillance indicator, not laboratory-confirmed influenza [기존 문헌]; outputs should inform, not replace, laboratory and clinical judgement."),
    ("How is influenza vaccine effectiveness typically estimated?",
     "Most often by a test-negative design comparing vaccination odds among test-positive and test-negative patients [기존 문헌]; report the point estimate with its interval rather than a single number."),
    ("What does the basic reproduction number R0 represent?",
     "R0 is the expected number of secondary infections from one case in a fully susceptible population [기존 문헌]; the time-varying effective reproduction number Rt reflects immunity and interventions."),
    ("Why report a weighted interval score (WIS) for forecasts?",
     "WIS is a proper score that jointly rewards calibrated central prediction and well-sized intervals [기존 문헌], so it evaluates probabilistic forecasts rather than point accuracy alone."),
    ("What does the behavioral contact multiplier capture in the SEIR-V-D model?",
     "It scales the force of infection by voluntary contact reduction driven by risk perception and adherence fatigue [data: SEIR-V-D]; with behaviour off it reverts to the baseline transmission."),
    ("How should district-level ABM outputs be interpreted?",
     "As mechanistic scenarios for spatially targeted intervention, not as validated sub-city point forecasts [data: ABM]; district parameters are read from city-aggregate data, so inference is ecological."),
    ("What limits cross-season generalization of these results?",
     "The behavioural evidence rests on a single post-COVID season with few effectively independent rolling origins [data: forward], so cross-season and cross-pathogen transfer are future work, not claims."),
    ("What is reporting lag and why does it matter for forecasting?",
     "Recent surveillance weeks are provisional and revised upward as late reports arrive [data: vintage]; a model trained on finalized values can be optimistic on provisional, right-censored data."),
    ("How are prediction intervals calibrated here?",
     "Raw intervals under-cover before recalibration [기존 문헌]; an adaptive conformal step is applied so empirical coverage approaches the nominal level."),
    ("What does a leaky vaccine assumption mean in the model?",
     "A leaky vaccine reduces the per-contact infection probability by (1 - VE) for vaccinated individuals rather than fully protecting a fraction [기존 문헌]."),
    ("Why are some behavioural parameters called weakly identified?",
     "Profile-likelihood and ABC-SMC analysis show the fatigue and threshold parameters trade off along sloppy directions [data: identifiability], so they are reported as a calibrated regime, not pinned constants."),
    ("What is the role of non-pharmaceutical interventions in Rt?",
     "NPIs such as distancing and masking lower contact rates and thus Rt [기존 문헌]; attributing an Rt change to a single NPI is confounded by concurrent behaviour change."),
    ("How is the spatial down-scaling to 25 districts driven?",
     "By a real-time-available daytime living-population density feature [data: density], making the product a density-driven nowcasting layer rather than a retrospective reconstruction."),
    ("Should the advisory layer's output be acted on directly?",
     "No. It is an interpretation surface whose outputs require human ratification [law: clinical governance]; high-risk or ungrounded claims are routed for review."),
]

# Each citation -> one grounded Q&A in the ARIA style.
def _citation_examples() -> list[tuple[str, str]]:
    try:
        from simulation.server.static_citations import STATIC_CITATIONS
    except Exception:
        return []
    out = []
    for c in STATIC_CITATIONS:
        d = c.to_dict() if hasattr(c, "to_dict") else c
        title = d.get("title", "").strip()
        rel = (d.get("relevance") or d.get("abstract") or "").strip()
        if not title or not rel:
            continue
        topic = (d.get("tags") or ["this topic"])
        topic = topic[0] if topic else "this topic"
        q = f"What is the evidence on {topic} relevant to Seoul ILI?"
        a = f"{rel} [기존 문헌: {title[:80]}]"
        out.append((q, a))
    return out


def _to_chat(q: str, a: str) -> dict:
    return {"messages": [
        {"role": "user", "content": q},
        {"role": "assistant", "content": a},
    ]}


def main() -> int:
    pairs = DOMAIN + _citation_examples()
    rng = random.Random(42)
    rng.shuffle(pairs)
    n = len(pairs)
    n_val = max(2, n // 8)
    n_test = max(2, n // 8)
    valid = pairs[:n_val]
    test = pairs[n_val:n_val + n_test]
    train = pairs[n_val + n_test:]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (("train", train), ("valid", valid), ("test", test)):
        with (OUT / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for q, a in rows:
                fh.write(json.dumps(_to_chat(q, a), ensure_ascii=False) + "\n")
    print(f"total={n}  train={len(train)} valid={len(valid)} test={len(test)}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
