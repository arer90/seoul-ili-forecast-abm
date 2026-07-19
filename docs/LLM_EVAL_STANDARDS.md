# ARIA P5 — LLM Evaluation Standards & Protocol

> Aligns the evaluation of ARIA's LLM layer with medical, RAG, and multi-LLM standards. Reflects
> two external reviews (2026-06-06). Code: `simulation/llm_compare/`. Related: [ARIA.md](ARIA.md) §7,
> the external-review response (not shipped).

---

## 0. Authority Foundation + Standard Selection (relocated 2026-06-07 — explicit user instruction)
**Authority comes from the "source of content," not from the "reporting format."** Three separated layers — layer 1 (institutional authority) is the foundation; layer 3 (reporting format) is merely optional transparency.

### 0.1 ★Primary authority — content and correctness (public-health / public-health-epidemiology institutions)
Whether an ARIA answer is right or wrong is adjudicated by **official public-health institutions and statutes**. All ground truth is anchored here:
- **WHO** (GISRS · FluNet · LMM governance 2024) · **KDCA (Korea Disease Control and Prevention Agency)** (sentinel surveillance · epidemic advisories · immunization guidelines) · **US CDC** (FluView · ILINet · clinical guidelines)
- **Infectious Disease Control and Prevention Act** (articles on the National Law Information Center, law.go.kr) · **MFDS (Ministry of Food and Drug Safety)** (2025 guidance on generative-AI medical devices) · **Korean Society of Infectious Diseases / Korean Society for Preventive Medicine** · **Cochrane** · peer-reviewed epidemiological literature
- Code anchors: `kr_epi_bench.py` 40 QA + `golden_set.py` — all from the sources above. **Zero anchors point to TRIPOD-LLM or academic consensus statements** (authority is borne by the institutions).

### 0.2 Evaluation methodology (domain-independent, rigor)
accuracy · RAGAS faithfulness · ALCE citation · calibration/abstention · harm critical-gate · expert κ · Wilcoxon/Holm (§1–§5).

### 0.3 Reporting format (optional supplement — transparency only, no authority claim)
ARIA is **generative, conversational, retrieval-grounded advisory** → prediction-model standards (TRIPOD+AI / PROBAST+AI) are not governing (a category error). Formats used as a **supplement** for reporting transparency (all newly created in 2025 — not an authority layer; removing them leaves the thesis's authority unchanged):
- **CHART** (Chatbot Assessment Reporting Tool, 2025) — reporting for chatbot health advice. High fit in form.
- **TRIPOD-LLM** (Gallifant, *Nat Med* 2025;31:60-69) — reporting for medical LLM evaluation. **It is an evaluation criterion (checklist) that inspects the *content* of the 4-layer design and implementation, not the *foundation*** (explicit user instruction 2026-06-07 — the foundation is the institutions in §0.1). It descends from the EQUATOR/TRIPOD lineage but is a **supplement**. Mapping: TRIPOD_LLM_CHECKLIST.md (TRIPOD-LLM checklist, not shipped).
- **METRICS** (Sallam 2024) — design hygiene.
- These are merely a *transparency floor* for reviewers who demand a reporting checklist.

> ⚠ Scope boundary: if epidemiological advisory (ILI interpretation) is the main track, focus on the common core items. Keep consistency with §M.7 (no clinical assertions).

---

## 1. Evaluation dimensions — Bedi 7-dimension (JAMA 2025;333:319)
> Review of 519 papers: 95.4% covered accuracy only, only 5% used real data, calibration 1.2%. Covering all 7 dimensions + real queries is in itself a contribution.

| Dimension | ARIA measurement | Code/status |
|------|-----------|-----------|
| 1 Accuracy | agreement with the answer set | `golden_set.py` + `judge.py` ✓ |
| 2 Comprehensiveness | coverage of the answer | golden-set pillars ✓ |
| 3 Factuality/grounding | **RAGAS + citation** | §3 ✓ (new) |
| 4 Robustness | prompt-sensitivity range | △ additionally recommended (Sclar 2023) |
| 5 Fairness/bias | stratification by district, age, language | △ additionally recommended (Obermeyer lens) |
| 6 Calibration/uncertainty | repetition variance · confidence | `repetition_variance` ✓ (new) |
| 7 Deployment | latency · cost | △ additionally recommended |

---

## 2. RAG grounding (RAGAS 4-metric, Es 2024)
**Closes the core gap** — the existing "tool faithfulness" was only part of *generation* fidelity and had no *retrieval* quality (a faithful answer to the wrong chunk still scored high). Separated into 4 metrics:
- **faithfulness** (generation): the proportion of claims in the answer that can be inferred from the context — `faithfulness()`. Measured: 1.0 on statute-wiki answers.
- **context_precision** (retrieval): did it retrieve the correct article / KDCA source *near the top* — `context_precision()`. Measured 1.0.
- **context_recall** (retrieval): did it retrieve *all* of the correct articles/sources — `context_recall()`. Measured 1.0.
- answer_relevancy — △ (question-answer alignment).
- citation: **ALCE** precision/recall (Gao 2023) + **FActScore** atomic claim (Min 2023) + **Med-HALT** hallucination probe (Pal 2023) — recommended.

---

## 3. Judge reliability (Zheng 2023)
- The current production judge = **rule-based deterministic** (`judge.py`) → **unaffected by** position/verbosity/self-enhancement **bias**.
- When using the optional LLM-judge: `judge_position_debias` (consistency under position swap) + `verbosity_bias` (length) + **separation of judge≠candidate** + comparison against a human subset κ. κ interpretation = `landis_koch_band` (the 1977 bands, with the caveat that they are not statistical cutoffs).

---

## 4. Safety and harm (DECIDE-AI, Vasey 2022)
- Mandatory citation and the provenance envelope provide *traceability*, not harm measurement.
- `harm_summary`: error severity (none/minor/major/**critical**) + a **critical zero-tolerance gate**. It separates out fatal errors (wrong grade, wrong isolation, wrong dosage) that would otherwise be buried in an average metric.
- Additionally recommended: resistance to jailbreak and system-prompt injection (Modi 2025: 88/100 disinformation), plus logging of refusal behavior.

---

## 5. Statistics and reproducibility
- **Sample size / power**: `n_for_power` (number of prior queries, d=0.5 · power 0.8 → 32) + effect size + 95% CI + a pre-specified MCID. (Statistical significance ≠ practical difference, §M.4)
- **Reproducibility**: `repetition_variance` (LLM non-determinism — "zero hallucination" cannot be asserted from n=1) + reporting of model snapshot, temperature, top_p, repetition count, and evaluation vintage (FAIR; TRIPOD-LLM repro).

---

## 6. Status of B-1 – B-7 (issues raised by external review)
| # | Issue raised | Status |
|---|------|------|
| B-1 | faithfulness too narrow (no retrieval) | ✅ context_precision/recall implemented |
| B-2 | judge bias not mitigated | ✅ functions (b222785) + current judge is rule-based (unaffected) |
| B-3 | no gold set | ✅ **the existing `golden_set.py`** (source-anchored) — recommend adding freeze-date and statute version |
| B-4 | no hallucination/harm handling | ✅ `harm_summary` + critical gate |
| B-5 | sample size / power / MCID not reported | ✅ `n_for_power` (reporting effect size and CI recommended) |
| B-6 | LLM non-determinism and reproducibility | ✅ `repetition_variance` + snapshot reporting |
| B-7 | TRIPOD-LLM checklist not enclosed | ✅ **TRIPOD_LLM_CHECKLIST.md (TRIPOD-LLM checklist, not shipped)** — mapping of 26 domain items (22 ✅ / 3 N/A or planned / 1 △ fairness), with pandoc PDF/HTML render commands enclosed |

---

## 7. Regulatory context (at deployment)
- **Korea MFDS** issued the world's first guidance on generative-AI medical devices (2025-01) — 6 harm categories, clinician grading of severity, clinician-in-the-loop. (A KDCA public-health LLM guideline has not been confirmed — needs direct verification.)
- **US FDA** lifecycle/PCCP/GMLP, **EU AI Act** (medical AI = high-risk), **WHO LMM** governance. Even at the research stage, keeping the documentation aligned with those expectations means the deployment stage inherits it.

---

## References (review sources)
CHART (Huo, *Ann Fam Med* 2025;23:389, doi:10.1370/afm.250386) · TRIPOD-LLM (Gallifant, *Nat Med* 2025;31:60, doi:10.1038/s41591-024-03425-5) · Bedi (*JAMA* 2025;333:319, doi:10.1001/jama.2024.21700) · RAGAS (Es, arXiv:2309.15217) · ALCE (Gao, EMNLP 2023, arXiv:2305.14627) · FActScore (Min, arXiv:2305.14251) · Med-HALT (Pal, CoNLL 2023) · Zheng (NeurIPS 2023, arXiv:2306.05685) · Landis&Koch (*Biometrics* 1977;33:159) · DECIDE-AI (Vasey, *Nat Med* 2022;28:924) · METRICS (Sallam, *IJMR* 2024;13:e54704) · QUEST (Tam, *npj Digit Med* 2024;7:258) · Sclar (arXiv:2310.11324) · MFDS overview (Park, *KJR* 2025;26:519) · WHO LMM 2024.

---

## 8. Implementation status (2026-06-07) — priorities 1, 2, 3, 5 + re-evaluation of #4

Implemented and tested (14 green) in `simulation/llm_compare/`:

| Priority | Function/module | Notes |
|---------|-----------|------|
| **1 Korean benchmark** | `kr_epi_bench.py` — `KR_EPI_LAW_QA` (**n=40**; classification 5 / notification 5 / control 5 / immunization 5 / epidemiology 14 / data 6, **anchored to official sources**: law.go.kr articles, KDCA, KOSIS, WHO — not thesis sections → avoids contamination) + `load_kormedmcqa` (**live connection complete** — sean0042/KorMedMCQA, test **3,009 items**, scored with `format_mcqa`/`score_mcqa`) | Since ARIA is epidemiological advisory, epidemiology and statute QA is the main track for the domain (n=40, exceeding the power requirement of 32 at d=0.5); KorMedMCQA's 3,009 items serve as the external contamination-check anchor. **Bespoke items are capped by quality** (padding to 80 = contamination risk, per 3-LLM consensus) — total benchmark 40 + 3,009 ≫ 80 |
| **2 External factuality** | `citation_metrics` (ALCE precision/recall) + the existing `harm_summary` and `context_precision/recall` | **Comparison against external authoritative sources** — a self-wiki context_precision of 1.0 is a ceiling artifact |
| **3 Abstention** | `risk_coverage_curve` (selective accuracy · AURC) | The thinnest dimension of Bedi-7D |
| **5 Reproducibility** | `repro_manifest` (model snapshot · temp · prompt hash · gold-set freeze · statute version · seed · repetition + config_sha256) | LLMs are not reproducible via seed → repetition is mandatory |

### Re-evaluation of #4 (in response to the user's challenge, "why the demotion?") — **not demoted, but corrected**
- The "64–68% expert agreement" figure is for an LLM-judge acting as **sole adjudicator**. Zheng 2023 also reports 80%+ (human level) once bias is mitigated.
- The current production judge is **rule-based** (token presence only; it cannot assess reasoning quality). The LLM-judge **compensates for** that weakness.
- Therefore, **3 tiers**: ① rules (reproducibility gate) ② **LLM-judge (reasoning, with bias mitigation — `judge_position_debias`/`verbosity_bias`)** ③ humans (calibration subset). `judge_tier_agreement` **calibrates each tier against humans using Cohen's κ** → expand at κ≥0.60, human adjudication when below = **conditional trust**.
- Demoting it would forfeit the entire assessment of reasoning quality, which neither the rules (too shallow) nor a small number of humans (limited n) can perform.

### Data/execution stage — complete (2026-06-07)
| Item previously outstanding | Action | Output |
|--------------|------|------|
| Bulk load of KorMedMCQA (network) | ✅ installed `datasets` + live connection | doctor 435 · nurse · pharm 885 · dentist 811 = **test 3,009**; `format_mcqa`/`score_mcqa` normalization and scoring helpers |
| Expand epidemiology QA to n≥80 | ✅ 12→**40** (capped by quality; exceeds the power requirement of 32) + KorMedMCQA 3,009 external = **total ≫80** | balanced across 6 domains, all anchored to official sources |
| Run the expert panel (κ) | ✅ **harness + protocol + real κ pipeline** `expert_panel.py` (blinded sheet · Fleiss κ · tier Cohen κ · templates) | dry-run measurements: expert Fleiss κ · rule κ=0.70 (scale-ok) · llm κ=0.55 (human adjudication). **Human ratings are not fabricated** — they are collected via the template |
| TRIPOD-LLM checklist | ✅ **TRIPOD_LLM_CHECKLIST.md (TRIPOD-LLM checklist, not shipped)** (closes B-7) | 26-item mapping, explicitly stated as not CVPR/ICCV, pandoc render |

> **Tests**: `test_llm_compare_hardening.py` (15) + `test_expert_panel.py` (5) = **20 green**.
> **Remaining real-data stage** (collection, not code): ① generate real LLM responses (live API, gated) → blinded sheet → ②
> recruit ≥3 real experts and collect their ratings (`rating_template_csv` template) → real κ from `panel_report`. The harness is ready.
