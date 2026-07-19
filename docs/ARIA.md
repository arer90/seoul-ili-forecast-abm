# ARIA — LLM Advisory Layer (Stage 6a)

> An LLM layer that advises on Seoul influenza forecasting and simulation. Tool calling + multi-LLM
> orchestration + **grounding in Korean infectious-disease-prevention law and KDCA sources**.
> Hallucination is blocked; every piece of advice carries a source.
> Code: `simulation/server/` (MCP epi server) · `web/lib/hermes.ts` (orchestrator) ·
> `web/components/{ChatPanel,AriaSimPanel}.tsx`. Related: [ABM.md](ABM.md) · [WEB_DASHBOARD_GUIDE.md](WEB_DASHBOARD_GUIDE.md).

---

## 1. 3-tier architecture

| tier | contents | code |
|------|------|------|
| **tool** | Read-only DB queries · forecasts · scenarios · literature RAG — invoked by the LLM | `server/mcp_epi.py` (11 tools) |
| **retrieval** | Real-data digest + literature + [법령·KDCA wiki] (law and KDCA wiki) grounding block | `web/lib/aria-wiki.json`, static_citations |
| **summarization** | Multi-LLM orchestration + synthesis | `web/lib/hermes.ts` |

The LLM is used **as an orchestrator over general-purpose models** — it only calls tools and summarizes their output; it never generates the numbers itself (thesis S6).

---

## 2. MCP epi server (`simulation/server/`)

| file | role |
|------|------|
| `mcp_epi.py` | **11 epi tools** (below) + unified provenance envelope (M7) |
| `mcp_stdio.py` | JSON-RPC 2.0 stdio transport |
| `sql_guard.py` | Read-only guard for `epi.query_db` — allows **SELECT-class statements only** |
| `epimas_adapter.py` | Thin facade over the Stage 5 simulation (`run_metapop_scenario`, `list_scenarios`) |
| `static_citations.py`, `metrics.py` | Static citations and evaluation metrics |

**11 tools**: `epi.query_db` · `epi.forecast` · `epi.model_compare` · `epi.shap_features` · `epi.rt_estimate` · `epi.lead_time_analysis` · `epi.outbreak_detect` · `epi.validity_check` · `epi.literature_rag` · `epi.scenario_run` · `epi.international_compare`

The web app calls them through `/api/mcp/[tool]` (a thin proxy → Python MCP bridge).

---

## 3. Orchestrator (`web/lib/hermes.ts`) — 4 modes

| mode | behaviour |
|------|------|
| **solo** | Streaming from a single provider |
| **parallel** | N providers in parallel → column-by-column comparison, tagged by `providerId` |
| **synthesis** | Runs in parallel, then a synthesiser merges the answers |
| **relay** | Sequential chain — each answer is passed to the next provider as `[Previous from X]` |

`/api/chat` (SSE stream, `requireAuth` + rate-limit) ↔ `ChatPanel.tsx`.

---

## 4. Grounding — citation policy (`SYSTEM_GROUNDING`)

**Prepended in common to every provider × every mode**, so each LLM follows the same evidence policy. Cite in tier order; do not refuse just because a higher tier is unavailable:

1. **MCP tools** (when available) — `[tool:<name>]`
2. **[법령·KDCA wiki]** (law and KDCA wiki, §5) — Korean infectious-disease law + the Korea Disease Control and Prevention Agency — `[law:감염병예방법]` / `[data:KDCA]`. **No hallucinating article numbers or disease classes.**
3. **[실데이터 digest]** (real-data digest) — real values baked in from epi_real_seoul.db — `[tool:data_digest]`
4. **[현재 컨텍스트]** (current context) — simulation day / selected district / scenario — `[tool:sim_context]`
5. **Established epidemiological knowledge** (KDCA/WHO/meta-analyses, with caveats stated) — `[기존 문헌]`

Always state which tier a statement came from. If MCP is unavailable, compute by hand from the digest + SEIR and prefix the answer with `(추정 — 실 모델 미연동)` ("estimate — real model not connected").

---

## 5. KDCA-law wiki (grounding for each LLM)

```
web/scripts/build_aria_wiki.py   # DB(disease_master statutory class·sentinel ILI·kosis_kdca_notifiable)
   → web/lib/aria-wiki.json        + 「감염병의 예방 및 관리에 관한 법률」 (Infectious Disease Control and Prevention Act) articles (digest)
   → hermes.ts KDCA_WIKI block (~510 tokens)  → appended to SYSTEM_GROUNDING (received by each LLM)
```

**Core facts**: influenza = **제4급감염병 (Class 4 notifiable infectious disease) → 표본감시 (sentinel surveillance)** → clinics nationwide report ILI (per 1,000 patients) weekly = **the prediction target of this project**. Article 11 notification (300만원 / KRW 3 million fine for failure to report), Article 18 epidemiological investigation, Articles 45, 47 and 49 control measures (assembly bans, school closures, isolation), Articles 24 and 25 vaccination. Sources: law.go.kr · KDCA dportal · easylaw.go.kr.

**Live verification** (2026-06-06, `claude-haiku-4-5`, real key): with the wiki injected, influenza was cited **accurately** as 제4급감염병 (Class 4), 표본감시 (sentinel surveillance), 7-day notification, the 300만원 fine, and Articles 47 and 49, with **zero article hallucinations** (6/6 on self-verification). OpenAI and Google get the same grounding automatically once their keys are filled in (it is a shared layer).

> Reproducibility: the statutory articles are authoritative references (summarized, verified against law.go.kr), and the KDCA data is generated from the DB — so updates are traceable.

---

## 6. Two-way map integration (`AriaSimPanel.tsx`, `/map3d/abm`)

- **Map → ARIA**: a rule-based briefing is generated automatically from the scenario trajectory (top infected districts · incidence rate · intervention effect · legal basis). Works even without an LLM key.
- **ARIA → map**: clicking a district chip in an answer or briefing highlights that district on the map.
- Free-form questions → `/api/chat` (hermes) → `SYSTEM_GROUNDING` (+wiki) applied. Without a key it degrades gracefully to the briefing.

---

## 7. Evaluation (P5 — LLM comparison benchmark)

`simulation/llm_compare/` — evaluation of generative, RAG and multi-LLM advice. **The authoritative
foundation is public-health / epidemiology institutions** (WHO · KDCA · US CDC · the Infectious Disease
Control and Prevention Act · MFDS — the authority for correct answers is anchored here);
**reporting format is supplementary**: CHART + TRIPOD-LLM (not TRIPOD+AI, which is for prediction
models; no authority is claimed for it), and the dimensions are the **Bedi 7-dimension** set (JAMA 2025).
For the full 3-layer authority structure and the status of B-1 through B-7, see
**[LLM_EVAL_STANDARDS.md](LLM_EVAL_STANDARDS.md) §0**.

| axis | function (`comparison.py`) |
|----|------|
| Accuracy · answer key | `golden_set.py` (source-anchored) + `judge.py` (rule-based, deterministic) |
| RAG generation faithfulness | `faithfulness` (RAGAS, claim grounding) |
| RAG retrieval quality | `context_precision` / `context_recall` (did it retrieve the right articles and KDCA facts?) |
| judge bias | `judge_position_debias` · `verbosity_bias` (Zheng 2023) |
| κ interpretation | `landis_koch_band` (1977 bands, with the caveat that they are not cutoffs) |
| Clinical harm | `harm_summary` (zero-tolerance gate for critical harm, DECIDE-AI) |
| Power · MCID | `n_for_power` (number of queries needed, determined in advance) + effect size · 95% CI |
| Reproducibility | `repetition_variance` (LLM non-determinism; snapshot and temperature reported) |

Statistics: pairwise Wilcoxon + Holm correction. Details in `docs/PROOF_VALIDATION_PROTOCOL_20260606.md`.

---

## 8. Safety and governance

- **Provenance envelope (M7, D1)**: every piece of advice is traceable to its evidence tier.
- **Staleness guard (D2)**: warns about stale vintage data.
- **Read-only SQL** (`sql_guard.py`): `epi.query_db` is SELECT-class only.
- **Citation enforcement**: blocks unstated tiers and hallucinations.

---

## 9. Running it

```bash
# MCP epi server (stdio)
python -m simulation mcp-server               # = cmd_mcp_server

# Web ARIA chat (a provider key is required)
cd web && npm run dev                          # → left-hand panel of /map3d/abm, or ChatPanel
#   ANTHROPIC_API_KEY / OPENAI_API_KEY / GOOGLE_API_KEY in web/.env.local

# Regenerate the wiki (after the DB is updated)
python web/scripts/build_aria_wiki.py
```
