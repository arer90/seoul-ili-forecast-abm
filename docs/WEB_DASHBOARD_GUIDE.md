# Web Dashboard Guide (Seoul ILI · Metapop ABM · ARIA)

> Dashboard built on Next.js 14 / deck.gl / MapLibre. Two map pages plus the ARIA LLM advisor.
> Run: `cd web && npm run dev` → `http://localhost:3000/map3d` (dark basemap when internet access is available).

---

## 1. Pages

| Route | Contents |
|--------|------|
| `/map3d` | **Overview map** — 2D/3D × 21 toggleable layers × 3 presets + legend box |
| `/map3d/abm` | **ABM dashboard** — 25-gu SEIR-V-D infection animation × ARIA advisory (bidirectional) |
| `/llm-compare` | LLM comparison benchmark (separate page) |

Navigate via the "→ ABM 대시보드" (ABM dashboard) button in the `/map3d` header. Return via "← 전체 지도" (overview map) on the ABM page.

---

## 2. Overview map (`/map3d`)

### Complexity control (top right)
- **2D / 3D toggle**: switches pitch 50°↔0° (LinearInterpolator).
- **Presets**: `간단` (simple: gu boundaries + ILI) / `상세` (detail: adds subway, air quality, transfer stations, agents) / `전체` (full: everything).
- **21 individual toggles**: per-layer on/off. Default is `간단` (simple — keeps the view clean).
- **Legend box (bottom left)**: automatically shows what the sizes and colors of the enabled layers mean.

### Layers (all real data)
| Group | Layer | Encoding |
|------|--------|--------|
| Base | Gu boundaries · ILI heatmap | color = ILI rate |
| **Epidemiology** | Air quality / particulate matter · influenza vaccination · notifiable infectious disease incidence | gu color (PM10 / vaccination rate % / case count) |
| Subway | Lines (8) · stations (boardings/alightings, age) · transfer stations (interchange points) · congestion (POI) | size = boardings/alightings, ring = transfer |
| Bus | Routes (1,453) · stops (11,253) · boardings/alightings (709) | trunk/branch color, size = boardings/alightings |
| Real-time | Population and age · population forecast · road traffic · wind (S-DoT) | size = population, color = congestion/flow |
| Facilities | Hospitals and emergency rooms (ER available = green / full = red) · schools (1,422) · agent movement · commuting | size = beds, color = type |
| Tourism | Tourist area classification | color = place type (special tourist zone / park / palace / station area / commercial district) |

> ⚠ Real-time population and traffic POI data were last collected on 2026-04-16 (stale; the vintage is shown in the panel). Air quality, weather, subway, and bus data are current.

---

## 3. ABM dashboard (`/map3d/abm`)

- **4 scenarios** (linked to their legal basis in the Infectious Disease Control and Prevention Act):
  - 기준 (baseline: no intervention, Article 16 sentinel surveillance) / 봉쇄·거리두기 (lockdown and distancing, Article 49 ban on gatherings) / 학교 폐쇄 (school closure, Article 49 school suspension) / 백신 캠페인 (vaccination campaign, Articles 24 and 25 NIP)
- **SEIR-V-D animation**: gu color = infection rate I(t)/N; use the day slider to replay the spread.
- **Statistics**: peak day · attack rate · deaths · epidemiological-plausibility gate.
- **Trajectory source**: precomputed by `export_abm_scenarios.py` (Stage 5 `run_metapop_scenario` → (T,25,6)).

### ARIA advisory (left panel, bidirectional)
- **Map → ARIA**: a rule-based briefing is generated automatically from the trajectory (top infected gu, attack rate, intervention effect, legal basis). Works even without an LLM key.
- **ARIA → map**: click a gu chip → the corresponding gu is highlighted on the map.
- **Free-form questions** → `/api/chat` (best-effort; falls back gracefully to the briefing when no key is present).

---

## 4. ARIA KDCA / legal-statute grounding

`web/lib/hermes.ts` `SYSTEM_GROUNDING` (shared across every provider × solo/parallel/synthesis) injects a **[법령·KDCA wiki]** (statutes · KDCA wiki) tier:
- 「감염병의 예방 및 관리에 관한 법률」 (Infectious Disease Control and Prevention Act) — classes 1 through 4, **influenza = class 4, sentinel surveillance**, Article 11 notification, Article 18 epidemiological investigation, Articles 45, 47, and 49 control measures, Articles 24 and 25 immunization.
- Korea Disease Control and Prevention Agency data (`disease_master` legal class · `sentinel_influenza` ILI · `kosis_kdca_notifiable`) — `build_aria_wiki.py` generates these from the DB → `web/lib/aria-wiki.json`.
- Effect: **each LLM cites actual statutory articles and KDCA data when making claims about Korean law and notifiable infectious diseases**, preventing hallucinated articles and disease classes.

---

## 5. Data pipeline (updates)

```bash
python web/scripts/refresh_web_data.py          # 15 generators (DB→web)
```
- **LIVE layers** (automatic): `/api/mcp` (forecast · Rt · scenario) · `/api/overlays` (commuter · hospitals · live) · `/api/sim` — computed from the DB on every request.
- **SNAPSHOT layers** (on refresh): air quality · subway · weather · POI · vaccination · bus · schools · SEIR init · abm-scenarios · aria-wiki.
- **Automatic wiring**: `python -m simulation collect` runs the refresh automatically at the end (opt out with `--no-web-refresh`) → the map is brought up to date after collection.
- The generators are deterministic (byte-identical regeneration) → reproducibility ✓.

> For the list of generators and their sources, see the docstring in each `web/scripts/build_*.py`.
