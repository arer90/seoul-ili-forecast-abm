# API Key Locations — Separated by Domain

**Written**: 2026-04-28
**Purpose**: Prevent confusion — make it clear which key lives where

---

## 📁 Two domains, never merge them

```
project root/
├── simulation/data/api_key.txt    ← Python backend (data collection)
│                                       KOSIS, KMA (기상청), FluNet, Seoul Open Data
│                                       Used by the collectors that fill the SQLite DB
│
└── web/.env.local                  ← Next.js frontend (UI / ARIA)
                                        Maps (VWorld, Mapbox, Kakao, Naver)
                                        LLM (Anthropic, OpenAI, Google)
                                        ARIA Demo Token, MCP Bridge URL
```

**Why separate?**
- **Different languages** (Python vs TypeScript)
- **Different load mechanisms** (`open(...).read()` vs `process.env`)
- **Different build timing** (Python runtime vs Next.js build time)
- **Different deployment environments** (Python: server; Next.js: registered directly in the Vercel UI)

---

## 1. `simulation/data/api_key.txt` — data collection

### Keys currently held (example)
```
서울 열린 데이터 광장 (Seoul Open Data Plaza)
일반인증키 (인구/상점):    444...   # general auth key (population / shops)
지하철인증키:              464...   # subway auth key
지하철실시간 인증키:        4f4...   # subway real-time auth key

기상청: (KMA)
ASOS 일별 자료:           xxx      # ASOS daily data
중기예보:                  xxx      # medium-range forecast

FluNet WHO:
API token:                 xxx

KOSIS (통계청): (Statistics Korea)
인증키:                    xxx      # auth key
```

### Where it is used
```
simulation/collectors/group_a..r/*.py
simulation/collectors/import_external.py
   ↓
SQLite DB (simulation/data/db/epi_real_seoul.db)
```

### Issuing sites
| API | Homepage | Cost |
|-----|---------|------|
| Seoul Open Data Plaza | https://data.seoul.go.kr | Free |
| KMA ASOS | https://data.kma.go.kr | Free |
| FluNet WHO | https://www.who.int/tools/flunet | Free |
| KOSIS | https://kosis.kr/openapi | Free |
| HIRA | https://opendata.hira.or.kr | Free |

---

## 2. `web/.env.local` — UI / ARIA / LLM / maps

### Categories

#### A. ARIA / demo authentication (required)
```bash
DEMO_TOKEN=                          # demo gate
MCP_BRIDGE_URL=http://localhost:8787 # MCP server URL
```

#### B. LLM provider (at least one required)
```bash
ANTHROPIC_API_KEY=sk-ant-...         # Claude (most recommended)
OPENAI_API_KEY=sk-...                # GPT-4 (optional)
GOOGLE_API_KEY=AIza...               # Gemini (optional)
```
Issuing:
- Claude: https://console.anthropic.com/
- OpenAI: https://platform.openai.com/api-keys
- Gemini: https://aistudio.google.com/apikey

#### C. Map provider (optional; the 12 free layers need no key)
```bash
NEXT_PUBLIC_VWORLD_KEY=              # Korea 1m precision (optional)
NEXT_PUBLIC_MAPBOX_TOKEN=            # global (optional)
NEXT_PUBLIC_KAKAO_MAP_KEY=           # Kakao (next sprint)
NEXT_PUBLIC_NAVER_MAP_CLIENT_ID=     # Naver (next sprint)
NASA_EARTHDATA_TOKEN=                # NASA advanced (not needed for GIBS)
```
Issuing: see `simulation/results/MAP_API_SETUP_GUIDE.md`

#### D. UI settings (optional)
```bash
NEXT_PUBLIC_HIDE_OLLAMA=1
NEXT_PUBLIC_DEFAULT_LOCALE=ko
```

---

## 3. Never mix them up — responsibility per domain

| Task | Which key? | Which file? |
|------|---------------|-----------|
| ILI data collection (collectors) | KOSIS / KMA (기상청) / FluNet | `simulation/data/api_key.txt` |
| Python training (Phase 1-13) | None (uses the DB only) | — |
| Web UI map display | VWorld / Mapbox | `web/.env.local` |
| ARIA LLM answers | Claude / OpenAI | `web/.env.local` |
| MCP server (Python) | None | — |

→ **Do not confuse the two files** — the domains are completely different.

---

## 4. .gitignore protection (both)

```gitignore
# Python backend key
simulation/data/api_key.txt
simulation/data/api_key*.txt

# Next.js frontend key
web/.env.local
web/.env.production.local
.env*.local
```

→ Neither one is committed to git (secure).

---

## 5. Procedure for adding a new key

### Data collection API (Python)
```bash
# 1. Add it to simulation/data/api_key.txt
echo "신규 API: ..." >> simulation/data/api_key.txt   # "신규 API" = new API

# 2. Make simulation/collectors/group_X/X_collector.py read it
```

### UI / map API (Next.js)
```bash
# 1. Add it to web/.env.local
echo "NEXT_PUBLIC_NEW_KEY=..." >> web/.env.local

# 2. Use process.env.NEXT_PUBLIC_NEW_KEY in the web/ code

# 3. When deploying to Vercel, also register it under Settings → Environment Variables
```

---

## 6. Quick reference table

```
Question: "Where should I put this key?"

The place this key is used is...
  ✓ Python (simulation/) → simulation/data/api_key.txt
  ✓ Next.js (web/)       → web/.env.local
  ✓ Both                 → both (but the same value must be kept in sync)

The API you issued is...
  ✓ Data       (KOSIS / KMA / WHO)  → Python
  ✓ Maps       (VWorld / Mapbox)    → Next.js
  ✓ LLM        (Claude / OpenAI)    → Next.js (currently in the ARIA UI)
  ✓ Monitoring (Sentry / Datadog)   → both (if needed)
```

---

## One-line summary

> 🐍 **Python backend** = `simulation/data/api_key.txt` (KOSIS / KMA / FluNet)
> ⚛️ **Next.js frontend** = `web/.env.local` (maps / LLM / ARIA)
>
> **Do not merge them** — the domains differ. Refer to this document to avoid confusion.
>
> ⚠️ Both are covered by `.gitignore` — not committed to git (security).
