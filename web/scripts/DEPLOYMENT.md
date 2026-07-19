# ARIA 배포 파이프라인 — 3-stage safety net

**목표**: API 비용을 태우기 **전에** 로컬에서 모든 것을 검증한 뒤,
가장 싼 방식으로 실제 provider 핑을 확인하고, 그제야 Vercel prod 배포.

```
  API 없이                                 API 키 필요                    실제 배포
  비용 $0                                  비용 < $0.001                   비용 0 (배포 고정)
  ─────────────                            ──────────────                  ─────────────
  Stage 1                                  Stage 2                         Stage 3
  01_preflight_local.ps1      ─────▶      02_preflight_providers.ps1      ─────▶    03_deploy_prod.ps1
    • python venv                            • Claude Haiku 3.5 ping          • S1+S2 리포트 강제 확인
    • db quick_check                         • GPT-4o-mini ping               • Turso seed import
    • MCP 10-tool smoke                      • Gemini 2.0 Flash ping          • Vercel env diff/주입
    • static aggregates                      • Turso SELECT 1                 • vercel deploy --prod
    • turso seed 크기                        • Upstash PING                   • 프로덕션 URL smoke
    • node/npm + build                                                        (4 엔드포인트)
    • Ollama + MCP E2E  ★
  ─────────────                             ──────────────                  ─────────────
  fail=0 이어야                             fail=0 이어야                    실패 시 exit 1
  Stage 2 진입                              Stage 3 진입                    리포트 저장
```

★ = 이게 핵심. **Ollama 로 MCP 10-tool chain 이 실제로 돌아가는지**
확인하므로 "Claude 호출 전 Hermes orchestration 이 깨졌는지"를 돈
안 쓰고 감지할 수 있음.

---

## Stage 1 — 로컬 무비용 검증

```powershell
.\web\scripts\01_preflight_local.ps1
```

### 옵션
| 플래그             | 효과                                                     |
|:-------------------|:---------------------------------------------------------|
| `-SkipBuild`       | `npm run build` 건너뛰기 (최근 빌드 성공 시)             |
| `-SkipOllama`      | Ollama 관련 모든 체크 스킵 (오프라인)                    |
| `-OllamaModel`     | E2E 스모크에 쓸 모델 (default: `qwen2.5:14b-instruct-q5_K_M`) |
| `-ReportPath`      | 리포트 JSON 경로                                         |

### 체크 항목 (9개)
1. **python_venv**       `.venv\Scripts\python.exe` 존재
2. **db_quick_check**    epi_real_seoul.db `PRAGMA quick_check` + 주요 테이블 행수
3. **mcp_10tool_smoke**  `simulation.scripts.smoke_stage6_mcp` → 10 tools, 0 err
4. **static_aggregates** `web/public/aggregates/*.json` 25 gu + 50 edges
5. **turso_seed**        `web/scripts/turso_seed.sql` ≥ 18 MB
6. **node_npm**          node ≥ 18, npm 존재
7. **web_node_modules**  `web/node_modules/` 존재 (없으면 WARN → `npm ci --legacy-peer-deps`)
8. **web_build**         `npm run build` rc=0 (생략 가능)
9. **ollama_installed**  `http://localhost:11434/api/tags` + 원하는 모델 존재
10. **ollama_e2e_smoke** Qwen 이 MCP 10툴 호출 → 한국어로 답변 합성

### 출력
`web/scripts/preflight_local_report.json` — 각 체크 status/detail/elapsed_ms.

### 주요 실패 패턴 & 복구
| 실패 체크            | 복구                                                      |
|:---------------------|:----------------------------------------------------------|
| `db_quick_check`     | `python -m simulation bootstrap` (DB 재빌드)              |
| `mcp_10tool_smoke`   | `python -m simulation.scripts.smoke_stage6_mcp` 직접 실행해 어느 tool 인지 확인 |
| `static_aggregates`  | `python web/scripts/build-static-aggregates.py`           |
| `turso_seed`         | `python web/scripts/export-turso.py --out web/scripts/turso_seed.sql` |
| `web_build`          | 출력 마지막 줄 확인 — 보통 TS 에러 or peer-dep conflict   |
| `ollama_installed`   | `ollama serve` (별도 터미널) + `ollama pull <model>`      |
| `ollama_e2e_smoke`   | `python -m simulation.scripts.smoke_ollama_e2e` 직접 실행해 hop 단위 로그 확인 |

---

## Stage 2 — 최소비용 API 핑

```powershell
# .env 파일 사용 (repo 루트)
.\web\scripts\02_preflight_providers.ps1

# 이미 $env:ANTHROPIC_API_KEY 등 shell 에 세팅된 경우
.\web\scripts\02_preflight_providers.ps1 -NoEnvFile
```

### 총 예상 비용 (max_tokens=8, temperature=0, 각 1 call)
| Provider             | 모델                       | in tok | out tok | 단가 (USD/MTok)   | 비용      |
|:---------------------|:---------------------------|-------:|--------:|:------------------|----------:|
| Anthropic            | claude-3-5-haiku-20241022  | ~10    | ~3      | $0.80 / $4.00     | $0.00001  |
| OpenAI               | gpt-4o-mini                | ~9     | ~3      | $0.15 / $0.60     | $0.00001  |
| Google               | gemini-2.0-flash           | ~2     | ~3      | free tier         | $0.00000  |
| Turso                | `SELECT 1`                 | —      | —       | free tier         | $0.00000  |
| Upstash              | `PING`                     | —      | —       | free tier         | $0.00000  |
| **합계**             |                            |        |         |                   | **< $0.00003** |

### 체크 항목 (6개)
1. **env_keys_present**    8개 env 변수 존재 여부
2. **anthropic_haiku_ping** Claude 3.5 Haiku "say hi"
3. **openai_4omini_ping**   GPT-4o-mini "say hi"
4. **google_flash_ping**    Gemini 2.0 Flash "say hi"
5. **turso_select1**        Turso HTTP pipeline `SELECT 1`
6. **upstash_ping**         Upstash REST `/ping` → `PONG`

### 출력
`web/scripts/preflight_providers_report.json` — 각 provider 의
`in_tokens`/`out_tokens`/`cost_usd` 기록.  총 비용은 `totals.cost_usd`.

---

## Stage 3 — Vercel 프로덕션 배포

```powershell
# DryRun: 실제로 배포 안 하고 plan 만
.\web\scripts\03_deploy_prod.ps1 -DryRun

# 실배포
$env:TURSO_DB_NAME = "frame-d"     # Turso import 에 필요
.\web\scripts\03_deploy_prod.ps1
```

### 옵션
| 플래그          | 효과                                                     |
|:----------------|:---------------------------------------------------------|
| `-DryRun`       | `vercel deploy` / `turso shell` 실행 안 함; plan 만      |
| `-SkipTurso`    | Turso 시드 import 스킵 (이미 완료했거나 production 분리) |
| `-SkipGate`     | **위험**: Stage 1+2 리포트 검사 스킵 (긴급 핫픽스 전용)  |
| `-VercelProject`| Vercel 프로젝트명 override                               |

### 실행 단계 (5개)
1. **gate_stage12**  `preflight_local_report.json` + `preflight_providers_report.json` 둘 다 `n_fail=0` 확인.  하나라도 없거나 fail>0 면 **exit 2**.
2. **vercel_cli**    `vercel --version`
3. **turso_cli**     `turso --version` (있으면)
4. **turso_import**  `turso db shell $TURSO_DB_NAME < web/scripts/turso_seed.sql`
5. **vercel_env_diff** `vercel env ls production` 와 비교해 누락 env 만 `vercel env add`
6. **vercel_deploy**  `vercel deploy --prod` → 반환 URL 추출
7. **prod_smoke**     4개 엔드포인트 HTTP 핑
    - `/`                              (앱 쉘)
    - `/api/mcp/_list`                 (10 tools 노출)
    - `/api/providers`                 (3 provider availability)
    - `/aggregates/latest-choropleth.json` (25-gu 정적 데이터)

### 출력
`web/scripts/deploy_report.json` — 각 스텝 + 프로덕션 URL + 4 smoke 결과.

---

## 전체 파이프라인 실행 (발표 당일)

### 시나리오 A — 발표 1일 전
```powershell
# 1. 모든 체크 fail 0 까지 반복
.\web\scripts\01_preflight_local.ps1
# → preflight_local_report.json 확인

# 2. API 키 .env 확인 후 핑
.\web\scripts\02_preflight_providers.ps1
# → preflight_providers_report.json cost < $0.001 확인

# 3. 배포 계획만 미리 확인
.\web\scripts\03_deploy_prod.ps1 -DryRun
```

### 시나리오 B — 발표 당일 아침
```powershell
# 빠른 재확인 (빌드 스킵)
.\web\scripts\01_preflight_local.ps1 -SkipBuild

# 실배포
$env:TURSO_DB_NAME = "frame-d"
.\web\scripts\03_deploy_prod.ps1
# → deploy_report.json 의 prod_url 확인 → QR 생성
```

### 시나리오 C — 배포 중 문제 발생
```powershell
# 로컬 fallback: docker compose 로 localhost:3000 돌려 발표자 노트북 hotspot 공유
docker compose -f web/docker-compose.yml up --build
# 참석자는 http://<presenter-ip>:3000 접속
```

---

## 로컬 모델별 특성 (Ollama E2E 통과 확인 테이블)

`01_preflight_local.ps1 -OllamaModel <tag>` 로 각각 검증할 것:

| 모델 태그                            | RAM   | 강점                           | tool-use 안정성 |
|:-------------------------------------|:------|:-------------------------------|:----------------|
| `qwen2.5:14b-instruct-q5_K_M`  ★     | 12 GB | 한국어 + tool-call 균형        | ★★★★★          |
| `qwen2.5:32b-instruct-q4_K_M`        | 24 GB | 최상위 품질 (GPU 여유 시)      | ★★★★★          |
| `exaone3.5:7.8b`                     | 6 GB  | 한국어 정책 문장 자연스러움    | ★★★☆☆          |
| `biomistral:7b`                      | 6 GB  | 의료/clinical 해석 (PubMed FT) | ★★☆☆☆          |

★ = 기본 추천 (발표 노트북 기준 16GB RAM 이면 충분)

**추천 운영**: `qwen2.5:14b` 를 기본으로 두고, "이번 주 Rt 해석"
같은 임상 질문에는 ProviderPicker 에서 `biomistral:7b` 로 수동 스위치.
Claude/Gemini 끊겼을 때 즉시 로컬 fallback 가능.

---

## FAQ

**Q. Stage 2 가 돈이 정말 $0.00003 인가?**
A. Claude Haiku 3.5 의 입력 토큰 10개 × $0.80/1M = $0.000008, 출력
3개 × $4.00/1M = $0.000012 → Claude 만 $0.00002.  OpenAI 비슷,
Gemini/Turso/Upstash 는 free tier.  "say hi" 를 "서울 ILI 요약" 같은
긴 프롬프트로 바꾸지 마라.

**Q. Stage 3 에서 Turso import 이 2 번째 실행 시 어떻게 되나?**
A. `turso db shell` 은 `INSERT OR REPLACE` / `CREATE TABLE IF NOT EXISTS`
를 존중하지만, `turso_seed.sql` 자체가 idempotent 하게 생성돼야 함.
이미 돌아간 DB 에 두 번 import 하면 PRIMARY KEY 충돌 가능 → 이
경우 `-SkipTurso` 로 우회하고 수동 `turso db destroy && turso db create`.

**Q. Stage 1 의 Ollama E2E 가 fail 했는데 뭐 해야 하나?**
A. `python -m simulation.scripts.smoke_ollama_e2e --model <tag>` 직접 실행.
각 hop 의 tool_call 시도가 로그에 찍힘.  흔한 원인:
  1. 모델이 tool-use 를 안 지원 (예: `llama2` 같은 구버전) → 지원 모델로 교체
  2. 모델이 한국어 prompt 를 영어 답변으로 내놓음 → SYSTEM_PROMPT 튜닝
  3. MCP 툴 이름 mangling 실패 (`epi.forecast` ↔ `epi_forecast`) — bridge 로직 버그

**Q. `-SkipGate` 는 언제 쓰나?**
A. 발표 15분 전, Turso 만 변경했고 전체 검증 다시 돌리기엔 시간이
없을 때.  그 외에는 절대 쓰지 마라.
