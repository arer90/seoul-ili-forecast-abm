export const meta = {
  name: 'baseline-negativity-diagnosis',
  description: 'Diagnose root cause + concrete fix for each baseline-negative active model (loop-engineering)',
  phases: [
    { title: 'Diagnose', detail: 'one agent per negative model: identity -> root cause -> concrete fix' },
    { title: 'Verify', detail: 'adversarially verify each diagnosis + whether the fix truly makes baseline positive' },
    { title: 'Synthesize', detail: 'group by root cause, order fixes, flag structural removals' },
  ],
}

let NEG = args
if (typeof NEG === 'string') { try { NEG = JSON.parse(NEG) } catch (e) { NEG = [] } }
NEG = Array.isArray(NEG) ? NEG : (NEG && Array.isArray(NEG.models) ? NEG.models : [])
if (!NEG.length) { log(`no negative active models — args type=${typeof args}`); return { diagnoses: [], synthesis: null, debug_args_type: typeof args } }
log(`${NEG.length} baseline-negative active models: ${NEG.map(m => m.model).join(', ')}`)

const DIAG_SCHEMA = {
  type: 'object',
  required: ['model', 'identity', 'root_cause', 'fix_type', 'fix_detail', 'file_refs', 'expected_after', 'confidence'],
  properties: {
    model: { type: 'string' },
    identity: { type: 'string', description: 'class name + file path. NOTE pf-wrapper vs custom net.' },
    root_cause: { type: 'string', description: 'WHY baseline R2 is negative, with code evidence' },
    fix_type: { type: 'string', enum: ['rolling_1step', 'transform_cap', 'input_bug', 'meta_identity', 'structural_remove', 'other'] },
    fix_detail: { type: 'string', description: 'concrete change: file:line + what to change, OR removal rationale' },
    file_refs: { type: 'array', items: { type: 'string' } },
    expected_after: { type: 'string', description: 'expected baseline R2 sign after fix' },
    confidence: { type: 'string', enum: ['high', 'medium', 'low'] },
  },
}

const VERIFY_SCHEMA = {
  type: 'object',
  required: ['agree', 'fix_sound', 'corrected', 'note'],
  properties: {
    agree: { type: 'boolean', description: 'is the root_cause correct per the code?' },
    fix_sound: { type: 'boolean', description: 'would the proposed fix actually make baseline R2 >= 0?' },
    corrected: { type: 'string', description: 'corrected root cause / fix if disagree, else empty' },
    note: { type: 'string' },
  },
}

function diagPrompt(m) {
  return `MPH 서울 ILI 주간예측 파이프라인. 모델 "${m.model}" 이 baseline(BASIC 13 feature = lag+계절성, raw x/y, transform 없음)에서 R²=${m.r2 ?? m.error}(음수/에러) 를 냈다. 근본 원인과 baseline 을 양수(약해도 OK, ≥0)로 만드는 구체 수정안을 진단하라. 읽기 전용 — 코드 수정 금지, 진단만.

절차:
1. 정체성: \`grep -rn "${m.model}" simulation/models/registry.py\` + REGISTRY 로 클래스+파일 확인.
   ⚠ N-BEATS/N-HiTS/TiDE 의 LIVE 구현 = simulation/models/pf_models.py 의 Pf*Forecaster wrapper(pytorch-forecasting), modern_ts/ 의 custom net 이 아님 — 반드시 실제 클래스 확인.
2. 그 클래스의 fit/fit_series/predict/forecast 코드를 Read.
3. 근본 원인 판정 — 후보(코드 증거 명시):
   (a) single-origin: predict=forecast(len(X_test))=한 출발점 68주 외삽→mean-revert→음수. TimeSeriesForecaster 계열(DLinear/TimesFM/TiRex 등). 고침=rolling 1-step (G-321 = base.py supports_rolling_eval + rolling_1step, ts_models append override 참고). 단 META(identity)가 아니면 transform 공간 주의(y_observed 를 학습공간으로).
   (b) static/sliding collapse: pf wrapper 의 정적 68주 predict 가 zero/평탄 collapse(N-BEATS/N-HiTS/TiDE).
   (c) inverse-transform 폭발: log/boxcox 역변환 발산 → cap(G-319b: clip in log-space BEFORE expm1).
   (d) input bug: feat_proj(398→1)가 과거 y 뭉갬 → lag-backbone(G-319d, PatchTST/iTransformer/Mamba 는 이미 적용; 미적용 모델 확인).
   (e) count/renewal META 누락: epi 모델이 y-transform 받아 곱셈/round 깨짐 → META_MODELS 등록(identity×none).
   (f) structural: BASIC feature 로 68주 hold-out 진짜 불가 → 라인업 제거 권고.
4. baseline 을 양수로 만드는 구체 수정(file:line + 변경) 제시, 또는 structural 이면 제거 권고 + 근거.

참고: docs/PERMODEL_FIX_LESSONS_20260619.md, simulation/models/base.py(supports_rolling_eval/rolling_1step), ENGINEERING_PRINCIPLES.md.`
}

const results = await pipeline(
  NEG,
  m => agent(diagPrompt(m), { label: `diag:${m.model}`, phase: 'Diagnose', schema: DIAG_SCHEMA, agentType: 'general-purpose' }),
  (diag, m) => {
    if (!diag) return null
    return agent(
      `적대적 검증(읽기 전용, 코드 재확인). 아래 baseline-음수 진단을 검증하라:\n` +
      `① root_cause 가 코드와 일치하나? ② 제안 fix 가 정말 baseline R²≥0 으로 만드나(과장/논리오류 지적)? ` +
      `③ rolling_1step 제안이면 transform 공간(META=raw, 아니면 transformed y_observed)이 맞나? ` +
      `④ structural_remove 제안이면 진짜 구조한계인가(고칠 여지 없나)?\n모델=${m.model}\n진단=${JSON.stringify(diag)}`,
      { label: `verify:${m.model}`, phase: 'Verify', schema: VERIFY_SCHEMA, agentType: 'general-purpose' }
    ).then(v => ({ ...diag, verdict: v }))
  }
)

const clean = results.filter(Boolean)

const SYNTH_SCHEMA = {
  type: 'object',
  required: ['groups', 'fix_order', 'remove_candidates', 'summary'],
  properties: {
    groups: { type: 'array', items: { type: 'object', properties: { root_cause: { type: 'string' }, models: { type: 'array', items: { type: 'string' } }, shared_fix: { type: 'string' } } } },
    fix_order: { type: 'array', items: { type: 'string' }, description: 'ordered model names (group shared-file fixes together)' },
    remove_candidates: { type: 'array', items: { type: 'string' } },
    summary: { type: 'string' },
  },
}

const synth = await agent(
  `다음은 baseline-음수 active 모델들의 진단+적대적검증 결과다. 종합하라:\n` +
  `1. 근본원인별 그룹(같은 원인+같은 fix 패턴 묶기, 공유 파일 식별).\n` +
  `2. 수정 순서(공유 파일끼리 묶어 충돌 방지; verify 가 fix_sound=false 인 건 재검토 표시).\n` +
  `3. structural_remove 후보(verify 통과한 것만) 명시.\n` +
  `4. 한 문단 요약 + 전체 위험도.\n진단들:\n${JSON.stringify(clean, null, 1)}`,
  { phase: 'Synthesize', schema: SYNTH_SCHEMA }
)

return { n_negative: NEG.length, diagnoses: clean, synthesis: synth }
