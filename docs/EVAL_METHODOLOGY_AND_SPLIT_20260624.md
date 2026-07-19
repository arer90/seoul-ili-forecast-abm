# MPH ILI 예측 — 평가 방법론 & train/val/test 레퍼런스

> **날짜**: 2026-06-24 · **commit**: `7a4d4af` · **코드 검증 스냅샷 (code-verified snapshot)**
> 이 문서의 모든 주장은 `file:line` 으로 추적 가능하다. 본문 수치/주장은 코드에서 검증 추출한 것이며, 코드에 없는 값은 지어내지 않았다.
> **목적**: 타 AI/리뷰어가 본 파이프라인의 평가 방법론을 **비교·검증**할 수 있는 단일 진실 원천(SSOT) 레퍼런스.

---

## 목차 (Table of Contents)

- [§0 한눈 요약 (TL;DR)](#0-한눈-요약-tldr)
- [§1 데이터 분할 & 교차검증 구조](#1-데이터-분할--교차검증-구조)
- [§2 모든 단계가 동적인가? (data-driven)](#2-모든-단계가-동적인가-data-driven)
- [§3 모델별 train/val/test + rolling 1-step](#3-모델별-trainvaltest--rolling-1-step)
- [§4 챔피언 선정 & 평가 지표](#4-챔피언-선정--평가-지표)
- [§5 방법론 출처 (인용)](#5-방법론-출처-인용)
- [§6 리뷰어 검증 체크리스트](#6-리뷰어-검증-체크리스트)
- [§7 eval-fairness 캠페인 타임라인](#7-eval-fairness-캠페인-타임라인)
- [부록 A. 알려진 한계 / 주의 (정직)](#부록-a-알려진-한계--주의-정직)

---

## §0 한눈 요약 (TL;DR)

1. **1-step rolling-origin 평가**: 모든 sequence 모델(classic-ts/epi/foundation/pf)은 각 test 주를 **관측 과거로만 1-step** 예측한다(배포 충실, Tashman 2000). 단일원점 68주 외삽이 아니다 (`base.py:355-385`).
2. **Leak-free**: 주 `i` 예측에는 `y_observed[:i]` 만 쓴다 — 미래값 미열람 (`base.py:380`, `fused_epi.py:267`, `simulation/models/modern_ts/pf_models.py:282`). WF-CV fold 의 global-summary feature 도 `[:end_tr]` 로 재코딩(G-309).
3. **선택과 보고의 물리 분리 (2-층위, G-339 leak-free)**: config(preproc/mc/feature/HP) **선정 = WF-CV(OOF, 5-fold, 242)** 로 test 미접촉(Layer 1, leak-free); **cross-model 챔피언도 G-339 로 OOF 1-SE band 안 fold안정성·parsimony(test 미사용)** 선정(Layer 2). hold-out test(68)는 **최종 1회 보고만** → winner's curse 0 (`per_model_eval.py:select_champion_g318` G-339, §1.3). ⚠ 옛 G-318(shortlist 내 test-best)은 부분 winner's curse라 폐기(외부 reviewer #1).
4. **전 단계 동적(data-driven)**: preproc·mc·feature·HP·transform·rolling-eval 6단계 모두 데이터(OOF-WIS / per-origin)가 per-model 로 결정한다. 사람이 박은 것은 *결정값*이 아니라 *탐색 공간·정직성 규율*이다.
5. **Uniform fit-once B-protocol**: 패널 전체가 train 에서 1회 추정 → 매 origin 관측값 feed(재추정 X). Theta 가 마지막 A-style 잔존이었으나 G-338 로 B 전환되어 eval 패널이 균일해졌다 (`base.py:355-364`, `ts_models.py:413-492`).

---

## §1 데이터 분할 & 교차검증 구조

### 1.1 split 크기 — 현재 코드의 진실 = **n=337 → 269(train+val) | 68(test) | + real**

메모리에 혼재하던 두 수치(`337 = 242|27|68` vs `pool 269|68`)는 **같은 split 의 다른 절단면**이며 둘 다 진실이다. in-sample n=337 을 4-way 로 자른다:

| slab | 크기 | 인덱스 구간 | 산출 코드 | 용도 |
|---|---|---|---|---|
| **in-sample (n)** | **337** | `[:337]` | `paper_cutoff_week=337` → `real_start=337`, `X_all=X_all[:real_start]` (`data.py:252-274`) | train+val+test |
| **train** | **242** | `[0:242]` | `n_train = pool - n_val` (`data.py:43`); 소비 `X_train=X_all[:n_train]` (`per_model_optimize.py:3817`) | fit + config/HP/feature 선택의 fold 풀 |
| **val** | **27** | `[242:269]` | `n_val = round(pool*0.10)` (`data.py:42`); `X_val=X_all[242:269]` (`per_model_optimize.py:3819`) | 단일 train→val WIS (`best['wis']`) 보고용; ⚠ champion 선정엔 미사용 |
| **train_pool (train+val)** | **269** | `[0:269]` | `pool = n - n_test = 337-68` (`data.py:41`); `X_pool=vstack([X_train,X_val])` (`per_model_optimize.py:2517`) | test refit 학습 풀 |
| **test (hold-out)** | **68** | `[269:337]` | `n_test = ceil(337*0.20)` (`data.py:40`); `X_test=X_all[269:337]` (`per_model_optimize.py:3822`) | **평가·champion 보고 전용** (선정 미접촉) |
| **real** | 가변(≈8~15) | `[337:]` | `real_X=X_all[real_start:]` (`data.py:263`) | P1 forecasting target — 학습/WF-CV/test **절대 금지** (`data.py:258`) |

**분할 방식 = 순수 시간순(temporal/chronological)**. 비율 파라미터: `paper_cutoff_week=337`, `in_sample_test_ratio=0.20`, `in_sample_val_ratio=0.10` (`pipeline/config.py:87-89`). 계산 SSOT = `compute_split_indices(n, config)` (`data.py:20-49`):

```
n_test  = ceil(n * 0.20)        # 337 → 68
pool    = n - n_test            # 337 - 68 = 269
n_val   = round(pool * 0.10)    # round(26.9) = 27
n_train = pool - n_val          # 269 - 27 = 242
```

(HWP §3 주석 `data.py:38-40`, `pipeline/config.py:61-65` 와 일치.) `config_global.py:530-535` 의 하드코드 `n_train=242/val_size=27/test_size=68` 은 같은 결과의 **정적 참조값**이며, **실제 split 은 ratio 에서 동적 계산**된다.

> ⚠ **보조 사실**: `config_global.DataSplitConfig.conformal_holdout_weeks=26` (`config_global.py:537`) 이지만 **런타임 split 은 `pipeline/config.py:138` 의 `conformal_holdout_weeks=0`** 를 쓴다 → 현재 conformal holdout slab 은 비활성(`holdout_start=n`, `data.py:318-319`). WF-CV 의 `holdout_start` 가드(`wfcv.py:29`)는 코드상 존재하나 default 로 trigger 안 됨.

### 1.2 WF-CV / OOF fold 구조 = **5-fold expanding-window walk-forward**

champion·HP·feature 선정에 쓰이는 OOF 의 canonical 구현은 두 함수 — flat `_oof_cv_wis` (`per_model_optimize.py:1859`) 와 hierarchical-preproc replay `_oof_cv_wis_hier` (`_inline_optuna_3stage.py:411`) — 이며 **fold 생성 로직이 동일**하다:

```
n = len(X_train)                         # = 242 (train pool; val/test/real 미포함)
fold_size = n // (n_folds + 1)           # n_folds=5 → fold_size = 242//6 = 40
for k in 1..n_folds:
    end_tr = k * fold_size               # 40, 80, 120, 160, 200
    end_va = (k+1)*fold_size (k<5) else n # 80, 120, 160, 200, 242
    fit on [:end_tr], validate on [end_tr:end_va]   # expanding train, sliding val
```

(`per_model_optimize.py:1894-1908`, `_inline_optuna_3stage.py:461-483`.) **expanding window** — train 은 항상 `[:end_tr]` (origin 0 고정, 누적 확장), val 은 `[end_tr:end_va]` (비중첩 전진). **'OOF'(out-of-fold) 의 정확한 의미** = 각 fold 의 val 구간은 그 fold 의 train 에 미포함 → 미래 누수 0. fold 별 WIS 를 regime-conditional 집계(`_oof_regime_aggregate`/`_aggregate_oof_folds`, outbreak fold 가 median 에 묻히지 않도록 G-255/256b) + fold-variance penalize 한 스칼라가 `oof_wis` 다 (`per_model_optimize.py:1930-1932`).

- **fold 수**: `n_folds=5` (`GLOBAL.training.oof_folds`, default 5, `config_global.py:136-140`; `_inline_optuna_3stage.py:819` 가 best_by 별로 5 또는 oof_folds 적용). `research_5fold` 모드는 항상 5.
- **train_pool(269) 이 selection 에 쓰이는가? → 아니다.** OOF 선정은 `X_train`/`y_train` = **242 (val 제외)** 만 fold 풀로 사용(`per_model_optimize.py:1890` `n=len(X_train)`, caller 가 `X_train=X_all[:n_train]` 전달). 269(train+val)은 **선정이 끝난 뒤 test refit 학습**에만 쓰인다(`per_model_optimize.py:2509-2518`). 따라서 val(27) 도 hold-out test(68) 도 OOF fold 에 절대 안 들어간다.
- **leakage 가드**: fold 마다 global-summary feature(quantile bin / above_threshold / interaction max-norm)를 `[:end_tr]` 로 재코딩(`_recode_advanced_per_fold`, `per_model_optimize.py:1904` / `wfcv.py:_recode_*_per_fold:66-254`) → build-time GLOBAL(test+real era) 통계 누수 차단(G-309). inverse-cap 기준도 fold-불변 train max(G-334, `_inline_optuna_3stage.py:466`).
- **R4 (wfcv)** (`wfcv.py:run_wfcv`)는 step_size=1 의 별도 1-주-step walk-forward(`pipeline/config.py:221-222`, `wfcv.py:22-37`)이나 이는 **진단/8-모델 baseline factory 용**이며, champion 선정 OOF 는 위 **R9 의 5-fold** 가 SSOT.

### 1.3 selection-vs-reporting 분리 — 2-층위 (G-339, 2026-06-24 외부 reviewer 정정)

> ⚠ **정정 (외부 AI reviewer #1)**: 이 문서 초판이 "champion 선정·보고 분리 = winner's curse 없음"이라 한 건 **2층위를 뭉갠 과장**이었다. 정확히는:
> - **Layer 1 (within-model config: preproc/mc/feature/HP/transform)** = WF-CV(OOF, 242) 만으로 선택, test 미접촉 → **genuinely leak-free** ✓ (불변).
> - **Layer 2 (cross-model champion)** = 옛 **G-318 이 OOF top-8 shortlist 안에서 hold-out test WIS argmin** 으로 골랐다 → K=8 로 줄였을 뿐 **test 로 1-of-8 선택 = winner's curse 재유입** (reviewer #1, Cawley & Talbot 2010 JMLR; Varma & Simon 2006 BMC Bioinformatics 7:91). **이건 leak-free 가 아니었다.**

**수정 = G-339 (LEAK-FREE 챔피언, test 선정 완전 제거):**

- **Layer 1 (불변, leak-free)**: 반환 dict 의 **선정 키 `oof_wis`(5-fold WF-CV) 와 보고 키 `test_metrics`(hold-out 68) 물리 분리**(`per_model_optimize.py:2914,2917`). config 전부 `MPH_BEST_BY=oof_cv` 로 OOF keying(`_oof_cv_wis:1859` test 미접촉).
- **Layer 2 (G-339 신규, leak-free)**: `select_champion_g318`(`per_model_eval.py:115` — 함수명 SSOT 유지, body=G-339) 가 ① **OOF 1-SE 통계동률 band**(Breiman 1984: best 의 per-fold OOF 로 SE, `oof_wis ≤ best+max(SE,2%)`) → ② band 안 **leak-free tiebreaker**: fold 안정성(`_oof_fold_cv`=분포이동 견고성 proxy, G-318 의 'hold-out 일반화'를 test 없이 대체) → parsimony(`n_features`) → OOF-WIS. **hold-out test 미사용.** G-307 SVR-RBF OOF-노이즈-우승도 band+안정성 흡수.
- **hold-out test = 진단 병기만**: `select_champion_holdout_best`(`per_model_eval.py:194`, test-best)는 배포 아닌 투명성 진단(G-339 와 같으면 강한 증거). **test 는 최종 보고에서 1회만 접촉 → winner's curse 0.**
- **가드 test**: `tests/test_g339_champion_leakfree.py` — `test_leak_free_test_wis_does_not_change_champion`(test 교란 불변), `test_champion_is_stable_not_oof_noise_winner`(SVR-노이즈-우승 거부), `test_parsimony_breaks_stability_ties`, `test_holdout_best_is_separate_diagnostic`(7 green).
- post-hoc 도구 = `scripts/rerank_champion.py`(재학습 0; test WIS·DM 은 진단 표시, 선정 미참여).

### 1.4 val(27) 의 용도

- **champion/HP/feature 선정엔 미사용** — G-132("n=27 single-val 거절"). `MPH_BEST_BY=oof_cv` (학습 운영 표준 export, ENGINEERING_PRINCIPLES.md §학습 운영) 가 선정 기준을 OOF(5-fold, 242 누적)로 강제. 함수 내부 docstring default 는 `val_wis`(`per_model_optimize.py:2056`)지만 **라이브 run 은 env 로 oof_cv override** → val noise overfit 방지(`per_model_optimize.py:2055-2056`, `_inline_optuna_3stage.py:817-819`).
- **실제 쓰임**:
  1. 단일 train→val WIS = `best['wis']` (보고/진단용 스칼라, 선정엔 안 들어감, `per_model_optimize.py:2909, 2912`);
  2. DL early-stopping 용 forward-holdout (HWP §3 "val = DL early stopping", `pipeline/config.py:65`);
  3. test refit 시 train 에 합쳐져 269 pool 의 일부가 됨 (`per_model_optimize.py:2517-2518`).
- do-no-harm floor 도 단일-27-val 게이트를 5-fold OOF 로 교체(G-12V, `per_model_optimize.py:2547-2549`).

---

## §2 모든 단계가 동적인가? (data-driven)

사용자 질문 "다 동적이지?" 에 대한 코드 검증 결과:

> **선택(selection)은 6단계 전부 데이터(OOF-WIS)가 정한다. 단, 각 단계의 _후보 풀·탐색 범위·규율 파라미터_ 는 사람이 하드코딩한 고정 메뉴이고, 데이터는 그 메뉴 안에서 per-model 로 최적을 고른다.**

stale 노트("preproc 100 Optuna trial", "anchor blend per-model α") 2건은 코드와 불일치 — 아래에 정정한다.

| 단계 | 동적? | 선택 기준 (무엇을 무슨 기준으로) | 코드 위치 | 하드코딩 fallback |
|------|-------|-------------------------------|-----------|------------------|
| **1. preproc/transform** | ✅ per-model 동적 (**G-335 flat-grid 확인, "100 Optuna trial" 노트는 stale**) | 고정 7-transform 메뉴(`identity` + log1p/sqrt/fourth_root/asinh/laplace/mcmc_robust) 각 **1회** OOF-CV-WIS 평가 → **fold-paired 1-SE** 로 best 선택(비-identity 가 identity 를 1 fold-SE 이상 유의하게 이길 때만 채택, 아니면 identity 유지). **Optuna study/sampler/pruner 없는 순수 loop** | `_inline_optuna_3stage.py:99` (STABLE_Y 6종), `:858-909` (pure-grid loop, "no Optuna"), `:604-617` (1-SE), `:697-700` (grid=모든 transform seed); 후보 메뉴 `preproc_optuna_hierarchical.py:99` | 메뉴 자체는 고정. grid 실패 시 `identity/none` (`:907-908`). `MPH_PREPROC_GRID=0` → legacy TPE 복귀 |
| **2. mc (multicollinearity)** | ✅ per-model 동적 (G-242) | {none/vif/corr/pca} 4-method 각각 per-fold OOF-WIS 측정 → 모델별 최소 method. **do-no-harm margin**: `none` 대비 상대 개선 ≥ `MPH_MC_MARGIN`(0.02) 일 때만 이탈, 애매하면 `none`(overfit 가드) | 선택 `_per_model_mc_choice` `per_model_optimize.py:3682-3718`; 비교 `_compare_mc_per_model:3383`; 적용 loop `:4069-4088` | `none` (측정 부재/guard/실패 시). `MPH_MC_PER_MODEL=0` → legacy global |
| **3. feature 선택** | ✅ per-model 동적 (STABILITY + nested 1-SE) | Meinshausen-Bühlmann 재표본: B subsample × 점수 상위 inner_k → **선택 빈도 ≥ π feature** 만 keep(출력 크기=데이터 창발). nested size-path(π=0.8/0.6/0.4+full) 에서 per-model **1-SE/parsimony** OOF 선택. deep-NN(`category=='dl'`)은 작은-fold OOF 불신 → **binary** 자동 전환 | `per_model_optimize.py:2278-2417`; `feature_select_corr1se.py:42` (stability), `:154` (nested), `:192` (1-SE), `:138` (data-derived floor) | full-feature 복원(subset 명백 열위 시, `:2412-2414`); guard 실패→subset(`:2415-2417`). AR-lag mandatory-include(음수 R² 가드, `:2291-2296`) |
| **4. HP (hyperparameter)** | ✅ per-model 동적 (Optuna TPE) | 각 모델 `fit()` 내부 Optuna TPE study 가 WIS 최소화로 HP 선택. trial 수 = per-model JSON 예산(env `MPH_OPTUNA_TRIALS_JSON`/`MPH_HP_OPTUNA_TRIALS`, 기본 20) | budget `_optuna_budget.py:40-65`; 탐색 `tree_models.py:101-151` 등 모델별 `suggest_*`; sampler `_optuna_samplers.py` | **탐색 _범위_ 는 하드코딩**(예: `max_depth 3-8`, `learning_rate 0.01-0.1`); 데이터는 그 범위 내에서만 탐색 |
| **5a. transform per-model** | ✅ 동적 | 단계 1 의 산물 — best transform 이 모델마다 다르게 선택됨 (count-GLM=force-identity, foundation=x-identity 등 모델 속성 기반 제약 포함) | `_inline_optuna_3stage.py:765-783` (per-model 제약), `:900` (per-model best) | 모델 속성으로 강제 identity(GLM log-link/pf 등) |
| **5b. anchor/blend α per-model** | ⚠️ **DEAD — stale 노트** | ENGINEERING_PRINCIPLES.md "anchor blend per-model α(G-141/144)" 는 **현 코드에 없음**. G-231(2026-05-22)이 α-blend 를 전 DNN 에서 제거; `suggest_*_hp` 의 "Package K alpha_blend" 는 **주석만 남은 댕글링 마커**(실제 `suggest_float("alpha_blend")` 호출 0) | 제거 기록 `_optuna_torch.py:170-172`("alpha_blend HP 제거"), `_optuna_samplers.py:294`(주석만), `runner.py:978`/`registry.py:631`(dl_anchored 폐기) | α-blend 비활성 = 항상 미적용 |
| **6. rolling eval observed-feed** | ✅ per-origin 동적 (G-321/327/337) | sequence 모델(classic-ts/epi/foundation/pf)은 각 test 주 i 를 **관측 과거 `y_observed[:i]` 로만 1-step** 예측(leak-free loop). feature 모델은 lag 로 이미 1-step. transform-space 모델(N-BEATS/TiDE)은 train+test 동시 affine transform 후 슬라이스 | `base.py:355-385` (`rolling_1step` per-origin loop), `:351-352` (predict 분기); eval 배선 `per_model_optimize.py:1374-1395`; 대상 집합 `base.py:404` (`ROLLING_EVAL_MODELS`) | static `forecast(steps=N)` (series 미저장/transform 실패 시, `base.py:376`, `:1393`) |

### 다 동적? — 평결: **YES (선택 결정은 6단계 전부 데이터가 정한다)**

하드코딩된 것은 *결정값* 이 아니라 *결정의 틀* 이다:

- **데이터가 정하는 것 (per-model, OOF-WIS argmin/1-SE)**: 어느 transform·어느 mc method·몇 개 feature·어느 HP 조합·어느 origin 의 observed-feed. 전부 모델마다 다르게 나오고, 노이즈급 우세는 단순한 baseline(identity/none)으로 되돌리는 **do-no-harm / 1-SE 가드**가 일관되게 걸려 있다(preproc 1-SE, mc 0.02 margin, feature 1-SE/parsimony — 동일 Occam 철학).
- **사람이 하드코딩한 것 (고정 메뉴·규율 파라미터)**: 7-transform 풀, {none/vif/corr/pca} 4-method, HP suggest 범위, π=0.8/0.6/0.4 ladder, epv_ratio=20, margin 임계값. 이건 "사람이 답을 박았다"가 아니라 "탐색 공간·정직성 규율을 박았다"이며, 데이터가 그 안에서 자유롭게 고른다.
- **항상 하드코딩 fallback 존재**: 각 단계 실패 시 가장 단순한 안전값(identity / none / subset / static)으로 graceful degrade — 이건 동적성을 깨는 게 아니라 fail-safe.

**stale 노트 2건 정정(앵무새 거부)**:

1. ENGINEERING_PRINCIPLES.md/task 의 "preproc 100 Optuna trial" → **틀림**. 현 코드는 G-335 flat-grid: Optuna 제거, 7-transform 각 1회 OOF + 1-SE. `MPH_PREPROC_GRID=1`(기본). → **G-335 가 맞다.**
2. ENGINEERING_PRINCIPLES.md 의 "anchor/blend α per-model HP(G-141/144)" → **현 코드에 dead**. G-231 로 제거됨, 주석만 잔존.

> **caveats (이 섹션)**: HP suggest 범위(max_depth/learning_rate 등)와 규율 파라미터(π ladder 0.8/0.6/0.4, epv_ratio=20, 각 margin 임계)는 하드코딩이지만, 이는 '결정값'이 아니라 '탐색공간/정직성 규율'이고 그 안에서 데이터가 per-model 선택함. runner 가 launch 시 `set_budget` 으로 tree/DL 예산을 넣는 경로(`runner.py`)는 직접 확인하지 않음 — `_optuna_budget.get_trials` 의 기본값은 코드상 20이며 task 의 'tree 50/DL 40'은 launch-time 주입값으로 추정(미검증). 검증은 import-경로의 라이브 코드 기준이며 실제 학습 run 로그는 미확인. mc 비교는 n_folds=2(cheap pre-pass), preproc/feature OOF 는 `GLOBAL.training.oof_folds`(기본 5).

---

## §3 모델별 train/val/test + rolling 1-step

### 3.1 공통 골격 — 세 평가 표면

모든 모델은 동일한 추상 인터페이스(`BaseForecaster.fit`/`predict`)를 공유하되, 평가 시점에 따라 세 개의 surface 를 지난다.

| 평가 표면 | 위치 | feature pool | rolling 적용 |
|---|---|---|---|
| **baseline (R2)** | `runner.py` `run_baseline` | BASIC(lag+계절성 13) | classic-ts + epi + foundation/pf 모두 rolling 1-step (`_sre`/`_sbr` 게이트, `runner.py:1407`) |
| **선택 OOF (R9)** | `per_model_optimize.py` `_evaluate_config` / `_evaluate_config_hierarchical` | full pool | sequence 모델 rolling 1-step (`_rolling_or_static_predict_oof`, `:468`, `:774`) |
| **최종 test (R10)** | `per_model_optimize.py` `_refit_and_predict_test` | 선택된 config 그대로 | sequence 모델 rolling 1-step (`:1374`–`:1393`) |

핵심 변화는 **선택 OOF가 이제 최종 test와 글자 그대로 동일한 1-step 잣대**라는 점이다. 두 OOF 진입점(flat `_evaluate_config:468`, hierarchical `_evaluate_config_hierarchical:774`)이 동일 헬퍼 `_rolling_or_static_predict_oof`(`per_model_optimize.py:326`)를 호출하고, 그 헬퍼는 `_refit_and_predict_test`(`:1374`)와 같은 게이트(`supports_rolling_eval ∪ supports_transform_rolling`)·같은 `predict(..., y_observed=...)` 경로를 탄다(G-337b). 따라서 preproc·mc·feature·HP 선택이 static multi-step collapse 기준이 아니라 배포-충실 1-step 기준으로 이뤄진다(`per_model_optimize.py:327-328`, `:484` "selection==eval").

### 3.2 B-protocol (fit-once + 관측-feed) — 패널 전체 통일

**B-protocol** = train 에서 파라미터를 **1회만** 추정하고, 매 origin 마다 **관측값**(hold-out `y_observed`)을 고정-파라미터 모델에 흘려 1-step 예측하는 방식(재추정 X, Tashman 2000 = eval이 실제 배포 조건을 복제). `rolling_1step` docstring(`base.py:355-364`)이 이를 명문화한다: "운영 rolling 모델은 전부 fit-once + 관측-feed (B)로 통일… per-origin 파라미터 재추정 (A) 모델 0".

leak-free: i 예측에는 `y_observed[:i]`만 사용한다(`base.py:380`, fused `fused_epi.py:267`, pf `simulation/models/modern_ts/pf_models.py:282`).

G-338 (symmetric-refit, §7) 이전에는 **Theta만 매 origin 재적합(A-style)** 하는 마지막 잔존이었다. 이제 Theta도 `fit_series`에서 α/seasonal/trend 를 1회 추정하고(`ts_models.py:413-444`) rolling 시 고정-α SES level 재귀 + 고정 seasonal + 고정 drift 로 관측값만 흘린다(`ts_models.py:457-492`). 이로써 ARIMA/SARIMA `append(refit=False)` + epi 관측-feed 와 대칭이 되어 **eval 패널 전체가 uniform B**가 되었다(`ts_models.py:460-465`). 검증: legacy per-origin refit 과 max|Δ|≈1.1, R² 0.938 vs 0.934(`ts_models.py:418`, `:467`).

### 3.3 카테고리별 fit / rolling 방식 / eval 표면

| 카테고리 | 대표 모델 | fit(train) | rolling 1-step 방식 (B-protocol) | rolling eval 표면 |
|---|---|---|---|---|
| **feature/tabular** | XGBoost, RF, NegBinGLM, TabPFN, SVR | full-pool fit-once batch | rolling **불필요** — lag feature 로 `predict(X_test)`가 이미 1-step. `y_observed` 미사용 | (해당 없음; static `predict(X)`) |
| **classic-ts** | ARIMA, SARIMA, SARIMAX, Theta, FluSight-Baseline | `fit_series`(y만) 1회; AIC grid | ARIMA/SARIMA = statsmodels `append(refit=False)`(filter only, refit X) `ts_models.py:147`,`:328`; SARIMAX = exog 포함 `append(refit=False)` `:251`; Theta = 고정 α/seasonal/trend override `:457` (G-338) | baseline + OOF + test (`ROLLING_EVAL_MODELS` `base.py:404`) |
| **epi/count** | PoissonAutoreg, hhh4-equivalent, EpiEstim, Wallinga-Teunis, GLARMA | `fit`(GLM/NegBin·AR-lag) 1회 | 관측 lag/잔차 feed (G-327): PoissonAutoreg AR-lag=관측값 `epi_models.py:900`; hhh4 AR(1) 입력=관측 `hhh4_models.py:169`; EpiEstim/Wallinga history append=관측 `epiestim_models.py:138`/`wallinga_teunis.py:102`; GLARMA pearson 잔차 매주 관측 재계산 `glarma_models.py:140-161` | baseline + OOF + test (`ROLLING_EVAL_MODELS` `base.py:411`,`:415`) |
| **foundation** | TimesFM-2.5, TiRex, DLinear | DLinear `fit_series`(lstsq) / TimesFM·TiRex context-store O(1) | DLinear = 학습 가중치를 관측 슬라이딩 윈도에 적용(refit 0) `dlinear.py:122`; TimesFM/TiRex = base `rolling_1step` context-feed `base.py:451-452` | **baseline 전용** rolling (`BASELINE_ROLLING_MODELS` `base.py:455`); R9 OOF/test 는 transform-space·챔피언 무영향이라 단일원점 유지 `base.py:446-450` |
| **pf (N-HiTS·N-BEATS·TiDE)** | PfNHiTS/PfNBeats/PfTiDE (`simulation/models/modern_ts/pf_models.py:392`,`:364`,`:420`) | `_PfBase.fit` pytorch-forecasting 1회 | encoder context-feed: test target 을 관측값으로 채워 placeholder 0.0 collapse 회피 `simulation/models/modern_ts/pf_models.py:285-288` (leak-free: decoder 가 예측할 target[t] 미열람) | **N-HiTS** = identity transform → raw `y_observed` rolling, baseline+OOF+test (`ROLLING_EVAL_MODELS` `base.py:423`, G-337); **N-BEATS(mcmc_robust)·TiDE(laplace)** = transform-space → `transform(y_observed)` 필요 (`TRANSFORM_ROLLING_MODELS` `base.py:465`, `:1376-1389`) |
| **fused** | FusedEpi, SeirCount-TabPFN | FusedEpi: TiRex base(train rolling 캐시) + TabPFN 잔차 1회 fit; SeirCount-TabPFN: feature fit-once | FusedEpi = TiRex 1-step 에 관측 history feed + TabPFN corr 보정 `fused_epi.py:266-272`; SeirCount-TabPFN = `predict(X_test)` static(lag feature 1-step) | **FusedEpi** = identity(y_mode=none)·내부 TiRex 출력 → raw `y_observed`, baseline+OOF+test (`ROLLING_EVAL_MODELS` `base.py:419`, G-336); **SeirCount-TabPFN** = `y_observed` 인자 없음(`seir_count.py:226`), 어느 rolling set 에도 없음 → feature-path static |

### 3.4 선택 OOF == 최종 eval 동일 잣대 (G-337b)

- **단일원점 forecast(len)** 은 sequence 모델을 68주 외삽→mean-revert→불공정 음수로 만들어 feature 모델(실 lag 1-step)과 비교 불가였다(`base.py:347-349`, 실측 ARIMA −0.89→+0.86, SARIMA −1.01→+0.86).
- 이제 OOF 선택(`_evaluate_config:468`, `_evaluate_config_hierarchical:774`)과 최종 test(`_refit_and_predict_test:1374`)가 **같은 게이트·같은 `predict(y_observed)` 경로**를 공유한다. OOF 헬퍼는 transform 모델을 위해 train→y_train_t affine map(polyfit deg-1)으로 `y_observed`를 모델 학습공간으로 변환(identity→raw, mcmc_robust/laplace→정확)한다(`per_model_optimize.py:332-352`).
- floor 도 동일: OOF 가 원단위에서 `transform_inv` 후 ≥0 floor 를 최종 eval 과 동일하게 적용(`per_model_optimize.py:481-485`).

**순효과**: preproc/mc/feature/HP 가 collapse 기준이 아니라 배포-충실 1-step 기준으로 선택되고, 패널 전체가 **uniform fit-once + 관측-feed (B)** 로 통일되어(Theta 가 마지막 A 잔존) 모델 간 챔피언 비교가 공정해졌다.

---

## §4 챔피언 선정 & 평가 지표

### 4.1 선정 funnel — G-339 LEAK-FREE (OOF 1-SE band → fold안정성 → parsimony → OOF-WIS, test 미사용)

챔피언은 **순수 OOF-argmin(G-307)** 도 **옛 G-318(shortlist 내 hold-out test argmin)** 도 아니다. SSOT 함수 = `simulation/pipeline/per_model_eval.py:115` `select_champion_g318`(함수명 유지, **body=G-339 leak-free**); post-hoc CLI(`rerank_champion.py`)도 동일 함수 import 로 100% 일치(`:28-30`).

| 단계 | 내용 (전부 leak-free — test 미사용) | 근거 file:line |
|------|------|----------------|
| ① **OOF 1-SE band** | eligible(유한 `oof_wis`) 정렬 → best 의 per-fold OOF 로 SE → `oof_wis ≤ best+max(SE, 2%·best)` band = 통계적 동률 cluster (Breiman 1984; top-K=8 cap) | `per_model_eval.py:135-150` |
| ② **fold 안정성** (primary tiebreak) | band 안 `_oof_fold_cv`(per-fold WIS 변동계수) 최소 = 분포이동 견고성 — **G-318 의 'hold-out 일반화'를 test 없이 WF-CV fold 안정성으로 대체** | `per_model_eval.py:106-126` (`_oof_fold_cv`), `:155` |
| ③ **parsimony** | 안정성 동률이면 `n_features` 최소 (Breiman 1-SE / Occam) | `per_model_eval.py:155` |
| ④ **OOF-WIS** | 최종 tiebreak (단조 — 유일 float) | `per_model_eval.py:155` |
| (병기) **epi 해석가능 count 모델** | **NegBinGLM**(정식 NB count) 별도 병기 — 챔피언 아님 | `rerank_champion.py:169` |

이는 G-307 SVR-RBF OOF-노이즈-우승(band+안정성)도 옛 G-318 winner's curse(test 완전 제거)도 동시 회피한다. `rerank_champion.py` 는 **무재학습**(기존 `per_model_optimal/*.json` 읽기)으로 재실행.

**hold-out test 의 역할 = 진단 병기만**(배포 아님): `select_champion_holdout_best`(`per_model_eval.py:194`, test WIS 최저) + DM 유의성(`rerank_champion.py`)은 "leak-free 챔피언이 미관측 시즌서도 잘했나?" 투명성 표시 — G-339 와 같으면 강한 증거, 다르면 둘 다 보고하되 **test 는 선정에 참여하지 않는다.**

> 폐기 이력: **G-307**(순수 OOF-argmin = OOF 과적합 SVR-RBF) → **G-318**(OOF top-8 shortlist 내 hold-out test argmin = 부분 winner's curse, 외부 reviewer #1 2026-06-24) → **G-339**(OOF 1-SE band 내 leak-free tiebreaker).

### 4.2 챔피언은 run-dependent (배포 배선)

`rerank_champion.py` 의 챔피언 이름은 **하드코딩이 아니라** 현재 run 산출물 `simulation/results/per_model_optimal/*.json` 의 `val_metrics.oof_wis` + `test_metrics.wis/r2` 로 매 run 재계산된다 (`rerank_champion.py:36-58, :115-119`). 따라서 챔피언은 그 run 의 데이터에 의존한다. docstring 의 "TabPFN(empirical 1위)/NegBinGLM(epi 해석가능 count 모델)"은 검증 시점(2026-06-19 run)의 실측 예시이며 (`rerank_champion.py:78`), **영구 고정값이 아니다**.

배포 경로: R10(`per_model_eval`)이 G-318 챔피언을 `ranking_top10[0]` **맨 앞**에 박고 명시 `champion` 필드로 반환한다 — `real_eval.py:797`·web·`.pt` 소비자가 `ranking_top10[0]` 을 챔피언으로 읽으므로 OOF-best([0]=G-307)를 박으면 G-318 이 우회되기 때문 (`per_model_eval.py:1663-1676`).

### 4.3 평가 지표 — 129-metric full eval SSOT, WIS 헤드라인

- **129-key SSOT**: `evaluate_predictions_full` (`simulation/pipeline/phase_evaluator.py:47`) 이 모델당 **129-key battery** 를 계산한다 — docstring 명시 "Compute the 129-key SSOT battery per-model" (`phase_evaluator.py:68, :85`). 실제로는 **128 metric + `phase_id` 1키** (`phase_evaluator.py:110, :120`). `MPH_FAST_METRIC=1` / `MPH_FULL_EVAL_TRAJECTORY=0` 면 skip-marker 만 반환 (`phase_evaluator.py:106-113`).
- **WIS = primary proper scoring rule (헤드라인)**: R10 report 가 "**Headline metric = WIS (Bracher 2021)**; also reported: log-WIS (Bosse 2023)" 로 명시 (`per_model_eval.py:1615`, `:1646-1647`). 챔피언 selection 기준도 WIS(`oof_wis` shortlist → `test_wis` 일반화). WIS 는 점추정 + interval(11 quantile) + miss penalty 를 통합한 proper interval score 로 계산된다 (`phase_evaluator.py:298-335` `weighted_interval_score_empirical`). rubric 상 WIS direction=lower, FluSight primary (`simulation/analytics/metric_rubric.py:163-168`).
- **R²/MAPE/PICP 는 개별 metric 으로만 병기** (게이트 아님): 챔피언은 "순수 best-WIS" 이고 R²/MAPE/WIS/PICP95 의 bootstrap CI 는 개별 진단으로만 산출 (`per_model_eval.py:1484, :1531, :1580-1582`).
- **4-criteria 필터(g175) 완전 폐기**: 이전 4-criteria(R²≥0.80 AND MAPE≤20% AND WIS≤6.0 AND PI95≥0.90) gate 는 2026-06-05 사용자 명시로 제거 — `phase_evaluator`·`per_model_eval` 모두 g175-free (`per_model_eval.py:1580-1581`; `simulation/analytics/metric_rubric.py:355-361`). 유지되는 것은 DM family(BH-FDR)와 Bootstrap CI 뿐 (`simulation/analytics/metric_rubric.py:363-372`).

### 4.4 논문 우선순위 metric — TOP_3

⚠ **위치 정정**: 프롬프트는 `metric_rubric.py:PAPER_TOP_*` 를 지목했으나, `PAPER_TOP_{2,3,5,10}` 와 `classify_paper_tier` 는 **2026-05-26 코드에서 제거**되었다 (사용자 명시 "paper top tier은 왜 만들어?!", `simulation/analytics/metric_rubric.py:355-358`). TOP set 의 현재 SSOT 는 문서 `docs/_archive/METRIC_TOP_PRIORITY.md` 이다 (프롬프트의 `docs/METRIC_TOP_PRIORITY.md` 는 `_archive/` 하위로 이동됨).

핵심 추천 = **TOP 3 = {WIS, alert_f1, lead_time_weeks}** (`METRIC_TOP_PRIORITY.md:22-28, :123`):

| Rank | Metric | 측면 | 의미 | 근거 |
|------|--------|------|------|------|
| 1 | **WIS** | AI | proper interval score, FluSight primary | Bracher 2021 |
| 2 | **alert_f1** | 보건역학 | KDCA threshold(8.6/1000) 위 주차 발견 F1 | Buckeridge 2007 / EARS |
| 3 | **lead_time_weeks** | 운영 | 모델이 관측 threshold crossing 보다 몇 주 일찍 경보? | RespiCast 2023+ |

코드 rubric(`simulation/analytics/metric_rubric.py:RUBRIC`)은 각 metric 의 direction + excellent/good/acceptable threshold + 인용을 보유 — WIS(lower, FluSight primary, `:163-168`), alert_f1(higher≥0.90 excellent, `:240-245`), peak_week_err·epi_peak_mae 등 epi-curve metric도 등록 (`:227-238`). phase→metric SSOT 는 `PHASE_METRICS` (`simulation/analytics/metric_rubric.py:391-428`): R14(per-model eval SSOT)=`ALL_134`(논문 Table 1 row, `:420-421`); 정수 phase 13(HP-optimize, slab oof_cv)의 primary=`wis,r2,mape,pi95_coverage`(`:419`)이며 정수 phase 10='PI + conformal'(`:406`)은 별개다. ⚠ R/P 라벨과 legacy 정수 phase 번호가 codebase 에 혼재하므로(R/P↔정수 SSOT=`simulation/pipeline/phases.py`), `wis,r2,mape,pi95_coverage` primary 를 per_model_eval(정수 14, primary=ALL_134)에 붙이지 말 것.

> **caveats (이 섹션)**:
> (1) 프롬프트가 지목한 `metric_rubric.py:PAPER_TOP_*` 는 2026-05-26 코드에서 제거됨 — TOP_3 SSOT 는 코드가 아니라 `docs/_archive/METRIC_TOP_PRIORITY.md` 문서.
> (2) `METRIC_TOP_PRIORITY.md` 는 프롬프트 경로(`docs/`)가 아니라 `docs/_archive/` 하위에 있고, §109 등 일부 본문은 폐기된 G-175 4-criteria filter 를 stale 하게 언급함(코드는 제거됨, `simulation/analytics/metric_rubric.py:355-361`).
> (3) "129-metric": `phase_evaluator` docstring=129-key SSOT, 실제=128 metric + phase_id 1키 (MEMORY.md 정정과 일치).
> (4) `PHASE_METRICS` 의 R14 primary=`ALL_134` 라벨과 docstring `full_134` 는 역사적 라벨로, 의미는 '전체 129'(`simulation/analytics/metric_rubric.py:389-390` 주석 명시).
> (5) phase→metric 매핑은 R/P 라벨이 아니라 legacy **정수** phase 키로 인덱싱됨 — 정수 13(HP-optimize)=`wis,r2,mape,pi95_coverage`, 정수 10='PI + conformal', 정수 14(per_model_eval)=ALL_134. R/P↔정수 매핑 SSOT 는 `simulation/pipeline/phases.py`.
> (6) 챔피언 실제 이름(TabPFN 등)은 현재 `per_model_optimal/*.json` 디렉토리 내용 의존 — JSON 개별 파일은 읽지 않고 funnel 로직만 검증함.

---

## §5 방법론 출처 (인용)

본 파이프라인이 사용하는 모든 방법의 표준 출처를 테마별로 정리한다. 각 항목은 정확한 표준 인용(저자·연도·venue)과 함께 **코드/문서에서 실제 인용되는 위치**(file:line)를 병기한다. 코드에 명시적 인용 문자열이 있는 경우 file:line 으로 못박았고, 코드엔 없으나 방법론 문서(`docs/`, `paper/`)에서 인용하는 경우 해당 위치를 표시했다.

### 5.1 평가 프로토콜 · 교차검증 (Evaluation / CV)

- **Rolling-origin / out-of-sample 평가** — Tashman, L.J. (2000). "Out-of-sample tests of forecasting accuracy: an analysis and review." *International Journal of Forecasting* 16(4):437–450.
  - 코드 인용: `simulation/models/base.py:361` ("배포 충실, Tashman 2000"), `simulation/models/ts_models.py:464`. 문서: `docs/NEW_MODEL_IDEAS_20260622.md:159` (§8.6 refit-대칭 = fit-once+관측-feed, Tashman 2000 = eval가 실제 배포조건 복제), `paper/_methodology_external_review_20260622/code/rolling_origin.py:9`.
  - 구현부: `_rolling_origin_forecast` / `_rolling_origin_multihorizon` (`simulation/pipeline/real_eval.py:182, 258`).
- **시계열 CV 타당성** — Bergmeir, C. & Benítez, J.M. (2012). "On the use of cross-validation for time series predictor evaluation." *Information Sciences* 191:192–213.
  - 인용: `docs/SCI_EVAL_PROTOCOL_AND_LIMITATIONS_20260614.md:19,34` (rolling-origin 표준 근거), `paper/_methodology_external_review_20260622/METHODOLOGY_AND_HONEST_LIMITATIONS.md:212` ("k-fold 무효, walk-forward/rolling-origin 표준"). 관련 보강: Bergmeir, Hyndman & Koo (2018, *CSDA* 120) — lagged-predictor autoregression 정당화 (`METHODOLOGY_AND_HONEST_LIMITATIONS.md:214`).
- **예측 원리/교과서** — Hyndman, R.J. & Athanasopoulos, G. *Forecasting: Principles and Practice* (FPP, OTexts) — rolling-origin / time-series cross-validation 절.
  - 코드 인용(2nd ed.): `simulation/server/static_citations.py:252` (`hyndman_2018_sMAPE`). 추가: sMAPE 대칭성 = Hyndman & Athanasopoulos 2021 (`simulation/pipeline/per_model_optimize.py:1779`), MASE = Hyndman 2006 (`simulation/pipeline/phase_evaluator.py:174`). 프로토콜 문서: `docs/SCI_EVAL_PROTOCOL_AND_LIMITATIONS_20260614.md:19`.

### 5.2 점수 규칙 (Scoring rules)

- **WIS (Weighted Interval Score) / 구간 예측 평가** — Bracher, J., Ray, E.L., Gneiting, T. & Reich, N.G. (2021). "Evaluating epidemic forecasts in an interval format." *PLOS Computational Biology* 17(2):e1008618.
  - 코드 인용: `simulation/server/static_citations.py:310` (`flusight_bracher_2021`, DOI 10.1371/journal.pcbi.1008618), WIS 공식 `simulation/pipeline/per_model_optimize.py:1202` ("(Bracher 2021)"), WIS decomposition `simulation/pipeline/metric_eval.py:344,434`, K=4 multi-level `simulation/pipeline/real_eval.py:582,1316,1665`, ABM WIS `simulation/abm/behavior_disease_validation.py:14,31,54`.
- **Strictly proper scoring rules (CRPS / WIS 이론 기초)** — Gneiting, T. & Raftery, A.E. (2007). "Strictly proper scoring rules, prediction, and estimation." *Journal of the American Statistical Association* 102(477):359–378.
  - 코드 인용: `simulation/server/static_citations.py:296` (`gneiting_2007_wis`, DOI 10.1198/016214506000001437).
- **Brier score 분해** — Murphy, A.H. (1973). "A new vector partition of the probability score." *Journal of Applied Meteorology* 12(4):595–600.
  - 코드 인용: `simulation/server/static_citations.py:236` (`murphy_brier_1973`).
- **Conditional calibration / PIT** — Czado, C., Gneiting, T. & Held, L. (2009). "Predictive model assessment for count data." *Biometrics* 65(4):1254–1261.
  - 코드 인용: `simulation/pipeline/metric_eval.py:391,462` (DOI 10.1111/j.1541-0420.2009.01191.x).

### 5.3 유의성 검정 (Significance tests)

- **Diebold-Mariano 검정** — Diebold, F.X. & Mariano, R.S. (1995). "Comparing predictive accuracy." *Journal of Business & Economic Statistics* 13(3):253–263.
  - 코드 인용: `simulation/server/static_citations.py:325` (`diebold_mariano_1995`, DOI 10.1080/07350015.1995.10524599); 구현 R6 = `simulation/pipeline/dm_test.py:2,17,93`; real_eval DM = `simulation/pipeline/real_eval.py:761,1562`.
- **소표본 보정 (HLN)** — Harvey, D., Leybourne, S. & Newbold, P. (1997). "Testing the equality of prediction mean squared errors." *International Journal of Forecasting* 13(2):281–291.
  - 코드 인용(HLN-corrected DM, t_{n-1}): `simulation/abm/behavior_disease_validation.py:19,142–155` (`dm_hln`), `simulation/abm/epi_proof.py:668–686` (`_hln_dm`). 문서: ABM behavioural 검증 §, `paper/_methodology_external_review_20260622/METHODOLOGY_AND_HONEST_LIMITATIONS.md:247`.
- **Model Confidence Set (보조)** — Hansen, P.R., Lunde, A. & Nason, J.M. (2011). "The model confidence set." *Econometrica* 79(2):453–497.
  - 코드 인용: `simulation/server/static_citations.py:223` (`hansen_mcs_2011`).
- **Unconditional coverage (Kupiec)** — Kupiec, P.H. (1995). "Techniques for verifying the accuracy of risk measurement models." *Journal of Derivatives* 3(2):73–84.
  - 코드 인용: `simulation/server/static_citations.py:340` (PICP coverage 검정).

### 5.4 변수 선택 (Feature selection)

- **Stability selection** — Meinshausen, N. & Bühlmann, P. (2010). "Stability selection." *JRSS: Series B* 72(4):417–473.
  - 코드 인용: `simulation/pipeline/feature_select_corr1se.py:9,47,67` (`select_features_stability`), `simulation/pipeline/per_model_optimize.py:2278` (Stage 2 FEATURE OPTIMIZATION = STABILITY SELECTION).
- **1-SE (one-standard-error) rule** — Breiman, L., Friedman, J.H., Olshen, R.A. & Stone, C.J. (1984). *Classification and Regression Trees (CART)*. Wadsworth.
  - 코드 인용: `simulation/pipeline/feature_select_corr1se.py:13,192–205` (`select_size_path_1se`), Tree split-rationale `simulation/pipeline/per_model_optimize.py:112` ("Breiman 1984").
- **ESL (parsimony / shrinkage 배경)** — Hastie, T., Tibshirani, R. & Friedman, J. *The Elements of Statistical Learning (ESL)*. Springer.
  - 코드 인용: `simulation/pipeline/per_model_optimize.py:113,117` (Linear "Friedman ESL §3.4", GAM "Hastie-Tibshirani §3.2").
- **다중공선성 (VIF / PCA)** — VIF = Kutner, Nachtsheim, Neter & Li, *Applied Linear Statistical Models* (5th ed., 2005); PCA = Jolliffe, *Principal Component Analysis* (2nd ed., 2002, Springer).
  - 코드 사용(인용 문자열 없이 구현만): per-model multicollinearity 선택 `none/vif/corr/pca` — `simulation/pipeline/per_model_optimize.py` (`MPH_MC_PER_MODEL`). 표준 인용은 코드에 명시되어 있지 않음(부록 참조).

### 5.5 구간 예측 / Conformal (Prediction intervals)

- **Conformal prediction (이론)** — Vovk, V., Gammerman, A. & Shafer, G. (2005). *Algorithmic Learning in a Random World*. Springer.
  - 코드 인용: `simulation/models/conformal.py:8,353`, `simulation/pipeline/real_eval.py:39,1010`, `simulation/models/modern_ts/conformal.py:48` (Vovk 2012 finite-sample correction).
- **Conformalized Quantile Regression (CQR)** — Romano, Y., Patterson, E. & Candès, E.J. (2019). "Conformalized quantile regression." *NeurIPS* 32.
  - 코드 인용: `simulation/models/conformal.py:723,748`, 클래스 `CQRSplit` (`conformal.py:903`); 파이프라인 적용 `simulation/pipeline/intervals.py:512–514,568–601`.
- **Split conformal (유한표본 보장)** — Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R.J. & Wasserman, L. (2018). "Distribution-free predictive inference for regression." *JASA* 113(523):1094–1111.
  - 코드 인용: `simulation/models/conformal.py:219`, `simulation/pipeline/real_eval.py:1022,1660`.
- **Jackknife+ / CV+** — Barber, R.F., Candès, E.J., Ramdas, A. & Tibshirani, R.J. (2021). "Predictive inference with the jackknife+." *Annals of Statistics* 49(1):486–507.
  - 코드 인용: `simulation/models/conformal.py:52,153`.
- **Adaptive Conformal Inference (ACI)** — Gibbs, I. & Candès, E.J. (2021). "Adaptive conformal inference under distribution shift." *NeurIPS* 34.
  - 코드 인용: `simulation/models/conformal.py:12`, `simulation/__main__.py:574`, `simulation/pipeline/per_model_optimize.py:1609,1748,2728`, `simulation/pipeline/real_eval.py:1163`, `simulation/pipeline/config.py:118` (Zaffran 2022 보강 참조).

### 5.6 역학 모델 (Epidemiological models)

- **Theta method** — Assimakopoulos, V. & Nikolopoulos, K. (2000). "The theta model." *IJF* 16(4):521–530. SES 등가성: Hyndman, R.J. & Billah, B. (2003). "Unmasking the Theta method." *IJF* 19(2):287–290.
  - 코드 인용: `simulation/models/ts_models.py:335,338–363` ("Assimakopoulos & Nikolopoulos 2000 Theta method — M3 winner"). (Hyndman & Billah 2003 SES-등가는 코드에 별도 명시 없음 — 부록 참조.)
- **Endemic-epidemic / hhh4** — Held, L., Höhle, M. & Hofmann, M. (2005). *Statistical Modelling* 5(3):187–199. + Held, L. & Paul, M. (2012). "Modeling seasonality in space-time infectious disease surveillance data." *Biometrical Journal* 54(6):824–843.
  - 코드 인용: `simulation/models/hhh4_benchmark.py:5,19,22,146` ("Held & Paul 2012"; Meyer, Held & Höhle 2017 보강), `simulation/pipeline/real_eval.py:972`; 구현 `HHH4Equivalent` (`real_eval.py:978–986`).
- **Rt 추정 (EpiEstim)** — Cori, A., Ferguson, N.M., Fraser, C. & Cauchemez, S. (2013). *American Journal of Epidemiology* 178(9):1505–1512.
  - 코드 인용: `simulation/models/epiestim_models.py:4,7,20,47,67`, `simulation/server/static_citations.py:281` (DOI 10.1093/aje/kwt133).
- **Wallinga-Teunis Rt** — Wallinga, J. & Teunis, P. (2004). *American Journal of Epidemiology* 160(6):509–516. (집계형 = Fraser 2007 재정식화)
  - 코드 인용: `simulation/models/wallinga_teunis.py:4,7,21,42–49`.
- **GLARMA (observation-driven count)** — Davis, R.A., Dunsmuir, W.T.M. & Streett, S.B. (2003). "Observation-driven models for Poisson counts." *Biometrika* 90(4):777–790. (Dunsmuir & Scott 2015 보강)
  - 코드 인용: `simulation/models/glarma_models.py:4,12,23`, `simulation/models/base.py:412`. (코드는 venue 미기재 — 부록 참조.)
- **EARS 이상탐지 (보조 outbreak detection)** — Hutwagner, L., Thompson, W., Seeman, G.M. & Treadwell, T. (2003). *Journal of Urban Health* 80(2 Suppl 1):i89–i96.
  - 코드 인용: `simulation/models/ears_models.py:4,22,89,138`.
- **SEIR 컴파트먼트 (Stage 5 / ABM 엔진 기반)** — Kermack, W.O. & McKendrick, A.G. (1927). *Proc. Royal Society A* 115(772):700–721. + Hethcote, H.W. (2000). *SIAM Review* 42(4):599–653.
  - 코드 인용: `simulation/server/static_citations.py:131,145`.
- **FluSight baseline (CDC / Reich lab)** — CDC FluSight Hub flat/persistence baseline; Reich, N.G. et al. (FluSight 평가 표준); Mathis, S.M. et al. (2024). *Nature Communications* 15:6289.
  - 코드 인용: `simulation/pipeline/per_model_eval.py:785` ("FluSight Hub standard, Bracher 2021 / Reich 2019"), `:1486` (Mathis et al. 2024), `simulation/pipeline/phase_evaluator.py:78` (WOY climatology BSS = Reich 2019). FluSight α-grid: `FLUSIGHT_ALPHAS` (`per_model_eval.py:400,601,667`).

### 5.7 딥러닝 / Foundation backbone (Model backbones)

- **TabPFN** — Hollmann, N., Müller, S., Eggensperger, K. & Hutter, F. (2023). *ICLR 2023*. v2 회귀: Hollmann, N. et al. (2025). "Accurate predictions on small data with a tabular foundation model." *Nature* 638:319–326.
  - 코드 인용: `simulation/models/tabpfn_wrapper.py:5`, `simulation/models/registry.py:481`.
- **TimesFM** — Das, A., Kong, W., Sen, R. & Zhou, Y. (2024). "A decoder-only foundation model for time-series forecasting." *ICML 2024* (arXiv:2310.10688). 본 코드는 TimesFM 2.5 (200M, Google) ship.
  - 코드 인용: `simulation/models/timesfm_wrapper.py:1,5`. (저자/venue 문자열은 wrapper 헤더에 미기재 — 부록 참조.)
- **TiRex / xLSTM** — Beck, M., Pöppel, K., Spanring, M., Auer, A. et al. (2024). "xLSTM: Extended long short-term memory." *NeurIPS 2024* (arXiv:2405.04517). TiRex = NX-AI xLSTM 기반 zero-shot TS foundation (2025).
  - 코드 인용: `simulation/models/tirex_wrapper.py:1,3,42,54`.
- **DLinear** — Zeng, A., Chen, M., Zhang, L. & Xu, Q. (2023). "Are transformers effective for time series forecasting?" *AAAI 2023* 37(9):11121–11128.
  - 코드 인용: `simulation/models/dlinear.py:1,45,56`, `simulation/models/registry.py:491`.
- **N-BEATS** — Oreshkin, B.N., Carpov, D., Chapados, N. & Bengio, Y. (2020). *ICLR 2020*.
  - 코드 인용: `simulation/models/modern_ts/nbeats.py:4,23`, `simulation/models/registry.py:452`, `simulation/models/_optuna_samplers.py:415`. (live N-BEATS/N-HiTS/TiDE 라인업은 pf wrapper — MEMORY 참조.)
- **N-HiTS** — Challu, C., Olivares, K.G., Oreshkin, B.N., Garza, F., Mergenthaler-Canseco, M. & Dubrawski, A. (2023). *AAAI 2023* 37(6):6989–6997.
  - 코드 인용: `simulation/models/modern_ts/nhits.py:4,22`, `simulation/models/_optuna_samplers.py:429`.
- **TiDE** — Das, A., Kong, W., Leach, A., Mathur, S., Sen, R. & Yu, R. (2023). "Long-term forecasting with TiDE: Time-series Dense Encoder." *TMLR*.
  - 코드 인용: `simulation/models/modern_ts/tide.py:6,95`, `simulation/models/_optuna_samplers.py:444`.
- **GNN backbone (보조)** — GCN: Kipf, T.N. & Welling, M. (2017). *ICLR 2017* (`simulation/models/graph_models_pyg.py:516`). GIN: Xu, K., Hu, W., Leskovec, J. & Jegelka, S. (2019). *ICLR 2019* (`graph_models_pyg.py:532`).

---

## §6 리뷰어 검증 체크리스트

타 AI / 리뷰어가 코드로 직접 확인할 수 있는 항목. 각 행은 **파일:함수**(또는 file:line)와 **기대 동작**을 짝지었다.

| # | 검증 질문 | 확인 위치 (file:function/line) | 기대 동작 (PASS 조건) |
|---|-----------|-------------------------------|----------------------|
| 1 | **선택이 test 를 보는가?** | `per_model_optimize.py:2914,2917` (선정 `oof_wis` vs 보고 `test_metrics`); `per_model_eval.py:_assign_oof_and_test_ranks:71` (정렬 `sorted(rows, key=_oof)` `:93`) | 선정 키는 `oof_wis`(5-fold WF-CV)뿐 — hold-out 으로 정렬하면 docstring(`:74-77`)이 winner's curse 라고 금지 |
| 2 | **챔피언 funnel 이 G-318 인가?** | `per_model_eval.py:select_champion_g318:115-141` (`CHAMPION_SHORTLIST_K=8`) | OOF top-8 shortlist → 그 안에서만 hold-out test WIS 최저. 53개 직접 hold-out 선정 불가 |
| 3 | **shortlist 밖 모델이 챔피언 될 수 있나?** | `tests/test_champion_best_wis.py:test_holdout_influences_only_within_shortlist:51-56` | OOF-shortlist 밖 "lucky" hold-out-best 모델은 챔피언 불가 (test PASS) |
| 4 | **OOF fold 에 val(27)/test(68)/real 이 새는가?** | `per_model_optimize.py:_oof_cv_wis:1859` (`n=len(X_train)=242`), `:1894-1908` | fold 풀 = train(242)만; val/test/real 미포함. expanding `[:end_tr]` train, sliding `[end_tr:end_va]` val |
| 5 | **fold global-summary feature 누수 차단?** | `per_model_optimize.py:_recode_advanced_per_fold:1904`; `wfcv.py:66-254` | quantile bin/above_threshold/interaction 을 fold 마다 `[:end_tr]` 로 재코딩 (G-309) |
| 6 | **선택 OOF == 최종 test 동일 1-step?** | `per_model_optimize.py:_rolling_or_static_predict_oof:326`; `_refit_and_predict_test:1374` | 두 경로가 같은 게이트(`supports_rolling_eval ∪ supports_transform_rolling`)·같은 `predict(y_observed)` 호출 (G-337b) |
| 7 | **rolling 이 leak-free 인가?** | `base.py:rolling_1step:355-385` (`:380` `y_observed[:i]`); `fused_epi.py:267`; `simulation/models/modern_ts/pf_models.py:282` | 주 i 예측에 `y_observed[:i]` 만 사용; decoder 가 예측할 target[t] 미열람 |
| 8 | **eval 패널이 uniform fit-once B-protocol 인가?** | `base.py:355-364` (docstring "per-origin 재추정 A 모델 0"); `ts_models.py:Theta:413-492` | A-style 재적합 모델 0. Theta 가 G-338 로 고정 α/seasonal/trend B 전환 |
| 9 | **`MPH_BEST_BY=oof_cv` 가 val-single 을 막는가?** | `per_model_optimize.py:2055-2056`; `_inline_optuna_3stage.py:817-819` | env override 시 best 결정=OOF WIS, n=27 single-val 금지 (G-132) |
| 10 | **preproc 가 Optuna 아닌 flat-grid 인가?** | `_inline_optuna_3stage.py:852-909` ("PURE-GRID (no Optuna)"); 1-SE `:604-617` | 7-transform 각 1회 OOF + fold-paired 1-SE. study/sampler/pruner 없음 (G-335; "100 trial" 노트 stale) |
| 11 | **mc 가 do-no-harm 가드를 갖는가?** | `per_model_optimize.py:_per_model_mc_choice:3682-3718` | `none` 대비 상대 개선 ≥ 0.02 일 때만 vif/corr/pca 채택, 애매하면 `none` |
| 12 | **post-hoc rerank 가 in-pipeline 챔피언과 일치하나?** | `scripts/rerank_champion.py:28-30` (`select_champion_g318` import) | 동일 SSOT 함수 import — 무재학습 재실행이 100% 일치 |

---

## §7 eval-fairness 캠페인 타임라인

sequence 모델을 feature 모델과 동일한 1-step 잣대로 공정 비교하기까지의 수정 이력.

| Gotcha | 무엇을 고쳤나 | 핵심 코드 | 실측 효과 |
|--------|--------------|-----------|----------|
| **G-321** | classic-ts hold-out 음수 R² 의 진짜 원인 = 평가 task 불일치(feature=1-step via lag vs sequence=단일원점 68주 외삽). rolling 1-step 으로 classic-ts 공정화. raw `y_orig` 사용(transform 공간 함정 회피) | `base.py:342-353` (predict 분기), rolling loop | ARIMA −0.89→+0.92 등 양수 회복 (scope=classic-ts) |
| **G-327** | epi/count 모델을 관측 lag/잔차 rolling 으로 전환 (PoissonAutoreg/hhh4/EpiEstim/Wallinga/GLARMA) | `epi_models.py:900`, `hhh4_models.py:169`, `epiestim_models.py:138`, `wallinga_teunis.py:102`, `glarma_models.py:140-161` | epi 모델 전 baseline 양수 |
| **G-336** | FusedEpi eval 배선 — identity(y_mode=none)·내부 TiRex 출력 → raw `y_observed` rolling 으로 ROLLING_EVAL_MODELS 편입 (baseline+OOF+test) | `fused_epi.py:266-272`; `base.py:419` | FusedEpi 가 R9 에서도 rolling 공정화 |
| **G-337** | N-HiTS·N-BEATS·TiDE static-collapse 수정 — sequence 모델 rolling 공정화. N-HiTS=identity raw rolling; N-BEATS/TiDE=transform-space rolling | `simulation/models/modern_ts/pf_models.py:285-288`; `base.py:423,465` | static placeholder 0.0 collapse 회피 |
| **G-337b** | preproc/mc/feature/HP 선택의 OOF 도 sequence 모델 rolling 1-step 으로 통일 — 선택 OOF == 최종 eval 동일 헬퍼 | `per_model_optimize.py:326-354,466-485,772-774` | selection==eval (collapse 기준 → 배포-충실 1-step 기준) |
| **G-338** | Theta symmetric-refit — 마지막 A-style(매 origin 재적합) 잔존을 fit-once B 로 전환. eval 패널 전체 uniform B 완성 | `ts_models.py:413-492` | legacy refit 대비 max\|Δ\|≈1.1, R² 0.938 vs 0.934 |

---

## 부록 A. 알려진 한계 / 주의 (정직)

데이터/코드 검증 시 발견된 한계와 주의점. 리뷰어가 over-claim 으로 오해하지 않도록 정직하게 명시한다.

1. **polyfit deg-1 affine map (G-337b OOF transform)**: OOF 헬퍼가 transform 모델의 `y_observed` 를 학습공간으로 옮길 때 train→y_train_t affine map(polyfit deg-1)을 쓴다 — **identity / affine(laplace, mcmc_robust 의 location-scale) 후보는 정확**하나, 비-affine transform 후보는 **근사**다 (`per_model_optimize.py:332-352`). 실무상 라이브 라인업의 transform-rolling 모델(N-BEATS=mcmc_robust, TiDE=laplace)은 affine 계열이라 정확. (pf 라인 인용 = `simulation/models/modern_ts/pf_models.py:285-288,:364,:392,:420`.)
2. **baseline (R2) 는 진단용**: §3.1 의 baseline 표면은 BASIC feature(lag+계절성 13)·8-모델 factory 진단/sanity 용이며 **champion 선정 SSOT 가 아니다**. champion 선정 OOF 는 R9 의 5-fold(전체 feature pool, **fold 데이터 = X_train 242**; val/test/real 미포함 — `wfcv.py:run_wfcv` 의 step=1 walk-forward 진단과 별개).
3. **1-step-primary, multi-horizon 은 진단 분리**: 본 평가의 headline 은 1-step rolling-origin 이다. multi-horizon(`_rolling_origin_multihorizon`, `real_eval.py:258`)·K=4 multi-level WIS 는 별도 진단 트랙이며 챔피언 1-step 비교와 분리해 읽어야 한다.
4. **HP 예산 launch-time 주입 미검증**: `_optuna_budget.get_trials` 코드 기본값은 **20**이다. 메모리 노트의 "tree 50 / DL 40"은 launch-time 주입값으로 추정되며, runner 의 `set_budget` 주입 경로는 본 스냅샷에서 직접 검증하지 않았다. mc 비교는 cheap pre-pass(n_folds=2), preproc/feature OOF 는 `oof_folds`(기본 5).
5. **stale 노트 정정 (코드와 불일치)**: (a) "preproc 100 Optuna trial" → 현 코드는 G-335 flat-grid(Optuna 제거). (b) "anchor blend per-model α(G-141/144)" → G-231 로 dead, 주석만 잔존(`_optuna_torch.py:170-172`, `_optuna_samplers.py:294`). (c) `metric_rubric.py:PAPER_TOP_*` → 2026-05-26 제거됨, TOP_3 SSOT 는 `docs/_archive/METRIC_TOP_PRIORITY.md` 문서.
6. **"129-metric" 정확 의미**: `phase_evaluator` docstring 은 "129-key SSOT" 라 적지만 실제는 **128 metric + `phase_id` 1키** = 129 (`phase_evaluator.py:110,120`). `PHASE_METRICS` 의 `ALL_134` / `full_134` 라벨은 역사적 라벨이며 의미는 '전체 129'(`simulation/analytics/metric_rubric.py:389-390` 주석).
7. **표준 인용 코드 미기재 항목**: VIF/PCA(Kutner 2005 / Jolliffe 2002), Theta SES-등가(Hyndman & Billah 2003), TimesFM 저자/venue(Das 2024), GLARMA venue(Biometrika 90(4))는 **방법은 구현·사용되나 코드에 인용 문자열이 없다** — §5 의 표준 인용은 방법론적으로 정확하나 코드 grep 으로는 못 잡힌다.
8. **`conformal_holdout_weeks` 이중값**: `config_global.py:537` 의 26 은 **미적용 정적 참조값**이고, 런타임 split 은 `pipeline/config.py:138` 의 0 을 써 conformal holdout slab 이 비활성(`holdout_start=n`). 두 값이 공존하므로 코드를 읽을 때 어느 config 가 라이브인지 확인 필요.
9. **챔피언 이름은 run-dependent**: "TabPFN / NegBinGLM" 은 2026-06-19 run 실측 예시이며 영구 고정값이 아니다. funnel **로직**은 검증했으나 개별 `per_model_optimal/*.json` 파일 내용은 본 스냅샷에서 읽지 않았다.
10. **검증 범위**: 모든 file:line 은 **import-경로의 라이브 코드** 기준이며 실제 학습 run 로그·산출물은 미확인. 동작 주장은 정적 코드 검증이고, 실행 시점의 env override(`MPH_*`)에 따라 분기가 달라질 수 있다.
