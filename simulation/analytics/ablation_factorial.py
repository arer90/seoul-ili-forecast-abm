"""2^3 factorial ablation orchestrator (SAFE approximation, 2026-06-02).

12-family 패널 × 8 cell(preproc × HP × feature) → `ablation_stats.factorial_effects`.

SAFE approximation (사용자 선택 2026-06-02): feature-off = BASIC pool(MPH_PHASE13_FEATURE_POOL=basic;
STABILITY 가 13개에 돌아 ≈전부 유지) — 위험한 selection-off 핵심블록 편집 회피. 토글은 전부 **기존 env**
(MPH_PREPROC_OPTUNA / MPH_HP_OPTUNA_TRIALS / MPH_PHASE13_FEATURE_POOL) → 코드 무변경.

실행 메커니즘 (2026-06-02 코드 검증): GLOBAL config 는 **frozen dataclass**(재현성 #5) 라 in-process
변이 불가(FrozenInstanceError). env 는 GLOBAL **생성(import) 시점**에만 읽힘 → 유일한 깨끗한 토글 =
**cell당 subprocess**(env → fresh GLOBAL) + G-158 메모리 격리. cell_env 의 3 env 가 그대로 fresh GLOBAL
로 연결: MPH_PREPROC_OPTUNA→training.preproc_optuna, MPH_HP_OPTUNA_TRIALS→optuna.hp_trials_default,
MPH_PHASE13_FEATURE_POOL→runner A1 flag(BASIC slice). MPH_OUTPUT_ROOT 로 main run 과 완전 격리.

이 모듈 = **orchestration + subprocess 빌딩블록** (CELLS·env-map·패널·CLI 명령·집계). 실제 cell 적합은
caller 가: (테스트) fit_fn mock 주입 → run_factorial; (실행) cell_cli_command + cell_run_env 로 8 cell
subprocess → per_model_optimal/{model}.json 의 refit_test_predictions 수집 → factorial_effects.
"""
from __future__ import annotations

import itertools

from simulation.analytics.ablation_stats import factorial_effects

__all__ = ["FACTORS", "CELLS", "PREPROC_SHALLOW", "HP_SHALLOW",
           "cell_env", "select_panel", "run_factorial", "aggregate_effects",
           "cell_run_env", "cell_cli_command", "collect_cell_predictions", "reshape_to_per_model"]

#: 요인 순서 (cell 튜플 인덱스와 일치) — factorial_effects 의 factors 와 동일.
FACTORS = ("preproc", "hp", "feature")
#: 2^3 = 8 cell. 각 = (preproc, hp, feature) ∈ {0,1}³.
CELLS = [tuple(c) for c in itertools.product((0, 1), repeat=3)]


#: DEPTH ablation shallow 값 — 파이프라인이 preproc·HP 를 **항상 최적화**(off 불가): preproc
#: per_model_optimize L1350 `use_preproc_optuna=(feature_cols and not meta)` 가 env 무시하고 always-ON;
#: HP _optuna_budget.get_trials 가 `max(5, …)` 하한 + unset 시 default 20. 따라서 bit=0 은 끄기가
#: 아니라 **최소-탐색(shallow)**: PREPROC_SHALLOW=1(preproc 1 trial ≈ 기본 config), HP_SHALLOW=5(하한).
#: (2026-06-02 사용자 B 채택 — on/off 불가 검증 후 depth 로 재정의. 코어 학습코드 무편집.)
PREPROC_SHALLOW = 1
HP_SHALLOW = 5


def cell_env(cell, *, hp_trials: int = 20, preproc_trials: int = 10) -> dict:
    """cell (preproc,hp,feature) 0/1 → 그 cell 의 env override dict. **DEPTH ablation**.

    파이프라인이 preproc·HP 를 항상 최적화(위 PREPROC_SHALLOW/HP_SHALLOW 주석) → 요인을 **깊이**로:
    bit=1=deep, bit=0=shallow.
      • preproc: deep=preproc_trials, shallow=1 (MPH_PREPROC_TRIALS → GLOBAL.training.preproc_trials).
      • HP: deep=hp_trials, shallow=5 (MPH_HP_OPTUNA_TRIALS → GLOBAL.optuna.hp_trials_default).
      • feature: full(STABILITY) vs BASIC(13) (cell-runner 가 MPH_PHASE13_FEATURE_POOL 로 슬라이스).
    env 는 subprocess 의 fresh GLOBAL 생성 시 읽힘 (frozen config 라 in-process 무효).

    Args:
        cell: (p,h,f) ∈ {0,1}³.
        hp_trials: HP **deep** trial 수 (bit=1). shallow=HP_SHALLOW(5) 고정.
        preproc_trials: preproc **deep** trial 수 (bit=1). shallow=PREPROC_SHALLOW(1) 고정.
    Returns:
        env override dict (str→str). caller 가 subprocess env 에 적용.
    """
    p, h, f = cell
    return {
        "MPH_PREPROC_TRIALS": str(preproc_trials if p else PREPROC_SHALLOW),
        "MPH_HP_OPTUNA_TRIALS": str(hp_trials if h else HP_SHALLOW),
        "MPH_PHASE13_FEATURE_POOL": "full" if f else "basic",
    }


def select_panel(champion, category_models, *, n_per_family: int = 1) -> list:
    """champion + family 당 n_per_family 대표 → 패널 (전 family 커버, 중복 제거, 순서 안정).

    champion 은 자기 family 의 대표로 우선 선택 + 어떤 경우든 패널에 포함.

    Args:
        champion: 최종 champion model name (없으면 "" / None).
        category_models: {family: [model, ...]} (registry.CATEGORY_MODELS).
        n_per_family: family 당 대표 수 (기본 1).
    Returns:
        패널 model name 리스트 (≈ family수 × n_per_family + champion).
    """
    panel: list = []
    for _fam, models in category_models.items():
        picks: list = []
        if champion and champion in models:
            picks.append(champion)
        for m in models:
            if len(picks) >= n_per_family:
                break
            if m not in picks:
                picks.append(m)
        for m in picks[:n_per_family]:
            if m not in panel:
                panel.append(m)
    if champion and champion not in panel:
        panel.append(champion)
    return panel


def run_factorial(panel, y_test, *, fit_fn, hp_trials: int = 20, preproc_trials: int = 10) -> dict:
    """각 model 의 8 cell 예측 생성(fit_fn) → factorial_effects (주효과+상호작용).

    Args:
        panel: model name 리스트.
        y_test: 실측 (len n_test).
        fit_fn: (model_name, cell, env) -> test_pred(len n_test). 테스트=mock / 실행=subprocess 수집 래퍼.
        hp_trials/preproc_trials: cell_env budget.
    Returns:
        {model: factorial_effects(...) 결과(main/interactions/n)}. fit_fn 이 None 반환한 cell 은 제외.
    """
    out: dict = {}
    for model in panel:
        cells = {}
        for cell in CELLS:
            pred = fit_fn(model, cell, cell_env(cell, hp_trials=hp_trials, preproc_trials=preproc_trials))
            if pred is not None:
                cells[cell] = pred
        out[model] = factorial_effects(cells, y_test, factors=FACTORS)
    return out


def aggregate_effects(results) -> dict:
    """모델별 factorial → 요인별 평균 effect + 유의 모델 수 (패널 전체 종합).

    Args:
        results: run_factorial 반환 {model: {"main":[...], ...}}.
    Returns:
        {factor: {"mean_effect", "n_sig", "n_models"}}. mean_effect>0 = 평균적으로 그 요인이 개선.
    """
    agg: dict = {}
    for _model, res in results.items():
        for m in (res or {}).get("main", []):
            a = agg.setdefault(m["factor"], {"effects": [], "n_sig": 0})
            a["effects"].append(m["effect"])
            if m.get("sig") == "yes":
                a["n_sig"] += 1
    return {
        f: {
            "mean_effect": (sum(v["effects"]) / len(v["effects"]) if v["effects"] else 0.0),
            "n_sig": v["n_sig"],
            "n_models": len(v["effects"]),
        }
        for f, v in agg.items()
    }


# ════════════════════════════════════════════════════════════════════════════
# 실행 glue — cell당 subprocess 빌딩블록 (env → fresh GLOBAL; frozen config 우회의 유일 경로).
#
# 실제 실행 = 8 cell × {cell_cli_command(panel) + cell_run_env(cell, output_root)} subprocess.
# 각 run = phase-13(per_model_optimize) on 12-panel(--models), MPH_OUTPUT_ROOT 격리 → main run
# 충돌 0. 수집 = 각 cell 의 per_model_optimal/{model}.json 의 refit_test_predictions →
# {model: {cell: pred}} → factorial_effects. 무거운 적합(8 run × panel) 이라 실제 launch +
# 수집 드라이버는 사용자가 "지금-실행(main 과 CPU 경합) vs main 후" 결정 후 빌드/검증.
# ════════════════════════════════════════════════════════════════════════════

def cell_run_env(cell, output_root, *, hp_trials: int = 20, preproc_trials: int = 10,
                 base_env=None) -> dict:
    """cell 을 격리 subprocess 로 재현하는 env dict (cell_env + 출력격리 + 메모리격리).

    Args:
        cell: (p,h,f) ∈ {0,1}³.
        output_root: 이 cell 의 MPH_OUTPUT_ROOT (main run 과 분리 → Optuna/per_model_optimal 충돌 0).
        hp_trials/preproc_trials: on-budget.
        base_env: 시작 env (None → 빈 dict; caller 가 os.environ.copy() 권고 — PATH/venv 보존).
    Returns:
        env dict (str→str). cell 3 토글 + MPH_OUTPUT_ROOT + OPTUNA_ISOLATE=1(G-158) 포함.

    Side effects: none (dict 반환만). caller 가 subprocess(env=...) 로 사용.
    """
    env = dict(base_env or {})
    env.update(cell_env(cell, hp_trials=hp_trials, preproc_trials=preproc_trials))
    env["MPH_OUTPUT_ROOT"] = str(output_root)
    env["OPTUNA_ISOLATE"] = "1"          # G-158 child memory 100% 회수
    env["MPH_MULTI_SEED_RUN"] = "0"      # ★ 단일 seed 강제 — main 의 5-seed(MPH_MULTI_SEED_RUN=1)
                                          #   상속 시 cell 마다 trial 5× 폭증. 요인 ablation 은 단일 seed.
    return env


def cell_cli_command(panel, *, resume: str = "per_model_optimize", scenario: str = "full_light",
                     python_exe: str = ".venv/bin/python") -> list:
    """cell subprocess 의 CLI argv (12-panel 만 --models 로 제한, phase-13 resume).

    Args:
        panel: model 이름 리스트 (select_panel 결과). 빈 리스트면 --models 생략(전체) — 호출 전 검증 권고.
        resume: --resume-from 단계 (의미이름; per_model_optimize=phase 13).
        scenario: --scenario (full_light = 경량 full).
        python_exe: 인터프리터 경로.
    Returns:
        argv 리스트 (subprocess.run 용). env 는 cell_run_env 로 별도 주입.
    """
    cmd = [python_exe, "-m", "simulation", "train",
           "--resume-from", resume, "--scenario", scenario]
    if panel:
        cmd += ["--models", ",".join(panel)]
    return cmd


def _coerce_pred(tp):
    """refit_test_predictions(json) → 1D float array | None. list(정상) + str-repr(default=str) 양쪽."""
    import numpy as np
    if tp is None:
        return None
    if isinstance(tp, str):                  # json.dumps(default=str) 가 np.array 를 str 로
        s = tp.strip().lstrip("[").rstrip("]")
        for sep in (" ", ","):
            try:
                arr = np.fromstring(s, sep=sep)
            except Exception:
                arr = np.empty(0)
            if arr.size:
                return arr
        return None
    try:
        arr = np.asarray(tp, dtype=np.float64).ravel()
        return arr if arr.size else None
    except Exception:
        return None


def collect_cell_predictions(output_root, panel, *, n_test=None) -> dict:
    """한 cell 출력(per_model_optimal/{model}.json)에서 모델별 refit_test_predictions 로드.

    Args:
        output_root: 그 cell 의 MPH_OUTPUT_ROOT (save_dir = output_root/results).
        panel: 수집할 model 이름 리스트.
        n_test: 주어지면 끝 n_test 개만 (test slab 길이 정렬).
    Returns:
        {model: np.ndarray(len n_test)}. 파일없음/예측없음/파싱실패 → 그 model skip.
    """
    import json
    from pathlib import Path
    base = Path(output_root) / "results" / "per_model_optimal"
    out: dict = {}
    for m in panel:
        fp = base / f"{m}.json"
        if not fp.exists():
            continue
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        arr = _coerce_pred(d.get("refit_test_predictions"))
        if arr is not None and arr.size:
            out[m] = arr[-n_test:] if n_test else arr
    return out


def reshape_to_per_model(per_cell, panel) -> dict:
    """{cell: {model: pred}} → {model: {cell: pred}} (factorial_effects 입력 형태로 전치).

    Args:
        per_cell: {cell_tuple: {model: pred_array}} (cell 별 수집).
        panel: model 이름 리스트.
    Returns:
        {model: {cell: pred}}. 예측 0개인 model 은 제외.
    """
    out: dict = {}
    for m in panel:
        cells = {cell: preds[m] for cell, preds in per_cell.items() if m in preds}
        if cells:
            out[m] = cells
    return out
