"""simulation/models/_optuna_torch.py ( P0-D)
=========================================================
Shared Optuna search-space helpers for Torch-based forecasters.

Motivation
----------
Before only ``DNN-Optuna`` and ``TCN-Optuna`` exposed their
architecture hyperparameters (layer count, hidden unit, dropout,
activation, norm, init, optimizer, loss) to Optuna. TFT, GE-DNN,
GE-DNN-GAT and every ``modern_ts`` model (PatchTST, N-BEATS, N-HiTS,
TiDE, Mamba, TimesNet, iTransformer) had *all* of these values
hard-coded.

The user asked for Optuna coverage across all torch wrappers with a
hidden-unit range of **2 – 9999 (log scale)**, activation picked from
relu / leaky_relu / gelu / mish, and initialiser from kaiming / xavier
(= glorot_uniform) / default. This module centralises those choices
so each wrapper can opt-in with a tiny ``fit``-time call.

Public API
----------
``suggest_mlp_hp(trial, *, min_unit, max_unit, max_layers)``
 MLP / Linear-stack search space (used by DNN-style and GE-DNN).
``suggest_transformer_hp(trial, *, min_d_model, max_d_model)``
 Transformer search space (d_model / n_heads / n_layers / dim_ff).
``suggest_seq_hp(trial, *, min_hidden, max_hidden)``
 Generic sequential-encoder search space (PatchTST patch params,
 N-BEATS blocks, TiDE hidden, …).
``suggest_training_hp(trial)``
 Shared training knobs (lr, weight_decay, optimizer, loss, augment,
 batch size).
``run_optuna_loop(name, objective, n_trials, default_fn)``
 Wrap ``optuna.create_study(...)``; falls back to ``default_fn``
 when ``n_trials <= 0`` or optuna is not installed.
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Callable, Dict, Optional

log = logging.getLogger(__name__)


def _trial_gpu_cleanup() -> None:
    """FIX: Optuna trial 간 VRAM / 메모리 단편화 방지.

 각 trial 끝에 호출해 파이썬 객체 → CUDA allocator → driver 순서로
 메모리를 회수한다. `del model + gc.collect` 만으로는 CUDA allocator
 의 cached block 이 남아 있어 100 trial 돌리면 6 GB VRAM 이 fragment
 로 가득 차 hidden=9999 같은 큰 trial 이 OOM 을 낸다.

 G-158 (2026-05-02) 강화:
 - gc.collect() 2 회 호출: 첫 번째 호출에서 unreachable cycle 의 __del__
 trigger, 두 번째 호출에서 gc.garbage 청소 (PEP 442 cycle finalizer).
 - macOS Linux malloc_trim(0) 추가: glibc heap arena 회수 (Linux 만).
 - in-process 모드 (OPTUNA_ISOLATE=0) 에서는 이 cleanup 만으로 부족 ->
 subprocess 격리 (OPTUNA_ISOLATE=1) 권장. ENGINEERING_PRINCIPLES.md 원칙 2 참조.
 """
    # gc 2 회 호출 (cyclic reference, PEP 442) - in-process 모드 메모리 누수 완화
    try:
        gc.collect()
    except Exception:
        pass
    try:
        gc.collect()  # G-158: 2nd pass for finalizer-resurrected cycles
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # cuBLAS workspace (stream 별로 256MB 씩 쌓임) 회수
            if hasattr(torch, "_C") and hasattr(torch._C, "_cuda_clearCublasWorkspaces"):
                try:
                    torch._C._cuda_clearCublasWorkspaces()
                except Exception:
                    pass
            # IPC handle (multi-process DataLoader) 회수
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # macOS Apple Silicon: MPS allocator 캐시 회수
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass
    except Exception:
        pass
    # G-158: Linux glibc heap arena 회수 (RSS 가 가상 메모리에 남는 문제)
    # macOS / Windows 는 noop (ctypes 가 libc.so.6 를 못 찾음).
    try:
        import ctypes
        import ctypes.util
        _libc_path = ctypes.util.find_library("c")
        if _libc_path:
            _libc = ctypes.CDLL(_libc_path)
            if hasattr(_libc, "malloc_trim"):
                _libc.malloc_trim(0)
    except Exception:
        pass


def make_trial_cleanup_callback(name: str = "") -> Callable[[Any, Any], None]:
    """Optuna study.optimize(..., callbacks=[...]) 에 꽂을 GPU cleanup 콜백.

    TPE/pruner 가 trial 결과를 기록한 **직후** 호출되므로 objective 끝에
    `del model` 해둔 것과 합쳐 trial 사이에서 VRAM 이 확실히 비워진다.
    """
    def _cb(study: Any, trial: Any) -> None:
        _trial_gpu_cleanup()
    return _cb

# : 사용자 요청 -- unit 탐색 범위는 2 .. 9999 log-scale.
UNIT_MIN_DEFAULT = 2
UNIT_MAX_DEFAULT = 9999

# FIX: memory-heavy TS DL (N-BEATS/N-HiTS/iTransformer/Mamba/TimesNet/TiDE)
# stack hidden across blocks × lookback → hidden=9999 OOM on 6GB VRAM.
# Cap to 512 keeps search diverse but VRAM-safe on RTX 3060 Laptop (6GB).
UNIT_MAX_TS_DL = 512

# FIX (2026-04-22, vanilla-GAT 진단 결과):
#   GATv2Conv 는 edge × heads × hidden² 로 메모리/연산이 폭증하고, torch_geometric
#   구현은 (N, in_dim) 기대이므로 batch 차원은 `_PYGGAT.forward` 에서 Python for-loop
#   로 풀린다. node_hidden 이 ~9000 이면 batch=32 × 6 step × 600 edge × 8 heads 의
#   커널 launch 오버헤드가 누적돼 GE-DNN-GAT retry 가 1200s+ stall 발생.
#   vanilla 진단 (node_hidden=32, heads=2, 25 gu, 600 edge, CPU):
#     fwd 0.44s/ep, bwd 0.34s/ep — 정상. 즉 병목은 HP 상한이지 구조가 아님.
#
# -B (2026-04-22, Iter-11 재실패 후 추가 축소):
#   Iter-11 에서 UNIT_MAX_GAT=256, heads∈{2,4}, ep80, pt15 를 적용했는데도
#   R2 baseline 학습 (구 Phase 2.1) [2/3] 에서 subprocess 실패 (20분 30초 OOM/timeout).
#   → 추가로 UNIT_MAX_GAT=128 로 반감, heads={2} 고정, ep60/pt10 으로 축소하여
#     TPE 가 dead-end trial 을 더 빠르게 잘라내도록 강제.
UNIT_MAX_GAT = 128

# r2 (사용자 지시): activation 은 Optuna 탐색 대신 고정 추천 'gelu'.
# ---------------------------------------------------------------
# 왜 GELU 인가 (소표본 n=337, LayerNorm 사용, tabular+TS 혼합):
#   * Gaussian Error Linear Unit — 모든 점에서 smooth · 미분 가능.
#   * BERT/GPT/LLaMA/ViT 등 LayerNorm + Transformer 조합의 사실상 표준.
#   * ReLU: dead neurons + non-smooth gradient.  LeakyReLU 는 완화하나 piecewise linear.
#   * Mish: GELU 와 거의 동일한 성능, 연산 비용 약 10% 비쌈.
#   * SELU: LeCun init + AlphaDropout + 배치놈 제거 조건 위반 시 오히려 악화.
# → trial 예산을 architecture/dropout/lr 에 몰아줌.
ACTIVATIONS = ["gelu"]
NORMS = ["none", "layer", "batch"]
INITS = ["kaiming", "xavier", "default"]
OPTIMIZERS = ["adamw", "adam", "radam", "rmsprop", "sgd_momentum"]
LOSSES = ["mse", "mae"]  # G-218: huber 영구 제거 (huber-loss-banned-20260520)
BATCH_SIZES = [16, 32, 64]

# r2 (사용자 지시): lr 탐색 범위 1e-5 .. 1e-2 (log-scale).
LR_LOW = 1e-5
LR_HIGH = 1e-2


def suggest_training_hp(trial: Any) -> Dict[str, Any]:
    """lr / weight_decay / optimizer / loss / augment / batch_size.

    r2: lr ∈ [1e-5, 1e-2] log-scale (더 넓은 범위 → TPE 가
    매우 작은 lr + 긴 training 조합도 시도).

    Package I-1 (G-144): augment_factor 0-3 (small data leakage 회피).
    G-231 (2026-05-22): alpha_blend HP 제거 — α-blend 완전 제거.
        이전: Package J (G-141) alpha_blend ∈ [0, 1.5] (Bühlmann 2018)
        현재: DNN 모델이 alpha_blend 를 사용하지 않으므로 Optuna HP 공간에서 제거.
    """
    return {
        "lr": trial.suggest_float("lr", LR_LOW, LR_HIGH, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
        "optimizer": trial.suggest_categorical("optimizer", OPTIMIZERS),
        "loss": trial.suggest_categorical("loss", LOSSES),
        # G-231 + 2026-05-26 archive: PI augmentation permanently disabled.
        # augment_factor is fixed at 0 (was suggest_int with MPH_PI_AUGMENT_LO/HI env vars).
        "augment_factor": 0,
        "batch_size": trial.suggest_categorical("batch_size", BATCH_SIZES),
    }


def suggest_mlp_hp(
    trial: Any,
    *,
    min_unit: int = UNIT_MIN_DEFAULT,
    max_unit: int = UNIT_MAX_DEFAULT,
    max_layers: int = 5,
) -> Dict[str, Any]:
    """Layer count + per-layer (hidden, dropout) + activation/norm/init."""
    n_layers = trial.suggest_int("n_layers", 1, max_layers)
    hidden_dims = []
    dropouts = []
    for i in range(n_layers):
        hidden_dims.append(
            trial.suggest_int(f"hidden_{i}", min_unit, max_unit, log=True)
        )
        dropouts.append(trial.suggest_float(f"dropout_{i}", 0.0, 0.5))

    return {
        "n_layers": n_layers,
        "hidden_dims": hidden_dims,
        "dropouts": dropouts,
        "activation": trial.suggest_categorical("activation", ACTIVATIONS),
        "norm": trial.suggest_categorical("norm", NORMS),
        "init": trial.suggest_categorical("init", INITS),
    }


def suggest_transformer_hp(
    trial: Any,
    *,
    min_d_model: int = 8,
    max_d_model: int = 512,
    max_layers: int = 4,
) -> Dict[str, Any]:
    """d_model / n_heads / n_layers / dim_ff / dropout.

    ``d_model`` must be divisible by ``n_heads``; the helper chooses
    ``n_heads`` from a small set and rounds ``d_model`` up to the
    nearest multiple.
    """
    # log-scale in requested range but capped by UNIT_MAX_DEFAULT for safety
    upper = min(max_d_model, UNIT_MAX_DEFAULT)
    raw = trial.suggest_int("d_model_raw", max(min_d_model, 2), upper, log=True)
    n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
    d_model = max(n_heads, ((raw + n_heads - 1) // n_heads) * n_heads)

    return {
        "d_model": int(d_model),
        "n_heads": int(n_heads),
        "n_layers_tf": trial.suggest_int("n_layers_tf", 1, max_layers),
        "dim_ff": trial.suggest_int("dim_ff", 16, 1024, log=True),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "activation": trial.suggest_categorical("activation", ACTIVATIONS),
        "norm": trial.suggest_categorical("norm", NORMS),
        "init": trial.suggest_categorical("init", INITS),
    }


def suggest_seq_hp(
    trial: Any,
    *,
    min_hidden: int = UNIT_MIN_DEFAULT,
    max_hidden: int = UNIT_MAX_DEFAULT,
    max_layers: int = 4,
) -> Dict[str, Any]:
    """Generic sequential encoder (RNN/TCN/N-BEATS/TiDE) search space."""
    return {
        "hidden": trial.suggest_int("hidden", min_hidden, max_hidden, log=True),
        "n_layers_seq": trial.suggest_int("n_layers_seq", 1, max_layers),
        "dropout": trial.suggest_float("dropout", 0.0, 0.5),
        "activation": trial.suggest_categorical("activation", ACTIVATIONS),
        "norm": trial.suggest_categorical("norm", NORMS),
        "init": trial.suggest_categorical("init", INITS),
    }


def run_optuna_loop(
    name: str,
    objective: Callable[[Any], float],
    n_trials: int,
    default_fn: Callable[[], Any],
    direction: str = "minimize",
    n_startup_trials: int = 5,
    n_warmup_steps: int = 20,
    eval_fn_qual: Optional[str] = None,    # ← 2026-04-26 추가
    param_space: Optional[Callable[[Any], Dict[str, Any]]] = None,  # ← 2026-04-26 추가
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> tuple[Dict[str, Any], float]:
    """Run an Optuna study; fall back to ``default_fn`` if trials<=0.

    Returns
    -------
    (best_params, best_value)
        ``best_params`` is an empty dict when the fallback fires — the
        caller is then expected to use its own defaults.

    OPTUNA_ISOLATE=1 통합 (2026-04-26):
      eval_fn_qual + param_space 가 모두 명시되면, OPTUNA_ISOLATE=1 시 매 trial
      을 subprocess 로 격리해서 메모리 누적 0. 둘 중 하나라도 없으면 기존
      in-process 경로 (back-compat).
    """
    if n_trials is None or n_trials <= 0:
        log.info(f"  [{name}] Optuna disabled (trials={n_trials}); using static defaults")
        default_fn()
        return {}, float("nan")

    try:
        import optuna
    except ImportError:
        log.warning(f"  [{name}] optuna not installed; using static defaults")
        default_fn()
        return {}, float("nan")

    # ── Optuna logging verbosity (OPTUNA_VERBOSE) ──
    import os as _os
    _verb = _os.environ.get("OPTUNA_VERBOSE", "1")    # 학습 default = INFO
    if _verb == "2":
        optuna.logging.set_verbosity(optuna.logging.DEBUG)
    elif _verb == "0":
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    else:
        optuna.logging.set_verbosity(optuna.logging.INFO)

    def _logger(study, trial):
        if trial.value is None:
            return
        try:
            best = study.best_value
        except Exception:
            best = float("inf")
        log.info(
            f"  [{name}] Trial {trial.number:>3d}/{n_trials}: "
            f"value={trial.value:+.4f} (best={best:+.4f})"
        )

    # 2026-04-26: Sampler 통합 — OPTUNA_SAMPLER env 변수로 선택
    #   "best"    → per-model best (get_best_sampler_for(name))
    #   "tpe-mv"  → universal multivariate TPE (권장 default)
    #   "tpe"     → 기존 univariate TPE (back-compat)
    #   "gp"/"cma"/"nsga2"/"qmc"/"random" → build_sampler 의 다른 옵션
    import os as _os
    _sampler_choice = _os.environ.get("OPTUNA_SAMPLER", "tpe").lower()
    try:
        from simulation.models._optuna_samplers import (
            build_sampler, get_best_sampler_for,
        )
        if _sampler_choice == "best":
            _sampler = get_best_sampler_for(name, seed=42,   # G-13F: deterministic HPO
                                                n_startup_trials=n_startup_trials)
            log.info(f"  [{name}] sampler: best per-model "
                      f"({type(_sampler).__name__})")
        elif _sampler_choice in ("tpe", "tpe-mv", "tpe_mv", "gp", "cma",
                                    "cmaes", "nsga2", "qmc", "random"):
            _sampler = build_sampler(_sampler_choice, seed=42,   # G-13F: deterministic HPO
                                        n_startup_trials=n_startup_trials)
            log.info(f"  [{name}] sampler: {_sampler_choice} "
                      f"({type(_sampler).__name__})")
        else:
            _sampler = optuna.samplers.TPESampler(seed=42)
            log.info(f"  [{name}] sampler: tpe (legacy default)")
    except Exception as _se:
        # 새 모듈 import 실패 시 legacy fallback (학습 안전)
        log.debug(f"  [{name}] sampler factory fallback: {_se}")
        _sampler = optuna.samplers.TPESampler(seed=42)

    # ══════════════════════════════════════════════════════════════
    # Storage 통합 (2026-04-27 사용자 요청)
    # ──────────────────────────────────────────────────────────────
    # 환경변수 정책:
    #   MPH_OPTUNA_STORAGE=1 (default ON) → SQLite, 기본은 resume
    #   MPH_OPTUNA_STORAGE=0              → 강제 in-memory (back-compat)
    #   MPH_OPTUNA_FORCE=1                → 기존 study 삭제 후 새로 (force fresh)
    #   MPH_OPTUNA_FORCE=0 (default)      → 있으면 resume, 없으면 새로 (사용자 권장)
    # ══════════════════════════════════════════════════════════════
    from simulation.config_global import GLOBAL as _GCFG  # SSOT (2026-05-28)
    _use_storage = _GCFG.optuna.use_storage
    _force_fresh = _GCFG.optuna.force
    _storage = None
    # Package N (G-143 진짜 fix): study_name 에 phase/fold suffix 추가.
    # 이전: _study_name = f"{name}_v1" → WF-CV 3-fold × R9 per_model_optimize × R11 SHAP 등
    #       multiple stages 가 같은 study 누적 → trial 폭주 (N-BEATS 1360, TCN 960 등)
    # 수정: MPH_OPTUNA_PHASE_TAG 환경변수로 phase/fold 별 study 분리
    _phase_tag = _GCFG.optuna.phase_tag
    # G-14H (2026-06-21, codex): study 이름에 content hash(git commit+dirty+schema) → 코드/환경이
    #   바뀌면 stale warm-start 를 자동 무효화(새 study), 동일 컨텍스트면 안전 재사용(속도). 옛 고정
    #   {name}_v1 은 unseeded·구버전 trial 을 그대로 물려받아 #13 seeding 을 무력화하던 것 차단(#13↔#14).
    from simulation.models._study_ctx import study_ctx_hash as _sctx
    _ctx = _sctx()
    _base = f"{name}_v2_{_phase_tag}" if _phase_tag else f"{name}_v2"
    _study_name = f"{_base}_{_ctx}"
    if _use_storage:
        try:
            from simulation.utils.paths import get_results_dir  # SSOT MPH_OUTPUT_ROOT (2026-05-29)
            _study_db = get_results_dir() / "optuna_study.db"
            _study_db.parent.mkdir(parents=True, exist_ok=True)
            _storage = f"sqlite:///{_study_db}"

            # MPH_OPTUNA_FORCE=1 일 때만 기존 study 삭제 (사용자 권장 동작)
            if _force_fresh:
                try:
                    optuna.delete_study(study_name=_study_name, storage=_storage)
                    log.info(f"  [{name}] 🗑 MPH_OPTUNA_FORCE=1 → 기존 study 삭제, 새로 시작")
                except Exception:
                    pass    # study 가 없으면 자연스럽게 새로 만듦
        except Exception as _ste:
            log.debug(f"  [{name}] storage init failed: {_ste} → in-memory")
            _storage = None

    # 2026-04-28: Pruner 선택 — MPH_PRUNER 환경변수
    #   "hyperband"  (default) — HyperbandPruner (공격적, 빠른 cut, Li 2017)
    #   "median"     — MedianPruner (보수적, 옛 default)
    #   "halving"    — SuccessiveHalvingPruner (가장 공격적)
    # 2026-04-28: default 를 hyperband 로 변경 (15h → 7h 단축 효과)
    _pruner_choice = _GCFG.optuna.pruner_name.lower()
    if _pruner_choice == "median":
        _pruner = optuna.pruners.MedianPruner(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=n_warmup_steps,
        )
    elif _pruner_choice in ("halving", "successive_halving"):
        _pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=1, reduction_factor=4, min_early_stopping_rate=0,
        )
    else:    # default hyperband
        _pruner = optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=20, reduction_factor=3,
        )

    study = optuna.create_study(
        direction=direction,
        sampler=_sampler,
        pruner=_pruner,
        storage=_storage,
        study_name=_study_name if _storage else None,
        load_if_exists=bool(_storage),     # force 가 아니면 항상 resume 시도
    )

    # Package N (G-143): cap=50 (200 → 50, OOM 방지)
    # 이전: cap=200, 사용자가 학습 중단 시 매 호출에 200 추가 → 무한 누적 (N-BEATS 1360 trials)
    # 수정: cap=50 + 재호출 시 existing 만큼 빼고 추가 (max 50 추가)
    # Q23 A (2026-05-03): default 50 → 25 (dl-tabular DNN trial 17-27h → ~50% 단축).
    # env override 가능 (`MPH_OPTUNA_REMAINING_CAP=50` 으로 복원 가능).
    _MAX_PER_CALL = _GCFG.optuna.dnn_remaining_cap
    # existing 고려: target n_trials 에서 existing 빼고 max 50 추가
    if _storage:
        try:
            n_existing_before = len(study.trials)
        except Exception:
            n_existing_before = 0
        _n_target = max(0, int(n_trials) - n_existing_before)
        _n_trials_remaining = min(_MAX_PER_CALL, _n_target)
    else:
        _n_trials_remaining = min(_MAX_PER_CALL, int(n_trials))
    if _storage:
        try:
            n_existing = len(study.trials)
            if n_existing > 0:
                log.info(f"  [{name}] 🔁 existing {n_existing} + {_n_trials_remaining} 추가 "
                         f"(cap={_MAX_PER_CALL})")
            else:
                log.info(f"  [{name}] 🆕 new study (storage={_study_db.name})")
        except Exception:
            pass

    # ── OPTUNA_ISOLATE=1 통합 (2026-04-26) ──
    # Trial 단위 subprocess 격리. eval_fn_qual + param_space 가 모두 명시될 때만 활성.
    # objective 가 closure 인 경우는 자동으로 in-process fallback (back-compat).
    _isolate = (_os.environ.get("OPTUNA_ISOLATE", "0") == "1")
    # 2026-04-27 fix: _n_trials_remaining 사용 (warm-start 시 누적 방지)
    # 이미 target 채웠으면 0 — optimize 건너뛰고 best 만 사용.
    if _n_trials_remaining == 0:
        log.info(f"  [{name}] target trials 이미 도달 — optimize skip")
    elif _isolate and eval_fn_qual and param_space is not None:
        try:
            from simulation.models._optuna_subprocess import optimize_with_isolation
            log.info(f"  [{name}] OPTUNA_ISOLATE=1 → trial 단위 subprocess 격리 활성화")
            optimize_with_isolation(
                study=study,
                eval_fn=lambda p, **k: 0.0,    # not called in subprocess mode
                param_space=param_space,
                n_trials=_n_trials_remaining,
                model_name=name,
                eval_fn_qual=eval_fn_qual,
                timeout_per_trial=600,
                isolate_trials=True,
                extra_kwargs=extra_kwargs or {},
                show_progress=True,
            )
        except Exception as _ie:
            log.warning(f"  [{name}] OPTUNA_ISOLATE 실패 → in-process fallback: {_ie}")
            study.optimize(
                objective, n_trials=_n_trials_remaining, show_progress_bar=False,
                callbacks=[_logger, make_trial_cleanup_callback(name)],
            )
    else:
        if _isolate:
            log.info(f"  [{name}] OPTUNA_ISOLATE=1 ignored (eval_fn_qual/param_space 미제공) → in-process")
        # FIX: trial 간 GPU cleanup 콜백을 기본 장착 (VRAM 단편화 방지)
        study.optimize(
            objective,
            n_trials=_n_trials_remaining,
            show_progress_bar=False,
            callbacks=[_logger, make_trial_cleanup_callback(name)],
        )
    try:
        best_params = dict(study.best_params)
        best_value = float(study.best_value)
    finally:
        # study 객체에도 trial 결과가 매달려 있어 잠재 누수 — 명시 해제
        del study
        _trial_gpu_cleanup()
    return best_params, best_value


__all__ = [
    "UNIT_MIN_DEFAULT",
    "UNIT_MAX_DEFAULT",
    "UNIT_MAX_TS_DL",
    "ACTIVATIONS",
    "NORMS",
    "INITS",
    "OPTIMIZERS",
    "LOSSES",
    "BATCH_SIZES",
    "suggest_training_hp",
    "suggest_mlp_hp",
    "suggest_transformer_hp",
    "suggest_seq_hp",
    "run_optuna_loop",
    "make_trial_cleanup_callback",
    "_trial_gpu_cleanup",
]
