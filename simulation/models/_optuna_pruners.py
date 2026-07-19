"""Optuna pruner — 모델 카테고리별 최적 선택 (locally-specified per study).

ENGINEERING_PRINCIPLES.md §원칙 #5 (재현성) — 환경변수 의존 X, 각 study 가 직접 호출.
설계 철학: `_optuna_samplers.py` 와 대칭. study 생성 시 명시 호출.

사용:
    from simulation.models._optuna_pruners import get_best_pruner_for, build_pruner

    study = optuna.create_study(
        direction="minimize",
        sampler=get_best_sampler_for(model_name),
        pruner=get_best_pruner_for(model_name),     # ← 명시 호출
    )

OR universal:
    pruner = build_pruner("hyperband", min_resource=1, max_resource=20)

References:
- Li et al. 2017 — Hyperband (대규모 HP search 의 표준)
- Jamieson & Talwalkar 2016 — Successive Halving
- Optuna docs: HyperbandPruner / SuccessiveHalvingPruner / MedianPruner
"""
from __future__ import annotations

from typing import Optional

import optuna


# ════════════════════════════════════════════════════════════════
# 모델 카테고리 → pruner 매핑
# ════════════════════════════════════════════════════════════════
#
# Hyperband (Li 2017) — 대부분 모델의 default. 이론적 보장 있음, 50%+ pruning.
# SuccessiveHalving — Kernel 같이 trial 비용이 큰 경우 (공격적 cut).
# Median — Stat 모델 같이 epoch 개념 없을 때 (intermediate value 단순 비교).
# None — 시계열/메커니즘 모델 (HP grid 가 작아 pruning 무의미).

_PRUNER_BY_MODEL: dict[str, str] = {
    # Tree (XGBoost, LightGBM, RandomForest 등) — Hyperband, fast convergence
    "XGBoost": "hyperband", "LightGBM": "hyperband", "RandomForest": "hyperband",
    "ExtraTrees": "hyperband", "GBM": "hyperband",
    # Linear / GLM — Hyperband, trial 빠르므로 공격적 prune OK
    "ElasticNet": "hyperband", "Ridge": "hyperband", "Lasso": "hyperband",
    "BayesianRidge": "hyperband", "NegBinGLM": "hyperband",
    "PoissonGLM": "hyperband", "QuantileRegressor": "hyperband",
    # Kernel — Successive Halving, kernel fit 비용 큼
    "SVR-RBF": "halving", "SVR-Linear": "halving", "KRR": "halving",
    "GP-RBF": "halving", "GaussianProcess": "halving",
    # DL — Hyperband (default for DL HP search, 이미 _optuna_torch.py 에서 사용)
    "DNN": "hyperband", "TabularDNN": "hyperband", "TabularDNNLite": "hyperband",
    "TinyMLP": "hyperband", "TCN": "hyperband", "LSTM": "hyperband",
    "PatchTST": "hyperband", "iTransformer": "hyperband",
    "Mamba": "hyperband", "TimesNet": "hyperband", "N-BEATS": "hyperband",
    "GE-DNN": "hyperband", "GE-GAT": "hyperband",
    # 시계열 통계 — pruning 무의미 (HP grid 작음, intermediate value 없음)
    "ARIMA": "median", "SARIMA": "median", "SARIMAX": "median",
    # Mechanistic — pruning X (deterministic ODE, HP grid 작음)
    "Bayesian-SEIR": "none", "Metapop-SEIR": "none",
    "SEIRForcedForecaster": "none", "Rt-Augmented": "none",
    "PINN": "none",
    # Foundation — pruning 어려움 (transfer learning epochs 적음)
    # G-261 (2026-06-13): Chronos-T5 제거 — Chronos retire (대체 = TimesFM-2.5 + TiRex).
    "FoundationModelTransfer": "median",
    "TimesFM": "median", "TiRex": "median",
    # Bayesian MCMC — pruning 무의미 (chain convergence)
    "BayesianMCMC": "none", "PoissonAR": "median",
}


def build_pruner(
    name: str,
    min_resource: int = 1,
    max_resource: int = 20,
    reduction_factor: int = 3,
    n_startup_trials: int = 5,
    n_warmup_steps: int = 1,
) -> Optional[optuna.pruners.BasePruner]:
    """이름 → Pruner 인스턴스. None 반환 시 pruning 비활성.

    Args:
        name: "hyperband" | "halving" | "median" | "none"
        min_resource: epochs minimum (Hyperband / Halving)
        max_resource: epochs maximum (Hyperband)
        reduction_factor: 각 단계마다 1/k 통과 (Hyperband / Halving)
        n_startup_trials: Median 의 warmup
        n_warmup_steps: Median 의 step warmup
    """
    name = (name or "hyperband").lower()
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner(
            min_resource=min_resource,
            max_resource=max_resource,
            reduction_factor=reduction_factor,
        )
    if name in ("halving", "successive_halving"):
        return optuna.pruners.SuccessiveHalvingPruner(
            min_resource=min_resource,
            reduction_factor=reduction_factor,
            min_early_stopping_rate=0,
        )
    if name == "median":
        return optuna.pruners.MedianPruner(
            n_startup_trials=n_startup_trials,
            n_warmup_steps=n_warmup_steps,
        )
    if name in ("none", "nop", "null"):
        return optuna.pruners.NopPruner()
    # Unknown → 안전 default
    return optuna.pruners.HyperbandPruner(
        min_resource=min_resource,
        max_resource=max_resource,
        reduction_factor=reduction_factor,
    )


def get_best_pruner_for(
    model_name: str,
    *,
    min_resource: int = 1,
    max_resource: int = 20,
    reduction_factor: int = 3,
) -> optuna.pruners.BasePruner:
    """모델 이름 → 최적 pruner 인스턴스.

    환경변수로 override 가능 (전역 적용):
      MPH_PRUNER=hyperband|halving|median|none

    개별 study 가 호출 시점에 명시적으로 결정 — 전역 의존 X.
    """
    import os as _os

    # 전역 override 우선 (운영 시 toggle 가능)
    override = _os.environ.get("MPH_PRUNER", "").lower().strip()
    if override:
        return build_pruner(
            override,
            min_resource=min_resource,
            max_resource=max_resource,
            reduction_factor=reduction_factor,
        )

    # 모델별 default
    name = _PRUNER_BY_MODEL.get(model_name, "hyperband")
    return build_pruner(
        name,
        min_resource=min_resource,
        max_resource=max_resource,
        reduction_factor=reduction_factor,
    )


def get_pruner_for_stage(
    stage: str,
    *,
    min_resource: int = 1,
    max_resource: int = 20,
) -> optuna.pruners.BasePruner:
    """Stage 1 (preproc) / Stage 2 (feature) / Stage 3 (HP) 별 pruner.

    Stage 1/2 는 trial 수가 적고 (30) max_resource 작음 (epochs 무관) →
    HyperbandPruner with min_resource=1, max_resource=10.
    """
    if stage in ("stage1", "stage2", "preproc", "feature"):
        return build_pruner(
            "hyperband",
            min_resource=min_resource,
            max_resource=10,
            reduction_factor=3,
        )
    # Stage 3 = HP search, model-aware
    return build_pruner("hyperband", min_resource=min_resource, max_resource=max_resource)


__all__ = [
    "build_pruner",
    "get_best_pruner_for",
    "get_pruner_for_stage",
]
