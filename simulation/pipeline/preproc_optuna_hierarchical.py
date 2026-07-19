#!/usr/bin/env python3
"""Hierarchical Preprocessing Optuna — Y + X 4-mode symmetric structure.

사용자 명시 (2026-05-23):
  Y와 X 모두 동일한 4-mode top-level 구조:
    none        → passthrough
    individual  → METRIC 단일 선택 (1개)
    group       → METRIC chain (1-N, 중복 허용) / per-feature-group ColumnTransformer
    categorical → CATEGORICAL 단일 선택 (fitted/heavy)

  Optuna가 단계적으로 conditional HP 탐색:
    Step 1: suggest_categorical("y_mode", ["none","individual","group","categorical"])
    Step 2: 모드에 따라 조건부 세부 선택

설계:
─────────────────────────────────────────────────────────
Y label preproc:
  y_mode = none | individual | group | categorical
    ├─ none          : passthrough (y → y)
    ├─ individual    : METRIC_Y_TRANSFORMS 중 1개 (단일, no chain)
    ├─ group         : METRIC_Y_TRANSFORMS chain (1-N, 중복 허용)
    │                  순차 적용 + 역순 inverse
    └─ categorical   : CATEGORICAL_Y_TRANSFORMS 중 1개 (boxcox/yeo_johnson/gaussian)

X label scaler:
  x_mode = none | individual | group | categorical
    ├─ none          : passthrough (X → X)
    ├─ individual    : METRIC_X_SCALERS 중 1개 (전체 X 동일 적용)
    ├─ group         : per-feature-group ColumnTransformer
    │                  각 group 별 METRIC_X_SCALERS 독립 선택
    └─ categorical   : CATEGORICAL_X_SCALERS 중 1개 (전체 X 동일 적용)

ENGINEERING_PRINCIPLES.md 원칙:
  K-1: silent assumption X — 모든 모드 명시적
  D-1: grill 후 구조 확정 (2026-05-23)
  D-3 TDD: smoke test 포함
  D-4 Deep module: 작은 interface + rich impl
  D-5 Gray-box: 각 함수 contract 명시
"""
from __future__ import annotations

import os
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple

# ════════════════════════════════════════════════════════════════
# Section 1. Constants — transform / scaler lists
# ════════════════════════════════════════════════════════════════

# Y: Metric-style transforms — deterministic, invertible, chain 가능.
# 2026-05-26: anscombe / freeman_tukey / arcsine_sqrt 3종 추가 — Sprint 1.5 audit R1
# (Poisson-VST family from `simulation/models/per_feature_preprocessor.py`,
#  with G-146 inverse caps applied to be consistent with the rest of the family).
METRIC_Y_TRANSFORMS = [
    "log1p", "sqrt", "asinh", "mcmc_robust", "laplace",
    "anscombe", "freeman_tukey",
]
# G-254 (2026-06-12): "rank" + "arcsine_sqrt" removed from the y-TARGET pool — train-bounded.
#   rank's inverse maps predictions back into the sorted TRAIN values, so a tree/additive
#   model is hard-capped at (and, for muted ranks, well below) the train max → it cannot
#   forecast the Seoul ILI peak (test 100.7 vs train ~30) and even mutes the upper-mid range,
#   collapsing LightGBM/CatBoost test r2 to 0.31 / -0.31. arcsine_sqrt clips the target to
#   [0,1], destroying ILI magnitude entirely. Both stay valid as X-feature scalers (bounded
#   scaling is fine there) and their inverse branches are kept for old-artifact replay.
#   Property-guarded by simulation/tests/test_y_transform_extrapolation.py (headroom > 1.10×).

# Y: Categorical transforms — fitted state heavy, single-use only
CATEGORICAL_Y_TRANSFORMS = [
    "boxcox", "yeo_johnson",   # G-254: "gaussian" removed — QuantileTransformer→normal is
                               # bounded to the train empirical CDF (headroom 1.00×), the same
                               # peak-capping failure as rank.
]

# ── Stable R9 (per_model_optimize) search space (2026-06-16): paper-grade reproducibility ───
# MPH_STABLE_TRANSFORMS=1 was already the production/preflight default, but hierarchical
# Optuna could still reach y group/categorical modes and the divergent metric transforms.
# In stable mode, all models use the same small y space:
#   identity (via y_mode="none") + log1p/sqrt/asinh/laplace (via y_mode="individual").
# The wider primitives stay implemented for old-artifact inverse/replay and explicit
# MPH_STABLE_TRANSFORMS=0 experiments, but they are not sampled in the stable path.
# G-329 (2026-06-20, Y-transform A/B 권위 split n=337 실측): affine-safe 군만 stable 에 둠.
#   laplace·mcmc_robust = affine(x·s+m, 역변환 폭발 원천 불가) + A/B서 identity 와 ties/beats.
#   제거: asinh·log1p = free-form regressor(linear/NN) catastrophic(DNN asinh −2.14·BayesianRidge log1p
#   −3.75, G-328 cap 있어도 amplifying), sqrt = risky(BayesianRidge 0.469·TabularDNN 0.211). identity 는
#   y_mode="none" 으로 항상 가용. → 어떤 모델이 무엇을 골라도 폭발/catastrophe 0 → extrapolation_safe
#   force-identity(G-328) 불요(되돌림). (boxcox/yeo/anscombe/freeman_tukey 는 별개로 비-stable.)
# G-330 (2026-06-20, 49-모델 변환 audit + 3AI 워크플로 + 사용자 "동적으로"): G-329 전역 affine-only
#   복원. audit(19 non-deep × 6변환 × in-range R² @ R9 feature)이 입증: ① R9 feature 개수에선 어떤
#   변환도 폭발 안 함(DNN log1p max159 < 3×peak=303). 예전 "log1p 폭발"=full-feature 과적합 artifact.
#   ② OOF-CV는 in-range fold = 사용자 기준 → 모델별 in-range 최고 변환 동적 자동선택(NegBinGLM→sqrt .739,
#   BayesianRidge→laplace[log1p .363 자동회피], robust 9종→best). 전역 affine-only 가 NegBinGLM의
#   OOF-selected sqrt/log1p 를 막아 identity −1.07 로 붕괴시킨 게 버그. → 전 변환 개방, OOF 가 모델마다
#   동적 선택. 폭발 backstop = G-328 cap(역변환 bounded) + _sanity_penalize_wis. identity=y_mode="none".
# G-333 (2026-06-22, flat-grid 재설계 + SCI 리서치): 5→6 확장(+identity=7 flat grid). fourth_root
#   (France 2022 Taylor's Power Law 과분산 VST, capped:285 패턴 — rate 적합) 추가. ★anscombe/freeman_tukey
#   는 **제외 유지**: SCI-grade이나 Poisson **count 전용**이라 ILI=rate 엔 부적절(rate 를 count 로 취급해야
#   동작) + G-256 불변식(test_g256:78,85)이 stable pool 서 금지. boxcox/yeo(λ train-fit)·logit·rank·gaussian
#   = OOD train-bounded 라 제외. identity 는 y_mode="none". _NONCENTERED_STABLE_Y·LINEAR_INVERSE alias 자동반영.
STABLE_Y_TRANSFORMS = ["log1p", "sqrt", "fourth_root", "asinh", "laplace", "mcmc_robust"]

# G-334 (2026-06-22): inverse cap 의 y_max 기준을 fold-불변 전역 참조로 통일. OOF(_oof_cv_wis_hier)가
#   fold 마다 preproc 를 replay하면서 cap=10×fold_y_max → 작은 fold 에서 cap 이 작아져 asinh(sinh 역변환)
#   의 상위 PI 분위를 잘라 OOF-WIS 부풀림(asinh OOF 1.901→4.344 회귀, 06-18 대비). set_y_ref_max(전체
#   train max)로 모든 fold·모델·transform 에 동일 cap → fold 편향 0(통일성, 사용자 지시). full-train refit
#   은 이미 full y.max() = 같은 값. 미설정(None) 시 y.max() fallback(back-compat).
_Y_REF_MAX = None


def set_y_ref_max(v) -> None:
    """Fold-불변 inverse-cap 기준 y_max 설정(전체 train max). None = 해제(fold-local fallback)."""
    global _Y_REF_MAX
    try:
        _Y_REF_MAX = float(v) if (v is not None and np.isfinite(float(v))) else None
    except (TypeError, ValueError):
        _Y_REF_MAX = None


def _cap_base(y) -> float:
    """Inverse cap 의 기준 y_max: 전역 참조(fold-불변) 우선, 없으면 fold-local(back-compat)."""
    if _Y_REF_MAX is not None:
        return _Y_REF_MAX
    _ya = np.asarray(y, dtype=np.float64)
    return float(np.maximum(_ya, 0).max()) if np.size(_ya) else 0.0
STABLE_PREPROC_MODES = ["none", "individual"]
# 2026-06-16 (사용자: X/Y 별개 접근): X 스케일링은 역변환 없음=발산 위험 0, 그룹별 도메인-인지가
#   가치(TabPFN x-group→testR²0.906). 단 3²⁰ Optuna 탐색이 불안정 원인. → X stable 모드는
#   {none, group} 만 두되, group 의 per-group scaler 를 Optuna 탐색이 아니라 데이터-기반(그룹 분포
#   통계)으로 결정적 선택(아래 data_driven_group_scalers). 도메인-인지 유지 + 재현성 + 안정.
STABLE_X_MODES = ["none", "group"]


def _stable_preproc_space_enabled() -> bool:
    """True for the production thesis/paper preprocessing search space."""
    return os.environ.get("MPH_STABLE_TRANSFORMS", "1").strip().lower() not in {
        "0", "false", "no", "off",
    }


def _choices_for_trial(
    trial: Any,
    param_name: str,
    stable_choices: List[str],
    full_choices: List[str],
    *,
    force_stable: bool = False,
) -> List[str]:
    """Return stable choices for new trials, but allow replay of frozen legacy params."""
    try:
        params = getattr(trial, "params", None)
        if not isinstance(params, dict):
            params = getattr(trial, "_params", {})
        frozen_value = params.get(param_name)
        if frozen_value is None:
            frozen_value = getattr(trial, "_params", {}).get(param_name)
    except Exception:
        frozen_value = None
    if frozen_value in full_choices and frozen_value not in stable_choices:
        return full_choices
    return stable_choices if (force_stable or _stable_preproc_space_enabled()) else full_choices


# ── G-256 (2026-06-12): model-family-aware y-transform safety ────────────────
# The env name MPH_LINEAR_INVERSE_MODELS is retained for compatibility with earlier DL
# allow-lists, but the active stable pool is now the paper-grade set above, not the old
# affine-only mcmc_robust/laplace pair. MPH_STABLE_TRANSFORMS=1 applies that pool to all
# models; the per-model env allow-list remains a secondary backstop if stable mode is off.
LINEAR_INVERSE_Y_TRANSFORMS = STABLE_Y_TRANSFORMS  # legacy env name; identity via y_mode="none"

# Model categories (registry.CATEGORY_MODELS keys) whose members can extrapolate freely in
# transformed space → were historically restricted via MPH_LINEAR_INVERSE_MODELS.
# Free-form regressors: linear/GLM, neural-tabular, modern-ts transformers, GAM/Bayesian,
# graph neural nets (GAT/GCN), and foundation models (TimesFM/TiRex). Trees/kernel-RBF/KNN/
# statistical-ts/epi cap or mean-revert, so they keep the full VST pool (no blow-up risk).
_EXTRAPOLATION_RESTRICT_FAMILIES = frozenset({
    "linear", "dl-tabular", "modern-ts", "other", "graph", "foundation",
})


def model_needs_linear_inverse_y(model_name: Optional[str]) -> bool:
    """True if ``model_name`` belongs to a free-form-regressor family (linear/neural/GAM) that
    must avoid nonlinear-inverse y-transforms (else expm1-style inverse blow-up on peaks).

    Args:
        model_name: registry model name (e.g. "TabularDNN", "ElasticNet", "XGBoost"). ``None``
            or an unknown name → ``False`` (permissive, tree-like default; trees cap anyway).

    Returns:
        ``True`` for linear / dl-tabular / modern-ts / other families; ``False`` otherwise.
    """
    if not model_name:
        return False
    try:
        from simulation.models.registry import CATEGORY_MODELS
    except Exception:
        return False
    for family, members in CATEGORY_MODELS.items():
        if model_name in members:
            return family in _EXTRAPOLATION_RESTRICT_FAMILIES
    return False


# ── G-311 (2026-06-18): models whose internal logic resolves specific features BY NAME ──
# OverseasTransfer's transfer encoder reads ili_rate_lag1-4 columns by name to build the ILI
# sequence fed to the frozen overseas-pretrained LSTM (overseas_transfer.py:656-668). mc=pca
# renames every feature to anonymous PCs → the name lookup fails → transfer silently degrades
# to feature-only (the "slow phantom"). Such models must keep features NAMED (mc forced "none").
_NAMED_FEATURE_MODELS = frozenset({"OverseasTransfer"})


def model_requires_named_features(model_name: Optional[str]) -> bool:
    """True if ``model_name`` resolves specific feature columns BY NAME internally.

    Args:
        model_name: registry model name (e.g. "OverseasTransfer"). ``None``/unknown → ``False``.

    Returns:
        ``True`` for models whose fit/predict looks up named columns (e.g. ``ili_rate_lag1-4``);
        these must run with a name-preserving multicollinearity method (``"none"``), since
        ``pca`` anonymises columns to PCs and breaks the lookup.
    """
    return bool(model_name) and model_name in _NAMED_FEATURE_MODELS


# ── G-300 (2026-06-17, per-model 감사): models that apply their OWN y-transform internally ──
# The 4 log-link GLMs (NB/Poisson family g(μ)=log μ) and the 3 pytorch-forecasting wrappers
# (Pf{NBeats,NHiTS,TiDE}, conditional log1p when y≥0) re-transform y a SECOND time on top of a
# R9 (per_model_optimize) nonlinear y-transform (log1p∘log1p / log-link-on-log1p'd-y). The round-trip ~cancels
# but the internal cap is computed on already-transformed y (mis-scaled) and the OOF selection
# scores a mis-specified likelihood. → feed these models RAW y (R9 y_mode forced to "none");
# each then applies its single intended internal transform. Verified the internal transforms exist
# (epi_models/negbin_glm GLM log-link + family; pf_models.py:183 conditional log1p).
_INTERNAL_Y_TRANSFORM_MODELS = frozenset({
    # G-330 (2026-06-20, 49-모델 변환 audit 실증으로 축소): force-identity 는 "비-identity 가 진짜
    #   catastrophic" = 내부 log-link 이중적용으로 OOF가 골라도 무조건 망가지는 것만 유지.
    # transform-fix (2026-06-21) ★ PART C — no-op blocker: PoissonAutoreg / N-BEATS / N-HiTS / TiDE
    #   REMOVED. Their internal y-transform was un-hardcoded in PART A (PoissonAutoreg np.log AR link
    #   removed → Ridge-AR on raw y; pf _PfBase softplus/log1p coupling removed → transformation=None).
    #   They now carry NO internal transform, so keeping them here would force y_mode="none" and make
    #   PART A a NO-OP (the preproc transform search could never run for them). Removing them lets the
    #   data-driven OOF select the single y-transform per model. NegBinGLM / SARIMA / GAM-Spline are
    #   already absent (un-META'd / un-hardcoded). hhh4-equivalent STAYS: its integer-rounded
    #   (np.round) NB log-link genuinely breaks under an external y-transform (audit: 비-id −78.0).
    "hhh4-equivalent",
})


def model_applies_internal_y_transform(model_name: Optional[str]) -> bool:
    """True if ``model_name`` re-transforms y itself → R9 (per_model_optimize) must feed RAW y (y_mode="none")
    to avoid a double y-transform (G-300). False for all other models."""
    return bool(model_name) and model_name in _INTERNAL_Y_TRANSFORM_MODELS


# ── G-303 (2026-06-17, 검증 적발): models that floor predictions at 0 in their OWN (R9
# per_model_optimize TRANSFORMED) output space — np.maximum(scaler_y.inverse(pred), 0). That floor is correct in
# original units for direct use + MONOTONE transforms (log1p/sqrt/asinh/identity: transformed-0 ≈
# original-0) but WRONG under MEDIAN-CENTERED transforms (laplace/mcmc_robust: transformed-0 =
# median → floors sub-median to the median = quiet-season upward bias). The floor must stay
# (G-275 direct-use contract; removing it broke test_g275_linear_cap on −47.8 linear extrapolation),
# so instead we exclude the 2 centered transforms from these models' y-search. Linear/kernel models
# are ~shift-invariant (intercept/rho absorbs centering) so centered transforms add ~nothing anyway.
_NONCENTERED_STABLE_Y = [t for t in STABLE_Y_TRANSFORMS if t not in ("laplace", "mcmc_robust")]
_TRANSFORMED_ZERO_FLOOR_MODELS = frozenset({
    "SVR-Linear", "SVR-RBF", "ElasticNet", "KRR", "BayesianRidge",
    # G-319e (2026-06-19, 전체 라인업 감사): custom modern-ts 도 predict 에서 scaler_y.inverse →
    #   np.maximum(0) = runner-transformed(T(y)) 공간 floor. T=laplace/mcmc_robust(median-centered)
    #   면 T(y)=0=median → sub-median 절단 = 비수기 상향 bias(Mamba 실측). 이 3개도 centered transform
    #   제외. (N-BEATS/N-HiTS/TiDE 는 pf=_INTERNAL_Y_TRANSFORM 라 별도.)
    "PatchTST", "iTransformer", "Mamba",
})


def model_floors_at_transformed_zero(model_name: Optional[str]) -> bool:
    """True if ``model_name`` applies an in-model 0-floor in R9 (per_model_optimize) transformed space → its
    y-search must exclude median-centered transforms (G-303). False for all other models."""
    return bool(model_name) and model_name in _TRANSFORMED_ZERO_FLOOR_MODELS

# X: Metric scalers — per-group ColumnTransformer 및 individual 용
METRIC_X_SCALERS = ["standard", "robust", "quantile"]

# X: Categorical scalers — 전체 X 단일 적용 (grouped 포함)
CATEGORICAL_X_SCALERS = ["standard", "robust", "quantile", "grouped"]


# ════════════════════════════════════════════════════════════════
# Section 2. Single-transform primitives (Y + X)
# ════════════════════════════════════════════════════════════════

def _apply_single_y_transform(y: np.ndarray, name: str) -> Tuple[np.ndarray, Callable, Dict]:
    """Single Y transform — returns (transformed, inverse_func, state).

    Args:
        y: 1D array of Y values
        name: transform name from METRIC_Y_TRANSFORMS or CATEGORICAL_Y_TRANSFORMS

    Returns:
        (y_transformed, inverse_fn, transform_state_dict)

    Performance: O(n) for monotonic, O(n log n) for rank/quantile
    Side effects: None (pure function)
    """
    y = np.asarray(y, dtype=np.float64)

    if name == "identity":
        return y.copy(), lambda x: np.asarray(x), {}

    if name == "log1p":
        _y_max = _cap_base(y)
        _safe_hi = float(np.log1p(max(_y_max * 10.0, 100.0)))
        return (np.log1p(np.maximum(y, 0)),
                lambda x, hi=_safe_hi: np.clip(np.expm1(np.clip(np.asarray(x), -2, hi)), 0, None),
                {"safe_hi": _safe_hi})

    if name == "sqrt":
        # 2026-06-16 (3자 감사): inverse=x² 가 STABLE_Y 중 유일하게 input clip·output cap 부재
        #   → peak 외삽서 2차 발산(z=50→2500). log1p(G-146) 와 대칭으로 하드캡(10×y_max).
        _y_max_sq = _cap_base(y)
        _cap_sq = max(_y_max_sq * 10.0, 100.0)
        return (np.sqrt(np.maximum(y, 0)),
                lambda x, c=_cap_sq: np.clip(np.maximum(np.asarray(x), 0) ** 2, 0, c),
                {"safe_cap": _cap_sq})

    if name == "asinh":
        # G-328 (2026-06-20, 패널 만장일치): inverse=sinh 가 STABLE_Y 중 유일하게 data-driven
        #   output cap 부재였음 — input clip(±10)은 sinh(10)≈11013 이라 ILI(~100) 규모서 무력.
        #   log1p(267)·sqrt(276) 와 대칭으로 하드캡(10×y_max) + 입력을 asinh(cap) 으로 제한.
        _y_max_as = _cap_base(y)
        _cap_as = max(_y_max_as * 10.0, 100.0)
        _hi_as = float(np.arcsinh(_cap_as))
        return (np.arcsinh(y),
                lambda x, hi=_hi_as, c=_cap_as: np.clip(
                    np.sinh(np.clip(np.asarray(x), -hi, hi)), 0, c),
                {"safe_cap": _cap_as})

    if name == "fourth_root":
        # G-333 (2026-06-22, flat-grid 재설계 + SCI 리서치): Taylor's Power Law 과분산 VST
        #   (France 2022) — sqrt(b=1)·log1p(b=2) 사이 분산-안정화 강도. inverse=t⁴ 는 sqrt(t²)보다
        #   훨씬 빨리 발산하므로 sqrt(:284)·asinh(:293) 와 대칭 하드캡(10×y_max) + 입력 t∈[0, cap^0.25]
        #   제한 필수(codex 검토). 음수 입력은 t⁴ NaN 이라 0-clip.
        _y_max_4r = _cap_base(y)
        _cap_4r = max(_y_max_4r * 10.0, 100.0)
        _hi_4r = float(_cap_4r ** 0.25)
        return (np.power(np.maximum(y, 0.0), 0.25),
                lambda x, hi=_hi_4r, c=_cap_4r: np.clip(
                    np.power(np.clip(np.asarray(x, dtype=np.float64), 0.0, hi), 4.0), 0, c),
                {"safe_cap": _cap_4r})

    if name == "rank":
        from scipy.stats import rankdata
        ranks = rankdata(y) / len(y)
        y_sorted = np.sort(y)
        n = len(y)
        top_slope = max(float(y_sorted[-1] - y_sorted[-2]) * n, 1e-6) if n >= 2 else 1.0
        bot_slope = max(float(y_sorted[1] - y_sorted[0]) * n, 1e-6) if n >= 2 else 1.0

        def _inv_rank(x, ys=y_sorted, n=n, top_slope=top_slope, bot_slope=bot_slope):
            x = np.asarray(x, dtype=np.float64)
            y_out = np.empty_like(x)
            in_range = (x >= 1.0 / n) & (x <= 1.0)
            x_in = np.clip(x[in_range], 1.0 / n, 1.0)
            idx = np.clip((x_in * n).astype(int) - 1, 0, n - 1)
            y_out[in_range] = ys[idx]
            above = x > 1.0
            y_out[above] = ys[-1] + (x[above] - 1.0) * top_slope
            below = x < 1.0 / n
            y_out[below] = ys[0] - (1.0 / n - x[below]) * bot_slope
            return y_out

        return (ranks, _inv_rank, {
            "sorted_y": y_sorted.tolist(),
            "top_slope": float(top_slope),
            "bot_slope": float(bot_slope),
            "n_train": int(n),
        })

    if name == "boxcox":
        from scipy.stats import boxcox
        try:
            yt, lam = boxcox(np.maximum(y, 1e-3))
            _y_train_max = _cap_base(y)
            _safe_cap = max(_y_train_max * 10.0, 100.0)
            if abs(lam) > 1e-6:
                def inv(x, lam=lam, safe_cap=_safe_cap):
                    x = np.asarray(x, dtype=np.float64)
                    raw = np.power(np.maximum(x * lam + 1, 1e-8), 1.0 / lam)
                    return np.clip(raw, 0, safe_cap)
            else:
                def inv(x, safe_cap=_safe_cap):
                    x = np.asarray(x, dtype=np.float64)
                    return np.clip(np.exp(x), 0, safe_cap)
            return yt, inv, {"lambda": float(lam), "y_train_max": _y_train_max, "safe_cap": _safe_cap}
        except Exception:
            return y.copy(), lambda x: np.asarray(x), {}

    if name == "yeo_johnson":
        from sklearn.preprocessing import PowerTransformer
        pt = PowerTransformer(method="yeo-johnson", standardize=False)
        yt = pt.fit_transform(y.reshape(-1, 1)).ravel()
        _y_train_max = float(np.max(y))
        _safe_cap = max(_y_train_max * 10.0, 100.0)

        def inv_yj(x, pt=pt, safe_cap=_safe_cap):
            x = np.asarray(x, dtype=np.float64)
            raw = pt.inverse_transform(x.reshape(-1, 1)).ravel()
            return np.clip(raw, 0, safe_cap)
        return yt, inv_yj, {"power_transformer": pt, "y_train_max": _y_train_max, "safe_cap": _safe_cap}

    if name == "gaussian":
        from sklearn.preprocessing import QuantileTransformer
        qt = QuantileTransformer(n_quantiles=min(100, len(y)),
                                  output_distribution="normal",
                                  random_state=42)
        yt = qt.fit_transform(y.reshape(-1, 1)).ravel()
        _y_train_max = float(np.max(y))
        _safe_cap = max(_y_train_max * 10.0, 100.0)

        def inv_g(x, qt=qt, safe_cap=_safe_cap):
            x = np.asarray(x, dtype=np.float64)
            raw = qt.inverse_transform(x.reshape(-1, 1)).ravel()
            return np.clip(raw, 0, safe_cap)
        return yt, inv_g, {"quantile_transformer": qt, "y_train_max": _y_train_max, "safe_cap": _safe_cap}

    if name == "mcmc_robust":
        med = float(np.median(y))
        mad = float(np.median(np.abs(y - med)))
        scale = max(mad * 1.4826, 1e-6)
        return ((y - med) / scale,
                lambda x, m=med, s=scale: np.asarray(x) * s + m,
                {"median": med, "mad_scale": scale})

    if name == "laplace":
        med = float(np.median(y))
        mad = float(np.median(np.abs(y - med)))
        b = max(mad / 0.6745, 1e-6)
        return ((y - med) / b,
                lambda x, m=med, b=b: np.asarray(x) * b + m,
                {"median": med, "laplace_b": b})

    # ── Poisson-VST family (2026-05-26 — Sprint 1.5 R1, from per_feature_preprocessor) ──
    if name == "anscombe":
        # Anscombe (1948): 2*sqrt(x + 3/8). Variance-stabilizing for Poisson with mean > 1.
        # Inverse: max((z/2)² - 3/8, 0), with G-146 safe_cap (max(y_max×10, 100)) to
        # prevent divergence on extrapolated z.
        _y_max = float(np.maximum(y, 0).max())
        _safe_cap = max(_y_max * 10.0, 100.0)
        yt = 2.0 * np.sqrt(np.maximum(y, 0.0) + 0.375)
        return (yt,
                lambda x, cap=_safe_cap: np.clip(
                    np.maximum((np.asarray(x, dtype=np.float64) / 2.0) ** 2 - 0.375, 0.0),
                    0, cap,
                ),
                {"safe_cap": _safe_cap})

    if name == "freeman_tukey":
        # Freeman-Tukey (1950): sqrt(x) + sqrt(x+1). VST for low-mean Poisson.
        # Inverse: ((z² - 1) / (2z))² — singular near z=0, hence the 1e-9 guard.
        _y_max = float(np.maximum(y, 0).max())
        _safe_cap = max(_y_max * 10.0, 100.0)
        x = np.maximum(y, 0.0)
        yt = np.sqrt(x) + np.sqrt(x + 1.0)
        def _inv_ft(z, cap=_safe_cap):
            z = np.asarray(z, dtype=np.float64)
            return np.clip(
                np.maximum(((z * z - 1.0) / (2.0 * np.maximum(z, 1e-9))) ** 2, 0.0),
                0, cap,
            )
        return yt, _inv_ft, {"safe_cap": _safe_cap}

    if name == "arcsine_sqrt":
        # 2 * arcsin(sqrt(p)) — VST for proportions p ∈ [0, 1].
        # Input clipped to [0, 1] (caller responsibility, but defensive here too).
        # Inverse: sin(z/2)² — range [0, 1] naturally bounded.
        x = np.clip(np.asarray(y, dtype=np.float64), 0.0, 1.0)
        yt = 2.0 * np.arcsin(np.sqrt(x))
        return (yt,
                lambda z: np.clip(np.sin(np.asarray(z, dtype=np.float64) / 2.0) ** 2, 0.0, 1.0),
                {"output_range": [0.0, 1.0]})

    raise ValueError(f"Unknown Y transform: {name}")


def _categorize_feature_groups(feat_names: list) -> dict:
    """Feature name → group index dict (Sprint 1.5 R2, 2026-05-26).

    19-bucket classifier promoted from ``simulation.models.grouped_preprocessor.
    classify_feature``. Hierarchical is the single source of truth; R9 (per_model_optimize)
    and grouped_preprocessor both call this helper. Empty groups are dropped
    so the returned dict only contains buckets with at least one column.

    Args:
        feat_names: ordered list of feature names matching X column order.

    Returns:
        ``{group_name: [column_index, ...]}`` — 18 group names possible (the
        19th, "other", is included only when nothing else matches).

    Groups (priority order, first match wins):
        advanced       — IMF / Hilbert / perment / catch22 / takens / RQA / SAX
        cyclic         — sin_/cos_/season_
        spectral       — wavelet / fft / spectral
        discrete       — qbin / qnorm / _bit
        lag_ili        — ili_rate_lag / ili_lag / ili_rate_l
        rmean          — rolling mean/std/min/max + ili_/ari_ derivatives
        weather        — temp_/ta_/humidity/rainfall/rn_/pressure/wind/sunshine
        fcst_weather   — fcst_*
        disease_count  — dis_/sari_/hfmd/enterovirus/disease
        mobility_rt    — rt_*
        mobility       — pop_/subway_/bus_/commute/inflow/hourly_pop/...
        search_trend   — gt_*
        vaccine        — vax/vacc
        health_resource — hospital_*
        claims         — hira_*
        epi_indicator  — mr_*
        binary         — closure/above_thr/consec_rise/school_/*_flag/...
        composite      — comp_/_x_
        other          — everything else (fallback)
    """
    groups: dict[str, list[int]] = {}
    for i, name in enumerate(feat_names):
        groups.setdefault(_classify_single_feature(name), []).append(i)
    return {k: v for k, v in groups.items() if v}


def _classify_single_feature(name: str) -> str:
    """Single feature name → group label (priority-ordered)."""
    cl = name.lower()

    # 1) 가장 좁은 매칭부터 (priority order)
    if any(x in cl for x in ("_imf", "hilbert_amp", "hilbert_phase", "hilbert_freq",
                              "perment", "spec_ent", "fft_slope", "hjorth_",
                              "takens_", "rqa_", "catch22_", "quantum_",
                              "stl_trend", "stl_seasonal", "stl_resid",
                              "savgol_", "hampel_", "sax_", "paa_")):
        return "advanced"
    if "sin_" in cl or "cos_" in cl or "season_" in cl:
        return "cyclic"
    if "wavelet" in cl or "fft" in cl or "spectral" in cl:
        return "spectral"
    if "qbin" in cl or "qnorm" in cl or "_bit" in cl:
        return "discrete"

    # 2) ILI 파생 (rmean before lag_ili 분기 — rmean 도 ili_ 시작 가능)
    if "ili_rate_lag" in cl or "ili_lag" in cl or "ili_rate_l" in cl:
        return "lag_ili"
    if "rmean" in cl or "rstd" in cl or "rmin" in cl or "rmax" in cl:
        return "rmean"
    if cl.startswith("ili_") or cl.startswith("ari_"):
        return "rmean"

    # 3) 기상 (관측 + 예보)
    if any(p in cl for p in ("temp_", "ta_", "humidity", "rainfall", "rn_",
                              "pressure", "wind", "sunshine", "cold_", "humid_")):
        return "weather"
    if cl.startswith("fcst_"):
        return "fcst_weather"

    # 4) 질병 카운트
    if cl.startswith("dis_") or "disease" in cl or cl.startswith("sari_") \
            or cl.startswith("hfmd") or cl.startswith("enterovirus"):
        return "disease_count"

    # 5) 인구이동 — 실시간 vs 통상
    if cl.startswith("rt_"):
        return "mobility_rt"
    if any(p in cl for p in ("pop_", "subway_", "bus_", "commute", "inflow",
                              "hourly_pop", "_metro", "_traffic", "sub_",
                              "dong_", "hotspot_", "emp_")):
        return "mobility"

    # 6) 검색트렌드
    if cl.startswith("gt_"):
        return "search_trend"

    # 7) 백신
    if "vax" in cl or "vacc" in cl:
        return "vaccine"

    # 8) 의료자원
    if cl.startswith("hospital_"):
        return "health_resource"

    # 9) HIRA claims
    if cl.startswith("hira_"):
        return "claims"

    # 10) Mortality rate
    if cl.startswith("mr_"):
        return "epi_indicator"

    # 11) Binary flag
    if any(p in cl for p in ("closure", "above_thr", "consec_rise", "school_",
                              "_flag", "is_", "binary", "_indicator", "_era")):
        return "binary"

    # 12) Composite interaction
    if "comp_" in cl or "_x_" in cl:
        return "composite"

    return "other"


def _build_single_x_scaler(name: str):
    """Single X scaler — returns sklearn-compatible scaler (or None for passthrough).

    Args:
        name: from METRIC_X_SCALERS or CATEGORICAL_X_SCALERS

    Returns:
        sklearn scaler with .fit_transform()/.transform(), or None.
    """
    if name == "none":
        return None
    if name == "standard":
        from sklearn.preprocessing import StandardScaler
        return StandardScaler()
    if name == "robust":
        from sklearn.preprocessing import RobustScaler
        return RobustScaler()
    if name == "quantile":
        from sklearn.preprocessing import QuantileTransformer
        return QuantileTransformer(n_quantiles=100, output_distribution="normal",
                                    random_state=42)
    if name == "grouped":
        from sklearn.preprocessing import RobustScaler
        return RobustScaler()
    raise ValueError(f"Unknown X scaler: {name}")


# ════════════════════════════════════════════════════════════════
# Section 3. Y hierarchical preproc — 4-mode symmetric
# ════════════════════════════════════════════════════════════════

def suggest_y_preproc(
    trial: Any,
    y_train: np.ndarray,
    max_chain_length: int = 2,
    extrapolation_safe: bool = False,
    force_y_identity: bool = False,
    restrict_centered_y: bool = False,
) -> Tuple[np.ndarray, Callable, Dict]:
    """Hierarchical Y preprocessing — 4-mode Optuna selection.

    Modes (top-level, symmetric with suggest_x_scaler):
        none        → passthrough
        individual  → METRIC_Y_TRANSFORMS 중 1개 (단일, no chain)
        group       → METRIC_Y_TRANSFORMS chain (1-N, 중복 허용)
        categorical → CATEGORICAL_Y_TRANSFORMS 중 1개 (fitted: boxcox/yeo_johnson)

    Stable R9 (per_model_optimize) mode (``MPH_STABLE_TRANSFORMS=1``, the production default) restricts
    every model to ``none`` (identity) or one stable individual transform
    (log1p/sqrt/asinh/laplace). ``extrapolation_safe=True`` preserves the old
    MPH_LINEAR_INVERSE_MODELS call pattern as a secondary backstop. Group/categorical
    branches remain only for old frozen-param replay or explicit stable-mode opt-out.

    Search space (max_chain_length=2):
        none:       1
        individual: 7  (METRIC_Y_TRANSFORMS — G-254: rank/arcsine_sqrt 제외)
        group:      7 + 49 = 56  (chain len 1 + len 2, 중복 허용)
        categorical: 2  (boxcox/yeo_johnson — G-254: gaussian 제외)
        total: ~66 configs

    Args:
        trial: Optuna trial object
        y_train: 1D training Y (original units)
        max_chain_length: group chain 최대 길이 (default 2)

    Returns:
        (y_transformed, inverse_function, transform_state)

    Caller responsibility:
        Apply inverse_function to y_pred BEFORE evaluating against y_test (original units).
    """
    # ── G-300: model owns its y-transform → R9 (per_model_optimize) feeds RAW y (identity). Record y_mode="none"
    #    on the trial so the FixedTrial refit/OOF replay reproduces identity WITHOUT this flag
    #    (FixedTrial picks the frozen "none" from the offered choices). Single early-return.
    # ── G-329: tail-amplification surcharge replaces G-328 force-identity blunt hack.
    #    force_y_identity still means model-owned y-transform → identity. extrapolation_safe now
    #    enters the stable allow-list below, so low-error models can still choose asinh/log1p.
    if force_y_identity:
        trial.suggest_categorical("y_mode", ["none"])
        return y_train.copy(), lambda x: np.asarray(x), {"y_mode": "none"}

    # ── Stable / allow-list path: all production R9 (per_model_optimize) models get a small, deterministic
    #    y space. ``extrapolation_safe`` preserves the existing MPH_LINEAR_INVERSE_MODELS
    #    call pattern; MPH_STABLE_TRANSFORMS=1 generalizes it to every model.
    if extrapolation_safe or _stable_preproc_space_enabled():
        y_mode = trial.suggest_categorical(
            "y_mode",
            _choices_for_trial(
                trial, "y_mode", STABLE_PREPROC_MODES,
                ["none", "individual", "group", "categorical"],
                force_stable=extrapolation_safe,
            ),
        )
        if y_mode == "none":
            return y_train.copy(), lambda x: np.asarray(x), {"y_mode": "none"}
        if y_mode != "individual":
            # Frozen legacy group/categorical params can still replay when present.
            if y_mode == "group":
                n = trial.suggest_int("y_group_n", 1, max_chain_length)
                chain_names, inv_funcs, chain_states = [], [], []
                y_curr = y_train.copy()
                for i in range(n):
                    tf = trial.suggest_categorical(f"y_group_{i}", METRIC_Y_TRANSFORMS)
                    chain_names.append(tf)
                    y_curr, inv_i, st_i = _apply_single_y_transform(y_curr, tf)
                    inv_funcs.append(inv_i)
                    chain_states.append({"transform": tf, "state": st_i})

                def combined_inverse(x, invs=tuple(inv_funcs)):
                    x = np.asarray(x)
                    for inv in reversed(invs):
                        x = inv(x)
                    return x

                return y_curr, combined_inverse, {
                    "y_mode": "group",
                    "y_group_n": n,
                    "y_group_chain": chain_names,
                    "y_group_chain_states": chain_states,
                }
            assert y_mode == "categorical"
            name = trial.suggest_categorical("y_categorical", CATEGORICAL_Y_TRANSFORMS)
            y_tr, inv_fn, state = _apply_single_y_transform(y_train, name)
            state["y_mode"] = "categorical"
            state["y_categorical"] = name
            return y_tr, inv_fn, state
        # G-303: models that floor at transformed-zero exclude the 2 median-centered transforms
        #   (laplace/mcmc_robust) so their in-model 0-floor coincides with original-0.
        _ind_stable = _NONCENTERED_STABLE_Y if restrict_centered_y else STABLE_Y_TRANSFORMS
        # G-329b (2026-06-20, 3AI 최종 preproc): G-329 가 STABLE_Y 를 affine-only(laplace/mcmc_robust)로
        #   줄이며 _NONCENTERED_STABLE_Y(=centered 제외) 가 빈 set 이 됨 → floor 모델(restrict_centered_y)
        #   이 individual trial 서 suggest_categorical([]) ValueError 폭사(실측 23/40). 유효 individual
        #   transform 0개 = identity 가 유일·정확한 y baseline → identity 로 단락(crash 0, 누수 0).
        if not _ind_stable:
            return y_train.copy(), lambda x: np.asarray(x), {
                "y_mode": "none", "y_individual": "none"}
        name = trial.suggest_categorical(
            "y_individual",
            _choices_for_trial(
                trial, "y_individual", _ind_stable, METRIC_Y_TRANSFORMS,
                force_stable=extrapolation_safe,
            ),
        )
        y_tr, inv_fn, state = _apply_single_y_transform(y_train, name)
        state["y_mode"] = "individual"
        state["y_individual"] = name
        return y_tr, inv_fn, state

    y_mode = trial.suggest_categorical("y_mode", ["none", "individual", "group", "categorical"])

    # ── none: passthrough
    if y_mode == "none":
        return y_train.copy(), lambda x: np.asarray(x), {"y_mode": "none"}

    # ── individual: single METRIC transform (no chain, no fitted)
    if y_mode == "individual":
        name = trial.suggest_categorical("y_individual", METRIC_Y_TRANSFORMS)
        y_tr, inv_fn, state = _apply_single_y_transform(y_train, name)
        state["y_mode"] = "individual"
        state["y_individual"] = name
        return y_tr, inv_fn, state

    # ── group: chain of METRIC transforms (중복 허용, 순차 적용)
    if y_mode == "group":
        n = trial.suggest_int("y_group_n", 1, max_chain_length)
        chain_names, inv_funcs, chain_states = [], [], []
        y_curr = y_train.copy()
        for i in range(n):
            tf = trial.suggest_categorical(f"y_group_{i}", METRIC_Y_TRANSFORMS)
            chain_names.append(tf)
            y_curr, inv_i, st_i = _apply_single_y_transform(y_curr, tf)
            inv_funcs.append(inv_i)
            chain_states.append({"transform": tf, "state": st_i})

        def combined_inverse(x, invs=tuple(inv_funcs)):
            x = np.asarray(x)
            for inv in reversed(invs):
                x = inv(x)
            return x

        return y_curr, combined_inverse, {
            "y_mode": "group",
            "y_group_n": n,
            "y_group_chain": chain_names,
            "y_group_chain_states": chain_states,
        }

    # ── categorical: fitted transforms (boxcox/yeo_johnson/gaussian)
    assert y_mode == "categorical"
    name = trial.suggest_categorical("y_categorical", CATEGORICAL_Y_TRANSFORMS)
    y_tr, inv_fn, state = _apply_single_y_transform(y_train, name)
    state["y_mode"] = "categorical"
    state["y_categorical"] = name
    return y_tr, inv_fn, state


# ════════════════════════════════════════════════════════════════
# Section 4. X hierarchical preproc — 4-mode symmetric
# ════════════════════════════════════════════════════════════════

def data_driven_group_scalers(
    X_train: np.ndarray, feature_groups: Dict[str, List[int]]
) -> Dict[str, str]:
    """그룹별 분포 통계로 scaler 를 **결정적** 선택 (Optuna 탐색 X → 재현성·안정).

    사용자(2026-06-16) "X group 데이터-기반 고정": 그룹별 도메인-인지 스케일링은 살리되
    3²⁰ Optuna 탐색의 비결정성·미수렴을 제거. G-131(per-feature stats, 블라인드 텍스트북 아님)
    준수 — 각 그룹 feature 들의 꼬리-무거움/왜도로 scaler 결정:
      excess-kurtosis > 5 (heavy-tail/outlier) → "quantile"(rank-기반, 가장 robust)
      |skew| > 1.5 (왜곡)                       → "robust"(median/IQR)
      그 외(well-behaved)                       → "standard"
    그룹 내 feature 별 통계의 중앙값을 사용(단일 outlier feature 가 그룹 전체를 끌지 않게).

    Args:
        X_train: (n, p) 학습 X (원단위).
        feature_groups: {group_name: [col_indices]} — 빈 그룹/상수 feature 안전.

    Returns:
        {group_name: scaler_name in {"standard","robust","quantile"}} — 결정적(같은 X→같은 맵).

    Performance: O(n·p). Side effects: 없음(순수).
    """
    Xa = np.asarray(X_train, dtype=float)
    out: Dict[str, str] = {}
    for grp, idx in feature_groups.items():
        cols = [c for c in (idx or []) if 0 <= c < Xa.shape[1]]
        if not cols:
            out[grp] = "standard"
            continue
        kurts, skews = [], []
        for c in cols:
            v = Xa[:, c]
            v = v[np.isfinite(v)]
            if v.size < 8:
                continue
            sd = float(np.std(v))
            if sd < 1e-12:                      # 상수 feature → 통계 무의미
                continue
            z = (v - float(np.mean(v))) / sd
            kurts.append(float(np.mean(z ** 4) - 3.0))   # excess kurtosis
            skews.append(abs(float(np.mean(z ** 3))))    # |skew|
        kmed = float(np.median(kurts)) if kurts else 0.0
        smed = float(np.median(skews)) if skews else 0.0
        if kmed > 5.0:
            out[grp] = "quantile"
        elif smed > 1.5:
            out[grp] = "robust"
        else:
            out[grp] = "standard"
    return out


def suggest_x_scaler(
    trial: Any,
    X_train: np.ndarray,
    X_test: np.ndarray,
    feature_groups: Optional[Dict[str, List[int]]] = None,
    force_x_identity: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Any, Dict]:
    """Hierarchical X preprocessing — 4-mode Optuna selection (symmetric with suggest_y_preproc).

    Modes (top-level):
        none        → passthrough
        individual  → METRIC_X_SCALERS 중 1개 (전체 X 동일 적용)
        group       → per-feature-group ColumnTransformer (각 group 별 METRIC_X_SCALERS)
        categorical → CATEGORICAL_X_SCALERS 중 1개 (전체 X 동일 적용, grouped 포함)

    Args:
        trial: Optuna trial object
        X_train: 2D array (n_train × n_features)
        X_test: 2D array (n_test × n_features)
        feature_groups: {group_name: [feature_indices]} — group mode 필수

    Returns:
        (X_train_scaled, X_test_scaled, fitted_scaler, scaler_state)

    Caller responsibility:
        Save fitted_scaler in Champion artifact for Pinf (inference) replay.
    """
    # ── G-301 (2026-06-17, budget 감사): USES_FEATURES=False models (TimesFM-2.5 / TiRex) ignore X
    #    entirely → searching the x-scaler is pure-wasted Optuna budget (every x_mode gives the same
    #    output). Force x_mode="none" (passthrough). Record on the trial so FixedTrial replay
    #    reproduces it WITHOUT this flag (mirrors G-300 force_y_identity). The y-transform search is
    #    unaffected (these models forecast y, so the y dimension stays meaningful).
    if force_x_identity:
        trial.suggest_categorical("x_mode", ["none"])
        return X_train.copy(), X_test.copy(), None, {"x_mode": "none"}

    x_mode = trial.suggest_categorical(
        "x_mode",
        _choices_for_trial(trial, "x_mode", STABLE_X_MODES, ["none", "individual", "group", "categorical"]),
    )

    # ── none: passthrough
    if x_mode == "none":
        return X_train.copy(), X_test.copy(), None, {"x_mode": "none"}

    # ── individual: single METRIC scaler, 전체 X 동일 적용
    if x_mode == "individual":
        name = trial.suggest_categorical("x_individual", METRIC_X_SCALERS)
        sc = _build_single_x_scaler(name)
        X_tr_s = sc.fit_transform(X_train)
        X_te_s = sc.transform(X_test)
        return X_tr_s, X_te_s, sc, {"x_mode": "individual", "x_individual": name}

    # ── group: per-feature-group ColumnTransformer
    if x_mode == "group":
        if feature_groups is None:
            feature_groups = {"all_features": list(range(X_train.shape[1]))}
        from sklearn.compose import ColumnTransformer
        # 2026-06-16 (사용자: X group 데이터-기반 고정): stable 모드는 per-group scaler 를 Optuna
        #   탐색(3²⁰ 비결정성) 대신 그룹 분포 통계로 **결정적** 선택 → 도메인-인지 유지 + 재현성.
        #   stable off(레거시/실험) 일 때만 Optuna per-group 탐색 유지.
        _ddmap = (data_driven_group_scalers(X_train, feature_groups)
                  if _stable_preproc_space_enabled() else None)
        transformers = []
        per_group_choices = {}
        for grp_name in sorted(feature_groups):
            col_indices = feature_groups[grp_name]
            if _ddmap is not None:
                chosen = _ddmap.get(grp_name, "standard")                    # 데이터-기반 결정적
            else:
                chosen = trial.suggest_categorical(f"x_group_{grp_name}", METRIC_X_SCALERS)  # legacy Optuna
            per_group_choices[grp_name] = chosen
            transformers.append((f"grp_{grp_name}", _build_single_x_scaler(chosen), col_indices))
        sc = ColumnTransformer(transformers, remainder="passthrough")
        X_tr_s = sc.fit_transform(X_train)
        X_te_s = sc.transform(X_test)
        return X_tr_s, X_te_s, sc, {
            "x_mode": "group",
            "x_group_choices": per_group_choices,
            "x_group_source": "data_driven" if _ddmap is not None else "optuna",
        }

    # ── categorical: single CATEGORICAL scaler, 전체 X 동일 적용 (grouped 포함)
    assert x_mode == "categorical"
    name = trial.suggest_categorical("x_categorical", CATEGORICAL_X_SCALERS)
    sc = _build_single_x_scaler(name)
    if sc is None:
        return X_train.copy(), X_test.copy(), None, {"x_mode": "categorical", "x_categorical": name}
    X_tr_s = sc.fit_transform(X_train)
    X_te_s = sc.transform(X_test)
    return X_tr_s, X_te_s, sc, {"x_mode": "categorical", "x_categorical": name}


# ════════════════════════════════════════════════════════════════
# Section 5. Full Optuna objective — train/test pipeline
# ════════════════════════════════════════════════════════════════

def preproc_objective(
    trial: Any,
    factory_fn: Callable,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_groups: Optional[Dict[str, List[int]]] = None,
    metric_fn: Optional[Callable] = None,
    strict_leakage_check: bool = True,
) -> float:
    """Optuna objective — hierarchical Y/X preproc + train/predict/inverse/evaluate.

    Full pipeline:
        1. Y preproc: y_train only (suggest_y_preproc)
        2. X preproc: X_train fit + X_test transform (suggest_x_scaler)
        3. Model fit: (X_train_s, y_train_t)
        4. Predict: model.predict(X_test_s) → y_pred_t
        5. Inverse Y: y_pred = inv_y(y_pred_t) → original units
        6. Evaluate: metric(y_test_original, y_pred)

    Args:
        trial: Optuna trial
        factory_fn: model constructor (no args → sklearn-compat estimator)
        X_train, y_train: training data (y in original units)
        X_test, y_test: test data (y in original units — NO transform)
        feature_groups: {group_name: [col_indices]} for x.group mode
        metric_fn: scoring fn metric(y_true, y_pred) → float (default: RMSE)
        strict_leakage_check: shape + identity check (default True)

    Returns:
        float metric score
    """
    import optuna

    if strict_leakage_check:
        assert X_train.shape[0] == y_train.shape[0], \
            f"Train shape: X_train {X_train.shape[0]} vs y_train {y_train.shape[0]}"
        assert X_test.shape[0] == y_test.shape[0], \
            f"Test shape: X_test {X_test.shape[0]} vs y_test {y_test.shape[0]}"
        assert X_train.shape[1] == X_test.shape[1], \
            f"Feature count: X_train {X_train.shape[1]} vs X_test {X_test.shape[1]}"
        if X_train is X_test or y_train is y_test:
            raise ValueError("Data leakage: train/test are the same object reference.")

    try:
        y_train_t, inv_y_fn, y_state = suggest_y_preproc(trial, y_train)
    except Exception as e:
        raise optuna.TrialPruned(f"Y preproc failed: {e}")

    try:
        X_train_s, X_test_s, x_scaler, x_state = suggest_x_scaler(
            trial, X_train, X_test, feature_groups
        )
    except Exception as e:
        raise optuna.TrialPruned(f"X preproc failed: {e}")

    if strict_leakage_check:
        assert X_train_s.shape[0] == y_train_t.shape[0]
        assert X_test_s.shape[0] == y_test.shape[0]

    model = factory_fn()
    try:
        model.fit(X_train_s, y_train_t)
    except Exception as e:
        raise optuna.TrialPruned(f"Model fit failed: {e}")

    try:
        y_pred_t = model.predict(X_test_s)
    except Exception as e:
        raise optuna.TrialPruned(f"Model predict failed: {e}")

    try:
        y_pred = np.asarray(inv_y_fn(y_pred_t)).ravel()
    except Exception as e:
        raise optuna.TrialPruned(f"Inverse Y failed: {e}")

    if len(y_pred) != len(y_test):
        raise optuna.TrialPruned(f"Shape mismatch: y_pred {len(y_pred)} vs y_test {len(y_test)}")

    if not np.all(np.isfinite(y_pred)):
        finite_median = float(np.median(y_train))
        n_bad = int(np.sum(~np.isfinite(y_pred)))
        y_pred = np.where(np.isfinite(y_pred), y_pred, finite_median)
        trial.set_user_attr("n_nan_inf_replaced", n_bad)

    try:
        score = float(np.sqrt(np.mean((y_test - y_pred) ** 2))) if metric_fn is None \
            else float(metric_fn(y_test, y_pred))
    except Exception as e:
        raise optuna.TrialPruned(f"Metric failed: {e}")

    if not np.isfinite(score):
        raise optuna.TrialPruned(f"Score not finite: {score}")

    trial.set_user_attr("y_preproc_state", y_state)
    trial.set_user_attr("x_preproc_state", x_state)
    trial.set_user_attr("score", score)
    return score


# ════════════════════════════════════════════════════════════════
# Section 6. Champion artifact helper — Pinf (inference) replay
# ════════════════════════════════════════════════════════════════

def apply_y_preproc_inverse_only(y_pred_t: np.ndarray, y_state: Dict) -> np.ndarray:
    """Pinf (inference) replay: re-apply trained Y inverse without re-fitting.

    Args:
        y_pred_t: predictions in transformed space
        y_state: state dict from suggest_y_preproc

    Returns:
        y_pred in original units
    """
    y_mode = y_state.get("y_mode", "none")

    if y_mode == "none":
        return np.asarray(y_pred_t, dtype=np.float64)

    if y_mode == "individual":
        name = y_state["y_individual"]
        return _reapply_primitive_inverse(y_pred_t, name, y_state)

    if y_mode == "group":
        x = np.asarray(y_pred_t, dtype=np.float64)
        for chain_entry in reversed(y_state["y_group_chain_states"]):
            tf = chain_entry["transform"]
            st = chain_entry["state"]
            x = _reapply_primitive_inverse(x, tf, st)
        return x

    if y_mode == "categorical":
        name = y_state["y_categorical"]
        safe_cap = y_state.get("safe_cap", float("inf"))
        x = np.asarray(y_pred_t, dtype=np.float64)
        if name == "boxcox":
            lam = y_state["lambda"]
            raw = np.power(np.maximum(x * lam + 1, 1e-8), 1.0 / lam) if abs(lam) > 1e-6 \
                else np.exp(x)
            return np.clip(raw, 0, safe_cap)
        if name == "yeo_johnson":
            pt = y_state["power_transformer"]
            return np.clip(pt.inverse_transform(x.reshape(-1, 1)).ravel(), 0, safe_cap)
        if name == "gaussian":
            qt = y_state["quantile_transformer"]
            return np.clip(qt.inverse_transform(x.reshape(-1, 1)).ravel(), 0, safe_cap)
        raise ValueError(f"Unknown categorical transform: {name}")

    raise ValueError(f"Unknown y_mode: {y_mode}")


def _reapply_primitive_inverse(x: np.ndarray, name: str, state: Dict) -> np.ndarray:
    """Re-apply inverse for METRIC (primitive) transforms from saved state."""
    x = np.asarray(x, dtype=np.float64)
    if name == "identity":
        return x
    if name == "log1p":
        if "safe_hi" not in state:
            raise ValueError(f"Malformed log1p state — missing 'safe_hi'. Keys: {list(state.keys())}")
        return np.clip(np.expm1(np.clip(x, -2, state["safe_hi"])), 0, None)
    if name == "sqrt":
        # 2026-06-16: forward 와 대칭으로 safe_cap 적용(신규 state 는 cap 보유; 레거시는 inf=무캡 back-compat).
        return np.clip(np.maximum(x, 0) ** 2, 0, state.get("safe_cap", float("inf")))
    if name == "asinh":
        # G-334 (2026-06-22): forward(_apply_single_y_transform)와 대칭 inverse — sqrt/fourth_root 처럼
        #   fold-불변 safe_cap 으로 sinh 출력 제한(input clip ±arcsinh(cap) + output clip [0,cap]). 미수정
        #   시 sinh(±10)=±11013 catastrophic 발산 가능(다른 transform 은 이미 capped). 레거시 state
        #   (safe_cap 부재)는 ±10 + cap=inf = 옛 동작(back-compat).
        cap = float(state.get("safe_cap", float("inf")))
        hi = float(np.arcsinh(cap)) if np.isfinite(cap) else 10.0
        return np.clip(np.sinh(np.clip(np.asarray(x, dtype=np.float64), -hi, hi)), 0, cap)
    if name == "rank":
        if not all(k in state for k in ("sorted_y", "top_slope", "bot_slope", "n_train")):
            raise ValueError(f"Malformed rank state. Keys: {list(state.keys())}")
        ys = np.asarray(state["sorted_y"], dtype=np.float64)
        n = int(state["n_train"])
        top_slope = float(state["top_slope"])
        bot_slope = float(state["bot_slope"])
        y_out = np.empty_like(x)
        in_range = (x >= 1.0 / n) & (x <= 1.0)
        x_in = np.clip(x[in_range], 1.0 / n, 1.0)
        idx = np.clip((x_in * n).astype(int) - 1, 0, n - 1)
        y_out[in_range] = ys[idx]
        above = x > 1.0
        y_out[above] = ys[-1] + (x[above] - 1.0) * top_slope
        below = x < 1.0 / n
        y_out[below] = ys[0] - (1.0 / n - x[below]) * bot_slope
        return y_out
    if name == "mcmc_robust":
        return x * state["mad_scale"] + state["median"]
    if name == "laplace":
        return x * state["laplace_b"] + state["median"]
    # ── Poisson-VST family (2026-05-30 — G-233 artifact-inverse completeness) ──
    # Mirrors _apply_single_y_transform's inverses so apply_y_preproc_inverse_only covers
    # every METRIC_Y_TRANSFORMS member (else a champion that picked a VST y-transform would
    # raise at Pinf (inference) / Pov (overseas) replay instead of replaying faithfully).
    if name == "anscombe":
        cap = float(state.get("safe_cap", float("inf")))
        return np.clip(np.maximum((x / 2.0) ** 2 - 0.375, 0.0), 0, cap)
    if name == "freeman_tukey":
        cap = float(state.get("safe_cap", float("inf")))
        return np.clip(
            np.maximum(((x * x - 1.0) / (2.0 * np.maximum(x, 1e-9))) ** 2, 0.0), 0, cap)
    if name == "fourth_root":
        # G-333: inverse-replay mirror of the forward (input clip [0, cap^0.25] + output clip [0, cap]).
        cap = float(state.get("safe_cap", float("inf")))
        hi = cap ** 0.25 if np.isfinite(cap) else float("inf")
        return np.clip(np.power(np.clip(np.asarray(x, dtype=np.float64), 0.0, hi), 4.0), 0, cap)
    if name == "arcsine_sqrt":
        return np.clip(np.sin(x / 2.0) ** 2, 0.0, 1.0)
    raise ValueError(f"Cannot re-apply primitive inverse for: {name}")


# ════════════════════════════════════════════════════════════════
# Section 7. Smoke tests (TDD per ENGINEERING_PRINCIPLES.md D-3)
# ════════════════════════════════════════════════════════════════

def _smoke_test_y_modes():
    """4 Y modes (none/individual/group/categorical) — transform/inverse roundtrip."""
    import optuna
    rng = np.random.RandomState(42)
    y = np.maximum(0, rng.lognormal(2, 0.5, size=100))

    cases = [
        ("none",           {"y_mode": "none"}),
        ("individual",     {"y_mode": "individual", "y_individual": "log1p"}),
        ("individual_sqrt",{"y_mode": "individual", "y_individual": "sqrt"}),
        ("group_n1",       {"y_mode": "group", "y_group_n": 1, "y_group_0": "log1p"}),
        ("group_n2",       {"y_mode": "group", "y_group_n": 2,
                            "y_group_0": "log1p", "y_group_1": "sqrt"}),
        ("group_dup",      {"y_mode": "group", "y_group_n": 2,
                            "y_group_0": "log1p", "y_group_1": "log1p"}),
        ("categorical_bc", {"y_mode": "categorical", "y_categorical": "boxcox"}),
        ("categorical_yj", {"y_mode": "categorical", "y_categorical": "yeo_johnson"}),
    ]

    for name, fixed in cases:
        trial = optuna.trial.FixedTrial(fixed)
        try:
            y_t, inv_fn, state = suggest_y_preproc(trial, y)
            y_recon = inv_fn(y_t)
            assert np.all(np.isfinite(y_t)), f"{name}: y_t non-finite"
            assert np.all(np.isfinite(y_recon)), f"{name}: roundtrip non-finite"
            mse = float(np.mean((y - y_recon) ** 2))
            print(f"  [OK] {name:20s} | mode={state['y_mode']}, roundtrip MSE={mse:.2e}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            raise


def _smoke_test_x_modes():
    """4 X modes (none/individual/group/categorical) — scaler/transform."""
    import optuna
    rng = np.random.RandomState(42)
    X_train = rng.randn(80, 12)
    X_test = rng.randn(20, 12)
    feature_groups = {"grp_A": [0, 1, 2, 3], "grp_B": [4, 5, 6, 7], "grp_C": [8, 9, 10, 11]}

    cases = [
        ("none",         {"x_mode": "none"}),
        ("individual",   {"x_mode": "individual", "x_individual": "robust"}),
        ("group",        {"x_mode": "group",
                          "x_group_grp_A": "standard",
                          "x_group_grp_B": "robust",
                          "x_group_grp_C": "quantile"}),
        ("categorical",  {"x_mode": "categorical", "x_categorical": "standard"}),
        ("cat_grouped",  {"x_mode": "categorical", "x_categorical": "grouped"}),
    ]

    for name, fixed in cases:
        trial = optuna.trial.FixedTrial(fixed)
        try:
            X_tr_s, X_te_s, sc, state = suggest_x_scaler(trial, X_train, X_test, feature_groups)
            assert X_tr_s.shape[0] == X_train.shape[0]
            assert X_te_s.shape[0] == X_test.shape[0]
            assert np.all(np.isfinite(X_tr_s))
            assert np.all(np.isfinite(X_te_s))
            print(f"  [OK] {name:15s} | mode={state['x_mode']}, "
                  f"scaler={type(sc).__name__ if sc else 'None'}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            raise


def _smoke_test_symmetry():
    """Y/X 4-mode 구조 대칭 검증."""
    import optuna
    y_modes = optuna.trial.FixedTrial({"y_mode": "none"}).suggest_categorical(
        "y_mode", ["none", "individual", "group", "categorical"]
    )
    x_modes = optuna.trial.FixedTrial({"x_mode": "none"}).suggest_categorical(
        "x_mode", ["none", "individual", "group", "categorical"]
    )
    # Both use same 4-mode list
    assert y_modes == x_modes == "none"
    print("  [OK] Y/X top-level modes 대칭 — none/individual/group/categorical")


def _smoke_test_full_objective():
    """Full preproc_objective — 20 trials with Ridge."""
    import optuna
    from sklearn.linear_model import Ridge
    rng = np.random.RandomState(42)
    X_train, X_test = rng.randn(80, 12), rng.randn(20, 12)
    beta = rng.randn(12) * 0.5
    y_train = np.maximum(0, X_train @ beta + rng.randn(80) * 0.3 + 5)
    y_test = np.maximum(0, X_test @ beta + rng.randn(20) * 0.3 + 5)
    feature_groups = {"grp_A": [0,1,2,3], "grp_B": [4,5,6,7], "grp_C": [8,9,10,11]}

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(multivariate=True, group=True),
    )
    study.optimize(
        lambda t: preproc_objective(t, lambda: Ridge(alpha=1.0),
                                    X_train, y_train, X_test, y_test,
                                    feature_groups=feature_groups),
        n_trials=20, show_progress_bar=False,
    )
    from collections import Counter
    y_modes = Counter(t.params.get("y_mode") for t in study.trials if t.state.name == "COMPLETE")
    x_modes = Counter(t.params.get("x_mode") for t in study.trials if t.state.name == "COMPLETE")
    print(f"  [OK] Full objective — best RMSE={study.best_value:.4f}")
    print(f"       y_modes: {dict(y_modes)}")
    print(f"       x_modes: {dict(x_modes)}")
    assert len(y_modes) >= 2, f"y_modes 탐색 부족: {y_modes}"
    assert len(x_modes) >= 2, f"x_modes 탐색 부족: {x_modes}"


if __name__ == "__main__":
    print("=" * 60)
    print("Smoke 1: Y 4-mode (none/individual/group/categorical)")
    print("=" * 60)
    _smoke_test_y_modes()

    print()
    print("=" * 60)
    print("Smoke 2: X 4-mode (none/individual/group/categorical)")
    print("=" * 60)
    _smoke_test_x_modes()

    print()
    print("=" * 60)
    print("Smoke 3: Y/X symmetry check")
    print("=" * 60)
    _smoke_test_symmetry()

    print()
    print("=" * 60)
    print("Smoke 4: Full objective (20 trials, Ridge)")
    print("=" * 60)
    _smoke_test_full_objective()

    print()
    print("[ALL OK] preproc_optuna_hierarchical.py — 4-mode symmetric structure")
