"""Stable hierarchical y-transform pool (G-256 + 2026-06-16 restriction).

Free-form regressors (linear / neural / GAM) can extrapolate in transformed space, so a
nonlinear-inverse y-transform (log1p→expm1, sqrt→x², boxcox/yeo_johnson) amplifies the
extrapolation into a blow-up (exp_peak_extrapolation.py Part A: MLP/Ridge + log1p → r2 -9).
The paper-grade Phase-13 path now restricts every model to ``none`` (identity) or one
stable individual transform (log1p / sqrt / asinh / laplace); the old full pool is reachable
only when ``MPH_STABLE_TRANSFORMS=0`` or when replaying frozen legacy params.
"""
from __future__ import annotations

import os
from contextlib import contextmanager

import numpy as np
import optuna

from simulation.pipeline.preproc_optuna_hierarchical import (
    suggest_y_preproc,
    suggest_x_scaler,
    model_needs_linear_inverse_y,
    STABLE_Y_TRANSFORMS,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
_Y = np.random.RandomState(0).uniform(4.0, 80.0, 200)
_SAFE_LABELS = {"none"} | set(STABLE_Y_TRANSFORMS)


@contextmanager
def _stable_env(value: str):
    old = os.environ.get("MPH_STABLE_TRANSFORMS")
    os.environ["MPH_STABLE_TRANSFORMS"] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("MPH_STABLE_TRANSFORMS", None)
        else:
            os.environ["MPH_STABLE_TRANSFORMS"] = old


def _collect_labels(extrapolation_safe: bool, stable: str = "1", n: int = 60) -> set:
    seen: set = set()

    def obj(trial):
        _yt, _inv, state = suggest_y_preproc(trial, _Y.copy(), extrapolation_safe=extrapolation_safe)
        seen.add(state.get("y_individual") or state.get("y_group_chain", [None])[0] or state["y_mode"])
        return 0.0

    with _stable_env(stable):
        optuna.create_study().optimize(obj, n_trials=n)
    return seen


def test_classifier_extrapolating_families() -> None:
    # free-form regressors → restricted
    # 2026-07-19: was "TabularDNN", which has since moved to DEFER_MODELS and so
    # is absent from CATEGORY_MODELS. The classifier walks CATEGORY_MODELS and
    # returns False for anything it does not find — its documented permissive
    # default — so the test was reading a retirement as a regression. Use a
    # dl-tabular model that is actually in the active lineup.
    assert model_needs_linear_inverse_y("DNN") is True          # dl-tabular
    # A deferred model is unknown to the classifier by design; pin that too so
    # the permissive default stays deliberate rather than accidental.
    assert model_needs_linear_inverse_y("TabularDNN") is False  # deferred → unknown
    assert model_needs_linear_inverse_y("ElasticNet") is True   # linear
    assert model_needs_linear_inverse_y("N-BEATS") is True      # modern-ts
    assert model_needs_linear_inverse_y("GAM-Spline") is True   # other
    assert model_needs_linear_inverse_y("GAT") is True          # graph (neural)
    # G-261 (2026-06-13): Chronos-2 → TimesFM-2.5/TiRex 대체 (Chronos retire). 동일 foundation family.
    assert model_needs_linear_inverse_y("TimesFM-2.5") is True  # foundation
    assert model_needs_linear_inverse_y("TiRex") is True        # foundation (xLSTM)


def test_classifier_bounded_families() -> None:
    # cap-at-train-max models → full pool (permissive)
    assert model_needs_linear_inverse_y("XGBoost") is False     # tree
    assert model_needs_linear_inverse_y("CatBoost") is False
    assert model_needs_linear_inverse_y("SVR-RBF") is False     # kernel (local)
    assert model_needs_linear_inverse_y("ARIMA") is False       # ts (mean-reverting)
    assert model_needs_linear_inverse_y(None) is False
    assert model_needs_linear_inverse_y("NotARealModel") is False


def test_stable_space_blocks_divergent_y_transforms_for_all_models() -> None:
    seen = _collect_labels(extrapolation_safe=False, stable="1")
    assert seen <= _SAFE_LABELS, f"stable pool leaked a divergent transform: {seen}"
    for bad in ("anscombe", "freeman_tukey", "boxcox", "yeo_johnson", "rank"):
        assert bad not in seen


def test_extrapolation_safe_uses_same_stable_pool() -> None:
    seen = _collect_labels(extrapolation_safe=True, stable="0")
    assert seen <= _SAFE_LABELS, f"restricted pool leaked a divergent transform: {seen}"
    for bad in ("anscombe", "freeman_tukey", "boxcox", "yeo_johnson", "rank"):
        assert bad not in seen


def test_full_pool_unrestricted_when_flag_off() -> None:
    seen = _collect_labels(extrapolation_safe=False, stable="0")
    # the full VST pool is reachable (sanity: at least one nonlinear-inverse transform appears)
    assert seen - _SAFE_LABELS, f"expected full pool to offer nonlinear-inverse transforms, got {seen}"


def test_stable_x_modes_none_or_data_driven_group() -> None:
    # 2026-06-16 (사용자 X/Y 별개): X stable = {none, group}, group=데이터-기반 결정적(Optuna 탐색 X).
    # individual/categorical 차단; group 은 허용하되 x_group_<name> Optuna dim 없어야.
    seen: set = set()
    srcs: set = set()
    params: set = set()
    X = np.random.RandomState(1).normal(size=(30, 4))

    def obj(trial):
        _xt, _xv, _sc, state = suggest_x_scaler(
            trial, X, X[:5], feature_groups={"b": [0, 1], "a": [2, 3]}
        )
        seen.add(state["x_mode"])
        srcs.add(state.get("x_group_source"))
        params.update(trial.params.keys())
        return 0.0

    with _stable_env("1"):
        optuna.create_study().optimize(obj, n_trials=30)
    assert seen <= {"none", "group"}, f"stable X leaked individual/categorical: {seen}"
    assert srcs <= {None, "data_driven"}, f"X group Optuna 탐색 누출: {srcs}"
    assert not any(p.startswith("x_group_") for p in params), f"x_group Optuna dim 잔존: {params}"
