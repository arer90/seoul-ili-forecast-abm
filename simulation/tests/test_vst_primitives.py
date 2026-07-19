"""Sprint α R1 (2026-05-26): _vst_primitives SoT round-trip tests.

Tests 7 canonical helpers + 14 legacy aliases. Pure functions, no DB.
Covers: identity round-trip / inverse divergence safety / alias resolution.
"""
from __future__ import annotations

import numpy as np

from simulation.models._vst_primitives import (
    # canonical
    log1p_transformer, sqrt_transformer, yeo_johnson_transformer,
    anscombe_transformer, freeman_tukey_transformer,
    arcsinh_transformer, arcsine_sqrt_transformer,
    # per_feature legacy aliases
    _log1p_t, _sqrt_t, _yeo_johnson_t, _anscombe_t,
    _freeman_tukey_t, _arcsinh_t, _arcsine_sqrt_t,
    # grouped legacy aliases
    _log1p_transformer, _sqrt_transformer, _yeo_johnson_safe,
    _anscombe_transformer, _freeman_tukey_transformer,
    _arcsinh_transformer, _arcsine_sqrt_transformer,
)


# Sample y arrays covering count-like ILI ranges + edge cases
Y_POSITIVE = np.array([0.0, 1.0, 5.0, 10.0, 50.0])
Y_PROPORTION = np.array([0.0, 0.1, 0.5, 0.9, 1.0])


def test_count_vst_round_trip():
    """count-like Y → forward+inverse 회복 (5 VST: log1p/sqrt/anscombe/freeman_tukey/arcsinh).

    yeo_johnson 은 PowerTransformer 가 2D 요구 → 별도 test (test_yeo_johnson_2d).
    """
    for name, fn in [
        ("log1p", log1p_transformer),
        ("sqrt", sqrt_transformer),
        ("anscombe", anscombe_transformer),
        ("freeman_tukey", freeman_tukey_transformer),
        ("arcsinh", arcsinh_transformer),
    ]:
        t = fn()
        y_t = t.fit_transform(Y_POSITIVE)
        y_back = t.inverse_transform(y_t)
        err = float(np.max(np.abs(y_back - Y_POSITIVE)))
        assert err < 1e-6, f"{name} round-trip error {err}"


def test_yeo_johnson_2d_round_trip():
    """yeo_johnson 의 sklearn PowerTransformer 는 2D 입력 + standardize=True 후 inverse 가 정확하지 않을 수 있음 (caller responsibility)."""
    t = yeo_johnson_transformer()
    Y_2D = Y_POSITIVE.reshape(-1, 1)
    y_t = t.fit_transform(Y_2D)
    y_back = t.inverse_transform(y_t).ravel()
    err = float(np.max(np.abs(y_back - Y_POSITIVE)))
    # standardize=True 때문에 더 큰 tolerance
    assert err < 1e-3, f"yeo_johnson 2D round-trip err={err}"


def test_arcsine_sqrt_proportion_round_trip():
    """proportion p ∈ [0,1] → 회복."""
    t = arcsine_sqrt_transformer()
    p_t = t.fit_transform(Y_PROPORTION)
    p_back = t.inverse_transform(p_t)
    err = float(np.max(np.abs(p_back - Y_PROPORTION)))
    assert err < 1e-10, f"arcsine_sqrt round-trip {err}"


def test_arcsinh_default_scale_is_10():
    """per_feature legacy 호환 — _arcsinh_t() default = scale=10.0."""
    t = arcsinh_transformer()
    # asinh(10/10) = asinh(1) ≈ 0.8813735
    out = t.transform(np.array([[10.0]]))
    assert abs(float(out[0, 0]) - 0.8813735870195430) < 1e-9


def test_freeman_tukey_inverse_div_by_zero_guard():
    """freeman_tukey inverse 의 div-by-zero guard 동작."""
    t = freeman_tukey_transformer()
    # z = 0 시 1e-9 guard 적용 → exception 없이 finite 결과
    out = t.inverse_transform(np.array([0.0]))
    assert np.all(np.isfinite(out))


def test_per_feature_legacy_aliases_resolve_to_canonical():
    """per_feature 의 7 alias 가 canonical 과 동일 함수."""
    pairs = [
        (_log1p_t, log1p_transformer),
        (_sqrt_t, sqrt_transformer),
        (_yeo_johnson_t, yeo_johnson_transformer),
        (_anscombe_t, anscombe_transformer),
        (_freeman_tukey_t, freeman_tukey_transformer),
        (_arcsinh_t, arcsinh_transformer),
        (_arcsine_sqrt_t, arcsine_sqrt_transformer),
    ]
    for alias, canonical in pairs:
        assert alias is canonical, f"per_feature alias {alias.__name__} mismatch"


def test_grouped_legacy_aliases_resolve_to_canonical():
    """grouped 의 7 alias 가 canonical 과 동일 함수."""
    pairs = [
        (_log1p_transformer, log1p_transformer),
        (_sqrt_transformer, sqrt_transformer),
        (_yeo_johnson_safe, yeo_johnson_transformer),
        (_anscombe_transformer, anscombe_transformer),
        (_freeman_tukey_transformer, freeman_tukey_transformer),
        (_arcsinh_transformer, arcsinh_transformer),
        (_arcsine_sqrt_transformer, arcsine_sqrt_transformer),
    ]
    for alias, canonical in pairs:
        assert alias is canonical, f"grouped alias {alias.__name__} mismatch"


def test_arcsinh_explicit_scale_overrides_default():
    """grouped caller 가 scale=100 explicit 전달 시 default 무시."""
    t = arcsinh_transformer(scale=100.0)
    # asinh(100/100) = asinh(1)
    out = t.transform(np.array([[100.0]]))
    assert abs(float(out[0, 0]) - 0.8813735870195430) < 1e-9


def test_log1p_handles_negative_input_safely():
    """log1p 의 np.maximum(0) clamp 동작."""
    t = log1p_transformer()
    # 음수 input → log1p(0) = 0
    out = t.transform(np.array([[-5.0]]))
    assert float(out[0, 0]) == 0.0


def test_arcsine_sqrt_clips_out_of_range_proportion():
    """proportion [0,1] 밖 input → clip → 안전한 변환."""
    t = arcsine_sqrt_transformer()
    out = t.fit_transform(np.array([1.5, -0.5]))
    assert np.all(np.isfinite(out))
