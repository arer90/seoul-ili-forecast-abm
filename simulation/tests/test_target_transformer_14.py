"""Sprint α R1 (2026-05-26): TargetTransformer 14-method round-trip.

R6-A 결과 14 method (legacy 5 + hierarchical 9). 각 method 의 fit/transform/
inverse_transform round-trip 검증. ChampionArtifact pickle path 도 cover.
"""
from __future__ import annotations

import numpy as np
import pickle

from simulation.models.target_transform import TargetTransformer, _HIERARCHICAL_METHODS


LEGACY_METHODS = ["log1p", "sqrt", "boxcox", "robust", "none"]
HIERARCHICAL_METHODS = sorted(_HIERARCHICAL_METHODS)
ALL_METHODS = LEGACY_METHODS + HIERARCHICAL_METHODS


# Sample Y (positive, count-like ILI range)
Y_COUNT = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
Y_PROP = np.array([0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 0.95])


def test_legacy_5_methods_round_trip():
    """legacy 5 method (log1p/sqrt/boxcox/robust/none) round-trip."""
    for method in LEGACY_METHODS:
        tt = TargetTransformer(method=method, clip_negative=True)
        y_t = tt.fit_transform(Y_COUNT)
        y_back = tt.inverse_transform(y_t)
        err = float(np.max(np.abs(y_back - Y_COUNT)))
        assert err < 1e-5, f"{method} round-trip err = {err}"


def test_hierarchical_count_methods_round_trip():
    """hierarchical 9 method 중 count-applicable 8 round-trip."""
    for method in ["log1p", "sqrt", "asinh", "rank", "mcmc_robust", "laplace",
                    "yeo_johnson", "anscombe", "freeman_tukey"]:
        tt = TargetTransformer(method=method, clip_negative=False)
        y_t = tt.fit_transform(Y_COUNT)
        y_back = tt.inverse_transform(y_t)
        err = float(np.max(np.abs(y_back - Y_COUNT)))
        assert err < 1e-4, f"{method} round-trip err = {err}"


def test_arcsine_sqrt_proportion_round_trip():
    tt = TargetTransformer(method="arcsine_sqrt", clip_negative=False)
    p_t = tt.fit_transform(Y_PROP)
    p_back = tt.inverse_transform(p_t)
    err = float(np.max(np.abs(p_back - Y_PROP)))
    assert err < 1e-10, f"arcsine_sqrt err = {err}"


def test_gaussian_quantile_round_trip():
    rng = np.random.default_rng(7)
    y = rng.exponential(scale=5.0, size=200) + 0.1
    tt = TargetTransformer(method="gaussian", clip_negative=False)
    y_t = tt.fit_transform(y)
    y_back = tt.inverse_transform(y_t)
    # QuantileTransformer 는 monotonic 단 분포에서 작은 numerical drift
    assert np.allclose(y_back, y, atol=1.0)


def test_robust_legacy_uses_iqr():
    """legacy robust = IQR-based (NOT MAD)."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 100.0])
    tt = TargetTransformer(method="robust")
    tt.fit(y)
    # median = 6.0, IQR = q75 - q25 = 8.5 - 3.5 = 5.0
    assert tt._median == 6.0
    assert abs(tt._iqr - 5.0) < 1e-9


def test_mcmc_robust_uses_mad_not_iqr():
    """hierarchical mcmc_robust = (y - median) / (1.4826 × MAD), NOT IQR."""
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    tt_iqr = TargetTransformer(method="robust")
    tt_mad = TargetTransformer(method="mcmc_robust")
    y_iqr = tt_iqr.fit_transform(y)
    y_mad = tt_mad.fit_transform(y)
    # 두 결과는 서로 다름 (단위 scale 차이)
    assert not np.allclose(y_iqr, y_mad)


def test_none_method_passthrough():
    tt = TargetTransformer(method="none")
    y_t = tt.fit_transform(Y_COUNT)
    assert np.array_equal(y_t, Y_COUNT)
    y_back = tt.inverse_transform(y_t)
    assert np.array_equal(y_back, Y_COUNT)


def test_clip_negative_default_true():
    """clip_negative=True default — log1p inverse 음수값 clip 검증."""
    tt = TargetTransformer(method="log1p", clip_negative=True)
    tt.fit(Y_COUNT)
    # 음수 input → inverse → max(_, 0)
    result = tt.inverse_transform(np.array([-2.0]))
    assert result[0] >= 0


def test_hierarchical_inverse_before_fit_raises():
    """hierarchical method 는 fit 없이 inverse_transform 호출 시 ValueError."""
    tt = TargetTransformer(method="mcmc_robust")
    try:
        tt.inverse_transform(np.array([1.0]))
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "requires .fit" in str(e)


def test_fitted_target_transformer_is_picklable():
    """ChampionArtifact bundle 에 pickle 됨 — closure capture 검증."""
    tt = TargetTransformer(method="log1p")
    tt.fit(Y_COUNT)
    bytes_ = pickle.dumps(tt)
    tt_loaded = pickle.loads(bytes_)
    assert tt_loaded._fitted is True
    y_t = tt_loaded.transform(Y_COUNT)
    assert np.allclose(y_t, tt.transform(Y_COUNT))


def test_hierarchical_fitted_state_picklable_via_cloudpickle():
    """ChampionArtifact 는 cloudpickle 사용 (G-154). yeo_johnson 같은 nested
    inv_fn 도 cloudpickle 로 round-trip."""
    try:
        import cloudpickle
    except ImportError:
        # cloudpickle 없으면 skip — preflight 가 G-154 로 보장하므로 정상 환경에서 통과
        return
    tt = TargetTransformer(method="yeo_johnson")
    tt.fit(Y_COUNT)
    bytes_ = cloudpickle.dumps(tt)
    tt_loaded = cloudpickle.loads(bytes_)
    y_t = tt_loaded.transform(Y_COUNT)
    y_back = tt_loaded.inverse_transform(y_t)
    assert np.allclose(y_back, Y_COUNT, atol=1e-4)


def test_hierarchical_closure_methods_picklable_via_cloudpickle():
    """hierarchical 의 모든 method inv_fn (lambda 또는 nested def) 는 cloudpickle
    round-trip 통과. Champion artifact 는 cloudpickle 사용 (G-154 — preflight 강제).

    stdlib pickle 은 nested lambda/def 모두 실패 → cloudpickle path 가 운영 standard.
    """
    try:
        import cloudpickle
    except ImportError:
        return  # cloudpickle 없으면 skip (preflight G-154 가 정상 환경에 보장)
    for method in ["mcmc_robust", "laplace", "asinh", "rank", "anscombe",
                    "freeman_tukey", "arcsine_sqrt", "gaussian", "yeo_johnson"]:
        tt = TargetTransformer(method=method)
        if method == "arcsine_sqrt":
            tt.fit(Y_PROP)
            y_back = tt.inverse_transform(cloudpickle.loads(cloudpickle.dumps(tt)).transform(Y_PROP))
            assert np.allclose(y_back, Y_PROP, atol=1e-6)
        else:
            tt.fit(Y_COUNT)
            tt_loaded = cloudpickle.loads(cloudpickle.dumps(tt))
            y_t = tt_loaded.transform(Y_COUNT)
            y_back = tt_loaded.inverse_transform(y_t)
            assert np.allclose(y_back, Y_COUNT, atol=1e-3), f"{method} cloudpickle round-trip failed"


def test_legacy_method_field_ABI_preserved():
    """ChampionArtifact ABI: legacy 5 method 의 internal field 이름 잠금."""
    tt = TargetTransformer(method="robust")
    tt.fit(Y_COUNT)
    # ABI fields (champion .pt 가 의존)
    assert hasattr(tt, "_median")
    assert hasattr(tt, "_iqr")
    assert hasattr(tt, "_boxcox_lambda")
    assert hasattr(tt, "_fitted")


def test_hierarchical_method_field_ABI_preserved():
    """R6-A 신규 hierarchical state field 이름 잠금."""
    tt = TargetTransformer(method="mcmc_robust")
    tt.fit(Y_COUNT)
    assert hasattr(tt, "_hier_inv_fn")
    assert hasattr(tt, "_hier_state")
    assert hasattr(tt, "_hier_y_t")
    assert tt._hier_state.get("median") is not None


def test_method_enum_has_14_options():
    """현재 14 method 지원 (legacy 5 + hierarchical 9). 추가/제거 시 test 갱신 필수."""
    assert len(ALL_METHODS) == 14
    # 모두 instantiable
    for m in ALL_METHODS:
        TargetTransformer(method=m)
