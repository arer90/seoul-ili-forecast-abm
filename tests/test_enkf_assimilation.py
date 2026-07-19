"""폐루프 ABM↔forecaster EnKF 데이터동화 smoke test.

검증 불변식:
  1. 보존(conservation): EnKF 갱신이 앙상블 평균을 관측 쪽으로 끌어당겨 RMSE 가
     줄어든다(rmse_post <= rmse_prior).
  2. Kalman 정확성: 합성 선형-Gaussian scalar case 에서 분석 평균이 닫힌형
     posterior 평균에 수렴(perturbed-obs EnKF 의 점근 정확성).
  3. shape: 출력 앙상블 shape = 입력 shape.
  4. edge: 붕괴 앙상블(분산 0) → gain 0 → 항등 갱신. 멤버 1개 → ValueError.
  5. leak-free: cutoff 이후 forecast 값이 갱신에 영향 없음(슬라이싱 폐기).
  6. model-agnostic: FusedEpi 외 임의 모델명 동작(2개 모델명 테스트).
"""
from __future__ import annotations

import numpy as np
import pytest

from simulation.abm.enkf_assimilation import (
    assimilate_forecast,
    ensemble_kalman_update,
)


# --------------------------------------------------------------------------- #
# fixture: 2개 모델명을 가진 합성 forecast CSV (model-agnostic 검증용)
# --------------------------------------------------------------------------- #
@pytest.fixture()
def forecast_csv(tmp_path):
    path = tmp_path / "predictions_per_model.csv"
    lines = ["week_idx,model,y_pred"]
    # 두 모델: 서로 다른 nowcast 레벨 → model_name 만으로 다른 obs 가 잡혀야 함
    fused = [2.0, 2.5, 3.0, 4.0, 5.0]      # FusedEpi 마지막 과거점 → 5.0
    negbin = [1.0, 1.2, 1.4, 1.6, 1.8]     # NegBinGLM-V7 마지막 과거점 → 1.8
    for wk, (a, b) in enumerate(zip(fused, negbin)):
        lines.append(f"{wk},FusedEpi,{a}")
        lines.append(f"{wk},NegBinGLM-V7,{b}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


# === 1. 보존: RMSE 감소 (합성 선형 case) ==================================== #
def test_update_reduces_rmse_scalar():
    """전역 1D 관측: 갱신 후 앙상블 평균이 obs 에 더 가까워진다."""
    rng = np.random.default_rng(0)
    # prior 앙상블 평균 ≈ 0, 관측 y = 5 → 갱신은 평균을 5 쪽으로 당겨야 함
    Xf = rng.standard_normal((40, 1))            # mean ~ 0
    y = np.array([5.0])
    H = np.array([[1.0]])
    R = np.array([[1.0]])
    Xa = ensemble_kalman_update(Xf, y, H, R, seed=1)
    rmse_prior = abs(Xf.mean() - 5.0)
    rmse_post = abs(Xa.mean() - 5.0)
    assert rmse_post < rmse_prior


# === 2. Kalman 정확성: 닫힌형 posterior 평균 수렴 =========================== #
def test_kalman_gain_matches_closed_form():
    """선형-Gaussian scalar: 분석 평균 → posterior = (R·μ_f + σ_f²·y)/(σ_f²+R).

    prior N(μ_f, σ_f²), 관측 y, 관측오차 R 의 닫힌형 posterior 평균과 EnKF 분석
    평균이 큰 앙상블에서 일치(perturbed-obs EnKF 의 정확성).
    """
    rng = np.random.default_rng(7)
    mu_f, sigma_f2 = 1.0, 4.0
    y_obs, R_val = 5.0, 1.0
    m = 200_000
    Xf = (mu_f + np.sqrt(sigma_f2) * rng.standard_normal(m)).reshape(m, 1)
    H = np.array([[1.0]])
    R = np.array([[R_val]])
    Xa = ensemble_kalman_update(Xf, np.array([y_obs]), H, R, seed=3)

    post_mean_closed = (R_val * mu_f + sigma_f2 * y_obs) / (sigma_f2 + R_val)
    assert abs(float(Xa.mean()) - post_mean_closed) < 0.05

    # 분석 분산도 닫힌형 (1/σ_f² + 1/R)⁻¹ 에 근접
    post_var_closed = 1.0 / (1.0 / sigma_f2 + 1.0 / R_val)
    assert abs(float(Xa.var()) - post_var_closed) < 0.1


# === 3. shape 보존 + 다변량(25구) ========================================== #
def test_multivariate_shape_and_partial_obs():
    """25구 상태, 전역 평균 관측 1개 → 출력 shape (m, 25) 유지."""
    rng = np.random.default_rng(2)
    m, n = 30, 25
    Xf = 0.05 + 0.01 * rng.standard_normal((m, n))   # 구별 prevalence
    H = np.full((1, n), 1.0 / n)                     # 전역 ILI
    y = np.array([0.10])                             # 관측: 더 높은 prevalence
    R = np.array([[1e-4]])
    Xa = ensemble_kalman_update(Xf, y, H, R, seed=5)
    assert Xa.shape == (m, n)
    # 전역 평균이 obs(0.10) 쪽으로 이동
    assert abs((H @ Xa.mean(0))[0] - 0.10) < abs((H @ Xf.mean(0))[0] - 0.10)


# === 4a. edge: 붕괴 앙상블(분산 0) → 항등 갱신 ============================= #
def test_collapsed_ensemble_is_identity():
    """모든 멤버 동일(표본 공분산 0) → gain 0 → 갱신이 항등."""
    Xf = np.full((10, 3), 0.5)
    y = np.array([9.0])
    H = np.full((1, 3), 1.0 / 3.0)
    R = np.array([[1.0]])
    Xa = ensemble_kalman_update(Xf, y, H, R, seed=0)
    np.testing.assert_allclose(Xa, Xf, atol=1e-12)


# === 4b. edge: 멤버 1개 → ValueError ====================================== #
def test_single_member_raises():
    with pytest.raises(ValueError, match=">= 2 members"):
        ensemble_kalman_update(
            np.zeros((1, 2)), np.array([1.0]), np.array([[1.0, 0.0]]), np.array([[1.0]])
        )


# === 4c. edge: 차원 불일치 / 비유한 → ValueError ========================== #
def test_dimension_and_nan_guards():
    Xf = np.zeros((5, 3))
    with pytest.raises(ValueError, match="H must have shape"):
        ensemble_kalman_update(Xf, np.array([1.0]), np.eye(2, 3), np.array([[1.0]]))
    with pytest.raises(ValueError):
        ensemble_kalman_update(
            np.array([[np.nan, 0.0], [1.0, 1.0]]),
            np.array([1.0]),
            np.array([[1.0, 0.0]]),
            np.array([[1.0]]),
        )


# === 5. 결정성: 같은 시드 → 비트동일 ====================================== #
def test_determinism_same_seed():
    rng = np.random.default_rng(11)
    Xf = rng.standard_normal((20, 4))
    y = np.array([2.0, 3.0])
    H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    R = np.eye(2)
    a = ensemble_kalman_update(Xf, y, H, R, seed=99)
    b = ensemble_kalman_update(Xf, y, H, R, seed=99)
    np.testing.assert_array_equal(a, b)


# === 6. assimilate_forecast: model-agnostic (2개 모델명) ================== #
def test_assimilate_is_model_agnostic(forecast_csv):
    """FusedEpi 와 NegBinGLM-V7 둘 다 동작 + 각자 다른 nowcast obs 사용."""
    rng = np.random.default_rng(4)
    ens = 0.04 + 0.01 * rng.standard_normal((25, 1))   # 전역 ILI proxy

    res_fused = assimilate_forecast(
        "FusedEpi", ensemble=ens, predictions_csv=forecast_csv, obs_var=0.5
    )
    res_negbin = assimilate_forecast(
        "NegBinGLM-V7", ensemble=ens, predictions_csv=forecast_csv, obs_var=0.5
    )

    # 모델-비종속: 모델명 에코 + 둘 다 정상 dict
    assert res_fused["model_name"] == "FusedEpi"
    assert res_negbin["model_name"] == "NegBinGLM-V7"
    # 각 모델의 마지막 과거점이 obs 로 잡힘(서로 다름 → 진짜 모델별 조회)
    assert res_fused["obs_used"] == pytest.approx(5.0)
    assert res_negbin["obs_used"] == pytest.approx(1.8)
    # 불변식: 갱신 후 RMSE 가 줄어듦(개선 실측)
    assert res_fused["improved"] and res_fused["rmse_post"] <= res_fused["rmse_prior"]
    assert res_negbin["improved"] and res_negbin["rmse_post"] <= res_negbin["rmse_prior"]
    # shape 보존
    assert res_fused["updated_ensemble"].shape == ens.shape


# === 7. leak-free: cutoff 이후 forecast 값 변경이 갱신에 영향 없음 ========= #
def test_leak_free_future_does_not_affect_update(tmp_path):
    """cutoff 까지 동일하고 미래만 다른 두 forecast → 동일 갱신(미래 미사용)."""
    def write_csv(name, future_val):
        p = tmp_path / name
        rows = ["week_idx,model,y_pred"]
        series = [2.0, 3.0, 4.0, future_val]      # cutoff=2 → 과거 = [2,3,4]
        for wk, v in enumerate(series):
            rows.append(f"{wk},FusedEpi,{v}")
        p.write_text("\n".join(rows) + "\n", encoding="utf-8")
        return str(p)

    csv_a = write_csv("a.csv", 99.0)              # 미래 = 99 (폭주)
    csv_b = write_csv("b.csv", -50.0)             # 미래 = 다른 값
    ens = np.linspace(1.0, 2.0, 16).reshape(16, 1)

    res_a = assimilate_forecast(
        "FusedEpi", ensemble=ens, predictions_csv=csv_a, cutoff=2, obs_var=1.0
    )
    res_b = assimilate_forecast(
        "FusedEpi", ensemble=ens, predictions_csv=csv_b, cutoff=2, obs_var=1.0
    )
    # cutoff 시점 obs 동일(=4.0), 미래값(99 vs −50)은 갱신에 안 들어감
    assert res_a["obs_used"] == pytest.approx(4.0)
    assert res_b["obs_used"] == pytest.approx(4.0)
    assert res_a["leak_free"] and res_b["leak_free"]
    np.testing.assert_array_equal(
        res_a["updated_ensemble"], res_b["updated_ensemble"]
    )


# === 8. assimilate_forecast: 외부 obs_series + 입력 미변경 ================ #
def test_external_obs_and_input_unchanged(forecast_csv):
    ens = np.array([[0.1], [0.2], [0.3], [0.4]])
    ens_before = ens.copy()
    res = assimilate_forecast(
        "FusedEpi",
        ensemble=ens,
        obs_series=[0.05, 0.06, 0.07],            # 외부 ground-truth
        predictions_csv=forecast_csv,
        cutoff=2,
        obs_var=0.01,
    )
    assert res["obs_used"] == pytest.approx(0.07)   # 외부 시리즈 cutoff 점
    # 입력 ensemble 미변경(side-effect free)
    np.testing.assert_array_equal(ens, ens_before)
    # obs_var <= 0 가드
    with pytest.raises(ValueError, match="obs_var"):
        assimilate_forecast("FusedEpi", ensemble=ens, obs_var=0.0,
                            predictions_csv=forecast_csv)


# === 9. conformal PI 반폭 → 관측 오차분산 결합(Appendix D.4) ================ #
def test_conformal_halfwidth_derives_obs_var(forecast_csv):
    """conformal_halfwidth 지정 시 obs_var 를 (hw/1.96)^2 로 유도 — default 는 불변."""
    rng = np.random.default_rng(11)
    ens = 0.04 + 0.01 * rng.standard_normal((25, 1))

    # (a) default(None): obs_var 그대로(기존 동작 불변)
    res_default = assimilate_forecast(
        "FusedEpi", ensemble=ens, predictions_csv=forecast_csv, obs_var=0.5
    )
    assert res_default["obs_var_effective"] == pytest.approx(0.5)

    # (b) conformal 반폭 지정: obs_var_eff = (hw/z)^2, z from the confidence LEVEL
    #     (dynamic, not the hard-coded 1.96) — default level 0.95 → z ≈ 1.95996.
    from scipy.stats import norm
    z95 = float(norm.ppf(0.975))
    hw = 2.0 * z95                                   # → (hw/z)^2 = 4.0
    res_conf = assimilate_forecast(
        "FusedEpi", ensemble=ens, predictions_csv=forecast_csv,
        obs_var=0.5, conformal_halfwidth=hw,
    )
    assert res_conf["obs_var_effective"] == pytest.approx((hw / z95) ** 2)
    assert res_conf["obs_var_effective"] == pytest.approx(4.0)


def test_conformal_wider_pi_trusts_observation_less(forecast_csv):
    """넓은 PI(불확실) → obs_var↑ → 칼만 이득↓ → 갱신이 obs 를 덜 당김(prior 유지)."""
    rng = np.random.default_rng(12)
    ens = 0.04 + 0.01 * rng.standard_normal((60, 1))
    prior_mean = float(ens.mean())

    narrow = assimilate_forecast(
        "FusedEpi", ensemble=ens, predictions_csv=forecast_csv,
        conformal_halfwidth=0.5,                     # 확신 → 좁은 PI
    )
    wide = assimilate_forecast(
        "FusedEpi", ensemble=ens, predictions_csv=forecast_csv,
        conformal_halfwidth=20.0,                    # 불확실 → 넓은 PI
    )
    obs = narrow["obs_used"]                          # 두 경우 동일 obs
    # 넓은 PI 쪽 post-mean 이 prior 에 더 가깝다(obs 로부터 덜 당겨짐)
    pull_narrow = abs(narrow["ens_mean_post"][0] - prior_mean)
    pull_wide = abs(wide["ens_mean_post"][0] - prior_mean)
    assert pull_wide < pull_narrow
    # 방향 sanity: obs 가 prior 보다 큼 → 둘 다 위로 갱신
    assert obs > prior_mean


def test_conformal_halfwidth_guards(forecast_csv):
    """비유한/비양수 반폭 → 0초 ValueError(fail-fast)."""
    ens = np.array([[0.1], [0.2], [0.3], [0.4]])
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="conformal_halfwidth"):
            assimilate_forecast(
                "FusedEpi", ensemble=ens, predictions_csv=forecast_csv,
                conformal_halfwidth=bad,
            )
