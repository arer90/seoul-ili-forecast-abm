"""TDD — sbi 신경 사후추정 파이프라인이 동작하는가 (외부평가 3차 권고).

검증 전략: ABM은 비싸고 약식별이라 '복원 실패'가 모델 탓인지 파이프라인 버그인지 구분이
어렵다. 그래서 먼저 **toy Gaussian 시뮬레이터**(잘 식별됨)로 sbi가 known param을 복원하는지
검증한다(파이프라인 sanity). 통과하면 같은 검증된 run_sbi를 ABM에 적용(scripts/...).
"""
import numpy as np
import pytest

sbi = pytest.importorskip("sbi")          # sbi 미설치 시 skip (ABC fallback 존재)

from simulation.abm.sbi_calibration import run_sbi


def test_sbi_recovers_known_params_on_toy():
    """★ TDD 핵심: x = θ + 소노이즈 (잘 식별된 toy)에서 sbi 사후가 진짜 θ를 복원.
    복원하면 파이프라인이 옳고, 그 다음 ABM 약식별 결과를 신뢰할 수 있다."""
    rng = np.random.default_rng(0)
    theta_true = np.array([0.3, 0.7])      # in [0,1]^2

    def simulator(theta):                   # 2-D in → 4-D summary out (잘 식별)
        n = rng.normal(0, 0.03, size=4)
        return np.array([theta[0], theta[1], theta[0] * theta[1], theta[0] + theta[1]]) + n

    x_obs = np.array([theta_true[0], theta_true[1],
                      theta_true[0] * theta_true[1], theta_true[0] + theta_true[1]])
    res = run_sbi(simulator, [0.0, 0.0], [1.0, 1.0], x_obs,
                  n_sims=500, n_posterior=1500, seed=42)
    mean = np.array(res["posterior_mean"])
    # ① 사후 평균이 진짜 θ 근처 (파이프라인이 정보를 추출)
    assert np.all(np.abs(mean - theta_true) < 0.15), f"복원 실패: {mean} vs {theta_true}"
    # ② 사후가 prior보다 좁음 (잘 식별된 toy는 CI < prior 폭)
    assert all(w < 0.6 for w in res["ci_width_vs_prior"]), \
        f"잘 식별된 toy인데 CI가 prior만큼 넓음: {res['ci_width_vs_prior']}"


def test_credible_intervals_well_formed():
    """CI 추출이 [lo<hi] 형태 + samples shape 정상."""
    rng = np.random.default_rng(1)

    def simulator(theta):
        return np.array([theta[0], theta[0] ** 2]) + rng.normal(0, 0.05, size=2)

    res = run_sbi(simulator, [0.0], [1.0], np.array([0.5, 0.25]),
                  n_sims=300, n_posterior=1000, seed=7)
    assert res["samples"].shape == (1000, 1)
    lo, hi = res["ci95"][0]
    assert lo < hi
    assert res["library"] == "sbi.NPE"
