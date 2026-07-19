"""G-266 (2026-06-13, ABM 정밀 audit): run_invariant_test 가 kernel·ABM 적분기 비대칭에도 통과.

audit 적발(적대적 재검증 TRUE POSITIVE): kernel 기본=RK4(metapop_seirvd.py:257) vs ABM 기본=exp-Euler
(behavioural.py:344, 2026-06-10~). env 미설정(운영 기본) 시 invariant 가 *적분기 차이*를 잡아 passed=False
(rmse≈5508) → verify_all.check_abm_invariant latent 회귀. fix=run_invariant_test 가 양측을 동일 적분기로
고정(MPH_STABLE_INTEGRATOR=1) 후 비교 → byte-exact. 이 테스트가 운영 기본서 통과 + env 복원을 가드.

Run: KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python -m pytest simulation/tests/test_g266_invariant_integrator.py -x -q
"""
from __future__ import annotations

import os


def _toy():
    from simulation.abm.behavior_disease_eval import build_demo_metapop
    return build_demo_metapop(G=5, days=120, R0=1.5)


def test_invariant_passes_under_operational_default():
    """env 미설정(운영 기본)서 behaviour-off ABM == kernel (byte-exact)."""
    from simulation.abm.behavioural import run_invariant_test
    os.environ.pop("MPH_STABLE_INTEGRATOR", None)        # 운영 기본 = 미설정
    r = run_invariant_test(_toy())
    assert r["passed"], f"invariant FAIL under operational default: rmse={r['rmse']:.3e}"
    assert r["rmse"] < 1e-6 and r["abm_mean_compliance"] == 0.0


def test_invariant_restores_env():
    """run_invariant_test 가 MPH_STABLE_INTEGRATOR 를 누수 없이 복원."""
    from simulation.abm.behavioural import run_invariant_test
    # (a) 미설정 → 미설정 유지
    os.environ.pop("MPH_STABLE_INTEGRATOR", None)
    run_invariant_test(_toy())
    assert "MPH_STABLE_INTEGRATOR" not in os.environ, "env 누수 (미설정→설정됨)"
    # (b) 사용자값 → 그대로 복원
    os.environ["MPH_STABLE_INTEGRATOR"] = "0"
    try:
        run_invariant_test(_toy())
        assert os.environ.get("MPH_STABLE_INTEGRATOR") == "0", "사용자 env 값 미복원"
    finally:
        os.environ.pop("MPH_STABLE_INTEGRATOR", None)


if __name__ == "__main__":
    test_invariant_passes_under_operational_default(); print("PASS operational-default")
    test_invariant_restores_env(); print("PASS env-restore")
    print("=== ALL PASS ===")
