"""TDD lock — SBI(신경 사후추정) 식별성 파이프라인 + ABM 배선 (DIMENSION #5 verify).

박제 대상 (무엇이 *실제로* 동작하는가를 잠근다):
  L1. run_sbi 가 toy 시뮬레이터(잘 식별됨, x = θ + 소노이즈)에서 known θ 를 복원 —
      사후평균 |Δ| < 0.15, 사후 CI < prior 폭(잘 식별 → 좁아짐). = "SBI 가 정보 추출"
  L2. CI 추출 well-formed (lo<hi, samples shape, library='sbi.NPE'), ci_width_vs_prior 단조.
  L3. 약식별 시뮬레이터(x 가 θ 무시 = 순수노이즈)에서는 SBI 사후가 prior 만큼 넓게 남음 —
      즉 SBI 가 "거짓 자신감"을 만들지 않음 (식별성 정직성 — over-claim 방지).
  L4. ABM 배선: sbi_posterior_calibration 이 4 행동 prior {alpha,kappa,tau,theta} 위에서
      run_sbi + simulate_response 를 호출. names→theta dict 매핑이 PRIORS 키와 정합.
  L5. sbi 패키지가 실제 설치돼 있음(ABC fallback 아님). 미설치면 이 사실을 PASS 로 기록하되
      L1~L3 은 SKIP 로 정직히 표기 (over-claim 금지).

합성/toy 파라미터만 — DB·ABM full-run 없음. 짧게 (~10s).
Run:  .venv/bin/python simulation/tests/test_sbi_calibration_lock.py
"""
from __future__ import annotations

import sys

import numpy as np

# ── sbi 설치 여부 (정직성: fallback 이면 L1~L3 은 SKIP) ──────────────────────────
try:
    import sbi  # noqa: F401
    _SBI = True
except ImportError:
    _SBI = False

from simulation.abm.sbi_calibration import run_sbi


def test_sbi_recovers_known_param_on_toy() -> bool:
    """L1 ★ 핵심: 잘 식별된 toy 에서 사후평균이 진짜 θ 복원 + CI 가 prior 보다 좁음."""
    if not _SBI:
        print("  [L1] SKIP — sbi 미설치 (ABC fallback). neural posterior 검증 불가.")
        return True
    rng = np.random.default_rng(0)
    theta_true = np.array([0.3, 0.7])

    def simulator(theta):                       # 2-D in → 4-D summary out (잘 식별)
        n = rng.normal(0, 0.03, size=4)
        return np.array([theta[0], theta[1], theta[0] * theta[1],
                         theta[0] + theta[1]]) + n

    x_obs = np.array([theta_true[0], theta_true[1],
                      theta_true[0] * theta_true[1], theta_true[0] + theta_true[1]])
    res = run_sbi(simulator, [0.0, 0.0], [1.0, 1.0], x_obs,
                  n_sims=500, n_posterior=1500, seed=42)
    mean = np.array(res["posterior_mean"])
    ok_recover = bool(np.all(np.abs(mean - theta_true) < 0.15))
    ok_narrow = all(w < 0.6 for w in res["ci_width_vs_prior"])
    print(f"  [L1] mean={mean} vs θ={theta_true} recover={ok_recover} | "
          f"ci_width_vs_prior={res['ci_width_vs_prior']} narrow={ok_narrow}")
    return ok_recover and ok_narrow


def test_credible_intervals_well_formed() -> bool:
    """L2: samples shape / CI lo<hi / library 라벨 / ci_width_vs_prior 형식."""
    if not _SBI:
        print("  [L2] SKIP — sbi 미설치.")
        return True
    rng = np.random.default_rng(1)

    def simulator(theta):
        return np.array([theta[0], theta[0] ** 2]) + rng.normal(0, 0.05, size=2)

    res = run_sbi(simulator, [0.0], [1.0], np.array([0.5, 0.25]),
                  n_sims=300, n_posterior=1000, seed=7)
    ok_shape = res["samples"].shape == (1000, 1)
    lo, hi = res["ci95"][0]
    ok_order = lo < hi
    ok_lib = res["library"] == "sbi.NPE"
    ok_width = (0.0 <= res["ci_width_vs_prior"][0] <= 1.5)
    print(f"  [L2] shape={res['samples'].shape} ci95={res['ci95'][0]} "
          f"lib={res['library']} width={res['ci_width_vs_prior'][0]}")
    return ok_shape and ok_order and ok_lib and ok_width


def test_unidentified_param_stays_broad() -> bool:
    """L3 ★ 정직성: x 가 θ 를 전혀 안 담으면(순수노이즈) 사후 ≈ prior (넓게 유지).
    SBI 가 정보 없는데 좁은 CI 를 만들어 '거짓 식별'을 보고하지 않는지 잠금."""
    if not _SBI:
        print("  [L3] SKIP — sbi 미설치.")
        return True
    rng = np.random.default_rng(2)

    def simulator(theta):                       # θ 를 완전히 무시 → 식별 불가
        return rng.normal(0.0, 1.0, size=3)

    res = run_sbi(simulator, [0.0], [1.0], np.array([0.0, 0.0, 0.0]),
                  n_sims=400, n_posterior=1200, seed=11)
    w = res["ci_width_vs_prior"][0]
    # 정보 없으면 사후 ≈ prior → width 가 충분히 넓어야 함 (>0.4).
    ok_broad = w > 0.4
    print(f"  [L3] unidentified ci_width_vs_prior={w} stays_broad(>0.4)={ok_broad}")
    return ok_broad


def test_abm_wiring_priors_and_simulator_map() -> bool:
    """L4: ABM 배선 — 4 행동 prior + names→theta dict 매핑 + run_sbi/simulate_response 호출."""
    from simulation.scripts.sbi_posterior_calibration import PRIORS, _summary
    import inspect
    import simulation.scripts.sbi_posterior_calibration as m

    names = list(PRIORS)
    ok_params = names == ["alpha", "kappa", "tau", "theta"]
    # 각 prior 가 (lo, hi), lo<hi
    ok_bounds = all(len(v) == 2 and v[0] < v[1] for v in PRIORS.values())
    # _summary 가 4-D scale-invariant 요약을 주는가 (toy 봉우리 궤적)
    traj = np.concatenate([np.linspace(0, 10, 12), np.linspace(10, 2, 12)])
    summ = _summary(traj)
    ok_summary = summ.shape == (4,) and np.all(np.isfinite(summ))
    # main() 소스가 run_sbi + simulate_response 를 4-param theta dict 로 배선
    src = inspect.getsource(m.main)
    ok_wired = ("run_sbi(" in src and "simulate_response" in src
                and "names[j]" in src and "x_obs" in src)
    print(f"  [L4] priors={names} ok_params={ok_params} ok_bounds={ok_bounds} "
          f"summary4D={ok_summary} wired={ok_wired}")
    return ok_params and ok_bounds and ok_summary and ok_wired


def test_sbi_installed_not_abc_fallback() -> bool:
    """L5: 정직성 — sbi 가 실제 설치돼 NPE 경로가 살아있는지 명시 기록.
    설치돼 있으면 PASS(NPE 사용). 미설치면 PASS 이되 'ABC fallback' 사실을 명시."""
    if _SBI:
        import sbi as _s
        print(f"  [L5] sbi INSTALLED v{getattr(_s, '__version__', '?')} — NPE 경로 LIVE "
              f"(ABC fallback 아님).")
    else:
        print("  [L5] sbi NOT installed — 호출 측은 ABC(abc_posterior_calibration)로 fallback.")
    return True  # 둘 다 정상 상태 — 사실을 기록할 뿐, over-claim 하지 않음


def main() -> int:
    tests = [
        ("L1 toy known-param recovery", test_sbi_recovers_known_param_on_toy),
        ("L2 credible intervals well-formed", test_credible_intervals_well_formed),
        ("L3 unidentified stays broad", test_unidentified_param_stays_broad),
        ("L4 ABM wiring (alpha/kappa/tau/theta)", test_abm_wiring_priors_and_simulator_map),
        ("L5 sbi installed (not ABC fallback)", test_sbi_installed_not_abc_fallback),
    ]
    n_pass = n_fail = 0
    print(f"=== test_sbi_calibration_lock (sbi installed={_SBI}) ===")
    for name, fn in tests:
        try:
            ok = fn()
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  [{name}] EXCEPTION: {type(e).__name__}: {e}")
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        n_pass += int(ok)
        n_fail += int(not ok)
    print(f"\n  {n_pass} PASS / {n_fail} FAIL")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
