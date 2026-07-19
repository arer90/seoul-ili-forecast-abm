"""ABM integrator-gate wiring + calibrate divergence guard (audit MEDIUM fix).

박제:
  - B-P1 안정 integrator(expeuler_step)가 behavioural/agent ABM 경로에서 reachable —
    MPH_STABLE_INTEGRATOR=1 이면 expeuler 가 실제로 호출되고, 미설정이면 rk4(back-compat).
    (이전: ABM 이 rk4 하드코딩이라 안정solver 가 영원히 미사용 — 감사 MEDIUM broken-wiring).
  - calibrate._evaluate 가 발산 run(비유한/>1e12 peak)을 loss=inf 로 거부(validate_real 처럼).
    (이전: 가드 없어 nan_to_num→0 으로 'died-out' 처럼 보이는 발산이 점수에 들어감).

Synthetic params (no DB). Run:  .venv/bin/python simulation/tests/test_abm_integrator_gate.py
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np

import simulation.abm.behavioural as bh
import simulation.abm.agent_based as ab
import simulation.abm.calibrate as cal
from simulation.abm.behavioural import BehaviouralParams, run_coupled_abm
from simulation.abm.agent_based import run_agent_abm
from simulation.sim.parameters import DEFAULT_FLU_PARAMS, MetapopParams

REBOUND = BehaviouralParams(alpha=2.0, kappa=0.3, tau=90.0, theta=0.1)


def _toy(G=5, days=30, dt=0.25, infected0=500.0):
    pops = np.full(G, 100_000.0)
    M = np.full((G, G), 0.05 / (G - 1)); np.fill_diagonal(M, 0.95)
    return MetapopParams(
        disease=DEFAULT_FLU_PARAMS, populations=pops, mobility=M,
        district_names=[f"d{i}" for i in range(G)],
        initial_infected=np.full(G, infected0), days=days, dt=dt, seed=0)


class _Spy:
    """call-counting wrapper preserving the wrapped fn's behaviour."""
    def __init__(self, fn):
        self._fn = fn; self.calls = 0
    def __call__(self, *a, **k):
        self.calls += 1; return self._fn(*a, **k)


def _with_env(val, fn):
    prev = os.environ.get("MPH_STABLE_INTEGRATOR")
    try:
        if val is None:
            os.environ.pop("MPH_STABLE_INTEGRATOR", None)
        else:
            os.environ["MPH_STABLE_INTEGRATOR"] = val
        return fn()
    finally:
        if prev is None:
            os.environ.pop("MPH_STABLE_INTEGRATOR", None)
        else:
            os.environ["MPH_STABLE_INTEGRATOR"] = prev


def _routes(mod, env_val):
    """monkeypatch the module's rk4/expeuler with spies, run, return (rk4_calls, exp_calls)."""
    rk4_spy, exp_spy = _Spy(mod.rk4_step), _Spy(mod.expeuler_step)
    o_rk4, o_exp = mod.rk4_step, mod.expeuler_step
    mod.rk4_step, mod.expeuler_step = rk4_spy, exp_spy
    try:
        runner = (lambda: run_coupled_abm(_toy(), REBOUND)) if mod is bh \
            else (lambda: run_agent_abm(_toy(), REBOUND, n_agents=200, seed=1))
        _with_env(env_val, runner)
    finally:
        mod.rk4_step, mod.expeuler_step = o_rk4, o_exp
    return rk4_spy.calls, exp_spy.calls


def test_behavioural_default_is_stable():
    """behavioural: 기본(미설정/=1) → expeuler(안정), MPH_STABLE_INTEGRATOR=0 → 레거시 rk4.
    2026-06-10: 안정 integrator 를 기본화(발산 근절). rk4 는 명시 opt-out 만."""
    rk4_def, exp_def = _routes(bh, None)
    assert exp_def > 0 and rk4_def == 0, f"기본인데 expeuler(안정) 미호출 (rk4={rk4_def}, exp={exp_def})"
    rk4_on, exp_on = _routes(bh, "1")
    assert exp_on > 0 and rk4_on == 0, f"=1 인데 expeuler 미호출 (rk4={rk4_on}, exp={exp_on})"
    rk4_legacy, exp_legacy = _routes(bh, "0")
    assert rk4_legacy > 0 and exp_legacy == 0, f"=0(레거시)인데 rk4 미사용 (rk4={rk4_legacy}, exp={exp_legacy})"


def test_agent_default_is_stable():
    """agent_based: 동일 — 기본 expeuler, =0 만 레거시 rk4."""
    rk4_def, exp_def = _routes(ab, None)
    assert exp_def > 0 and rk4_def == 0, f"agent 기본인데 expeuler 미호출 (rk4={rk4_def}, exp={exp_def})"
    rk4_legacy, exp_legacy = _routes(ab, "0")
    assert rk4_legacy > 0 and exp_legacy == 0, f"agent =0 인데 rk4 미사용 (rk4={rk4_legacy}, exp={exp_legacy})"


def test_stable_path_finite_nonneg():
    """안정 경로 출력 = 유한 + 비음수 (band-aid 없이도 안전)."""
    out = _with_env("1", lambda: run_coupled_abm(_toy(), REBOUND))
    ci = out.city_I()
    assert np.all(np.isfinite(ci)) and np.all(ci >= 0) and ci.max() > 0


def test_calibrate_rejects_diverged_run():
    """발산 run(>1e12 peak) → _evaluate peak=NaN, loss=inf (min-선택서 제외)."""
    class _FakeOut:
        def __init__(self, peak):
            self._I = np.array([1.0, peak, 1.0]); self.compliance = np.zeros((3, 4))
        def city_I(self):
            return self._I
    orig = cal.run_coupled_abm
    cal.run_coupled_abm = lambda p, b: _FakeOut(1e15)
    try:
        r = cal._evaluate(_toy(), REBOUND)
    finally:
        cal.run_coupled_abm = orig
    assert math.isnan(r.peak_val), f"발산인데 peak_val 유한: {r.peak_val}"
    assert math.isinf(r.loss), f"발산인데 loss 유한: {r.loss}"


def test_calibrate_accepts_normal_run():
    """정상 run 은 통과(가드가 정상값 막지 않음)."""
    class _FakeOut:
        def __init__(self, peak):
            self._I = np.array([1.0, peak, 1.0]); self.compliance = np.zeros((3, 4))
        def city_I(self):
            return self._I
    orig = cal.run_coupled_abm
    cal.run_coupled_abm = lambda p, b: _FakeOut(45000.0)
    try:
        r = cal._evaluate(_toy(), REBOUND)
    finally:
        cal.run_coupled_abm = orig
    assert math.isfinite(r.peak_val) and abs(r.peak_val - 45000.0) < 1.0


if __name__ == "__main__":
    funcs = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    p = f = 0
    for fn in funcs:
        try:
            fn(); print(f"  ✓ PASS  {fn.__name__}"); p += 1
        except Exception as e:
            print(f"  ✗ FAIL  {fn.__name__}: {e}"); f += 1
    print(f"\n  {p} PASS / {f} FAIL")
    sys.exit(1 if f else 0)
