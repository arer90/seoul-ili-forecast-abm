"""ABM cross-process DETERMINISM lock (the old "~1/3 발산 / CV20%" 근절 박제).

배경 (MEMORY: abm-mobility-orderby-rootcause):
  agent SEIR 의 프로세스-간 변동(R² 0.67~0.85, "CV20%")의 진짜 원인 두 가지 —
    (a) mobility SQL 의 ORDER BY 누락 → SQLite 가 프로세스마다 다른 행 순서로
        반환 → ``M[i,j] += coupling`` 누적 반올림이 프로세스마다 달라짐
        (수정 commit 95bdba9, simulation/sim/io.py:load_mobility_matrix).
    (b) RK4 integrator 의 FP-chaos → 안정 exp-Euler 를 2026-06-10 기본화하여
        근절 (run_coupled_abm / run_agent_abm 기본 = expeuler_step).

이 파일이 박제하는 INVARIANT:
  1. run_coupled_abm — 동일 (seed, params) 를 3~5 개 독립 OS 프로세스에서 돌리면
     city_I 궤적이 BYTE-IDENTICAL (max|Δ|=0). 안정 기본 integrator 에서 프로세스-간
     chaos 가 사라졌다.
  2. run_agent_abm — 동일하게 byte-identical (n_agents Monte-Carlo 도 seed 고정이면 결정적).
  3. mobility load 가 결정적 — commuter_matrix 의 물리적 행 삽입 순서를 셔플해도
     load_mobility_matrix(ORDER BY) 가 byte-identical (G,G) 행렬을 만든다.
     (실 스키마는 (origin,dest) 쌍이 unique=625 → ORDER BY origin,dest 가 total order.)

정직성 노트 (anti-overclaim):
  - 이 toy 파라미터에서는 레거시 RK4(MPH_STABLE_INTEGRATOR=0)조차 byte-identical 이다.
    원래의 발산은 특정 stiff regime / 실 25-gu mobility 행렬에서만 나타났다. 따라서
    이 테스트는 "RK4 가 toy 에서 발산한다"고 주장하지 않는다 — **안정 기본** 경로가
    프로세스-간 정확히 재현됨을 박제한다(사용자가 원하는 보장이자 2026-06-10 기본화의 약속).
  - mobility 테스트는 DB(13GB)를 건드리지 않고 simulation.sim.io.safe_connect 를
    in-memory 커넥션으로 monkeypatch 한다(read-only, hermetic).

Synthetic/toy params (no real DB). 실행:
    .venv/bin/python simulation/tests/test_abm_xprocess_determinism.py
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import random
import sqlite3
import subprocess
import sys
import textwrap

import numpy as np

import simulation.sim.io as io


# ── shared toy-run program used by the cross-process subprocesses ──────────
# Inlined so each subprocess is a fully-independent OS process (no shared
# module state, no shared RNG, no shared numba cache warmth) — exactly the
# condition under which the old ORDER-BY / RK4 jitter showed up.
_RUN_SRC = textwrap.dedent(
    """
    import sys, hashlib
    import numpy as np
    from simulation.abm.behavioural import BehaviouralParams, run_coupled_abm
    from simulation.abm.agent_based import run_agent_abm
    from simulation.sim.parameters import DEFAULT_FLU_PARAMS, MetapopParams

    def toy(G=5, days=40, dt=0.25, infected0=500.0):
        pops = np.full(G, 100_000.0)
        M = np.full((G, G), 0.05 / (G - 1)); np.fill_diagonal(M, 0.95)
        return MetapopParams(
            disease=DEFAULT_FLU_PARAMS, populations=pops, mobility=M,
            district_names=[f"d{i}" for i in range(G)],
            initial_infected=np.full(G, infected0), days=days, dt=dt, seed=0)

    b = BehaviouralParams(alpha=2.0, kappa=0.3, tau=90.0, theta=0.1)
    which = sys.argv[1]
    if which == "coupled":
        ci = run_coupled_abm(toy(), b).city_I()
    elif which == "agent":
        ci = run_agent_abm(toy(), b, n_agents=200, seed=1).city_I()
    else:
        raise SystemExit("unknown run kind: " + which)
    ci = np.ascontiguousarray(ci, dtype=np.float64)
    # emit a sha256 of the raw bytes so the parent can compare without parsing floats
    sys.stdout.write(hashlib.sha256(ci.tobytes()).hexdigest())
    """
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _spawn_hash(kind: str, env_overrides: dict | None = None) -> str:
    """Run the toy ABM in a brand-new OS process and return the city_I sha256.

    Each call is an independent ``subprocess.run`` (own interpreter, own memory,
    own RNG state) — the genuine multi-process condition.
    """
    env = dict(os.environ)
    # Make the run hermetic + deterministic across machines: no thread-level FP
    # reordering and no stable-integrator surprise unless explicitly overridden.
    env.setdefault("MPH_STABLE_INTEGRATOR", "1")
    if env_overrides:
        env.update(env_overrides)
    proc = subprocess.run(
        [sys.executable, "-c", _RUN_SRC, kind],
        cwd=_REPO_ROOT, env=env,
        capture_output=True, text=True, encoding="utf-8", timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"subprocess {kind} failed (rc={proc.returncode}):\n{proc.stderr[-2000:]}"
        )
    out = proc.stdout.strip()
    if len(out) != 64:
        raise RuntimeError(f"unexpected hash output for {kind}: {out!r}")
    return out


# ── 1. run_coupled_abm cross-process byte-identity ────────────────────────
def test_coupled_xprocess_byte_identical():
    """동일 (seed, params) 를 4 개 독립 OS 프로세스 → city_I sha256 전부 동일."""
    hashes = [_spawn_hash("coupled") for _ in range(4)]
    assert len(set(hashes)) == 1, (
        "run_coupled_abm 가 프로세스-간 비결정적 (chaos 미근절): "
        + " ".join(h[:12] for h in hashes)
    )


# ── 2. run_agent_abm cross-process byte-identity ──────────────────────────
def test_agent_xprocess_byte_identical():
    """agent ABM (n_agents Monte-Carlo, seed 고정) 도 5 개 독립 프로세스서 동일."""
    hashes = [_spawn_hash("agent") for _ in range(5)]
    assert len(set(hashes)) == 1, (
        "run_agent_abm 가 프로세스-간 비결정적: "
        + " ".join(h[:12] for h in hashes)
    )


# ── 3. exact max|Δ|=0 cross-process (not just hash equality) ───────────────
def test_coupled_xprocess_max_abs_delta_zero():
    """해시뿐 아니라 실제 trajectory 차이 max|Δ| == 0.0 임을 in-process 로도 박제.

    동일 프로세스 내 2 회 실행이 byte-identical 이어야(결정성의 필요조건).
    cross-process 동일성(테스트 1·2)과 합치면 max|Δ|=0 의 완전 증거.
    """
    from simulation.abm.behavioural import BehaviouralParams, run_coupled_abm
    from simulation.sim.parameters import DEFAULT_FLU_PARAMS, MetapopParams

    def toy():
        G = 5
        pops = np.full(G, 100_000.0)
        M = np.full((G, G), 0.05 / (G - 1)); np.fill_diagonal(M, 0.95)
        return MetapopParams(
            disease=DEFAULT_FLU_PARAMS, populations=pops, mobility=M,
            district_names=[f"d{i}" for i in range(G)],
            initial_infected=np.full(G, 500.0), days=40, dt=0.25, seed=0)

    b = BehaviouralParams(alpha=2.0, kappa=0.3, tau=90.0, theta=0.1)
    prev = os.environ.get("MPH_STABLE_INTEGRATOR")
    os.environ["MPH_STABLE_INTEGRATOR"] = "1"
    try:
        a = run_coupled_abm(toy(), b).city_I()
        c = run_coupled_abm(toy(), b).city_I()
    finally:
        if prev is None:
            os.environ.pop("MPH_STABLE_INTEGRATOR", None)
        else:
            os.environ["MPH_STABLE_INTEGRATOR"] = prev
    assert float(np.max(np.abs(a - c))) == 0.0, "동일 입력 재실행이 byte-identical 아님"
    assert np.all(np.isfinite(a)) and a.max() > 0.0


# ── mobility-load determinism (ORDER BY) ──────────────────────────────────
_SEOUL_FOUR = ["강남구", "서초구", "송파구", "강동구"]


@contextlib.contextmanager
def _patched_safe_connect(rows):
    """Yield an in-memory commuter_matrix seeded with ``rows`` (origin,dest,coupling).

    Patches simulation.sim.io.safe_connect so load_mobility_matrix reads the
    toy table instead of the 13GB production DB (hermetic, read-only).
    """
    orig = io.safe_connect

    @contextlib.contextmanager
    def _sc(*a, **k):
        con = sqlite3.connect(":memory:")
        con.execute(
            "CREATE TABLE commuter_matrix "
            "(origin_gu TEXT, dest_gu TEXT, coupling REAL, night_population REAL)"
        )
        con.executemany(
            "INSERT INTO commuter_matrix(origin_gu,dest_gu,coupling) VALUES (?,?,?)",
            rows,
        )
        con.commit()
        try:
            yield con
        finally:
            con.close()

    io.safe_connect = _sc
    try:
        yield
    finally:
        io.safe_connect = orig


def _unique_pair_rows(seed: int) -> list[tuple[str, str, float]]:
    """One row per (origin,dest) — mirrors the REAL schema (625 = 25x25 unique)."""
    rng = random.Random(seed)
    rows = []
    for o in _SEOUL_FOUR:
        for d in _SEOUL_FOUR:
            rows.append((o, d, round(rng.uniform(1e-4, 1e-2), 9)))
    return rows


def test_mobility_load_has_order_by():
    """소스-수준 락: load_mobility_matrix 의 SQL 에 ORDER BY 가 존재.

    ORDER BY 제거(95bdba9 회귀)를 정적으로 차단. 함수 소스 inspect.
    """
    import inspect

    src = inspect.getsource(io.load_mobility_matrix)
    assert "ORDER BY" in src.upper(), (
        "load_mobility_matrix 의 SQL 에서 ORDER BY 가 사라짐 — "
        "프로세스-간 누적 반올림 jitter 재발(95bdba9 회귀)"
    )


def test_mobility_load_shuffle_invariant():
    """행동 락: commuter_matrix 의 물리적 삽입 순서를 셔플해도 M 이 byte-identical.

    실 스키마처럼 (origin,dest) unique 행을 사용 → ORDER BY origin,dest 가 total
    order 라 누적 순서가 프로세스/삽입순서와 무관하게 고정된다.
    """
    rows = _unique_pair_rows(seed=11)

    with _patched_safe_connect(rows):
        M_a = io.load_mobility_matrix(_SEOUL_FOUR)

    shuffled = list(rows)
    random.Random(987).shuffle(shuffled)
    assert shuffled != rows, "셔플이 실제로 순서를 바꿔야 테스트가 의미 있음"
    with _patched_safe_connect(shuffled):
        M_b = io.load_mobility_matrix(_SEOUL_FOUR)

    assert M_a.tobytes() == M_b.tobytes(), (
        "삽입 순서 셔플에 mobility 행렬이 비결정적 — ORDER BY 가 누적 순서를 "
        f"고정하지 못함 (max|Δ|={float(np.max(np.abs(M_a - M_b))):.2e})"
    )
    # 추가: 행-확률성 invariant 유지(정상 행렬임을 보장).
    assert np.allclose(M_a.sum(axis=1), 1.0), "row-stochastic 위반"


def test_mobility_load_cross_process_identical():
    """진짜 다중-프로세스: 동일 unique-pair 테이블을 2 개 독립 프로세스서 로드 →
    M 의 sha256 동일. (in-memory patch 를 각 subprocess 안에서 재현.)"""
    src = textwrap.dedent(
        """
        import sys, json, hashlib, contextlib, sqlite3
        import numpy as np
        import simulation.sim.io as io
        rows = json.loads(sys.argv[1])
        rows = [tuple(r) for r in rows]
        districts = json.loads(sys.argv[2])
        @contextlib.contextmanager
        def _sc(*a, **k):
            con = sqlite3.connect(":memory:")
            con.execute("CREATE TABLE commuter_matrix (origin_gu TEXT, dest_gu TEXT, coupling REAL, night_population REAL)")
            con.executemany("INSERT INTO commuter_matrix(origin_gu,dest_gu,coupling) VALUES (?,?,?)", rows)
            con.commit()
            try: yield con
            finally: con.close()
        io.safe_connect = _sc
        M = io.load_mobility_matrix(districts)
        M = np.ascontiguousarray(M, dtype=np.float64)
        sys.stdout.write(hashlib.sha256(M.tobytes()).hexdigest())
        """
    )
    import json

    rows = _unique_pair_rows(seed=5)
    # two different physical orders, each in its own OS process
    order_a = list(rows)
    order_b = list(rows)
    random.Random(404).shuffle(order_b)
    args_a = [sys.executable, "-c", src, json.dumps(order_a), json.dumps(_SEOUL_FOUR)]
    args_b = [sys.executable, "-c", src, json.dumps(order_b), json.dumps(_SEOUL_FOUR)]
    h_a = subprocess.run(args_a, cwd=_REPO_ROOT, capture_output=True, text=True, encoding="utf-8", timeout=120)
    h_b = subprocess.run(args_b, cwd=_REPO_ROOT, capture_output=True, text=True, encoding="utf-8", timeout=120)
    assert h_a.returncode == 0, h_a.stderr[-1500:]
    assert h_b.returncode == 0, h_b.stderr[-1500:]
    assert h_a.stdout.strip() == h_b.stdout.strip() != "", (
        "다중-프로세스 mobility 로드가 비결정적: "
        f"{h_a.stdout.strip()[:12]} vs {h_b.stdout.strip()[:12]}"
    )


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
