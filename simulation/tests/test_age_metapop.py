"""Age-structured WAIFW metapopulation SEIR-V-D (#4 full age_i×age_j mixing) — TDD.

박제(기존 scalar/agent 는 age 를 row-mean 가중만; 여기선 진짜 WAIFW):
  - normalize_contact: 스펙트럼 반경 1 (β=R0·γ 해석 유지), A=1 → [[1]].
  - age_foi: λ_{i,a}=β·Σ_b Ĉ[a,b]·(M@(I_b/N_b)) — 공간(M)+나이(C) 이중결합, 손계산 일치.
  - 질량보존(S+E+I+R+V+D)·비음수, A=1 시 scalar commuter SEIR 로 정확 환원.
  - **WAIFW signature**: 한 age-band 만 seed → 초기 교차감염이 그 band 의 접촉 column Ĉ[:,b] 비율로
    퍼짐(scalar/row-mean 모델은 불가능 — age_i×age_j 의 핵심 증거).
  - 균일 접촉 → age 효과 0(거짓양성 가드).
  - 실데이터 검증: 모델 age attack-rate 가 실제 sentinel age-ILI gradient(학령기 高·노년 低)와 양의 상관.

Run:  .venv/bin/python simulation/tests/test_age_metapop.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from simulation.sim.age_metapop import (  # noqa: E402
    normalize_contact, age_foi, run_age_metapop, age_expeuler_step, AgeDiseaseParams)
from simulation.sim.parameters import IDX_S, IDX_I  # noqa: E402
from simulation.abm.agent_kernel import CONTACT_MATRIX_7x7  # noqa: E402


def _spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    return float((rx @ ry) / (np.sqrt(rx @ rx) * np.sqrt(ry @ ry) + 1e-12))


def _toy(G=3, A=7, pop=1e5):
    populations = np.full((G, A), pop)
    M = np.full((G, G), 0.1); np.fill_diagonal(M, 0.8)
    return populations, M


def test_normalize_contact_spectral_radius_one():
    Cn = normalize_contact(CONTACT_MATRIX_7x7)
    rho = float(np.max(np.abs(np.linalg.eigvals(Cn))))
    assert abs(rho - 1.0) < 1e-9, rho


def test_normalize_contact_A1_identity():
    assert np.allclose(normalize_contact(np.array([[3.7]])), [[1.0]])


def test_age_foi_known_values():
    """손계산: G=2,A=2, Ĉ=I(정규화 후), M=I → λ_{i,a}=β·I_{i,a}/N_{i,a}."""
    C = np.eye(2) * 2.0          # 정규화 시 Ĉ=I (ρ=2)
    M = np.eye(2)
    I = np.array([[10.0, 0.0], [0.0, 20.0]]); N = np.full((2, 2), 100.0)
    lam = age_foi(I, N, beta=0.5, contact_norm=normalize_contact(C), mobility=M)
    assert np.allclose(lam, [[0.5 * 0.1, 0.0], [0.0, 0.5 * 0.2]]), lam


def test_mass_conserved_and_nonneg():
    pop, M = _toy()
    seed = np.zeros_like(pop); seed[0, 1] = 100
    r = run_age_metapop(pop, M, CONTACT_MATRIX_7x7, AgeDiseaseParams(beta=0.6, delta=0.001),
                        initial_infected_age=seed, days=80, dt=0.5)
    tot = r.S + r.E + r.I + r.R + r.V + r.D
    assert np.max(np.abs(tot - pop[None])) < 1e-6, "질량 비보존"
    for arr in (r.S, r.E, r.I, r.R, r.V, r.D):
        assert np.all(arr >= -1e-9), "음수 발생"


def test_reduces_to_scalar_when_A1():
    """A=1 + Ĉ=[[1]] → scalar commuter SEIR exp-Euler 와 정확 일치."""
    G = 4
    pop = np.full((G, 1), 1e5); M = np.full((G, G), 0.05); np.fill_diagonal(M, 0.8)
    seed = np.zeros((G, 1)); seed[0, 0] = 200
    dis = AgeDiseaseParams(beta=0.5, sigma=1/1.5, gamma=1/3, delta=0.0)
    r = run_age_metapop(pop, M, np.array([[2.0]]), dis, initial_infected_age=seed, days=60, dt=0.5)
    # 독립 scalar 참조: 같은 exp-Euler, λ=β·(M@(I/N))
    state = np.zeros((G, 1, 6)); state[:, 0, IDX_I] = seed[:, 0]; state[:, 0, IDX_S] = pop[:, 0] - seed[:, 0]
    kw = {"beta": 0.5, "sigma": 1/1.5, "gamma": 1/3, "delta": 0.0, "nu": 0.0, "omega": 0.0,
          "v_waning": 0.0, "contact_norm": np.array([[1.0]]), "mobility": M, "populations_age": pop}
    for _ in range(60):
        for _ in range(2):
            state = age_expeuler_step(state, 0.5, params_kwargs=kw)
    assert np.max(np.abs(r.I[-1] - state[:, :, IDX_I])) < 1e-6, "A=1 이 scalar 로 환원 안 됨"


def test_waifw_spreads_by_contact_column():
    """핵심: band b 만 seed → 초기 교차-age 발생이 Ĉ[:,b] 비율(WAIFW). row-mean 모델은 불가."""
    pop, M = _toy(G=1)            # 단일 gu 로 공간효과 제거, age 혼합만
    b = 1                         # 고접촉 band seed
    seed = np.zeros_like(pop); seed[0, b] = 500
    r = run_age_metapop(pop, M, CONTACT_MATRIX_7x7, AgeDiseaseParams(beta=0.8),
                        initial_infected_age=seed, days=6, dt=0.5)
    early = r.incidence[:4].sum(axis=(0, 1))    # (A,) 초기 누적 발생
    Cn = normalize_contact(CONTACT_MATRIX_7x7)
    others = [a for a in range(7) if a != b]
    rank_corr = _spearman(early[others], Cn[others, b])   # 교차감염 ∝ 접촉 column
    assert rank_corr > 0.8, f"교차-age 발생이 접촉 column 을 안 따름(WAIFW 아님): ρ={rank_corr:.2f}"


def test_uniform_contact_no_age_effect():
    """균일 접촉행렬 → 모든 age 동일 attack(거짓 age 효과 없음)."""
    pop, M = _toy()
    seed = np.zeros_like(pop); seed[0, :] = 20   # 모든 age 동일 seed
    C_flat = np.ones((7, 7))
    r = run_age_metapop(pop, M, C_flat, AgeDiseaseParams(beta=0.6),
                        initial_infected_age=seed, days=80, dt=0.5)
    ar = r.attack_rate_by_age(pop)
    assert ar.std() / ar.mean() < 1e-3, f"균일 접촉인데 age 차등 발생: {np.round(ar,4)}"


def test_age_attack_follows_contact_structure():
    """이질 접촉 → age attack-rate 가 접촉 row-mean 순위를 따름(고접촉=고발생)."""
    pop, M = _toy()
    seed = np.zeros_like(pop); seed[0, :] = 10
    r = run_age_metapop(pop, M, CONTACT_MATRIX_7x7, AgeDiseaseParams(beta=0.55),
                        initial_infected_age=seed, days=120, dt=0.5)
    ar = r.attack_rate_by_age(pop)
    rho = _spearman(ar, CONTACT_MATRIX_7x7.mean(axis=1))
    assert ar.std() / ar.mean() > 0.1, "age attack 가 사실상 균일(WAIFW 효과 없음)"
    assert rho > 0.8, f"age attack 가 접촉구조를 안 따름: ρ={rho:.2f}"


def _real_ili_by_decade_band(db):
    """실제 sentinel 버킷 ILI → 모델의 10년-밴드(_age_band_from_label: 0-9,10-19,…,60+)로
    **연-단위 crosswalk** 정렬(정직: 버킷↔decade 경계가 안 맞으므로 연령별로 매핑 후 밴드평균)."""
    import sqlite3
    con = sqlite3.connect(str(db))
    bk = {ag: con.execute("SELECT AVG(ili_rate) FROM sentinel_influenza WHERE age_group=?", (ag,)).fetchone()[0]
          for ag in ["0세", "1-6세", "7-12세", "13-18세", "19-49세", "50-64세", "65세 이상"]}
    con.close()
    if any(v is None for v in bk.values()):
        return None

    def yr(y):
        return (bk["0세"] if y == 0 else bk["1-6세"] if y <= 6 else bk["7-12세"] if y <= 12
                else bk["13-18세"] if y <= 18 else bk["19-49세"] if y <= 49
                else bk["50-64세"] if y <= 64 else bk["65세 이상"])
    bands = []
    for b in range(7):
        yrs = range(b * 10, b * 10 + 10) if b < 6 else range(60, 80)
        bands.append(float(np.mean([yr(y) for y in yrs])))
    return np.array(bands)


def test_validation_vs_real_sentinel_age_gradient():
    """실데이터: 모델 decade-밴드 attack-rate 가 실제 sentinel age-ILI gradient 와 양의 상관.
    정직: 7밴드=10년단위(0-9…60+), sentinel 버킷과 경계가 달라 연-crosswalk 로 정렬해야 함.
    측정 ρ≈0.68 — 학령기 高·노년 低 는 재현하나 working-age 는 실제가 더 평탄(접촉만으론 불완전)."""
    db = ROOT / "simulation" / "data" / "db" / "epi_real_seoul.db"
    if not db.is_file():
        print("  (DB 없음 — 검증 skip)"); return
    real = _real_ili_by_decade_band(db)
    if real is None:
        print("  (age 버킷 불일치 — skip)"); return
    pop, M = _toy()
    seed = np.zeros_like(pop); seed[0, :] = 10
    r = run_age_metapop(pop, M, CONTACT_MATRIX_7x7, AgeDiseaseParams(beta=0.55),
                        initial_infected_age=seed, days=120, dt=0.5)
    model = r.attack_rate_by_age(pop)
    rho = _spearman(model, real)
    print(f"  실제 ILI(decade crosswalk): {list(np.round(real,1))}")
    print(f"  모델 attack-rate:          {list(np.round(model,3))}")
    print(f"  Spearman(model, real-decade) = {rho:.2f}")
    assert model.argmax() <= 1, "모델 peak 가 학령기(밴드 0-1)가 아님"        # 10-19 peak
    assert model.argmin() == 6, "모델 trough 가 노년(60+)이 아님"
    assert rho > 0.5, f"모델 age 패턴이 실제와 양의 상관 부족(ρ={rho:.2f})"


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
