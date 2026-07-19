"""Sobol 전역민감도 (분산 분해, variance-based global sensitivity).

기존 `simulation.abm.sensitivity` 는 LHS + PRCC (단조 상관 순위)만 제공한다.
PRCC 는 *단조* 영향만 잡고 **상호작용(interaction)** 을 분리하지 못한다.
Sobol 분산분해(Saltelli 2010, CPC 181:259; Sobol 2001, MCS 55:271; Jansen
1999, CPC 120:1)는 출력 분산 Var(Y) 를 입력별 기여로 정확히 분해한다:

    Var(Y) = Σᵢ Vᵢ + Σ_{i<j} V_{ij} + …          (ANOVA-HDMR 분해)

  - **First-order Sᵢ = Vᵢ / Var(Y)**: 파라미터 i *단독* 의 주효과 (상호작용 제외).
  - **Total-order STᵢ = 1 − V_{~i}/Var(Y)**: i 가 관여한 *모든* 효과 (주효과 +
    i 를 포함하는 모든 상호작용). STᵢ ≥ Sᵢ 항상 성립.

가산모델(additive) 에서는 상호작용이 없어 STᵢ ≈ Sᵢ, ΣSᵢ ≈ 1.
상호작용 모델에서는 STᵢ > Sᵢ, ΣSᵢ < 1.

추정량 (estimator):
  - Sᵢ: Saltelli 2010 식(b) — Vᵢ = (1/n)Σ f(B)·(f(A_B^i) − f(A)).
  - STᵢ: Jansen 1999 — V_{~i} = (1/2n)Σ (f(A) − f(A_B^i))², STᵢ = V_{~i}/Var.

의존성-free (numpy 만). 결정성 (`np.random.default_rng(seed)`).
ABM kernel 직접 호출 불요 — 임의 `model_fn(params: dict) -> float` 에 동작
(합성 test 함수로 불변식 검증 가능). 스모크: `python -m simulation.abm.sobol_sensitivity`.
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence

import numpy as np


def _saltelli_matrices(
    param_bounds: Mapping[str, tuple[float, float]],
    n_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[str]]:
    """Saltelli 교차-샘플 행렬 A, B, 그리고 A_B^i (i=각 파라미터) 를 만든다.

    A, B 는 독립 준-난수 표본 (각 (n, p)); A_B^i 는 A 의 모든 열을 쓰되 i 번째
    열만 B 에서 가져온 행렬. 출력 평가는 A·B·{A_B^i} 총 n(p+2) 회.

    Args:
        param_bounds: 파라미터명 → (low, high). 순서가 출력 인덱스 순서.
        n_samples: 기저 표본 수 n (총 모델 평가 = n*(p+2)).
        rng: 결정성 난수원.

    Returns:
        (A, B, AB_list, names) — A,B 는 (n,p), AB_list 는 길이 p 의 (n,p) 배열들.
    """
    names = list(param_bounds)
    p = len(names)
    lows = np.array([param_bounds[k][0] for k in names], float)
    highs = np.array([param_bounds[k][1] for k in names], float)
    span = highs - lows

    # 단위입방체 [0,1)^p 에서 두 독립 표본 → 실제 범위로 스케일
    A_unit = rng.random((n_samples, p))
    B_unit = rng.random((n_samples, p))
    A = lows + A_unit * span
    B = lows + B_unit * span

    AB_list: list[np.ndarray] = []
    for i in range(p):
        ab = A.copy()
        ab[:, i] = B[:, i]
        AB_list.append(ab)
    return A, B, AB_list, names


def _evaluate(model_fn: Callable[[dict], float], M: np.ndarray, names: Sequence[str]) -> np.ndarray:
    """행렬 M 의 각 행을 dict 로 만들어 model_fn 평가 → (n,) 결과 벡터."""
    n = M.shape[0]
    out = np.empty(n, float)
    for r in range(n):
        params = {name: float(M[r, j]) for j, name in enumerate(names)}
        out[r] = float(model_fn(params))
    return out


def sobol_indices(
    model_fn: Callable[[dict], float],
    param_bounds: Mapping[str, tuple[float, float]],
    *,
    n_samples: int = 1024,
    seed: int = 42,
) -> dict:
    """Sobol first-order(S1) · total-order(ST) 민감도 지수 (분산 분해).

    Saltelli 교차-샘플 설계로 출력 분산을 입력별로 분해한다. 모델은 임의의
    스칼라-반환 함수 — ABM kernel 또는 합성 test 함수 모두 가능.

    추정량:
        Vᵢ  = (1/n) Σ f(B)·(f(A_B^i) − f(A))            (Saltelli 2010, S1)
        V_{~i} = (1/2n) Σ (f(A) − f(A_B^i))²            (Jansen 1999, ST)
        S1ᵢ = Vᵢ / Var(Y),  STᵢ = V_{~i} / Var(Y)

    Args:
        model_fn: ``params(dict: name→float) -> float`` 스칼라 출력 모델.
            범위 내에서 deterministic 가정 (stochastic 모델은 seed 고정/평균화
            를 호출자가 책임 — caller responsibility).
        param_bounds: 파라미터명 → (low, high) 균등 표본 범위. low < high.
            dict 순서가 결과 인덱스 순서.
        n_samples: 기저 표본 수 n. 총 모델 평가 = n*(p+2). 수렴엔 2의 거듭제곱
            권장 (≥512; 노이즈 작은 함수는 256 도 충분).
        seed: 결정성 시드.

    Returns:
        ``{"names": [...], "S1": {name: float}, "ST": {name: float},
           "S1_array": np.ndarray, "ST_array": np.ndarray,
           "var_Y": float, "n_samples": n, "n_evals": n*(p+2)}``.
        Var(Y)=0 (상수 출력) 이면 모든 지수 0.0.

    Raises:
        ValueError: param_bounds 비었거나, 어떤 low ≥ high, n_samples < 2.

    Performance: 모델 평가 n*(p+2) 회 — model_fn 비용이 지배적. 메모리 O(n*p).
    Side effects: 없음 (순수 compute). model_fn 부작용은 호출자 책임.
    Caller responsibility: model_fn 은 param_bounds 범위 전역에서 정의·유한해야
        하며, stochastic 이면 seed 를 params 외부에서 고정해 결정성 보장.
    """
    if not param_bounds:
        raise ValueError("param_bounds 가 비었습니다 (≥1 파라미터 필요).")
    for k, (lo, hi) in param_bounds.items():
        if not (lo < hi):
            raise ValueError(f"파라미터 '{k}': low({lo}) < high({hi}) 위반.")
    if n_samples < 2:
        raise ValueError(f"n_samples({n_samples}) ≥ 2 필요.")

    rng = np.random.default_rng(seed)
    A, B, AB_list, names = _saltelli_matrices(param_bounds, n_samples, rng)
    p = len(names)

    fA = _evaluate(model_fn, A, names)
    fB = _evaluate(model_fn, B, names)
    fAB = [_evaluate(model_fn, AB_list[i], names) for i in range(p)]

    # 분산은 A∪B 결합 표본으로 추정 (안정성 ↑). 평균-센터링은 first-order
    # 추정량 분산을 크게 줄임 (Saltelli 2010 권장) — total-order(차이 제곱)는
    # 평행이동 불변이라 영향 없음.
    f_all = np.concatenate([fA, fB])
    var_Y = float(f_all.var(ddof=1)) if f_all.size > 1 else 0.0
    mu = float(f_all.mean())
    fA_c = fA - mu
    fB_c = fB - mu

    s1 = np.zeros(p)
    st = np.zeros(p)
    if var_Y > 0.0:
        for i in range(p):
            fAB_c = fAB[i] - mu
            # Saltelli 2010 first-order: Vᵢ = mean(fB_c * (fAB_i_c − fA_c))
            v_i = float(np.mean(fB_c * (fAB_c - fA_c)))
            # Jansen 1999 total-order: V_{~i} = mean((fA − fAB_i)² ) / 2
            v_not_i = float(np.mean((fA - fAB[i]) ** 2) / 2.0)
            s1[i] = v_i / var_Y
            st[i] = v_not_i / var_Y

    return {
        "names": names,
        "S1": {names[i]: float(s1[i]) for i in range(p)},
        "ST": {names[i]: float(st[i]) for i in range(p)},
        "S1_array": s1,
        "ST_array": st,
        "var_Y": var_Y,
        "n_samples": n_samples,
        "n_evals": n_samples * (p + 2),
    }


def sobol_rank(indices: dict, *, by: str = "ST") -> list[tuple[str, float]]:
    """Sobol 지수 결과를 중요도 내림차순으로 정렬해 (이름, 값) 리스트 반환.

    Args:
        indices: `sobol_indices` 반환 dict.
        by: 정렬 기준 — "ST" (총효과, 기본) 또는 "S1" (주효과). 동률은
            파라미터명 사전순으로 안정 정렬.

    Returns:
        [(param, index_value), …] — index_value 내림차순. by="ST" 면 상호작용
        포함 전체 영향 순, by="S1" 면 주효과 순.

    Raises:
        ValueError: by 가 "S1"/"ST" 아니거나 해당 키가 indices 에 없을 때.

    Performance: O(p log p). Side effects: 없음.
    """
    if by not in ("S1", "ST"):
        raise ValueError(f"by 는 'S1' 또는 'ST' — got {by!r}.")
    if by not in indices:
        raise ValueError(f"indices 에 '{by}' 키 없음 (sobol_indices 결과 전달?).")
    table = indices[by]
    return sorted(table.items(), key=lambda kv: (-kv[1], kv[0]))


def _smoke() -> None:
    """선형 가산 + 상호작용 모델로 불변식을 콘솔 확인."""
    # 선형 가산: y = 3*x0 + 1*x1 + 0*x2, x∈[0,1] → S1 해석해 ∝ aᵢ²
    bounds = {"x0": (0.0, 1.0), "x1": (0.0, 1.0), "x2": (0.0, 1.0)}

    def additive(p: dict) -> float:
        return 3.0 * p["x0"] + 1.0 * p["x1"] + 0.0 * p["x2"]

    res = sobol_indices(additive, bounds, n_samples=4096, seed=0)
    print("[additive] y = 3x0 + 1x1 + 0x2")
    for name, val in sobol_rank(res, by="ST"):
        print(f"  {name}: S1={res['S1'][name]:+.3f}  ST={res['ST'][name]:+.3f}")
    print(f"  ΣS1 = {sum(res['S1'].values()):.3f}  (가산 → ≈1)")

    # 해석해: Vᵢ = aᵢ²·Var(U[0,1]) = aᵢ²/12; ΣV = (9+1+0)/12
    total = (9 + 1 + 0) / 12.0
    print(f"  analytic S1: x0={9/12/ (10/12):.3f} x1={1/12/(10/12):.3f} x2=0")

    def interact(p: dict) -> float:
        return p["x0"] + p["x1"] + 4.0 * p["x0"] * p["x1"]

    res2 = sobol_indices(interact, {"x0": (0.0, 1.0), "x1": (0.0, 1.0)},
                         n_samples=4096, seed=0)
    print("[interaction] y = x0 + x1 + 4·x0·x1")
    for name in res2["names"]:
        print(f"  {name}: S1={res2['S1'][name]:+.3f}  ST={res2['ST'][name]:+.3f}"
              f"  (ST>S1: {res2['ST'][name] > res2['S1'][name]})")


if __name__ == "__main__":
    _smoke()
