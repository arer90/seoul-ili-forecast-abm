"""simulation.abm.adaptive_agent_count

**동적(가변) agent 수** — 고정 N 이 아니라 시뮬레이션 **dynamics 에 따라 수렴 기준으로 자동 선택**.

배경: ABM 추정치(공격률·peak·WIS)는 n_agents 가 작으면 stochastic 잡음이 크고, 크면 안정되지만
계산비용↑. "맞는 N" 은 데이터/시즌의 dynamics 에 의존하므로 **임의 고정이 아니라 수렴으로
정해야** 한다 (사용자 제안 2026-06-05). 본 모듈은 증가하는 N 사다리에서 추정치가 **상대변화 <
tol 이고 seed-CV < cv_tol** 을 patience 회 연속 만족하면 그 N 을 채택한다.

content(구성)도 dynamics 적응: agent 속성(gu·연령·직업·고위험군)은 `synthetic_population` 이
실제 인구 strata 로 생성하므로, N 만 동적으로 정하면 구성은 자동으로 실데이터 비율을 따른다
(시즌별 strata 가중은 estimate_fn 에서 시즌 인구로 주입 가능).

Gray-box 계약
-------------
- `select_n_agents_adaptive(estimate_fn, candidates, ...)` 순수 — estimate_fn(n)->(point, cv)
  를 N 사다리에 호출. estimate_fn 은 pluggable (TDD=합성, 실사용=epi_proof 기반).
- 부작용 없음. Performance: O(평가된 N 개수 × estimate_fn 비용).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def select_n_agents_adaptive(
    estimate_fn: Callable[[int], tuple[float, float]],
    candidates: Sequence[int],
    *,
    tol: float = 0.02,
    cv_tol: float = 0.05,
    patience: int = 1,
) -> dict:
    """증가하는 N 에서 ABM 추정치 수렴 지점을 자동 선택 (동적 agent 수).

    Args:
        estimate_fn: ``n -> (point_estimate, seed_cv)``. point=관심 추정치(AR·peak·WIS),
            seed_cv = seed 간 변동계수(stochastic 안정성).
        candidates: 오름차순 N 사다리 (예: [2000,4000,8000,16000,32000]).
        tol: 연속 N 간 추정치 **상대변화** 수렴 임계 (기본 2%).
        cv_tol: seed-CV 임계 (기본 5%) — stochastic 잡음 충분히 작아야.
        patience: 수렴 조건을 연속 몇 번 만족해야 채택 (기본 1).

    Returns:
        {"n_optimal", "converged", "trace":[{n,estimate,cv,rel_change}]}.
        수렴 못 하면 n_optimal = 최대 candidate (안전 fallback) + converged=False.

    Raises:
        ValueError: candidates 비었거나 비오름차순.
    """
    cand = [int(n) for n in candidates]
    if not cand:
        raise ValueError("candidates 비었음")
    if any(cand[i] >= cand[i + 1] for i in range(len(cand) - 1)):
        raise ValueError(f"candidates 는 strictly 오름차순이어야: {cand}")

    trace: list[dict] = []
    prev: float | None = None
    stable = 0
    n_optimal = cand[-1]
    converged = False
    for n in cand:
        est, cv = estimate_fn(n)
        est = float(est)
        cv = float(cv)
        rel = (abs((est - prev) / prev) if (prev is not None and prev != 0)
               else float("inf"))
        trace.append({"n": n, "estimate": est, "cv": cv, "rel_change": rel})
        if prev is not None and rel < tol and cv < cv_tol:
            stable += 1
            if stable >= patience:
                n_optimal = n
                converged = True
                break
        else:
            stable = 0
        prev = est
    return {"n_optimal": int(n_optimal), "converged": bool(converged), "trace": trace}


def make_epi_estimate_fn(
    *,
    db_path,
    K: int = 5,
    cal_season: int | None = None,
    eval_season: int | None = None,
    target: str = "peak",
) -> Callable[[int], tuple[float, float]]:
    """실사용 estimate_fn — epi_proof 를 N 으로 돌려 (adaptive arm 추정치, seed-CV) 반환.

    target: "peak"(eval mapped 앙상블 평균의 peak) | "wis"(SCI WIS). seed-CV = seed별 peak CV.
    주의: 무겁다(N·K·season). 학습과 동시 실행 시 CPU 경쟁 — 학습 후 권장.
    """
    from simulation.abm.epi_proof import run_epi_proof

    def _fn(n: int) -> tuple[float, float]:
        r = run_epi_proof(K=K, seeds=list(range(K)), n_agents=int(n),
                          cal_season=cal_season, eval_season=eval_season,
                          db_path=db_path,
                          output_path=f"simulation/results/_trash/_nselect_{n}.json")
        arm = r["comparison_1_behaviour"]["on"]
        reps = np.asarray(arm.get("mapped_replicates", []), dtype=np.float64)
        if reps.ndim != 2 or reps.size == 0:
            # public summary may omit reps; fall back to SCI WIS point
            sci = r["comparison_1_behaviour"].get("sci_validation", {})
            w = float(sci.get("adaptive", {}).get("wis", float("nan")))
            return w, 0.0
        seed_peaks = reps.max(axis=1)                 # seed별 peak
        point = float(np.mean(seed_peaks))
        cv = float(np.std(seed_peaks) / point) if point > 0 else 0.0
        if target == "wis":
            sci = r["comparison_1_behaviour"].get("sci_validation", {})
            point = float(sci.get("adaptive", {}).get("wis", point))
        return point, cv

    return _fn
