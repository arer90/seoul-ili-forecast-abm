"""simulation.abm.adaptive_allocation

**In-run 동적 agent 할당** — 시뮬레이션 도중 유행 dynamics 에 따라 agent **수(resolution)와
구성(strata content)** 을 적응적으로 바꾼다 (사용자 제안 2026-06-05).

핵심 아이디어 (adaptive mesh refinement for ABM):
  (a) 유효 N 스케일: 유병률↑(peak) → agent↑(고해상도), 유병률↓(off-season) → agent↓(절약).
  (b) strata 재할당: 활동(감염) 높은 gu/연령에 agent 더 배분(중요도 표집), 단 floor 로 저활동
      strata 소실 방지 → 구성이 dynamics 를 추종.
  (c) **편향 없는 보존 재샘플링**: 가중(particle-filter systematic) 재샘플 — 각 agent 가중
      w_i(대표 인원수) 기준, 재샘플 후 총 인구 P=Σw 보존 + E[복제수_i] ∝ w_i (불편).

이 컨트롤러는 ABM epoch 루프가 호출한다 (kernel 무관·순수 → 단독 TDD 가능).

Gray-box 계약
-------------
- `target_n(prevalence, ...)`: 현 유병률 → 다음 epoch 유효 N (clamp [floor_n, max_n]).
- `allocate_by_activity(activity, budget, floor_frac)`: strata 별 정수 agent 수 (합=budget, floor 보장).
- `resample_weighted(weights, target_count, rng)`: (idx, new_weight) — 불편·인구보존.
- 모두 순수 함수(부작용 없음). Performance: O(N).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AdaptiveAllocator:
    """In-run 동적 할당 정책 (frozen — 재현성).

    Attributes:
        base_n: 기저 유효 agent 수 (저활동 시).
        max_n: 최대 유효 agent 수 (peak 시 고해상도).
        floor_n: 최소 유효 agent 수 (절대 하한).
        sensitivity: 유병률→N 반응 지수 (>0; 1=선형, >1=peak 집중).
    """
    base_n: int = 8000
    max_n: int = 32000
    floor_n: int = 2000
    sensitivity: float = 1.0

    def __post_init__(self):
        if not (0 < self.floor_n <= self.base_n <= self.max_n):
            raise ValueError(f"floor_n≤base_n≤max_n 위반: {self.floor_n},{self.base_n},{self.max_n}")
        if not (self.sensitivity > 0):
            raise ValueError(f"sensitivity>0 위반: {self.sensitivity}")

    def target_n(self, prevalence: float, peak_ref: float) -> int:
        """현 유병률 → 다음 epoch 유효 N. prevalence/peak_ref ∈ [0,1] 비율로 스케일.

        n = floor_n + (max_n − floor_n) · clip(prevalence/peak_ref, 0, 1)^sensitivity.
        peak_ref ≤ 0 이면 base_n.
        """
        if peak_ref <= 0 or not np.isfinite(peak_ref):
            return int(self.base_n)
        frac = min(max(float(prevalence) / float(peak_ref), 0.0), 1.0) ** self.sensitivity
        n = self.floor_n + (self.max_n - self.floor_n) * frac
        return int(round(min(max(n, self.floor_n), self.max_n)))


def allocate_by_activity(activity, budget: int, *, floor_frac: float = 0.05) -> np.ndarray:
    """strata 활동(감염) ∝ agent budget 배분 + per-stratum floor. 합 = budget (정수).

    각 strata 최소 floor_frac·budget/G 보장(저활동 소실 방지), 나머지를 활동 비례 배분.
    Largest-remainder 로 정수합 = budget.
    """
    a = np.asarray(activity, dtype=np.float64)
    if a.ndim != 1 or a.size == 0:
        raise ValueError(f"activity 1-D non-empty 필요, got {a.shape}")
    if np.any(a < 0) or not np.all(np.isfinite(a)):
        raise ValueError("activity 음수/비유한 불가")
    G = a.size
    budget = int(budget)
    if budget < G:
        raise ValueError(f"budget({budget}) < strata 수({G}) — 각 strata 최소 1")
    floor = max(int(np.floor(floor_frac * budget / G)), 1)
    base = np.full(G, floor, dtype=np.int64)
    remaining = budget - base.sum()
    if remaining < 0:                       # floor 합이 budget 초과 → floor 균등 축소
        base = np.full(G, budget // G, dtype=np.int64)
        remaining = budget - base.sum()
    tot = a.sum()
    if tot <= 0:                            # 활동 0 → 잔여 균등
        share = np.full(G, remaining / G)
    else:
        share = remaining * (a / tot)
    add = np.floor(share).astype(np.int64)
    out = base + add
    # largest-remainder 로 합 맞춤
    short = budget - int(out.sum())
    if short > 0:
        frac_rank = np.argsort(-(share - add))
        out[frac_rank[:short]] += 1
    return out


def resample_weighted(weights, target_count: int, rng: np.random.Generator):
    """가중 systematic 재샘플 (particle-filter). 불편 + 인구 보존.

    Returns:
        (idx, new_weight): idx = 복제될 원 agent 인덱스 (len=target_count);
        new_weight = P/target_count (각 새 agent 대표 인원 — Σ=P 보존).
    불편: E[#copies of i] = target_count · w_i / P.
    """
    if rng is None:
        raise ValueError("rng 필수 (재현성)")
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.size == 0 or np.any(w < 0):
        raise ValueError("weights 1-D 비음수 non-empty 필요")
    M = int(target_count)
    if M <= 0:
        raise ValueError("target_count > 0 필요")
    P = float(w.sum())
    if P <= 0:
        raise ValueError("weights 합 0 — 재샘플 불가")
    probs = w / P
    positions = (rng.random() + np.arange(M)) / M     # systematic
    idx = np.searchsorted(np.cumsum(probs), positions)
    idx = np.clip(idx, 0, w.size - 1)
    return idx.astype(np.int64), float(P / M)
