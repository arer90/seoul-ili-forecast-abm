"""FusedEpi 불확실성 분해 (aleatoric / epistemic) — 예측 투명성.

"예측폭이 넓다 = 자료부족(epistemic uncertainty) 이지 모델이 나쁜 것이 아니다" 를
정량적으로 보여주기 위한 **모델-비종속** 분해 모듈.

분산의 전체 법칙(law of total variance)으로 앙상블 예측 분산을 두 성분으로 가른다:

    Var[y] = E_θ[ Var(y | θ) ]  +  Var_θ[ E(y | θ) ]
             └── aleatoric ──┘     └── epistemic ──┘

- **epistemic** (인식론적/축소가능): 앙상블 멤버 간 평균예측의 분산. 멤버들이
  서로 의견이 갈리는 시점 = 모델이 자료로부터 확신하지 못하는 시점 → 자료부족 경고.
- **aleatoric** (우연적/축소불가능): 멤버 내부 분산의 평균. 멤버가 점예측이면
  내부 분산이 0 이므로, 잔차 분산(`residual_var`)을 우연적 noise 추정치로 사용.
- **total** = aleatoric + epistemic (법칙에 의해 두 성분의 합).

참고문헌:
    - Kendall A & Gal Y (2017) "What Uncertainties Do We Need in Bayesian Deep
      Learning for Computer Vision?" NeurIPS 30. (aleatoric vs epistemic 분해 정의)
    - Hüllermeier E & Waegeman W (2021) "Aleatoric and epistemic uncertainty in
      machine learning" Machine Learning 110:457-506. doi:10.1007/s10994-021-05946-3
    - Lakshminarayanan et al. (2017) "Simple and Scalable Predictive Uncertainty
      Estimation using Deep Ensembles" NeurIPS 30. (deep-ensemble 분산분해)

D-5 gray-box contract:
    - 순수 함수(pure) + 결정성(`np.random.default_rng(seed)` 만 사용; 본 모듈은 난수 불필요).
    - NaN-safe 아님 — caller 가 finite 보장 (NaN 입력 시 NaN 전파, fail-loud).
    - shape 보존: 입력 (m, n) → 각 성분 (n,).
    - 음수 없음: 분산은 항상 ≥ 0 (수치오차 floor 0).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = [
    "decompose_uncertainty",
    "flag_high_epistemic",
]


def decompose_uncertainty(
    ensemble_preds: np.ndarray,
    *,
    member_vars: Optional[np.ndarray] = None,
    residual_var: Optional[np.ndarray] = None,
) -> dict:
    """앙상블 예측 분산을 aleatoric/epistemic 으로 분해 (법칙: total variance).

    분산의 전체 법칙으로 앙상블 분산을 두 성분으로 가른다::

        epistemic[t] = Var_members( ensemble_preds[:, t] )       # 멤버 간 불일치
        aleatoric[t] = mean_members( member_vars[:, t] )         # 멤버 내부 noise
                       또는 residual_var[t] (멤버=점예측일 때)
        total[t]     = aleatoric[t] + epistemic[t]

    멤버가 **점예측**(per-member variance 없음)이면 멤버 내부 분산 = 0 이므로
    aleatoric 은 `residual_var`(잔차 분산 추정치)로 공급한다. `member_vars` 와
    `residual_var` 가 모두 주어지면 `member_vars` 가 우선한다.

    Args:
        ensemble_preds: (m, n) — m=앙상블 멤버 수, n=시점 수. 각 멤버의 점예측 행.
            finite 보장(caller 책임). m ≥ 2 권장(m=1 이면 epistemic=0).
        member_vars: (m, n) | None — 멤버별 시점별 예측 분산(우연적 성분).
            주어지면 aleatoric = 멤버 평균. 단위 = 예측값²(분산).
        residual_var: (n,) | scalar | None — 멤버 내부 분산 대용 잔차 분산.
            `member_vars` 없을 때 aleatoric 으로 사용. 음수 금지. scalar 면 broadcast.

    Returns:
        dict {
            "epistemic": (n,) float64,   # 멤버 간 분산 (자료부족·모델불확실)
            "aleatoric": (n,) float64,   # 멤버 내부 분산 평균 또는 residual_var
            "total":     (n,) float64,   # aleatoric + epistemic
            "epistemic_frac": (n,) float64,  # epistemic / total (0~1, total=0→0)
            "n_members": int,
            "n_steps": int,
        }
        모든 성분 ≥ 0 (수치오차 floor 0). aleatoric+epistemic == total (법칙).

    Raises:
        ValueError: ensemble_preds 가 2-D (m, n) 아니거나 m<1; member_vars/residual_var
            shape 불일치; residual_var 음수.

    Performance: O(m * n) — 단일 np.var 패스. n=337, m≤50 → < 1 ms.
    Side effects: 없음 (순수 함수, 입력 미변형).
    Caller responsibility:
        - ensemble_preds finite (NaN 시 NaN 전파, fail-loud)
        - residual_var ≥ 0 (분산은 비음)
    """
    preds = np.asarray(ensemble_preds, dtype=np.float64)
    if preds.ndim != 2:
        raise ValueError(
            f"ensemble_preds must be 2-D (m_members, n_steps); got ndim={preds.ndim}, "
            f"shape={preds.shape}"
        )
    m, n = preds.shape
    if m < 1:
        raise ValueError(f"ensemble_preds must have >=1 member; got m={m}")

    # epistemic = 멤버 간 분산 (모집단 분산 ddof=0; m=1 → 0). 자료부족·모델불확실 신호.
    epistemic = np.var(preds, axis=0, ddof=0)

    # aleatoric = 멤버 내부 분산 평균 (member_vars) 또는 잔차 분산(residual_var) 또는 0.
    if member_vars is not None:
        mv = np.asarray(member_vars, dtype=np.float64)
        if mv.shape != preds.shape:
            raise ValueError(
                f"member_vars shape {mv.shape} != ensemble_preds shape {preds.shape}"
            )
        if np.any(mv < 0):
            raise ValueError("member_vars must be non-negative (variance)")
        aleatoric = np.mean(mv, axis=0)
    elif residual_var is not None:
        rv = np.asarray(residual_var, dtype=np.float64)
        if rv.ndim == 0:
            aleatoric = np.full(n, float(rv), dtype=np.float64)
        else:
            rv = rv.ravel()
            if rv.shape[0] != n:
                raise ValueError(
                    f"residual_var length {rv.shape[0]} != n_steps {n}"
                )
            aleatoric = rv.astype(np.float64, copy=True)
        if np.any(aleatoric < 0):
            raise ValueError("residual_var must be non-negative (variance)")
    else:
        aleatoric = np.zeros(n, dtype=np.float64)

    # 수치오차로 인한 미세 음수 floor (분산은 비음).
    epistemic = np.clip(epistemic, 0.0, None)
    aleatoric = np.clip(aleatoric, 0.0, None)
    total = aleatoric + epistemic

    with np.errstate(divide="ignore", invalid="ignore"):
        epistemic_frac = np.where(total > 0.0, epistemic / total, 0.0)

    return {
        "epistemic": epistemic,
        "aleatoric": aleatoric,
        "total": total,
        "epistemic_frac": epistemic_frac,
        "n_members": int(m),
        "n_steps": int(n),
    }


def flag_high_epistemic(
    decomp: dict,
    *,
    quantile: float = 0.8,
    min_total: float = 0.0,
) -> np.ndarray:
    """epistemic 불확실성이 높은 시점을 플래그 (자료부족 경고).

    `decompose_uncertainty` 결과에서 epistemic 성분이 상위 `quantile` 분위를
    초과하는 시점을 True 로 표시한다. 이 시점들은 "앙상블 멤버 간 의견이 크게
    갈리는 = 자료가 부족해 확신 못 하는" 지점으로, 예측폭이 넓은 이유가
    모델 결함이 아니라 자료부족(epistemic) 임을 투명하게 보여준다.

    Args:
        decomp: `decompose_uncertainty` 반환 dict. "epistemic" (n,) 키 필수.
        quantile: 0~1, 임계 분위(threshold = 해당 분위값). 0.8 = 상위 20% 플래그.
            이 분위를 **초과**(strict >)하는 시점만 True. 동률 다수 시 플래그 0개 가능.
        min_total: total 불확실성이 이 값 이하인 시점은 플래그 제외(절대 noise 바닥
            방어). 0.0 = 비활성(기본). "total" 키 없으면 무시.

    Returns:
        (n,) bool ndarray — epistemic 높은 시점 True. shape 는 epistemic 과 일치.

    Raises:
        ValueError: decomp 에 "epistemic" 없음; quantile 가 [0,1] 밖.

    Performance: O(n log n) — 단일 quantile. n=337 → < 1 ms.
    Side effects: 없음 (순수 함수).
    Caller responsibility: decomp 는 decompose_uncertainty 산출물(키 보존).
    """
    if "epistemic" not in decomp:
        raise ValueError("decomp must contain 'epistemic' key (decompose_uncertainty output)")
    if not (0.0 <= quantile <= 1.0):
        raise ValueError(f"quantile must be in [0, 1]; got {quantile}")

    epi = np.asarray(decomp["epistemic"], dtype=np.float64)
    if epi.size == 0:
        return np.zeros(0, dtype=bool)

    thresh = float(np.quantile(epi, quantile))
    flags = epi > thresh

    if min_total > 0.0 and "total" in decomp:
        tot = np.asarray(decomp["total"], dtype=np.float64)
        flags = flags & (tot > min_total)

    return flags
