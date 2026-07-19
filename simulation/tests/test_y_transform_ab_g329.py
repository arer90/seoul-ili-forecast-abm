"""G-330 (2026-06-20): 동적 per-model 변환 선택 — preproc OOF가 결정, force-identity 최소화.

G-329(전역 STABLE_Y=affine-only) 되돌림. 49-모델 변환 audit(in-range R² @ R9 feature) 실증:
  ① R9 feature 개수에선 어떤 변환도 폭발 안 함(예전 "log1p 폭발"=full-feature 과적합 artifact).
  ② OOF-CV는 in-range fold = 사용자 기준 → preproc OOF가 모델별 in-range 최고 변환을 동적 선택
     (NegBinGLM→log1p/sqrt, BayesianRidge→laplace[log1p 자동회피], robust→best).
전역 affine-only 가 NegBinGLM의 OOF-selected sqrt/log1p 를 막아 identity −1.07 붕괴시킨 게 버그.
→ 전 변환 STABLE_Y 개방 + OOF 동적 선택. force-identity 는 비-identity 가 진짜 catastrophic 인
  Poisson/hhh4(내부 log-link 이중적용 −46/−78) + pf wrapper 만 유지. 폭발 backstop = G-328 cap.
"""
from simulation.pipeline.preproc_optuna_hierarchical import (
    STABLE_Y_TRANSFORMS, _apply_single_y_transform, model_applies_internal_y_transform,
)
import numpy as np


# ── G-330: STABLE_Y = 전체 변환셋 복원 (OOF 가 동적 선택) ────────────────────
def test_stable_y_is_full_set_restored():
    # G-329 가 제거했던 log1p/sqrt/asinh 복원 → OOF 가 모델별로 고를 수 있어야
    for t in ("log1p", "sqrt", "asinh"):
        assert t in STABLE_Y_TRANSFORMS, f"{t} 가 STABLE_Y 에 복원돼야(NegBinGLM 등 필요)"
    for t in ("laplace", "mcmc_robust"):
        assert t in STABLE_Y_TRANSFORMS, f"{t}(affine) 도 유지"


# ── G-330 + transform-fix (2026-06-21): force-identity 는 audit-증명 catastrophic 모델만 ───────
def test_force_identity_minimal_and_dynamic():
    # transform-fix (2026-06-21): PoissonAutoreg 의 내부 np.log AR link 를 제거(PART A) → Ridge-AR
    #   on raw y. 더 이상 내부 변환이 없으므로 force-identity 면제(PART C 에서 gate 제거) → 데이터
    #   기반 preproc OOF 가 단일 y-transform 을 동적 선택. (옛 G-330 −46 은 내부 log-link 가정 전제.)
    assert not model_applies_internal_y_transform("PoissonAutoreg"), \
        "PoissonAutoreg 는 내부 log AR link 제거(PART A) → force-identity 면제(OOF 동적 선택)"
    # 정수반올림∘NB-log-link 이라 외부 변환이 해상도 파괴 = 유지 (preproc OOF 도 안 고를 구조)
    assert model_applies_internal_y_transform("hhh4-equivalent"), "hhh4 비-id −78 → force-identity 유지"
    # 외부 변환이 in-range 개선(audit) = 제거 → OOF 가 동적 선택
    for m in ("NegBinGLM", "NegBinGLM-V7", "GAM-Spline", "GLARMA", "NegBinGLM-Glum"):
        assert not model_applies_internal_y_transform(m), \
            f"{m} 는 force-identity 제거돼야(OOF 가 sqrt/log1p/laplace 동적 선택)"


# ── G-330: 비-affine 역변환도 G-328 cap 으로 bounded (외삽 backstop) ──────────
def test_nonaffine_inverses_capped_not_unbounded():
    y = np.linspace(0, 67, 200)              # train max ~67
    for name in ("log1p", "sqrt", "asinh"):
        _, inv, _ = _apply_single_y_transform(y, name)
        # 외삽(큰 transform-space 입력)서도 cap 이하로 bounded (지수폭발 X)
        big = float(inv(np.array([20.0]))[0])
        assert np.isfinite(big), f"{name} inverse 가 외삽서 inf"
        assert big <= 67 * 10.0 + 1e-6, f"{name} inverse 가 10×y_max cap 초과(폭발)"
        # 음수 floor
        assert float(inv(np.array([-5.0]))[0]) >= 0.0
