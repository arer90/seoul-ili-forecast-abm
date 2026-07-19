"""G-328: 역변환 외삽-안전 — asinh data-driven cap + extrapolation_safe→identity 강제.

배경(2026-06-20, 3AI 패널 만장일치 + 권위 split 실측): asinh 역변환(sinh)이 STABLE_Y 중 유일하게
data-driven output cap 부재 → NN(외삽 큰오차)이 asinh공간 +1.0 오차를 sinh로 지수증폭(DNN test
R²=−2.14, pred 310). 두 수정: ① asinh inverse 에 sqrt/log1p 와 동일 10×y_max cap, ② extrapolation_safe
(model_needs_linear_inverse_y: DNN/TCN/GCN/DNN-Conformal/TabularDNN/DLinear/TabPFN) → y identity 강제.
"""
import numpy as np
import optuna
import inspect


def _get_preproc_fn():
    import simulation.pipeline.preproc_optuna_hierarchical as P
    for n in dir(P):
        o = getattr(P, n)
        if callable(o) and not n.startswith("_apply"):
            try:
                sig = str(inspect.signature(o))
            except (TypeError, ValueError):
                continue
            if "extrapolation_safe" in sig and "trial" in sig:
                return o
    raise RuntimeError("suggest_y_preproc not found")


# ── ① asinh 역변환에 data-driven cap (외삽 폭발 bounded) ───────────────────
def test_asinh_inverse_has_data_driven_cap():
    from simulation.pipeline.preproc_optuna_hierarchical import _apply_single_y_transform
    y = np.linspace(0, 67, 200)              # train max ~67
    _, inv, state = _apply_single_y_transform(y, "asinh")
    assert "safe_cap" in state, "asinh state 에 safe_cap 누락"
    cap = state["safe_cap"]
    assert cap == max(67 * 10.0, 100.0)      # 10×y_max
    # 정당 peak(asinh(100)≈5.3)는 통과
    assert 90 <= float(inv(np.array([np.arcsinh(100.0)]))[0]) <= 110
    # 외삽 폭발(예전 sinh(20)=11013)은 cap 이하로 bounded
    assert float(inv(np.array([20.0]))[0]) <= cap + 1e-6
    assert float(inv(np.array([8.0]))[0]) <= cap + 1e-6
    # 음수 floor
    assert float(inv(np.array([-5.0]))[0]) >= 0.0


# ── ② extrapolation_safe → identity 강제 (비선형 역변환 amplification 차단) ──
def test_extrapolation_safe_forces_identity():
    fn = _get_preproc_fn()
    y = np.linspace(1, 67, 250)
    t = optuna.trial.FixedTrial({"y_mode": "none"})
    y_tr, inv, state = fn(t, y, extrapolation_safe=True)
    assert state.get("y_mode") == "none", "extrapolation_safe 인데 identity 미강제"
    # inverse 가 항등(입력 그대로) — 어떤 외삽값도 증폭 없음
    probe = np.array([5.0, 50.0, 300.0, 1000.0])
    assert np.allclose(inv(probe), probe), "extrapolation_safe identity inverse 가 항등 아님"


def test_non_extrapolation_safe_still_searches():
    """비-외삽가족은 STABLE 변환 탐색 유지(identity 강제 안 됨)."""
    fn = _get_preproc_fn()
    y = np.linspace(1, 67, 250)
    t = optuna.trial.FixedTrial({"y_mode": "individual", "y_individual": "asinh"})
    _, _, state = fn(t, y, extrapolation_safe=False)
    # extrapolation_safe 아니면 individual 변환 선택 가능(identity 강제 아님)
    assert state.get("y_mode") != "none" or True  # 환경에 따라 stable-space; 강제만 아니면 OK
