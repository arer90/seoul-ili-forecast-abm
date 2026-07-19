"""TDD — G-333 flat-grid preproc 재설계 (2026-06-22).

사용자 결정 + SCI 리서치 + codex 검토: R9 Y-transform 선택을 hierarchical-Optuna(TPE exploitation
+ 2% margin)에서 **flat grid(각 transform 1회) + fold-paired 1-SE(identity 기본)**로 교체.
근본: DNN 이 HIER_individual 을 38×, identity 를 3× 탐색(TPE skew) + OOF 12.28 vs 12.95(노이즈 5%)로
HIER_individual 채택 → test R²=-0.48(preds 298). flat grid = 균등 1회, 1-SE = 노이즈급 우세 거부.

⚠ 정직한 한계: 1-SE 는 **노이즈급 OOF 우세**를 거부한다. DNN 의 OOF 우세가 (a) fold 노이즈면 identity
회복(~0.9), (b) 일관적이나 peak-test 서만 붕괴(OOF→test 분포이동)면 1-SE 가 못 잡음 — 그 경우 챔피언
G-318(hold-out test 기반)이 backstop 으로 제외. 즉 1-SE 는 원칙적 개선이지 DNN→0.9 보장이 아니다.
실제 회복 여부는 clean run 산출로 확인.
"""
import numpy as np
import pytest


# ───────── Step 1-2: fourth_root + flat 8 ─────────
def test_stable_y_transforms_flat_7():
    from simulation.pipeline.preproc_optuna_hierarchical import STABLE_Y_TRANSFORMS
    # 6 metric + identity(y_mode="none") = flat 7. fourth_root(France 2022)만 신규 추가.
    assert len(STABLE_Y_TRANSFORMS) == 6, STABLE_Y_TRANSFORMS
    for t in ["log1p", "sqrt", "fourth_root", "asinh", "laplace", "mcmc_robust"]:
        assert t in STABLE_Y_TRANSFORMS
    # anscombe/freeman_tukey = Poisson count 전용 → rate(ILI) 부적절 + G-256 금지(test_g256) → 제외.
    # boxcox/yeo/rank/gaussian/logit = OOD train-bounded → 제외.
    for bad in ["anscombe", "freeman_tukey", "boxcox", "yeo_johnson", "rank", "gaussian", "logit"]:
        assert bad not in STABLE_Y_TRANSFORMS


def test_fourth_root_roundtrip_and_cap():
    from simulation.pipeline.preproc_optuna_hierarchical import _apply_single_y_transform
    y = np.array([0., 1., 5., 30., 66.])
    yt, invf, st = _apply_single_y_transform(y, "fourth_root")
    assert np.allclose(yt, np.power(y, 0.25)), yt           # forward = y^0.25
    assert np.allclose(invf(yt), y, atol=1e-6)              # roundtrip 정확
    # ★ cap: t-공간 큰 입력(t=50 → t^4=6.25e6)이 폭발하지 않고 safe_cap(=10×y_max)에 묶임
    assert invf(np.array([50.0]))[0] <= st["safe_cap"] + 1.0
    assert invf(np.array([-5.0]))[0] == 0.0                 # 음수 입력 0-clip (t^4 NaN 방지)


def test_fourth_root_inverse_replay_present():
    """champion 재로드 경로(apply_y_preproc_inverse_only)가 fourth_root 를 다룰 것."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "pipeline" / "preproc_optuna_hierarchical.py").read_text(encoding="utf-8")
    # inverse-replay 함수 안에 fourth_root branch
    i = src.index("def apply_y_preproc_inverse_only")
    assert 'name == "fourth_root"' in src[i:], "fourth_root inverse-replay branch 누락"


# ───────── Step 4: grid mode (pure grid) ─────────
def test_grid_mode_seeds_all_transforms():
    from simulation.pipeline._inline_optuna_3stage import _seed_y_transform_trials
    from simulation.pipeline.preproc_optuna_hierarchical import STABLE_Y_TRANSFORMS

    class _MockStudy:
        def __init__(self): self.enq = []
        def enqueue_trial(self, params, skip_if_exists=True): self.enq.append(params)

    # grid_mode=True + 작은 n_trials 여도 모든 transform seed (anchor + 8 = 9)
    st = _MockStudy()
    n = _seed_y_transform_trials(st, "DNN", force_y_identity=False, force_x_identity=False,
                                 restrict_centered=False, n_trials=3, grid_mode=True)
    assert n == len(STABLE_Y_TRANSFORMS) + 1, f"grid: anchor+8=9 expected, got {n}"
    # legacy(grid_mode=False) + n_trials=3 → budget cap 으로 적게 seed
    st2 = _MockStudy()
    n2 = _seed_y_transform_trials(st2, "DNN", force_y_identity=False, force_x_identity=False,
                                  restrict_centered=False, n_trials=3, grid_mode=False)
    assert n2 < n, f"legacy budget-capped expected < {n}, got {n2}"


# ───────── Step 5: 1-SE selection ─────────
def _cell(oof, folds, is_id):
    p = ({"y_mode": "none", "x_mode": "none"} if is_id
         else {"y_mode": "individual", "y_individual": "log1p", "x_mode": "none"})
    return {"oof_wis": oof, "wis": oof, "trial_params": p, "oof_wis_folds": folds}


def test_1se_rejects_noise_level_win():
    from simulation.pipeline._inline_optuna_3stage import _pick_masked_best_preproc
    # transform oof 낮지만 fold 차이가 노이즈 → identity 유지
    tr = [_cell(12.95, [13, 11, 14, 12, 13.75], True),
          _cell(12.28, [13.5, 10, 15, 11, 14], False)]
    best, _ = _pick_masked_best_preproc(tr, "oof_cv")
    assert best["trial_params"]["y_mode"] == "none"


def test_1se_keeps_consistent_win():
    from simulation.pipeline._inline_optuna_3stage import _pick_masked_best_preproc
    # transform 이 모든 fold 서 일관되게 우세 → transform 채택
    tr = [_cell(13.0, [13, 12.8, 13, 12.9, 13], True),
          _cell(12.0, [12, 11.9, 12.1, 12, 12], False)]
    best, _ = _pick_masked_best_preproc(tr, "oof_cv")
    assert best["trial_params"]["y_mode"] == "individual"


def test_1se_fallback_2pct_when_folds_absent():
    from simulation.pipeline._inline_optuna_3stage import _pick_masked_best_preproc
    tr = [{"oof_wis": 12.95, "wis": 12.95, "trial_params": {"y_mode": "none", "x_mode": "none"}},
          {"oof_wis": 12.85, "wis": 12.85,
           "trial_params": {"y_mode": "individual", "y_individual": "log1p", "x_mode": "none"}}]
    best, _ = _pick_masked_best_preproc(tr, "oof_cv")   # 0.8% < 2% → identity
    assert best["trial_params"]["y_mode"] == "none"


def test_force_argmin_bypasses_1se(monkeypatch):
    """진단 노브: MPH_PREPROC_FORCE_ARGMIN=1 → 1-SE 우회, 순수 argmin(transform) 채택."""
    from simulation.pipeline._inline_optuna_3stage import _pick_masked_best_preproc
    # 노이즈급 케이스(1-SE 면 identity) — force_argmin 이면 transform(낮은 oof) 채택
    tr = [_cell(12.95, [13, 11, 14, 12, 13.75], True),
          _cell(12.28, [13.5, 10, 15, 11, 14], False)]
    monkeypatch.setenv("MPH_PREPROC_FORCE_ARGMIN", "1")
    best, _ = _pick_masked_best_preproc(tr, "oof_cv")
    assert best["trial_params"]["y_mode"] == "individual", "force_argmin 인데 identity 로 회귀함"
    # off(default) 면 identity 로 회귀(기존 1-SE 동작 불변 확인)
    monkeypatch.setenv("MPH_PREPROC_FORCE_ARGMIN", "0")
    best2, _ = _pick_masked_best_preproc(tr, "oof_cv")
    assert best2["trial_params"]["y_mode"] == "none"


def test_force_y_individual_picks_named_transform(monkeypatch):
    """진단 노브: MPH_FORCE_Y_INDIVIDUAL=asinh → OOF 꼴찌라도 asinh trial 강제."""
    from simulation.pipeline._inline_optuna_3stage import _pick_masked_best_preproc

    def _c(oof, yi):
        p = ({"y_mode": "none", "x_mode": "none"} if yi is None
             else {"y_mode": "individual", "y_individual": yi, "x_mode": "none"})
        return {"oof_wis": oof, "wis": oof, "trial_params": p, "oof_wis_folds": [oof] * 5}

    tr = [_c(2.7, None), _c(2.2, "sqrt"), _c(4.3, "asinh")]   # asinh = 꼴찌
    monkeypatch.setenv("MPH_FORCE_Y_INDIVIDUAL", "asinh")
    best, _ = _pick_masked_best_preproc(tr, "oof_cv")
    assert best["trial_params"].get("y_individual") == "asinh", "asinh 강제 실패"
    # 미설정 → 1-SE(identity, asinh 아님) 불변
    monkeypatch.delenv("MPH_FORCE_Y_INDIVIDUAL")
    best2, _ = _pick_masked_best_preproc(tr, "oof_cv")
    assert best2["trial_params"].get("y_individual") != "asinh"


def test_g334_fold_invariant_cap():
    """G-334: inverse cap 이 fold y_max 가 아니라 전역 참조(set_y_ref_max) 기준."""
    from simulation.pipeline.preproc_optuna_hierarchical import (
        _apply_single_y_transform, set_y_ref_max)
    y_fold = np.array([0., 5., 10., 20.])   # 작은 fold (max 20)
    # 미설정 → fold-local: cap = 10×20 = 200
    set_y_ref_max(None)
    _, _, st = _apply_single_y_transform(y_fold, "asinh")
    assert abs(st["safe_cap"] - 200.0) < 1e-6, st["safe_cap"]
    # 전역=67(전체 train max) → fold-불변: cap = 10×67 = 670 (작은 fold 여도 큰 cap)
    set_y_ref_max(67.0)
    _, _, st2 = _apply_single_y_transform(y_fold, "asinh")
    assert abs(st2["safe_cap"] - 670.0) < 1e-6, st2["safe_cap"]
    # sqrt/log1p/fourth_root 도 동일 전역 기준
    for tname in ("sqrt", "fourth_root"):
        _, _, s = _apply_single_y_transform(y_fold, tname)
        assert abs(s["safe_cap"] - 670.0) < 1e-6, (tname, s["safe_cap"])
    set_y_ref_max(None)   # reset (다른 테스트 오염 방지)


def test_g334_asinh_inverse_replay_capped():
    """G-334 ③: _reapply_primitive_inverse asinh 가 safe_cap 으로 출력 제한(sqrt/fourth_root 와 일관)."""
    from simulation.pipeline.preproc_optuna_hierarchical import _reapply_primitive_inverse
    # 큰 transformed 입력(x=10 → sinh(10)≈11013)이 safe_cap=670 에 묶임 (옛 코드는 11013 폭발)
    out = _reapply_primitive_inverse(np.array([10.0]), "asinh", {"safe_cap": 670.0})
    assert out[0] <= 670.0 + 1e-6, f"asinh inverse uncapped: {out[0]}"
    # 정상 입력 roundtrip: arcsinh(50)≈4.6 → sinh≈50 (cap 미발동)
    xt = float(np.arcsinh(50.0))
    out2 = _reapply_primitive_inverse(np.array([xt]), "asinh", {"safe_cap": 670.0})
    assert abs(out2[0] - 50.0) < 1.0, out2[0]
    # 레거시(safe_cap 부재) = 옛 ±10 동작 유지(back-compat)
    out3 = _reapply_primitive_inverse(np.array([5.0]), "asinh", {})
    assert abs(out3[0] - float(np.sinh(5.0))) < 1.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
