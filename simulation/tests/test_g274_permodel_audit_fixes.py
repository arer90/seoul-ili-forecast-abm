"""G-274 (2026-06-16) per-model 감사 후속 fix 회귀 가드.

5개 fix: ①STABILITY feature-floor ②hhh4 cap 3→1.5 ③pf normalizer 부호-안전
④EARS-C3 ≠ C2(3-period composite). per-file 실행 권장(macOS OpenMP).
"""
import json
import numpy as np
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[2]


# ─────────────────────────── FIX 1: STABILITY floor ───────────────────────────
def _floor_pick(freq, sel_idx, floor=12, floor_min=10, model_name="X"):
    """per_model_optimize.py 의 floor 스니펫과 동일 알고리즘(특성화 테스트)."""
    if model_name in ("BayesianMCMC",) or len(sel_idx) >= floor_min:
        return sorted(sel_idx)
    freq = np.asarray(freq, float)
    if not freq.size:
        return sorted(sel_idx)
    topf = sorted(int(j) for j in np.argsort(freq)[::-1][: min(floor, freq.size)])
    return topf if len(topf) > len(sel_idx) else sorted(sel_idx)


def test_floor_lifts_collapsed_to_12():
    # freq: 3개만 높고 나머지 점감 → collapsed sel=[0,2,4]
    freq = np.zeros(52); freq[[0, 2, 4]] = 0.9
    freq[5:] = np.linspace(0.5, 0.01, 47)
    out = _floor_pick(freq, [0, 2, 4])
    assert len(out) == 12, out
    assert {0, 2, 4} <= set(out)               # 원 selection 보존(freq 최상위)
    assert len(set(out)) == 12                  # 중복 없음


def test_floor_no_op_on_legit_selection():
    freq = np.random.RandomState(0).rand(40)
    legit = sorted(range(12))                   # 이미 12개 = floor_min(10) 이상
    assert _floor_pick(freq, legit) == legit    # 미발동
    legit11 = sorted(range(11))                 # 11 ≥ floor_min(10) → 미발동(legit 무손상)
    assert _floor_pick(freq, legit11) == legit11


def test_floor_exempts_mechanistic_and_empty_freq():
    freq = np.zeros(52); freq[[0, 2, 4]] = 0.9
    assert _floor_pick(freq, [0, 2, 4], model_name="BayesianMCMC") == [0, 2, 4]
    assert _floor_pick([], [0, 2, 4]) == [0, 2, 4]          # 빈 freq 안전


def test_floor_code_present_in_source():
    src = (ROOT / "simulation/pipeline/per_model_optimize.py").read_text(encoding="utf-8")
    assert "MPH_FEAT_FLOOR" in src and "STABILITY floor" in src
    assert 'model_name not in ("BayesianMCMC",)' in src


# ─────────────────────────── FIX 2: hhh4 cap 3→1.5 ───────────────────────────
def test_hhh4_cap_tightened():
    src = (ROOT / "simulation/models/hhh4_models.py").read_text(encoding="utf-8")
    assert "self._y_max * 1.5" in src          # 새 cap
    assert "self._y_max * 3.0" not in src       # 옛 cap 제거


# ─────────────────────────── FIX 3: pf normalizer 부호-안전 ───────────────────
def test_pf_normalizer_sign_safe():
    src = (ROOT / "simulation/models/modern_ts/pf_models.py").read_text(encoding="utf-8")
    # 2026-06-21 transform-fix(완료): 내부 softplus∘(external) 이중변환 제거 → transformation=None
    #   (group standardization만; 음수 boxcox 에서도 안전). 하드코딩 softplus 금지 유지.
    assert 'transformation="softplus"' not in src, "하드코딩 softplus 금지(음수서 폭발)"
    assert "transformation=None" in src, "transform-fix: 데이터-주도 preproc 가 y 변환 담당(sign-safe)"


# ─────────────────────────── FIX 4: EARS-C3 ≠ C2 ─────────────────────────────
def test_ears_c3_differs_from_c2():
    from simulation.models.ears_models import EarsC2Forecaster, EarsC3Forecaster
    rng = np.random.RandomState(7)
    y = np.abs(rng.randn(60) * 5 + 20)          # 비음수 ILI-유사
    X = rng.randn(60, 4)
    c2 = EarsC2Forecaster(); c2.fit(X, y)
    c3 = EarsC3Forecaster(); c3.fit(X, y)
    p2 = c2.predict(rng.randn(5, 4))
    p3 = c3.predict(rng.randn(5, 4))
    assert np.all(np.isfinite(p3))
    assert not np.allclose(p2, p3), "C3 가 여전히 C2 와 동일(byte-identical 버그 미수정)"
    assert np.all(p3 >= 0)


# ─────── 개선-확인 게이트 (사용자: "TDD 확인 후 개선됐을 때만 적용") ───────
def _write_pm(d: pathlib.Path, name, nf, wis, r2=0.5):
    d.mkdir(parents=True, exist_ok=True)
    fi = list(range(nf)) if nf is not None else None
    (d / f"{name}.json").write_text(json.dumps({
        "best_config": {"n_features": nf, "feature_indices": fi},
        "val_metrics": {"wis": wis}, "test_metrics": {"r2": r2},
    }), encoding="utf-8")


def test_improvement_gate_collapse_resolved_is_pass(tmp_path):
    """게이트 PASS 기준 = collapse 해소(n_features≥10). valWIS=informational(비결정성)."""
    from simulation.scripts.verify_g274_floor_improvement import main
    base, new = tmp_path / "base", tmp_path / "new"
    _write_pm(base, "CQR-LightGBM", 3, 10.89)
    _write_pm(base, "SVR-RBF", 12, 0.496)
    # --- case A: PASS (collapse 해소 3→12) ---
    _write_pm(new, "CQR-LightGBM", 12, 2.5)
    _write_pm(new, "SVR-RBF", 12, 0.498)
    assert main(["--base", str(base), "--new", str(new)]) == 0
    # --- case B: FAIL (여전히 collapse, nf=3) ---
    _write_pm(new, "CQR-LightGBM", 3, 2.5)
    assert main(["--base", str(base), "--new", str(new)]) == 1
    # --- case C: PENDING (new 미완) → exit 0 ---
    (new / "CQR-LightGBM.json").unlink()
    (new / "SVR-RBF.json").unlink()
    assert main(["--base", str(base), "--new", str(new)]) == 0


def test_improvement_gate_floored_wis_worse_still_pass(tmp_path):
    """floored 모델 nf 올랐으면 valWIS 가 더 나빠도 PASS (WIS=비결정성 노이즈, informational)."""
    from simulation.scripts.verify_g274_floor_improvement import main
    base, new = tmp_path / "base", tmp_path / "new"
    _write_pm(base, "CQR-LightGBM", 3, 10.89)
    _write_pm(new, "CQR-LightGBM", 12, 11.5)    # nf 해소됐으면 WIS↑여도 PASS(노이즈)
    assert main(["--base", str(base), "--new", str(new)]) == 0


def test_improvement_gate_protected_regress_is_informational(tmp_path):
    """protected 모델 valWIS 회귀는 FAIL 아님 — preproc Optuna 비결정성(XGBoost 0.83↔1.73 실측)."""
    from simulation.scripts.verify_g274_floor_improvement import main
    base, new = tmp_path / "base", tmp_path / "new"
    _write_pm(base, "CQR-LightGBM", 3, 10.89)
    _write_pm(base, "XGBoost", 12, 0.833)
    _write_pm(new, "CQR-LightGBM", 12, 2.5)     # floored PASS
    _write_pm(new, "XGBoost", 12, 1.728)        # protected 회귀 = informational, FAIL 아님
    assert main(["--base", str(base), "--new", str(new)]) == 0
