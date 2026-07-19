"""preproc stable space 재설계 (2026-06-16, 사용자 X/Y 별개 + 재현성).

설계:
  Y: STABLE 모드 → y_mode ∈ {none, individual} + transform ∈ STABLE_Y_TRANSFORMS
     (group/categorical·발산군 mcmc_robust/anscombe/boxcox/yeo 제외) — 1D 타깃, 역변환 발산 차단.
  X: STABLE 모드 → x_mode ∈ {none, group}; group = data_driven_group_scalers(결정적, Optuna 탐색 X)
     — 그룹별 도메인-인지 스케일링 유지하되 3²⁰ 비결정성 제거.
  재현성: PYTHONHASHSEED=42 를 python 기동 전 run_pipeline.sh 가 export.

per-file 실행(OMP_NUM_THREADS=1) 권장.
"""
import os
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]


def _study(obj, n=30):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    s = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42))
    s.optimize(obj, n_trials=n, show_progress_bar=False)
    return s


# ───────────────────────── Y: stable 제한 ─────────────────────────
def test_y_stable_only_none_individual_safe():
    os.environ["MPH_STABLE_TRANSFORMS"] = "1"
    from simulation.pipeline.preproc_optuna_hierarchical import (
        suggest_y_preproc, STABLE_Y_TRANSFORMS)
    y = np.abs(np.random.RandomState(0).randn(200) * 5 + 20)
    modes, tfs = set(), set()

    def obj(t):
        _, _, st = suggest_y_preproc(t, y, max_chain_length=2)
        modes.add(st.get("y_mode"))
        if st.get("y_mode") == "individual":
            tfs.add(st.get("y_individual"))
        return 0.0
    _study(obj, 40)
    assert modes <= {"none", "individual"}, f"group/categorical 누출: {modes}"
    assert tfs <= set(STABLE_Y_TRANSFORMS), f"발산군 누출: {tfs - set(STABLE_Y_TRANSFORMS)}"
    # 발산군이 실제로 제외됐는지 명시 (mcmc_robust=affine 안전이라 STABLE 에 포함, 제외 아님)
    for bad in ("anscombe", "freeman_tukey", "boxcox", "yeo_johnson"):
        assert bad not in tfs


# ───────────────────────── X: stable {none, group-datadriven} ─────────────────────────
def test_x_stable_none_group_datadriven_only():
    os.environ["MPH_STABLE_TRANSFORMS"] = "1"
    from simulation.pipeline.preproc_optuna_hierarchical import suggest_x_scaler
    rng = np.random.RandomState(0)
    X = rng.randn(200, 6)
    fg = {"a": [0, 1], "b": [2, 3], "c": [4, 5]}
    modes, srcs, params = set(), set(), set()

    def obj(t):
        _, _, _, st = suggest_x_scaler(t, X, X, feature_groups=fg)
        modes.add(st["x_mode"])
        srcs.add(st.get("x_group_source"))
        params.update(t.params.keys())
        return 0.0
    _study(obj, 30)
    assert modes <= {"none", "group"}, f"individual/categorical 누출: {modes}"
    assert srcs <= {None, "data_driven"}, f"Optuna per-group 누출: {srcs}"
    # 데이터-기반이면 x_group_<name> Optuna 파라미터가 없어야 (탐색 제거 증명)
    assert not any(p.startswith("x_group_") for p in params), f"x_group Optuna dim 잔존: {params}"


# ───────────────────────── data_driven_group_scalers 규칙 ─────────────────────────
def test_data_driven_scaler_rule_and_determinism():
    from simulation.pipeline.preproc_optuna_hierarchical import data_driven_group_scalers
    rng = np.random.RandomState(1)
    X = np.column_stack([
        rng.standard_t(2, 300), rng.standard_t(2, 300),   # g_heavy: heavy-tail
        rng.randn(300), rng.randn(300),                    # g_norm: well-behaved
    ])
    fg = {"g_heavy": [0, 1], "g_norm": [2, 3]}
    m1 = data_driven_group_scalers(X, fg)
    m2 = data_driven_group_scalers(X, fg)
    assert m1 == m2, "비결정적 (같은 X → 다른 맵)"
    assert m1["g_heavy"] == "quantile", m1            # heavy-tail → rank
    assert m1["g_norm"] == "standard", m1             # normal → standard
    assert set(m1.values()) <= {"standard", "robust", "quantile"}


def test_data_driven_empty_and_constant_safe():
    from simulation.pipeline.preproc_optuna_hierarchical import data_driven_group_scalers
    X = np.column_stack([np.ones(100), np.random.RandomState(0).randn(100)])  # col0 상수
    fg = {"empty": [], "const": [0], "oob": [99], "ok": [1]}
    m = data_driven_group_scalers(X, fg)
    assert m["empty"] == "standard"      # 빈 그룹 안전
    assert m["const"] == "standard"      # 상수 feature 안전(std≈0)
    assert m["oob"] == "standard"        # 범위 밖 인덱스 안전
    assert m["ok"] in {"standard", "robust", "quantile"}


# ───────────────────────── 재현성: PYTHONHASHSEED ─────────────────────────
def test_pythonhashseed_exported_in_launch():
    src = (ROOT / "run_pipeline.sh").read_text(encoding="utf-8")
    assert "export PYTHONHASHSEED=42" in src, "PYTHONHASHSEED 미export → TPE 비결정"


def test_stable_constants_present():
    from simulation.pipeline import preproc_optuna_hierarchical as h
    # 6개 = 발산-안전(affine: laplace,mcmc_robust / capped: log1p,sqrt,fourth_root,asinh).
    #   G-333(2026-06-22): flat-grid 재설계서 fourth_root(Taylor's Power Law VST, France 2022) 추가.
    assert h.STABLE_Y_TRANSFORMS == ["log1p", "sqrt", "fourth_root", "asinh", "laplace", "mcmc_robust"]
    assert h.STABLE_PREPROC_MODES == ["none", "individual"]   # Y
    assert h.STABLE_X_MODES == ["none", "group"]               # X (별개)


def test_sqrt_inverse_capped():
    """sqrt inverse 가 cap 적용 (uncapped x² 2차 발산 회귀 가드 — 3자 감사 2026-06-16)."""
    from simulation.pipeline.preproc_optuna_hierarchical import _apply_single_y_transform
    y = np.abs(np.random.RandomState(0).randn(200) * 5 + 20)
    cap = max(float(y.max()) * 10.0, 100.0)
    yt, inv, state = _apply_single_y_transform(y, "sqrt")
    assert "safe_cap" in state and abs(state["safe_cap"] - cap) < 1e-6
    # large-z 외삽: 모델이 sqrt-공간서 z=50/100/200 → 옛날 2500/10000/40000 발산, 이제 cap 에서 멈춤
    out = inv(np.array([50.0, 100.0, 200.0]))
    assert np.all(out <= cap + 1e-6), f"sqrt inverse 발산: {out} > cap {cap}"
    assert np.all(np.isfinite(out))
    # round-trip 정확 (normal range 는 cap 미만이라 무손실)
    assert np.allclose(inv(yt), y, atol=1e-4)


def test_all_stable_y_inverses_finite_and_nonexplosive():
    """STABLE_Y 5개 모두 round-trip·외삽서 finite. capped(log1p/sqrt/asinh)는 bounded."""
    from simulation.pipeline.preproc_optuna_hierarchical import (
        _apply_single_y_transform, STABLE_Y_TRANSFORMS)
    y = np.abs(np.random.RandomState(1).randn(200) * 5 + 20)
    cap = max(float(y.max()) * 10.0, 100.0)
    for name in STABLE_Y_TRANSFORMS:
        yt, inv, _ = _apply_single_y_transform(y, name)
        assert np.all(np.isfinite(inv(yt))), f"{name}: round-trip non-finite"
        big = inv(np.array([40.0, 80.0]))
        assert np.all(np.isfinite(big)), f"{name}: 외삽 non-finite"
        if name in ("log1p", "sqrt", "asinh"):   # capped/clipped → bounded
            assert np.all(big <= cap * 100), f"{name}: capped 인데 발산 {big}"


def test_optuna_hp_search_is_seeded_for_reproducibility():
    """Optuna samplers fix a seed, with a pinned set of exceptions.

    2026-07-19: this asserted the opposite — that NO file may fix a sampler seed,
    citing "TPE is exploration, reproducibility unnecessary". That decision was
    reversed by G-257 and G-13F, whose comments at the call sites say so
    ("HPO 탐색도 재현"). Measured across the tree: 17 of 21 TPESampler call sites
    fix a seed. Seeding is the policy; the test was the leftover.

    This matters for what the distribution claims. An unseeded HP search cannot
    be re-run to the same hyperparameters, so the champion's recorded config
    would not be reproducible — see SETUP.md section 6.3.

    The four exceptions are pinned rather than waived, so a new unseeded sampler
    fails here instead of quietly eroding the guarantee.
    """
    import re

    # (file, distinguishing fragment of the call's arguments)
    ALLOWED_UNSEEDED = {
        # Builder helpers — the caller supplies the seed via seed=seed.
        ("simulation/models/_optuna_samplers.py", "multivariate=False"),
        ("simulation/models/_optuna_samplers.py", "multivariate=True, group=True"),
        # Pipeline-level structure search; preproc no longer uses Optuna at all
        # after G-335 replaced it with a flat grid.
        ("simulation/pipeline/_inline_optuna_3stage.py", "n_startup_trials=_n_startup"),
        ("simulation/pipeline/preproc_optuna_hierarchical.py", "multivariate=True, group=True"),
    }

    import subprocess
    tracked = subprocess.run(["git", "ls-files", "*.py"], cwd=ROOT,
                             capture_output=True, text=True, encoding="utf-8").stdout.split()
    pat = re.compile(r"TPESampler\(([^)]*)\)", re.S)
    seeded, unexpected = 0, []
    for rel in tracked:
        if rel.startswith(("simulation/tests/", "tests/")):
            continue
        try:
            src = (ROOT / rel).read_text(encoding="utf-8")
        except OSError:
            continue
        for raw in pat.findall(src):
            args = " ".join(raw.split())
            if "seed=" in args:
                seeded += 1
                continue
            if any(rel == f and frag in args for f, frag in ALLOWED_UNSEEDED):
                continue
            unexpected.append(f"{rel}: TPESampler({args[:70]})")

    assert not unexpected, (
        "unseeded Optuna sampler(s) outside the pinned exceptions — an HP search "
        "that cannot be replayed breaks the reproducibility claim:\n  "
        + "\n  ".join(unexpected)
    )
    assert seeded >= 15, f"only {seeded} seeded samplers found — did the HP layer move?"
