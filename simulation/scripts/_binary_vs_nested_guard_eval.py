"""실측 비교: phase-13 feature guard = BINARY{subset,full} vs NESTED size-path+1-SE (사용자 결정 2026-06-01 "둘 다 — 구현 + 실측 비교").

codex+Gemini 권고대로 nested(π ladder 0.8/0.6/0.4 + full, 1-SE/parsimony)를 구현(opt-in
MPH_FEAT_PATH=nested) 후, **production 코드(_oof_cv_wis · _evaluate_config · 새 helper)를 직접 호출**해
family당 대표 1모델에서 두 전략의 OOF-WIS + **held-out test-WIS(누수 0)**를 비교.

고정 preproc: _prep_full()이 이미 QuantileTransformer 적용 → transform=identity/scaler=none 으로
feature 선택(guard)만 격리. binary/nested 동일 데이터·동일 preproc → 차이 = 선택 전략뿐.

worker: python -m simulation.scripts._binary_vs_nested_guard_eval --model XGBoost
parent: python -m simulation.scripts._binary_vs_nested_guard_eval
"""
import argparse
import json
import os
import subprocess
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
PANEL = ["XGBoost", "ElasticNet", "KRR", "GAM-Spline", "NegBinGLM", "CQR-LightGBM", "TabularDNN"]
PER_MODEL_TIMEOUT = 900
PI_LEVELS = (0.8, 0.6, 0.4)
MARGIN = 0.02


def _test_wis(evaluate_config, fac, Pp, yp, Pt, yt, idx, p):
    """Held-out test WIS for a feature set (refit on full train, eval on test). 누수 0."""
    fi = None if len(idx) >= p else list(idx)
    try:
        cell = evaluate_config(
            fac, Pp, yp, Pt, yt, transform_name="identity", scaler_name="none",
            feature_indices=fi, sigma_for_wis=max(float(np.std(yp)), 1e-3),
            feature_cols=None, calib_residuals=None)
        w = cell.get("wis", 1e9) if isinstance(cell, dict) else 1e9
        return float(w) if np.isfinite(w) else 1e9
    except Exception:
        return 1e9


def _load_model_class(name):
    """REGISTRY lookup with torch-free import for LightGBM-based models.

    torch + lightgbm in the SAME process → macOS OpenMP segfault (OMP Error #179
    pthread_mutex_init, observed on CQR-LightGBM). The real training pipeline isolates each
    model in its own subprocess so they never coexist; this comparison worker must do the same
    by NOT importing torch (via dl_models) when running a LightGBM model. sklearn/statsmodels
    models (KRR/GAM/NegBin/XGBoost) tolerate force_import (no lightgbm OpenMP clash).
    """
    from simulation.models.base import REGISTRY
    if "CQR" in name.upper() or "LightGBM" in name or "LGBM" in name.upper():
        # macOS: lightgbm n_jobs>1 + 다른 OpenMP 런타임(torch via phase13, MKL) → pthread_mutex_init
        # #179 segfault. 모델은 n_jobs=2 하드코딩 → harness 전용 monkeypatch 로 단일스레드 강제
        # (production 코드 불변; 실 파이프라인은 subprocess 격리라 무관).
        try:
            import functools
            import lightgbm as _lgb
            _orig_init = _lgb.LGBMRegressor.__init__

            @functools.wraps(_orig_init)   # preserve signature → sklearn get_params introspection OK
            def _force_single_thread(self, *a, **k):
                k["n_jobs"] = 1
                return _orig_init(self, *a, **k)

            _lgb.LGBMRegressor.__init__ = _force_single_thread
        except Exception:
            pass
        import simulation.models.cqr_models   # noqa: F401  torch-free → lightgbm safe
        import simulation.models.tree_models   # noqa: F401  other LightGBM live here (torch-free)
    else:
        from simulation.models.registry import verify_registry_coverage
        verify_registry_coverage(force_import=True)
    return REGISTRY.get(name)


def worker_main(name):
    from simulation.tests._real_data_prep import _prep_full
    from simulation.pipeline.per_model_optimize import _oof_cv_wis, _evaluate_config
    from simulation.pipeline.feature_select_corr1se import (
        select_features_stability, feature_guard_keep, build_nested_size_path, select_size_path_1se)
    cls = _load_model_class(name)
    if cls is None:
        print("RESULT_JSON null", flush=True); return
    fac = (lambda c=cls: c())

    Pp, Pt, yp, yt, _u1, _u2, cols = _prep_full()
    p = int(Pp.shape[1])
    ylog = np.log1p(np.clip(yp, 0, None))
    tf, sc = "identity", "none"            # Pp already QT'd → isolate the guard

    _sel = select_features_stability(Pp, ylog, epv_ratio=20, seed=42)
    freq = _sel["stability"]; subset = _sel["selected_indices"]

    # ── BINARY guard {subset, full} (parsimony, 현행 default) ──────────────────
    oof_full = _oof_cv_wis(fac, Pp, yp, tf, sc, feature_indices=None, feature_cols=cols)
    oof_sub = _oof_cv_wis(fac, Pp, yp, tf, sc, feature_indices=subset, feature_cols=cols)
    if feature_guard_keep(oof_full, oof_sub, MARGIN, prefer_subset=True):
        bin_idx, bin_oof, bin_pick = subset, oof_sub, f"SUBSET(k={len(subset)})"
    else:
        bin_idx, bin_oof, bin_pick = list(range(p)), oof_full, f"FULL(k={p})"

    # ── NESTED size-path + 1-SE/parsimony ─────────────────────────────────────
    cands = build_nested_size_path(freq, p, pi_levels=PI_LEVELS, min_keep=1)
    means, folds, sizes = [], [], []
    for c in cands:
        fi = None if len(c) >= p else c
        m, fl = _oof_cv_wis(fac, Pp, yp, tf, sc, feature_indices=fi,
                            feature_cols=cols, return_folds=True)
        means.append(m); folds.append(fl); sizes.append(len(c))
    pick = select_size_path_1se(means, sizes, fold_scores=folds, margin=MARGIN, se_mult=1.0)
    nst_idx, nst_oof = cands[pick], means[pick]

    out = {
        "name": name, "p": p, "stability_subset": len(subset), "nested_sizes": sizes,
        "binary": {"k": len(bin_idx), "oof": bin_oof, "pick": bin_pick,
                   "test": _test_wis(_evaluate_config, fac, Pp, yp, Pt, yt, bin_idx, p)},
        "nested": {"k": len(nst_idx), "oof": nst_oof, "pick_size": sizes[pick],
                   "test": _test_wis(_evaluate_config, fac, Pp, yp, Pt, yt, nst_idx, p)},
    }
    print("RESULT_JSON " + json.dumps(out), flush=True)


def parent_main():
    print("=" * 104, flush=True)
    print("BINARY{subset,full} vs NESTED(π ladder+1-SE) guard — OOF-WIS + held-out TEST-WIS (↓) [고정 preproc]", flush=True)
    print("=" * 104, flush=True)
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="1")
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rows = {}
    for name in PANEL:
        try:
            cp = subprocess.run(
                [sys.executable, "-u", "-m", "simulation.scripts._binary_vs_nested_guard_eval", "--model", name],
                cwd=repo, env=env, capture_output=True, text=True, timeout=PER_MODEL_TIMEOUT)
        except subprocess.TimeoutExpired:
            print(f"  {name:14s} TIMEOUT", flush=True); continue
        if cp.returncode != 0:
            print(f"  {name:14s} CRASH rc={cp.returncode}: {(cp.stderr or '')[-160:]}", flush=True); continue
        line = next((l for l in cp.stdout.splitlines() if l.startswith("RESULT_JSON ")), None)
        r = json.loads(line[len("RESULT_JSON "):]) if line and line != "RESULT_JSON null" else None
        if r:
            rows[name] = r
            print(f"  ✓ {name}", flush=True)
        else:
            print(f"  {name:14s} no-result", flush=True)

    print("\n  === 결과 (lower=better; TEST = held-out 누수0 = 일반화 진실) ===", flush=True)
    print(f"  {'model':14s}{'binary k':>10s}{'bin OOF':>10s}{'bin TEST':>11s}"
          f"{'nested k':>10s}{'nst OOF':>10s}{'nst TEST':>11s}{'TEST Δ(nst-bin)':>17s}", flush=True)
    n_better, n_worse, n_tie = 0, 0, 0
    for name in PANEL:
        if name not in rows:
            continue
        r = rows[name]; b, nst = r["binary"], r["nested"]
        dt = nst["test"] - b["test"]
        tag = "≈" if abs(dt) < 0.02 * max(b["test"], 1e-9) else ("nst↓better" if dt < 0 else "bin↓better")
        if tag == "≈": n_tie += 1
        elif dt < 0: n_better += 1
        else: n_worse += 1
        print(f"  {name:14s}{b['k']:>10d}{b['oof']:>10.3f}{b['test']:>11.3f}"
              f"{nst['k']:>10d}{nst['oof']:>10.3f}{nst['test']:>11.3f}{dt:>+12.3f} {tag}", flush=True)
        print(f"      nested sizes={r['nested_sizes']} → pick k={nst['pick_size']}; binary {b['pick']}", flush=True)
    print(f"\n  === TEST 종합: nested 우세 {n_better} / 동등 {n_tie} / binary 우세 {n_worse} (of {len(rows)}) ===", flush=True)
    print("  (nested 가 test 에서 일관 우세면 채택; 동등이면 binary 유지가 더 단순/안전 — codex+Gemini 권고)", flush=True)
    print("=" * 104, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default=None); a = ap.parse_args()
    if a.model:
        worker_main(a.model)
    else:
        parent_main()


if __name__ == "__main__":
    main()
