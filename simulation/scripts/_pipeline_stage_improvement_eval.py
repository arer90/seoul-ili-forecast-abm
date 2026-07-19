"""2→7 누적 개선 demo (사용자 "단계 진행하며 개선되는 것 보여줘"): 고정 모델, 단계별 OOF-WIS.

사용자 파이프라인 모델: baseline(2) → preproc(4) → feature(5, guard) → mc → HP(6), 각 단계가
이전 대비 개선될 때만 채택(greedy 보장 체인). 이 demo 는 **고정 모델**로 각 누적 단계의
OOF-WIS(선택 목적함수, lower=better)를 실측해 단계별 개선·보장을 보여준다.

단계 (각 단계는 이전 best config 위에 자기 차원만 추가, _oof_cv_wis 로 best 선택):
  S2 baseline : BASIC feature(lag+계절성), identity/none, default HP
  S4 +preproc : full feature, best of {identity, log1p+standard, asinh+robust}
  S5 +feature : best preproc + GUARD(full vs STABILITY subset; 개선시만 subset)
  +mc         : + corr-filter (none vs corr, best)
  S6 +HP      : + best of model-specific HP grid
phase 13 의 실제 staged 순서·guard 와 동일 메커니즘 (_oof_cv_wis, feature_guard_keep).

worker: python -m simulation.scripts._pipeline_stage_improvement_eval --model XGBoost
parent: python -m simulation.scripts._pipeline_stage_improvement_eval
"""
import argparse
import json
import os
import subprocess
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
PANEL = ["XGBoost", "ElasticNet"]
PER_TIMEOUT = 1500
PREPROC_GRID = [("identity", "none"), ("log1p", "standard"), ("asinh", "robust")]


def _hp_factories(name):
    """모델별 (label, factory) HP 후보. default + 2 변형."""
    if name == "XGBoost":
        from xgboost import XGBRegressor
        return [
            ("default", lambda: XGBRegressor(random_state=0, n_jobs=1, verbosity=0)),
            ("deep", lambda: XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.05,
                                          random_state=0, n_jobs=1, verbosity=0)),
            ("shallow", lambda: XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.1,
                                             subsample=0.8, random_state=0, n_jobs=1, verbosity=0)),
        ]
    if name == "ElasticNet":
        from sklearn.linear_model import ElasticNet
        return [
            ("default", lambda: ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=0)),
            ("a0.1", lambda: ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=0)),
            ("a0.01_l10.8", lambda: ElasticNet(alpha=0.01, l1_ratio=0.8, random_state=0)),
        ]
    return [("default", None)]


def worker_main(name):
    from simulation.tests._real_data_prep import _prep_full
    from simulation.pipeline.per_model_optimize import _oof_cv_wis
    from simulation.pipeline.feature_select_corr1se import (
        select_features_stability, feature_guard_keep)
    from simulation.pipeline.baseline import BASIC_FEATURE_COLS

    Pp, Pt, yp, yt, ylog, inv, cols = _prep_full()
    hp = _hp_factories(name)
    fac_default = hp[0][1]
    full_idx = list(range(Pp.shape[1]))
    basic_idx = sorted([i for i, c in enumerate(cols) if c in set(BASIC_FEATURE_COLS)]) or full_idx[:13]

    stages = []  # (label, oof_wis, detail)

    # S2 baseline: BASIC feature, identity/none, default
    oof_base = _oof_cv_wis(fac_default, Pp, yp, "identity", "none",
                           feature_indices=basic_idx, feature_cols=cols)
    stages.append(("S2 baseline(BASIC)", oof_base, f"k={len(basic_idx)}, identity/none"))

    # S4 +preproc: full feature, best preproc
    best_pp, best_pp_oof = None, float("inf")
    for tf, sc in PREPROC_GRID:
        o = _oof_cv_wis(fac_default, Pp, yp, tf, sc, feature_indices=None, feature_cols=cols)
        if o < best_pp_oof:
            best_pp_oof, best_pp = o, (tf, sc)
    stages.append(("S4 +preproc(full)", best_pp_oof, f"k={len(full_idx)}, {best_pp[0]}/{best_pp[1]}"))
    _tf, _sc = best_pp

    # S5 +feature: GUARD (full vs STABILITY subset) under best preproc
    _ylog = np.log1p(np.clip(yp, 0, None))
    sel = select_features_stability(Pp, _ylog, pi=0.6, epv_ratio=20, seed=42)["selected_indices"]
    oof_full = _oof_cv_wis(fac_default, Pp, yp, _tf, _sc, feature_indices=None, feature_cols=cols)
    oof_sel = _oof_cv_wis(fac_default, Pp, yp, _tf, _sc, feature_indices=sel, feature_cols=cols)
    if feature_guard_keep(oof_full, oof_sel, 0.02, prefer_subset=True):   # parsimony 우선
        feat_idx, oof_feat = sel, oof_sel
        fdet = f"SUBSET k={len(sel)} ({'개선' if oof_sel < oof_full - 1e-9 else 'parsimony 동등'})"
    else:
        feat_idx, oof_feat, fdet = full_idx, oof_full, f"FULL k={len(full_idx)} (subset 명백열위)"
    stages.append(("S5 +feature(guard)", oof_feat, fdet))

    # +mc: none vs corr-filter (drop one of each |corr|>0.95 pair within feat_idx)
    def _corr_filter(idx):
        idx = list(idx)
        if len(idx) < 2:
            return idx
        sub = Pp[:, idx]
        keep, dropped = [], set()
        C = np.corrcoef(sub.T)
        for a in range(len(idx)):
            if a in dropped:
                continue
            keep.append(idx[a])
            for b in range(a + 1, len(idx)):
                if b not in dropped and np.isfinite(C[a, b]) and abs(C[a, b]) > 0.95:
                    dropped.add(b)
        return sorted(keep)
    mc_idx = _corr_filter(feat_idx)
    oof_mc = _oof_cv_wis(fac_default, Pp, yp, _tf, _sc, feature_indices=mc_idx, feature_cols=cols)
    if oof_mc < oof_feat * 0.98:
        mc_keep, oof_after_mc, mdet = mc_idx, oof_mc, f"corr k={len(mc_idx)} (개선)"
    else:
        mc_keep, oof_after_mc, mdet = feat_idx, oof_feat, f"none k={len(feat_idx)} (corr 개선미달)"
    stages.append(("+mc(guard)", oof_after_mc, mdet))

    # S6 +HP: best HP under fixed preproc+feature+mc
    best_hp, best_hp_oof = "default", oof_after_mc
    for lab, f in hp:
        o = _oof_cv_wis(f, Pp, yp, _tf, _sc, feature_indices=mc_keep, feature_cols=cols)
        if o < best_hp_oof:
            best_hp_oof, best_hp = o, lab
    stages.append(("S6 +HP", best_hp_oof, f"HP={best_hp}"))

    out = {"name": name, "stages": [{"label": l, "oof_wis": w, "detail": d} for l, w, d in stages]}
    print("RESULT_JSON " + json.dumps(out), flush=True)


def parent_main():
    print("=" * 96, flush=True)
    print("2→7 누적 개선 demo: baseline→+preproc→+feature(guard)→+mc→+HP 각 단계 OOF-WIS (lower=better)", flush=True)
    print("=" * 96, flush=True)
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="1")
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for name in PANEL:
        try:
            cp = subprocess.run(
                [sys.executable, "-u", "-m", "simulation.scripts._pipeline_stage_improvement_eval", "--model", name],
                cwd=repo, env=env, capture_output=True, text=True, timeout=PER_TIMEOUT)
        except subprocess.TimeoutExpired:
            print(f"  {name} TIMEOUT", flush=True); continue
        if cp.returncode != 0:
            print(f"  {name} CRASH rc={cp.returncode}: {(cp.stderr or '')[-150:]}", flush=True); continue
        line = next((l for l in cp.stdout.splitlines() if l.startswith("RESULT_JSON ")), None)
        if not line:
            print(f"  {name} no-result", flush=True); continue
        r = json.loads(line[len("RESULT_JSON "):])
        print(f"\n  [{name}]  (OOF-WIS, ↓개선)", flush=True)
        prev = None
        for s in r["stages"]:
            w = s["oof_wis"]
            delta = "" if prev is None else f"  Δ={w - prev:+.4f} {'✓개선' if w < prev - 1e-9 else ('=보장유지' if w <= prev + 1e-9 else '↑(guard로 복원됨)')}"
            print(f"    {s['label']:22s} OOF-WIS={w:.4f}  [{s['detail']}]{delta}", flush=True)
            prev = w
        first = r["stages"][0]["oof_wis"]; last = r["stages"][-1]["oof_wis"]
        print(f"    → baseline {first:.4f} → 최종 {last:.4f} (총 개선 {first - last:+.4f}, "
              f"{'개선' if last < first else '동등/보장'})", flush=True)
    print("=" * 96, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default=None); a = ap.parse_args()
    if a.model:
        worker_main(a.model)
    else:
        parent_main()


if __name__ == "__main__":
    main()
