"""통합 feature 방법 비교 (사용자 "지금까지 feature-optuna+optimization 한 것 전부 포함, family당 1모델").

방법 13 (feature-optuna 계열 + feature-optimization 계열 + FULL 기준):
  FULL          선택 X (전체 feature)                              [기준]
  STABILITY     |corr| 재표본 빈도 (Meinshausen-Bühlmann)           [optimization, 채택]
  MODEL_BASED   각 모델 importance 재표본 stability                 [optimization]
  POOL          |corr| stability π=0.4 (큰 풀)                       [optimization]
  BLEND         z(|corr|)+z(importance) stability                    [optimization]
  MI            mutual-info 재표본 stability (비선형 marginal)         [optimization]
  RFE           Recursive Feature Elimination (RF surrogate)          [optimization]
  mRMR          max-relevance − min-redundancy                       [optimization]
  EMBEDDED      모델 coef_/feature_importances_ > mean               [optimization]
  BINARY        Optuna 0/1 TPE (조합 탐색)                            [feature-optuna, 사용자 원안]
  FORWARD       greedy 추가                                          [feature-optuna]
  BACKWARD      greedy 제거                                          [feature-optuna]
  THRESHOLD     Optuna τ on |corr| (일정수치 미만 버림)                [feature-optuna, 사용자 원안]
모델 7 (family당 대표): XGBoost(tree)·ElasticNet(linear)·KRR(kernel)·GAM-Spline(GAM)
  ·NegBinGLM(count/GLM)·CQR-LightGBM(cqr)·TabularDNN(dl).
고정 preproc(QuantileTransformer + log1p y) → **feature 방법만 격리**. OOF-WIS(3-fold)+test-WIS, per-model subprocess.

═══ 2026-06-01 LEAKAGE-FREE + DE-CIRCULARIZED REWRITE (codex+Gemini 3-way eval) ═══
두 CRITICAL 수정 (사용자 "수정 후 재실행"):
  1. **누수 제거**: feature 선택을 OOF fold **안에서** training prefix(Pp[:etr])로만 재실행
     (`_oof3_nested`). 이전: full pool 한 번 선택 → fold 채점 = selection 이 validation row 를 봄.
     test 열은 full-train 선택 → held-out test 평가(test 미접근 = 누수 아님, 표준)로 유지.
  2. **POOL_M 탈circular화**: wrapper 후보 풀 = `_fold_cand` = union(top-|corr|, top-RF-importance).
     이전: top-18 |corr| 만 → wrapper 가 |corr| 놓친 interaction feature 도달 불가(편향).
  3. **error bar**: OOF 는 fold 간 mean±sd 보고 (판별력 = 차이가 sd 안인지 가시화).
선택 = pool-only OOF (test 인덱스 미접근). 절대값은 phase13 _oof_cv_wis(5-fold)와 다름 — 상대 ranking + sd 가 해석 단위.

worker: python -m simulation.scripts._unified_feature_method_comparison --model XGBoost
parent: python -m simulation.scripts._unified_feature_method_comparison
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
METHODS = ["FULL", "STABILITY", "MODEL_BASED", "POOL", "BLEND", "MI", "RFE",
           "mRMR", "EMBEDDED", "BINARY", "FORWARD", "BACKWARD", "THRESHOLD"]
POOL_CORR = 12     # wrapper 후보 풀: top-|corr| 부분
POOL_IMP = 10      # wrapper 후보 풀: top-RF-importance 부분 (탈circular화 — |corr| 놓친 feature 도달)
N_TRIALS = 10      # binary/threshold Optuna (cost-control; fold 안 3회 재실행 고려)
STAB_B = 20        # stability 재표본
FWD_CAP = 6        # forward greedy 상한 (fold 안 3회 재실행 고려)
PER_MODEL_TIMEOUT = 3600   # nested(fold당 재선택) → 비용 증가, DL/GAM 여유 (초과 시 TIMEOUT 보고)


def _ewis(y, pred, resid):
    from simulation.analytics.diagnostics import weighted_interval_score_empirical
    from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
    r = np.asarray(resid, float).ravel(); r = r[np.isfinite(r)]
    if r.size < 2:
        r = np.array([np.std(y) or 1.0, 0.0])
    return np.asarray(weighted_interval_score_empirical(
        np.asarray(y, float), np.asarray(pred, float), residuals=r, alphas=FLUSIGHT_ALPHAS), float)


def _ac(X, y):
    y = np.asarray(y, float).ravel()
    out = np.array([abs(np.corrcoef(X[:, j], y)[0, 1]) if (np.std(X[:, j]) > 1e-9 and np.std(y) > 1e-9) else 0.0
                    for j in range(X.shape[1])], float)
    out[~np.isfinite(out)] = 0.0
    return np.clip(out, 0.0, 1.0)


def _pred1d(m, X):
    p = np.asarray(m.predict(X), float)
    return p[:, p.shape[1] // 2] if p.ndim == 2 and p.shape[1] > 1 else p.ravel()


def _fold_cand(Pp_tr, ylog_tr):
    """De-circularized wrapper candidate pool, computed on TRAINING data only.

    Returns (sorted candidate indices, |corr| vector). The pool is the UNION of top-|corr|
    and top-RF-importance features, so wrapper methods (forward/backward/binary/threshold/mRMR)
    can reach features that marginal |corr| alone misses (interaction-only features have low
    |corr| but high RF importance). Fixes the "top-18 |corr| straitjacket" that rigged the
    bake-off toward |corr|-family methods (codex+Gemini 3-way eval).

    Args:
        Pp_tr: (n_tr, p) training-prefix design matrix (preprocessed).
        ylog_tr: (n_tr,) log1p target on the same rows.
    Returns:
        (cand, cs): cand = sorted unique indices (|corr| ∪ importance); cs = |corr| vector (len p).
    Performance: one RandomForest fit (n_estimators=80) per call. Caller pre-computes once per fold.
    """
    cs = _ac(Pp_tr, ylog_tr)
    by_corr = np.argsort(cs)[::-1][:POOL_CORR].tolist()
    by_imp = []
    try:
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=80, max_depth=6, random_state=0, n_jobs=1).fit(Pp_tr, ylog_tr)
        imp = np.asarray(rf.feature_importances_, float).ravel()
        if imp.shape[0] == Pp_tr.shape[1] and np.all(np.isfinite(imp)):
            by_imp = np.argsort(imp)[::-1][:POOL_IMP].tolist()
    except Exception:
        by_imp = []
    return sorted(set(by_corr) | set(by_imp)), cs


def _oof3_nested(method, fac, Pp, yp, ylog, inv, folds):
    """Leakage-free OOF-WIS: feature selection is re-run INSIDE each fold using the training
    prefix (Pp[:etr]) only; the validation rows (etr:eva) are never seen by the selector.

    Args:
        method: feature-selection method name (routed through `_select`).
        fac: 0-arg model factory.
        Pp, yp, ylog: full train-pool design / raw target / log1p target.
        inv: inverse transform (log1p → raw).
        folds: list of (etr, eva, cand_tr, cs_tr) — precomputed per-fold training prefixes +
               de-circularized candidate pools (from `_fold_cand` on Pp[:etr]).
    Returns:
        (mean_wis, std_wis) across folds (std = fold dispersion for error bars); (1e9, 0.0) if all fail.
    """
    ws = []
    for (etr, eva, cand_tr, cs_tr) in folds:
        Pp_tr = Pp[:etr]; ylog_tr = ylog[:etr]; yp_tr = yp[:etr]
        try:
            idx = _select(method, fac, Pp_tr, yp_tr, ylog_tr, inv, cand_tr, cs_tr)
            if not idx:
                continue
            m = fac(); m.fit(Pp_tr[:, idx], ylog_tr)
            pe = inv(_pred1d(m, Pp[etr:eva][:, idx])); rs = yp_tr - inv(_pred1d(m, Pp_tr[:, idx]))
            w = float(np.median(_ewis(yp[etr:eva], pe, rs)))
            if np.isfinite(w):
                ws.append(w)
        except Exception:
            pass
    return (float(np.mean(ws)), float(np.std(ws))) if ws else (1e9, 0.0)


def _test(fac, Pp, Pt, yp, yt, ylog, inv, idx):
    idx = list(idx)
    if not idx:
        return 1e9
    try:
        m = fac(); m.fit(Pp[:, idx], ylog)
        pred = inv(_pred1d(m, Pt[:, idx])); rs = yp - inv(_pred1d(m, Pp[:, idx]))
        return float(np.mean(_ewis(yt, pred, rs)))
    except Exception:
        return 1e9


def _holdout(fac, Pp, yp, ylog, inv, idx):
    idx = list(idx); n = len(yp); cut = int(n * 0.8)
    if cut < 4 or n - cut < 2 or not idx:
        return 1e9
    try:
        m = fac(); m.fit(Pp[:cut][:, idx], ylog[:cut])
        pe = inv(_pred1d(m, Pp[cut:][:, idx])); rs = yp[:cut] - inv(_pred1d(m, Pp[:cut][:, idx]))
        w = float(np.median(_ewis(yp[cut:], pe, rs)))
        return w if np.isfinite(w) else 1e9
    except Exception:
        return 1e9


def _select(method, fac, Pp, yp, ylog, inv, cand, cs):
    """method → 선택 feature 인덱스. cand=top-POOL_M pool, cs=|corr| 전체."""
    from simulation.pipeline.feature_select_corr1se import (
        select_features_stability, make_model_importance_fn, forward_select, backward_select)
    full = list(range(Pp.shape[1]))
    if method == "FULL":
        return full
    if method == "STABILITY":
        return select_features_stability(Pp, ylog, pi=0.6, B=STAB_B, epv_ratio=20, seed=42)["selected_indices"]
    if method == "POOL":
        return select_features_stability(Pp, ylog, pi=0.4, B=STAB_B, epv_ratio=20, seed=42)["selected_indices"]
    if method == "MODEL_BASED":
        return select_features_stability(Pp, ylog, pi=0.6, B=STAB_B, epv_ratio=20, seed=42,
                                         importance_fn=make_model_importance_fn(fac), model_based_min_n=1)["selected_indices"]
    if method == "BLEND":
        mb = make_model_importance_fn(fac)
        def _z(v):
            v = np.asarray(v, float).ravel(); s = float(np.std(v))
            return (v - float(np.mean(v))) / s if s > 1e-9 else v * 0.0
        def blend(Xs, ys):
            c = _ac(Xs, ys); m = np.asarray(mb(Xs, ys), float).ravel()
            return c if m.shape[0] != c.shape[0] or not np.all(np.isfinite(m)) else _z(c) + _z(m)
        return select_features_stability(Pp, ylog, pi=0.6, B=STAB_B, epv_ratio=20, seed=42,
                                         importance_fn=blend, model_based_min_n=1)["selected_indices"]
    if method == "MI":
        from sklearn.feature_selection import mutual_info_regression
        return select_features_stability(Pp, ylog, pi=0.6, B=STAB_B, epv_ratio=20, seed=42,
                                         importance_fn=lambda Xs, ys: mutual_info_regression(Xs, ys, random_state=0),
                                         model_based_min_n=1)["selected_indices"]
    if method == "RFE":
        from sklearn.feature_selection import RFE
        from sklearn.ensemble import RandomForestRegressor
        k = max(1, len(Pp) // 20)
        sel = RFE(RandomForestRegressor(n_estimators=60, max_depth=6, random_state=0, n_jobs=1),
                  n_features_to_select=min(k, Pp.shape[1]), step=0.3).fit(Pp, ylog)
        return sorted(np.where(sel.support_)[0].tolist())
    if method == "mRMR":
        k = max(1, len(Pp) // 20); rel = {c: cs[c] for c in cand}
        sel, rem = [], list(cand)
        while len(sel) < k and rem:
            best, bs = None, -1e18
            for c in rem:
                red = max((abs(np.corrcoef(Pp[:, c], Pp[:, s])[0, 1]) for s in sel), default=0.0) if sel else 0.0
                sc = rel[c] - (red if np.isfinite(red) else 0.0)
                if sc > bs:
                    bs, best = sc, c
            sel.append(best); rem.remove(best)
        return sorted(sel)
    if method == "EMBEDDED":
        mb = make_model_importance_fn(fac)
        imp = np.asarray(mb(Pp, ylog), float).ravel()
        if imp.shape[0] != Pp.shape[1] or not np.all(np.isfinite(imp)):
            return full
        thr = float(np.mean(imp))
        return sorted([j for j in range(Pp.shape[1]) if imp[j] > thr]) or full
    if method == "FORWARD":
        return forward_select(lambda idx: _holdout(fac, Pp, yp, ylog, inv, idx), cand, k_cap=FWD_CAP)
    if method == "BACKWARD":
        return backward_select(lambda idx: _holdout(fac, Pp, yp, ylog, inv, idx), cand, k_min=1)
    if method == "BINARY":
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        def obj(t):
            idx = [cand[i] for i in range(len(cand)) if t.suggest_categorical(f"u{i}", [0, 1]) == 1]
            return _holdout(fac, Pp, yp, ylog, inv, idx) if idx else 1e9
        st = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=42),
                                 pruner=optuna.pruners.MedianPruner(n_startup_trials=8))
        st.optimize(obj, n_trials=N_TRIALS, show_progress_bar=False)
        return sorted([cand[i] for i in range(len(cand)) if st.best_params.get(f"u{i}", 0) == 1]) or [cand[0]]
    if method == "THRESHOLD":
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        hi = float(max(cs[c] for c in cand)) if cand else 1.0
        if not np.isfinite(hi) or hi <= 0:
            hi = 1.0
        def obj(t):
            tau = t.suggest_float("tau", 0.0, hi)
            idx = [c for c in cand if cs[c] >= tau]
            return _holdout(fac, Pp, yp, ylog, inv, idx) if idx else 1e9
        st = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        st.optimize(obj, n_trials=N_TRIALS, show_progress_bar=False)
        tau = st.best_params["tau"]; return sorted([c for c in cand if cs[c] >= tau]) or [cand[0]]
    return full


def worker_main(name):
    from simulation.tests._real_data_prep import _prep_full
    from simulation.models.registry import verify_registry_coverage
    verify_registry_coverage(force_import=True)
    from simulation.models.base import REGISTRY
    cls = REGISTRY.get(name)
    if cls is None:
        print("RESULT_JSON null", flush=True); return
    fac = (lambda c=cls: c())
    Pp, Pt, yp, yt, ylog_unused, inv_unused, cols = _prep_full()
    ylog = np.log1p(np.clip(yp, 0, None)); inv = lambda z: np.expm1(np.clip(z, -2, 20))
    # precompute per-fold training prefixes + de-circularized candidate pools (leakage-free).
    n = len(yp); nf = 3; fs = n // (nf + 1)
    folds = []
    for k in range(1, nf + 1):
        etr = k * fs; eva = (k + 1) * fs if k < nf else n
        if eva - etr < 4:
            continue
        cand_tr, cs_tr = _fold_cand(Pp[:etr], ylog[:etr])
        folds.append((etr, eva, cand_tr, cs_tr))
    # full-train-pool candidate pool for the held-out TEST refit (test rows untouched = no leakage).
    cand_full, cs_full = _fold_cand(Pp, ylog)
    out = {"name": name, "methods": {}}
    for mth in METHODS:
        try:
            oof_m, oof_sd = _oof3_nested(mth, fac, Pp, yp, ylog, inv, folds)
            idx_full = _select(mth, fac, Pp, yp, ylog, inv, cand_full, cs_full)  # select on full train (clean)
            out["methods"][mth] = {"k": len(idx_full), "oof": oof_m, "oof_sd": oof_sd,
                                   "test": _test(fac, Pp, Pt, yp, yt, ylog, inv, idx_full)}
        except Exception as e:
            out["methods"][mth] = {"k": -1, "oof": 1e9, "oof_sd": 0.0, "test": 1e9, "err": f"{type(e).__name__}"}
    print("RESULT_JSON " + json.dumps(out), flush=True)


def parent_main():
    print("=" * 110, flush=True)
    print("통합 feature 방법 비교: 13 방법 × 7 모델(family당 1) — OOF-WIS(↓) [고정 preproc=QT+log1p]", flush=True)
    print("=" * 110, flush=True)
    env = dict(os.environ, KMP_DUPLICATE_LIB_OK="TRUE", OMP_NUM_THREADS="1")
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    rows = {}
    for name in PANEL:
        try:
            cp = subprocess.run(
                [sys.executable, "-u", "-m", "simulation.scripts._unified_feature_method_comparison", "--model", name],
                cwd=repo, env=env, capture_output=True, text=True, timeout=PER_MODEL_TIMEOUT)
        except subprocess.TimeoutExpired:
            print(f"  {name} TIMEOUT", flush=True); continue
        if cp.returncode != 0:
            print(f"  {name} CRASH rc={cp.returncode}: {(cp.stderr or '')[-150:]}", flush=True); continue
        line = next((l for l in cp.stdout.splitlines() if l.startswith("RESULT_JSON ")), None)
        r = json.loads(line[len("RESULT_JSON "):]) if line and line != "RESULT_JSON null" else None
        if r:
            rows[name] = r["methods"]
            print(f"  ✓ {name} 완료", flush=True)
        else:
            print(f"  {name} no-result", flush=True)
    avail = [m for m in PANEL if m in rows]

    def _avg_rank(key):
        ar = {}
        for mth in METHODS:
            ranks = []
            for m in avail:
                ordered = sorted(METHODS, key=lambda x: rows[m].get(x, {}).get(key, 1e9))
                ranks.append(ordered.index(mth) + 1)
            ar[mth] = float(np.mean(ranks)) if ranks else 99.0
        return ar

    # OOF matrix (leakage-free, mean±sd) — best per model = *
    print("\n  === OOF-WIS matrix [LEAKAGE-FREE, fold당 재선택] (행=방법, 열=모델; mean±sd; lower=better; 각 모델 best=*) ===", flush=True)
    print("  " + f"{'method':12s}" + "".join(f"{m[:9]:>13s}" for m in avail), flush=True)
    best_oof = {m: min((rows[m].get(mth, {}).get("oof", 1e9) for mth in METHODS)) for m in avail}
    for mth in METHODS:
        cells = []
        for m in avail:
            d = rows[m].get(mth, {}); o = d.get("oof", 1e9); sd = d.get("oof_sd", 0.0)
            star = "*" if abs(o - best_oof[m]) < 1e-6 else " "
            cells.append(f"{o:>6.2f}±{sd:<4.2f}{star}" if o < 1e8 else f"{'—':>13s}")
        print(f"  {mth:12s}" + "".join(cells), flush=True)

    # TEST matrix (held-out, clean generalization signal)
    print("\n  === TEST-WIS matrix [held-out, 누수 0] (lower=better; 각 모델 best=*) ===", flush=True)
    print("  " + f"{'method':12s}" + "".join(f"{m[:9]:>13s}" for m in avail), flush=True)
    best_test = {m: min((rows[m].get(mth, {}).get("test", 1e9) for mth in METHODS)) for m in avail}
    for mth in METHODS:
        cells = []
        for m in avail:
            t = rows[m].get(mth, {}).get("test", 1e9)
            star = "*" if abs(t - best_test[m]) < 1e-6 else " "
            cells.append(f"{t:>11.3f}{star}" if t < 1e8 else f"{'—':>13s}")
        print(f"  {mth:12s}" + "".join(cells), flush=True)

    # dual ranking (OOF + TEST)
    ar_oof = _avg_rank("oof"); ar_test = _avg_rank("test")
    print("\n  === 방법 평균순위 (각 모델 순위 평균; 낮을수록 일관 우수) — OOF | TEST ===", flush=True)
    for mth in sorted(METHODS, key=lambda x: ar_test[x]):
        print(f"    {mth:12s} OOF_rank={ar_oof[mth]:.2f}   TEST_rank={ar_test[mth]:.2f}", flush=True)
    print(f"\n  → OOF 1위: {min(METHODS, key=lambda x: ar_oof[x])}   |   TEST 1위: {min(METHODS, key=lambda x: ar_test[x])}", flush=True)

    # 판별력(equivalence) note: 각 모델서 best 의 우위가 best 의 fold-sd 안인가? (차이가 noise 수준이면 동등)
    print("\n  === 판별력 note: 각 모델 'best 와 통계적 동등(차이 ≤ best의 fold-sd)' 방법 수 ===", flush=True)
    for m in avail:
        b = best_oof[m]
        b_sd = next((rows[m].get(mth, {}).get("oof_sd", 0.0) for mth in METHODS
                     if abs(rows[m].get(mth, {}).get("oof", 1e9) - b) < 1e-6), 0.0)
        n_equiv = sum(1 for mth in METHODS
                      if rows[m].get(mth, {}).get("oof", 1e9) < 1e8
                      and (rows[m].get(mth, {}).get("oof", 1e9) - b) <= max(b_sd, 1e-9))
        print(f"    {m:13s} best={b:.3f} (±{b_sd:.3f}) → {n_equiv}개 방법이 1-sd 안 (동등)", flush=True)
    print("  (동등 방법이 많을수록 '방법 차이가 noise' → parsimony 로 |corr| stability 선택 정당화)", flush=True)
    print("=" * 110, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", default=None); a = ap.parse_args()
    if a.model:
        worker_main(a.model)
    else:
        parent_main()


if __name__ == "__main__":
    main()
