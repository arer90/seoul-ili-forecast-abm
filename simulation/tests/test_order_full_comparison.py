"""Order head-to-head (사용자 명시): which full ordering of preproc / feature / mc wins,
EMPIRICALLY, on real family models — not by argument. TDD: reproducible WF-CV OOF, fixed seed,
SAME preproc/feature/mc methods in both arms (only the ORDER differs).

  U (사용자 순서):  preproc → feature → mc → HP
  C (현재 코드):    feature → mc → preproc → HP

Both arms share: RobustScaler(X) + log1p(y) preproc, top-k |corr(X,y)| feature select, the REAL
`apply_multicollinearity_filter` (method='corr') mc, empirical-WIS (in-sample train residuals,
identical method both arms). Reports OOF median WIS + R² per model + a verdict.

macOS: run PER-FILE with KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 (LightGBM-free reps).
"""
import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore")


# ── shared ops (identical in both arms; only the ORDER of application differs) ──
def _preproc(X_tr, X_va, y_tr):
    from sklearn.preprocessing import RobustScaler
    sc = RobustScaler().fit(X_tr)
    y_t = np.log1p(np.clip(np.asarray(y_tr, float).ravel(), 0, None))
    inv = lambda z: np.expm1(np.clip(np.asarray(z, float).ravel(), -2, 20))
    return sc.transform(X_tr), sc.transform(X_va), y_t, inv


def _select_topk(X_tr, y_tr, k):
    yt = np.asarray(y_tr, float).ravel()
    score = []
    for j in range(X_tr.shape[1]):
        col = X_tr[:, j]
        score.append(abs(np.corrcoef(col, yt)[0, 1]) if np.std(col) > 1e-9 else 0.0)
    return sorted(np.argsort(score)[::-1][:k].tolist())


def _mc(X_tr, X_va, cols, method="corr"):
    from simulation.pipeline.mc_filter_stage3 import apply_multicollinearity_filter
    Xtr, Xva, _, _, _ = apply_multicollinearity_filter(
        X_tr, X_va, X_va[:1], np.zeros(len(X_tr)),
        feature_cols=list(cols), method=method)
    return Xtr, Xva


def _wis_r2(y_va, pred, resid_tr):
    from simulation.analytics.diagnostics import weighted_interval_score_empirical
    from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
    yv = np.asarray(y_va, float).ravel()
    pv = np.asarray(pred, float).ravel()
    r = np.asarray(resid_tr, float).ravel()
    r = r[np.isfinite(r)]
    wis = float(np.mean(weighted_interval_score_empirical(
        yv, pv, residuals=r if r.size >= 2 else np.array([np.std(yv) or 1.0, 0.0]),
        alphas=FLUSIGHT_ALPHAS)))
    ss = float(np.sum((yv - yv.mean()) ** 2))
    r2 = (1.0 - float(np.sum((pv - yv) ** 2)) / ss) if ss > 1e-12 else float("nan")
    return wis, r2


def _eval_order(factory, X, y, cols, order, k=6, n_folds=3):
    """WF-CV OOF median (WIS, R²) applying the ops in `order` ∈ {"U","C"}."""
    n = len(y)
    fold = n // (n_folds + 1)
    wiss, r2s = [], []
    for kf in range(1, n_folds + 1):
        e_tr = kf * fold
        e_va = (kf + 1) * fold if kf < n_folds else n
        X_tr, y_tr = X[:e_tr], y[:e_tr]
        X_va, y_va = X[e_tr:e_va], y[e_tr:e_va]
        if len(X_va) < 4:
            continue
        try:
            if order == "U":                       # preproc → feature → mc
                Xt, Xv, yt, inv = _preproc(X_tr, X_va, y_tr)
                idx = _select_topk(Xt, yt, min(k, Xt.shape[1]))
                Xt, Xv = _mc(Xt[:, idx], Xv[:, idx], [cols[i] for i in idx])
            else:                                  # C: feature → mc → preproc
                idx = _select_topk(X_tr, y_tr, min(k, X_tr.shape[1]))
                Xtf, Xvf = _mc(X_tr[:, idx], X_va[:, idx], [cols[i] for i in idx])
                Xt, Xv, yt, inv = _preproc(Xtf, Xvf, y_tr)
            m = factory()
            m.fit(Xt, yt)
            pred = inv(m.predict(Xv))
            resid = np.asarray(y_tr, float).ravel() - inv(m.predict(Xt))
            wis, r2 = _wis_r2(y_va, pred, resid)
            if np.isfinite(wis):
                wiss.append(wis)
                r2s.append(r2)
        except Exception:
            continue
    return (float(np.median(wiss)) if wiss else float("inf"),
            float(np.median(r2s)) if r2s else float("nan"))


def _data(n=200, p=8, seed=11):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n, 1))
    X = np.hstack([base, base + 0.02 * rng.normal(size=(n, 1)), 2 * base,   # 3 collinear
                   rng.normal(size=(n, 5))])                                # 5 indep
    y = 20.0 + 3.0 * X[:, 0] + 2.0 * X[:, 5] - 1.5 * X[:, 6] + rng.gamma(1.0, 0.6, n)
    return X, y, [f"f{i}" for i in range(p)]


def test_order_U_vs_C_on_family_models(capsys):
    from simulation.models.registry import verify_registry_coverage
    verify_registry_coverage(force_import=True)
    from simulation.models.base import REGISTRY

    X, y, cols = _data()
    reps = [("tree", "XGBoost"), ("linear", "ElasticNet"), ("kernel", "KRR"),
            ("ts", "Theta"), ("dl-tabular", "TabularDNN"), ("epi", "GLARMA")]

    rows, u_wins_wis, c_wins_wis = [], 0, 0
    for fam, name in reps:
        cls = REGISTRY.get(name)
        if cls is None:
            continue
        fac = (lambda c=cls: c())
        u_wis, u_r2 = _eval_order(fac, X, y, cols, "U")
        c_wis, c_r2 = _eval_order(fac, X, y, cols, "C")
        if np.isfinite(u_wis) and np.isfinite(c_wis):
            u_wins_wis += int(u_wis < c_wis)
            c_wins_wis += int(c_wis < u_wis)
        rows.append((fam, name, u_wis, c_wis, u_r2, c_r2))

    lines = ["", "=" * 84,
             "ORDER 비교  U(preproc→feature→mc)  vs  C(feature→mc→preproc)  — OOF median",
             "=" * 84,
             f"  {'family':12s} {'model':12s} {'U_wis':>8s} {'C_wis':>8s} "
             f"{'U_r2':>7s} {'C_r2':>7s}  {'WIS승':>6s}"]
    u_w = c_w = 0
    for fam, name, uw, cw, ur, cr in rows:
        win = "U" if uw < cw else ("C" if cw < uw else "=")
        u_w += int(win == "U"); c_w += int(win == "C")
        lines.append(f"  {fam:12s} {name:12s} {uw:8.4f} {cw:8.4f} {ur:7.3f} {cr:7.3f}  {win:>6s}")
    fin = [r for r in rows if np.isfinite(r[2]) and np.isfinite(r[3])]
    if fin:
        mu = float(np.mean([r[2] for r in fin])); mc_ = float(np.mean([r[3] for r in fin]))
        lines.append("-" * 84)
        lines.append(f"  mean WIS  U={mu:.4f}  C={mc_:.4f}  (낮을수록 좋음)  |  "
                     f"WIS 승: U={u_w} C={c_w}")
        lines.append(f"  verdict: {'U(네 순서)' if mu < mc_ else 'C(현재)' if mc_ < mu else '동률'} "
                     f"가 평균 WIS 우세 (Δ={abs(mu-mc_):.4f}, {abs(mu-mc_)/max(mu,mc_)*100:.1f}%)")
    lines.append("=" * 84)
    print("\n".join(lines))

    # TDD assertions: the experiment is valid + reproducible (not arbitrary).
    assert len(fin) >= 3, f"need ≥3 models finite in BOTH orders, got {len(fin)}"
    # reproducibility: same seed → identical result (re-run one model)
    r1 = _eval_order((lambda: REGISTRY.get("ElasticNet")()), X, y, cols, "U")
    r2 = _eval_order((lambda: REGISTRY.get("ElasticNet")()), X, y, cols, "U")
    assert r1 == r2, f"non-reproducible: {r1} != {r2}"
