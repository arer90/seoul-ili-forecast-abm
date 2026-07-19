"""Best SINGLE selection objective (사용자: "WIS가 best? R²나 다른 게 더 좋게 찾지 않나? 단일이
세밀, 복합은 둔감"). User + codex/gemini both reject composite → compare SINGLE objectives only.

For each model we enumerate (y-transform × x-scaler) preproc configs, score each by OOF
{wis, r2, mape, pi95} via the REAL `_oof_cv_metrics` (Q1 OOF residuals), then ask: if we SELECT
by objective O, does the chosen config also satisfy the OTHER criteria? A good single objective
picks configs that pass the 4-criteria gate (R²≥0.8 ∧ MAPE≤20 ∧ WIS≤6 ∧ PICP95≥0.9). The
hypothesis (codex/gemini): WIS-selection is balanced (WIS contains point-accuracy + calibration);
R²-selection maximizes fit but can miss PICP (miscalibrated). Data decides.

macOS: run PER-FILE with KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1.
"""
import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore")

_TRANSFORMS = ["identity", "log1p", "sqrt", "asinh"]
_SCALERS = ["none", "robust", "standard"]
_CRIT = dict(r2=0.80, mape=20.0, wis=6.0, pi95=0.90)


def _data(n=250, p=12, seed=5):
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(n, 1))
    X = np.hstack([g, g + 0.03 * rng.normal(size=(n, 1)), 2 * g,   # 3 collinear
                   rng.normal(size=(n, 9))])
    y = 15.0 + 3.0 * X[:, 0] + 2.0 * X[:, 3] - 1.5 * X[:, 4] + rng.gamma(1.0, 0.5, n)
    return X, y, [f"f{i}" for i in range(p)]


def _passes(c):
    return int(c["r2"] >= _CRIT["r2"] and c["mape"] <= _CRIT["mape"]
              and c["wis"] <= _CRIT["wis"] and c["pi95"] >= _CRIT["pi95"])


def _config_cells(factory, X, y, cols):
    from simulation.pipeline.per_model_optimize import _oof_cv_metrics
    cells = []
    for t in _TRANSFORMS:
        for s in _SCALERS:
            m = _oof_cv_metrics(factory, X, y, t, s, feature_cols=cols, n_folds=2)
            cell = {"t": t, "s": s, "wis": m["wis"], "r2": m["r2"],
                    "mape": m["mape"], "pi95": m["pi95_coverage"]}
            if all(np.isfinite([cell["wis"], cell["r2"], cell["mape"], cell["pi95"]])):
                cells.append(cell)
    return cells


def _pick(cells, obj):
    if obj == "wis":
        return min(cells, key=lambda c: c["wis"])
    if obj == "r2":
        return max(cells, key=lambda c: c["r2"])
    if obj == "mape":
        return min(cells, key=lambda c: c["mape"])
    raise ValueError(obj)


def test_single_objective_comparison(capsys):
    from simulation.models.registry import verify_registry_coverage
    verify_registry_coverage(force_import=True)
    from simulation.models.base import REGISTRY

    X, y, cols = _data()
    reps = ["ElasticNet", "BayesianRidge", "KRR", "XGBoost", "GLARMA"]
    objs = ["wis", "r2", "mape"]
    pass_count = {o: 0 for o in objs}
    n_models = 0
    lines = ["", "=" * 90,
             "단일 objective 비교 — 각 objective로 고른 config의 OOF {wis,r2,mape,pi95} + 4-criteria",
             "  (4-criteria: R²≥0.8 ∧ MAPE≤20 ∧ WIS≤6 ∧ PICP95≥0.9)",
             "=" * 90]
    for name in reps:
        cls = REGISTRY.get(name)
        if cls is None:
            continue
        cells = _config_cells((lambda c=cls: c()), X, y, cols)
        if len(cells) < 3:
            lines.append(f"  {name}: insufficient finite configs — skip")
            continue
        n_models += 1
        lines.append(f"  {name}:")
        for o in objs:
            p = _pick(cells, o)
            ok = _passes(p)
            pass_count[o] += ok
            lines.append(f"     by {o:4s} → ({p['t']:7s},{p['s']:8s})  "
                         f"wis={p['wis']:.3f} r2={p['r2']:.3f} mape={p['mape']:.2f} "
                         f"pi95={p['pi95']:.3f}  4crit={'PASS' if ok else 'fail'}")
    lines.append("-" * 90)
    lines.append(f"  {n_models} models | 4-criteria 통과 수: "
                 + "  ".join(f"{o}={pass_count[o]}" for o in objs))
    best = max(pass_count, key=pass_count.get) if n_models else None
    lines.append(f"  VERDICT: 단일 objective '{best}' 가 4-criteria 가장 많이 통과 "
                 f"→ best single selection metric")
    lines.append("=" * 90)
    print("\n".join(lines))

    assert n_models >= 3, f"need ≥3 models, got {n_models}"
    # reproducibility (TDD): same seed → identical cells
    c1 = _config_cells((lambda: REGISTRY.get("ElasticNet")()), X, y, cols)
    c2 = _config_cells((lambda: REGISTRY.get("ElasticNet")()), X, y, cols)
    assert [c["wis"] for c in c1] == [c["wis"] for c in c2], "non-reproducible"
