"""sci_regime_calibration.py — regime-conditional PI calibration (coverage + PIT).

Goal (SCI ANALYSIS 3): show that the champion's raw 95% PI undercoverage is
*concentrated in the high-incidence / peak regime*, not a global failure, and
quantify where adaptive conformal restores nominal coverage.

Read-only. Reuses the SSOT split-conformal + adaptive-conformal logic from
`simulation.scripts.adaptive_pi_eval` / `simulation.analytics.adaptive_conformal`
(G-365) — NO retraining, NO model load. Inputs are the saved per-week test
predictions (predictions_<champion>.csv: test slab y_true/y_pred) plus the
leak-free in-sample residuals in per_model_optimal/<champion>.json.

Regime labels (each test week):
  (a) incidence tertile  : low / medium / high  (by ILI y_true tertiles)
  (b) wave phase         : pre-peak / peak / tail (relative to test-set peak;
                           peak = peak week ± a window; pre = before, tail = after)

Per regime we report empirical 95% PI coverage for:
  - raw split-conformal (static, in-sample residual half-width)
  - adaptive conformal (Conformal-PID, rolling past obs)

PIT (probability integral transform): per test point, the empirical CDF of the
predictive distribution evaluated at y. We build a piecewise-linear CDF from the
K=11 central PI bounds (median + symmetric quantiles). Uniformity of PIT under
calibration is summarized by a one-sample KS statistic vs U(0,1). Reported
globally and per regime. We also run a Christoffersen-style unconditional
coverage (Kupiec POF) LR test on the 95% PI hit sequence.

Outputs (under simulation/results/figures/ + csv/):
  - figures/regime_calibration_<champion>.png  (PIT hist + coverage-by-regime bars)
  - csv/regime_calibration_<champion>.csv       (regime x coverage table)

Usage: .venv/bin/python -m simulation.scripts.sci_regime_calibration [CHAMPION]
Returns: prints JSON summary + output paths. Side effects: writes png + csv.
"""
from __future__ import annotations

import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

RESULTS = "simulation/results"
CSV_DIR = f"{RESULTS}/csv"
FIG_DIR = f"{RESULTS}/figures"
DEFAULT_CHAMPION = "FusedEpi"


def _residuals(name: str):
    """Leak-free in-sample residuals from per_model_optimal/<name>.json (or None)."""
    f = f"{RESULTS}/per_model_optimal/{name}.json"
    if not os.path.exists(f):
        return None
    try:
        d = json.load(open(f))
        r = (d.get("val_metrics", {}) or {}).get("insample_residuals")
        if r is None:
            return None
        a = np.asarray(r, dtype=np.float64)
        a = a[np.isfinite(a)]
        return a if len(a) >= 2 else None
    except Exception:
        return None


def _load_test(name: str):
    """Return (y_true, y_pred) for the test slab of predictions_<name>.csv."""
    import pandas as pd
    pf = f"{CSV_DIR}/predictions_{name}.csv"
    df = pd.read_csv(pf)
    t = df[df["split"] == "test"]
    return (t["y_true"].values.astype(np.float64),
            t["y_pred"].values.astype(np.float64))


def _regime_labels(y: np.ndarray):
    """Return (incidence_label[], phase_label[]) for each test week.

    incidence: low/medium/high by y_true tertiles.
    phase: pre-peak / peak / tail relative to the test-set peak index.
      peak = within +/- `win` weeks of argmax; pre = earlier; tail = later.
    """
    n = len(y)
    q1, q2 = np.quantile(y, [1.0 / 3.0, 2.0 / 3.0])
    inc = np.where(y <= q1, "low", np.where(y <= q2, "medium", "high"))
    pk = int(np.argmax(y))
    win = max(2, int(round(0.08 * n)))  # peak window ~+/-2-3 wk on n~68
    phase = np.empty(n, dtype=object)
    for i in range(n):
        if abs(i - pk) <= win:
            phase[i] = "peak"
        elif i < pk:
            phase[i] = "pre-peak"
        else:
            phase[i] = "tail"
    return inc, phase, (float(q1), float(q2), pk, win)


def _pit_from_bounds(y: np.ndarray, bounds: dict, alphas, median: np.ndarray):
    """Probability integral transform via piecewise-linear predictive CDF.

    Build per-point CDF support from the K=11 central PI bounds: each alpha
    gives a (1-a) central interval => lower tail prob a/2 at lo, upper tail
    prob 1-a/2 at hi, plus 0.5 at the median. Interpolate CDF(y) and clip to
    [eps, 1-eps]. Returns (n,) PIT values in (0,1).
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    median = np.asarray(median, dtype=np.float64).ravel()
    n = len(y)
    ks = sorted([a for a in alphas if a in bounds])
    pit = np.full(n, 0.5)
    for i in range(n):
        xs = [median[i]]
        ps = [0.5]
        for a in ks:
            lo, hi = bounds[a]
            xs.append(float(lo[i])); ps.append(a / 2.0)
            xs.append(float(hi[i])); ps.append(1.0 - a / 2.0)
        xs = np.asarray(xs); ps = np.asarray(ps)
        order = np.argsort(xs)
        xs = xs[order]; ps = ps[order]
        # enforce monotone, de-dup x
        ux, idx = np.unique(xs, return_index=True)
        up = np.maximum.accumulate(ps[idx])
        if ux.size == 1:
            pit[i] = 0.5
            continue
        val = float(np.interp(y[i], ux, up, left=up[0], right=up[-1]))
        # tails beyond support: linear-ish saturation already via left/right clip
        pit[i] = float(np.clip(val, 1e-3, 1 - 1e-3))
    return pit


def _ks_uniform(pit: np.ndarray):
    """One-sample KS statistic + p-value of pit vs Uniform(0,1)."""
    from scipy import stats
    p = np.asarray(pit, dtype=np.float64)
    p = p[np.isfinite(p)]
    if len(p) < 3:
        return float("nan"), float("nan")
    res = stats.kstest(p, "uniform")
    return float(res.statistic), float(res.pvalue)


def _kupiec_pof(hits_in: np.ndarray, target_cov: float = 0.95):
    """Kupiec POF unconditional-coverage LR test on a 95% PI hit sequence.

    hits_in[i] = 1 if y[i] inside the 95% PI (covered). Tests whether the
    empirical coverage equals nominal `target_cov`. Returns (LR_stat, p_value,
    emp_cov). Christoffersen-style unconditional coverage component.
    """
    from scipy import stats
    h = np.asarray(hits_in, dtype=np.float64)
    h = h[np.isfinite(h)]
    n = len(h)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    x = int(h.sum())                       # number covered
    pi_hat = x / n                          # empirical coverage
    p = target_cov
    # LR_uc on the *failure* indicator is symmetric; use coverage directly.
    eps = 1e-12
    ll_null = x * np.log(p + eps) + (n - x) * np.log(1 - p + eps)
    ll_alt = x * np.log(pi_hat + eps) + (n - x) * np.log(1 - pi_hat + eps)
    lr = -2.0 * (ll_null - ll_alt)
    pval = float(stats.chi2.sf(lr, df=1))
    return float(lr), pval, float(pi_hat)


def compute(champion: str = DEFAULT_CHAMPION):
    """Run full regime-conditional calibration. Returns a result dict."""
    from simulation.analytics.hub_metrics import (
        FLUSIGHT_ALPHAS, k11_pi_widths_from_residuals,
    )
    from simulation.analytics.adaptive_conformal import adaptive_conformal_bounds

    y, pred = _load_test(champion)
    res = _residuals(champion)
    if res is None:
        raise RuntimeError(f"no leak-free residuals for {champion}")

    inc, phase, meta = _regime_labels(y)
    q1, q2, pk, win = meta

    # --- static split-conformal half-widths (K=11) ---
    k11 = k11_pi_widths_from_residuals(np.abs(res), FLUSIGHT_ALPHAS)
    q95 = k11.get(0.05)  # 95% PI half-width

    # static bounds per level (symmetric pred +/- q)
    static_bounds = {}
    for a in FLUSIGHT_ALPHAS:
        q = k11.get(float(a))
        if q is not None and np.isfinite(q):
            static_bounds[a] = (pred - q, pred + q)

    # --- adaptive conformal bounds (Conformal-PID, rolling) ---
    adapt_bounds = adaptive_conformal_bounds(pred, k11, res, y, FLUSIGHT_ALPHAS)

    # --- per-week 95% coverage indicators ---
    lo_s, hi_s = static_bounds[0.05]
    cov_static = ((y >= lo_s) & (y <= hi_s)).astype(float)
    lo_a, hi_a = adapt_bounds[0.05]
    cov_adapt = ((y >= lo_a) & (y <= hi_a)).astype(float)

    # --- coverage by regime ---
    def by(labels, vals):
        out = {}
        for lab in sorted(set(labels)):
            m = labels == lab
            out[lab] = {"n": int(m.sum()), "cov": round(float(vals[m].mean()), 3)}
        return out

    coverage = {
        "incidence": {
            "static": by(inc, cov_static),
            "adaptive": by(inc, cov_adapt),
        },
        "phase": {
            "static": by(phase, cov_static),
            "adaptive": by(phase, cov_adapt),
        },
        "overall": {
            "static": round(float(cov_static.mean()), 3),
            "adaptive": round(float(cov_adapt.mean()), 3),
        },
    }

    # --- PIT (static + adaptive) ---
    pit_static = _pit_from_bounds(y, static_bounds, FLUSIGHT_ALPHAS, pred)
    pit_adapt = _pit_from_bounds(y, adapt_bounds, FLUSIGHT_ALPHAS, pred)
    ks_s, ksp_s = _ks_uniform(pit_static)
    ks_a, ksp_a = _ks_uniform(pit_adapt)

    # --- Kupiec POF on 95% hit sequence ---
    lr_s, lrp_s, emp_s = _kupiec_pof(cov_static, 0.95)
    lr_a, lrp_a, emp_a = _kupiec_pof(cov_adapt, 0.95)

    return {
        "champion": champion, "n_test": int(len(y)),
        "tertiles": [round(q1, 2), round(q2, 2)], "peak_idx": pk, "peak_win": win,
        "y": y, "pred": pred, "inc": inc, "phase": phase,
        "q95_halfwidth": round(float(q95), 3) if q95 is not None else None,
        "coverage": coverage,
        "pit_static": pit_static, "pit_adapt": pit_adapt,
        "ks": {"static": {"stat": round(ks_s, 3), "p": round(ksp_s, 3)},
               "adaptive": {"stat": round(ks_a, 3), "p": round(ksp_a, 3)}},
        "kupiec_pof": {
            "static": {"lr": round(lr_s, 3), "p": round(lrp_s, 4), "emp_cov": round(emp_s, 3)},
            "adaptive": {"lr": round(lr_a, 3), "p": round(lrp_a, 4), "emp_cov": round(emp_a, 3)},
        },
    }


def make_figure(R: dict) -> str:
    """PIT histograms (static vs adaptive) + 95% coverage-by-regime grouped bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    champ = R["champion"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    # (0,0) PIT static
    nb = 10
    ax[0, 0].hist(R["pit_static"], bins=nb, range=(0, 1), color="#c0392b",
                  edgecolor="white", alpha=0.85)
    ax[0, 0].axhline(len(R["pit_static"]) / nb, ls="--", color="k", lw=1,
                     label="uniform (calibrated)")
    ax[0, 0].set_title(f"PIT — raw split-conformal\nKS={R['ks']['static']['stat']:.3f} "
                       f"(p={R['ks']['static']['p']:.3f})")
    ax[0, 0].set_xlabel("PIT"); ax[0, 0].set_ylabel("count"); ax[0, 0].legend(fontsize=8)

    # (0,1) PIT adaptive
    ax[0, 1].hist(R["pit_adapt"], bins=nb, range=(0, 1), color="#27ae60",
                  edgecolor="white", alpha=0.85)
    ax[0, 1].axhline(len(R["pit_adapt"]) / nb, ls="--", color="k", lw=1,
                     label="uniform (calibrated)")
    ax[0, 1].set_title(f"PIT — adaptive conformal\nKS={R['ks']['adaptive']['stat']:.3f} "
                       f"(p={R['ks']['adaptive']['p']:.3f})")
    ax[0, 1].set_xlabel("PIT"); ax[0, 1].set_ylabel("count"); ax[0, 1].legend(fontsize=8)

    # (1,0) coverage by incidence tertile
    def bar_panel(axx, key, order, title):
        st = R["coverage"][key]["static"]
        ad = R["coverage"][key]["adaptive"]
        labs = [o for o in order if o in st]
        x = np.arange(len(labs)); w = 0.38
        sv = [st[l]["cov"] for l in labs]
        av = [ad[l]["cov"] for l in labs]
        axx.bar(x - w / 2, sv, w, label="raw split-conformal", color="#c0392b")
        axx.bar(x + w / 2, av, w, label="adaptive conformal", color="#27ae60")
        axx.axhline(0.95, ls="--", color="k", lw=1, label="nominal 0.95")
        axx.set_xticks(x)
        axx.set_xticklabels([f"{l}\n(n={st[l]['n']})" for l in labs])
        axx.set_ylim(0, 1.05); axx.set_ylabel("95% PI coverage")
        axx.set_title(title); axx.legend(fontsize=8)
        for xi, v in zip(x - w / 2, sv):
            axx.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
        for xi, v in zip(x + w / 2, av):
            axx.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    bar_panel(ax[1, 0], "incidence", ["low", "medium", "high"],
              "95% PI coverage by incidence tertile")
    bar_panel(ax[1, 1], "phase", ["pre-peak", "peak", "tail"],
              "95% PI coverage by wave phase")

    fig.suptitle(f"Regime-conditional PI calibration — champion {champ} "
                 f"(test n={R['n_test']})", fontsize=13, y=1.00)
    fig.tight_layout()
    os.makedirs(FIG_DIR, exist_ok=True)
    out = f"{FIG_DIR}/regime_calibration_{champ}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def write_csv(R: dict) -> str:
    import csv as _csv
    champ = R["champion"]
    os.makedirs(CSV_DIR, exist_ok=True)
    out = f"{CSV_DIR}/regime_calibration_{champ}.csv"
    rows = []
    for key in ("incidence", "phase"):
        st = R["coverage"][key]["static"]
        ad = R["coverage"][key]["adaptive"]
        for lab in st:
            rows.append({
                "regime_type": key, "regime": lab, "n": st[lab]["n"],
                "cov95_static": st[lab]["cov"], "cov95_adaptive": ad[lab]["cov"],
            })
    rows.append({"regime_type": "overall", "regime": "all", "n": R["n_test"],
                 "cov95_static": R["coverage"]["overall"]["static"],
                 "cov95_adaptive": R["coverage"]["overall"]["adaptive"]})
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = _csv.DictWriter(fh, fieldnames=["regime_type", "regime", "n",
                                            "cov95_static", "cov95_adaptive"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return out


def main() -> int:
    champ = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CHAMPION
    R = compute(champ)
    fig = make_figure(R)
    csvp = write_csv(R)
    summary = {
        "champion": R["champion"], "n_test": R["n_test"],
        "tertiles": R["tertiles"], "peak_idx": R["peak_idx"],
        "coverage": R["coverage"], "ks": R["ks"], "kupiec_pof": R["kupiec_pof"],
        "figure": fig, "csv": csvp,
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
