#!/usr/bin/env python
"""External-series validation: is the TiRex+Tweedie interval advantage GENERAL or ILI-specific?

Downloads DIVERSE public time series spanning the heteroscedasticity spectrum, rolls TiRex 1-step
(zero-shot) on each, and compares the Tweedie residual-scale interval (q=mu+Qz*mu^(p/2), p on VAL,
expanding split-CQR) to the fair baseline (TiRex + empirical past-residual quantiles + expanding CQR),
leak-free. HYPOTHESIS: Tweedie helps HETEROSCEDASTIC (variance-scales-with-level) non-negative series
and is neutral/slightly-worse on HOMOSCEDASTIC ones. Reports per-series WIS delta% + DM p, and whether
the delta correlates with a heteroscedasticity index (corr of local std vs local mean).

Leak-free: tirex[t] uses y[:t]; every residual/std quantile at t uses weeks < t; p on VAL only; cap
train-only. Non-negativity: series shifted to min 0 (Tweedie assumes non-negative). No live/pipeline edits.
"""
from __future__ import annotations
import json, ssl, sys, time, urllib.request, io
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts._exp_crosscountry as X

CACHE = ROOT / "scripts" / "_external_ts_cache"
CACHE.mkdir(exist_ok=True)
CTX = ssl.create_default_context(); CTX.check_hostname = False; CTX.verify_mode = ssl.CERT_NONE
MAXLEN = 500          # cap long series (keep rolling-TiRex compute bounded)
MIN_CTX = 52

# (name, url, value_col, expected_regime) — jbrownlee/Datasets (reliable raw CSVs)
BASE = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
SERIES = [
    ("airline_passengers", BASE + "airline-passengers.csv", 1, "HETERO (multiplicative seasonal)"),
    ("monthly_car_sales",  BASE + "monthly-car-sales.csv", 1, "HETERO (trend+seasonal sales)"),
    ("monthly_robberies",  BASE + "monthly-robberies.csv", 1, "HETERO (rising count)"),
    ("writing_paper_sales", BASE + "monthly-writing-paper-sales.csv", 1, "HETERO (sales)"),
    ("female_births",      BASE + "daily-total-female-births.csv", 1, "COUNT (low-var)"),
    ("sunspots",           BASE + "monthly-sunspots.csv", 1, "HOMO-ish (cyclical, >=0)"),
    ("min_temperatures",   BASE + "daily-min-temperatures.csv", 1, "HOMO (seasonal temp, shifted>=0)"),
]
M3_IDS = ["M1005", "M1200", "M1400", "M1600", "M1800", "M2000"]   # diverse M3 Monthly economic series


def download(name, url, col):
    f = CACHE / f"{name}.csv"
    if not f.exists():
        with urllib.request.urlopen(url, timeout=30, context=CTX) as r:
            f.write_bytes(r.read())
    txt = f.read_text().strip().splitlines()
    vals = []
    for line in txt[1:]:
        parts = line.split(",")
        if len(parts) <= col:
            continue
        try:
            vals.append(float(parts[col].strip().strip('"')))
        except ValueError:
            continue
    y = np.array(vals, float)
    return y[-MAXLEN:] if len(y) > MAXLEN else y


def hetero_index(y):
    """Corr(local std, local mean) over rolling windows -> +1 = strong heteroscedastic (var scales w/ level)."""
    w = 12
    means, stds = [], []
    for i in range(0, len(y) - w, w // 2):
        seg = y[i:i + w]
        means.append(seg.mean()); stds.append(seg.std())
    means, stds = np.array(means), np.array(stds)
    if len(means) < 5 or means.std() == 0 or stds.std() == 0:
        return 0.0
    return float(np.corrcoef(means, stds)[0, 1])


def roll_tirex(model, y):
    import torch
    out = np.full(len(y), np.nan)
    with torch.no_grad():
        for t in range(MIN_CTX, len(y)):
            ctx = torch.tensor(y[max(0, t - 512):t], dtype=torch.float32).unsqueeze(0)
            _q, mean = model.forecast(context=ctx, prediction_length=1)
            out[t] = float(np.asarray(mean).ravel()[0])
    return out


def eval_series(name, y, model):
    y = np.asarray(y, float)
    y = y - min(0.0, float(y.min()))                      # shift to non-negative (Tweedie assumption)
    N = len(y)
    if N < MIN_CTX + 40:
        return None
    tirex = roll_tirex(model, y)
    n_test = min(150, (N - MIN_CTX) // 2)
    T0 = N - n_test
    cap = 2.0 * float(np.nanmax(y[:T0]))
    origins = np.arange(T0, N); n = len(origins); y_te = y[origins]
    val = np.arange(T0 - X.K_CAL, T0); y_val = y[val]
    # baseline
    bqy = X.baseline_qy(y, tirex, origins, cap); bB = X.expanding_cqr_bounds(bqy, y_te, cap)
    b_w = X.wis_of(bB, y_te, bqy[:, X.MED_COL])
    # Tweedie (p on VAL)
    vw = {}
    for p in X.P_GRID:
        vqy = X.tweedie_qy(y, tirex, val, p, cap); vB = X.expanding_cqr_bounds(vqy, y_val, cap)
        vw[p] = float(X.wis_of(vB, y_val, vqy[:, X.MED_COL]).mean())
    p_star = min(vw, key=vw.get)
    tqy = X.tweedie_qy(y, tirex, origins, p_star, cap); tB = X.expanding_cqr_bounds(tqy, y_te, cap)
    t_w = X.wis_of(tB, y_te, tqy[:, X.MED_COL])
    lo, hi = tB[0.05]; k = int(((y_te >= lo) & (y_te <= hi)).sum())
    dmp, dbar = X.dm(t_w, b_w)
    return {
        "name": name, "N_used": int(N), "n_test": int(n), "hetero_index": round(hetero_index(y), 3),
        "p_star": p_star, "baseline_wis": round(float(b_w.mean()), 4), "tweedie_wis": round(float(t_w.mean()), 4),
        "delta_pct": round(100 * (t_w.mean() - b_w.mean()) / b_w.mean(), 2), "dm_p": round(dmp, 4),
        "tweedie_picp95": round(k / n, 3), "tweedie_helps_sig": bool(t_w.mean() < b_w.mean() and dmp < 0.05),
    }


def main():
    t0 = time.time()
    from tirex import load_model
    model = load_model("NX-AI/TiRex", device="cpu")
    rows = []
    jobs = [(name, (lambda n=name, u=url, c=col: download(n, u, c)), regime)   # n=name: capture (late-binding fix)
            for name, url, col, regime in SERIES]
    # M3 Monthly economic series (diverse)
    try:
        import tempfile, warnings as _w; _w.filterwarnings("ignore")
        from datasetsforecast.m3 import M3
        m3df, _, _ = M3.load(directory=tempfile.gettempdir(), group="Monthly")
        for uid in M3_IDS:
            v = m3df[m3df.unique_id == uid]["y"].values.astype(float)
            jobs.append((f"M3_{uid}", (lambda vv=v: vv), "M3 Monthly economic"))
    except Exception as e:
        print(f"  [M3 load skipped: {type(e).__name__}]", flush=True)
    for name, getter, regime in jobs:
        try:
            y = getter()
            r = eval_series(name, y, model)
            if r:
                r["regime"] = regime; rows.append(r)
                sig = "*" if r["tweedie_helps_sig"] else " "
                print(f"  {name:20s} hetero={r['hetero_index']:+.2f} p*={r['p_star']} "
                      f"base={r['baseline_wis']:.4f} tweedie={r['tweedie_wis']:.4f} "
                      f"Δ={r['delta_pct']:+6.1f}%{sig} DMp={r['dm_p']:.3f} PICP95={r['tweedie_picp95']}  [{regime}]",
                      flush=True)
        except Exception as e:
            print(f"  {name}: FAIL {type(e).__name__} {str(e)[:80]}", flush=True)
    # correlation: does Tweedie's benefit scale with heteroscedasticity?
    if len(rows) >= 3:
        h = np.array([r["hetero_index"] for r in rows]); d = np.array([r["delta_pct"] for r in rows])
        corr = float(np.corrcoef(h, -d)[0, 1])   # -delta = benefit; corr>0 => more hetero -> more benefit
    else:
        corr = None
    out = {"series": rows, "corr_hetero_vs_benefit": round(corr, 3) if corr is not None else None,
           "elapsed_s": round(time.time() - t0, 0)}
    (ROOT / "scripts" / "_exp_external_ts.json").write_text(json.dumps(out, indent=2))
    print(f"\n=== SUMMARY ({time.time()-t0:.0f}s) ===")
    helped = [r["name"] for r in rows if r["tweedie_helps_sig"]]
    hurt = [r["name"] for r in rows if r["delta_pct"] > 0]
    print(f"  Tweedie helps (DM-sig): {helped or 'none'}")
    print(f"  Tweedie worse (Δ>0):    {hurt or 'none'}")
    print(f"  corr(heteroscedasticity, Tweedie benefit) = {corr}")
    print("  -> Tweedie advantage is", "HETERO-DRIVEN (general for level-dependent-variance series)"
          if corr and corr > 0.3 else "not clearly hetero-linked (small n / mixed)")


if __name__ == "__main__":
    raise SystemExit(main())
