#!/usr/bin/env python
"""External validation of the Tweedie distributional-head forecaster on US & Japan national ILI.

Mirrors the Seoul recipe EXACTLY, leak-free, on genuinely new data:
  * point   = TiRex 1-step rolling (zero-shot; context y[:t] only) — cached per country.
  * baseline= TiRex point + empirical PAST-residual FLUSIGHT quantiles (unconditional) + expanding CQR.
  * Tweedie = TiRex point + residual-scale quantiles q = mu + Qz*mu^(p/2), Qz = empirical past
              STANDARDIZED-residual quantiles (z = (y-mu)/mu^(p/2)); p re-selected per country by
              argmin pre-T0 validation WIS (NOT assumed 1.5); expanding split-CQR.
Leak-free: tirex[t] uses y<t; every residual/standardized quantile at week t uses weeks < t; expanding
CQR uses conformity of origins < t seeded pre-T0; p on pre-T0 val only; cap = 2*max(y_train) train-only.
DM (HLN h=1) per-origin WIS vs the baseline. Reports WIS, DM p, PICP95(+CP CI), peak coverage, last-N.
No live/pipeline edits. Canonical single-source series (US=delphi_national %ILI 1404wk; JP=japan_jihs 316wk).
"""
from __future__ import annotations
import json, sqlite3, sys, time
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS, FLUSIGHT_QUANTILES
from simulation.analytics.adaptive_conformal import wis_from_bounds

DB = ROOT / "simulation/data/db/epi_real_seoul.db"
FQ = np.asarray(FLUSIGHT_QUANTILES, float)
FQL = [round(float(q), 4) for q in FQ]
MED_COL = FQL.index(0.5)
ALPHAS = list(FLUSIGHT_ALPHAS)
MIN_CTX = 52          # 1-yr min context (same as Seoul)
K_CAL = 40            # calibration/seed tail
P_GRID = (1.1, 1.3, 1.5, 1.7, 1.9)
MAX_CTX = 512


def load_series(country, source):
    con = sqlite3.connect(DB)
    rows = con.execute("SELECT year,week_no,ili_rate FROM overseas_ili WHERE country=? AND source=? "
                       "AND ili_rate IS NOT NULL ORDER BY year,week_no", (country, source)).fetchall()
    con.close()
    y = np.array([r[2] for r in rows], float)
    return np.clip(y, 0.0, None)


def roll_tirex(y, cache):
    if cache.exists():
        d = np.load(cache)
        if len(d["tirex"]) == len(y):
            return d["tirex"]
    import torch
    from tirex import load_model
    model = load_model("NX-AI/TiRex", device="cpu")
    out = np.full(len(y), np.nan)
    with torch.no_grad():
        for t in range(MIN_CTX, len(y)):
            ctx = torch.tensor(y[max(0, t - MAX_CTX):t], dtype=torch.float32).unsqueeze(0)
            _q, mean = model.forecast(context=ctx, prediction_length=1)
            out[t] = float(np.asarray(mean).ravel()[0])
    np.savez(cache, tirex=out)
    return out


def baseline_qy(y, tirex, idxs, cap):
    """Unconditional: mu + empirical past-residual FLUSIGHT quantiles (expanding, past-only)."""
    r = y - tirex
    qy = np.zeros((len(idxs), len(FQ)))
    for k, t in enumerate(idxs):
        past = r[MIN_CTX:t]; past = past[np.isfinite(past)]
        off = np.quantile(past, FQ) if len(past) >= 5 else np.zeros(len(FQ))
        row = np.clip(tirex[t] + off, 0.0, cap); row.sort(); qy[k] = row
    return qy


def tweedie_qy(y, tirex, idxs, p, cap):
    """Residual-scale: mu + Qz*mu^(p/2), Qz = past standardized-residual quantiles (expanding)."""
    mu = np.clip(tirex, 1e-6, None)
    z = (y - tirex) / np.power(mu, p / 2.0)                 # standardized residual (past-only when sliced)
    qy = np.zeros((len(idxs), len(FQ)))
    for k, t in enumerate(idxs):
        past = z[MIN_CTX:t]; past = past[np.isfinite(past)]
        qz = np.quantile(past, FQ) if len(past) >= 5 else np.zeros(len(FQ))
        row = np.clip(tirex[t] + qz * (mu[t] ** (p / 2.0)), 0.0, cap); row.sort(); qy[k] = row
    return qy


def expanding_cqr_bounds(qy, y_at, cap):
    """Expanding split-CQR: per-alpha offset = past (1-a) quantile of conformity E (origins<j)."""
    n = qy.shape[0]
    B = {a: (np.zeros(n), np.zeros(n)) for a in ALPHAS}
    Ehist = {a: [] for a in ALPHAS}
    for j in range(n):
        for a in ALPHAS:
            cl = FQL.index(round(a / 2.0, 4)); ch = FQL.index(round(1 - a / 2.0, 4))
            past = np.asarray(Ehist[a])
            Q = np.quantile(past, min(1.0, (1 - a) * (1 + 1 / max(len(past), 1)))) if len(past) >= 5 else 0.0
            lo = np.clip(qy[j, cl] - Q, 0, cap); hi = np.clip(qy[j, ch] + Q, 0, cap)
            B[a][0][j] = lo; B[a][1][j] = hi
            Ehist[a].append(max(qy[j, cl] - y_at[j], y_at[j] - qy[j, ch]))
    return B


def wis_of(B, y, med):
    return np.asarray(wis_from_bounds(y, B, ALPHAS, median=med), dtype=float)


def dm(wa, wb):
    d = wa - wb; n = len(d); dbar = d.mean()
    v = np.var(d, ddof=1) / n
    if v <= 0: return 1.0, dbar
    st = dbar / np.sqrt(v) * np.sqrt((n + 1) / n)
    return float(2 * (1 - stats.t.cdf(abs(st), df=n - 1))), float(dbar)


def cp(k, nn, a=0.05):
    lo = 0.0 if k == 0 else stats.beta.ppf(a / 2, k, nn - k + 1)
    hi = 1.0 if k == nn else stats.beta.ppf(1 - a / 2, k + 1, nn - k)
    return round(float(lo), 3), round(float(hi), 3)


def run_country(country, source):
    y = load_series(country, source)
    N = len(y)
    tirex = roll_tirex(y, ROOT / "scripts" / f"_tirex_{country}.npz")
    # test window: last ~40% of usable origins after a pool; val = the K_CAL before T0
    usable = N - MIN_CTX
    n_test = min(300, max(100, usable // 2))
    T0 = N - n_test
    train_max = float(np.nanmax(y[:T0]))
    cap = 2.0 * train_max
    peak_thr = float(np.quantile(y, 0.90))                 # top-decile = "peak" (country-relative)
    origins = np.arange(T0, N)
    y_te = y[origins]
    cal = np.arange(T0 - K_CAL, T0)

    # baseline (per-origin WIS = DM reference)
    bqy = baseline_qy(y, tirex, origins, cap)
    bB = expanding_cqr_bounds(bqy, y_te, cap)
    b_wis = wis_of(bB, y_te, bqy[:, MED_COL])

    # p-selection on pre-T0 val [T0-K_CAL, T0)
    val = np.arange(T0 - K_CAL, T0); y_val = y[val]
    val_wis = {}
    for p in P_GRID:
        vqy = tweedie_qy(y, tirex, val, p, cap)
        vB = expanding_cqr_bounds(vqy, y_val, cap)
        val_wis[p] = float(wis_of(vB, y_val, vqy[:, MED_COL]).mean())
    p_star = min(val_wis, key=val_wis.get)

    tqy = tweedie_qy(y, tirex, origins, p_star, cap)
    tB = expanding_cqr_bounds(tqy, y_te, cap)
    t_wis = wis_of(tB, y_te, tqy[:, MED_COL])

    n = len(origins)
    lo95, hi95 = tB[0.05]; covv = (y_te >= lo95) & (y_te <= hi95); k = int(covv.sum())
    peak = y_te >= peak_thr
    p_dm, dbar = dm(t_wis, b_wis)
    last = np.zeros(n, bool); last[max(0, n - 34):] = True
    return {
        "country": country, "source": source, "N_weeks": int(N), "n_test_origins": int(n),
        "T0": int(T0), "peak_thr_p90": round(peak_thr, 2), "cap_train_only": round(cap, 2),
        "p_star": p_star, "val_wis_by_p": {str(p): round(v, 4) for p, v in val_wis.items()},
        "baseline_wis": round(float(b_wis.mean()), 4),
        "tweedie_wis": round(float(t_wis.mean()), 4),
        "delta_pct": round(100 * (t_wis.mean() - b_wis.mean()) / b_wis.mean(), 2),
        "dm_p": p_dm, "dm_meandiff": round(dbar, 4),
        "tweedie_picp95": round(k / n, 4), "k_of_n": f"{k}/{n}", "cp95ci": list(cp(k, n)),
        "peak_picp95": round(float(covv[peak].mean()), 3), "n_peak": int(peak.sum()),
        "last34_tweedie_wis": round(float(t_wis[last].mean()), 4),
        "last34_baseline_wis": round(float(b_wis[last].mean()), 4),
        "beats_baseline_sig": bool(t_wis.mean() < b_wis.mean() and p_dm < 0.05),
    }


def main():
    t0 = time.time()
    res = {}
    for country, source in [("US", "delphi_national"), ("JP", "japan_jihs")]:
        r = run_country(country, source)
        res[country] = r
        print(f"\n===== {country} ({source}) — {r['N_weeks']} wk, {r['n_test_origins']} test origins =====")
        print(f"  p* (pre-T0 val) = {r['p_star']}  (val WIS by p: {r['val_wis_by_p']})")
        print(f"  baseline WIS = {r['baseline_wis']}   Tweedie WIS = {r['tweedie_wis']}   Δ = {r['delta_pct']}%")
        print(f"  DM p vs baseline = {r['dm_p']:.3e}  (mean diff {r['dm_meandiff']})   -> beats_sig = {r['beats_baseline_sig']}")
        print(f"  Tweedie PICP95 = {r['tweedie_picp95']} ({r['k_of_n']}, CP {r['cp95ci']})   peak(top10%) PICP95 = {r['peak_picp95']} (n={r['n_peak']})")
        print(f"  last34 WIS: Tweedie {r['last34_tweedie_wis']} vs baseline {r['last34_baseline_wis']}")
    (ROOT / "scripts" / "_exp_crosscountry.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote scripts/_exp_crosscountry.json  ({time.time()-t0:.0f}s)")
    print("\n=== VERDICT ===")
    for c, r in res.items():
        v = "GENERALIZES (beats baseline, DM p<0.05)" if r["beats_baseline_sig"] else "does NOT reach significance"
        print(f"  {c}: {v} — Δ{r['delta_pct']}% WIS, DM p={r['dm_p']:.2e}, PICP95 {r['tweedie_picp95']}")


if __name__ == "__main__":
    raise SystemExit(main())
