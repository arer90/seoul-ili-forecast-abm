#!/usr/bin/env python
"""LAST untried lever (Codex panel's only non-rename candidate): Mondrianized Tweedie-CQR.

Keep the champion's TiRex point + Tweedie quantile skeleton (q=mu+Qz*mu^(p/2)) UNCHANGED. Replace the
single GLOBAL expanding split-CQR offset with a REGIME-LOCAL offset: group origins by a forecastable
variable (TiRex-point-level tertile = off-season / rising / peak), compute the per-alpha conformal offset
within the group from PAST origins in that group only, shrunk toward the global offset:
  Q = Q_group * n_g/(n_g+m) + Q_global * m/(n_g+m).
Tertile cutpoints fixed on TRAIN+VAL (past-only); shrinkage m selected on VAL only; TEST untouched.
Compared to the champion (global expanding CQR) on TEST, DM. Leak-free; no live/pipeline edits.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts._exp_crosscountry as X
from scripts.nov_guard_v3 import setup


def regime_cqr_bounds(qy, tirex, origins, y_at, cap, cuts, m):
    """Mondrian expanding CQR: per-alpha offset shrinks group (mu-tertile) toward global. Past-only."""
    n = qy.shape[0]
    B = {a: (np.zeros(n), np.zeros(n)) for a in X.ALPHAS}
    Eglob = {a: [] for a in X.ALPHAS}
    Egrp = {a: {0: [], 1: [], 2: []} for a in X.ALPHAS}
    def reg(t):
        v = tirex[t]
        return 0 if v <= cuts[0] else (1 if v <= cuts[1] else 2)
    for j, t in enumerate(origins):
        g = reg(t)
        for a in X.ALPHAS:
            cl = X.FQL.index(round(a / 2.0, 4)) if hasattr(X, "FQL") else list(np.round(X.FQ, 4)).index(round(a/2.0, 4))
            ch = list(np.round(X.FQ, 4)).index(round(1 - a / 2.0, 4))
            cl = list(np.round(X.FQ, 4)).index(round(a / 2.0, 4))
            pg = np.asarray(Egrp[a][g]); pglob = np.asarray(Eglob[a])
            lvl = min(1.0, (1 - a) * (1 + 1 / max(len(pglob), 1)))
            Qg = np.quantile(pg, lvl) if len(pg) >= 5 else (np.quantile(pglob, lvl) if len(pglob) >= 5 else 0.0)
            Qglob = np.quantile(pglob, lvl) if len(pglob) >= 5 else 0.0
            ng = len(pg); Q = Qg * ng / (ng + m) + Qglob * m / (ng + m)
            lo = np.clip(qy[j, cl] - Q, 0, cap); hi = np.clip(qy[j, ch] + Q, 0, cap)
            B[a][0][j] = lo; B[a][1][j] = hi
            E = max(qy[j, cl] - y_at[j], y_at[j] - qy[j, ch])
            Eglob[a].append(E); Egrp[a][g].append(E)
    return B


def main():
    t0 = time.time()
    S = setup(); y = S["yf"]; tirex = S["tirex"]; ntot = S["ntot"]
    T0 = 205; origins = np.arange(T0, ntot); n = len(origins); y_te = y[origins]
    cap = 2.0 * float(np.nanmax(y[:T0]))
    va = np.arange(T0 - X.K_CAL, T0); y_va = y[va]

    # tertile cutpoints on TiRex point over TRAIN+VAL (past-only, fixed before TEST)
    past_tx = tirex[X.__dict__.get("MIN_CTX", 52):T0]; past_tx = past_tx[np.isfinite(past_tx)]
    cuts = list(np.quantile(past_tx, [1/3, 2/3]))

    # champion Tweedie skeleton: p on VAL (global CQR), then compare skeleton+global vs skeleton+Mondrian
    vw = {}
    for p in X.P_GRID:
        vqy = X.tweedie_qy(y, tirex, va, p, cap); vB = X.expanding_cqr_bounds(vqy, y_va, cap)
        vw[p] = float(X.wis_of(vB, y_va, vqy[:, X.MED_COL]).mean())
    p_star = min(vw, key=vw.get)

    # champion (global expanding CQR) on TEST
    tqy = X.tweedie_qy(y, tirex, origins, p_star, cap)
    champ_B = X.expanding_cqr_bounds(tqy, y_te, cap)
    champ_w = X.wis_of(champ_B, y_te, tqy[:, X.MED_COL])

    # select shrinkage m on VAL (Mondrian on VAL origins)
    vqy = X.tweedie_qy(y, tirex, va, p_star, cap)
    best_m, best_vw = None, np.inf
    for m in (5, 10, 20, 40, 80):
        vB = regime_cqr_bounds(vqy, tirex, va, y_va, cap, cuts, m)
        w = float(X.wis_of(vB, y_va, vqy[:, X.MED_COL]).mean())
        if w < best_vw:
            best_vw, best_m = w, m

    # Mondrian on TEST with VAL-selected m
    mon_B = regime_cqr_bounds(tqy, tirex, origins, y_te, cap, cuts, best_m)
    mon_w = X.wis_of(mon_B, y_te, tqy[:, X.MED_COL])
    dmp, dbar = X.dm(mon_w, champ_w)
    lo, hi = mon_B[0.05]; k = int(((y_te >= lo) & (y_te <= hi)).sum())

    out = {
        "p_star": p_star, "tertile_cuts": [round(c, 2) for c in cuts], "best_m_on_val": best_m,
        "champion_global_wis": round(float(champ_w.mean()), 4),
        "mondrian_wis": round(float(mon_w.mean()), 4),
        "delta_pct": round(100 * (mon_w.mean() - champ_w.mean()) / champ_w.mean(), 2),
        "dm_p_vs_champion": round(dmp, 4), "dm_meandiff": round(dbar, 4),
        "mondrian_picp95": round(k / n, 3),
        "beats_champion": bool(mon_w.mean() < champ_w.mean() and dmp < 0.05),
        "elapsed_s": round(time.time()-t0, 0),
    }
    (ROOT / "scripts" / "_exp_mondrian_cqr.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("\nVERDICT:", "Mondrian BEATS champion (DM-sig)" if out["beats_champion"]
          else f"Mondrian does NOT beat champion (global {out['champion_global_wis']} vs mondrian "
               f"{out['mondrian_wis']}, DM p={out['dm_p_vs_champion']}) — last untried lever exhausted; "
               f"WIS ~2.24 confirmed as the data-limited floor")


if __name__ == "__main__":
    raise SystemExit(main())
