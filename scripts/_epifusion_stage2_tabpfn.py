#!/usr/bin/env python
"""EpiFusion-XL Stage 2 — TabPFN in-context distributional head on the TiRex residual (the part
that carried the fusion). Clean 3-way split; TEST untouched. Completes the honest assembly
test: does adding a TabPFN in-context residual head beat the Tweedie head (2.2378) on Seoul TEST?

Point = zero-shot TiRex (Stage 1 showed LoRA overfits/reverts). Head = TabPFN predicting the
FLUSIGHT quantiles of r_t = y_t - TiRex_t from past features [lag residuals, TiRex level, seasonal],
refit per block on strictly-past pairs; q_y = TiRex + TabPFN residual quantiles; expanding split-CQR.
Compared to Tweedie head + fair baseline on TEST [205,337). Leak-free; no live/pipeline edits.
"""
from __future__ import annotations
import os, json, sys, time, warnings
os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")   # offline; no cloud-token flow
from pathlib import Path
import numpy as np
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts._exp_crosscountry as X
from scripts.nov_guard_v3 import setup


def features(y, tirex, idxs, L=6):
    """Past-only tabular features at each week t: lag residuals, TiRex level, seasonal, roll|r|."""
    r = y - tirex
    woy = (np.arange(len(y)) % 52) / 52.0
    rows = []
    for t in idxs:
        lagr = [r[t - k] if t - k >= 0 and np.isfinite(r[t - k]) else 0.0 for k in range(1, L + 1)]
        rollabs = np.mean(np.abs(r[max(0, t - 8):t][np.isfinite(r[max(0, t - 8):t])])) if t > 8 else 0.0
        rows.append(lagr + [tirex[t], np.sin(2*np.pi*woy[t]), np.cos(2*np.pi*woy[t]), rollabs])
    return np.asarray(rows, float)


def _make_tabpfn(n_est=8):
    """Offline TabPFN loader (cached public weights, no token flow) — matches campaign _exp_tabpfn."""
    from simulation.models.tabpfn_wrapper import _ensure_weights, _load_tabpfn_token
    _load_tabpfn_token(); ckpt = _ensure_weights()
    from tabpfn import TabPFNRegressor
    kw = dict(device="cpu", ignore_pretraining_limits=True, n_estimators=int(n_est), random_state=42)
    if ckpt is not None:
        kw["model_path"] = str(ckpt)
    return TabPFNRegressor(**kw)


def tabpfn_resid_qy(y, tirex, origins, cap, refit_k=8, min_ctx=52):
    """TabPFN in-context residual quantiles per block (past-only), q_y = TiRex + q_r."""
    r = y - tirex
    qlist = [float(q) for q in X.FQ]
    qy = np.zeros((len(origins), len(X.FQ)))
    reg = None; last_fit = -999
    for j, t in enumerate(origins):
        if t - last_fit >= refit_k or reg is None:
            tr = np.arange(min_ctx, t)
            tr = tr[np.isfinite(r[tr])]
            Xtr = features(y, tirex, tr); ytr = r[tr]
            reg = _make_tabpfn(8)
            reg.fit(Xtr, ytr); last_fit = t
        xt = features(y, tirex, [t])
        try:
            qr = reg.predict(xt, output_type="quantiles", quantiles=qlist)
            qr = np.array([np.ravel(a)[0] for a in qr], float)
        except Exception:
            qr = np.zeros(len(qlist))
        qy[j] = np.clip(tirex[t] + np.sort(qr), 0.0, cap)
    return qy


def main():
    t0 = time.time()
    S = setup(); y = S["yf"]; tirex = S["tirex"]; ntot = S["ntot"]
    T0 = 205; origins = np.arange(T0, ntot); n = len(origins); y_te = y[origins]
    cap = 2.0 * float(np.nanmax(y[:T0]))
    val = np.arange(T0 - X.K_CAL, T0); y_val = y[val]

    def eval_qy(qy):
        B = X.expanding_cqr_bounds(qy, y_te, cap); w = X.wis_of(B, y_te, qy[:, X.MED_COL])
        lo, hi = B[0.05]; k = int(((y_te >= lo) & (y_te <= hi)).sum())
        return w, k

    # references (same clean machinery)
    bqy = X.baseline_qy(y, tirex, origins, cap); b_w, b_k = eval_qy(bqy)
    # Tweedie (p on VAL)
    vw = {}
    for p in X.P_GRID:
        vqy = X.tweedie_qy(y, tirex, val, p, cap); vB = X.expanding_cqr_bounds(vqy, y_val, cap)
        vw[p] = float(X.wis_of(vB, y_val, vqy[:, X.MED_COL]).mean())
    p_star = min(vw, key=vw.get)
    tqy = X.tweedie_qy(y, tirex, origins, p_star, cap); t_w, t_k = eval_qy(tqy)
    # TabPFN head
    print("[stage2] fitting TabPFN in-context residual head...", flush=True)
    ptqy = tabpfn_resid_qy(y, tirex, origins, cap); pt_w, pt_k = eval_qy(ptqy)

    dm_vs_tw, _ = X.dm(pt_w, t_w); dm_vs_base, _ = X.dm(pt_w, b_w)
    out = {
        "fair_baseline_wis": round(float(b_w.mean()), 4), "baseline_picp95": round(b_k / n, 3),
        "tweedie_wis": round(float(t_w.mean()), 4), "tweedie_p": p_star, "tweedie_picp95": round(t_k / n, 3),
        "tabpfn_head_wis": round(float(pt_w.mean()), 4), "tabpfn_picp95": round(pt_k / n, 3),
        "tabpfn_dm_p_vs_tweedie": round(dm_vs_tw, 4), "tabpfn_dm_p_vs_baseline": round(dm_vs_base, 4),
        "tabpfn_beats_tweedie": bool(pt_w.mean() < t_w.mean() and dm_vs_tw < 0.05),
        "elapsed_s": round(time.time() - t0, 0),
    }
    (ROOT / "scripts" / "_epifusion_stage2.json").write_text(json.dumps(out, indent=2))
    print("\n=== STAGE 2 (TabPFN head) RESULT ===")
    print(json.dumps(out, indent=2))
    print("\nVERDICT:", "TabPFN head BEATS Tweedie -> add to assembly"
          if out["tabpfn_beats_tweedie"] else
          f"TabPFN head does NOT beat Tweedie (WIS {out['tabpfn_head_wis']} vs {out['tweedie_wis']}, "
          f"DM p={out['tabpfn_dm_p_vs_tweedie']}) -> Tweedie head remains the winning interval")


if __name__ == "__main__":
    raise SystemExit(main())
