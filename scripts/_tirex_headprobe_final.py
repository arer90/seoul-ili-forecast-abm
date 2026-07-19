#!/usr/bin/env python
"""TiRex-HeadProbe FINAL — leak-free & rigorous. Frozen 32M xLSTM backbone + a small Seoul head on the
cached 512-dim out_norm rep, EXPANDING per-origin training (fair vs TiRex's expanding context). The head
HYPERPARAMETERS (PCA dim, hidden, weight-decay, dropout) are selected ON VAL origins [165,205) ONLY —
NOT by peeking at TEST (that earlier test-peek is corrected here). The VAL-selected config is then run on
TEST[205,337), reported with multi-seed DM(HLN)+bootstrap vs plain TiRex-native. Leak-free throughout.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import scripts._exp_crosscountry as X
from scripts.nov_guard_v3 import setup
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
from simulation.analytics.adaptive_conformal import wis_from_bounds

FQ = np.asarray(X.FQ, float); FQr = list(np.round(FQ, 4)); MEDI = FQr.index(0.5); NQ = len(FQ)
DEC = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
MIN_CTX = 52; VAL0 = 165; T0 = 205; VAL_TAIL = 40
CONFIGS = [(16,16,1e-2,0.3), (24,24,5e-3,0.25), (32,48,3e-3,0.2), (32,32,3e-3,0.2), (8,16,1e-2,0.3)]


def flusight(dec_row, cap):
    q = np.interp(FQ, DEC, dec_row)
    ls = (dec_row[1]-dec_row[0])/(DEC[1]-DEC[0]); hs = (dec_row[-1]-dec_row[-2])/(DEC[-1]-DEC[-2])
    for i, a in enumerate(FQ):
        if a < 0.1: q[i] = dec_row[0]-ls*(0.1-a)
        elif a > 0.9: q[i] = dec_row[-1]+hs*(a-0.9)
    return np.clip(np.sort(q), 0.0, cap)


def wis_arr(qmat, y, cap):
    qmat = np.clip(np.sort(qmat, 1), 0, cap); B = {}
    for a in FLUSIGHT_ALPHAS:
        B[a] = (qmat[:, FQr.index(round(a/2,4))], qmat[:, FQr.index(round(1-a/2,4))])
    return np.asarray(wis_from_bounds(y, B, list(FLUSIGHT_ALPHAS), median=qmat[:, MEDI]), float)


def dm_boot(wa, wb, L=6, nb=10000):
    d = wa-wb; n = len(d); v = np.var(d, ddof=1)/n
    hln = 2*(1-stats.t.cdf(abs(d.mean()/np.sqrt(v))*np.sqrt((n+1)/n), df=n-1)) if v > 0 else 1.0
    rng = np.random.RandomState(0); k = n//L; c = 0
    for _ in range(nb):
        idx = (rng.randint(0, n-L+1, size=k)[:, None]+np.arange(L)).ravel()[:n]
        if d[idx].mean() >= 0: c += 1
    return float(hln), float(c/nb)


def run_origins(reps, nat, y, cap, origins, cfg, seed):
    """Expanding per-origin head-probe over `origins`; returns deployed quantile matrix (do-no-harm)."""
    import torch, torch.nn as nn
    K, HID, WD, DP = cfg; fq = torch.tensor(FQ, dtype=torch.float32)
    Q = np.zeros((len(origins), NQ))
    for j, t in enumerate(origins):
        tr = np.arange(MIN_CTX, t-VAL_TAIL); vl = np.arange(t-VAL_TAIL, t)
        mu = reps[tr].mean(0); U, Sg, Vt = np.linalg.svd(reps[tr]-mu, full_matrices=False)
        pf = lambda idx: (reps[idx]-mu) @ Vt[:K].T/(Sg[:K]+1e-6)
        Xtr = torch.tensor(pf(tr), dtype=torch.float32); ytr = torch.tensor(y[tr], dtype=torch.float32)
        natr = torch.tensor(np.array([nat[s] for s in tr]), dtype=torch.float32)
        navl = np.array([nat[s] for s in vl]); Xvl = torch.tensor(pf(vl), dtype=torch.float32)
        best = (np.inf, None)
        for rs in range(2):
            torch.manual_seed(seed+rs)
            net = nn.Sequential(nn.Linear(K, HID), nn.GELU(), nn.Dropout(DP), nn.Linear(HID, NQ))
            with torch.no_grad(): net[-1].weight.mul_(0.01); net[-1].bias.mul_(0)
            opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=WD)
            bv, bad, bs = np.inf, 0, None
            for ep in range(250):
                net.train(); opt.zero_grad()
                q = torch.sort(natr+net(Xtr), 1)[0]; e = ytr.unsqueeze(1)-q
                (torch.mean(torch.maximum(fq*e, (fq-1)*e))).backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); net.eval()
                with torch.no_grad():
                    vw = wis_arr(torch.sort(torch.tensor(navl, dtype=torch.float32)+net(Xvl),1)[0].numpy(), y[vl], cap).mean()
                if vw < bv-1e-4: bv, bad, bs = vw, 0, {k_: v.clone() for k_, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 30: break
            if bv < best[0]: best = (bv, (net, bs, bv))
        net, bs, vw_head = best[1]; net.load_state_dict(bs); net.eval()
        vw_nat = wis_arr(navl, y[vl], cap).mean()
        po = (reps[t:t+1]-mu) @ Vt[:K].T/(Sg[:K]+1e-6)
        if vw_head < vw_nat:                                   # do-no-harm
            with torch.no_grad():
                Q[j] = torch.sort(torch.tensor(nat[t], dtype=torch.float32)+net(torch.tensor(po, dtype=torch.float32)),1)[0].numpy()
        else:
            Q[j] = nat[t]
    return Q


def main():
    t0 = time.time()
    S = setup(); y = S["yf"]; N = S["ntot"]; cap = 2.0*float(np.nanmax(y[:T0]))
    reps = np.load(ROOT/"scripts"/"_tirex_reps.npz")["rep"]; dec = np.load(ROOT/"scripts"/"_tirex_native_deciles.npz")["dec"]
    nat = {t: flusight(dec[t], cap) for t in range(MIN_CTX, N)}
    val_o = np.arange(VAL0, T0); te = np.arange(T0, N)
    nat_val = wis_arr(np.array([nat[t] for t in val_o]), y[val_o], cap)
    nat_te = wis_arr(np.array([nat[t] for t in te]), y[te], cap)

    # ---- SELECT config on VAL origins [165,205) ONLY (leak-free; TEST untouched) ----
    val_scores = {}
    for cfg in CONFIGS:
        Qv = run_origins(reps, nat, y, cap, val_o, cfg, seed=42)
        val_scores[cfg] = float(wis_arr(Qv, y[val_o], cap).mean())
    best_cfg = min(val_scores, key=val_scores.get)
    print(f"[VAL selection] native VAL WIS={nat_val.mean():.4f}")
    for cfg, w in sorted(val_scores.items(), key=lambda x: x[1]):
        print(f"   {cfg}: VAL WIS {w:.4f}  {'<- selected' if cfg==best_cfg else ''}")

    # ---- run VAL-selected config on TEST, multi-seed ----
    seeds = [42, 7, 100, 2024, 5]; te_wis = []; picps = []
    for s in seeds:
        Qt = run_origins(reps, nat, y, cap, te, best_cfg, seed=s)
        w = wis_arr(Qt, y[te], cap); te_wis.append(w)
        lo = np.clip(np.sort(Qt,1)[:,FQr.index(0.05)],0,cap); hi=np.clip(np.sort(Qt,1)[:,FQr.index(0.95)],0,cap)
        picps.append(float(((y[te]>=lo)&(y[te]<=hi)).mean()))
    W = np.mean(te_wis, 0)                                       # seed-averaged per-origin WIS
    hln, bp = dm_boot(W, nat_te)
    per_seed = [(round(float(w.mean()),4), *[round(p,4) for p in dm_boot(w, nat_te)]) for w in te_wis]
    out = {
        "protocol": "head hyperparams selected on VAL[165,205) ONLY; expanding per-origin training; TEST[205,337) untouched",
        "selected_cfg": {"K_pca": best_cfg[0], "hidden": best_cfg[1], "wd": best_cfg[2], "dropout": best_cfg[3]},
        "val_wis_native": round(float(nat_val.mean()),4), "val_wis_selected": round(val_scores[best_cfg],4),
        "native_wis_test": round(float(nat_te.mean()),4), "headprobe_wis_test_seedavg": round(float(W.mean()),4),
        "delta_pct": round(100*(W.mean()-nat_te.mean())/nat_te.mean(),2), "picp95_seedavg": round(float(np.mean(picps)),3),
        "dm_hln_p_seedavg": round(hln,4), "dm_boot_p_seedavg": round(bp,4),
        "per_seed_[wis,hln,boot]": per_seed,
        "beats_native_robust": bool(W.mean()<nat_te.mean() and hln<0.05 and bp<0.05 and all(ps[1]<0.05 and ps[2]<0.05 for ps in per_seed)),
        "elapsed_s": round(time.time()-t0,0),
    }
    (ROOT/"scripts"/"_tirex_headprobe_final.json").write_text(json.dumps(out, indent=2))
    print("\n"+json.dumps(out, indent=2))
    print("\nVERDICT:", "ROBUST leak-free win over TiRex-native (VAL-selected cfg, all seeds DM-sig)"
          if out["beats_native_robust"] else
          f"VAL-selected cfg on TEST: {out['headprobe_wis_test_seedavg']} vs native {out['native_wis_test']} "
          f"(Δ{out['delta_pct']}%, HLN p={hln:.3f}, boot p={bp:.3f}) — "
          + ("seed-avg DM-sig but not every seed" if (hln<0.05 and bp<0.05) else "not robustly DM-sig"))


if __name__ == "__main__":
    raise SystemExit(main())
