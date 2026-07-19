#!/usr/bin/env python
"""TiRex-HeadProbe with EXPANDING training (FAIR vs zero-shot TiRex's expanding context).

Fix for the earlier fixed-113-week handicap: at each refit block the Seoul head is retrained on ALL past
data [52, block_start) (153..284 weeks, up to 2.5x the fixed window), matching TiRex's expanding context.
Frozen 32M xLSTM backbone; small PCA head on the cached 512-dim out_norm rep; rolling early-stop val
(last 40 wk of the training window); do-no-harm per block; hyperparameters fixed a-priori (no per-block
selection leak). Eval TEST[205,337) vs plain TiRex-native, DM(HLN)+bootstrap. Leak-free. No live/pipeline edits.
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
MIN_CTX = 52; T0 = 205; REFIT_K = 1; VAL_TAIL = 40; SEED = 42
K_PCA, HID, WD, DP = 32, 48, 3e-3, 0.2  # scaled to larger fair-expanding data


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


def main():
    import torch, torch.nn as nn
    t0 = time.time()
    S = setup(); y = S["yf"]; N = S["ntot"]; cap = 2.0*float(np.nanmax(y[:T0]))
    reps = np.load(ROOT/"scripts"/"_tirex_reps.npz")["rep"]
    dec = np.load(ROOT/"scripts"/"_tirex_native_deciles.npz")["dec"]
    nat = {t: flusight(dec[t], cap) for t in range(MIN_CTX, N)}
    fq = torch.tensor(FQ, dtype=torch.float32)

    def train_block(train_end):
        tr = np.arange(MIN_CTX, train_end-VAL_TAIL); vl = np.arange(train_end-VAL_TAIL, train_end)
        mu = reps[tr].mean(0); U, Sg, Vt = np.linalg.svd(reps[tr]-mu, full_matrices=False)
        def proj(idxs): return (reps[idxs]-mu) @ Vt[:K_PCA].T/(Sg[:K_PCA]+1e-6)
        Xtr = torch.tensor(proj(tr), dtype=torch.float32); ytr = torch.tensor(y[tr], dtype=torch.float32)
        natr = torch.tensor(np.array([nat[t] for t in tr]), dtype=torch.float32)
        Xvl = torch.tensor(proj(vl), dtype=torch.float32); navl = np.array([nat[t] for t in vl])
        best = (np.inf, None)
        for rs in range(3):
            torch.manual_seed(SEED+rs)
            net = nn.Sequential(nn.Linear(K_PCA, HID), nn.GELU(), nn.Dropout(DP), nn.Linear(HID, NQ))
            with torch.no_grad(): net[-1].weight.mul_(0.01); net[-1].bias.mul_(0)
            opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=WD)
            bv, bad, bs = np.inf, 0, None
            for ep in range(300):
                net.train(); opt.zero_grad()
                q = torch.sort(natr + net(Xtr), 1)[0]; e = ytr.unsqueeze(1)-q
                loss = torch.mean(torch.maximum(fq*e, (fq-1)*e)); loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                net.eval()
                with torch.no_grad():
                    qv = torch.sort(torch.tensor(navl, dtype=torch.float32)+net(Xvl), 1)[0].numpy()
                vw = wis_arr(qv, y[vl], cap).mean()
                if vw < bv-1e-4: bv, bad, bs = vw, 0, {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 40: break
            if bv < best[0]: best = (bv, (net, bs))
        (net, bs) = best[1]; net.load_state_dict(bs); net.eval()
        vw_nat = wis_arr(navl, y[vl], cap).mean()
        return net, mu, Vt, Sg, best[0], vw_nat

    te = np.arange(T0, N); qte = np.zeros((len(te), NQ)); used = 0; train_sizes = []
    for bstart in range(T0, N, REFIT_K):
        bend = min(bstart+REFIT_K, N)
        net, mu, Vt, Sg, vw_head, vw_nat = train_block(bstart)
        train_sizes.append(bstart-MIN_CTX)
        oi = np.arange(bstart, bend)
        proj_o = (reps[oi]-mu) @ Vt[:K_PCA].T/(Sg[:K_PCA]+1e-6)
        import torch
        if vw_head < vw_nat:                                   # do-no-harm per block
            with torch.no_grad():
                q = torch.sort(torch.tensor(np.array([nat[t] for t in oi]), dtype=torch.float32)
                               + net(torch.tensor(proj_o, dtype=torch.float32)), 1)[0].numpy()
            used += len(oi)
        else:
            q = np.array([nat[t] for t in oi])
        qte[oi-T0] = q

    head_w = wis_arr(qte, y[te], cap)
    nat_w = wis_arr(np.array([nat[t] for t in te]), y[te], cap)
    hln, bp = dm_boot(head_w, nat_w)
    lo = np.clip(np.sort(qte,1)[:,FQr.index(0.05)],0,cap); hi=np.clip(np.sort(qte,1)[:,FQr.index(0.95)],0,cap)
    picp = float(((y[te]>=lo)&(y[te]<=hi)).mean())
    out = {
        "training": "EXPANDING [52,block_start) — %d..%d weeks (fixed run was 113)" % (min(train_sizes), max(train_sizes)),
        "refit_blocks": len(train_sizes), "head_used_on_origins": used, "of_total": len(te),
        "native_wis_test": round(float(nat_w.mean()), 4), "headprobe_wis_test": round(float(head_w.mean()), 4),
        "delta_pct": round(100*(head_w.mean()-nat_w.mean())/nat_w.mean(), 2), "headprobe_picp95": round(picp, 3),
        "dm_hln_p": round(hln, 4), "dm_boot_p": round(bp, 4),
        "beats_native": bool(head_w.mean() < nat_w.mean() and hln < 0.05 and bp < 0.05),
        "elapsed_s": round(time.time()-t0, 0),
    }
    (ROOT/"scripts"/"_tirex_headprobe_expanding.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print("\nVERDICT:", "EXPANDING head-probe BEATS native (DM-sig)!" if out["beats_native"]
          else "with expanding training (up to 284 wk), head-probe still does NOT DM-beat native (%.4f vs %.4f, p=%.3f)"
               % (out["headprobe_wis_test"], out["native_wis_test"], hln))


if __name__ == "__main__":
    raise SystemExit(main())
