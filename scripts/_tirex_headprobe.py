#!/usr/bin/env python
"""TiRex-HeadProbe — decompose TiRex, FREEZE its 32M xLSTM backbone, extract the rich 512-dim internal
representation (out_norm output), and train a SMALL Seoul-specific calibration head on it (linear-probing /
head-retraining — the principled small-data transfer, unlike LoRA which retrained the backbone and overfit).

The head predicts a CORRECTION to TiRex's own native FluSight quantiles from a PCA-reduced (leak-free, TRAIN-fit)
projection of the 512-dim frozen feature. Anchored on native (correction init 0 = do-no-harm). Trained on
TRAIN[52,165), early-stopped/selected on VAL[165,205), evaluated on TEST[205,337) vs plain TiRex-native.
Leak-free: features use context y[:t]; PCA fit on TRAIN only; head trained on TRAIN only. DM(HLN)+bootstrap.
Caches the 512-dim reps. No live/pipeline edits.
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
MIN_CTX = 52; T0 = 205; TR_END = 165; SEED = 42
REP_CACHE = ROOT / "scripts" / "_tirex_reps.npz"; DEC_CACHE = ROOT / "scripts" / "_tirex_native_deciles.npz"


def extract():
    if REP_CACHE.exists() and DEC_CACHE.exists():
        return np.load(REP_CACHE)["rep"], np.load(DEC_CACHE)["dec"]
    import torch
    from tirex import load_model
    S = setup(); y = S["yf"]; N = S["ntot"]
    m = load_model("NX-AI/TiRex", device="cpu")
    cap = {}
    h = m.out_norm.register_forward_hook(lambda mod, i, o: cap.__setitem__("r", o.detach()))
    reps = np.full((N, 512), np.nan); dec = np.full((N, 9), np.nan)
    with torch.no_grad():
        for t in range(MIN_CTX, N):
            ctx = torch.tensor(y[max(0, t-512):t], dtype=torch.float32).unsqueeze(0)
            q, _ = m.forecast(context=ctx, prediction_length=1)
            reps[t] = cap["r"].reshape(-1, 512)[-1].numpy()
            dec[t] = np.sort(np.asarray(q).ravel())
    h.remove()
    np.savez(REP_CACHE, rep=reps);
    if not DEC_CACHE.exists(): np.savez(DEC_CACHE, dec=dec)
    return reps, dec


def flusight(dec_row, cap):
    q = np.interp(FQ, DEC, dec_row)
    ls = (dec_row[1]-dec_row[0])/(DEC[1]-DEC[0]); hs = (dec_row[-1]-dec_row[-2])/(DEC[-1]-DEC[-2])
    for i, a in enumerate(FQ):
        if a < 0.1: q[i] = dec_row[0] - ls*(0.1-a)
        elif a > 0.9: q[i] = dec_row[-1] + hs*(a-0.9)
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
    reps, dec = extract()
    print(f"[extract] reps {reps.shape}, {time.time()-t0:.0f}s")

    tr = np.arange(MIN_CTX, TR_END); va = np.arange(TR_END, T0); te = np.arange(T0, N)
    nat = {t: flusight(dec[t], cap) for t in range(MIN_CTX, N)}                 # native FluSight quantiles

    # PCA on TRAIN reps only (leak-free), keep K comps
    Rtr = reps[tr]; mu = Rtr.mean(0); Rc = Rtr - mu
    U, Sg, Vt = np.linalg.svd(Rc, full_matrices=False)
    def proj(idxs, K):
        return (reps[idxs] - mu) @ Vt[:K].T / (Sg[:K] + 1e-6)                   # whitened PCA scores

    def native_wis(idxs):
        return wis_arr(np.array([nat[t] for t in idxs]), y[idxs], cap)
    nat_te = native_wis(te); nat_va = native_wis(va)

    # head: PCA(K) -> small MLP -> per-quantile correction (added to native); init ~0 (do-no-harm)
    def train_head(K, hid, wd, dp, restarts=4):
        Xtr = torch.tensor(proj(tr, K), dtype=torch.float32); ytr = torch.tensor(y[tr], dtype=torch.float32)
        natr = torch.tensor(np.array([nat[t] for t in tr]), dtype=torch.float32)
        Xva = torch.tensor(proj(va, K), dtype=torch.float32); nava = torch.tensor(np.array([nat[t] for t in va]), dtype=torch.float32)
        fq = torch.tensor(FQ, dtype=torch.float32)
        best = (np.inf, None)
        for rs in range(restarts):
            torch.manual_seed(SEED+rs)
            net = nn.Sequential(nn.Linear(K, hid), nn.GELU(), nn.Dropout(dp), nn.Linear(hid, NQ))
            with torch.no_grad(): net[-1].weight.mul_(0.01); net[-1].bias.mul_(0)   # start near native
            opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=wd)
            bv, bad, bstate = np.inf, 0, None
            for ep in range(300):
                net.train(); opt.zero_grad()
                corr = net(Xtr); q = natr + corr; q, _ = torch.sort(q, 1)
                e = ytr.unsqueeze(1) - q; loss = torch.mean(torch.maximum(fq*e, (fq-1)*e))
                loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                net.eval()
                with torch.no_grad():
                    qv = torch.sort(nava + net(Xva), 1)[0].numpy()
                    vw = wis_arr(qv, y[va], cap).mean()
                if vw < bv-1e-4: bv, bad, bstate = vw, 0, {k: v.clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= 40: break
            if bv < best[0]: best = (bv, (net, bstate))
        return best

    # VAL-selected head hyperparameters
    grid = [(8, 8, 1e-2, 0.3), (16, 16, 1e-2, 0.3), (8, 16, 3e-2, 0.4), (16, 8, 3e-2, 0.4), (4, 8, 1e-2, 0.2)]
    best_cfg, best_val, best_net = None, np.inf, None
    for (K, hid, wd, dp) in grid:
        bv, (net, st) = train_head(K, hid, wd, dp)
        if bv < best_val: best_val, best_cfg, best_net = bv, (K, hid, wd, dp), (net, st, K)
    (net, st, K) = best_net; net.load_state_dict(st); net.eval()
    use_head = best_val < nat_va.mean()                                        # do-no-harm gate

    # TEST
    with torch.no_grad():
        qte = torch.sort(torch.tensor(np.array([nat[t] for t in te]), dtype=torch.float32)
                         + net(torch.tensor(proj(te, K), dtype=torch.float32)), 1)[0].numpy()
    head_te = wis_arr(qte, y[te], cap)
    dep_te = head_te if use_head else nat_te                                   # deployed
    hln, bp = dm_boot(dep_te, nat_te)
    lo = np.clip(np.sort(qte,1)[:,FQr.index(0.05)],0,cap); hi=np.clip(np.sort(qte,1)[:,FQr.index(0.95)],0,cap)
    picp = float(((y[te]>=lo)&(y[te]<=hi)).mean())

    out = {
        "backbone": "FROZEN 32.3M xLSTM (12 blocks); trained only a small head on the 512-dim out_norm rep",
        "best_head_cfg": {"K_pca": best_cfg[0], "hidden": best_cfg[1], "wd": best_cfg[2], "dropout": best_cfg[3]},
        "native_wis_test": round(float(nat_te.mean()), 4), "native_wis_val": round(float(nat_va.mean()), 4),
        "head_wis_val": round(float(best_val), 4), "head_wis_test": round(float(head_te.mean()), 4),
        "use_head_donoharm": bool(use_head), "deployed_wis_test": round(float(dep_te.mean()), 4),
        "deployed_picp95": round(picp, 3),
        "dm_hln_p_vs_native": round(hln, 4), "dm_boot_p_vs_native": round(bp, 4),
        "beats_native": bool(dep_te.mean() < nat_te.mean() and hln < 0.05 and bp < 0.05),
        "elapsed_s": round(time.time()-t0, 0),
    }
    (ROOT / "scripts" / "_tirex_headprobe.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    if out["beats_native"]:
        verdict = "HEAD-PROBE BEATS native TiRex (DM-sig, both HLN & bootstrap)"
    elif best_val < nat_va.mean():
        why = "do-no-harm reverted" if not use_head else ("TEST %.4f not DM-sig (p=%.3f)" % (out["head_wis_test"], hln))
        verdict = ("head IMPROVES VAL (%.4f vs native %.4f) but %s — frozen-backbone head-retraining also hits the n=113 floor"
                   % (out["head_wis_val"], out["native_wis_val"], why))
    else:
        verdict = "head cannot even beat native on VAL — 113 weeks insufficient to retrain the head"
    print("\nVERDICT:", verdict)


if __name__ == "__main__":
    raise SystemExit(main())
