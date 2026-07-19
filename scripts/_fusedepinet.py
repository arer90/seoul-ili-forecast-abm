#!/usr/bin/env python
"""FusedEpiNet — a GENUINE end-to-end fused FORECASTER (Korea/Seoul-specialized). Fuses ONLY the
successful FORECASTER internals into ONE network (NO SEIR-V-D / mechanistic model — that is the ABM's
separate, later concern; this is the champion-forecaster fusion). Grounded in TweedieGP/NegBin
distributional-head SOTA + DeepAR-style recurrent probabilistic forecasting. ONE network fuses:
  * recurrent backbone  : GRU over the recent y-context (temporal dynamics; the xLSTM/DeepAR idea)
  * foundation anchor    : TiRex 1-step point as an input (the successful foundation prior)
  * aux branch           : generic TS features (log-growth, acceleration, seasonal) — NO epidemiology
  * gated fusion         : learned gate combining the streams (lightweight cross-gating)
  * Tweedie distributional head : q(tau) = mu + z(tau)*mu^(p/2), z(tau) monotone learned, p learned in [1,2]
  * loss                 : pinball / WIS across the 23 FluSight quantiles (directly optimizes the metric)

This is FUSION (one net, end-to-end), not a pipeline of separate models. Small + heavily regularized for
the 113-week Seoul TRAIN. Clean 3-way split TRAIN[52,165)/VAL[165,205)/TEST[205,337); early-stop on VAL;
do-no-harm vs the TiRex+Tweedie champion. TEST untouched until final. Leak-free. No live/pipeline edits.
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

FQ = np.asarray(X.FQ, float); NQ = len(FQ); MEDI = list(np.round(FQ, 4)).index(0.5)
L = 16                 # recurrent context window
MIN_CTX = 52
SEED = 42


def aux_features(y, tirex):
    """Past-only GENERIC time-series features (NO epidemiology/SEIR): log-growth, acceleration,
    short/long momentum. 1-lag shifted so row t uses info up to t-1 only. Leak-free."""
    n = len(y)
    g1 = np.zeros(n); g1[1:] = np.log1p(y[1:]) - np.log1p(y[:-1])           # log-growth
    g2 = np.zeros(n); g2[2:] = g1[2:] - g1[1:-1]                            # acceleration
    m4 = np.zeros(n)
    for t in range(4, n):
        m4[t] = np.log1p(y[t]) - np.log1p(np.mean(y[t-4:t]) + 1e-9)         # level vs trailing mean
    lvl = np.zeros(n); lvl[1:] = y[:-1]                                     # last observed level
    M = np.column_stack([g1, g2, m4, lvl])                                 # (n,4) generic
    return np.vstack([M[:1], M[:-1]])                                      # 1-lag shift


def make_samples(y, tirex, mech, idxs):
    """For each target week t in idxs: (y-context[t-L:t] normalized, anchor/mech/calendar scalars, y[t])."""
    woy = (np.arange(len(y)) % 52) / 52.0
    ctx, sca, tgt = [], [], []
    for t in idxs:
        if t < MIN_CTX or t - L < 0 or not np.isfinite(tirex[t]):
            continue
        c = y[t - L:t].astype(np.float32)
        scale = max(float(np.std(c)), 1.0)
        ctx.append((c - c.mean()) / scale)
        sca.append([tirex[t], mech[t, 0], mech[t, 1], mech[t, 2], mech[t, 3],
                    np.sin(2*np.pi*woy[t]), np.cos(2*np.pi*woy[t]), scale, c[-1]])
        tgt.append(y[t])
    return (np.asarray(ctx, np.float32), np.asarray(sca, np.float32), np.asarray(tgt, np.float32))


def build_model(n_sca):
    import torch, torch.nn as nn
    class FusedEpiNet(nn.Module):
        def __init__(self, hid=24, mhid=12):
            super().__init__()
            self.gru = nn.GRU(1, hid, batch_first=True)
            self.mech = nn.Sequential(nn.Linear(n_sca, mhid), nn.SiLU(), nn.Dropout(0.2))
            self.gate = nn.Sequential(nn.Linear(hid + mhid, hid + mhid), nn.Sigmoid())
            self.trunk = nn.Sequential(nn.Linear(hid + mhid, 24), nn.SiLU(), nn.Dropout(0.2))
            self.delta = nn.Linear(24, 1)                       # point correction on TiRex anchor
            self.zraw = nn.Linear(24, NQ)                       # per-quantile offsets (pre-monotone)
            self.praw = nn.Parameter(torch.tensor(0.0))         # learned Tweedie power (->[1,2])
        def forward(self, ctx, sca):
            import torch
            h, _ = self.gru(ctx.unsqueeze(-1)); h = h[:, -1, :]
            m = self.mech(sca)
            z = torch.cat([h, m], -1); z = z * self.gate(z)
            u = self.trunk(z)
            anchor = sca[:, 0]                                  # TiRex point
            mu = torch.clamp(anchor + self.delta(u).squeeze(-1), min=0.01)
            p = 1.0 + torch.sigmoid(self.praw)                 # power in (1,2)
            zo = self.zraw(u)                                  # (B,NQ)
            zc = zo - zo[:, MEDI:MEDI+1]                        # center at median
            off = torch.cumsum(torch.nn.functional.softplus(torch.diff(
                zc, prepend=zc[:, :1]*0)), dim=1)              # monotone increasing
            off = off - off[:, MEDI:MEDI+1]                    # median offset = 0
            q = mu.unsqueeze(1) + off * mu.unsqueeze(1).clamp(min=1e-3) ** (p / 2.0)
            return torch.clamp(q, min=0.0), mu
    torch.manual_seed(SEED)
    return FusedEpiNet()


def pinball(q, y, fq):
    import torch
    y = y.unsqueeze(1)
    e = y - q
    return torch.mean(torch.maximum(fq * e, (fq - 1.0) * e))


def wis_np(q_mat, y, cap):
    """WIS from a (n,NQ) quantile matrix via the FluSight interval decomposition."""
    from simulation.analytics.adaptive_conformal import wis_from_bounds
    from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
    q_mat = np.clip(np.sort(q_mat, axis=1), 0, cap)
    B = {}
    for a in FLUSIGHT_ALPHAS:
        cl = list(np.round(FQ, 4)).index(round(a/2.0, 4)); ch = list(np.round(FQ, 4)).index(round(1-a/2.0, 4))
        B[a] = (q_mat[:, cl], q_mat[:, ch])
    med = q_mat[:, MEDI]
    return np.asarray(wis_from_bounds(y, B, list(FLUSIGHT_ALPHAS), median=med), float)


def main():
    import torch
    t0 = time.time()
    S = setup(); y = S["yf"]; tirex = S["tirex"]; ntot = S["ntot"]
    T0 = 205; TR_END = 165
    aux = aux_features(y, tirex)
    cap = 2.0 * float(np.nanmax(y[:T0]))
    fq_t = torch.tensor(FQ, dtype=torch.float32)

    tr_idx = np.arange(MIN_CTX, TR_END)
    va_idx = np.arange(TR_END, T0)
    te_idx = np.arange(T0, ntot)
    Xtr = make_samples(y, tirex, aux, tr_idx)
    Xva = make_samples(y, tirex, aux, va_idx)
    Xte = make_samples(y, tirex, aux, te_idx)
    print(f"[split] TRAIN {len(Xtr[2])} VAL {len(Xva[2])} TEST {len(Xte[2])} samples (Seoul-only)")

    ctr, str_, ytr = [torch.tensor(a) for a in Xtr]
    cva, sva, yva = [torch.tensor(a) for a in Xva]
    cte, ste, yte = [torch.tensor(a) for a in Xte]

    best = None; best_val = np.inf; best_state = None
    for restart in range(3):                                  # multi-restart (tiny data)
        model = build_model(str_.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-3)
        patience, bad, bv = 40, 0, np.inf
        for ep in range(400):
            model.train()
            perm = torch.randperm(len(ytr))
            for i in range(0, len(ytr), 32):
                b = perm[i:i+32]
                q, _ = model(ctr[b], str_[b])
                loss = pinball(q, ytr[b], fq_t)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            model.eval()
            with torch.no_grad():
                qv, _ = model(cva, sva)
                vw = float(wis_np(qv.numpy(), yva.numpy(), cap).mean())
            if vw < bv - 1e-4:
                bv = vw; bad = 0; state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if bv < best_val:
            best_val = bv; best_state = state; best = model
    best.load_state_dict(best_state); best.eval()
    print(f"[train] best VAL WIS={best_val:.4f}  ({time.time()-t0:.0f}s)")

    # ---- TEST: FusedEpiNet vs champion (TiRex+Tweedie) vs baseline ----
    with torch.no_grad():
        qte, _ = best(cte, ste)
    net_w = wis_np(qte.numpy(), yte.numpy(), cap)
    y_te = yte.numpy()
    # champion Tweedie (p on VAL) + baseline via crosscountry funcs
    origins = te_idx
    def champ_tweedie():
        vw = {}
        for p in X.P_GRID:
            vqy = X.tweedie_qy(y, tirex, va_idx, p, cap); vB = X.expanding_cqr_bounds(vqy, y[va_idx], cap)
            vw[p] = float(X.wis_of(vB, y[va_idx], vqy[:, X.MED_COL]).mean())
        ps = min(vw, key=vw.get)
        tqy = X.tweedie_qy(y, tirex, origins, ps, cap); tB = X.expanding_cqr_bounds(tqy, y_te, cap)
        return X.wis_of(tB, y_te, tqy[:, X.MED_COL]), ps
    tw_w, ps = champ_tweedie()
    bqy = X.baseline_qy(y, tirex, origins, cap); bB = X.expanding_cqr_bounds(bqy, y_te, cap)
    base_w = X.wis_of(bB, y_te, bqy[:, X.MED_COL])
    # align lengths (net may drop early samples needing t-L)
    m = min(len(net_w), len(tw_w))
    net_w, tw_w2, base_w2 = net_w[-m:], tw_w[-m:], base_w[-m:]
    dm_vs_tw, _ = X.dm(net_w, tw_w2); dm_vs_base, _ = X.dm(net_w, base_w2)
    lo = np.sort(qte.numpy(), 1)[:, list(np.round(FQ,4)).index(0.05)]
    hi = np.sort(qte.numpy(), 1)[:, list(np.round(FQ,4)).index(0.95)]
    picp = float(((y_te >= lo) & (y_te <= hi)).mean())

    out = {
        "n_test": int(m), "best_val_wis": round(best_val, 4),
        "fusedepinet_wis": round(float(net_w.mean()), 4), "fusedepinet_picp95": round(picp, 3),
        "champion_tweedie_wis": round(float(tw_w2.mean()), 4), "tweedie_p": ps,
        "baseline_wis": round(float(base_w2.mean()), 4),
        "dm_p_vs_tweedie": round(dm_vs_tw, 4), "dm_p_vs_baseline": round(dm_vs_base, 4),
        "beats_champion": bool(net_w.mean() < tw_w2.mean() and dm_vs_tw < 0.05),
        "beats_baseline": bool(net_w.mean() < base_w2.mean() and dm_vs_base < 0.05),
        "elapsed_s": round(time.time()-t0, 0),
    }
    (ROOT / "scripts" / "_fusedepinet.json").write_text(json.dumps(out, indent=2))
    print("\n=== FusedEpiNet (genuine end-to-end fusion, Seoul-only) ===")
    print(json.dumps(out, indent=2))
    print("\nVERDICT:", "BEATS champion TiRex+Tweedie (DM-sig)" if out["beats_champion"]
          else (f"does NOT beat champion (net {out['fusedepinet_wis']} vs Tweedie {out['champion_tweedie_wis']}, "
                f"DM p={out['dm_p_vs_tweedie']}); "
                + ("still beats baseline" if out["beats_baseline"] else "and does not beat baseline")
                + " — honest null: 113-week from-scratch fusion cannot outrun the foundation prior"))


if __name__ == "__main__":
    raise SystemExit(main())
