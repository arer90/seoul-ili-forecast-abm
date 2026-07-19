#!/usr/bin/env python
"""Genuine last fine-tuning variant: LoRA-fine-tune the PRETRAINED TiRex (output-side layers only)
with FAIR EXPANDING training (the fixed-113 LoRA overfit; here we use 153..284 weeks, per-block refit,
matching TiRex's expanding context). Output-side LoRA (output_patch_embedding + last 2 xLSTM blocks) —
avoids the NaN-gradient input_patch_embedding; grad-sanitize; do-no-harm gate on a rolling val.

At each refit block the LoRA adapters are trained on [52, block_start) then the fine-tuned TiRex rolls its
NATIVE quantiles over the block. Compared to plain default TiRex-native on TEST[205,337), DM(HLN)+bootstrap.
Leak-free (context y[:t]; adapters trained on past only). No live/pipeline edits.
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
from scripts._tirex_headprobe_final import flusight
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
from simulation.analytics.adaptive_conformal import wis_from_bounds

FQ = np.asarray(X.FQ, float); FQr = list(np.round(FQ, 4)); MEDI = FQr.index(0.5); NQ = len(FQ)
DEC = np.array([.1,.2,.3,.4,.5,.6,.7,.8,.9])
MIN_CTX = 52; T0 = 205; REFIT_K = 22; VAL_TAIL = 34; MAX_CTX = 512


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


class LoRALin:
    pass


def inject_output_lora(model, rank=4, alpha=8.0):
    """LoRA on OUTPUT-side nn.Linear only (name contains output_patch_embedding or blocks.10/blocks.11)."""
    import torch, torch.nn as nn, math
    class LoRALinear(nn.Module):
        def __init__(self, base):
            super().__init__(); self.base = base
            for p in base.parameters(): p.requires_grad = False
            self.A = nn.Parameter(torch.zeros(rank, base.in_features))
            self.B = nn.Parameter(torch.zeros(base.out_features, rank)); self.s = alpha/rank
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        def forward(self, x): return self.base(x) + self.s*((x @ self.A.t()) @ self.B.t())
    for p in model.parameters(): p.requires_grad = False
    n_adapt = 0
    for name, mod in list(model.named_modules()):
        for cn, child in list(mod.named_children()):
            full = f"{name}.{cn}" if name else cn
            if isinstance(child, nn.Linear) and min(child.in_features, child.out_features) >= 32 \
               and ("output_patch_embedding" in full or "blocks.10." in full or "blocks.11." in full):
                setattr(mod, cn, LoRALinear(child)); n_adapt += 1
    return model, n_adapt


def native_qy(model, y, idxs, cap):
    import torch
    Q = np.zeros((len(idxs), NQ))
    with torch.no_grad():
        for k, t in enumerate(idxs):
            ctx = torch.tensor(y[max(0, t-MAX_CTX):t], dtype=torch.float32).unsqueeze(0)
            q, _ = model.forecast(context=ctx, prediction_length=1)
            Q[k] = flusight(np.sort(np.asarray(q).ravel()), cap)
    return Q


def train_lora(model, y, train_end, epochs=2, lr=5e-4, stride=3):
    import torch
    fq = torch.tensor(FQ, dtype=torch.float32)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    tr = [p for p in model.parameters() if p.requires_grad]
    model.train()
    for _ in range(epochs):
        for t in range(MIN_CTX, train_end, stride):
            ctx = torch.tensor(y[max(0, t-MAX_CTX):t], dtype=torch.float32).unsqueeze(0)
            try:
                pred = model._forecast_tensor(ctx, prediction_length=1)      # (1,9,1) differentiable
                q9 = pred.reshape(pred.shape[0], pred.shape[1])[0]            # 9 deciles
                dl = torch.tensor([.1,.2,.3,.4,.5,.6,.7,.8,.9], dtype=torch.float32)
                e = float(y[t]) - q9
                loss = torch.mean(torch.maximum(dl*e, (dl-1)*e))
                if not torch.isfinite(loss): continue
                opt.zero_grad(); loss.backward()
                for p in tr:
                    if p.grad is not None: torch.nan_to_num_(p.grad, 0., 0., 0.)
                torch.nn.utils.clip_grad_norm_(tr, 1.0); opt.step()
            except Exception:
                continue
    model.eval()


def main():
    t0 = time.time()
    from tirex import load_model
    S = setup(); y = S["yf"]; N = S["ntot"]; cap = 2.0*float(np.nanmax(y[:T0]))
    dec = np.load(ROOT/"scripts"/"_tirex_native_deciles.npz")["dec"]
    te = np.arange(T0, N); n = len(te); y_te = y[te]
    nat = np.array([flusight(dec[t], cap) for t in te]); nat_w = wis_arr(nat, y_te, cap)

    qte = np.zeros((n, NQ)); used = 0; sizes = []
    for bstart in range(T0, N, REFIT_K):
        bend = min(bstart+REFIT_K, N); sizes.append(bstart-MIN_CTX)
        model = load_model("NX-AI/TiRex", device="cpu")
        model, na = inject_output_lora(model, rank=4, alpha=8.0)
        # rolling val for do-no-harm: [bstart-VAL_TAIL, bstart)
        vl = np.arange(bstart-VAL_TAIL, bstart)
        train_lora(model, y, bstart-VAL_TAIL, epochs=2)                       # train on [52, bstart-VAL_TAIL)
        ft_vl = native_qy(model, y, vl, cap); base_vl = np.array([flusight(dec[t], cap) for t in vl])
        use = wis_arr(ft_vl, y[vl], cap).mean() < wis_arr(base_vl, y[vl], cap).mean()
        oi = np.arange(bstart, bend)
        if use:
            qte[oi-T0] = native_qy(model, y, oi, cap); used += len(oi)
        else:
            qte[oi-T0] = np.array([flusight(dec[t], cap) for t in oi])
        print(f"  block {bstart}: train {bstart-MIN_CTX}wk, {na} LoRA mods, use_ft={use} ({time.time()-t0:.0f}s)", flush=True)

    ft_w = wis_arr(qte, y_te, cap); hln, bp = dm_boot(ft_w, nat_w)
    out = {
        "approach": "output-side LoRA fine-tune of pretrained TiRex, FAIR expanding (%d..%d wk), per-block do-no-harm" % (min(sizes), max(sizes)),
        "native_wis_test": round(float(nat_w.mean()), 4), "lora_ft_wis_test": round(float(ft_w.mean()), 4),
        "delta_pct": round(100*(ft_w.mean()-nat_w.mean())/nat_w.mean(), 2),
        "ft_used_on_origins": used, "of_total": n,
        "dm_hln_p": round(hln, 4), "dm_boot_p": round(bp, 4),
        "beats_native": bool(ft_w.mean() < nat_w.mean() and hln < 0.05 and bp < 0.05),
        "elapsed_s": round(time.time()-t0, 0),
    }
    (ROOT/"scripts"/"_tirex_lora_fair.json").write_text(json.dumps(out, indent=2))
    print("\n"+json.dumps(out, indent=2))
    print("\nVERDICT:", "FAIR-expanding LoRA fine-tune BEATS TiRex-native!" if out["beats_native"]
          else "fair-expanding LoRA fine-tune (%.4f) does NOT beat native (%.4f) — backbone fine-tune also hits the floor"
               % (out["lora_ft_wis_test"], out["native_wis_test"]))


if __name__ == "__main__":
    raise SystemExit(main())
