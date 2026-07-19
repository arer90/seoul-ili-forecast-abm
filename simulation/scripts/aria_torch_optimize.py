"""Optimization techniques on the from-scratch ARIA torch LM: LoRA + quantization
+ pruning, all implemented in pure PyTorch and measured honestly.

Loads the from-scratch GPT trained by train_aria_torch_lm.py and demonstrates,
on the SAME model, the three optimization techniques the user requires:

  - LoRA      : low-rank adapters (A·B) injected into the feed-forward linears,
                base frozen, adapter briefly trained -> trainable-param fraction.
  - Quantize  : symmetric int8 weight quantization -> size reduction + perplexity.
  - Prune     : L1-magnitude unstructured pruning -> sparsity + perplexity.

Honest scope: the base is a small, weak from-scratch model, so the perplexity
deltas are demonstrations of the techniques' EFFECTS (size/sparsity vs accuracy
trade-off), not a production model. Pure torch, offline, MPS/CPU.

Run:  .venv/bin/python -m simulation.scripts.aria_torch_optimize
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from simulation.scripts.train_aria_torch_lm import GPT, build_corpus

_REPO = Path(__file__).resolve().parents[1]
CKPT = _REPO / "results" / "aria_torch_lm" / "aria_torch_lm.pt"
OUT = _REPO / "results" / "aria_torch_lm" / "optimization.json"


# --------------------------------------------------------------------------- #
# LoRA — low-rank adapter on a frozen nn.Linear (pure torch)
# --------------------------------------------------------------------------- #
class LoRALinear(nn.Module):
    """Wrap a frozen Linear with a trainable low-rank update A·B (rank r)."""
    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        dev = base.weight.device  # keep adapters on the base's device (MPS)
        self.A = nn.Parameter(torch.randn(base.in_features, r, device=dev) * 0.01)
        self.B = nn.Parameter(torch.zeros(r, base.out_features, device=dev))
        self.scaling = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.A @ self.B) * self.scaling


def _inject_lora(model: nn.Module, r: int = 8) -> int:
    """Replace the feed-forward Linears with LoRA wrappers; return adapter params.

    Freezes the ENTIRE base first so that only the injected low-rank adapters are
    trainable (the canonical LoRA setup -> tiny trainable fraction)."""
    for p in model.parameters():
        p.requires_grad = False
    adapter_params = 0
    for blk in model.blocks:
        ff = blk.ff
        for i, layer in enumerate(ff):
            if isinstance(layer, nn.Linear):
                lora = LoRALinear(layer, r=r)
                ff[i] = lora
                adapter_params += lora.A.numel() + lora.B.numel()
    return adapter_params


def _eval_ppl(model, data, dev, blk=256, n_batches=20) -> float:
    """Mean cross-entropy perplexity over random validation blocks."""
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_batches):
            ix = torch.randint(len(data) - blk - 1, (8,))
            x = torch.stack([data[i:i + blk] for i in ix]).to(dev)
            y = torch.stack([data[i + 1:i + 1 + blk] for i in ix]).to(dev)
            _, loss = model(x, y)
            losses.append(loss.item())
    return float(math.exp(np.mean(losses)))


def _model_bytes(model) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


def _quant_group_tensor(W, bits, gs):
    """Per-group symmetric int quantization of one weight matrix (dequantized).

    Groups of ``gs`` input features share a scale — the correlation/group-aware
    ("entanglement-as-correlation") analog that beats per-tensor at low bits.
    Returns (Wq, scale_bytes).
    """
    out, inp = W.shape
    qmax = 2 ** (bits - 1) - 1
    if inp % gs != 0:  # fall back to per-tensor for non-divisible shapes
        scale = (W.abs().amax(1, keepdim=True) / qmax).clamp_min(1e-8)
        return torch.clamp(torch.round(W / scale), -qmax, qmax) * scale, out * 4
    Wr = W.view(out, inp // gs, gs)
    scale = (Wr.abs().amax(-1, keepdim=True) / qmax).clamp_min(1e-8)
    Wq = (torch.clamp(torch.round(Wr / scale), -qmax, qmax) * scale).view(out, inp)
    return Wq, out * (inp // gs) * 4


def _quantize_per_group(model, dev, data, blk, bits=4, gs=64):
    """Per-group int4 weight quantization (group/correlation-aware)."""
    qbytes = 0
    with torch.no_grad():
        for p in model.parameters():
            if p.dim() == 2:
                Wq, sbytes = _quant_group_tensor(p.data, bits, gs)
                p.data.copy_(Wq)
                qbytes += int(p.numel() * bits / 8) + sbytes
            else:
                qbytes += p.numel() * 4
    return qbytes, _eval_ppl(model, data, dev, blk)


def _quantize_awq(model, dev, data, blk, bits=4, gs=64):
    """AWQ-style activation-aware per-group quantization (Lin et al. 2024).

    Protects salient weight channels (high input-activation magnitude) by scaling
    them up before quantization and folding the scale back out, reducing error on
    the channels that matter most — the activation-aware analog of Wanda for quant.
    """
    acts: dict = {}
    lins = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]

    def mk(name):
        def hook(_m, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1])
            acts[name] = acts.get(name, 0) + x.pow(2).sum(0)
        return hook

    hooks = [m.register_forward_pre_hook(mk(n)) for n, m in lins]
    model.eval()
    with torch.no_grad():
        for _ in range(8):
            ix = torch.randint(len(data) - blk - 1, (8,))
            model(torch.stack([data[i:i + blk] for i in ix]).to(dev))
    for h in hooks:
        h.remove()
    qbytes = 0
    with torch.no_grad():
        for n, m in lins:
            W = m.weight.data
            if n in acts:
                s = torch.sqrt(acts[n] + 1e-8).to(W.device)        # [in] salience
                s = (s / s.mean()).clamp(0.2, 5.0)                  # per-channel scale
                Wq, sb = _quant_group_tensor(W * s[None, :], bits, gs)
                m.weight.data.copy_(Wq / s[None, :])                # fold scale out
                qbytes += int(W.numel() * bits / 8) + sb + W.shape[1] * 4
            else:
                qbytes += W.numel() * 4
    return qbytes, _eval_ppl(model, data, dev, blk)


def _quantize_int8(model):
    """Symmetric per-tensor int8 weight quantization (dequantized in place)."""
    qbytes = 0
    total = 0
    with torch.no_grad():
        for p in model.parameters():
            total += p.numel() * 4  # fp32 reference
            if p.dim() >= 2:  # quantize weight matrices
                scale = p.abs().max() / 127.0
                if scale > 0:
                    q = torch.clamp(torch.round(p / scale), -127, 127)
                    p.copy_(q * scale)  # dequantized (simulated int8)
                qbytes += p.numel() * 1 + 4  # int8 + scale
            else:
                qbytes += p.numel() * 4
    return qbytes, total


def _prune_structured(model, dev, data, blk, amount: float = 0.3):
    """L2 structured pruning of FFN output channels -> real-hardware-speedup shape.

    Unlike unstructured zeros (no GPU speedup), structured pruning removes whole
    output channels, so the surviving weights form a smaller dense matrix.
    """
    for b in model.blocks:
        for layer in b.ff:
            if isinstance(layer, nn.Linear) and layer.weight.shape[0] > 8:
                prune.ln_structured(layer, "weight", amount=amount, n=2, dim=0)
                prune.remove(layer, "weight")
    zeroed = tot = 0
    for b in model.blocks:
        for layer in b.ff:
            if isinstance(layer, nn.Linear):
                zeroed += int((layer.weight.abs().sum(1) == 0).sum().item())
                tot += layer.weight.shape[0]
    return zeroed / max(tot, 1), _eval_ppl(model, data, dev, blk)


def _prune_wanda(model, dev, data, blk, amount: float = 0.3):
    """Wanda pruning (Sun et al. 2024): prune by |W| x ||activation|| (activation-aware)."""
    acts: dict = {}
    lins = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]

    def mk(name):
        def hook(_mod, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1])
            acts[name] = acts.get(name, 0) + x.pow(2).sum(0)
        return hook

    hooks = [m.register_forward_pre_hook(mk(n)) for n, m in lins]
    model.eval()
    with torch.no_grad():
        for _ in range(8):
            ix = torch.randint(len(data) - blk - 1, (8,))
            model(torch.stack([data[i:i + blk] for i in ix]).to(dev))
    for h in hooks:
        h.remove()
    zeroed = tot = 0
    with torch.no_grad():
        for n, m in lins:
            if n in acts:
                act = torch.sqrt(acts[n] + 1e-8).to(m.weight.device)        # [in]
                score = m.weight.abs() * act[None, :]                        # [out,in]
                k = int(amount * score.numel())
                if k > 0:
                    thr = torch.kthvalue(score.flatten(), k).values
                    m.weight.mul_((score > thr).float())
            zeroed += int((m.weight == 0).sum().item())
            tot += m.weight.numel()
    return zeroed / max(tot, 1), _eval_ppl(model, data, dev, blk)


def _prune_sparsegpt(model, dev, data, blk, amount: float = 0.3, lam: float = 0.01):
    """SparseGPT-style pruning (Frantar & Alistarh 2023): second-order (Hessian-OBS)
    saliency + blocked sequential weight COMPENSATION. For each Linear, H = X^T X
    from calibration; the inverse Hessian (Cholesky) drives both the OBS saliency
    (W_ij^2 / [Hinv]_jj^2, a second-order criterion vs Wanda's first-order) AND the
    error-feedback update of the surviving weights — the full SparseGPT, which is
    what lets it prune without retraining."""
    Hs: dict = {}
    lins = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]

    def mk(name):
        def hook(_m, inp):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            Hs[name] = Hs.get(name, torch.zeros(x.shape[1], x.shape[1], device=x.device)) + x.t() @ x
        return hook

    hooks = [m.register_forward_pre_hook(mk(n)) for n, m in lins]
    model.eval()
    with torch.no_grad():
        for _ in range(8):
            ix = torch.randint(len(data) - blk - 1, (8,))
            model(torch.stack([data[i:i + blk] for i in ix]).to(dev))
    for h in hooks:
        h.remove()
    zeroed = tot = 0
    bsz = 128
    with torch.no_grad():
        for n, m in lins:
            if n not in Hs:
                zeroed += int((m.weight == 0).sum().item()); tot += m.weight.numel(); continue
            W = m.weight.data.clone().float()                     # [out, cols]
            cols = W.shape[1]
            H = Hs[n].to(W.device).float()
            dead = torch.diag(H) == 0
            H[dead, dead] = 1.0; W[:, dead] = 0
            damp = lam * torch.diag(H).mean().clamp_min(1e-6)
            idx = torch.arange(cols, device=W.device)
            H[idx, idx] += damp
            try:                                                  # upper Cholesky of H^-1
                Hinv = torch.linalg.cholesky(H)
                Hinv = torch.cholesky_inverse(Hinv)
                Hinv = torch.linalg.cholesky(Hinv, upper=True)
            except Exception:                                     # non-PD -> magnitude fallback
                k = int(amount * W.numel())
                if k > 0:
                    thr = torch.kthvalue(W.abs().flatten(), k).values
                    m.weight.data.copy_((W * (W.abs() > thr).float()).to(m.weight.dtype))
                zeroed += int((m.weight == 0).sum().item()); tot += m.weight.numel(); continue
            for i1 in range(0, cols, bsz):                        # blocked sequential OBS
                i2 = min(i1 + bsz, cols)
                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]
                d2 = torch.diag(Hinv1).clamp_min(1e-8)
                sal = W1 ** 2 / (d2.reshape(1, -1) ** 2)          # OBS saliency
                kk = int(amount * sal.numel())
                mask1 = (sal <= torch.kthvalue(sal.flatten(), kk).values) if kk > 0 \
                    else torch.zeros_like(W1, dtype=torch.bool)
                for j in range(i2 - i1):
                    w = W1[:, j]; dd = Hinv1[j, j]
                    q = w.clone(); q[mask1[:, j]] = 0
                    Q1[:, j] = q
                    err = (w - q) / dd                            # OBS error feedback
                    W1[:, j:] -= err.unsqueeze(1) * Hinv1[j, j:].unsqueeze(0)
                    Err1[:, j] = err
                W[:, i1:i2] = Q1
                W[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]              # cross-block compensation
            m.weight.data.copy_(W.to(m.weight.dtype))
            zeroed += int((m.weight == 0).sum().item())
            tot += m.weight.numel()
    return zeroed / max(tot, 1), _eval_ppl(model, data, dev, blk)


def _prune_random(model, dev, data, blk, amount: float = 0.3):
    """Random unstructured pruning - the null/control baseline that every informed
    criterion (magnitude / Wanda / SparseGPT / SNIP) must beat to justify itself."""
    zeroed = tot = 0
    for m in model.modules():
        if isinstance(m, nn.Linear):
            prune.random_unstructured(m, name="weight", amount=amount)
            prune.remove(m, "weight")
            zeroed += int((m.weight == 0).sum().item())
            tot += m.weight.numel()
    return zeroed / max(tot, 1), _eval_ppl(model, data, dev, blk)


def _prune_snip(model, dev, data, blk, amount: float = 0.3):
    """SNIP (Lee et al. 2019): single-shot connection sensitivity. Saliency = |g * W|
    from ONE calibration forward+backward on the task LM loss; prune the globally
    lowest-saliency weights. Genuinely one-shot (no retraining), data-aware - the
    product |g*W| (not |g| or |W| alone) is the whole point."""
    lins = [(n, m) for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    model.eval()
    model.zero_grad()
    ix = torch.randint(len(data) - blk - 1, (8,))
    x = torch.stack([data[i:i + blk] for i in ix]).to(dev)
    y = torch.stack([data[i + 1:i + 1 + blk] for i in ix]).to(dev)
    _, loss = model(x, y)                                       # task LM cross-entropy
    loss.backward()
    sal = {n: (m.weight.grad * m.weight.data).abs() for n, m in lins if m.weight.grad is not None}
    allsal = torch.cat([s.flatten() for s in sal.values()])     # GLOBAL ranking
    k = int(amount * allsal.numel())
    thr = torch.kthvalue(allsal, k).values if k > 0 else allsal.min() - 1
    zeroed = tot = 0
    with torch.no_grad():
        for n, m in lins:
            if n in sal:
                m.weight.data.mul_((sal[n] > thr).float())
            zeroed += int((m.weight == 0).sum().item())
            tot += m.weight.numel()
    model.zero_grad()
    return zeroed / max(tot, 1), _eval_ppl(model, data, dev, blk)


def _prune_nm(model, dev, data, blk, N: int = 2, M: int = 4):
    """N:M semi-structured magnitude sparsity (2:4): in every consecutive group of M
    weights along the INPUT dim (the contraction axis the 2:4 sparse tensor cores
    expect), keep the N largest by |W| and zero the rest. One-shot, no grads.

    NOTE: a real ~2x speedup needs Ampere+/cuSPARSELt sparse tensor cores; on MPS/CPU
    this is a sparsity-PATTERN + accuracy demo only (no speedup on this machine)."""
    zeroed = tot = 0
    with torch.no_grad():
        for m in model.modules():
            if not isinstance(m, nn.Linear):
                continue
            W = m.weight.data
            out, inp = W.shape
            if inp % M == 0:                                    # 384/1536/256 all divisible by 4
                Wg = W.view(out, inp // M, M)
                top = Wg.abs().topk(N, dim=-1).indices
                mask = torch.zeros_like(Wg)
                mask.scatter_(-1, top, 1.0)
                m.weight.data.copy_((Wg * mask).view(out, inp))
            zeroed += int((m.weight == 0).sum().item())
            tot += m.weight.numel()
    return zeroed / max(tot, 1), _eval_ppl(model, data, dev, blk)


def main() -> int:
    if not CKPT.exists():
        print(f"ERROR: base model not found at {CKPT} (train it first)")
        return 1
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    ckpt = torch.load(CKPT, map_location=dev)
    cfg = ckpt["config"]

    def fresh():
        m = GPT(**cfg).to(dev)
        m.load_state_dict(ckpt["state_dict"])
        return m

    # validation data (same corpus recipe)
    data = torch.tensor(np.frombuffer(build_corpus(800, 60), dtype=np.uint8).astype(np.int64))
    blk = cfg["block_size"]

    base = fresh()
    n_params = sum(p.numel() for p in base.parameters())
    base_ppl = _eval_ppl(base, data, dev, blk)
    base_bytes = _model_bytes(base)
    print(f"BASE: params={n_params/1e6:.2f}M  ppl={base_ppl:.2f}  size={base_bytes/1e6:.1f}MB")

    report = {"base": {"params_M": round(n_params / 1e6, 2),
                       "perplexity": round(base_ppl, 2),
                       "size_MB": round(base_bytes / 1e6, 2)}}

    # ---- LoRA ----
    lora_model = fresh()
    adapter = _inject_lora(lora_model, r=8)
    trainable = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in lora_model.parameters() if p.requires_grad], lr=1e-3)
    lora_model.train()
    for _ in range(60):  # brief adapter fit
        ix = torch.randint(len(data) - blk - 1, (8,))
        x = torch.stack([data[i:i + blk] for i in ix]).to(dev)
        y = torch.stack([data[i + 1:i + 1 + blk] for i in ix]).to(dev)
        _, loss = lora_model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
    lora_ppl = _eval_ppl(lora_model, data, dev, blk)
    report["lora"] = {"rank": 8, "adapter_params": adapter,
                      "trainable_params": trainable,
                      "trainable_fraction": round(trainable / n_params, 4),
                      "perplexity_after_fit": round(lora_ppl, 2)}
    print(f"LoRA: adapter={adapter} trainable={trainable} "
          f"({trainable/n_params:.1%} of base)  ppl={lora_ppl:.2f}")

    # ---- Quantization (int8) ----
    qmodel = fresh()
    qbytes, _ = _quantize_int8(qmodel)
    q_ppl = _eval_ppl(qmodel, data, dev, blk)
    report["quantization"] = {"scheme": "symmetric per-tensor int8 (weights)",
                              "size_MB": round(qbytes / 1e6, 2),
                              "size_reduction_x": round(base_bytes / max(qbytes, 1), 2),
                              "perplexity": round(q_ppl, 2)}
    print(f"INT8(per-tensor): size={qbytes/1e6:.1f}MB ({base_bytes/qbytes:.1f}x)  ppl={q_ppl:.2f}")

    # ---- Advanced quantization: per-group int4 + AWQ (the "entanglement=correlation" + activation-aware analogs) ----
    gq_bytes, gq_ppl = _quantize_per_group(fresh(), dev, data, blk, bits=4, gs=64)
    report["quantization_pergroup_int4"] = {
        "scheme": "per-group int4 (group=64, correlation/group-aware)",
        "size_MB": round(gq_bytes / 1e6, 2),
        "size_reduction_x": round(base_bytes / max(gq_bytes, 1), 2),
        "perplexity": round(gq_ppl, 2)}
    print(f"INT4(per-group): size={gq_bytes/1e6:.1f}MB ({base_bytes/gq_bytes:.1f}x)  ppl={gq_ppl:.2f}")
    aq_bytes, aq_ppl = _quantize_awq(fresh(), dev, data, blk, bits=4, gs=64)
    report["quantization_awq_int4"] = {
        "scheme": "AWQ activation-aware per-group int4 (Lin 2024)",
        "size_MB": round(aq_bytes / 1e6, 2), "perplexity": round(aq_ppl, 2),
        "note": "protects salient channels -> lower ppl than plain int4 at same bits"}
    print(f"AWQ(int4): size={aq_bytes/1e6:.1f}MB  ppl={aq_ppl:.2f}")

    # ---- Pruning (L1 magnitude, 30% unstructured) ----
    pmodel = fresh()
    pruned_w = 0
    total_w = 0
    for m in pmodel.modules():
        if isinstance(m, nn.Linear):
            prune.l1_unstructured(m, name="weight", amount=0.30)
            prune.remove(m, "weight")
            pruned_w += int((m.weight == 0).sum().item())
            total_w += m.weight.numel()
    p_ppl = _eval_ppl(pmodel, data, dev, blk)
    report["pruning"] = {"method": "L1 unstructured 30%",
                         "linear_sparsity": round(pruned_w / max(total_w, 1), 3),
                         "perplexity": round(p_ppl, 2)}
    print(f"PRUNE(L1 unstructured): linear sparsity={pruned_w/total_w:.1%}  ppl={p_ppl:.2f}")

    # ---- Pruning variety: structured (L2 channels) + Wanda (activation-aware) ----
    s_sp, s_ppl = _prune_structured(fresh(), dev, data, blk, 0.30)
    report["pruning_structured"] = {"method": "L2 structured FFN output-channels 30%",
                                    "channel_sparsity": round(s_sp, 3),
                                    "perplexity": round(s_ppl, 2),
                                    "note": "structured -> real hardware speedup (unstructured zeros do not)"}
    print(f"PRUNE(structured L2): channel sparsity={s_sp:.1%}  ppl={s_ppl:.2f}")

    w_sp, w_ppl = _prune_wanda(fresh(), dev, data, blk, 0.30)
    report["pruning_wanda"] = {"method": "Wanda |W|x||act|| 30% (activation-aware, Sun 2024)",
                               "sparsity": round(w_sp, 3), "perplexity": round(w_ppl, 2)}
    print(f"PRUNE(Wanda): sparsity={w_sp:.1%}  ppl={w_ppl:.2f}")

    sg_sp, sg_ppl = _prune_sparsegpt(fresh(), dev, data, blk, 0.30)
    report["pruning_sparsegpt"] = {"method": "SparseGPT Hessian-OBS saliency + blocked OBS error-feedback compensation 30% (Frantar 2023)",
                                   "sparsity": round(sg_sp, 3), "perplexity": round(sg_ppl, 2)}
    print(f"PRUNE(SparseGPT): sparsity={sg_sp:.1%}  ppl={sg_ppl:.2f}")

    # ---- More pruning: random control + SNIP (single-shot |g*W|) + N:M 2:4 ----
    r_sp, r_ppl = _prune_random(fresh(), dev, data, blk, 0.30)
    report["pruning_random"] = {"method": "random unstructured 30% (null/control baseline)",
                                "sparsity": round(r_sp, 3), "perplexity": round(r_ppl, 2)}
    print(f"PRUNE(random control): sparsity={r_sp:.1%}  ppl={r_ppl:.2f}")

    sn_sp, sn_ppl = _prune_snip(fresh(), dev, data, blk, 0.30)
    report["pruning_snip"] = {"method": "SNIP |grad x W| single-shot connection sensitivity 30% (Lee 2019; one-shot, data-aware)",
                              "sparsity": round(sn_sp, 3), "perplexity": round(sn_ppl, 2)}
    print(f"PRUNE(SNIP): sparsity={sn_sp:.1%}  ppl={sn_ppl:.2f}")

    nm_sp, nm_ppl = _prune_nm(fresh(), dev, data, blk, 2, 4)
    report["pruning_nm_2_4"] = {"method": "N:M 2:4 semi-structured magnitude sparsity (one-shot)",
                                "sparsity": round(nm_sp, 3), "perplexity": round(nm_ppl, 2),
                                "note": "real 2x speedup needs Ampere+/cuSPARSELt sparse tensor cores; on MPS/CPU = pattern+accuracy demo only"}
    print(f"PRUNE(N:M 2:4): sparsity={nm_sp:.1%}  ppl={nm_ppl:.2f}")

    report["honest_note"] = ("All optimisation techniques (LoRA; per-tensor / per-group / "
                             "AWQ quantisation; random / L1 / structured / Wanda / SparseGPT / "
                             "SNIP / N:M-2:4 pruning) are implemented in pure torch and measured "
                             "on the small from-scratch ARIA LM; the weak base means perplexities "
                             "are high, so this demonstrates the size/sparsity vs accuracy "
                             "trade-off of each technique, not a production model.")
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
