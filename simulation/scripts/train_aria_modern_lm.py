"""ARIA modern from-scratch LM (v2): Llama-style torch transformer + SubQ training.

Upgrades the plain GPT (train_aria_torch_lm.py) to the 2026 modern open-LLM stack
(Raschka's gallery / TinyLlama / Llama baseline), implemented from scratch in
pure torch:
  - RoPE   : rotary position embeddings (rotate Q/K) instead of learned pos emb;
  - RMSNorm: pre-norm RMS normalisation (no mean, no bias);
  - GQA    : grouped-query attention (fewer KV heads -> cheaper inference);
  - SwiGLU : gated FFN (Swish gate), outperforms GELU at equal compute;
  - no bias terms anywhere (Llama convention).

Trained on SubQ-format reasoning traces (Self-Ask: Question -> Sub-question ->
Intermediate answer -> ... -> Final answer), i.e. chain-of-thought / reasoning
distillation by template, so the model internalises the SubQ decomposition the
user asked for (SubQ is an inference METHOD, not an architecture — the meaningful
"SubQ-based LLM" is one trained on decomposition traces). Plus the curated Korean
public-health text and PubMed epidemiology.

HONEST SCOPE: still a SMALL (~weak) from-scratch model; this demonstrates the
modern architecture + SubQ-trace training in torch, not a production model.
Offline, MPS.

Run:  .venv/bin/python -m simulation.scripts.train_aria_modern_lm --steps 2000
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
KR_TXT = _REPO / "data" / "collected" / "aria_kr_health_domain.txt"
PUBMED = _REPO / "data" / "collected" / "pubmed_abstracts"
OUT = _REPO / "results" / "aria_modern_lm"
_REL = re.compile(r"influenza|\bflu\b|ILI|respiratory|vaccine|epidemic|outbreak|"
                  r"surveillance|forecast|seasonal", re.IGNORECASE)


# ── SubQ-format reasoning-trace corpus (Self-Ask distillation by template) ────
# ── Tokenizer: word + digit + Korean-syllable (numbers stay digit-sequences for copy) ──
# Keys (start with a letter: alpha, forward_r2, R0) -> ONE token => strong key identity.
# Numbers (start with a digit: 0.722, 14) -> per-digit tokens => values are copyable.
# Korean syllables -> ONE token each (vs 3 raw UTF-8 bytes) => shorter, cleaner sequences.
_TOK_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]|[가-힣]|\s|.", re.UNICODE)


def _tokenize(text: str) -> list:
    return _TOK_RE.findall(text)


def build_vocab(text: str, max_vocab: int = 4000) -> dict:
    """Frequency vocabulary (deterministic). Reserves <unk>, digits and common
    punctuation first, then the most frequent remaining tokens up to max_vocab."""
    from collections import Counter
    cnt = Counter(_tokenize(text))
    vocab = {"<unk>": 0}
    for t in [str(d) for d in range(10)] + list(" \n\t.,=:;()[]{}%-+/*<>?!\"'…~²³"):
        vocab.setdefault(t, len(vocab))
    for t, _c in cnt.most_common():
        if len(vocab) >= max_vocab:
            break
        vocab.setdefault(t, len(vocab))
    return vocab


def encode(text: str, vocab: dict) -> list:
    unk = vocab["<unk>"]
    return [vocab.get(t, unk) for t in _tokenize(text)]


def decode(ids, inv: dict) -> str:
    return "".join(inv.get(int(i), "") for i in ids)


def make_number_constraint(vocab: dict, inv: dict, numbers):
    """Logits-mask factory: forces every emitted NUMBER to be one of ``numbers`` (the
    context numbers). Bans numeric tokens that would not extend a valid context-number
    prefix, and — while mid-way through an incomplete number — bans non-numeric tokens.
    Guarantees the model can only cite numbers present in the context (n_spurious from
    invented values -> 0); it still freely chooses WHICH context numbers and all prose."""
    numchars = set("0123456789.-")
    num_ids = [i for tok, i in vocab.items() if tok and all(c in numchars for c in tok)]
    prefixes, complete = set(), set(numbers)
    for ns in numbers:
        for k in range(1, len(ns) + 1):
            prefixes.add(ns[:k])

    def constrain(gen_ids, logits):
        import torch
        suffix = ""
        for tid in reversed(gen_ids):                 # trailing run of numeric tokens
            ts = inv.get(tid, "")
            if ts and all(c in numchars for c in ts):
                suffix = ts + suffix
            else:
                break
        for i in num_ids:
            if (suffix + inv[i]) not in prefixes:
                logits[0, i] = float("-inf")
        if suffix and suffix not in complete:         # incomplete number -> numeric tokens only
            mask = torch.ones(logits.size(-1), dtype=torch.bool, device=logits.device)
            mask[num_ids] = False
            logits[0, mask] = float("-inf")
        return logits
    return constrain


def _subq_trace(title: str, abstract: str, journal: str, year: str, pmid: str) -> str:
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", abstract) if len(s.strip()) > 20]
    if len(sents) < 2:
        return ""
    final = sents[-1]
    return (
        f"Question: From an influenza-surveillance standpoint, what is known about "
        f"{title.strip().rstrip('.')[:90]}?\n"
        f"Sub-question: What did the study examine?\n"
        f"Intermediate answer: {sents[0][:200]}\n"
        f"Sub-question: What was found?\n"
        f"Intermediate answer: {sents[1][:200]}\n"
        f"So the final answer is: {final[:200]} "
        f"[기존 문헌: {title[:60].strip()}, {journal[:30].strip()} {year}, PMID {pmid}]\n\n"
    )


_GPROMPT = ("당신은 역학자에게 시뮬레이션 결과를 해석해 주는 자문가입니다. 아래 *제공된 수치만* "
            "근거로 2~3문장으로 해석하세요. 제공되지 않은 수치를 지어내지 마세요.\n\n")  # = eval GROUNDING_PROMPT (instruction match, facts stay random)
_GKEYS = ["R2", "RMSE", "MAE", "alpha", "beta", "tau", "theta", "kappa", "Rt", "R0",
          "WIS", "coverage", "민감도", "특이도", "적합도", "발생률", "유병률",
          "백신효과", "감소율", "상관", "검출률", "지연"]


def _grounding_examples(n: int, seed: int = 7) -> list[str]:
    """Synthetic in-context number-grounding (COPY) examples.

    Each is a context full of random ``key=value`` facts plus an answer that cites
    those exact numbers — teaching the GENERAL skill the grounding eval rewards
    (echo the numbers GIVEN in the context). Uses RANDOM values only, never the
    eval's gold facts, so it is an honest capability-builder, not teaching-to-the-
    test. The short context+answer fit inside one training window so the model sees
    the copy pattern (context number -> answer number) end to end.
    """
    import random
    rng = random.Random(seed)
    out: list[str] = []
    for _ in range(n):
        k = rng.randint(3, 6)
        keys = rng.sample(_GKEYS, k)
        vals = []
        for _ in range(k):
            t = rng.random()
            if t < 0.55:
                vals.append(f"{rng.uniform(0, 1):.3g}")
            elif t < 0.8:
                vals.append(str(rng.randint(1, 60)))
            else:
                vals.append(f"{rng.uniform(1, 99):.3g}")
        pairs = [f"{a}={b}" for a, b in zip(keys, vals)]
        ctx = "분석 결과: " + ", ".join(pairs) + "."
        if rng.random() < 0.5:                          # RETRIEVAL: locate one key, copy its value
            j = rng.randrange(k)
            out.append(f"{_GPROMPT}{ctx}\n{keys[j]}는 얼마입니까?\n답: {keys[j]}={vals[j]} 입니다.\n\n")
        else:                                           # OPEN interpretation citing the context values
            out.append(f"{_GPROMPT}{ctx}\n답: 제공된 수치에 따르면 {', '.join(pairs)} 입니다.\n\n")
    return out


def build_corpus(n_abstracts: int, kr_repeat: int, n_ground: int = 0) -> bytes:
    parts: list[str] = []
    if n_ground > 0:                       # synthetic in-context number-copy task (honest, random values)
        parts += _grounding_examples(n_ground)
    if KR_TXT.exists():
        parts += [KR_TXT.read_text(encoding="utf-8")] * kr_repeat
    used = 0
    for f in sorted(glob.glob(str(PUBMED / "*.csv"))):
        if used >= n_abstracts:
            break
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r = {k.lstrip("﻿"): v for k, v in r.items()}
                t, a = r.get("title", ""), r.get("abstract", "")
                if t and a and len(a) > 150 and _REL.search(f"{t} {a}"):
                    tr = _subq_trace(t, a, r.get("journal", ""), r.get("year", ""), r.get("pmid", ""))
                    if tr:
                        parts.append(tr)
                        used += 1
                        if used >= n_abstracts:
                            break
    return "\n".join(parts)


# ── Modern components (from scratch) ─────────────────────────────────────────
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


def precompute_rope(head_dim, max_seq, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq).float()
    f = torch.outer(t, inv)                       # [T, hd/2]
    return torch.cos(f), torch.sin(f)


def apply_rope(x, cos, sin):                       # x: [B, T, nh, hd]
    x1, x2 = x[..., 0::2], x[..., 1::2]            # [B,T,nh,hd/2]
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    return torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2)


class GQAttention(nn.Module):
    """Grouped-query causal self-attention with RoPE (no bias)."""
    def __init__(self, d, n_head, n_kv_head):
        super().__init__()
        self.nh, self.nkv = n_head, n_kv_head
        self.hd = d // n_head
        self.rep = n_head // n_kv_head
        self.wq = nn.Linear(d, n_head * self.hd, bias=False)
        self.wk = nn.Linear(d, n_kv_head * self.hd, bias=False)
        self.wv = nn.Linear(d, n_kv_head * self.hd, bias=False)
        self.wo = nn.Linear(n_head * self.hd, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.nh, self.hd)
        k = self.wk(x).view(B, T, self.nkv, self.hd)
        v = self.wv(x).view(B, T, self.nkv, self.hd)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        # GQA: repeat KV heads to match Q heads
        k = k.repeat_interleave(self.rep, dim=2)
        v = v.repeat_interleave(self.rep, dim=2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))   # [B, nh, T, hd]
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, T, self.nh * self.hd)
        return self.wo(o)


class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ModernBlock(nn.Module):
    def __init__(self, d, n_head, n_kv_head):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.attn = GQAttention(d, n_head, n_kv_head)
        self.n2 = RMSNorm(d)
        self.ff = SwiGLU(d, int(d * 8 / 3 // 16 * 16))   # ~8/3 d, rounded

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)
        return x + self.ff(self.n2(x))


class ModernGPT(nn.Module):
    """Llama-style decoder: RoPE + GQA + SwiGLU + RMSNorm, no bias (from scratch)."""
    def __init__(self, vocab=256, d=384, n_head=6, n_kv_head=2, n_layer=6, block_size=256):
        super().__init__()
        self.block_size = block_size
        self.hd = d // n_head
        self.tok = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList([ModernBlock(d, n_head, n_kv_head) for _ in range(n_layer)])
        self.norm = RMSNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.apply(self._init_weights)                       # principled init (std 0.02, not default N(0,1))
        for nm, p in self.named_parameters():                # GPT-2 residual-projection depth scaling
            if nm.endswith("wo.weight") or nm.endswith("w2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * n_layer) ** 0.5)
        self.head.weight = self.tok.weight                   # weight tying (input embedding <-> LM head)
        cos, sin = precompute_rope(self.hd, block_size)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        T = idx.size(1)
        x = self.tok(idx)
        cos, sin = self.cos[:T], self.sin[:T]
        for b in self.blocks:
            x = b(x, cos, sin)
        logits = self.head(self.norm(x))
        loss = (F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                if targets is not None else None)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n, temp=0.8, top_k=40, top_p=0.0, stop=None, min_new=8,
                 repetition_penalty=1.0, no_repeat_ngram=0, constrain=None):
        start = idx.size(1)
        for _ in range(n):
            logits = self(idx[:, -self.block_size:])[0][:, -1, :]
            if repetition_penalty != 1.0 and idx.size(1) > 0:    # CTRL-style (Keskar 2019)
                for t in set(idx[0].tolist()):
                    lt = logits[0, t]
                    logits[0, t] = lt / repetition_penalty if lt > 0 else lt * repetition_penalty
            if no_repeat_ngram > 0 and idx.size(1) >= no_repeat_ngram:  # block repeated n-grams
                seq = idx[0].tolist()
                pref = tuple(seq[-(no_repeat_ngram - 1):]) if no_repeat_ngram > 1 else ()
                for i in range(len(seq) - no_repeat_ngram + 1):
                    if tuple(seq[i:i + no_repeat_ngram - 1]) == pref:
                        logits[0, seq[i + no_repeat_ngram - 1]] = -float("inf")
            if constrain is not None:                            # numbers restricted to context
                logits = constrain(idx[0, start:].tolist(), logits)
            logits = logits / temp
            if top_k:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            if 0.0 < top_p < 1.0:                                # nucleus (Holtzman 2019)
                sl, si = torch.sort(logits, descending=True)
                cum = torch.cumsum(F.softmax(sl, -1), -1)
                drop = cum > top_p
                drop[..., 1:] = drop[..., :-1].clone(); drop[..., 0] = False
                sl[drop] = -float("inf")
                logits = torch.full_like(logits, -float("inf")).scatter(1, si, sl)
            idx = torch.cat([idx, torch.multinomial(F.softmax(logits, -1), 1)], 1)
            if stop is not None and idx.size(0) == 1 and (idx.size(1) - start) >= min_new \
                    and idx[0, -len(stop):].tolist() == list(stop):
                break                                  # stop-sequence: end at answer boundary
        return idx


def main(steps: int, n_abstracts: int, kr_repeat: int, d: int = 384,
         n_layer: int = 6, n_head: int = 6, n_kv_head: int = 2, n_ground: int = 0) -> int:
    import math
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(42)
    text = build_corpus(n_abstracts, kr_repeat, n_ground)
    vocab = build_vocab(text)
    inv = {v: k for k, v in vocab.items()}
    data = np.array(encode(text, vocab), dtype=np.int64)
    print(f"corpus tokens={len(data):,} vocab={len(vocab)} (n_ground={n_ground}) device={dev}")
    n = int(0.9 * len(data))
    tr, va = torch.tensor(data[:n]), torch.tensor(data[n:])
    bs, blk = 12, 512
    model = ModernGPT(vocab=len(vocab), d=d, n_head=n_head, n_kv_head=n_kv_head, n_layer=n_layer,
                      block_size=blk).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"ModernGPT params={nparam/1e6:.2f}M (RoPE+GQA+SwiGLU+RMSNorm, no bias) blk={blk}")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), weight_decay=0.1)
    base_lr, warmup = 3e-4, max(50, steps // 25)

    def lr_at(s: int) -> float:                                  # warmup -> cosine decay to 10%
        if s < warmup:
            return base_lr * s / warmup
        prog = (s - warmup) / max(1, steps - warmup)
        return base_lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog)))

    def batch(split):
        d = tr if split == "tr" else va
        ix = torch.randint(len(d) - blk - 1, (bs,))
        x = torch.stack([d[i:i + blk] for i in ix]).to(dev)
        y = torch.stack([d[i + 1:i + 1 + blk] for i in ix]).to(dev)
        return x, y

    model.train()
    last = 0.0
    for step in range(1, steps + 1):
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x, y = batch("tr")
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # stability (Llama recipe)
        opt.step()
        last = loss.item()
        if step % 200 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                vx, vy = batch("va"); _, vl = model(vx, vy)
            model.train()
            print(f"step {step:5d}  train {last:.3f}  val {vl.item():.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "tokenizer": vocab,
                "config": {"vocab": len(vocab), "d": d, "n_head": n_head, "n_kv_head": n_kv_head,
                           "n_layer": n_layer, "block_size": blk}},
               OUT / "aria_modern_lm.pt")
    model.eval()
    idx = torch.tensor([encode("Question: What is influenza vaccine effectiveness?\nSub-question:", vocab)],
                       dtype=torch.long, device=dev)
    sample = decode(model.generate(idx, 220)[0].tolist(), inv)
    (OUT / "sample.txt").write_text(sample, encoding="utf-8")
    (OUT / "meta.json").write_text(json.dumps({
        "arch": "Llama-style: RoPE+GQA(6/2 heads)+SwiGLU+RMSNorm, no bias",
        "params_M": round(nparam / 1e6, 2), "corpus_tokens": int(len(data)), "vocab": len(vocab),
        "steps": steps, "device": dev, "final_train_loss": round(last, 3),
        "training": "SubQ-format reasoning traces (Self-Ask / CoT distillation by template)",
        "honest_note": "small weak from-scratch model; demonstrates modern arch + SubQ-trace "
                       "training in torch, not a production model."}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print("SAMPLE:", sample[:220].replace("\n", " | "))
    print(f"-> {OUT}/aria_modern_lm.pt")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--abstracts", type=int, default=2500)
    ap.add_argument("--kr-repeat", type=int, default=120)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--n-layer", type=int, default=6)
    ap.add_argument("--n-head", type=int, default=6)
    ap.add_argument("--n-kv-head", type=int, default=2)
    ap.add_argument("--n-ground", type=int, default=0,
                    help="synthetic in-context number-copy examples (honest, random values)")
    args = ap.parse_args()
    raise SystemExit(main(args.steps, args.abstracts, args.kr_repeat,
                          args.d, args.n_layer, args.n_head, args.n_kv_head, args.n_ground))
