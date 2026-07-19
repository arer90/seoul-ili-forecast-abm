"""ARIA from-scratch language model — a small GPT implemented in pure PyTorch.

NOT Ollama, NOT a fine-tune of a pretrained model: this implements the LLM
architecture (token + positional embeddings, causal multi-head self-attention,
feed-forward blocks, layer norm, an LM head) in torch and trains it FROM SCRATCH
on the project's epidemiology corpus (PubMed influenza abstracts) plus curated
Korean public-health / infectious-disease-law / 보건복지부 domain text. Byte-level
tokenisation (256 vocab) handles English and Korean uniformly.

HONEST SCOPE: this is a deliberately SMALL model (~11M params) trained briefly on
a modest corpus, so it is WEAK — it learns domain vocabulary, n-gram structure,
and bilingual epidemiology phrasing, but it is a demonstration of "we built the
LLM structure ourselves in torch", NOT a usable advisory model (the pretrained
ARIA backends serve that role). Offline, MPS.

Run:  .venv/bin/python -m simulation.scripts.train_aria_torch_lm --steps 1500
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
KR_TXT = _REPO / "data" / "collected" / "aria_kr_health_domain.txt"
PUBMED = _REPO / "data" / "collected" / "pubmed_abstracts"
OUT = _REPO / "results" / "aria_torch_lm"
_RELEVANT = re.compile(r"influenza|\bflu\b|ILI|respiratory|vaccine|epidemic|outbreak|"
                       r"surveillance|forecast|seasonal", re.IGNORECASE)


def build_corpus(n_abstracts: int, kr_repeat: int) -> bytes:
    """Bilingual epidemiology byte corpus: Korean health text (weighted) + PubMed."""
    parts: list[str] = []
    if KR_TXT.exists():
        parts += [KR_TXT.read_text(encoding="utf-8")] * kr_repeat  # weight the Korean
    used = 0
    for f in sorted(glob.glob(str(PUBMED / "*.csv"))):
        if used >= n_abstracts:
            break
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                r = {k.lstrip("﻿"): v for k, v in r.items()}
                t, a = r.get("title", ""), r.get("abstract", "")
                if t and a and len(a) > 120 and _RELEVANT.search(f"{t} {a}"):
                    parts.append(f"\n[기존 문헌] {t}\n{a}\n")
                    used += 1
                    if used >= n_abstracts:
                        break
    text = "\n".join(parts)
    return text.encode("utf-8")


class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(n_embd), nn.LayerNorm(n_embd)
        self.attn = nn.MultiheadAttention(n_embd, n_head, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.GELU(),
                                nn.Linear(4 * n_embd, n_embd), nn.Dropout(dropout))
        self.register_buffer("mask", torch.triu(torch.ones(block_size, block_size), 1).bool())

    def forward(self, x):
        T = x.size(1)
        a = self.ln1(x)
        ao, _ = self.attn(a, a, a, attn_mask=self.mask[:T, :T], need_weights=False)
        x = x + ao
        return x + self.ff(self.ln2(x))


class GPT(nn.Module):
    """A small from-scratch GPT (causal transformer decoder) in pure torch."""
    def __init__(self, vocab=256, n_embd=384, n_head=6, n_layer=6, block_size=256, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok = nn.Embedding(vocab, n_embd)
        self.pos = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.lnf = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab, bias=False)

    def forward(self, idx, targets=None):
        T = idx.size(1)
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos)[None])
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.lnf(x))
        loss = (F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
                if targets is not None else None)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, n, temp=0.8, top_k=40):
        for _ in range(n):
            logits, _ = self(idx[:, -self.block_size:])
            logits = logits[:, -1, :] / temp
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float("inf")
            nxt = torch.multinomial(F.softmax(logits, -1), 1)
            idx = torch.cat([idx, nxt], 1)
        return idx


class TorchLMBackend:
    """LLMBackend-compatible wrapper around the from-scratch ARIA torch LM.

    Lets the backend-agnostic ARIA components (notably the SubQ / Self-Ask flow in
    simulation.llm_compare.subq, which drives any object with ``.generate(prompt)
    -> resp(.text/.error)``) use the from-scratch torch model. The model is WEAK
    (small from-scratch LM) so this demonstrates the *integration*, not advisory
    quality — the pretrained backends remain the quality tier.
    """
    backend_id = "torch:aria-torch-lm"
    model = "aria-torch-lm"
    provider = "local"
    tier = "local"

    def __init__(self, ckpt: Path = OUT / "aria_torch_lm.pt", device: str | None = None):
        self.dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        c = torch.load(ckpt, map_location=self.dev)
        self.m = GPT(**c["config"]).to(self.dev)
        self.m.load_state_dict(c["state_dict"])
        self.m.eval()
        self.block_size = c["config"]["block_size"]

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str, *, system: str | None = None,
                 temperature: float = 0.8, max_tokens: int = 160, **_):
        import time
        from simulation.llm_compare.backends import LLMResponse
        t0 = time.time()
        try:
            p = ((system + "\n") if system else "") + (prompt or "")
            pb = list(p.encode("utf-8"))[-self.block_size:]
            idx = torch.tensor([pb], dtype=torch.long, device=self.dev)
            out = self.m.generate(idx, int(max_tokens), temp=max(0.1, temperature))[0].tolist()
            gen = bytes(out[len(pb):]).decode("utf-8", errors="replace")
            return LLMResponse(self.backend_id, self.model, gen, (time.time() - t0) * 1000)
        except Exception as e:
            return LLMResponse(self.backend_id, self.model, "", (time.time() - t0) * 1000,
                               error=str(e))


def main(steps: int, n_abstracts: int, kr_repeat: int) -> int:
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(42)
    data = np.frombuffer(build_corpus(n_abstracts, kr_repeat), dtype=np.uint8).astype(np.int64)
    print(f"corpus bytes={len(data):,} device={dev}")
    n = int(0.9 * len(data))
    tr, va = torch.tensor(data[:n]), torch.tensor(data[n:])
    bs, blk = 16, 256
    model = GPT(block_size=blk).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"params={nparam/1e6:.1f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def batch(split):
        d = tr if split == "tr" else va
        ix = torch.randint(len(d) - blk - 1, (bs,))
        x = torch.stack([d[i:i + blk] for i in ix]).to(dev)
        y = torch.stack([d[i + 1:i + 1 + blk] for i in ix]).to(dev)
        return x, y

    model.train()
    for step in range(1, steps + 1):
        x, y = batch("tr")
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                vx, vy = batch("va"); _, vl = model(vx, vy)
            model.train()
            print(f"step {step:5d}  train {loss.item():.3f}  val {vl.item():.3f}")

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "config": {"vocab": 256, "n_embd": 384, "n_head": 6,
                           "n_layer": 6, "block_size": blk}},
               OUT / "aria_torch_lm.pt")
    # sample from a domain prompt
    model.eval()
    prompt = "인플루엔자 표본감시는".encode("utf-8")
    idx = torch.tensor([list(prompt)], dtype=torch.long, device=dev)
    out = model.generate(idx, 200)[0].tolist()
    sample = bytes(out).decode("utf-8", errors="replace")
    (OUT / "sample.txt").write_text(sample, encoding="utf-8")
    meta = {"params_M": round(nparam / 1e6, 2), "corpus_bytes": int(len(data)),
            "steps": steps, "device": dev, "final_train_loss": round(loss.item(), 3),
            "n_abstracts": n_abstracts, "kr_repeat": kr_repeat,
            "honest_note": "small from-scratch torch GPT; WEAK demonstration LM, "
                           "not a usable advisory model (pretrained backends serve that)."}
    (OUT / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SAMPLE:", sample[:200].replace("\n", " "))
    print(f"-> {OUT}/aria_torch_lm.pt")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--abstracts", type=int, default=2500)
    ap.add_argument("--kr-repeat", type=int, default=120)
    args = ap.parse_args()
    raise SystemExit(main(args.steps, args.abstracts, args.kr_repeat))
