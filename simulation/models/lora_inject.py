"""lora_inject.py — generic hand-rolled LoRA (peft 불필요, 순수 torch).

#2 TiRex-LoRA의 기반 (NEW_MODEL_IDEAS 가속 #3 / 강점 흡수 전이+적응):
- 사전학습 backbone에 **저차원(rank r) 어댑터만** 학습 → full fine-tune(과적합·heavy) 회피 = 소표본 안전·경량.
- peft 0.19.1이 xLSTM custom arch 자동지원 불확실 → **순수 torch로 어떤 nn.Linear든 일반 주입.**

API:
- ``LoRALinear(base, rank, alpha)``: frozen base + 저차원 어댑터 B@A (초기 ΔW=0, B=0).
- ``inject_lora(model, rank)``: 모든 nn.Linear → LoRALinear 교체, base freeze, 어댑터만 trainable.
- ``merge_all_lora(model)``: 어댑터를 base에 병합 → 추론 오버헤드 0 (가속 #3).

Performance: 어댑터 param = Σ rank·(in+out) ≪ full. base forward 비용 동일.
Side effects: model in-place 변경 (named_children 교체).
Caller responsibility: torch nn.Module. fine-tune 가능한 backbone(TiRex `_model` 등)에 주입.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """frozen base nn.Linear + 저차원 어댑터 (scaling·B@A). 초기 ΔW=0 (B=0 init)."""

    def __init__(self, base: nn.Linear, rank: int = 4, alpha: float = 8.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)                 # base freeze
        self.rank = int(rank)
        self.scaling = float(alpha) / float(rank)
        self.lora_A = nn.Parameter(torch.zeros(self.rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))   # A random, B=0 → 초기 ΔW=0

    # drop-in 호환: backbone 코드가 layer.weight/.bias/.in_features 직접 접근해도 crash 안 나게
    # (TiRex 등). forward(layer(x))는 LoRA delta 유지 — weight 직접 F.linear 하는 층만 LoRA 미적용.
    @property
    def weight(self):
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        delta = (x @ self.lora_A.t()) @ self.lora_B.t()         # (… , out)
        return out + self.scaling * delta

    @torch.no_grad()
    def merge(self) -> nn.Linear:
        """어댑터를 base weight에 병합 → 추론용 순수 nn.Linear (오버헤드 0, 가속 #3)."""
        merged = nn.Linear(self.base.in_features, self.base.out_features,
                           bias=self.base.bias is not None)
        W = self.base.weight.data + self.scaling * (self.lora_B.data @ self.lora_A.data)
        merged.weight.data.copy_(W)
        if self.base.bias is not None:
            merged.bias.data.copy_(self.base.bias.data)
        return merged


def inject_lora(model: nn.Module, rank: int = 4, alpha: float = 8.0,
                min_features: int = 8) -> tuple[nn.Module, int]:
    """model의 모든 nn.Linear를 LoRALinear로 교체. base freeze, 어댑터만 trainable.

    Args: rank·alpha = LoRA HP. min_features = 이 미만 Linear는 스킵(작은 head 보존).
    Returns: (model, trainable_param_count). collect-then-apply (재-wrap·mutation 함정 회피).
    """
    to_replace = []
    for _, module in model.named_modules():
        if isinstance(module, LoRALinear):
            continue                                # 이미 wrap된 것의 base 재-wrap 방지
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear) and min(child.in_features, child.out_features) >= min_features:
                to_replace.append((module, child_name, child))
    for module, child_name, child in to_replace:
        setattr(module, child_name, LoRALinear(child, rank=rank, alpha=alpha))

    n_train = 0
    for n, p in model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            p.requires_grad_(True); n_train += p.numel()
        else:
            p.requires_grad_(False)                 # base 전체(LoRA 외) freeze
    return model, n_train


def merge_all_lora(model: nn.Module) -> nn.Module:
    """모든 LoRALinear를 병합된 nn.Linear로 교체 (추론 오버헤드 0). in-place."""
    to_merge = []
    for _, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, LoRALinear):
                to_merge.append((module, child_name, child))
    for module, child_name, child in to_merge:
        setattr(module, child_name, child.merge())
    return model
