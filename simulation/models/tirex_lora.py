"""tirex_lora.py — TiRex(35M xLSTM foundation, 현 rolling rel-WIS 1위) + LoRA 적응 (#2 엔진).

NEW_MODEL_IDEAS #2: frozen TiRex backbone + 저차원 LoRA 어댑터(rank 4-8)만 학습 = 소표본 안전·경량.
핵심: TiRex ``_forecast_tensor``(미분가능, (1,9,1)=9분위) 로 fine-tune (``_forecast_quantiles`` 는
no_grad 라 학습 불가 — 그래서 _forecast_tensor 사용). ``forecast`` (no_grad) 로 추론.

Performance: fit = epochs × (n/stride) 회 forward+backward (LoRA 320K param만). CPU.
Caller responsibility: y ≥ 0. USES_FEATURES=False (y-series 사용, X 무시 — foundation). rolling eval.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

log = logging.getLogger(__name__)


class TiRexLoraForecaster(BaseForecaster):
    """TiRex foundation + LoRA fine-tune. y-series rolling 1-step.

    Args: rank·alpha=LoRA HP. epochs·lr·stride=경량 fine-tune. max_context=TiRex 컨텍스트.
    """

    USES_FEATURES = False
    meta = ModelMeta(
        name="TiRex-LoRA",
        category="foundation",
        level=16,
        min_data=60,
        description="TiRex(35M xLSTM, rolling 1위) + LoRA 어댑터 fine-tune (_forecast_tensor 미분경로). "
                    "frozen backbone + rank-r 만 학습 = 소표본 경량 적응. rolling 1-step.",
        requires_gpu=False,
        dependencies=["tirex", "torch"],
    )

    def __init__(self, rank: int = 4, alpha: float = 8.0, epochs: int = 4,
                 lr: float = 1e-4, max_context: int = 256, min_ctx: int = 52,
                 stride: int = 2, val_k: int = 20, grad_clip: float = 1.0,
                 repo_id: str = "NX-AI/TiRex"):
        super().__init__()
        self.rank = int(rank); self.alpha = float(alpha)
        self.epochs = int(epochs); self.lr = float(lr)
        self.max_context = int(max_context); self.min_ctx = int(min_ctx)
        self.stride = int(stride); self.val_k = int(val_k)
        self.grad_clip = float(grad_clip); self.repo_id = repo_id
        self._m = None
        self._train_y: Optional[np.ndarray] = None
        self._use_lora = False; self._val_base = None; self._val_tuned = None

    def _val_mae(self, y, val_idx) -> float:
        import torch
        errs = []
        with torch.no_grad():
            for t in val_idx:
                ctx = torch.tensor(y[max(0, t - self.max_context):t], dtype=torch.float32).unsqueeze(0)
                _q, m = self._m.forecast(context=ctx, prediction_length=1)
                errs.append(abs(float(np.asarray(m).ravel()[0]) - float(y[t])))
        return float(np.mean(errs)) if errs else float("inf")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "TiRexLoraForecaster":
        import torch
        from tirex import load_model
        from simulation.models.lora_inject import inject_lora

        y = np.asarray(y_train, dtype=float).ravel()
        self._train_y = y
        n = len(y)
        val_k = min(self.val_k, n // 5)
        val_idx = list(range(n - val_k, n))                  # do-no-harm 검증용 tail
        self._m = load_model(self.repo_id, device="cpu")
        base_val = self._val_mae(y, val_idx)                 # zero-shot 기준선

        self._m, n_tr = inject_lora(self._m, rank=self.rank, alpha=self.alpha)   # frozen base + LoRA만
        opt = torch.optim.Adam([p for p in self._m.parameters() if p.requires_grad], lr=self.lr)
        trainable = [p for p in self._m.parameters() if p.requires_grad]
        global_scale = max(float(np.std(y)), 1.0)            # robust scale floor (flat ctx 폭발 방지)
        self._m.train()
        for _ep in range(self.epochs):
            for t in range(self.min_ctx, n - val_k, self.stride):
                ctx_raw = y[max(0, t - self.max_context):t]
                scale = max(float(np.std(ctx_raw)), 0.5 * global_scale, 1.0)     # ★robust floor
                ctx = torch.tensor(ctx_raw, dtype=torch.float32).unsqueeze(0)
                try:
                    pred = self._m._forecast_tensor(ctx, prediction_length=1)    # 미분가능 (1,9,1)
                    med = pred.reshape(pred.shape[0], pred.shape[1], -1)[:, pred.shape[1] // 2, 0]
                    loss = (((med - float(y[t])) / scale) ** 2).mean()           # ★정규화 loss
                    if not torch.isfinite(loss):
                        continue                              # ★NaN/inf loss skip (발산 차단)
                    opt.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(trainable, self.grad_clip)    # ★grad clip
                    opt.step()
                except Exception:
                    continue
        self._m.eval()

        # ★do-no-harm: fine-tuned가 val서 zero-shot보다 나쁘면 LoRA 가중치 자체를 0으로 (NaN 제거 → 진짜 zero-shot)
        tuned_val = self._val_mae(y, val_idx)
        self._use_lora = bool(np.isfinite(tuned_val) and tuned_val < base_val)
        self._val_base, self._val_tuned = base_val, tuned_val
        if not self._use_lora:
            with torch.no_grad():
                for mod in self._m.modules():
                    if hasattr(mod, "lora_A") and hasattr(mod, "lora_B"):
                        mod.lora_A.zero_(); mod.lora_B.zero_()   # ★param=0 (scaling=0은 0×NaN=NaN 버그)
                        mod.scaling = 0.0
        self._fitted = True
        log.info(f"  [TiRex-LoRA] base_val={base_val:.3f} tuned_val={tuned_val:.3f} "
                 f"use_lora={self._use_lora} (LoRA {n_tr} param)")
        return self

    def predict(self, X_test: np.ndarray, y_observed=None, **kwargs) -> np.ndarray:
        import torch
        if not self._fitted or self._m is None:
            raise RuntimeError("TiRex-LoRA: fit() 먼저")     # fail-loud
        n = len(X_test)
        obs = (np.asarray(y_observed, dtype=float).ravel() if y_observed is not None
               else np.full(n, np.nan))
        preds = np.empty(n, dtype=float)
        with torch.no_grad():
            for i in range(n):
                hist = (np.concatenate([self._train_y, obs[:i]]) if (y_observed is not None and i > 0)
                        else self._train_y)
                ctx = torch.tensor(hist[-self.max_context:], dtype=torch.float32).unsqueeze(0)
                try:
                    _q, mean = self._m.forecast(context=ctx, prediction_length=1)   # no_grad 추론
                    preds[i] = float(np.asarray(mean).ravel()[0])
                except Exception:
                    preds[i] = float(hist[-1])              # fallback = persistence
        return np.clip(np.nan_to_num(preds, nan=0.0), 0.0, None)


def register_tirex_lora() -> None:
    """명시 등록 (검증·완성 후 호출 — 자동등록 회피)."""
    try:
        REGISTRY.register(TiRexLoraForecaster)
        log.info("[tirex_lora] TiRexLoraForecaster 등록됨")
    except Exception as e:
        log.debug(f"[tirex_lora] 등록 skip: {e}")
