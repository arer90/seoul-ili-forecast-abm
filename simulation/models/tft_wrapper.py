"""
simulation/models/tft_wrapper.py
================================
TFT (Temporal Fusion Transformer) -- BaseForecaster 호환 래퍼.

기존 analysis/forecasting/tft_model.py를 BaseForecaster 인터페이스로 래핑.
데이터 충분 시(500주+) 사용. ILI rate 전용.

TFT는 attention 기반 해석 가능 시계열 예측 모델.
- Variable Selection Network: 피처 중요도 자동 학습
- Multi-head Attention: 시간 패턴 포착
- Gating: 불필요 정보 억제

(P0-D): Optuna HPO 내장.
 d_model / n_heads / n_layers / dim_ff / dropout / lr / wd / optimizer /
 loss / activation / norm / init 모두 trial 탐색. 트라이얼 예산은
 ``_optuna_budget.get_trials('TFT')`` 로 조절. 0 이면 static config 로 폴백.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

log = logging.getLogger(__name__)


def _build_light_tft(n_features: int, seq_len: int, *,
                     d_model: int = 32, n_heads: int = 4,
                     dropout: float = 0.1, activation: str = "relu",
                     norm: str = "layer", init: str = "default",
                     n_enc_layers: int = 1, dim_ff: int = 64):
    """LightTFT builder (VSN + LSTM + MHA + gated FC head).

    `d_model` is rounded up to a multiple of `n_heads`.
    """
    import torch
    import torch.nn as nn
    from simulation.models.dl_models import (
        _get_activation_fn, _get_norm_layer, _apply_weight_init,
    )

    # d_model must be divisible by n_heads
    d_model = max(n_heads, ((d_model + n_heads - 1) // n_heads) * n_heads)

    class LightTFT(nn.Module):
        def __init__(self):
            super().__init__()
            self.d_model = d_model

            self.vsn_weights = nn.Sequential(
                nn.Linear(n_features, n_features),
                nn.Softmax(dim=-1),
            )
            self.input_proj = nn.Linear(n_features, d_model)

            self.lstm = nn.LSTM(
                d_model, d_model, num_layers=n_enc_layers,
                batch_first=True, dropout=dropout if n_enc_layers > 1 else 0.0,
            )
            self.attn = nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True,
            )
            self.attn_norm = _get_norm_layer(norm, d_model)

            self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
            self.fc = nn.Sequential(
                nn.Linear(d_model, max(4, dim_ff // 2)),
                _get_activation_fn(activation),
                nn.Dropout(dropout),
                nn.Linear(max(4, dim_ff // 2), 1),
            )

        def forward(self, x):
            w = self.vsn_weights(x)
            x = x * w
            x = self.input_proj(x)
            lstm_out, _ = self.lstm(x)
            attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
            # LayerNorm expects last dim == normalized_shape
            x = self.attn_norm(lstm_out + attn_out) if norm == "layer" else (lstm_out + attn_out)
            last = x[:, -1, :]
            g = self.gate(last)
            return self.fc(last * g)

    m = LightTFT()
    _apply_weight_init(m, init)
    return m


class TFTForecaster(BaseForecaster):
    """Temporal Fusion Transformer -- attention 기반 해석 가능 예측.

 : Optuna 를 통해 d_model/dim_ff/dropout/lr/wd/activation/norm/init/
 optimizer/loss 를 탐색. 예산 0 이면 기존 고정 HP (d_model=32, n_heads=4,
 dropout=0.1, lr=5e-4) 로 실행.
 """

    meta = ModelMeta(
        name="TFT",
        category="dl",
        level=14,
        min_data=120,
        description="Temporal Fusion Transformer. attention 기반 해석 가능 시계열 예측.",
        requires_gpu=False,
        dependencies=["torch"],
    )

    SEQ_LEN = 12  # TFT는 좀 더 긴 lookback

    def __init__(self):
        super().__init__()
        self._model = None
        self._scaler_X = None
        self._scaler_y = None
        self._best_params: dict | None = None

    def _make_sequences(self, X, y=None):
        Xs, ys = [], []
        for i in range(self.SEQ_LEN, len(X)):
            Xs.append(X[i - self.SEQ_LEN:i])
            if y is not None:
                ys.append(y[i])
        return np.array(Xs), np.array(ys) if y is not None else None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> TFTForecaster:
        from sklearn.preprocessing import StandardScaler
        from simulation.models.dl_models import _train_loop
        from simulation.models._optuna_budget import get_trials
        from simulation.models._optuna_torch import (
            suggest_transformer_hp, suggest_training_hp, run_optuna_loop,
        )

        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()
        X_s = self._scaler_X.fit_transform(X_train)
        y_s = self._scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()

        X_seq, y_seq = self._make_sequences(X_s, y_s)
        n_features = X_s.shape[1]

        n_trials = get_trials("TFT", default=0)

        def _static_defaults():
            self._model = _build_light_tft(
                n_features, self.SEQ_LEN,
                d_model=32, n_heads=4, dropout=0.1,
                activation="relu", norm="layer", init="default",
                n_enc_layers=1, dim_ff=64,
            )
            _train_loop(self._model, X_seq, y_seq, epochs=200, lr=5e-4, patience=25)

        def _objective(trial):
            import gc
            tf = suggest_transformer_hp(
                trial, min_d_model=8, max_d_model=512, max_layers=3,
            )
            tr = suggest_training_hp(trial)
            model = _build_light_tft(
                n_features, self.SEQ_LEN,
                d_model=tf["d_model"], n_heads=tf["n_heads"],
                dropout=tf["dropout"], activation=tf["activation"],
                norm=tf["norm"], init=tf["init"],
                n_enc_layers=tf["n_layers_tf"], dim_ff=tf["dim_ff"],
            )
            neg_r2 = _train_loop(
                model, X_seq, y_seq,
                epochs=150, lr=tr["lr"], batch_size=tr["batch_size"],
                patience=20, weight_decay=tr["weight_decay"],
                augment=True, augment_factor=tr["augment_factor"],
                trial=trial,
                optimizer_type=tr["optimizer"],
                loss_type=tr["loss"],
                return_r2=True,
            )
            del model
            gc.collect()
            return neg_r2

        best, _ = run_optuna_loop(
            "TFT", _objective, n_trials, _static_defaults,
        )
        if best:
            self._best_params = best
            self._model = _build_light_tft(
                n_features, self.SEQ_LEN,
                d_model=best.get("d_model_raw", 32),
                n_heads=best.get("n_heads", 4),
                dropout=best.get("dropout", 0.1),
                activation=best.get("activation", "relu"),
                norm=best.get("norm", "layer"),
                init=best.get("init", "default"),
                n_enc_layers=best.get("n_layers_tf", 1),
                dim_ff=best.get("dim_ff", 64),
            )
            _train_loop(
                self._model, X_seq, y_seq,
                epochs=300, lr=best.get("lr", 5e-4),
                batch_size=best.get("batch_size", 32),
                patience=30,
                weight_decay=best.get("weight_decay", 1e-4),
                augment=True, augment_factor=best.get("augment_factor", 3),
                optimizer_type=best.get("optimizer", "adamw"),
                loss_type=best.get("loss", "mse"),
            )
        self._fitted = True
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        X_s = self._scaler_X.transform(X_test)

        preds = []
        for i in range(len(X_s)):
            start = max(0, i + 1 - self.SEQ_LEN)
            seq = X_s[start:i + 1]
            if len(seq) < self.SEQ_LEN:
                pad = np.zeros((self.SEQ_LEN - len(seq), X_s.shape[1]))
                seq = np.vstack([pad, seq])
            preds.append(seq)
        X_seq = np.array(preds)

        from simulation.models.dl_models import _predict_torch
        pred_s = _predict_torch(self._model, X_seq)
        return np.maximum(
            self._scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel(), 0
        )


# ── 등록 ──
# 2026-05-12 (사용자 명시): -pf 정책 — PfTFTForecaster 가 "TFT" 이름 점유.
# Custom TFT wrapper class 정의 보존, REGISTRY 등록 차단.
# REGISTRY.register(TFTForecaster)
