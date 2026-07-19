"""
simulation/models/modern_ts/itransformer.py
============================================
iTransformer (Inverted Transformer) -- Level 16 [실험적]

Treats each variable (feature) as a token, not each timestep.
Learns inter-variable interactions via Transformer attention.
Particularly effective for multivariate time series.
"""

from __future__ import annotations

import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

__all__ = ["iTransformerForecaster"]


class _iTransformerNet:
    """iTransformer: treat each VARIABLE as a token, not each timestep."""

    @staticmethod
    def build(n_features: int, seq_len: int = 8,
              d_model: int = 24, n_heads: int = 4,
              n_layers: int = 1, dim_ff: int = 48,
              dropout: float = 0.3, init: str = "default"):
        """: 메모리/시간 축소판 — p=309·seq=12·d_model=32·2-layer 구성에서
 subprocess timeout (1800s) 걸리던 문제 해결:
 - SEQ_LEN 12 → 8 (약 33% 단축)
 - d_model 32 → 24
 - n_layers 2 → 1
 - head: flatten (309×32=9888) → **mean pool over feature tokens** (d_model)
 헤드 파라미터 수가 9888×32=316k → 24×24=576 으로 감소 (약 550×).

 (P0-D): Optuna 가 d_model / n_heads / n_layers / dim_ff /
 dropout / init 를 조정하도록 확장.
 """
        import torch
        import torch.nn as nn
        from simulation.models.dl_models import _apply_weight_init

        class iTransformerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_features = n_features

                # Each feature's time series → token embedding
                self.embed = nn.Linear(seq_len, d_model)

                # Transformer encoder operates on feature tokens
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model,
                    nhead=n_heads,
                    dim_feedforward=dim_ff,
                    dropout=dropout,
                    batch_first=True,
                )
                self.transformer = nn.TransformerEncoder(
                    encoder_layer, num_layers=n_layers,
                )

                # : flatten → mean-pool. n_features=309 에서
                # d_model*n_features = 9888 차원 linear layer 는 파라미터 과잉.
                self.head = nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(d_model, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, 1),
                )

            def forward(self, x):
                # x: (batch, seq_len, n_features)
                # INVERT: treat features as tokens
                # → (batch, n_features, seq_len)
                x_inv = x.transpose(1, 2)

                # Embed each feature's time series
                tokens = self.embed(x_inv)  # (batch, n_features, d_model)

                # Transformer: inter-feature attention
                out = self.transformer(tokens)  # (batch, n_features, d_model)

                # : mean pool over feature dim
                pooled = out.mean(dim=1)  # (batch, d_model)
                return self.head(pooled)

        m = iTransformerModel()
        _apply_weight_init(m, init)
        return m


class iTransformerForecaster(BaseForecaster):
    """iTransformer -- Inverted Transformer for Time Series.

    [실험적] 핵심 아이디어: 각 변수(feature)를 token으로 취급.
    변수 간 상호작용을 Transformer attention으로 학습.
    다변량 시계열에서 특히 효과적.
    """

    meta = ModelMeta(
        name="iTransformer",
        category="dl",
        level=16,
        min_data=120,
        description="[실험적] iTransformer. 변수를 token으로 반전시킨 Transformer. 다변량 특화.",
        dependencies=["torch"],
    )

    # : SEQ_LEN 12 → 8 (timeout 방어)
    SEQ_LEN = 8

    def __init__(self):
        super().__init__()
        self._model = None
        self._scaler_X = None
        self._scaler_y = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> iTransformerForecaster:
        """(P0-D): Optuna HPO 내장.

 timeout 방어를 위해 d_model ≤ 64, dim_ff ≤ 256 으로 search 범위 억제.
 p≈309 feature token 에 transformer 를 돌리므로 d_model 이 커지면
 O(p²·d) attention 이 급격히 무거워진다.
 """
        from sklearn.preprocessing import StandardScaler
        from simulation.models.dl_models import _make_sequences, _train_loop, lag_backbone_seq
        from simulation.models._optuna_budget import get_trials
        from simulation.models._optuna_torch import (
            suggest_training_hp, run_optuna_loop, INITS,
        )

        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()
        X_s = self._scaler_X.fit_transform(X_train)
        y_s = self._scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        self._y_train_max = float(np.max(y_train)) if len(y_train) else 100.0  # G-289 외삽 cap

        # G-319d: lag backbone(과거 y) 입력으로 입력버그 회복(n_features=1 closure 전파, lookback 불변).
        _lag_seq, self._lag_idx = lag_backbone_seq(X_s, kwargs.get("feature_names"), self.SEQ_LEN)
        if self._lag_idx is not None:
            X_seq, y_seq, n_features = _lag_seq, y_s, 1
        else:
            X_seq, y_seq = _make_sequences(X_s, y_s, self.SEQ_LEN)
            n_features = X_s.shape[1]
        n_trials = get_trials("iTransformer", default=0)

        def _static_defaults():
            self._model = _iTransformerNet.build(
                n_features=n_features, seq_len=self.SEQ_LEN,
                d_model=24, n_heads=4, n_layers=1,
                dim_ff=48, dropout=0.3, init="default",
            )
            _train_loop(
                self._model, X_seq, y_seq,
                epochs=150, lr=5e-4, patience=20,
                weight_decay=5e-4,
                augment=True, augment_factor=2,
            )

        def _objective(trial):
            import gc
            n_heads = trial.suggest_categorical("n_heads", [2, 4, 8])
            d_raw = trial.suggest_int("d_model_raw", 8, 64, log=True)
            d_model = ((d_raw + n_heads - 1) // n_heads) * n_heads
            n_layers_tf = trial.suggest_int("n_layers_tf", 1, 2)
            dim_ff = trial.suggest_int("dim_ff", 16, 256, log=True)
            dropout = trial.suggest_float("dropout", 0.0, 0.5)
            init_t = trial.suggest_categorical("init", INITS)
            tr = suggest_training_hp(trial)

            model = _iTransformerNet.build(
                n_features=n_features, seq_len=self.SEQ_LEN,
                d_model=d_model, n_heads=n_heads,
                n_layers=n_layers_tf, dim_ff=dim_ff,
                dropout=dropout, init=init_t,
            )
            neg_r2 = _train_loop(
                model, X_seq, y_seq,
                epochs=100, lr=tr["lr"], batch_size=tr["batch_size"],
                patience=20, weight_decay=tr["weight_decay"],
                augment=True, augment_factor=tr["augment_factor"],
                trial=trial,
                optimizer_type=tr["optimizer"], loss_type=tr["loss"],
                return_r2=True,
            )
            del model
            gc.collect()
            return neg_r2

        best, _ = run_optuna_loop("iTransformer", _objective, n_trials, _static_defaults)
        if best:
            n_heads = best.get("n_heads", 4)
            d_model = ((best.get("d_model_raw", 24) + n_heads - 1) // n_heads) * n_heads
            self._model = _iTransformerNet.build(
                n_features=n_features, seq_len=self.SEQ_LEN,
                d_model=d_model, n_heads=n_heads,
                n_layers=best.get("n_layers_tf", 1),
                dim_ff=best.get("dim_ff", 48),
                dropout=best.get("dropout", 0.3),
                init=best.get("init", "default"),
            )
            _train_loop(
                self._model, X_seq, y_seq,
                epochs=150, lr=best.get("lr", 5e-4),
                batch_size=best.get("batch_size", 32),
                patience=20, weight_decay=best.get("weight_decay", 5e-4),
                augment=True, augment_factor=best.get("augment_factor", 2),
                optimizer_type=best.get("optimizer", "adamw"),
                loss_type=best.get("loss", "mse"),
            )
        self._fitted = True
        return self

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        from simulation.models.dl_models import _predict_torch, lag_backbone_from_idx

        X_s = self._scaler_X.transform(X_test)
        if getattr(self, "_lag_idx", None) is not None:
            X_seq = lag_backbone_from_idx(X_s, self._lag_idx, self.SEQ_LEN)  # G-319d
        else:
            preds = []
            for i in range(len(X_s)):
                start = max(0, i + 1 - self.SEQ_LEN)
                seq = X_s[start:i + 1]
                if len(seq) < self.SEQ_LEN:
                    pad = np.zeros((self.SEQ_LEN - len(seq), X_s.shape[1]))
                    seq = np.vstack([pad, seq])
                preds.append(seq)
            X_seq = np.array(preds)
        pred_s = _predict_torch(self._model, X_seq)
        _pred = np.maximum(
            self._scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel(), 0
        )
        from simulation.models.safety import apply_extrapolation_cap  # G-289
        return apply_extrapolation_cap(_pred, getattr(self, "_y_train_max", None))
