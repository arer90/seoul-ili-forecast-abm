"""
simulation/models/modern_ts/tide.py
====================================
TiDE (Time-series Dense Encoder) -- Level 29

Google Research, 2023. Dense MLP encoder-decoder + residual connection.
SOTA for ILI prediction (JMIR 2025).
More parameter-efficient and stable than Transformers with small samples (~341 weeks).
"""

from __future__ import annotations

import logging

import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta, REGISTRY

log = logging.getLogger(__name__)

__all__ = ["TiDEForecaster"]


class _TiDENet:
    """TiDE: encoder-decoder with dense MLPs for long-term forecasting."""

    @staticmethod
    def build(n_features: int, lookback: int = 12,
              hidden_size: int = 128, decoder_size: int = 64,
              dropout: float = 0.25, num_encoder_layers: int = 2,
              activation: str = "gelu", init: str = "default"):
        """(P0-D): activation / init 도 hp 로 수신."""
        import torch
        import torch.nn as nn
        from simulation.models.dl_models import (
            _get_activation_fn, _apply_weight_init,
        )

        class TiDEModel(nn.Module):
            def __init__(self):
                super().__init__()
                flat_dim = lookback * n_features
                # Feature projection (reduce dimensionality)
                self.feature_proj = nn.Linear(n_features, min(32, n_features))
                proj_dim = lookback * min(32, n_features)

                # Dense Encoder
                enc_layers = []
                in_dim = proj_dim
                for _ in range(num_encoder_layers):
                    enc_layers.extend([
                        nn.Linear(in_dim, hidden_size),
                        _get_activation_fn(activation),
                        nn.LayerNorm(hidden_size),
                        nn.Dropout(dropout),
                    ])
                    in_dim = hidden_size
                self.encoder = nn.Sequential(*enc_layers)

                # Dense Decoder
                self.decoder = nn.Sequential(
                    nn.Linear(hidden_size, decoder_size),
                    _get_activation_fn(activation),
                    nn.Dropout(dropout),
                    nn.Linear(decoder_size, 1),
                )

                # Residual connection (temporal)
                self.residual_proj = nn.Linear(lookback, 1)

            def forward(self, x):
                # x: (batch, seq_len, features)
                batch_size = x.shape[0]

                # Feature projection
                x_proj = self.feature_proj(x)  # (batch, seq, proj_dim)
                x_flat = x_proj.reshape(batch_size, -1)

                # Encoder → Decoder
                encoded = self.encoder(x_flat)
                decoded = self.decoder(encoded)  # (batch, 1)

                # Residual: mean across features, project time
                x_mean = x.mean(dim=2)  # (batch, seq_len)
                residual = self.residual_proj(x_mean)  # (batch, 1)

                return decoded + residual

        m = TiDEModel()
        _apply_weight_init(m, init)
        return m


class TiDEForecaster(BaseForecaster):
    r"""TiDE -- Time-series Dense Encoder (Google Research, 2023).

    ILI 예측에서 LSTM, N-BEATS, TFT, Transformer 대비 최고 성능 (JMIR 2025).
    Dense MLP encoder-decoder + residual connection.
    소표본(~341주)에서도 Transformer보다 안정적 (파라미터 효율적).
    """

    meta = ModelMeta(
        name="TiDE",
        category="dl",
        level=29,
        min_data=80,
        description="TiDE. Dense encoder-decoder, ILI SOTA. Transformer 대비 파라미터 효율적.",
        dependencies=["torch"],
    )

    SEQ_LEN = 12

    def __init__(self):
        super().__init__()
        self._models = []
        self._model = None
        self._scaler_X = None
        self._scaler_y = None
        self._y_train_max = None
        self._y_train_std = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> TiDEForecaster:
        """(P0-D): Optuna HPO 내장 (search → 3-seed ensemble 재학습)."""
        import torch
        from sklearn.preprocessing import StandardScaler
        from simulation.models.dl_models import _make_sequences, _train_loop
        from simulation.models._optuna_budget import get_trials
        from simulation.models._optuna_torch import (
            suggest_training_hp, run_optuna_loop,
            UNIT_MIN_DEFAULT, UNIT_MAX_TS_DL, INITS,
        )

        self._scaler_X = StandardScaler()
        self._scaler_y = StandardScaler()
        X_s = self._scaler_X.fit_transform(X_train)
        y_s = self._scaler_y.fit_transform(y_train.reshape(-1, 1)).ravel()
        self._y_train_max = float(np.max(y_train)) if len(y_train) else 100.0  # G-289 외삽 cap
        self._y_train_std = float(np.std(y_train)) if len(y_train) > 1 else 1.0

        X_seq, y_seq = _make_sequences(X_s, y_s, self.SEQ_LEN)
        n_features = X_s.shape[1]

        # COVID-era curriculum weighting
        n = len(y_train)
        sw = np.ones(n)
        sw[int(n * 0.6):] = 2.0
        sw_seq = sw[self.SEQ_LEN:][:len(y_seq)]

        n_trials = get_trials("TiDE", default=0)

        # best_hp holder (Optuna 가 없어도 static 기본값 사용)
        bp = {
            "hidden_size": 128, "decoder_size": 64, "dropout": 0.25,
            "num_encoder_layers": 2, "init": "default", "lr": 3e-4,
            "weight_decay": 1e-4, "batch_size": 32, "augment_factor": 4,
            "optimizer": "adamw", "loss": "mse",
        }

        def _static_defaults():
            # 기본값 유지 — bp 수정 안 함
            pass

        def _objective(trial):
            import gc
            # FIX: TiDE encoder/decoder stacks → hidden × lookback OOM. cap 512.
            hidden = trial.suggest_int("hidden_size", UNIT_MIN_DEFAULT, UNIT_MAX_TS_DL, log=True)
            decoder = trial.suggest_int("decoder_size", UNIT_MIN_DEFAULT, UNIT_MAX_TS_DL, log=True)
            dropout = trial.suggest_float("dropout", 0.0, 0.5)
            n_enc = trial.suggest_int("num_encoder_layers", 1, 4)
            init_t = trial.suggest_categorical("init", INITS)
            tr = suggest_training_hp(trial)

            model = _TiDENet.build(
                n_features=n_features, lookback=self.SEQ_LEN,
                hidden_size=hidden, decoder_size=decoder, dropout=dropout,
                num_encoder_layers=n_enc, activation="gelu", init=init_t,
            )
            neg_r2 = _train_loop(
                model, X_seq, y_seq,
                epochs=150, lr=tr["lr"], batch_size=tr["batch_size"],
                patience=25, weight_decay=tr["weight_decay"],
                augment=True, augment_factor=tr["augment_factor"],
                trial=trial,
                optimizer_type=tr["optimizer"], loss_type=tr["loss"],
                return_r2=True,
            )
            del model
            gc.collect()
            return neg_r2

        best, _ = run_optuna_loop("TiDE", _objective, n_trials, _static_defaults)
        if best:
            bp.update({
                "hidden_size": best.get("hidden_size", 128),
                "decoder_size": best.get("decoder_size", 64),
                "dropout": best.get("dropout", 0.25),
                "num_encoder_layers": best.get("num_encoder_layers", 2),
                "init": best.get("init", "default"),
                "lr": best.get("lr", 3e-4),
                "weight_decay": best.get("weight_decay", 1e-4),
                "batch_size": best.get("batch_size", 32),
                "augment_factor": best.get("augment_factor", 4),
                "optimizer": best.get("optimizer", "adamw"),
                "loss": best.get("loss", "mse"),
            })

        # 3-seed ensemble with best HP
        self._models = []
        for seed in [42, 2024, 31415]:
            torch.manual_seed(seed)
            model = _TiDENet.build(
                n_features=n_features, lookback=self.SEQ_LEN,
                hidden_size=bp["hidden_size"], decoder_size=bp["decoder_size"],
                dropout=bp["dropout"], num_encoder_layers=bp["num_encoder_layers"],
                activation="gelu", init=bp["init"],
            )
            _train_loop(
                model, X_seq, y_seq,
                epochs=500, lr=bp["lr"], batch_size=bp["batch_size"],
                patience=40, weight_decay=bp["weight_decay"],
                augment=True, augment_factor=bp["augment_factor"],
                sample_weights=sw_seq, curriculum_mode='linear',
                optimizer_type=bp["optimizer"], loss_type=bp["loss"],
            )
            self._models.append(model)

        self._model = self._models[0]
        self._fitted = True
        log.info(f"  [TiDE] 3-seed ensemble + curriculum 완료 (trials={n_trials})")
        return self

    def _cap_reference(self) -> float:
        """Return the original-unit y reference used for TiDE extrapolation caps."""
        y_max = getattr(self, "_y_train_max", None)
        try:
            if y_max is not None and np.isfinite(float(y_max)) and float(y_max) > 0:
                return float(y_max)
        except (TypeError, ValueError):
            pass
        try:
            if self._scaler_y is not None:
                mean = float(np.asarray(self._scaler_y.mean_).ravel()[0])
                scale = float(np.asarray(self._scaler_y.scale_).ravel()[0])
                fallback = mean + 3.0 * scale
                if np.isfinite(fallback) and fallback > 0:
                    return fallback
        except Exception:
            pass
        return 100.0 / 1.5

    def _cap_original_units(self, values, mult: float = 1.5, floor: float = 100.0) -> np.ndarray:
        from simulation.models.safety import apply_extrapolation_cap

        return apply_extrapolation_cap(values, self._cap_reference(), mult=mult, floor=floor)

    def predict(self, X_test: np.ndarray, **kwargs) -> np.ndarray:
        from simulation.models.dl_models import _predict_torch

        X_s = self._scaler_X.transform(X_test)
        seqs = []
        for i in range(len(X_s)):
            start = max(0, i + 1 - self.SEQ_LEN)
            seq = X_s[start:i + 1]
            if len(seq) < self.SEQ_LEN:
                pad = np.zeros((self.SEQ_LEN - len(seq), X_s.shape[1]))
                seq = np.vstack([pad, seq])
            seqs.append(seq)
        X_seq = np.array(seqs)

        preds = [_predict_torch(m, X_seq) for m in self._models]
        pred_s = np.mean(preds, axis=0)
        _pred = np.maximum(
            self._scaler_y.inverse_transform(pred_s.reshape(-1, 1)).ravel(), 0
        )
        return self._cap_original_units(_pred)  # G-289

    def predict_interval(
        self, X_test: np.ndarray, alpha: float = 0.05, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prediction interval with the same original-unit cap as point forecasts."""
        if not self._fitted:
            raise RuntimeError("TiDE: fit() 먼저 호출")
        try:
            from scipy.stats import norm as _norm
            z = float(_norm.ppf(1.0 - alpha / 2.0))
        except Exception:
            z = 1.96
        point = self.predict(X_test)
        sigma = getattr(self, "_y_train_std", None)
        try:
            sigma = float(sigma)
        except (TypeError, ValueError):
            sigma = float("nan")
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = max(1.0, 0.1 * self._cap_reference())
        lower = np.maximum(point - z * sigma, 0.0)
        upper = point + z * sigma
        lower = self._cap_original_units(lower)
        upper = self._cap_original_units(upper)
        lower = np.minimum(lower, upper)
        return lower, upper
