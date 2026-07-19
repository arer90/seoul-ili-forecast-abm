"""
simulation/models/modern_ts/pf_models.py
=========================================
pytorch_forecasting 1.7.0 reference implementations .

기존 custom 모델들 (nbeats.py, nhits.py, tide.py, tft_wrapper.py) 과 **병행**
하여 A/B 비교와 재현성 벤치마크를 제공한다. 리뷰어가
`pip install pytorch-forecasting==1.7.0` 한 번으로 동일 결과를 재현 가능.

추가 모델:
 - PfTFTForecaster (name="TFT", Level 15) -- twin of tft_wrapper
 - PfNBeatsForecaster (name="N-BEATS", Level 11) -- twin of nbeats
 - PfNHiTSForecaster (name="N-HiTS", Level 12) -- twin of nhits
 - PfTiDEForecaster (name="TiDE", Level 29) -- twin of tide
 - PfRNNForecaster (name="RNN", Level 10) -- LSTM baseline (없던 category)
 - PfDeepARForecaster (name="DeepAR", Level 13) -- probabilistic (없던 category)

중요:
 - pf 1.7.0 은 `lightning.pytorch.LightningModule` 상속. Trainer 도 반드시
 `import lightning.pytorch as pl` -- `pytorch_lightning` 으로 감싸면
 isinstance check 가 실패한다.
 - NHiTS / TiDEModel 은 `add_relative_time_idx=False` 를 강제 (내부 assert).
 - NBeats 는 univariate (target only); covariate 무시.
 - DeepAR 는 probabilistic autoregressive -- NormalDistributionLoss 사용.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np

from simulation.models.base import BaseForecaster, ModelMeta
from simulation.config_global import GLOBAL  # SSOT (2026-05-28)
from simulation.utils.paths import get_results_dir

log = logging.getLogger(__name__)

__all__ = [
    "PfTFTForecaster",
    "PfNBeatsForecaster",
    "PfNHiTSForecaster",
    "PfTiDEForecaster",
    "PfRNNForecaster",
    "PfDeepARForecaster",
]


def _build_lightning_callbacks(checkpoint_dir: Path, history_sink: list) -> list:
    """Build val_loss early-stop/checkpoint callbacks plus epoch metric history."""
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    class _EpochMetricHistory(pl.Callback):
        def on_validation_epoch_end(self, trainer, pl_module) -> None:
            if trainer.sanity_checking:
                return
            record = {"epoch": int(trainer.current_epoch)}
            for key, value in trainer.callback_metrics.items():
                try:
                    if hasattr(value, "detach"):
                        value = value.detach().cpu()
                    record[str(key)] = float(value.item() if hasattr(value, "item") else value)
                except (TypeError, ValueError, RuntimeError) as exc:
                    log.warning("Lightning metric %s could not be serialized: %s", key, exc)
                    continue
            history_sink.append(record)

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            mode="min",
            strict=True,
            check_finite=True,
            verbose=False,
        ),
        ModelCheckpoint(
            dirpath=checkpoint_path,
            filename="best",
            monitor="val_loss",
            save_top_k=1,
            mode="min",
            save_last=False,
            enable_version_counter=False,
        ),
        _EpochMetricHistory(),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Base wrapper
# ═══════════════════════════════════════════════════════════════════════════


class _PfBase(BaseForecaster):
    """공통 wrapper: numpy (X, y) <-> pf TimeSeriesDataSet <-> Lightning Trainer.

    서브클래스는 `_build_model(training_ds)` 만 override 하면 된다.
    covariate 를 안 쓰는 univariate 모델 (NBeats) 은 USE_COVARIATES=False.
    NHiTS/TiDE 처럼 `add_relative_time_idx=False` 가 강제인 모델은
    ADD_RELATIVE_TIME_IDX=False 로 override.
    """

    # 서브클래스 override 가능한 knob
    MAX_ENCODER_LENGTH: int = 8
    MAX_PREDICTION_LENGTH: int = 1
    HIDDEN_SIZE: int = 16
    DROPOUT: float = 0.3
    EPOCHS: int = 80
    LR: float = 1e-3
    BATCH_SIZE: int = 32
    ADD_RELATIVE_TIME_IDX: bool = True
    USE_COVARIATES: bool = True

    def __init__(self):
        super().__init__()
        self._model = None
        self._training_ds = None
        self._df_train = None
        self._feat_names: Optional[list[str]] = None
        self._n_train: int = 0
        self._y_log_used: bool = False

    # ── hook for subclasses ──

    def _build_model(self, training_ds):
        raise NotImplementedError

    def _build_training_dataset(self, df):
        """default: USE_COVARIATES 분기. 서브클래스가 필요시 override."""
        from pytorch_forecasting import TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer

        common = dict(
            time_idx="time_idx",
            target="target",
            group_ids=["group_ids"],
            max_encoder_length=self.MAX_ENCODER_LENGTH,
            max_prediction_length=self.MAX_PREDICTION_LENGTH,
        )

        if self.USE_COVARIATES:
            return TimeSeriesDataSet(
                df,
                **common,
                static_categoricals=["group_ids"],
                time_varying_known_reals=self._feat_names,
                time_varying_unknown_reals=["target"],
                # transform-fix (2026-06-21): softplus/log1p coupling REMOVED — the single
                #   y-transform is now DATA-DRIVEN by the preproc Optuna search, not coupled to the
                #   normalizer here. transformation=None (plain group standardization) so there is
                #   no internal softplus∘(external transform) double-transform / expm1 blow-up on
                #   out-of-range peaks. (G-274's signed-target guard is moot: phase-13 supplies the
                #   single transform and the linear-space explosion-cap in predict is the backstop.)
                target_normalizer=GroupNormalizer(
                    groups=["group_ids"],
                    transformation=None,
                ),
                add_relative_time_idx=self.ADD_RELATIVE_TIME_IDX,
                add_target_scales=True,
            )
        else:
            # univariate
            return TimeSeriesDataSet(
                df,
                **common,
                time_varying_unknown_reals=["target"],
                target_normalizer=None,
            )

    # ── shared fit / predict ──

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            **kwargs) -> "_PfBase":
        import pandas as pd

        # transform-fix (2026-06-21): internal log1p REMOVED — fit on RAW y. The single y-transform
        #   is now DATA-DRIVEN by the preproc Optuna search; no internal log1p coupling. _y_log_used
        #   stays False so the predict path applies no inverse (only the linear-space cap backstop).
        self._y_log_used = False
        # G-180 P1 (2026-05-05): y_train 보존 (predict 시 linear-space explosion-cap 계산용)
        self._y_train_raw = np.asarray(y_train, dtype=np.float64).copy()
        y_fit = y_train.astype(np.float32)

        self._n_train = int(len(X_train))
        n_feat = int(X_train.shape[1]) if X_train.ndim == 2 else 1
        self._feat_names = [f"f{i:03d}" for i in range(n_feat)]

        df = pd.DataFrame(X_train.astype(np.float32), columns=self._feat_names)
        df["time_idx"] = np.arange(self._n_train, dtype=np.int64)
        df["group_ids"] = "seoul"
        df["target"] = y_fit.astype(np.float32)
        self._df_train = df

        val_n = max(8, int(self._n_train * 0.2))
        train_cutoff = self._n_train - val_n - 1
        df_training = df[df["time_idx"] <= train_cutoff].copy()
        training_ds = self._build_training_dataset(df_training)
        validation_ds = training_ds.__class__.from_dataset(
            training_ds,
            df,
            min_prediction_idx=train_cutoff + 1,
            predict=False,
            stop_randomization=True,
        )
        self._training_ds = training_ds
        model = self._build_model(training_ds)

        # Lightning Trainer
        # Package O (G-152 fix): max_time cap 강제 — Lightning final fit stuck 방지
        # 이전: 8h+ stuck 발생 (TFT/DeepAR-pf 등). max_time 으로 강제 종료.
        # G-188 (2026-05-14) — codex+gemini dual review fix:
        # default 1800 → 300 (TFT 14h+ infinite loop 차단, env var 미설정 시도 안전).
        _max_time_sec = GLOBAL.training.lightning_max_time_per_model
        _max_time_str = f"00:00:{_max_time_sec:02d}:00" if _max_time_sec < 3600 else f"00:{_max_time_sec // 3600:02d}:{(_max_time_sec % 3600) // 60:02d}:00"
        # 2026-05-20 사용자 영구 명시: "모든 epoch는 100이야"
        # 2026-05-21 Gemini fix: override → ceiling (prior stability fix 보존)
        from simulation.config_global import GLOBAL as _GCFG  # SSOT (2026-05-28)
        _moe = _GCFG.training.max_epochs_override
        _epochs_eff = min(self.EPOCHS, _moe) if _moe > 0 else self.EPOCHS
        import lightning.pytorch as pl
        self._history = []
        self._history_record_type = "lightning_epoch"
        _checkpoint_dir = (
            get_results_dir() / "checkpoints" / "lightning"
            / self.meta.name.replace(" ", "_").replace("/", "_")
        )
        # Phase-13 training-history task: val_loss early-stop + best restore + epoch capture.
        _callbacks = _build_lightning_callbacks(_checkpoint_dir, self._history)
        trainer = pl.Trainer(
            max_epochs=_epochs_eff,
            max_time={"seconds": _max_time_sec},  # G-152: Lightning timeout
            accelerator="auto",
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            enable_checkpointing=True,
            gradient_clip_val=0.1,
            callbacks=_callbacks,
        )
        train_dl = training_ds.to_dataloader(
            train=True, batch_size=self.BATCH_SIZE, num_workers=0,
        )
        val_dl = validation_ds.to_dataloader(
            train=False, batch_size=self.BATCH_SIZE, num_workers=0,
        )
        trainer.fit(model, train_dl, val_dl)
        best_model_path = _callbacks[1].best_model_path
        # G-237b (2026-06-15): best checkpoint 부재 시 silent last-epoch 유지 → refit-null
        # (N-HiTS/TiDE test_r2=None 의 2차 방어선; 1차 원인은 preproc best_idx error-trial).
        if not best_model_path:
            raise ValueError(
                f"{self.meta.name}: ModelCheckpoint best_model_path 빈값 — "
                f"val 개선 0/체크포인트 미저장 (refit-null fail-loud)")
        model = model.__class__.load_from_checkpoint(best_model_path)

        self._model = model
        self._fitted = True
        return self

    def predict(self, X_test: np.ndarray, y_observed=None, **kwargs) -> np.ndarray:
        import pandas as pd
        from pytorch_forecasting import TimeSeriesDataSet

        if not self._fitted or self._model is None:
            raise RuntimeError(f"{self.meta.name}: fit() 먼저 호출")

        n_test = int(len(X_test))
        df_test = pd.DataFrame(X_test.astype(np.float32), columns=self._feat_names)
        df_test["time_idx"] = np.arange(
            self._n_train, self._n_train + n_test, dtype=np.int64,
        )
        df_test["group_ids"] = "seoul"
        df_test["target"] = 0.0  # placeholder (inference 중 무시)
        # G-327c (2026-06-20, 사용자 "baseline rolling만"): 관측 y 주면 test target 을 **관측값**으로 채움 →
        #   encoder 가 placeholder 0.0 대신 실제 과거를 읽음(깊은 test 일수록 0 수렴하는 collapse 회피;
        #   N-HiTS pmean 0.1 → 정상). leak-free: pf 는 t 예측에 encoder window[<t] 만 사용(decoder 가
        #   예측할 target[t] 는 미열람). baseline=raw(tt=none) 라 y_observed=raw; 모델 내부 log1p 사용 시
        #   동일 공간(log1p)으로 맞춤. R9(transform-space)에는 미적용(supports_baseline_rolling 게이트).
        if y_observed is not None and len(y_observed) == n_test:
            # transform-fix (2026-06-21): no log1p conversion — RAW observed y (single transform is
            #   supplied by the preproc layer, so the encoder space == raw y).
            df_test["target"] = np.asarray(y_observed, dtype=np.float32)

        # train + test full df 로 dataset 재구성 -- encoder context 필요
        df_full = pd.concat([self._df_train, df_test], ignore_index=True)

        pred_ds = TimeSeriesDataSet.from_dataset(
            self._training_ds, df_full, predict=False, stop_randomization=True,
        )
        pred_dl = pred_ds.to_dataloader(
            train=False, batch_size=self.BATCH_SIZE, num_workers=0,
        )
        preds = self._model.predict(pred_dl, mode="prediction")
        if preds is None:
            raise ValueError(f"{self.meta.name}: predict() None 반환 (refit-null fail-loud)")
        if hasattr(preds, "detach"):
            preds_np = preds.detach().cpu().numpy()
        else:
            preds_np = np.asarray(preds)

        # (n, prediction_length=1) 또는 (n, pred, quantiles) -> (n,)
        if preds_np.ndim == 2 and preds_np.shape[1] == 1:
            preds_np = preds_np.squeeze(-1)
        elif preds_np.ndim >= 2:
            preds_np = preds_np.reshape(preds_np.shape[0], -1)[:, 0]

        # 마지막 n_test 개가 test time_idx 에 대응하는 예측
        preds_np = preds_np[-n_test:]

        # transform-fix (2026-06-21): no internal expm1 inverse (fit on raw y). A LINEAR-space
        #   explosion-cap backstop is RETAINED: a dense/transformer net can overshoot on out-of-
        #   range peaks (was TiDE-pf R²=-518, pred max=669). Cap at 10×train_max in original units
        #   (G-180/G-146 intent, now linear) + 0-floor; no exponential inverse to amplify it.
        try:
            _y_max_train = (float(np.nanmax(self._y_train_raw))
                            if getattr(self, "_y_train_raw", None) is not None else 200.0)
        except Exception:
            _y_max_train = 200.0
        _cap = 10.0 * _y_max_train if _y_max_train > 0 else np.inf  # 보수적 10× cap
        return np.clip(preds_np, 0.0, _cap).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════
# Concrete forecasters
# ═══════════════════════════════════════════════════════════════════════════


class PfTFTForecaster(_PfBase):
    """Temporal Fusion Transformer -- pytorch_forecasting 1.7.0 reference."""

    meta = ModelMeta(
        name="TFT",
        category="dl",
        level=15,
        min_data=120,
        description="TFT pf 1.7.0 reference. Attention+quantile, 재현성 벤치마크.",
        dependencies=["torch", "pytorch_forecasting", "lightning"],
    )

    MAX_ENCODER_LENGTH = 8
    HIDDEN_SIZE = 16

    def _build_model(self, training_ds):
        from pytorch_forecasting import TemporalFusionTransformer
        from pytorch_forecasting.metrics import QuantileLoss
        return TemporalFusionTransformer.from_dataset(
            training_ds,
            hidden_size=self.HIDDEN_SIZE,
            attention_head_size=2,
            hidden_continuous_size=8,
            dropout=self.DROPOUT,
            learning_rate=self.LR,
            loss=QuantileLoss(),
            log_interval=-1,
        )


class PfNBeatsForecaster(_PfBase):
    """N-BEATS -- pytorch_forecasting 1.7.0 reference (univariate)."""

    meta = ModelMeta(
        name="N-BEATS",
        category="dl",
        level=11,
        min_data=120,
        description="N-BEATS pf 1.7.0. Univariate basis expansion.",
        dependencies=["torch", "pytorch_forecasting", "lightning"],
    )

    USE_COVARIATES = False
    MAX_ENCODER_LENGTH = 16

    def _build_model(self, training_ds):
        from pytorch_forecasting import NBeats
        return NBeats.from_dataset(
            training_ds,
            learning_rate=self.LR,
            log_interval=-1,
            log_val_interval=-1,
            weight_decay=1e-2,
            widths=[32, 256],
            backcast_loss_ratio=0.1,
        )


class PfNHiTSForecaster(_PfBase):
    """N-HiTS -- pytorch_forecasting 1.7.0 reference (multivariate)."""

    meta = ModelMeta(
        name="N-HiTS",
        category="dl",
        level=12,
        min_data=120,
        description="N-HiTS pf 1.7.0. Hierarchical interpolation.",
        dependencies=["torch", "pytorch_forecasting", "lightning"],
    )

    MAX_ENCODER_LENGTH = 16
    ADD_RELATIVE_TIME_IDX = False  # NHiTS 내부 assert

    def _build_model(self, training_ds):
        from pytorch_forecasting import NHiTS
        from pytorch_forecasting.metrics import MAE
        return NHiTS.from_dataset(
            training_ds,
            hidden_size=self.HIDDEN_SIZE,
            loss=MAE(),
            learning_rate=self.LR,
            log_interval=-1,
            log_val_interval=-1,
        )


class PfTiDEForecaster(_PfBase):
    """TiDEModel -- pytorch_forecasting 1.7.0 reference (dense encoder)."""

    meta = ModelMeta(
        name="TiDE",
        category="dl",
        level=29,
        min_data=120,
        description="TiDEModel pf 1.7.0. Dense time-series encoder.",
        dependencies=["torch", "pytorch_forecasting", "lightning"],
    )

    MAX_ENCODER_LENGTH = 8
    ADD_RELATIVE_TIME_IDX = False  # TiDE 내부 assert

    def _build_model(self, training_ds):
        from pytorch_forecasting import TiDEModel
        from pytorch_forecasting.metrics import MAE
        return TiDEModel.from_dataset(
            training_ds,
            hidden_size=self.HIDDEN_SIZE,
            dropout=self.DROPOUT,
            loss=MAE(),
            learning_rate=self.LR,
            log_interval=-1,
        )


class PfRNNForecaster(_PfBase):
    """RNN (LSTM) via pytorch_forecasting RecurrentNetwork -- baseline."""

    meta = ModelMeta(
        name="RNN",
        category="dl",
        level=10,
        min_data=120,
        description="LSTM pf 1.7.0 RecurrentNetwork -- 재추가된 baseline.",
        dependencies=["torch", "pytorch_forecasting", "lightning"],
    )

    MAX_ENCODER_LENGTH = 8
    HIDDEN_SIZE = 16

    def _build_model(self, training_ds):
        from pytorch_forecasting import RecurrentNetwork
        from pytorch_forecasting.metrics import MAE
        return RecurrentNetwork.from_dataset(
            training_ds,
            cell_type="LSTM",
            hidden_size=self.HIDDEN_SIZE,
            rnn_layers=1,
            dropout=0.1,
            loss=MAE(),
            learning_rate=self.LR,
            log_interval=-1,
        )


class PfDeepARForecaster(_PfBase):
    """DeepAR -- probabilistic autoregressive RNN. Registry 최초 prob-forecaster."""

    meta = ModelMeta(
        name="DeepAR",
        category="dl",
        level=13,
        min_data=120,
        description="DeepAR pf 1.7.0. Probabilistic autoregressive (Gaussian).",
        dependencies=["torch", "pytorch_forecasting", "lightning"],
    )

    USE_COVARIATES = False  # target only; covariate 은 Prob-RNN 과 어긋남
    MAX_ENCODER_LENGTH = 16

    def _build_training_dataset(self, df):
        # DeepAR 는 univariate 지만 static_categoricals + normalizer 필요
        from pytorch_forecasting import TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer

        return TimeSeriesDataSet(
            df,
            time_idx="time_idx",
            target="target",
            group_ids=["group_ids"],
            max_encoder_length=self.MAX_ENCODER_LENGTH,
            max_prediction_length=self.MAX_PREDICTION_LENGTH,
            static_categoricals=["group_ids"],
            time_varying_unknown_reals=["target"],
            # 2026-06-16 G-274: 부호-안전 normalizer.
            # transform-fix (2026-06-21) FLAG: DeepAR keeps its softplus∘sign coupling here, UNLIKE
            #   _PfBase (which was switched to transformation=None this pass). DeepAR is DEFER/
            #   inactive in the active lineup, so its internal-transform un-hardcode is intentionally
            #   deferred to a follow-up; do NOT treat this softplus as a live double-transform bug.
            target_normalizer=GroupNormalizer(
                groups=["group_ids"],
                transformation=("softplus" if self._y_log_used else None),
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
        )

    def _build_model(self, training_ds):
        from pytorch_forecasting import DeepAR
        from pytorch_forecasting.metrics import NormalDistributionLoss
        return DeepAR.from_dataset(
            training_ds,
            cell_type="LSTM",
            hidden_size=self.HIDDEN_SIZE,
            rnn_layers=1,
            dropout=0.1,
            loss=NormalDistributionLoss(),
            learning_rate=self.LR,
            log_interval=-1,
        )
