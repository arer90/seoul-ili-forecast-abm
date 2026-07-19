"""
simulation/tests/smoke_pytorch_forecasting.py
=============================================
pytorch_forecasting 1.7.0 full model smoke test (exploration).

목적: pf 를 기존 custom 모델에 **추가** 하는 migration 이 가능한지 확인.
 - 각 모델: import / dataset / fit 2-epoch / predict 성공?
 - output shape 이 BaseForecaster.predict 와 호환?

pf 1.7.0 usable model classes (Baseline 제외):
 - Multivariate (covariate 지원):
 TemporalFusionTransformer, NHiTS, TiDEModel,
 DecoderMLP, RecurrentNetwork
 - Univariate (target only):
 NBeats, NBeatsKAN
 - Autoregressive (probabilistic):
 DeepAR

pf 1.7.0 은 `lightning.pytorch` 상속 -- `pytorch_lightning` 이 아님.
Trainer 도 반드시 `lightning.pytorch.Trainer` 사용.

직접 실행:
 .venv\\Scripts\\python.exe simulation/tests/smoke_pytorch_forecasting.py
"""
from __future__ import annotations

import time
import traceback
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# --------------------------------------------------------------------------
# data helpers
# --------------------------------------------------------------------------
def _make_dummy_data(n_weeks: int = 234, n_features: int = 30, seed: int = 42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_weeks, n_features)).astype(np.float32)
    base = np.linspace(0, 6, n_weeks)
    seasonal = 2.0 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
    noise = 0.5 * rng.standard_normal(n_weeks)
    y = np.maximum(base + seasonal + noise, 0.1).astype(np.float32)
    feat_names = [f"feat_{i:02d}" for i in range(n_features)]
    return X, y, feat_names


def _build_long_df(X, y, feat_names):
    n = len(y)
    df = pd.DataFrame(X, columns=feat_names)
    df["time_idx"] = np.arange(n, dtype=np.int64)
    df["group_ids"] = "seoul"
    df["target"] = y.astype(np.float32)
    return df


def _multivariate_dataset(X, y, feat_names, max_enc=8, max_pred=1, val_h=12,
                          add_relative_time_idx: bool = True):
    """covariate 지원 모델용 (TFT, NHiTS, TiDE, DecoderMLP, RNN).

    NHiTS / TiDEModel 은 `add_relative_time_idx=False` 를 강제 (내부 assert).
    """
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer

    df = _build_long_df(X, y, feat_names)
    n = len(df)
    cutoff = n - val_h - max_pred

    training = TimeSeriesDataSet(
        df[df["time_idx"] <= cutoff],
        time_idx="time_idx",
        target="target",
        group_ids=["group_ids"],
        max_encoder_length=max_enc,
        max_prediction_length=max_pred,
        static_categoricals=["group_ids"],
        time_varying_known_reals=feat_names,
        time_varying_unknown_reals=["target"],
        target_normalizer=GroupNormalizer(groups=["group_ids"],
                                          transformation="softplus"),
        add_relative_time_idx=add_relative_time_idx,
        add_target_scales=True,
    )
    validation = TimeSeriesDataSet.from_dataset(training, df, predict=False,
                                                stop_randomization=True)
    return training, validation


def _univariate_dataset(y, max_enc=16, max_pred=1, val_h=12):
    """target-only 모델용 (NBeats, NBeatsKAN)."""
    from pytorch_forecasting import TimeSeriesDataSet

    n = len(y)
    df = pd.DataFrame({
        "time_idx": np.arange(n, dtype=np.int64),
        "group_ids": "seoul",
        "target": y.astype(np.float32),
    })
    cutoff = n - val_h - max_pred
    training = TimeSeriesDataSet(
        df[df["time_idx"] <= cutoff],
        time_idx="time_idx",
        target="target",
        group_ids=["group_ids"],
        max_encoder_length=max_enc,
        max_prediction_length=max_pred,
        time_varying_unknown_reals=["target"],
        target_normalizer=None,
    )
    validation = TimeSeriesDataSet.from_dataset(training, df, predict=False,
                                                stop_randomization=True)
    return training, validation


def _deepar_dataset(y, max_enc=16, max_pred=1, val_h=12):
    """DeepAR: autoregressive, target-only, needs numeric group_id."""
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer

    n = len(y)
    df = pd.DataFrame({
        "time_idx": np.arange(n, dtype=np.int64),
        "group_ids": "seoul",
        "target": y.astype(np.float32),
    })
    cutoff = n - val_h - max_pred
    training = TimeSeriesDataSet(
        df[df["time_idx"] <= cutoff],
        time_idx="time_idx",
        target="target",
        group_ids=["group_ids"],
        max_encoder_length=max_enc,
        max_prediction_length=max_pred,
        static_categoricals=["group_ids"],
        time_varying_unknown_reals=["target"],
        target_normalizer=GroupNormalizer(groups=["group_ids"],
                                          transformation="softplus"),
        add_relative_time_idx=True,
        add_target_scales=True,
    )
    validation = TimeSeriesDataSet.from_dataset(training, df, predict=False,
                                                stop_randomization=True)
    return training, validation


# --------------------------------------------------------------------------
# trainer helper
# --------------------------------------------------------------------------
def _fit_and_predict(model, train_ds, val_ds, max_epochs=2, batch_size=16,
                     use_early_stop=False):
    # pf 1.7.0: LightningModule 이 lightning.pytorch 네임스페이스 소속.
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import EarlyStopping

    train_dl = train_ds.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dl = val_ds.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    callbacks = []
    if use_early_stop:
        callbacks.append(EarlyStopping(monitor="val_loss", patience=10,
                                       mode="min", verbose=False))

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        gradient_clip_val=0.1,
        callbacks=callbacks,
    )
    t_fit = time.time()
    trainer.fit(model, train_dl, val_dl)
    fit_sec = time.time() - t_fit

    t_pred = time.time()
    preds = model.predict(val_dl, mode="prediction")
    pred_sec = time.time() - t_pred
    preds_np = preds.detach().cpu().numpy() if hasattr(preds, "detach") else np.asarray(preds)
    return preds_np, fit_sec, pred_sec


# --------------------------------------------------------------------------
# per-model smoke runners
# --------------------------------------------------------------------------
def smoke_tft(X, y, feat):
    from pytorch_forecasting import TemporalFusionTransformer
    from pytorch_forecasting.metrics import QuantileLoss
    train_ds, val_ds = _multivariate_dataset(X, y, feat, max_enc=8)
    model = TemporalFusionTransformer.from_dataset(
        train_ds, hidden_size=16, attention_head_size=2,
        hidden_continuous_size=8, dropout=0.3, learning_rate=1e-3,
        loss=QuantileLoss(), log_interval=-1)
    return _fit_and_predict(model, train_ds, val_ds,
                            max_epochs=_EPOCHS, batch_size=_BATCH,
                            use_early_stop=_EARLY_STOP)


def smoke_nhits(X, y, feat):
    from pytorch_forecasting import NHiTS
    from pytorch_forecasting.metrics import MAE
    train_ds, val_ds = _multivariate_dataset(X, y, feat, max_enc=16,
                                              add_relative_time_idx=False)
    model = NHiTS.from_dataset(
        train_ds,
        hidden_size=16,
        loss=MAE(),
        learning_rate=1e-3,
        log_interval=-1,
        log_val_interval=-1,
    )
    return _fit_and_predict(model, train_ds, val_ds,
                            max_epochs=_EPOCHS, batch_size=_BATCH,
                            use_early_stop=_EARLY_STOP)


def smoke_tide(X, y, feat):
    from pytorch_forecasting import TiDEModel
    from pytorch_forecasting.metrics import MAE
    train_ds, val_ds = _multivariate_dataset(X, y, feat, max_enc=8,
                                              add_relative_time_idx=False)
    model = TiDEModel.from_dataset(
        train_ds,
        hidden_size=16,
        dropout=0.3,
        loss=MAE(),
        learning_rate=1e-3,
        log_interval=-1,
    )
    return _fit_and_predict(model, train_ds, val_ds,
                            max_epochs=_EPOCHS, batch_size=_BATCH,
                            use_early_stop=_EARLY_STOP)


def smoke_decoder_mlp(X, y, feat):
    from pytorch_forecasting import DecoderMLP
    from pytorch_forecasting.metrics import MAE
    train_ds, val_ds = _multivariate_dataset(X, y, feat, max_enc=8)
    model = DecoderMLP.from_dataset(
        train_ds,
        hidden_size=32,
        dropout=0.3,
        loss=MAE(),
        learning_rate=1e-3,
        log_interval=-1,
    )
    return _fit_and_predict(model, train_ds, val_ds,
                            max_epochs=_EPOCHS, batch_size=_BATCH,
                            use_early_stop=_EARLY_STOP)


def smoke_rnn(X, y, feat):
    from pytorch_forecasting import RecurrentNetwork
    from pytorch_forecasting.metrics import MAE
    train_ds, val_ds = _multivariate_dataset(X, y, feat, max_enc=8)
    model = RecurrentNetwork.from_dataset(
        train_ds,
        cell_type="LSTM",
        hidden_size=16,
        rnn_layers=1,
        dropout=0.1,
        loss=MAE(),
        learning_rate=1e-3,
        log_interval=-1,
    )
    return _fit_and_predict(model, train_ds, val_ds,
                            max_epochs=_EPOCHS, batch_size=_BATCH,
                            use_early_stop=_EARLY_STOP)


def smoke_nbeats(y):
    from pytorch_forecasting import NBeats
    train_ds, val_ds = _univariate_dataset(y, max_enc=16)
    model = NBeats.from_dataset(
        train_ds, learning_rate=1e-3, log_interval=-1, log_val_interval=-1,
        widths=[32, 256], backcast_loss_ratio=0.1)
    return _fit_and_predict(model, train_ds, val_ds,
                            max_epochs=_EPOCHS, batch_size=_BATCH,
                            use_early_stop=_EARLY_STOP)


def smoke_nbeats_kan(y):
    from pytorch_forecasting import NBeatsKAN
    train_ds, val_ds = _univariate_dataset(y, max_enc=16)
    model = NBeatsKAN.from_dataset(
        train_ds, learning_rate=1e-3, log_interval=-1, log_val_interval=-1,
        widths=[32, 128], backcast_loss_ratio=0.1)
    return _fit_and_predict(model, train_ds, val_ds,
                            max_epochs=_EPOCHS, batch_size=_BATCH,
                            use_early_stop=_EARLY_STOP)


def smoke_deepar(y):
    from pytorch_forecasting import DeepAR
    from pytorch_forecasting.metrics import NormalDistributionLoss
    train_ds, val_ds = _deepar_dataset(y, max_enc=16)
    model = DeepAR.from_dataset(
        train_ds,
        cell_type="LSTM",
        hidden_size=16,
        rnn_layers=1,
        dropout=0.1,
        loss=NormalDistributionLoss(),
        learning_rate=1e-3,
        log_interval=-1,
    )
    return _fit_and_predict(model, train_ds, val_ds,
                            max_epochs=_EPOCHS, batch_size=_BATCH,
                            use_early_stop=_EARLY_STOP)


# --------------------------------------------------------------------------
# runner
# --------------------------------------------------------------------------
# module-level knobs (main() overrides based on --realistic flag)
_EPOCHS = 2
_BATCH = 16
_EARLY_STOP = False

SUITES_FULL = [
    ("TemporalFusionTransformer", "multi", smoke_tft),
    ("NHiTS",                     "multi", smoke_nhits),
    ("TiDEModel",                 "multi", smoke_tide),
    ("DecoderMLP",                "multi", smoke_decoder_mlp),
    ("RecurrentNetwork",          "multi", smoke_rnn),
    ("NBeats",                    "uni",   smoke_nbeats),
    ("NBeatsKAN",                 "uni",   smoke_nbeats_kan),
    ("DeepAR",                    "ar",    smoke_deepar),
]

# Tier 1+2 (integration 후보 6개). realistic smoke 는 이 6개만.
SUITES_TIER12 = [
    ("TemporalFusionTransformer", "multi", smoke_tft),
    ("NHiTS",                     "multi", smoke_nhits),
    ("TiDEModel",                 "multi", smoke_tide),
    ("RecurrentNetwork",          "multi", smoke_rnn),
    ("NBeats",                    "uni",   smoke_nbeats),
    ("DeepAR",                    "ar",    smoke_deepar),
]


def main():
    import sys
    realistic = "--realistic" in sys.argv

    global _EPOCHS, _BATCH, _EARLY_STOP
    if realistic:
        n_features = 309
        _EPOCHS = 20
        _BATCH = 32
        _EARLY_STOP = True
        suites = SUITES_TIER12
        tag = "realistic shape (n=234 x p=309, 20 epoch, Tier 1+2 only)"
    else:
        n_features = 30
        _EPOCHS = 2
        _BATCH = 16
        _EARLY_STOP = False
        suites = SUITES_FULL
        tag = "quick smoke (n=234 x p=30, 2 epoch, all 8 models)"

    print("=" * 72)
    print(f"pytorch_forecasting 1.7.0 -- {tag}")
    print(f"torch {torch.__version__}, CUDA={torch.cuda.is_available()}")
    print("=" * 72)

    X, y, feat = _make_dummy_data(n_weeks=234, n_features=n_features)
    print(f"data: X={X.shape}, y={y.shape}, y_range=[{y.min():.2f}, {y.max():.2f}]")
    print()

    results = []
    for name, kind, fn in suites:
        t0 = time.time()
        try:
            if kind == "multi":
                preds_np, fit_sec, pred_sec = fn(X, y, feat)
            else:
                preds_np, fit_sec, pred_sec = fn(y)
            total = time.time() - t0
            row = {
                "model": name, "kind": kind, "status": "OK",
                "fit_sec": round(fit_sec, 2),
                "pred_sec": round(pred_sec, 2),
                "total_sec": round(total, 2),
                "shape": preds_np.shape,
                "pred_min": float(preds_np.min()),
                "pred_max": float(preds_np.max()),
            }
            print(f"[OK ]  {name:<28} {kind:<6} "
                  f"fit={fit_sec:5.2f}s pred={pred_sec:5.2f}s "
                  f"shape={preds_np.shape}")
        except Exception as e:
            total = time.time() - t0
            row = {
                "model": name, "kind": kind, "status": "FAIL",
                "error": f"{type(e).__name__}: {str(e)[:120]}",
                "total_sec": round(total, 2),
            }
            print(f"[FAIL] {name:<28} {kind:<6} "
                  f"{type(e).__name__}: {str(e)[:80]}")
            if "--trace" in __import__("sys").argv:
                traceback.print_exc()
        results.append(row)

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    n_ok = sum(1 for r in results if r["status"] == "OK")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    print(f"  passed: {n_ok}/{len(results)},  failed: {n_fail}")
    for r in results:
        line = f"  {r['status']:<4} {r['model']:<28}"
        if r["status"] == "OK":
            line += f" fit={r['fit_sec']}s pred={r['pred_sec']}s shape={r['shape']}"
        else:
            line += f" {r.get('error', '')}"
        print(line)
    return results


if __name__ == "__main__":
    main()
