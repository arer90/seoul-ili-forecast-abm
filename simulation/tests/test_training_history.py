"""Smoke tests for Phase-13 unified training-history persistence."""
from __future__ import annotations

import json
import math

import polars as pl

from simulation.pipeline.training_history import save_training_record


def _read(path):
    return pl.read_csv(path, schema_overrides={
        "model": pl.String,
        "scope": pl.String,
        "record_type": pl.String,
        "step": pl.Int64,
        "split": pl.String,
        "metric_name": pl.String,
        "value": pl.Float64,
        "params_json": pl.String,
        "saved_at": pl.String,
    })


def test_empty_history_writes_nan_row_and_png(tmp_path, caplog):
    csv_path = save_training_record(
        "EmptyModel", "pooled", "dl_epoch", None, tmp_path,
    )

    rows = _read(csv_path)
    assert rows.height == 1
    assert rows["record_type"][0] == "dl_epoch"
    assert math.isnan(rows["value"][0])
    assert (tmp_path / "figures" / "EmptyModel_pooled.png").exists()
    assert "empty" in caplog.text.lower()


def test_dl_history_writes_long_format_rows(tmp_path):
    csv_path = save_training_record(
        "DNN",
        "pooled",
        "dl_epoch",
        [
            {"epoch": 0, "train_loss": 3.0, "val_loss": 4.0, "lr": 1e-3},
            {"epoch": 1, "train_loss": 2.0, "val_loss": 2.5, "lr": 5e-4},
        ],
        tmp_path,
        params_json='{"layers": 2}',
    )

    rows = _read(csv_path)
    assert set(rows["split"]) >= {"train", "val"}
    assert set(rows["metric_name"]) >= {"loss", "lr"}
    assert rows.filter(pl.col("metric_name") == "loss").height == 4
    assert json.loads(rows["params_json"][0]) == {"layers": 2}


def test_optuna_mock_study_writes_trials_and_params(tmp_path):
    class MockStudy:
        def trials_dataframe(self):
            return pl.DataFrame({
                "number": [0, 1],
                "value": [5.0, 3.0],
                "params_depth": [2, 4],
            })

    csv_path = save_training_record(
        "LightGBM", "pooled", "optuna_trial", MockStudy(), tmp_path,
    )

    rows = _read(csv_path)
    assert rows.height == 2
    assert rows["step"].to_list() == [0, 1]
    assert set(rows["split"]) == {"optuna_trial"}
    assert set(rows["metric_name"]) == {"WIS"}
    assert json.loads(rows["params_json"][1]) == {"depth": 4}


def test_lightning_history_writes_available_metrics(tmp_path):
    csv_path = save_training_record(
        "N-BEATS",
        "pooled",
        "lightning_epoch",
        [
            {"epoch": 0, "train_loss_epoch": 1.5, "val_loss": 1.7},
            {"epoch": 1, "train_loss_epoch": 1.0, "val_loss": 1.2},
        ],
        tmp_path,
    )

    rows = _read(csv_path)
    assert set(rows["step"]) == {0, 1}
    assert set(rows["split"]) == {"train", "val"}
    assert set(rows["metric_name"]) == {"loss"}


def test_closed_form_metrics_and_summary_wis_png(tmp_path):
    csv_path = save_training_record(
        "ARIMA",
        "pooled",
        "closed_form",
        {"AIC": 120.5, "BIC": 130.0, "val_WIS": 2.75, "r2": 0.8},
        tmp_path,
    )

    rows = _read(csv_path)
    assert set(rows["metric_name"]) == {"AIC", "BIC", "WIS", "r2"}
    assert rows.filter(pl.col("metric_name") == "WIS")["split"][0] == "val"
    assert (tmp_path / "figures" / "summary_wis.png").exists()


def test_nan_metric_is_preserved(tmp_path):
    csv_path = save_training_record(
        "NaNModel", "11010", "closed_form", {"r2": float("nan")}, tmp_path,
    )

    rows = _read(csv_path)
    assert rows["scope"][0] == "11010"
    assert math.isnan(rows["value"][0])


def test_existing_model_scope_csv_is_bulk_appended(tmp_path):
    first = save_training_record(
        "Combo/Model", "pooled", "closed_form", {"r2": 0.5}, tmp_path,
    )
    second = save_training_record(
        "Combo/Model", "pooled", "closed_form", {"AIC": 10.0}, tmp_path,
    )

    assert first == second
    assert first.name == "Combo_Model_pooled.csv"
    assert _read(first).height == 2


def test_lightning_early_stopping_fires_before_max_epochs(tmp_path):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import lightning.pytorch as lightning

    from simulation.models.modern_ts.pf_models import _build_lightning_callbacks

    class ConstantValModule(lightning.LightningModule):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))

        def training_step(self, batch, batch_idx):
            return self.weight * 0.0

        def validation_step(self, batch, batch_idx):
            self.log("val_loss", torch.tensor(1.0), on_epoch=True)

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=0.1)

    history = []
    callbacks = _build_lightning_callbacks(tmp_path, history)
    loader = DataLoader(TensorDataset(torch.ones(2, 1)), batch_size=1)
    trainer = lightning.Trainer(
        max_epochs=20,
        callbacks=callbacks,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        accelerator="cpu",
        devices=1,
    )
    trainer.fit(ConstantValModule(), loader, loader)

    assert trainer.current_epoch < 19
    assert history
