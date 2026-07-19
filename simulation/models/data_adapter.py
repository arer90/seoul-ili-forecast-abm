"""
simulation.models.data_adapter
==============================
— single adapter that wraps the feature-selected (X, y) into any of
the three formats the registry consumes.

Context
-------
After R9 (per_model_optimize) feature-Optuna finishes (feature-Optuna →
selected_features → sliced (X_train, X_val, X_test)) we still need to hand
the arrays to three different model-family APIs:

 1. sklearn / statsmodels / xgboost / lightgbm — plain numpy.
 2. pytorch DNNs (TabularDNN-Lite / TinyMLP / TCN) — torch.utils.data.Dataset.
 3. pytorch_forecasting (TFT-pf / NBeats-pf / TiDE-pf / NHiTS-pf /
 RNN-pf / DeepAR-pf) — TimeSeriesDataSet.

Putting the adapter here (rather than inside each model) gives R9
(per_model_optimize) a single call-site:

 from simulation.models.data_adapter import adapt, AdapterKind
 bundle = adapt(
 X_train, y_train, X_val, y_val, X_test, y_test,
 feature_names=selected_names,
 kind=AdapterKind.infer(model))

Torch / pytorch_forecasting imports are lazy — the module imports cleanly
on numpy-only installs and only pulls the heavy deps when a torch / pf
format is actually requested.

Ordering guarantee
------------------
This adapter is called AFTER feature selection (feature-Optuna / policy
filter). The feature list passed in is the final one — the adapter does
not prune columns. That matches the user's explicit request:

 "feature optuna에서는 feature를 다하고 나서 dataset으로 만드는거 맞지?"
 → yes: select features first, then wrap into the chosen format.
"""
from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

import numpy as np

log = logging.getLogger(__name__)


__all__ = [
    "AdapterKind",
    "DataBundle",
    "adapt",
    "to_numpy",
    "to_torch_dataset",
    "to_pf_timeseries_dataset",
]


# ══════════════════════════════════════════════════════════════════════════
# Format enum
# ══════════════════════════════════════════════════════════════════════════
class AdapterKind(str, enum.Enum):
    NUMPY = "numpy"
    TORCH_DATASET = "torch_dataset"
    PF_TIMESERIES = "pf_timeseries"

    @classmethod
    def infer(cls, model: Any) -> "AdapterKind":
        """Heuristic format selection from a model instance or class name.

        Rule of thumb:
          - pytorch_forecasting wrappers (`Pf*Forecaster`, any class whose
            category is `dl` *and* whose module path sits under
            `simulation.models.modern_ts.pf_models`)      → PF_TIMESERIES
          - other DL / torch models                       → TORCH_DATASET
          - everything else (ts / linear / tree / epi)    → NUMPY
        """
        cls_obj = type(model) if not isinstance(model, type) else model
        modname = getattr(cls_obj, "__module__", "") or ""
        name = getattr(getattr(cls_obj, "meta", None), "name", "") or cls_obj.__name__

        if "pf_models" in modname or name.endswith("-pf"):
            return cls.PF_TIMESERIES

        category = getattr(getattr(cls_obj, "meta", None), "category", "")
        if category in ("dl", "physics"):
            return cls.TORCH_DATASET
        return cls.NUMPY


# ══════════════════════════════════════════════════════════════════════════
# Bundle returned to callers
# ══════════════════════════════════════════════════════════════════════════
@dataclass
class DataBundle:
    """The result of wrapping features into one of three formats.

    Only the fields relevant to `kind` are populated; the others stay None.
    Callers branch on `bundle.kind` (or rely on the keyword-matching
    `train / val / test` attributes).
    """
    kind: AdapterKind
    feature_names: list[str]
    n_train: int
    n_val: int
    n_test: int

    # NUMPY path
    X_train: Optional[np.ndarray] = None
    y_train: Optional[np.ndarray] = None
    X_val:   Optional[np.ndarray] = None
    y_val:   Optional[np.ndarray] = None
    X_test:  Optional[np.ndarray] = None
    y_test:  Optional[np.ndarray] = None

    # TORCH_DATASET path  (torch.utils.data.Dataset instances)
    train: Optional[Any] = None
    val:   Optional[Any] = None
    test:  Optional[Any] = None

    # PF_TIMESERIES path  (pytorch_forecasting.TimeSeriesDataSet instances)
    pf_train:      Optional[Any] = None
    pf_val:        Optional[Any] = None
    pf_test:       Optional[Any] = None
    pf_group_id:   str = "seoul"
    pf_params:     dict = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "n_features": len(self.feature_names),
            "n_train": self.n_train,
            "n_val": self.n_val,
            "n_test": self.n_test,
        }


# ══════════════════════════════════════════════════════════════════════════
# 1) numpy
# ══════════════════════════════════════════════════════════════════════════
def to_numpy(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None,
    *,
    feature_names: Iterable[str],
) -> DataBundle:
    """Passthrough; enforces float32 to match the torch paths."""
    fn = list(feature_names)
    if X_train.shape[1] != len(fn):
        raise ValueError(
            f"X_train.shape[1]={X_train.shape[1]} != len(feature_names)={len(fn)}"
        )
    return DataBundle(
        kind=AdapterKind.NUMPY,
        feature_names=fn,
        n_train=int(len(y_train)),
        n_val=int(0 if y_val is None else len(y_val)),
        n_test=int(0 if y_test is None else len(y_test)),
        X_train=X_train.astype(np.float32, copy=False),
        y_train=y_train.astype(np.float32, copy=False),
        X_val=None if X_val is None else X_val.astype(np.float32, copy=False),
        y_val=None if y_val is None else y_val.astype(np.float32, copy=False),
        X_test=None if X_test is None else X_test.astype(np.float32, copy=False),
        y_test=None if y_test is None else y_test.astype(np.float32, copy=False),
    )


# ══════════════════════════════════════════════════════════════════════════
# 2) torch.utils.data.Dataset
# ══════════════════════════════════════════════════════════════════════════
def _build_torch_dataset_class():
    """Return a local `TabularDataset` class, importing torch lazily.

    Defining the class inside a factory avoids importing torch at module
    load — `to_numpy()` users never pay the torch import cost.
    """
    import torch
    from torch.utils.data import Dataset

    class _TabularDataset(Dataset):
        __slots__ = ("X", "y")

        def __init__(self, X: np.ndarray, y: np.ndarray):
            self.X = torch.as_tensor(X, dtype=torch.float32)
            self.y = torch.as_tensor(y, dtype=torch.float32)

        def __len__(self) -> int:
            return int(self.y.shape[0])

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    return _TabularDataset


def to_torch_dataset(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None,
    *,
    feature_names: Iterable[str],
) -> DataBundle:
    """Wrap into torch.utils.data.Dataset tuples.

    The feature list must already be finalized (feature selection done).
    """
    fn = list(feature_names)
    TabularDataset = _build_torch_dataset_class()

    train = TabularDataset(X_train, y_train)
    val = None if X_val is None or y_val is None else TabularDataset(X_val, y_val)
    test = None if X_test is None or y_test is None else TabularDataset(X_test, y_test)

    return DataBundle(
        kind=AdapterKind.TORCH_DATASET,
        feature_names=fn,
        n_train=len(train),
        n_val=0 if val is None else len(val),
        n_test=0 if test is None else len(test),
        train=train, val=val, test=test,
    )


# ══════════════════════════════════════════════════════════════════════════
# 3) pytorch_forecasting.TimeSeriesDataSet
# ══════════════════════════════════════════════════════════════════════════
def to_pf_timeseries_dataset(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None,
    *,
    feature_names: Iterable[str],
    max_encoder_length: int = 8,
    max_prediction_length: int = 1,
    group_id: str = "seoul",
    use_covariates: bool = True,
    add_relative_time_idx: bool = True,
    target_log1p: bool = True,
) -> DataBundle:
    """Build pytorch_forecasting TimeSeriesDataSet objects.

    The train-set is constructed explicitly; val / test are derived via
    `TimeSeriesDataSet.from_dataset(train_ds, df, stop_randomization=True)`
    to guarantee identical categorical encodings and normalizers.
    """
    try:
        import pandas as pd
        from pytorch_forecasting import TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer
    except ImportError as e:
        raise ImportError(
            "pytorch_forecasting not installed -- run "
            "`uv pip install pytorch_forecasting` first."
        ) from e

    fn = list(feature_names)
    if X_train.shape[1] != len(fn):
        raise ValueError(
            f"X_train.shape[1]={X_train.shape[1]} != len(feature_names)={len(fn)}"
        )

    def _frame(X: np.ndarray, y: np.ndarray, start_idx: int) -> "pd.DataFrame":
        df = pd.DataFrame(X.astype(np.float32, copy=False), columns=fn)
        df["time_idx"] = np.arange(start_idx, start_idx + len(y), dtype=np.int64)
        df["group_ids"] = group_id
        y_fit = np.log1p(y) if (target_log1p and np.all(y >= 0)) else y
        df["target"] = y_fit.astype(np.float32)
        return df

    n_train = int(len(y_train))
    n_val = int(0 if y_val is None else len(y_val))
    n_test = int(0 if y_test is None else len(y_test))

    df_train = _frame(X_train, y_train, 0)

    common = dict(
        time_idx="time_idx",
        target="target",
        group_ids=["group_ids"],
        max_encoder_length=max_encoder_length,
        max_prediction_length=max_prediction_length,
    )

    if use_covariates:
        train_ds = TimeSeriesDataSet(
            df_train,
            **common,
            static_categoricals=["group_ids"],
            time_varying_known_reals=fn,
            time_varying_unknown_reals=["target"],
            target_normalizer=GroupNormalizer(groups=["group_ids"], transformation="softplus"),
            add_relative_time_idx=add_relative_time_idx,
            add_target_scales=True,
            allow_missing_timesteps=True,
        )
    else:
        train_ds = TimeSeriesDataSet(
            df_train,
            **common,
            time_varying_unknown_reals=["target"],
            target_normalizer=None,
            allow_missing_timesteps=True,
        )

    val_ds = None
    if X_val is not None and y_val is not None:
        df_val = _frame(X_val, y_val, n_train)
        val_ds = TimeSeriesDataSet.from_dataset(
            train_ds, pd.concat([df_train, df_val], ignore_index=True),
            predict=True, stop_randomization=True,
        )

    test_ds = None
    if X_test is not None and y_test is not None:
        df_test = _frame(X_test, y_test, n_train + n_val)
        test_ds = TimeSeriesDataSet.from_dataset(
            train_ds, pd.concat([df_train, df_test], ignore_index=True),
            predict=True, stop_randomization=True,
        )

    return DataBundle(
        kind=AdapterKind.PF_TIMESERIES,
        feature_names=fn,
        n_train=n_train, n_val=n_val, n_test=n_test,
        pf_train=train_ds, pf_val=val_ds, pf_test=test_ds,
        pf_group_id=group_id,
        pf_params={
            "max_encoder_length": max_encoder_length,
            "max_prediction_length": max_prediction_length,
            "use_covariates": use_covariates,
            "target_log1p": target_log1p,
        },
    )


# ══════════════════════════════════════════════════════════════════════════
# Dispatch
# ══════════════════════════════════════════════════════════════════════════
def adapt(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None, y_test: Optional[np.ndarray] = None,
    *,
    feature_names: Iterable[str],
    kind: AdapterKind = AdapterKind.NUMPY,
    **kwargs,
) -> DataBundle:
    """Route to the appropriate to_* function.

    `kwargs` are forwarded to the pf_timeseries branch only (numpy /
    torch ignore them).
    """
    if kind == AdapterKind.NUMPY:
        return to_numpy(X_train, y_train, X_val, y_val, X_test, y_test,
                        feature_names=feature_names)
    if kind == AdapterKind.TORCH_DATASET:
        return to_torch_dataset(X_train, y_train, X_val, y_val, X_test, y_test,
                                feature_names=feature_names)
    if kind == AdapterKind.PF_TIMESERIES:
        return to_pf_timeseries_dataset(
            X_train, y_train, X_val, y_val, X_test, y_test,
            feature_names=feature_names, **kwargs,
        )
    raise ValueError(f"unknown AdapterKind: {kind!r}")
