"""Unified Phase-13 training-history persistence.

The public interface accepts raw records from DL loops, Optuna, Lightning, or
closed-form estimators and owns all normalization, CSV append, and plotting.
"""
from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

log = logging.getLogger(__name__)

_SCHEMA = {
    "model": pl.String,
    "scope": pl.String,
    "record_type": pl.String,
    "step": pl.Int64,
    "split": pl.String,
    "metric_name": pl.String,
    "value": pl.Float64,
    "params_json": pl.String,
    "saved_at": pl.String,
}
_PUBLIC_RECORD_TYPES = {
    "dl_epoch", "optuna_trial", "lightning_epoch", "closed_form",
}
_STEP_KEYS = {"epoch", "step", "iteration", "trial", "trial_number", "number", "seed"}


def save_training_record(
    model_name: str,
    scope: str,
    record_type: str,
    history_obj,
    out_dir: Path,
    params_json: str = "{}",
) -> Path:
    """Persist one training record as a long-format CSV plus PNG figures.

    Args:
        model_name: Registered model name.
        scope: ``"pooled"`` or a gu-code string.
        record_type: ``dl_epoch``, ``optuna_trial``, ``lightning_epoch``,
            or ``closed_form``. Phase-13's fitted-model hook may pass ``auto``;
            family detection remains encapsulated here.
        history_obj: Raw model history, Optuna study, Lightning metrics, metrics
            dict, or an ``{"model": fitted_model, "metrics": metrics}`` bundle.
        out_dir: Training-history directory, normally
            ``get_results_dir() / "training_history"``.
        params_json: Best-parameter JSON used when rows do not carry trial params.

    Returns:
        Path to ``<out_dir>/<model_name>_<scope>.csv``.

    Raises:
        ValueError: If ``record_type`` is unsupported.
        Exception: Persistence or plotting failures are logged and re-raised.

    Performance:
        O(existing CSV rows + new rows); append uses one ``pl.concat`` and one
        full-file UTF-8 rewrite. Summary rebuild scans all history CSVs.
    Side effects:
        Writes one CSV, one per-model PNG, and ``figures/summary_wis.png``.
    """
    out_path = Path(out_dir)
    figures_dir = out_path / "figures"
    out_path.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    try:
        normalized_params = _normalize_params_json(params_json)
        effective_type, effective_history = _resolve_record(record_type, history_obj)
        rows = _normalize_rows(
            model_name=model_name,
            scope=scope,
            record_type=effective_type,
            history_obj=effective_history,
            params_json=normalized_params,
        )
        if not rows:
            log.warning(
                "Training history empty for model=%s scope=%s type=%s; writing NaN marker",
                model_name, scope, effective_type,
            )
            rows = [_empty_row(model_name, scope, effective_type, normalized_params)]

        new_df = pl.DataFrame(rows, schema=_SCHEMA)
        csv_path = out_path / f"{_safe_name(model_name)}_{_safe_name(scope)}.csv"
        if csv_path.exists():
            existing = pl.read_csv(csv_path, schema_overrides=_SCHEMA)
            combined = pl.concat([existing, new_df], how="vertical_relaxed")
        else:
            combined = new_df
        combined.write_csv(csv_path)

        _plot_record(new_df, figures_dir / f"{_safe_name(model_name)}_{_safe_name(scope)}.png")
        _rebuild_summary_wis(out_path, figures_dir / "summary_wis.png")
        return csv_path
    except Exception:
        log.exception(
            "Training-history persistence failed for model=%s scope=%s type=%s",
            model_name, scope, record_type,
        )
        raise


def _resolve_record(record_type: str, history_obj: Any) -> tuple[str, Any]:
    """Resolve the fitted-model bundle used by the surgical Phase-13 hook."""
    if record_type != "auto":
        if record_type not in _PUBLIC_RECORD_TYPES:
            raise ValueError(f"Unsupported training record type: {record_type}")
        return record_type, history_obj

    if hasattr(history_obj, "trials_dataframe"):
        return "optuna_trial", history_obj

    if isinstance(history_obj, dict) and "model" in history_obj:
        model = history_obj.get("model")
        if model is not None and hasattr(model, "_history"):
            family = getattr(model, "_history_record_type", "dl_epoch")
            if family not in {"dl_epoch", "lightning_epoch"}:
                raise ValueError(f"Unsupported model history record type: {family}")
            return family, getattr(model, "_history")
        return "closed_form", history_obj.get("metrics")

    if isinstance(history_obj, dict):
        return "closed_form", history_obj
    raise ValueError("record_type='auto' requires an Optuna study, metrics dict, or fitted-model bundle")


def _normalize_rows(
    *,
    model_name: str,
    scope: str,
    record_type: str,
    history_obj: Any,
    params_json: str,
) -> list[dict[str, Any]]:
    if _is_empty(history_obj):
        return []
    if record_type == "optuna_trial":
        return _optuna_rows(model_name, scope, history_obj, params_json)
    if record_type in {"dl_epoch", "lightning_epoch"}:
        return _epoch_rows(model_name, scope, record_type, history_obj, params_json)
    return _closed_form_rows(model_name, scope, history_obj, params_json)


def _epoch_rows(
    model_name: str,
    scope: str,
    record_type: str,
    history_obj: Any,
    params_json: str,
) -> list[dict[str, Any]]:
    records = _records_from_history(history_obj)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        step = _int_or_default(
            record.get("epoch", record.get("step", record.get("iteration", index))),
            index,
        )
        for key, raw_value in record.items():
            if str(key) in _STEP_KEYS:
                continue
            value = _as_float(raw_value)
            if value is None:
                continue
            split, metric = _split_metric(str(key), default_split="train")
            rows.append(_row(
                model_name, scope, record_type, step, split, metric, value, params_json,
            ))
    return rows


def _optuna_rows(
    model_name: str,
    scope: str,
    history_obj: Any,
    params_json: str,
) -> list[dict[str, Any]]:
    if hasattr(history_obj, "trials_dataframe"):
        raw_df = history_obj.trials_dataframe()
        if isinstance(raw_df, pl.DataFrame):
            trial_records = raw_df.to_dicts()
        elif hasattr(raw_df, "to_dict"):
            try:
                trial_records = pl.DataFrame(raw_df.to_dict(orient="list")).to_dicts()
            except TypeError:
                trial_records = pl.DataFrame(raw_df.to_dict()).to_dicts()
        else:
            raise TypeError("Optuna trials_dataframe() returned an unsupported object")
    else:
        trial_records = _records_from_history(history_obj)

    rows: list[dict[str, Any]] = []
    for index, trial in enumerate(trial_records):
        if not isinstance(trial, dict):
            continue
        step = _int_or_default(
            trial.get("number", trial.get("trial_number", trial.get("trial", index))),
            index,
        )
        value = None
        for key in ("value", "objective", "oof_wis", "wis", "values_0"):
            if key in trial:
                value = _as_float(trial.get(key))
                if value is not None:
                    break
        if value is None:
            value = float("nan")

        trial_params = trial.get("params")
        if not isinstance(trial_params, dict):
            trial_params = {
                str(key)[len("params_"):]: value_
                for key, value_ in trial.items()
                if str(key).startswith("params_")
            }
        row_params = (
            _json_dumps(trial_params) if trial_params
            else params_json
        )
        rows.append(_row(
            model_name, scope, "optuna_trial", step, "optuna_trial", "WIS", value, row_params,
        ))
    return rows


def _closed_form_rows(
    model_name: str,
    scope: str,
    history_obj: Any,
    params_json: str,
) -> list[dict[str, Any]]:
    records = _records_from_history(history_obj)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        step = _int_or_default(record.get("step", record.get("iteration", 0)), 0)
        for key, raw_value in record.items():
            if str(key) in _STEP_KEYS:
                continue
            value = _as_float(raw_value)
            if value is None:
                continue
            split, metric = _split_metric(str(key), default_split="test")
            rows.append(_row(
                model_name, scope, "closed_form", step, split, metric, value, params_json,
            ))
    return rows


def _records_from_history(history_obj: Any) -> list[dict[str, Any]]:
    if isinstance(history_obj, pl.DataFrame):
        return history_obj.to_dicts()
    if isinstance(history_obj, dict):
        sequence_lengths = [
            len(value) for value in history_obj.values()
            if isinstance(value, (list, tuple, np.ndarray))
        ]
        if sequence_lengths:
            n_rows = max(sequence_lengths)
            records = []
            for index in range(n_rows):
                record = {}
                for key, value in history_obj.items():
                    if isinstance(value, (list, tuple, np.ndarray)):
                        if index < len(value):
                            record[key] = value[index]
                    else:
                        record[key] = value
                records.append(record)
            return records
        return [history_obj]
    if isinstance(history_obj, (list, tuple)):
        return list(history_obj)
    if hasattr(history_obj, "items"):
        return [dict(history_obj.items())]
    raise TypeError(f"Unsupported history object: {type(history_obj).__name__}")


def _row(
    model_name: str,
    scope: str,
    record_type: str,
    step: int,
    split: str,
    metric_name: str,
    value: float,
    params_json: str,
) -> dict[str, Any]:
    return {
        "model": str(model_name),
        "scope": str(scope),
        "record_type": record_type,
        "step": int(step),
        "split": split,
        "metric_name": metric_name,
        "value": float(value),
        "params_json": params_json,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }


def _empty_row(model_name: str, scope: str, record_type: str, params_json: str) -> dict[str, Any]:
    split = "optuna_trial" if record_type == "optuna_trial" else "train"
    metric = "WIS" if record_type == "optuna_trial" else "loss"
    return _row(model_name, scope, record_type, 0, split, metric, float("nan"), params_json)


def _split_metric(key: str, *, default_split: str) -> tuple[str, str]:
    raw = key.strip()
    lower = raw.lower()
    split = default_split
    for prefix in ("train_", "val_", "test_", "optuna_trial_"):
        if lower.startswith(prefix):
            split = prefix[:-1]
            raw = raw[len(prefix):]
            lower = raw.lower()
            break
    for suffix in ("_step", "_epoch"):
        if lower.endswith(suffix):
            raw = raw[:-len(suffix)]
            lower = raw.lower()
            break
    metric = {
        "wis": "WIS",
        "aic": "AIC",
        "bic": "BIC",
        "auc": "AUC",
        "mae": "MAE",
        "mape": "MAPE",
        "mase": "MASE",
        "rmse": "RMSE",
        "smape": "SMAPE",
    }.get(lower, lower)
    return split, metric


def _plot_record(df: pl.DataFrame, png_path: Path) -> None:
    plt = _get_pyplot()
    record_type = str(df["record_type"][0])
    fig, ax = plt.subplots(figsize=(8, 5))

    if record_type == "dl_epoch":
        _plot_dl(ax, df)
    elif record_type == "optuna_trial":
        _plot_optuna(ax, df)
    elif record_type == "lightning_epoch":
        _plot_lightning(ax, df)
    else:
        _plot_closed_form(ax, df)

    fig.tight_layout()
    fig.savefig(png_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_dl(ax, df: pl.DataFrame) -> None:
    model = str(df["model"][0])
    losses = df.filter(pl.col("metric_name") == "loss")
    train = _mean_series(losses.filter(pl.col("split") == "train"))
    val = _mean_series(losses.filter(pl.col("split") == "val"))
    ax.set_title(f"{model} - DL learning curve")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss", color="#1f77b4")
    if train:
        ax.plot(*zip(*train), color="#1f77b4", label="train_loss")
    ax2 = ax.twinx()
    ax2.set_ylabel("validation loss", color="#d62728")
    if val:
        ax2.plot(*zip(*val), color="#d62728", label="val_loss")
    if not train and not val:
        _annotate_empty(ax)


def _plot_optuna(ax, df: pl.DataFrame) -> None:
    model = str(df["model"][0])
    points = [
        (int(step), float(value))
        for step, value in zip(df["step"], df["value"])
        if math.isfinite(float(value))
    ]
    ax.set_title(f"{model} - Optuna objective")
    ax.set_xlabel("trial number")
    ax.set_ylabel("WIS")
    if not points:
        _annotate_empty(ax)
        return
    points.sort(key=lambda item: item[0])
    x = np.asarray([item[0] for item in points], dtype=float)
    y = np.asarray([item[1] for item in points], dtype=float)
    ax.scatter(x, y, color="#7f8c8d", alpha=0.7, label="objective")
    ax.plot(x, np.minimum.accumulate(y), color="#c0392b", lw=1.6, label="rolling minimum")
    ax.legend()


def _plot_lightning(ax, df: pl.DataFrame) -> None:
    model = str(df["model"][0])
    ax.set_title(f"{model} - Lightning metrics")
    ax.set_xlabel("epoch")
    ax.set_ylabel("metric value")
    plotted = False
    for key in df.select("split", "metric_name").unique().iter_rows():
        series = _mean_series(
            df.filter((pl.col("split") == key[0]) & (pl.col("metric_name") == key[1]))
        )
        if series:
            ax.plot(*zip(*series), label=f"{key[0]}_{key[1]}")
            plotted = True
    if plotted:
        ax.legend(fontsize=8)
    else:
        _annotate_empty(ax)


def _plot_closed_form(ax, df: pl.DataFrame) -> None:
    model = str(df["model"][0])
    latest: dict[str, float] = {}
    for metric, value in df.select("metric_name", "value").iter_rows():
        if math.isfinite(float(value)):
            latest[str(metric)] = float(value)
    items = list(latest.items())[:20]
    ax.set_title(f"{model} - final metrics")
    ax.set_ylabel("value")
    if not items:
        _annotate_empty(ax)
        return
    names, values = zip(*items)
    ax.bar(range(len(names)), values, color="#4c78a8")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=55, ha="right", fontsize=8)


def _rebuild_summary_wis(history_dir: Path, png_path: Path) -> None:
    candidates: dict[tuple[str, str], tuple[int, float]] = {}
    for csv_path in sorted(history_dir.glob("*.csv")):
        try:
            df = pl.read_csv(csv_path, schema_overrides=_SCHEMA)
        except Exception as exc:
            log.warning("Marked unreadable training-history CSV %s: %s", csv_path, exc)
            continue
        wis = df.filter(
            pl.col("metric_name").str.to_uppercase() == "WIS"
        ).filter(pl.col("value").is_finite())
        if wis.is_empty():
            continue
        for model, scope in wis.select("model", "scope").unique().iter_rows():
            group = wis.filter((pl.col("model") == model) & (pl.col("scope") == scope))
            val = group.filter(pl.col("split") == "val").sort("step")
            if not val.is_empty():
                candidate = (2, float(val["value"][-1]))
            else:
                optuna = group.filter(pl.col("split") == "optuna_trial")
                if optuna.is_empty():
                    continue
                candidate = (1, float(optuna["value"].min()))
            key = (str(model), str(scope))
            previous = candidates.get(key)
            if previous is None or candidate[0] >= previous[0]:
                candidates[key] = candidate

    plt = _get_pyplot()
    width = max(8.0, len(candidates) * 0.45)
    fig, ax = plt.subplots(figsize=(width, 5))
    ax.set_title("Final validation WIS by model")
    ax.set_ylabel("WIS")
    if candidates:
        ordered = sorted(candidates.items(), key=lambda item: item[1][1])
        labels = [
            model if scope == "pooled" else f"{model} [{scope}]"
            for (model, scope), _ in ordered
        ]
        values = [candidate[1] for _, candidate in ordered]
        ax.bar(range(len(labels)), values, color="#59a14f")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=65, ha="right", fontsize=8)
    else:
        _annotate_empty(ax)
    fig.tight_layout()
    fig.savefig(png_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _get_pyplot():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    from matplotlib import pyplot as plt

    try:
        font_manager.findfont("AppleGothic", fallback_to_default=False)
        plt.rcParams["font.family"] = "AppleGothic"
    except ValueError:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _mean_series(df: pl.DataFrame) -> list[tuple[int, float]]:
    if df.is_empty():
        return []
    grouped = df.group_by("step").agg(pl.col("value").mean()).sort("step")
    return [
        (int(step), float(value))
        for step, value in grouped.iter_rows()
        if math.isfinite(float(value))
    ]


def _annotate_empty(ax) -> None:
    ax.text(0.5, 0.5, "No finite training metrics", ha="center", va="center",
            transform=ax.transAxes)


def _as_float(value: Any) -> float | None:
    if value is None:
        return float("nan")
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (ValueError, TypeError):
            return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (ValueError, TypeError):
        return int(default)


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (dict, list, tuple, pl.DataFrame)):
        return len(value) == 0
    return False


def _normalize_params_json(params_json: str) -> str:
    try:
        parsed = json.loads(params_json or "{}")
    except (json.JSONDecodeError, TypeError):
        log.warning("Invalid params_json marked as raw text")
        parsed = {"raw": str(params_json)}
    return _json_dumps(parsed)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return safe.strip("._") or "unnamed"


__all__ = ["save_training_record"]
