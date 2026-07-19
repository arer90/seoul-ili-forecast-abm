"""Component ablation for the FusedEpi champion forecaster.

This script is intentionally isolated from the live pipeline/model code.  It
uses the frozen run_data split, the pipeline BASIC eval feature subset, the
same rolling one-step y_observed protocol used for FusedEpi evaluation, and
imports the existing WIS scorer used by the R10 adaptive-conformal path.

Side effects:
    Writes one JSON file under /private/tmp/.../scratchpad/ablation.
"""
from __future__ import annotations

import csv
import gc
import io
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Callable

if sys.stdout.encoding and sys.stdout.encoding.lower().replace("-", "") != "utf8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("MPH_EVAL_FEATURES", "basic")
os.environ.setdefault("OPTUNA_ISOLATE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")

import numpy as np

from simulation.analytics.adaptive_conformal import wis_from_bounds
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS, FLUSIGHT_QUANTILES
from simulation.models.fused_epi import FusedEpiForecaster
from simulation.pipeline.config import PipelineConfig
from simulation.pipeline.data import run_data
from simulation.pipeline.runner import _resolve_eval_features


LOG = logging.getLogger("ablation_fusedepi")
OUT_JSON = Path(
    os.environ.get("MPH_SCRATCH", str(Path(__file__).resolve().parents[1] / "_scratch")) + "/ablation/"
    "fusedepi_ablation.json"
)


def seed_all(seed: int = 42) -> None:
    """Set deterministic seeds for numpy, random, and torch when available."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


class _ZeroCorrection:
    """Correction model that returns zero residual adjustment."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=float)


class _AlphaOverrideFusedEpi(FusedEpiForecaster):
    """FusedEpi fit path with only the alpha rule replaced."""

    alpha_mode = "full"

    def _alpha_from_errors(self, n: int, base_err: float, corr_err: float) -> float:
        alpha_size = float(np.clip(n / self.n_ref, self.alpha_min, 1.0))
        if self.alpha_mode == "static_alpha":
            return 1.0
        if self.alpha_mode == "no_donoharm":
            return alpha_size
        harm = float(np.clip(base_err / (corr_err + 1e-9), 0.0, 1.0))
        return alpha_size * harm

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "_AlphaOverrideFusedEpi":
        from tirex import load_model

        y = np.asarray(y_train, dtype=float).ravel()
        X = np.asarray(X_train, dtype=float)
        self._train_y = y
        self._y_max = float(np.max(y)) if y.size else 0.0
        n = len(y)
        if self._tx is None:
            self._tx = load_model(self.repo_id, device="cpu")

        tr_idx = list(range(self.min_ctx, n))
        tx_tr = self._tirex_roll(y, tr_idx)
        resid = y[self.min_ctx :] - tx_tr
        yf = y[self.min_ctx :]

        mech = self._mech_features(y)
        Xf_all = np.hstack([X, mech])[self.min_ctx :]

        K = max(10, int(len(yf) * self.cal_frac))
        self._mc_keep = self._select_mc_keep(Xf_all, resid, K)
        Xf = Xf_all[:, self._mc_keep] if self._mc_keep is not None else Xf_all

        corr_pt = self._tab()
        corr_pt.fit(Xf[:-K], resid[:-K])
        corr_cal = np.asarray(corr_pt.predict(Xf[-K:]), dtype=float)

        base_err = float(np.mean(resid[-K:] ** 2))
        corr_err = float(np.mean((resid[-K:] - corr_cal) ** 2))
        self._alpha = float(self._alpha_from_errors(n, base_err, corr_err))

        fused_cal = np.clip(tx_tr[-K:] + self._alpha * corr_cal, 0.0, None)
        self._nb_disp = self._nb_dispersion(fused_cal, yf[-K:])
        self._conf = self._fit_conformal(fused_cal, yf[-K:])
        self._resid_scale = float(np.std(yf[-K:] - fused_cal)) + 1e-6
        self._decide_asym(yf[-K:] - fused_cal, len(yf[-K:]))

        self._corr = self._tab()
        self._corr.fit(Xf, resid)
        self._calib_residuals = (yf[-K:] - fused_cal).tolist()
        self._fitted = True
        return self


class NoAnchorFusedEpi(_AlphaOverrideFusedEpi):
    """Remove mechanistic anchor features; residual model sees base X only."""

    def _mech_features(self, y_full: np.ndarray) -> np.ndarray:
        return np.empty((len(np.asarray(y_full)), 0), dtype=float)


class StaticAlphaFusedEpi(_AlphaOverrideFusedEpi):
    """Use alpha=1.0; no size ramp and no do-no-harm shrinkage."""

    alpha_mode = "static_alpha"


class NoDoNoHarmFusedEpi(_AlphaOverrideFusedEpi):
    """Keep the n/n_ref size ramp, force harm=1.0."""

    alpha_mode = "no_donoharm"


class NoResidualFusedEpi(FusedEpiForecaster):
    """Disable TabPFN residual correction by forcing alpha=0 and zero correction."""

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> "NoResidualFusedEpi":
        from tirex import load_model

        y = np.asarray(y_train, dtype=float).ravel()
        self._train_y = y
        self._y_max = float(np.max(y)) if y.size else 0.0
        n = len(y)
        if self._tx is None:
            self._tx = load_model(self.repo_id, device="cpu")

        tr_idx = list(range(self.min_ctx, n))
        tx_tr = self._tirex_roll(y, tr_idx)
        yf = y[self.min_ctx :]

        K = max(10, int(len(yf) * self.cal_frac))
        self._mc_keep = None
        self._corr = _ZeroCorrection()
        self._alpha = 0.0

        fused_cal = np.clip(tx_tr[-K:], 0.0, None)
        self._nb_disp = self._nb_dispersion(fused_cal, yf[-K:])
        self._conf = self._fit_conformal(fused_cal, yf[-K:])
        self._resid_scale = float(np.std(yf[-K:] - fused_cal)) + 1e-6
        self._decide_asym(yf[-K:] - fused_cal, len(yf[-K:]))
        self._calib_residuals = (yf[-K:] - fused_cal).tolist()
        self._fitted = True
        return self


def load_split() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Load frozen data once and return train-pool/test arrays in BASIC feature space."""
    data = run_data(PipelineConfig())
    X_all = np.asarray(data["X_all"], dtype=float)
    y_all = np.asarray(data["y_all"], dtype=float).ravel()
    feature_cols = list(data["feature_cols"])
    X_eval, eval_cols, _basic_idx = _resolve_eval_features(
        X_all, feature_cols, eval_basic=True
    )
    pool_end = int(data["pool_end"])
    n_test = int(data["n_test"])
    test_start = pool_end
    test_end = test_start + n_test
    meta = {
        "n": int(data["n"]),
        "pool_end": pool_end,
        "test_start": test_start,
        "test_end": test_end,
        "n_test": n_test,
        "n_features_basic": int(X_eval.shape[1]),
        "feature_cols": eval_cols,
    }
    return (
        np.asarray(X_eval[:pool_end], dtype=float),
        y_all[:pool_end],
        np.asarray(X_eval[test_start:test_end], dtype=float),
        y_all[test_start:test_end],
        meta,
    )


def _quantile_array(q: dict, level: float) -> np.ndarray:
    for key, val in q.items():
        if abs(float(key) - float(level)) < 1e-12:
            return np.asarray(val, dtype=float)
    raise KeyError(f"quantile level {level} absent from predict_quantiles output")


def score_native_wis(model: FusedEpiForecaster, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Score model-native quantile intervals with the existing R10 WIS helper."""
    levels = tuple(float(q) for q in FLUSIGHT_QUANTILES)
    q = model.predict_quantiles(X_test, y_observed=y_test, levels=levels)
    bounds = {}
    for alpha in FLUSIGHT_ALPHAS:
        lo = _quantile_array(q, float(alpha) / 2.0)
        hi = _quantile_array(q, 1.0 - float(alpha) / 2.0)
        bounds[float(alpha)] = (lo, hi)
    median = _quantile_array(q, 0.5)
    wis_arr = wis_from_bounds(y_test, bounds, FLUSIGHT_ALPHAS, median=median)
    wis_arr = np.asarray(wis_arr, dtype=float)
    if not np.isfinite(wis_arr).any():
        raise RuntimeError("native WIS produced no finite values")
    return float(np.nanmean(wis_arr))


def load_reported_wis_reference() -> tuple[float | None, str | None]:
    """Read the current thesis-facing FusedEpi WIS artifact, if present."""
    p = ROOT / "simulation/results/per_model_eval/per_model_metrics.csv"
    if not p.exists():
        return None, None
    with p.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("model") == "FusedEpi":
                try:
                    return float(row["wis"]), str(p)
                except Exception:
                    return None, str(p)
    return None, str(p)


def cleanup_model(model) -> None:
    """Release per-variant objects without clearing FusedEpi's safe prediction caches."""
    del model
    gc.collect()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:
                pass
    except Exception:
        pass


def run_variant(
    name: str,
    factory: Callable[[], FusedEpiForecaster],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Fit one variant and score it on the same rolling hold-out test span."""
    seed_all(42)
    t0 = time.time()
    LOG.info("[%s] fit start", name)
    model = factory()
    model.fit(X_train, y_train)
    test_wis = score_native_wis(model, X_test, y_test)
    out = {
        "test_wis": float(test_wis),
        "delta_vs_full": None,
        "n_test": int(len(y_test)),
        "alpha": float(getattr(model, "_alpha", float("nan"))),
        "adaptive_conf": bool(getattr(model, "adaptive_conf", False)),
        "elapsed_sec": round(time.time() - t0, 3),
    }
    LOG.info("[%s] WIS=%.6f alpha=%.4f", name, test_wis, out["alpha"])
    cleanup_model(model)
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    seed_all(42)
    X_train, y_train, X_test, y_test, split_meta = load_split()
    LOG.info(
        "loaded split: train_pool=%s test=%s basic_features=%s",
        len(y_train),
        len(y_test),
        X_train.shape[1],
    )

    variants: list[tuple[str, Callable[[], FusedEpiForecaster]]] = [
        ("full", lambda: FusedEpiForecaster()),
        ("no_anchor", lambda: NoAnchorFusedEpi()),
        ("no_residual", lambda: NoResidualFusedEpi()),
        ("static_alpha", lambda: StaticAlphaFusedEpi()),
        ("no_donoharm", lambda: NoDoNoHarmFusedEpi()),
        ("no_adaptive_conformal", lambda: FusedEpiForecaster(adaptive_conf=False)),
    ]

    results: dict[str, dict] = {}
    for name, factory in variants:
        try:
            results[name] = run_variant(name, factory, X_train, y_train, X_test, y_test)
        except Exception as e:
            LOG.exception("[%s] failed", name)
            results[name] = {
                "test_wis": None,
                "delta_vs_full": None,
                "n_test": int(len(y_test)),
                "error": str(e),
            }

    full_wis = results.get("full", {}).get("test_wis")
    if isinstance(full_wis, (int, float)) and np.isfinite(full_wis):
        for name, row in results.items():
            wis = row.get("test_wis")
            if isinstance(wis, (int, float)) and np.isfinite(wis):
                row["delta_vs_full"] = float(wis - full_wis)

    ref_wis, ref_source = load_reported_wis_reference()
    close = None
    gap = None
    if isinstance(full_wis, (int, float)) and ref_wis is not None:
        gap = float(full_wis - ref_wis)
        close = bool(abs(gap) <= max(0.10, 0.05 * abs(ref_wis)))
    results["replication_check"] = {
        "full_test_wis": full_wis,
        "reported_reference_wis": ref_wis,
        "reported_reference_source": ref_source,
        "close_to_reported_reference": close,
        "gap_vs_reported_reference": gap,
        "protocol": (
            "run_data(PipelineConfig) frozen split; BASIC eval features via "
            "_resolve_eval_features; fit train+val pool; rolling test "
            "predict_quantiles(..., y_observed=y_test); WIS via "
            "simulation.analytics.adaptive_conformal.wis_from_bounds"
        ),
        "split": split_meta,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    LOG.info("wrote %s", OUT_JSON)
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
