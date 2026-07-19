"""
simulation/models/expanding_cv.py
=================================
Expanding Window Cross-Validation for time series forecasting.

Purpose:
  Evaluate forecasting models using expanding window CV, where:
  - Initial train window: min_train_weeks
  - Each fold: expand training window, predict next h steps
  - Collect OOF (out-of-fold) predictions for ensemble analysis
  - Compute per-fold and aggregated metrics

Example (min_train=104, step=26, h=13):
  Fold 1: train=[0:104],   test=[104:117]    (h=13 steps)
  Fold 2: train=[0:130],   test=[130:143]    (h=13 steps)
  Fold 3: train=[0:156],   test=[156:169]    (h=13 steps)
  ...
  Continue until test window reaches end of data

Usage:
  from simulation.models.expanding_cv import run_expanding_cv
  
  cv_results = run_expanding_cv(
      feat_df, 
      target_col="ili_rate",
      target_transform="log1p",
      min_train=104,
      step=26,
      horizon=13
  )
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import polars as pl

log = logging.getLogger(__name__)


def _safe_print(*args, **kwargs):
    """
    Wrap print() with OSError handling for Windows console issues.
    Uses ASCII-only output (no Korean text in print).
    """
    try:
        print(*args, **kwargs)
    except OSError as e:
        log.debug(f"Print failed (console issue): {e}")


def _progress_bar(current: int, total: int, width: int = 30) -> str:
    """
    ASCII-only progress bar (no Unicode symbols).
    
    Args:
        current: current iteration (0-based)
        total: total iterations
        width: bar width in characters
    
    Returns:
        bar string like "[=========>      ] 67%"
    """
    if total == 0:
        return "[" + "=" * width + "] 100%"
    
    percent = current / total
    filled = int(width * percent)
    bar = "[" + "=" * filled + ">" + " " * (width - filled - 1) + "]"
    pct = f"{percent*100:.0f}%"
    return f"{bar} {pct}"


# ══════════════════════════════════════════════════════════════
# ExpandingWindowCV Class
# ══════════════════════════════════════════════════════════════

class ExpandingWindowCV:
    """
    Time series expanding window cross-validation.
    
    Evaluates forecasting models by expanding the training window
    and predicting the next h steps at each fold. Collects OOF predictions
    for ensemble stacking and out-of-sample evaluation.
    
    Parameters:
        min_train_weeks: Minimum training window size (default: 104 = 2 years)
        step_weeks: Window step/expansion size per fold (default: 26 = 6 months)
        horizon: Forecast horizon (steps ahead, default: 13 = 3 months)
        target_transform: Target transformation method.
            Sprint 1.5 R6 (2026-05-26): expanded to hierarchical menu —
              {"log1p", "sqrt", "boxcox", "asinh", "rank", "mcmc_robust",
               "laplace", "yeo_johnson", "gaussian", "anscombe",
               "freeman_tukey", "arcsine_sqrt", "none"}.
            "robust" now aliases to "mcmc_robust" (audit Q3 (a) — same
            semantic intent: robust-statistic-based Y rescaling).
    
    Attributes:
        fold_results: List of per-fold metric dicts
        model_summary: DataFrame with mean/std metrics per model
        oof_predictions: Dict[model_name, np.ndarray] of OOF predictions
        oof_actual: np.ndarray of actual values corresponding to OOF predictions
    """
    
    def __init__(
        self,
        min_train_weeks: int = 104,
        step_weeks: int = 26,
        horizon: int = 13,
        target_transform: str = "log1p",
    ):
        self.min_train_weeks = min_train_weeks
        self.step_weeks = step_weeks
        self.horizon = horizon
        self.target_transform = target_transform
        
        self.fold_results = []
        self.model_summary = None
        self.oof_predictions = {}
        self.oof_actual = None
    
    def _get_folds(self, n_samples: int) -> list[tuple[int, int, int]]:
        """
        Generate expanding window fold boundaries.
        
        Args:
            n_samples: Total number of time steps
        
        Returns:
            List of (train_end, test_start, test_end) tuples
        """
        folds = []
        train_end = self.min_train_weeks
        
        while train_end + self.horizon <= n_samples:
            test_start = train_end
            test_end = min(train_end + self.horizon, n_samples)
            
            # Only create fold if test window is full horizon
            if test_end - test_start == self.horizon:
                folds.append((train_end, test_start, test_end))
                train_end += self.step_weeks
            else:
                break
        
        return folds
    
    def run(
        self,
        feat_df: pl.DataFrame,
        target_col: str = "ili_rate",
        model_classes: Optional[list] = None,
        skip_categories: Optional[list] = None,
    ) -> dict:
        """
        Execute expanding window CV.

        Args:
            feat_df: DataFrame with features and target (pl.DataFrame or pandas.DataFrame accepted)
            target_col: Name of target column
            model_classes: List of model classes to evaluate (if None, use REGISTRY)
            skip_categories: Categories to skip (e.g., ["dl", "physics"])

        Returns:
            Dictionary containing:
                - fold_results: List of per-fold results
                - model_summary: DataFrame with aggregated metrics
                - oof_predictions: Dict[model_name, predictions]
                - oof_actual: Actual target values for OOF periods
                - n_folds: Number of completed folds
        """

        # Convert polars to pandas if needed (for backward compat with pandas input)
        if isinstance(feat_df, pl.DataFrame):
            feat_df_pandas = feat_df.to_pandas()
        else:
            feat_df_pandas = feat_df

        # Import models from registry
        if model_classes is None:
            from simulation.models.base import REGISTRY
            skip_cats = skip_categories or []
            model_classes = REGISTRY.get_available(
                data_size=len(feat_df_pandas),
                exclude_categories=skip_cats
            )

        if not model_classes:
            log.error("No models available after filtering")
            return {
                "fold_results": [],
                "model_summary": None,
                "oof_predictions": {},
                "oof_actual": None,
                "n_folds": 0,
            }

        # Setup
        from simulation.models.target_transform import TargetTransformer

        if target_col not in feat_df_pandas.columns:
            log.error(f"Target column '{target_col}' not in DataFrame")
            return {
                "fold_results": [],
                "model_summary": None,
                "oof_predictions": {},
                "oof_actual": None,
                "n_folds": 0,
            }

        # Extract features and target
        X_data = feat_df_pandas.drop(columns=[target_col]).values
        y_data = feat_df_pandas[target_col].values
        
        # Get fold boundaries
        folds = self._get_folds(len(y_data))
        n_folds = len(folds)
        
        if n_folds == 0:
            log.error("Cannot create any folds with given parameters")
            return {
                "fold_results": [],
                "model_summary": None,
                "oof_predictions": {},
                "oof_actual": None,
                "n_folds": 0,
            }
        
        _safe_print(f"\nExpanding Window CV: {n_folds} folds, {len(model_classes)} models")
        _safe_print(f"  min_train={self.min_train_weeks}, step={self.step_weeks}, horizon={self.horizon}\n")
        
        # Initialize OOF arrays
        oof_length = n_folds * self.horizon
        oof_actual = np.zeros(oof_length)
        oof_predictions_dict = {model.__class__.__name__: np.zeros(oof_length) for model in model_classes}
        
        # Fold-by-fold evaluation
        for fold_idx, (train_end, test_start, test_end) in enumerate(folds):
            _safe_print(f"Fold {fold_idx+1}/{n_folds}: train=[0:{train_end}], test=[{test_start}:{test_end}]")
            
            fold_result = {
                "fold": fold_idx + 1,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "models": {}
            }
            
            # Prepare data
            X_train = X_data[:train_end]
            y_train = y_data[:train_end]
            X_test = X_data[test_start:test_end]
            y_test = y_data[test_start:test_end]
            
            # Target transformation (fit on train only)
            # Sprint 1.5 R6 (2026-05-26): "robust" → "mcmc_robust" alias for
            # hierarchical compatibility (audit Q3 (a)). TargetTransformer is
            # preserved as the wrapper (it's still used by main runner +
            # phase4_baseline), only the name normalization happens here.
            _tt_method = ("mcmc_robust"
                          if self.target_transform == "robust"
                          else self.target_transform)
            tt = TargetTransformer(method=_tt_method)
            y_train_t = tt.fit_transform(y_train)
            
            # Evaluate each model
            for model_idx, model in enumerate(model_classes):
                model_name = model.__class__.__name__
                
                try:
                    # Check if model is time-series only
                    is_ts = hasattr(model, 'fit_series') and not hasattr(model, 'fit')
                    
                    if is_ts:
                        # Time series model
                        model.fit_series(y_train_t)
                        y_pred_t = model.forecast(self.horizon)
                    else:
                        # Standard model with features
                        model.fit(X_train, y_train_t)
                        y_pred_t = model.predict(X_test)
                    
                    # Inverse transform
                    y_pred = tt.inverse_transform(y_pred_t)
                    
                    # Ensure output shape
                    y_pred = np.atleast_1d(y_pred).flatten()
                    if len(y_pred) != self.horizon:
                        y_pred = np.resize(y_pred, self.horizon)
                    
                    # Metrics
                    residuals = y_test - y_pred
                    mse = float(np.mean(residuals ** 2))
                    rmse = float(np.sqrt(mse))
                    mae = float(np.mean(np.abs(residuals)))
                    
                    # R²
                    ss_res = float(np.sum(residuals ** 2))
                    ss_tot = float(np.sum((y_test - np.mean(y_test)) ** 2))
                    r2 = 1.0 - (ss_res / (ss_tot + 1e-10))
                    
                    # MAPE (avoid division by zero)
                    y_test_nonzero = np.where(np.abs(y_test) < 1e-6, 1e-6, y_test)
                    mape = float(np.mean(np.abs((y_test - y_pred) / y_test_nonzero)))
                    
                    fold_result["models"][model_name] = {
                        "R2": r2,
                        "RMSE": rmse,
                        "MAE": mae,
                        "MAPE": mape,
                    }
                    
                    # Store OOF prediction
                    oof_idx_start = fold_idx * self.horizon
                    oof_idx_end = oof_idx_start + self.horizon
                    oof_predictions_dict[model_name][oof_idx_start:oof_idx_end] = y_pred
                    
                except Exception as e:
                    log.warning(f"  [{model_name}] fold {fold_idx+1} failed: {type(e).__name__}")
                    fold_result["models"][model_name] = {
                        "R2": np.nan,
                        "RMSE": np.nan,
                        "MAE": np.nan,
                        "MAPE": np.nan,
                    }
            
            # Store OOF actual values
            oof_idx_start = fold_idx * self.horizon
            oof_idx_end = oof_idx_start + self.horizon
            oof_actual[oof_idx_start:oof_idx_end] = y_test
            
            self.fold_results.append(fold_result)
            
            # Progress indicator
            bar = _progress_bar(fold_idx, n_folds)
            _safe_print(f"  {bar}")
        
        # Aggregate results into summary table
        self._aggregate_results()
        
        self.oof_actual = oof_actual
        self.oof_predictions = oof_predictions_dict
        
        return {
            "fold_results": self.fold_results,
            "model_summary": self.model_summary,
            "oof_predictions": self.oof_predictions,
            "oof_actual": self.oof_actual,
            "n_folds": n_folds,
        }
    
    def _aggregate_results(self) -> None:
        """
        Aggregate per-fold results into a summary table.
        Computes mean and std of R², RMSE, MAE, MAPE across folds.
        """
        if not self.fold_results:
            self.model_summary = pd.DataFrame()
            return
        
        # Collect all model names
        model_names = set()
        for fold_result in self.fold_results:
            model_names.update(fold_result["models"].keys())
        
        summary_rows = []
        for model_name in sorted(model_names):
            metrics = {"R2": [], "RMSE": [], "MAE": [], "MAPE": []}
            
            for fold_result in self.fold_results:
                if model_name in fold_result["models"]:
                    m = fold_result["models"][model_name]
                    metrics["R2"].append(m["R2"])
                    metrics["RMSE"].append(m["RMSE"])
                    metrics["MAE"].append(m["MAE"])
                    metrics["MAPE"].append(m["MAPE"])
            
            if metrics["R2"]:
                summary_rows.append({
                    "Model": model_name,
                    "Mean_R2": float(np.nanmean(metrics["R2"])),
                    "Std_R2": float(np.nanstd(metrics["R2"])),
                    "Mean_RMSE": float(np.nanmean(metrics["RMSE"])),
                    "Std_RMSE": float(np.nanstd(metrics["RMSE"])),
                    "Mean_MAE": float(np.nanmean(metrics["MAE"])),
                    "Std_MAE": float(np.nanstd(metrics["MAE"])),
                    "Mean_MAPE": float(np.nanmean(metrics["MAPE"])),
                    "Std_MAPE": float(np.nanstd(metrics["MAPE"])),
                    "N_Folds": len([m for m in metrics["R2"] if not np.isnan(m)]),
                })
        
        self.model_summary = pd.DataFrame(summary_rows)


# ══════════════════════════════════════════════════════════════
# Convenience Function
# ══════════════════════════════════════════════════════════════

def run_expanding_cv(
    feat_df: pd.DataFrame,
    target_col: str = "ili_rate",
    target_transform: str = "log1p",
    skip_dl: bool = True,
    skip_physics: bool = False,
    min_train: int = 104,
    step: int = 26,
    horizon: int = 13,
) -> dict:
    """
    One-line convenience wrapper for expanding window CV.
    
    Args:
        feat_df: DataFrame with features and target
        target_col: Target column name
        target_transform: Transformation method
        skip_dl: Skip deep learning models
        skip_physics: Skip physics-based models
        min_train: Minimum training window (weeks)
        step: Window expansion step (weeks)
        horizon: Forecast horizon (weeks)
    
    Returns:
        CV results dictionary
    """
    skip_cats = []
    if skip_dl:
        # G-236 후속: "modern_ts" 는 dead 라벨 — modern-ts 모델은 meta.category=="dl"
        # 이므로 "dl" 하나로 커버 (runner._SUBPROCESS_CATEGORIES 와 동일 혼동 회피).
        skip_cats.append("dl")
    if skip_physics:
        skip_cats.append("physics")
    
    cv = ExpandingWindowCV(
        min_train_weeks=min_train,
        step_weeks=step,
        horizon=horizon,
        target_transform=target_transform,
    )
    
    return cv.run(
        feat_df,
        target_col=target_col,
        skip_categories=skip_cats if skip_cats else None,
    )
