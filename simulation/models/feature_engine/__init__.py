"""
Feature engineering package for the MPH infection simulation project.

This package provides a modular interface for advanced feature engineering,
combining external data sources with sophisticated transformations.

Main exports (backward compatible):
    - build_enriched_features: Main pipeline orchestrator
    - select_features: Feature selection via mutual information
    - _categorize_features: Feature categorization into groups
    - TimeSeriesAugmentor: Time series data augmentation class

Submodules:
    - loaders: Data loading functions (_load_*)
    - transforms: Feature transformation functions (_add_*)
    - builder: Orchestration and feature selection
    - utils: Database utilities, helpers, and augmentation
"""

from __future__ import annotations

# Re-export public API for backward compatibility
from .builder import build_enriched_features, select_features, _categorize_features, load_optuna_features
from .utils import TimeSeriesAugmentor

__all__ = [
    "build_enriched_features",
    "select_features",
    "_categorize_features",
    "load_optuna_features",
    "TimeSeriesAugmentor",
]
