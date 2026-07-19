"""PART A (transform-fix reconciliation, 2026-06-21): internal hardcoded y-transform removed.

Background: NegBinGLM's hardcoded internal log1p (fit) → expm1 (predict) explodes on an
out-of-range test peak (test 100.7 vs train_max 66.9), giving test R²=-0.41; identity gives
0.911 (verified on real KR ILI). The transform must be DATA-DRIVEN via the preproc Optuna
search (preproc supplies the single transform), not hardcoded inside each model.

This pass removes the internal log1p/expm1 (NegBinGLM, GAM-Spline, PoissonAutoreg, SARIMA)
and the softplus/log1p coupling (_PfBase) so each model fits on RAW y (identity); the
preproc layer then chooses the single y-transform per model.

These are SOURCE-level guards (no internal log1p/expm1 left) + a behavioral check that the
raw-y NegBinGLM no longer explodes on an out-of-range peak.
"""
from __future__ import annotations

import ast
import inspect
import io
import textwrap
import tokenize

import numpy as np


# ── source-level: no internal log1p/expm1 inside the changed models ─────────────

def _strip_comments_and_docstrings(src: str) -> str:
    """Return executable code only — comments, string literals, and docstrings removed.

    The guard is about the actual transform CALLS (np.log1p / np.expm1 / np.log / softplus arg),
    not about the word appearing in an explanatory comment, so we tokenize away comments/strings
    before substring-checking. Indentation is normalized so the method body parses standalone.
    """
    src = textwrap.dedent(src)
    out_tokens = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            out_tokens.append(tok)
        return tokenize.untokenize(out_tokens)
    except (tokenize.TokenError, IndentationError):
        # fallback: drop comment lines only
        return "\n".join(ln.split("#", 1)[0] for ln in src.splitlines())


def _fit_predict_src(cls, methods=("fit", "fit_series", "predict", "forecast", "rolling_1step")) -> str:
    """Concatenated, comment/string-stripped source of the given methods for a model class."""
    parts = []
    for meth in methods:
        fn = getattr(cls, meth, None)
        if fn is not None and callable(fn):
            try:
                parts.append(_strip_comments_and_docstrings(inspect.getsource(fn)))
            except (OSError, TypeError):
                pass
    return "\n".join(parts)


def test_negbinglm_no_internal_log1p_expm1():
    from simulation.models.epi_models import NegBinGLMForecaster
    src = _fit_predict_src(NegBinGLMForecaster)
    assert "np.log1p" not in src, "NegBinGLM still applies internal log1p in fit (must be data-driven)"
    assert "np.expm1" not in src, "NegBinGLM still applies internal expm1 in predict (must be data-driven)"


def test_gam_no_internal_log1p_expm1():
    from simulation.models.epi_models import GAMForecaster
    src = _fit_predict_src(GAMForecaster)
    assert "log1p" not in src, "GAM-Spline still log1p-transforms y internally"
    assert "expm1" not in src, "GAM-Spline still expm1-inverts y internally"


def test_poissonautoreg_no_internal_log_exp():
    from simulation.models.epi_models import PoissonAutoregForecaster
    src = _fit_predict_src(PoissonAutoregForecaster)
    # the AR-lag branch's np.log(prev) must also be gone (Ridge-AR on raw y)
    assert "np.log(" not in src, "PoissonAutoreg still applies np.log() to y/lags"
    assert "np.exp(" not in src, "PoissonAutoreg still applies np.exp() inverse"


def test_sarima_no_internal_log1p_expm1():
    from simulation.models.ts_models import SARIMAForecaster
    src = _fit_predict_src(SARIMAForecaster)
    assert "log1p" not in src, "SARIMA still log1p-transforms the series internally (G-271 path)"
    assert "expm1" not in src, "SARIMA still expm1-inverts the forecast internally"


def test_pfbase_no_softplus_or_log1p_coupling():
    from simulation.models.modern_ts.pf_models import _PfBase
    # _build_training_dataset (default), fit, predict — DeepAR's override is a separate (deferred)
    # class and is intentionally NOT checked here (PART A leaves it as-is with a flag comment).
    src = _fit_predict_src(_PfBase, methods=("_build_training_dataset", "fit", "predict"))
    assert "softplus" not in src, "_PfBase still couples softplus normalizer to y sign"
    assert "log1p" not in src, "_PfBase still applies internal log1p to y"
    assert "expm1" not in src, "_PfBase still applies internal expm1 inverse"


# ── behavioral: raw-y NegBinGLM does NOT explode on an out-of-range test peak ───

def _ili_like_with_oor_peak(seed=42):
    """Train max ~30, test contains an out-of-range peak ~100 (the explosion trigger)."""
    rng = np.random.default_rng(seed)
    n_train, n_test, p = 200, 20, 30
    t = np.arange(n_train + n_test)
    base = 6.0 + 8.0 * np.maximum(np.sin(2 * np.pi * t / 52 - 1.0), 0)
    noise = rng.normal(0, 1.0, n_train + n_test)
    y = np.maximum(base + noise, 0.1)
    # force an out-of-range peak in the test window
    y[n_train + 5] = 100.0
    X = rng.standard_normal((n_train + n_test, p))
    X[:, 0] = np.roll(y, 1)
    X[:, 1] = np.roll(y, 2)
    X[:, 2] = np.sin(2 * np.pi * t / 52)
    return (X[:n_train].astype(np.float64), y[:n_train].astype(np.float64),
            X[n_train:].astype(np.float64), y[n_train:].astype(np.float64))


def test_negbinglm_raw_does_not_explode_on_oor_peak():
    from simulation.models.epi_models import NegBinGLMForecaster
    X_tr, y_tr, X_te, y_te = _ili_like_with_oor_peak()
    train_max = float(y_tr.max())
    m = NegBinGLMForecaster()
    m.fit(X_tr, y_tr)
    pred = m.predict(X_te)
    assert np.all(np.isfinite(pred)), "NegBinGLM raw predictions contain NaN/inf"
    # the explosion symptom was pred >> train_max (expm1 of out-of-range log). With identity
    # fit + the retained 2×train_max clip, predictions stay bounded near training scale.
    assert pred.max() <= 2.0 * train_max + 1e-6, (
        f"NegBinGLM raw pred max {pred.max():.1f} exceeds 2×train_max {2*train_max:.1f} (explosion)")
