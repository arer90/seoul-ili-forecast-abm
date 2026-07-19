"""VST (Variance-Stabilizing Transformer) primitives — Sprint 1.5 R7 (2026-05-26).

Shared sklearn ``FunctionTransformer`` builders for the 7 helpers that were
byte-identical between ``per_feature_preprocessor.py`` and
``grouped_preprocessor.py``. New code SHOULD import from this module to avoid
adding a third copy.

Audit reference: ``docs/MODELS_PREPROC_AUDIT.md`` §4-R7 (lean dedupe path).

Coverage (7 helpers, two name conventions both exported as aliases):

  log1p          — log1p(max(x, 0)) ↔ expm1(z)
  sqrt           — sqrt(max(x, 0)) ↔ z²
  yeo_johnson    — sklearn PowerTransformer(method="yeo-johnson", standardize=True)
  anscombe       — 2·sqrt(x + 3/8) ↔ max((z/2)² − 3/8, 0)         (Anscombe 1948)
  freeman_tukey  — sqrt(x) + sqrt(x+1) ↔ max(((z² − 1) / (2z))², 0) (Freeman-Tukey 1950)
  arcsinh        — asinh(x/scale) ↔ sinh(z)·scale                  (heavy-tail safe)
  arcsine_sqrt   — 2·arcsin(sqrt(clip(p, 0, 1))) ↔ sin(z/2)²       (proportion VST)

Inverse safety: the freeman_tukey inverse divides by ``z``; we guard with
``np.maximum(z, 1e-9)`` to avoid div-by-zero on extrapolated inverse calls.

Aliases (legacy names preserved):
  ``_log1p_t``  / ``_log1p_transformer``  → ``log1p_transformer``
  ``_sqrt_t``   / ``_sqrt_transformer``   → ``sqrt_transformer``
  ``_yeo_johnson_t`` / ``_yeo_johnson_safe`` → ``yeo_johnson_transformer``
  ``_anscombe_t`` / ``_anscombe_transformer`` → ``anscombe_transformer``
  ``_freeman_tukey_t`` / ``_freeman_tukey_transformer`` → ``freeman_tukey_transformer``
  ``_arcsinh_t`` / ``_arcsinh_transformer`` → ``arcsinh_transformer``
  ``_arcsine_sqrt_t`` / ``_arcsine_sqrt_transformer`` → ``arcsine_sqrt_transformer``
"""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import FunctionTransformer


# ─────────────────────────────────────────────────────────────────────
# 1. log1p
# ─────────────────────────────────────────────────────────────────────

def log1p_transformer():
    """log1p(max(x, 0)) ↔ expm1(z).  Stateless, no G-146 cap (caller adds if needed)."""
    return FunctionTransformer(
        func=lambda x: np.log1p(np.maximum(np.asarray(x, dtype=np.float64), 0.0)),
        inverse_func=lambda x: np.expm1(np.asarray(x, dtype=np.float64)),
        validate=False,
    )


# ─────────────────────────────────────────────────────────────────────
# 2. sqrt
# ─────────────────────────────────────────────────────────────────────

def sqrt_transformer():
    """sqrt(max(x, 0)) ↔ z².  Stateless."""
    return FunctionTransformer(
        func=lambda x: np.sqrt(np.maximum(np.asarray(x, dtype=np.float64), 0.0)),
        inverse_func=lambda x: np.asarray(x, dtype=np.float64) ** 2,
        validate=False,
    )


# ─────────────────────────────────────────────────────────────────────
# 3. yeo-johnson (sklearn standardize=True)
# ─────────────────────────────────────────────────────────────────────

def yeo_johnson_transformer():
    """PowerTransformer(method="yeo-johnson", standardize=True).

    Fitted state heavy — caller is responsible for fit/transform discipline.
    """
    from sklearn.preprocessing import PowerTransformer
    return PowerTransformer(method="yeo-johnson", standardize=True)


# ─────────────────────────────────────────────────────────────────────
# 4. Anscombe VST  (Poisson, mean > 1)
# ─────────────────────────────────────────────────────────────────────

def anscombe_transformer():
    """Anscombe (1948): 2·sqrt(x + 3/8) ↔ max((z/2)² − 3/8, 0).

    Optimal Poisson VST when E[X] > 1; for low-mean Poisson prefer
    ``freeman_tukey_transformer``.
    """
    return FunctionTransformer(
        func=lambda x: 2.0 * np.sqrt(np.maximum(np.asarray(x, dtype=np.float64), 0.0) + 0.375),
        inverse_func=lambda x: np.maximum((np.asarray(x, dtype=np.float64) / 2.0) ** 2 - 0.375, 0.0),
        validate=False,
    )


# ─────────────────────────────────────────────────────────────────────
# 5. Freeman-Tukey VST  (low-mean Poisson)
# ─────────────────────────────────────────────────────────────────────

def freeman_tukey_transformer():
    """Freeman-Tukey (1950): sqrt(x) + sqrt(x+1) ↔ max(((z²−1)/(2z))², 0).

    More stable than Anscombe when many zeros (low ILI counts during off-season).
    Inverse guards against z → 0 (div-by-zero) with a 1e-9 floor.
    """
    def _ft(x):
        x = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
        return np.sqrt(x) + np.sqrt(x + 1.0)

    def _ft_inv(z):
        z = np.asarray(z, dtype=np.float64)
        z = np.maximum(z, 1e-9)
        x = ((z * z - 1.0) / (2.0 * z)) ** 2
        return np.maximum(x, 0.0)

    return FunctionTransformer(func=_ft, inverse_func=_ft_inv, validate=False)


# ─────────────────────────────────────────────────────────────────────
# 6. arcsinh  (heavy-tail, signed input safe)
# ─────────────────────────────────────────────────────────────────────

def arcsinh_transformer(scale: float = 10.0):
    """asinh(x/scale) ↔ sinh(z)·scale.

    Linear near 0, log-like for |x| → ∞. Handles negative values and heavy
    tails without lambda fitting (unlike Box-Cox / Yeo-Johnson).

    Default scale=10.0 matches per_feature_preprocessor's historical
    ``_arcsinh_t(scale=10.0)`` for ILI rate range compatibility. Callers
    needing the grouped_preprocessor's scale=1.0 (or 100.0) must pass it
    explicitly — grouped_preprocessor's callers already do so at every
    site (L307/314/323/329).
    """
    return FunctionTransformer(
        func=lambda x, s=scale: np.arcsinh(np.asarray(x, dtype=np.float64) / s),
        inverse_func=lambda x, s=scale: np.sinh(np.asarray(x, dtype=np.float64)) * s,
        validate=False,
    )


# ─────────────────────────────────────────────────────────────────────
# 7. arcsine-sqrt  (proportion VST p ∈ [0, 1])
# ─────────────────────────────────────────────────────────────────────

def arcsine_sqrt_transformer():
    """2·arcsin(sqrt(p)) ↔ sin(z/2)².  Variance ≈ 1/(4n) under binomial.

    Input is clipped to [0, 1] for numerical safety; inverse output is in
    [0, 1] by construction (sin(·)² range).
    """
    return FunctionTransformer(
        func=lambda x: 2.0 * np.arcsin(np.sqrt(np.clip(np.asarray(x, dtype=np.float64), 0.0, 1.0))),
        inverse_func=lambda x: np.sin(np.asarray(x, dtype=np.float64) / 2.0) ** 2,
        validate=False,
    )


# ─────────────────────────────────────────────────────────────────────
# Legacy name aliases — caller can use either ``_log1p_t`` (per_feature) or
# ``_log1p_transformer`` (grouped). New code SHOULD use the un-prefixed name.
# ─────────────────────────────────────────────────────────────────────

# per_feature_preprocessor.py style (short)
_log1p_t          = log1p_transformer
_sqrt_t           = sqrt_transformer
_yeo_johnson_t    = yeo_johnson_transformer
_anscombe_t       = anscombe_transformer
_freeman_tukey_t  = freeman_tukey_transformer
_arcsinh_t        = arcsinh_transformer
_arcsine_sqrt_t   = arcsine_sqrt_transformer

# grouped_preprocessor.py style (verbose)
_log1p_transformer          = log1p_transformer
_sqrt_transformer           = sqrt_transformer
_yeo_johnson_safe           = yeo_johnson_transformer
_anscombe_transformer       = anscombe_transformer
_freeman_tukey_transformer  = freeman_tukey_transformer
_arcsinh_transformer        = arcsinh_transformer
_arcsine_sqrt_transformer   = arcsine_sqrt_transformer


__all__ = [
    # canonical names
    "log1p_transformer", "sqrt_transformer", "yeo_johnson_transformer",
    "anscombe_transformer", "freeman_tukey_transformer",
    "arcsinh_transformer", "arcsine_sqrt_transformer",
    # legacy aliases (per_feature)
    "_log1p_t", "_sqrt_t", "_yeo_johnson_t", "_anscombe_t",
    "_freeman_tukey_t", "_arcsinh_t", "_arcsine_sqrt_t",
    # legacy aliases (grouped)
    "_log1p_transformer", "_sqrt_transformer", "_yeo_johnson_safe",
    "_anscombe_transformer", "_freeman_tukey_transformer",
    "_arcsinh_transformer", "_arcsine_sqrt_transformer",
]
