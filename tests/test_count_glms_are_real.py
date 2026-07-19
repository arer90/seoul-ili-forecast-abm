"""Guard: the count-family models must BE the models their names and the thesis claim.

``NegBinGLM`` and ``PoissonAutoreg`` shipped for months as ``RidgeCV`` — an identity-link
Gaussian fit — while the thesis printed an NB2 log link with a dispersion parameter and a
log-linear Poisson autoregression. Nothing caught it: the names were right, the numbers were
good, and no test ever asked what the estimator actually was.

This guard asks. It also pins the leak-free performance so a future "fix" cannot quietly
trade the model back for something that scores better by not being the model.
"""

from __future__ import annotations

import csv
import inspect
import math
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_CSV = _ROOT / "simulation" / "results" / "per_model_eval" / "per_model_metrics.csv"


def _row(model: str) -> dict:
    if not _CSV.exists():
        pytest.skip(f"{_CSV} absent")
    for r in csv.DictReader(_CSV.open(encoding="utf-8")):
        if r["model"] == model:
            return r
    pytest.skip(f"{model} absent from eval CSV")


def test_negbin_is_a_negative_binomial_glm() -> None:
    """NegBinGLM must fit an NB2 GLM with a log link — not a ridge regression."""
    from simulation.models.epi_models import NegBinGLMForecaster

    src = inspect.getsource(NegBinGLMForecaster.fit)
    assert "NegativeBinomial" in src, "NegBinGLM.fit does not fit a negative-binomial family"
    assert "RidgeCV" not in src, "NegBinGLM.fit is a ridge regression again"
    assert "dispersion" in src.lower(), "NegBinGLM does not estimate a dispersion parameter"


def test_poisson_autoreg_is_a_poisson_glm() -> None:
    """PoissonAutoreg must fit a Poisson GLM with a log link — not a ridge regression."""
    from simulation.models.epi_models import PoissonAutoregForecaster

    src = inspect.getsource(PoissonAutoregForecaster.fit)
    assert "PoissonRegressor" in src, "PoissonAutoreg.fit does not fit a Poisson family"
    assert "RidgeCV" not in src, "PoissonAutoreg.fit is a ridge regression again"


def test_log_link_design_uses_log_lags() -> None:
    """The lag block must enter in logs, and the whole design must be standardised.

    Raw lags through a log link explode on the out-of-range test peak (measured R2 = -281);
    an unstandardised lag block makes the L2 penalty land unevenly and costs ~4 points of R2.
    Both were real defects in the first implementation.
    """
    from simulation.models.epi_models import _LogLinkGLM

    src = inspect.getsource(_LogLinkGLM._design)
    assert "log1p" in src, "log-link design does not put the lags in logs"
    assert "transform" in src, "log-link design does not standardise the full matrix"


def test_intercept_is_not_penalised() -> None:
    """statsmodels penalises the constant by default; in a log link that under-predicts."""
    from simulation.models.epi_models import _SMGlm

    src = inspect.getsource(_SMGlm.fit)
    assert "a[0] = 0.0" in src, "the NB2 fit penalises its intercept (shrinks mu toward 1)"


@pytest.mark.parametrize("model,max_oof,min_r2", [
    ("NegBinGLM", 1.80, 0.88),
    ("PoissonAutoreg", 1.80, 0.88),
])
def test_count_glms_still_perform(model: str, max_oof: float, min_r2: float) -> None:
    """Leak-free OOF-WIS and test R2 must stay in the band the true GLMs reached.

    Measured 2026-07-15: NegBinGLM oof 1.6896 / R2 0.9022, PoissonAutoreg oof 1.7015 / R2 0.9012.
    Both sit inside the G-339 1-SE champion band. A regression past these bounds means the
    estimator changed or the penalty selection broke.
    """
    r = _row(model)
    oof, r2 = float(r["oof_wis"]), float(r["r2"])
    assert not math.isnan(oof) and oof <= max_oof, f"{model} OOF-WIS regressed to {oof}"
    assert not math.isnan(r2) and r2 >= min_r2, f"{model} test R2 regressed to {r2}"


def test_poisson_autoreg_has_a_native_interval() -> None:
    """The Poisson GLM has a predictive distribution, so its WIS must exist.

    Under the ridge impostor there was none: pi_source was "unavailable" and WIS was NaN, which
    is why the model carried a dash in Table 2 instead of a score.
    """
    r = _row("PoissonAutoreg")
    assert r["pi_source"] != "unavailable", "PoissonAutoreg has no native interval again"
    assert not math.isnan(float(r["wis"])), "PoissonAutoreg WIS is NaN again"
