"""4-axis XAI organization (feature/input/output/model) from SHAP values.

사용자 명시: "shap에서 explainable AI으로 feature, input, output, model에 대해서 정리".
These pin the structure + the additive-identity-derived OUTPUT axis.
"""
import numpy as np

from simulation.pipeline.shap_analysis import _four_axis_explanation


def _synthetic_shap(n=40, p=5, seed=0):
    rng = np.random.default_rng(seed)
    sv = rng.normal(0, 0.1, size=(n, p))
    sv[:, 0] += rng.normal(0, 1.0, size=n)        # feature 0 dominant
    base = 5.0
    preds = base + sv.sum(axis=1)                 # additive identity
    return sv, preds, base


def test_four_axes_present_and_shaped():
    sv, preds, _ = _synthetic_shap()
    ax = _four_axis_explanation("M", sv, [f"feat{i}" for i in range(5)], "tree", preds)
    assert set(ax) == {"feature_axis", "input_axis", "output_axis", "model_axis"}
    assert ax["feature_axis"][0]["feature"] == "feat0"          # dominant ranked first
    assert ax["model_axis"]["dominant_feature"] == "feat0"


def test_output_axis_recovers_base_value():
    sv, preds, base = _synthetic_shap()
    ax = _four_axis_explanation("M", sv, [f"feat{i}" for i in range(5)], "tree", preds)
    assert abs(ax["output_axis"]["base_value"] - base) < 1e-6      # base + Σφ identity
    assert 0.0 <= ax["output_axis"]["fraction_explained_by_top3"] <= 1.0


def test_input_axis_representative_rows():
    sv, preds, _ = _synthetic_shap()
    ax = _four_axis_explanation("M", sv, [f"feat{i}" for i in range(5)], "tree", preds)
    inp = ax["input_axis"]
    assert {"highest_prediction", "lowest_prediction", "median_prediction"} <= set(inp)
    hi = inp["highest_prediction"]
    assert hi["prediction"] >= inp["lowest_prediction"]["prediction"]
    assert len(hi["top_drivers"]) == 3


def test_squeeze_3d_and_no_predictions():
    sv, _, _ = _synthetic_shap()
    sv3 = sv[:, :, None]                                # (n, p, 1) squeeze gotcha
    ax = _four_axis_explanation("M", sv3, [f"feat{i}" for i in range(5)], "kernel", None)
    assert ax["model_axis"]["n_features_explained"] == 5
    assert ax["output_axis"]["base_value"] is None     # no preds → degraded gracefully
    assert "representative" in ax["input_axis"]
