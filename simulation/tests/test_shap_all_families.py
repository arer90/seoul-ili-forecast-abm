"""R11 (shap) SHAP — all-family coverage contract (2026-06-05).

Guards the comprehensive explainability layer:
  • Universal permutation importance covers EVERY family (model-agnostic, via
    ``artifact.predict``) — incl. covariate-free univariate-TS (→ ~0).
  • Native SHAP dispatches per family: tree→TreeExplainer, linear→LinearExplainer,
    dl→DeepExplainer/GradientExplainer (torch), kernel→KernelExplainer — each
    writes shap_values + figures.

Regression for the (n, p, 1) single-output torch squeeze bug that silently
dropped DL native SHAP.
"""
import numpy as np
import pytest
from pathlib import Path

from simulation.utils.model_artifact import make_artifact
from simulation.pipeline.shap_analysis import _permutation_importance, _explain_one

shap = pytest.importorskip("shap")
pytestmark = pytest.mark.filterwarnings("ignore")


class _Mock:
    """Minimal forecaster wrapper: ``.predict`` (numpy→numpy) + ``._model`` (raw)."""

    def __init__(self, raw, pf):
        self._model = raw
        self._pf = pf

    def predict(self, X, **k):
        return self._pf(np.asarray(X))


@pytest.fixture
def data():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(60, 6))
    y = 2 * X[:, 0] - X[:, 3] + 0.1 * rng.normal(size=60)
    return X, y, [f"f{i}" for i in range(6)]


def test_permutation_ranks_used_features():
    # self-contained + well-posed: y matches the predict_fn (uses f0, f2 only)
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 6))
    cols = [f"f{i}" for i in range(6)]
    pf = lambda Xi: 3 * Xi[:, 0] - 2 * Xi[:, 2]  # noqa: E731
    y = pf(X) + 0.1 * rng.normal(size=80)
    r = _permutation_importance(pf, X, y, cols, n_repeats=5)
    assert {f for f, _ in r[:2]} == {"f0", "f2"}, r


def test_permutation_zero_for_covariate_free_model(data):
    # univariate-TS family (ARIMA/Theta) ignores X → every feature ~0 (truthful)
    X, y, cols = data
    r = _permutation_importance(lambda Xi: np.full(len(Xi), y.mean()), X, y, cols)
    assert all(abs(v) < 1e-6 for _, v in r)


def _check(name, raw, pf, X, y, cols, tmp_path, family):
    st = _explain_one(name, make_artifact(model=_Mock(raw, pf), transform_name="identity"),
                      X, y, cols, Path(tmp_path))
    assert st["family"] == family, f"{name}: family {st['family']} != {family}"
    assert st["permutation"], f"{name}: universal permutation missing"
    assert st["native"], f"{name}: native SHAP missing"
    arts = {p.name for p in (Path(tmp_path) / name).iterdir()}
    assert {"importance.csv", "shap_values.npy"} <= arts, arts


def test_native_tree(data, tmp_path):
    from sklearn.ensemble import RandomForestRegressor
    X, y, cols = data
    rf = RandomForestRegressor(n_estimators=20, random_state=0).fit(X, y)
    _check("tree", rf, rf.predict, X, y, cols, tmp_path, "tree")


def test_native_linear(data, tmp_path):
    from sklearn.linear_model import Ridge
    X, y, cols = data
    rg = Ridge().fit(X, y)
    _check("linear", rg, rg.predict, X, y, cols, tmp_path, "linear")


def test_native_kernel(data, tmp_path):
    from sklearn.kernel_ridge import KernelRidge
    X, y, cols = data
    kr = KernelRidge(kernel="rbf").fit(X, y)
    _check("kernel", kr, kr.predict, X, y, cols, tmp_path, "kernel")


def test_native_dl_deepshap(data, tmp_path):
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    X, y, cols = data
    p = X.shape[1]

    class Net(nn.Module):
        def __init__(s):
            super().__init__()
            s.f = nn.Sequential(nn.Linear(p, 8), nn.ReLU(), nn.Linear(8, 1))

        def forward(s, x):
            return s.f(x)

    net = Net().eval()

    def npred(Xi):
        with torch.no_grad():
            return net(torch.as_tensor(np.asarray(Xi, dtype=np.float32))).numpy().ravel()

    _check("dl", net, npred, X, y, cols, tmp_path, "dl")
