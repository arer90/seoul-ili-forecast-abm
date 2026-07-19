"""G-312 (2026-06-18): prevent the two phase-13 stall-kills that lost SVR-Linear + TabPFN.

- SVR-Linear: SVR(kernel="linear") is libsvm SMO with max_iter=-1 (unbounded). A non-converging
  trial hung indefinitely → the isolate 900s stall-guard killed it (no .pt, dropped from ranking).
  Fix: cap max_iter on every SVR() in linear_models.py (converging fits unaffected; hangs terminate).
- TabPFN: a foundation model runs long INFERENCE without writing child.log → the 900s stall-guard
  false-killed it at 43min (the prior champion). Fix: quiet-foundation models get 3x stall tolerance.

macOS: run PER-FILE.
"""
import inspect


def test_g312_all_svr_calls_capped():
    """Every SVR() in linear_models.py passes a finite max_iter (no unbounded SMO hang)."""
    import simulation.models.linear_models as lm
    src = inspect.getsource(lm)
    import re
    svr_calls = re.findall(r"SVR\(kernel=\"(?:linear|rbf)\"[^\n]*", src)
    assert svr_calls, "expected SVR(kernel=...) calls in linear_models"
    for c in svr_calls:
        assert "max_iter" in c, f"SVR call missing max_iter (hang risk): {c.strip()}"


def test_g312_svr_max_iter_is_valid_and_terminates():
    """sklearn SVR accepts max_iter and a normal small fit converges/terminates."""
    import numpy as np
    from sklearn.svm import SVR
    rng = np.random.RandomState(0)
    X = rng.normal(size=(40, 3)); y = X[:, 0] * 2 + rng.normal(scale=0.1, size=40)
    m = SVR(kernel="linear", max_iter=200_000, C=1.0, epsilon=0.01).fit(X, y)
    assert m.predict(X).shape == (40,)


def test_g312_quiet_foundation_get_extended_stall():
    """Dispatch grants TabPFN/TiRex/TimesFM a larger isolate stall_timeout (3x)."""
    import simulation.pipeline.per_model_optimize as pmo
    src = inspect.getsource(pmo)
    assert '"TabPFN", "TiRex", "TimesFM-2.5"' in src, "quiet-foundation set must be in dispatch"
    assert "3.0 if mname in" in src, "stall_timeout must be multiplied for quiet-foundation"


def test_g312_normal_model_keeps_base_stall():
    """A non-foundation model is NOT in the extended-stall set (base 900s unchanged)."""
    import simulation.pipeline.per_model_optimize as pmo
    src = inspect.getsource(pmo)
    # the set is exactly the 3 quiet-foundation; SVR-Linear (hang fixed by max_iter) is excluded
    assert "SVR-Linear" not in src.split("3.0 if mname in")[1].split("else")[0]
