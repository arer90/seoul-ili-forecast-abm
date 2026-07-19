"""PART D (transform-fix reconciliation, 2026-06-21): seed each y-transform once before TPE.

de27fdf's stage1 already enqueues a single identity anchor (G-329c, y_mode="none") + plateau-stop
+ stable pools. PART D EXTENDS that (does not overwrite) to also enqueue one startup trial per
y-transform in the search pool, so TPE starts with every transform observed once (a transform can
no longer be skipped by random startup). Budget-capped, skip_if_exists, try/except per enqueue.

Now that NegBinGLM / GAM-Spline / PoissonAutoreg etc. fit on raw y (PART A) and the preproc search
runs for them (PART C), this seeding guarantees the data-driven OOF actually evaluates each
candidate transform rather than relying on the TPE startup lottery to surface them.
"""
from __future__ import annotations


class _FakeStudy:
    """Minimal stand-in for an optuna study: records enqueued param dicts."""

    def __init__(self):
        self.enqueued = []

    def enqueue_trial(self, params, skip_if_exists=False):
        self.enqueued.append(dict(params))


def test_seed_helper_enqueues_each_transform_plus_identity():
    from simulation.pipeline._inline_optuna_3stage import _seed_y_transform_trials
    from simulation.pipeline.preproc_optuna_hierarchical import STABLE_Y_TRANSFORMS

    study = _FakeStudy()
    _seed_y_transform_trials(
        study, model_name="NegBinGLM",
        force_y_identity=False, force_x_identity=False,
        restrict_centered=False, n_trials=100,
    )
    # identity anchor present (y_mode="none")
    assert any(p.get("y_mode") == "none" for p in study.enqueued), "identity anchor missing"
    # one individual trial per stable y-transform
    seeded = {p.get("y_individual") for p in study.enqueued if p.get("y_mode") == "individual"}
    for t in STABLE_Y_TRANSFORMS:
        assert t in seeded, f"y-transform {t} was not seeded before TPE"
    # every enqueued trial forces x_mode="none" (x search unaffected by this y-seeding)
    assert all(p.get("x_mode") == "none" for p in study.enqueued if "x_mode" in p)


def test_seed_helper_respects_force_y_identity():
    """force_y_identity models (intrinsic transform) get ONLY the identity anchor — no y-search."""
    from simulation.pipeline._inline_optuna_3stage import _seed_y_transform_trials

    study = _FakeStudy()
    _seed_y_transform_trials(
        study, model_name="hhh4-equivalent",
        force_y_identity=True, force_x_identity=False,
        restrict_centered=False, n_trials=100,
    )
    # no individual y-transform trials when y is forced identity
    assert not any(p.get("y_mode") == "individual" for p in study.enqueued), (
        "force_y_identity model must not seed y-transform trials")


def test_seed_helper_restrict_centered_excludes_centered():
    """restrict_centered models (transformed-zero floor) exclude laplace/mcmc_robust from seeds."""
    from simulation.pipeline._inline_optuna_3stage import _seed_y_transform_trials

    study = _FakeStudy()
    _seed_y_transform_trials(
        study, model_name="BayesianRidge",
        force_y_identity=False, force_x_identity=False,
        restrict_centered=True, n_trials=100,
    )
    seeded = {p.get("y_individual") for p in study.enqueued if p.get("y_mode") == "individual"}
    assert "laplace" not in seeded and "mcmc_robust" not in seeded, (
        "centered transforms must be excluded for transformed-zero-floor models")


def test_seed_helper_budget_capped():
    """Seeds must not consume the whole budget — leave room for TPE exploitation."""
    from simulation.pipeline._inline_optuna_3stage import _seed_y_transform_trials

    study = _FakeStudy()
    _seed_y_transform_trials(
        study, model_name="NegBinGLM",
        force_y_identity=False, force_x_identity=False,
        restrict_centered=False, n_trials=4,    # tiny budget
    )
    assert len(study.enqueued) < 4, "seeds must be budget-capped (leave trials for TPE)"


def test_seed_helper_never_raises():
    """A broken study.enqueue_trial must not break stage1 (try/except per enqueue)."""
    from simulation.pipeline._inline_optuna_3stage import _seed_y_transform_trials

    class _BrokenStudy:
        def enqueue_trial(self, params, skip_if_exists=False):
            raise RuntimeError("boom")

    # must not propagate
    _seed_y_transform_trials(
        _BrokenStudy(), model_name="NegBinGLM",
        force_y_identity=False, force_x_identity=False,
        restrict_centered=False, n_trials=100,
    )
