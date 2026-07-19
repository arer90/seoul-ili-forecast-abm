"""G-329b/c (3AI 최종 preproc 설계): floor empty-guard + identity-anchor.

배경: G-329 가 STABLE_Y 를 affine-only(laplace/mcmc_robust)로 줄이며 _NONCENTERED_STABLE_Y(=centered
제외) 가 빈 set → floor 모델(restrict_centered_y) individual trial 서 suggest_categorical([]) ValueError
폭사(실측 23/40). + 사용자 Q3: Optuna 가 identity 를 trial-0 로 안 시작(enqueue 없음, trial-0=mcmc_robust).
fix: ① empty-guard(빈 set → identity 단락) ② identity-anchor(enqueue_trial trial-0=identity).
"""
import numpy as np
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── G-329b: floor 모델 individual trial 이 crash 안 하고 identity 로 단락 ──────
def test_floor_model_empty_indstable_no_crash():
    import simulation.pipeline.preproc_optuna_hierarchical as P
    y = np.linspace(1, 67, 250)

    def obj(trial):
        P.suggest_y_preproc(trial, y, restrict_centered_y=True)
        return 0.0

    s = optuna.create_study()
    s.optimize(obj, n_trials=25, catch=(Exception,))
    fails = sum(1 for t in s.trials if t.state == optuna.trial.TrialState.FAIL)
    assert fails == 0, f"floor 모델 individual trial crash {fails}건 (empty-guard 실패)"


def test_floor_model_individual_uses_noncentered_g330():
    """G-330: STABLE_Y 전체 복원 → floor 모델 _NONCENTERED = non-centered(log1p/sqrt/asinh), 빈 set 아님.
    empty-guard(G-329b)는 backstop 으로 잔존(MPH_STABLE_TRANSFORMS affine-only 실험 시에만 발동)."""
    import simulation.pipeline.preproc_optuna_hierarchical as P
    from simulation.pipeline.preproc_optuna_hierarchical import _NONCENTERED_STABLE_Y
    # G-330 opened the set to log1p/sqrt/asinh (centered laplace/mcmc excluded);
    # G-335 added fourth_root when preproc became a flat 7-transform grid. The
    # invariant is that the set is non-empty and holds no CENTERED transform —
    # pinning the exact membership turned a documented addition into a failure.
    assert set(_NONCENTERED_STABLE_Y) == {"log1p", "sqrt", "asinh", "fourth_root"}, \
        f"non-centered set drifted: {_NONCENTERED_STABLE_Y}"
    assert not ({"laplace", "mcmc"} & set(_NONCENTERED_STABLE_Y)), \
        f"a centered transform leaked into the non-centered set: {_NONCENTERED_STABLE_Y}"
    y = np.linspace(1, 67, 250)
    # floor 모델 individual → non-centered 변환 사용 (crash 없음, centered 제외)
    t = optuna.trial.FixedTrial({"y_mode": "individual", "y_individual": "sqrt"})
    _, inv, state = P.suggest_y_preproc(t, y, restrict_centered_y=True)
    assert state.get("y_mode") == "individual", "floor 모델 individual 작동(crash 없음)"
    assert state.get("y_individual") not in ("laplace", "mcmc_robust"), "centered 제외 확인"


# ── G-329c: identity-anchor (enqueue → trial-0 = identity) ──────────────────
def test_identity_anchor_enqueue_trial0():
    import simulation.pipeline.preproc_optuna_hierarchical as P
    y = np.linspace(1, 67, 250)

    def obj(trial):
        P.suggest_y_preproc(trial, y, restrict_centered_y=False)
        return float(np.random.default_rng(trial.number).random())

    s = optuna.create_study()
    s.enqueue_trial({"y_mode": "none"})          # G-329c anchor
    s.optimize(obj, n_trials=5)
    assert s.trials[0].params.get("y_mode") == "none", \
        f"trial-0 가 identity 아님: {s.trials[0].params}"
