"""실데이터 prep + empirical WIS — 통합 테스트 fixture.

(2026-06-01) one-off 측정 스크립트(_user_design_eval 등)에서 이관 — 측정 스크립트는 삭제,
테스트가 쓰는 두 helper 만 깨끗한 위치로 보존. 측정 결정 기록:
`docs/FEATURE_SELECTION_STABILITY_DECISION_20260601.md`.

사용처: test_phase13_preproc_first_corr1se · test_feature_optuna_size.
NOTE: 파일명이 test_* 아님 → pytest 수집 대상 아님 (helper 전용).
"""
import numpy as np


def _ewis(y, pred, resid):
    """empirical WIS (FLUSIGHT alphas, split-conformal residuals). resid<2 → std fallback."""
    from simulation.analytics.diagnostics import weighted_interval_score_empirical
    from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS
    r = np.asarray(resid, float).ravel(); r = r[np.isfinite(r)]
    if r.size < 2:
        r = np.array([np.std(y) or 1.0, 0.0])
    return np.asarray(weighted_interval_score_empirical(
        np.asarray(y, float), np.asarray(pred, float), residuals=r, alphas=FLUSIGHT_ALPHAS), float)


def _prep_full():
    """phase 1 실데이터(서울 ILI) → (Pp, Pt, yp, yt, ylog, inv, cols).

    QuantileTransformer(normal) on X + 시간 train/test split. ylog=log1p(y), inv=expm1(clip).
    Returns: Pp/Pt (transform된 train/test X), yp/yt (raw y), ylog (log1p yp), inv (역변환), cols.
    """
    from simulation.pipeline.config import PipelineConfig
    from simulation.pipeline.data import run_data
    from sklearn.preprocessing import QuantileTransformer
    r = run_data(PipelineConfig())
    X_all, y_all = r["X_all"], r["y_all"]
    cols = r.get("feature_cols") or [f"f{i}" for i in range(X_all.shape[1])]
    n_train = int(r.get("n_train", int(len(y_all) * 0.8)))
    Xp, yp, Xt, yt = X_all[:n_train], y_all[:n_train], X_all[n_train:], y_all[n_train:]
    qt = QuantileTransformer(output_distribution="normal",
                             n_quantiles=min(100, len(Xp))).fit(np.nan_to_num(Xp, nan=0.0))
    Pp, Pt = qt.transform(np.nan_to_num(Xp, nan=0.0)), qt.transform(np.nan_to_num(Xt, nan=0.0))
    ylog = np.log1p(np.clip(yp, 0, None))
    inv = lambda z: np.expm1(np.clip(z, -2, 20))
    return Pp, Pt, np.asarray(yp, float), np.asarray(yt, float), ylog, inv, list(cols)
