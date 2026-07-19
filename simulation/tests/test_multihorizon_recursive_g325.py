"""G-325: recursive multi-horizon rolling-origin (leakage-free) — 사용자 #3 + 3-AI 검토.

이전 _rolling_origin_multihorizon 은 X_query=X_test[i+h-1] 로 **실제 미래 lag 를 feature 로 사용**
→ h≥2 누수(낙관적 decay table). 수정: origin 에서 미관측 주의 lag 를 모델 자신의 예측으로 recursive
채움. 이 테스트가 (1) leakage-free (2) horizon-decay 를 검증.

macOS: run PER-FILE — KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python -m pytest <file>.
"""
import numpy as np

from simulation.pipeline.real_eval import _rolling_origin_multihorizon


class _Persistence:
    """predict = lag1 컬럼(col 0) — 순수 persistence. recursive 시 미관측 주는 예측이 lag 로 들어감."""

    def fit(self, X, y, **kw):
        return self

    def predict(self, X, **kw):
        return np.asarray(X, dtype=float)[:, 0]   # lag1 column


def _factory():
    return _Persistence()


def _setup():
    # 단조 증가 series → persistence 의 recursive(평탄) 가 실제와 점점 어긋남(decay 가시).
    y = np.arange(10.0, 50.0)                       # 10,11,…,49 (len 40)
    n_in, n_test = 24, 8
    feature_cols = ["ili_rate_lag1", "season_idx"]  # lag1=col0, calendar=col1
    X_full = np.array([[y[i - 1] if i > 0 else y[0], float(i)] for i in range(len(y))], dtype=float)
    return (X_full[:n_in], y[:n_in], X_full[n_in:n_in + n_test], y[n_in:n_in + n_test], feature_cols)


def test_recursive_is_leakage_free():
    """h=2 예측은 모델 자신의 h=1 예측(recursive)이지 실제 미래 lag 가 아님."""
    X_in, y_in, X_test, y_test, fcols = _setup()
    out = _rolling_origin_multihorizon(_factory, "pers", X_in, y_in, X_test, y_test,
                                       horizons=(1, 2, 3, 4), feature_cols=fcols)
    # recursive: out[2][t] == out[1][t] (미관측 주 lag = h=1 예측 = 평탄)
    valid = np.isfinite(out[2][:6])
    np.testing.assert_allclose(out[2][:6][valid], out[1][:6][valid],
                               err_msg="h=2 는 recursive(h=1 예측 재사용)여야")
    # leakage 라면 out[2][t] == y_test[t](실제 미래). 단조증가라 out[2][t]=y_test[t-1] ≠ y_test[t].
    for t in range(5):
        assert abs(out[2][t] - y_test[t]) > 0.5, "h=2 가 실제 미래값과 같으면 누수"


def test_horizon_decay_increases():
    """예측 오차가 horizon 과 함께 증가(decay table 의 본질)."""
    X_in, y_in, X_test, y_test, fcols = _setup()
    out = _rolling_origin_multihorizon(_factory, "pers", X_in, y_in, X_test, y_test,
                                       horizons=(1, 2, 3, 4), feature_cols=fcols)

    def _mae_h(h):
        # out[h][t] 는 target 주 t+h-1 예측 → y_test[t+h-1] 과 정렬
        errs = []
        for t in range(len(y_test)):
            tgt = t + h - 1
            if tgt < len(y_test) and np.isfinite(out[h][t]):
                errs.append(abs(out[h][t] - y_test[tgt]))
        return float(np.mean(errs)) if errs else np.nan

    maes = [_mae_h(h) for h in (1, 2, 3, 4)]
    assert maes[0] < maes[1] < maes[2] < maes[3], f"horizon-decay 단조증가 기대, got {maes}"


def test_no_feature_cols_no_crash():
    """feature_cols=None(sequence univariate) 도 안전(lag override 없음, fixed model)."""
    X_in, y_in, X_test, y_test, _ = _setup()
    out = _rolling_origin_multihorizon(_factory, "pers", X_in, y_in, X_test, y_test,
                                       horizons=(1, 2), feature_cols=None)
    assert set(out.keys()) == {1, 2}
    assert len(out[1]) == len(y_test)


def test_full_metric_decay_not_just_mae():
    """사용자 정정: decay 는 MAE 하나가 아니라 horizon 별 **129-metric 전부 + 예측값/test값**(G-168 SSOT).
    R² 는 horizon 과 함께 감소(1-step 이 multi-week 일반화를 과대평가 = #3 가시화)."""
    from simulation.pipeline.phase_evaluator import evaluate_predictions_full
    rng = np.random.RandomState(0)
    t = np.arange(120.0)
    y = np.maximum(30 + 20 * np.sin(2 * np.pi * t / 52) + rng.normal(0, 2, 120), 0.0)
    n_in, n_test = 80, 30
    fcols = ["ili_rate_lag1", "season_idx"]
    X_full = np.array([[y[i - 1] if i > 0 else y[0], float(i % 52)] for i in range(len(y))], dtype=float)
    X_in, y_in, X_test, y_test = X_full[:n_in], y[:n_in], X_full[n_in:n_in + n_test], y[n_in:n_in + n_test]

    from sklearn.linear_model import Ridge

    class _S:
        def __init__(self):
            self.m = Ridge()

        def fit(self, X, Y, **k):
            self.m.fit(X, Y)
            return self

        def predict(self, X, model_name=None, **k):
            return self.m.predict(X)

    mh = _rolling_origin_multihorizon(lambda: _S(), "ridge", X_in, y_in, X_test, y_test,
                                      horizons=(1, 2, 3, 4), feature_cols=fcols)
    r2s = []
    for h in (1, 2, 3, 4):
        p = mh[h]
        pr, tr = [], []
        for ti in range(len(y_test)):
            if ti + h - 1 < len(y_test) and np.isfinite(p[ti]):
                pr.append(p[ti])
                tr.append(y_test[ti + h - 1])
        pr, tr = np.asarray(pr), np.asarray(tr)
        sig = float(np.std(tr - pr)) or 1.0
        m = evaluate_predictions_full(tr, pr, sigma=sig, y_train_pool=y_in, phase_id=f"h{h}")
        assert len(m) >= 100, f"129-metric SSOT 기대(MAE 하나 아님), got {len(m)}"
        assert "r2" in m and "wis" in m and "rmse" in m and "mape" in m
        r2s.append(float(m["r2"]))
    assert r2s[0] > r2s[-1], f"R² horizon-decay(1-step>4-step) 기대, got {r2s}"
