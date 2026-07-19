"""G-321 (2026-06-19, 사용자): eval-time rolling-origin 1-step for META classic-ts = 공정 평가.

배경: feature 모델은 predict(X_test) 가 실 lag(lag1=관측 전주)로 사실상 1-step-ahead → 양수. 그러나
sequence 모델(ARIMA/SARIMA/SARIMAX/Theta/FluSight) 은 predict=forecast(len(X_test))=단일원점 68주
외삽 → mean-revert → 불공정 음수. 같은 hold-out 인데 평가 task 가 달라 비교 불가(사용자의 "동일 환경
공정 경쟁" 위반). 해결: sequence 모델도 각 test 주를 관측 과거로 1주 예측(rolling 1-step) = feature 와
동일 task. A/B(실데이터): ARIMA −0.89→+0.92, SARIMA −1.01→+0.86, SARIMAX −0.84→+0.91.

leak-free: i 예측에 y_observed[:i] 만 사용(미래 미관측). META(identity transform) 라 raw y_observed 정확.

macOS: run PER-FILE — KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 .venv/bin/python -m pytest <file>.
"""
import numpy as np

from simulation.models.base import (
    ROLLING_EVAL_MODELS, supports_rolling_eval, TimeSeriesForecaster, ModelMeta,
)


def _r2(p, t):
    p, t = np.asarray(p, dtype=float), np.asarray(t, dtype=float)
    return 1.0 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()


# ── 계약(G-344): rolling-eval = classic-ts 5 + epi self-feeder 4(G-327) + GLARMA(G-327b) + FusedEpi(G-336)
#    + foundation 3종(TiRex/TimesFM-2.5/DLinear, G-344) — 전부 identity transform 이라 raw y_observed rolling
#    정확. N-HiTS 는 individual transform 이라 제외(→TRANSFORM_ROLLING; raw 면 test R²−13.7 폭발). ──
def test_rolling_eval_models_is_classic_ts():
    assert ROLLING_EVAL_MODELS == frozenset({
        "ARIMA", "SARIMA", "SARIMAX", "Theta", "FluSight-Baseline",
        "PoissonAutoreg", "hhh4-equivalent", "EpiEstim", "Wallinga-Teunis",
        "GLARMA", "FusedEpi", "TiRex", "TimesFM-2.5", "DLinear"})


# ── 계약(G-344): foundation migrated → ROLLING_EVAL(identity, R9도 rolling → 유한 OOF). BASELINE_ROLLING
#    비었음(helper 보존). N-HiTS/N-BEATS/TiDE = transform-space → TRANSFORM_ROLLING(raw 면 폭발). disjoint. ──
def test_baseline_rolling_disjoint_from_r9():
    from simulation.models.base import (
        BASELINE_ROLLING_MODELS, ROLLING_EVAL_MODELS, TRANSFORM_ROLLING_MODELS,
        supports_baseline_rolling,
    )
    assert BASELINE_ROLLING_MODELS == frozenset()                       # G-344: 전 멤버 migrated
    assert TRANSFORM_ROLLING_MODELS == frozenset({"N-BEATS", "N-HiTS", "TiDE"})  # G-344: +N-HiTS
    # 안전 핵심: raw-rolling(ROLLING_EVAL)과 transform-rolling(TRANSFORM_ROLLING)이 겹치면 라우팅 모호.
    assert TRANSFORM_ROLLING_MODELS.isdisjoint(ROLLING_EVAL_MODELS)
    assert BASELINE_ROLLING_MODELS.isdisjoint(ROLLING_EVAL_MODELS)      # 빈 set = 자명 disjoint

    # foundation 은 이제 ROLLING_EVAL(baseline-only 아님) → supports_baseline_rolling=False
    class _M:
        class meta: name = "DLinear"
    assert supports_baseline_rolling(_M()) is False
    class _N:
        class meta: name = "RandomForest"
    assert supports_baseline_rolling(_N()) is False


def test_supports_rolling_eval_detects_classic_ts():
    from simulation.models.ts_models import (
        SARIMAForecaster, ARIMAForecaster, SARIMAXForecaster,
    )
    for fac in (SARIMAForecaster, ARIMAForecaster, SARIMAXForecaster):
        assert supports_rolling_eval(fac()) is True, fac.__name__


def test_supports_rolling_eval_rejects_non_classic_ts():
    # meta.name 이 set 밖이면 False — feature/foundation/pf 는 rolling-eval 제외(scope: classic-ts).
    class _Dummy:
        meta = ModelMeta(name="RandomForest", category="tree", level=0,
                         min_data=1, description="", dependencies=[])
    assert supports_rolling_eval(_Dummy()) is False
    assert supports_rolling_eval(object()) is False   # meta 없어도 안전(False)


# ── 메커니즘: rolling_1step (base default = refit-per-step) ───────────────
class _PersistTS(TimeSeriesForecaster):
    """테스트용 persistence TS: forecast = 마지막 관측값 반복. rolling_1step(base default)이 매 step
    관측 과거로 refit → 1-step = 직전 관측값(persistence). 단일원점 = 상수(train 마지막값)."""
    meta = ModelMeta(name="_PersistTest", category="ts", level=0,
                     min_data=1, description="", dependencies=[])

    def fit_series(self, series, **kwargs):
        self._last = float(np.asarray(series, dtype=float).ravel()[-1])
        return self

    def forecast(self, steps, **kwargs):
        return np.full(int(steps), self._last, dtype=float)


def test_rolling_1step_is_leak_free_persistence():
    """rolling_1step[i] = y_observed[i-1] (직전 관측) — 미래(y_observed[i:]) 미사용 = leak-free."""
    m = _PersistTS()
    y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m.fit(np.zeros((5, 1)), y_train)            # base.fit → _train_series 저장
    y_obs = np.array([10.0, 20.0, 30.0, 40.0])
    roll = m.rolling_1step(y_obs)
    # step0 = train 마지막(5), step i = y_obs[i-1] (persistence, refit-per-step)
    np.testing.assert_allclose(roll, [5.0, 10.0, 20.0, 30.0])


def test_rolling_beats_single_origin_on_trend():
    """추세/계절 series 에서 rolling(관측 추종) >> 단일원점(상수 외삽). = 불공정 음수 해소의 본질."""
    t = np.arange(120, dtype=float)
    y = 30.0 + 25.0 * np.sin(2 * np.pi * t / 52.0)      # 계절 변동(상수 외삽이 크게 빗나감)
    y = np.maximum(y, 0.0)
    y_train, y_test = y[:80], y[80:]
    m = _PersistTS()
    m.fit(np.zeros((80, 1)), y_train)
    single = m.predict(np.zeros((len(y_test), 1)))                 # 단일원점 = 상수
    rolling = m.predict(np.zeros((len(y_test), 1)), y_observed=y_test)  # rolling 1-step
    assert _r2(single, y_test) < _r2(rolling, y_test), "rolling 이 단일원점보다 우수해야"
    assert _r2(rolling, y_test) > 0.0, "rolling = 양수(persistence 가 계절 추종)"


def test_predict_without_y_observed_is_single_origin():
    """y_observed 없으면 legacy 단일원점(forecast(len)) — back-compat 보존."""
    m = _PersistTS()
    m.fit(np.zeros((5, 1)), np.array([1.0, 2.0, 3.0, 4.0, 7.0]))
    out = m.predict(np.zeros((3, 1)))   # y_observed 미전달
    np.testing.assert_allclose(out, [7.0, 7.0, 7.0])   # 상수(마지막값) 외삽
