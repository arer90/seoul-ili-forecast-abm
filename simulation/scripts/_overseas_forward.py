"""_overseas_forward.py — calendar-locked forward-eval helpers (national + regional 공유).

두 챔피언 일반화 스크립트(``fig_overseas_national_champion`` · ``fig_overseas_regions_champion``)가
**동일 캘린더 경계**의 forward(전향) 평가를 쓰도록 공유하는 deep module.

설계 (ENGINEERING_PRINCIPLES.md D-4 deep module / K-2 simplicity / #5 reproducibility):
- **calendar-locked split (단일 규칙)**: 한 주의 대표 시작일이
  ``in_sample_end`` (기본 2026-02-09) **이하면 in-sample, 초과면 forward**.
  - overseas: ``(year, week_no)`` → ISO Monday 날짜 (``isoweek_monday``).
  - Seoul: ``week_start`` (일요일 앵커) 를 그대로 사용 — 둘 다 "02-09 초과 = forward".
  이 규칙으로 전 시계열의 forward 가 **2026-02-16 이후(overseas W08 / Seoul 02-15 주)** 로 정렬된다.
- **forward = rolling 1-step**: in-sample 로 fit → forward 구간을 관측 y 흘리며(``y_observed``)
  1주씩 1-step 예측. 단일원점 외삽이 아님(leak-free: i 예측에 y_observed[:i] 만).
- **공통 forward 창 (가짜 연장 금지)**: 시계열별 forward 길이가 데이터 끝 차이로 다르면
  ``common_forward_len`` 로 공통 최소 길이를 구해 각 시계열을 **앞에서부터 N주만 truncate**.
  figure/CSV 에 공통 창(주수)·시계열별 가용 길이를 명시 → 데이터 없는 주를 채우지 않는다.
- **Seoul 라이브 기준선 (하드코드 제거)**: 옛 ``SEOUL_CHAMPION`` r2=0.9357(test-slab, forward 아님)
  대신 ``compute_seoul_forward_baseline`` 가 **동일 프로토콜**(feature_cache ili_rate + week_start,
  in-sample≤02-09, FusedEpi rolling 1-step)로 Seoul forward R²/WIS 를 **실시간 계산**한다.

Performance: 시계열별 FusedEpi fit(TiRex rolling 캐시) + forward n×(1-step). CPU. 시계열당 ~수 분.
Side effects: 없음 (DB·파일 write 0; Seoul 기준선은 feature_cache.parquet read-only).
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ── 캘린더 경계 (SSOT) ─────────────────────────────────────────────────────
#   in-sample 종료일 = 2026-02-09 (HWP §3 in_sample_end; config_global DataSplitConfig 와 정합).
#   한 주의 대표 시작일이 이 날짜 **이하 = in-sample, 초과 = forward**.
#   env MPH_FORWARD_IN_SAMPLE_END 로 재정의 가능(detached 실험용; 기본 2026-02-09).
DEFAULT_IN_SAMPLE_END = "2026-02-09"


def get_in_sample_end() -> _dt.date:
    """forward split 경계일 반환 (env override 가능).

    Returns:
        in_sample_end (datetime.date) — 이 날짜 **이하** = in-sample, **초과** = forward.
    """
    s = os.environ.get("MPH_FORWARD_IN_SAMPLE_END", "").strip() or DEFAULT_IN_SAMPLE_END
    return _dt.date.fromisoformat(s)


def isoweek_monday(year: int, week_no: int) -> _dt.date | None:
    """(ISO year, ISO week) → 그 주 월요일 날짜. 부적합 week 면 None.

    Args:
        year: ISO 연도.
        week_no: ISO 주차 (1–53).

    Returns:
        해당 주 월요일(datetime.date) 또는 ISO 변환 실패 시 None.
    """
    try:
        return _dt.date.fromisocalendar(int(year), int(week_no), 1)
    except (ValueError, TypeError):
        return None


def split_forward_by_isoweek(
    yw: list[tuple[int, int]], in_sample_end: _dt.date
) -> int:
    """(year, week_no) 시퀀스(시간순)에서 forward 시작 인덱스 = n_train 계산.

    Args:
        yw: [(year, week_no), …] 시간 오름차순. (overseas 행 순서.)
        in_sample_end: 경계일 — 주 월요일 ≤ 이 날짜면 in-sample, 초과면 forward.

    Returns:
        n_train (int) — 처음 n_train 주가 in-sample(월요일 ≤ in_sample_end), 나머지가 forward.
        ISO 변환 불가 주는 보수적으로 in-sample 취급(경계 이전 데이터로 가정).

    Caller responsibility: yw 는 시간 오름차순이어야 한다(forward 가 연속 꼬리라고 가정).
    """
    n_train = 0
    for y, w in yw:
        d = isoweek_monday(y, w)
        if d is None or d <= in_sample_end:
            n_train += 1
        else:
            break
    return n_train


def split_forward_by_dates(
    dates: list, in_sample_end: _dt.date
) -> int:
    """날짜 배열(시간순)에서 forward 시작 인덱스 = n_train 계산 (Seoul week_start 용).

    Args:
        dates: datetime.date | datetime.datetime 시퀀스, 시간 오름차순.
        in_sample_end: 경계일 — 주 시작일 ≤ 이 날짜면 in-sample, 초과면 forward.

    Returns:
        n_train (int) — 처음 n_train 주가 in-sample.
    """
    n_train = 0
    for d in dates:
        dd = d.date() if isinstance(d, _dt.datetime) else d
        if dd <= in_sample_end:
            n_train += 1
        else:
            break
    return n_train


def common_forward_len(forward_lens: list[int], cap: int | None = None) -> int:
    """시계열별 forward 길이 목록에서 공통 평가 길이 결정 (가짜 연장 금지).

    공통 창 = min(forward_lens) — 가장 짧은 시계열 길이에 truncate. cap 이 주어지면
    min(공통, cap). 0 이하 길이는 제외(데이터 없는 시계열).

    Args:
        forward_lens: 각 시계열의 사용 가능 forward 주수(>0).
        cap: env/CONFIG forward 상한(예 18). None 이면 무제한.

    Returns:
        공통 forward 길이(int, ≥1). forward_lens 비면 0.
    """
    vals = [n for n in forward_lens if n > 0]
    if not vals:
        return 0
    c = min(vals)
    if cap is not None and cap > 0:
        c = min(c, cap)
    return int(c)


def build_basic_features(y: np.ndarray, period: int = 52) -> np.ndarray:
    """BASIC feature 행렬 (lag + 계절성) — leakage-free(과거 shift). national/regional 공유.

    프로젝트 ``BASIC_FEATURE_COLS`` 개념 재구성:
    lag1/2/4/52 + sin/cos month + Fourier(h1~h3) + season_idx. 각 시점 t feature 는
    y[<=t-1] 만 사용(인과). lag 부재구간은 첫 관측으로 패딩.

    Args:
        y: (T,) 관측 시계열.
        period: 계절 주기(주). ILI 주간 = 52.

    Returns:
        (T, 13) float ndarray — BASIC_FEATURE_COLS 순서/의미.

    Performance: O(T). Side effects: 없음.
    """
    T = len(y)
    t = np.arange(T)

    def _lag(k: int) -> np.ndarray:
        out = np.empty(T, dtype=float)
        out[:k] = y[0] if T else 0.0
        out[k:] = y[:-k]
        return out

    lag1, lag2, lag4, lag52 = _lag(1), _lag(2), _lag(4), _lag(min(period, max(1, T - 1)))

    woy = (t % period) + 1
    month = np.clip(np.ceil(woy / 4.345), 1, 12)
    sin_month = np.sin(2 * np.pi * month / 12.0)
    cos_month = np.cos(2 * np.pi * month / 12.0)

    ang = 2 * np.pi * (t % period) / float(period)
    fs1, fc1 = np.sin(ang), np.cos(ang)
    fs2, fc2 = np.sin(2 * ang), np.cos(2 * ang)
    fs3, fc3 = np.sin(3 * ang), np.cos(3 * ang)

    season_idx = (t // period).astype(float)

    return np.column_stack([
        lag1, lag2, lag4, lag52,
        sin_month, cos_month,
        fs1, fc1, fs2, fc2, fs3, fc3,
        season_idx,
    ])


def wis(y_true: np.ndarray, q_dict: dict) -> float:
    """Weighted Interval Score (proper, lower=better) — predict_quantiles 출력 소비.

    Args:
        y_true: (n,) 관측.
        q_dict: {level: (n,) 분위예측} — 중위(0.5)와 대칭쌍 포함.

    Returns:
        평균 WIS (float).
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    med = np.asarray(q_dict.get(0.5, q_dict.get(0.50)), dtype=float).ravel()
    total = 0.5 * np.abs(y_true - med)
    pairs = [(0.025, 0.975), (0.25, 0.75)]
    K = len(pairs)
    for lo, hi in pairs:
        if lo not in q_dict or hi not in q_dict:
            continue
        l = np.asarray(q_dict[lo], dtype=float).ravel()
        u = np.asarray(q_dict[hi], dtype=float).ravel()
        alpha = (1.0 - (hi - lo))
        is_score = (u - l) \
            + (2.0 / alpha) * (l - y_true) * (y_true < l) \
            + (2.0 / alpha) * (y_true - u) * (y_true > u)
        total = total + (alpha / 2.0) * is_score
    return float(np.mean(total / (K + 0.5)))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """결정계수 R² (1 - SS_res/SS_tot); SS_tot=0이면 nan.

    Args:
        y_true: (n,) 관측.
        y_pred: (n,) 예측.

    Returns:
        R² (float) 또는 분산 0이면 nan.
    """
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fused_epi_forward(
    y: np.ndarray, n_train: int, forward_len: int,
) -> dict | None:
    """in-sample(처음 n_train) fit → forward(다음 forward_len) FusedEpi rolling 1-step.

    Args:
        y: (T,) 관측 시계열(시간순). y[:n_train]=in-sample, y[n_train:n_train+forward_len]=forward.
        n_train: in-sample 길이(캘린더 경계 split 산출).
        forward_len: 평가할 forward 주수(공통 창으로 truncate 된 값).

    Returns:
        {n_train, n_forward, champ_r2, champ_wis, persist_r2, seasonal_r2,
         persist_wis, seasonal_wis, y_te(list), y_pred(list)} 또는 부적합/실패 시 None.

    Performance: TiRex rolling fit(캐시) + forward_len×(1-step). CPU. ~수 분.
    Side effects: 없음.
    """
    from simulation.models.fused_epi import FusedEpiForecaster

    T = len(y)
    if forward_len <= 0 or n_train < 70 or n_train + forward_len > T:
        return None

    X = build_basic_features(y[: n_train + forward_len])
    X_tr, y_tr = X[:n_train], y[:n_train]
    sl = slice(n_train, n_train + forward_len)
    X_te = X[n_train: n_train + forward_len]
    y_te = y[sl]
    y_obs = y[sl]                                   # rolling: 관측 흘리기 (leak-free 1-step)

    try:
        model = FusedEpiForecaster(repo_id="NX-AI/TiRex")
        model.fit(X_tr, y_tr)
        y_pred = np.asarray(model.predict(X_te, y_observed=y_obs), dtype=float)
        q = model.predict_quantiles(X_te, y_observed=y_obs,
                                    levels=(0.025, 0.25, 0.5, 0.75, 0.975))
    except Exception as e:
        log.warning("  [skip] FusedEpi forward 실패 — %s", e)
        return None

    champ_r2 = r2(y_te, y_pred)
    champ_wis = wis(y_te, q)

    # baseline (관측 기반, 모델-유래 아님) — forward 창 안에서 1-step.
    persist = y[n_train - 1: n_train + forward_len - 1]            # lag1
    seasonal = np.array([y[n_train + i - 52] if n_train + i - 52 >= 0 else y[n_train - 1]
                         for i in range(forward_len)])             # 작년 같은 주
    persist_r2 = r2(y_te, persist)
    seasonal_r2 = r2(y_te, seasonal)
    persist_wis = float(np.mean(0.5 * np.abs(y_te - persist) / 1.5))
    seasonal_wis = float(np.mean(0.5 * np.abs(y_te - seasonal) / 1.5))

    return {
        "n_train": n_train, "n_forward": forward_len,
        "champ_r2": champ_r2, "champ_wis": champ_wis,
        "persist_r2": persist_r2, "seasonal_r2": seasonal_r2,
        "persist_wis": persist_wis, "seasonal_wis": seasonal_wis,
        "y_te": y_te.tolist(), "y_pred": y_pred.tolist(),
    }


def _seoul_cache_path() -> Path:
    """Seoul feature_cache.parquet 경로 (ili_rate + week_start SSOT).

    Returns:
        feature_cache.parquet 절대경로 (simulation/cache/).
    """
    return Path(__file__).resolve().parents[1] / "cache" / "feature_cache.parquet"


def compute_seoul_forward_baseline(forward_cap: int | None = None) -> dict | None:
    """Seoul ILI 라이브 forward 기준선 — 옛 하드코드 0.9357(test-slab) 대체.

    동일 프로토콜: feature_cache(ili_rate + week_start) 로드 → week_start ≤ in_sample_end
    = in-sample, 초과 = forward → FusedEpi rolling 1-step → forward R²/WIS 실시간 계산.

    Args:
        forward_cap: 공통 forward 창(주수)로 Seoul forward 를 truncate (figure 정합용).
            None 이면 Seoul 의 전체 forward 길이 사용.

    Returns:
        {"model": "FusedEpi (Seoul, live forward)", "r2", "wis", "n_test",
         "n_train", "in_sample_end", "source"} 또는 캐시 부재/실패 시 None.
        n_test = 실제 평가에 쓰인 forward 주수(공통 창 truncate 반영).

    Performance: Seoul in-sample(~336) fit + forward(~14–17) rolling. CPU ~수 분.
    Side effects: feature_cache.parquet read-only. DB write 0.
    Caller responsibility: forward_cap 은 cross-series 공통 창과 정합해야 figure 가 정직.
    """
    import polars as pl

    path = _seoul_cache_path()
    if not path.exists():
        log.warning("[Seoul 기준선] feature_cache 부재(%s) — 라이브 기준선 계산 불가", path)
        return None
    df = pl.read_parquet(path)
    if "ili_rate" not in df.columns or "week_start" not in df.columns:
        log.warning("[Seoul 기준선] ili_rate/week_start 컬럼 부재 — 라이브 기준선 계산 불가")
        return None

    y = df["ili_rate"].to_numpy().astype(float)
    dates = df["week_start"].to_list()
    in_sample_end = get_in_sample_end()
    n_train = split_forward_by_dates(dates, in_sample_end)
    full_forward = len(y) - n_train
    forward_len = common_forward_len([full_forward], cap=forward_cap)
    if forward_len <= 0 or n_train < 70:
        log.warning("[Seoul 기준선] split 부적합 (n_train=%d forward=%d)", n_train, full_forward)
        return None

    log.info("[Seoul 기준선] in-sample=%d(≤%s) forward=%d(공통창 truncate, 가용 %d) — FusedEpi rolling …",
             n_train, in_sample_end.isoformat(), forward_len, full_forward)
    r = fused_epi_forward(y, n_train, forward_len)
    if r is None:
        log.warning("[Seoul 기준선] FusedEpi forward 실패 — 라이브 기준선 None")
        return None

    log.info("[Seoul 기준선] 라이브 forward R²=%.4f WIS=%.4f (n_forward=%d, in_sample_end=%s)",
             r["champ_r2"], r["champ_wis"], r["n_forward"], in_sample_end.isoformat())
    return {
        "model": "FusedEpi (Seoul, live forward)",
        "r2": float(r["champ_r2"]),
        "wis": float(r["champ_wis"]),
        "n_test": int(r["n_forward"]),
        "n_train": int(r["n_train"]),
        "in_sample_end": in_sample_end.isoformat(),
        "source": "model-derived (Seoul live forward, FusedEpi rolling 1-step, calendar-locked)",
    }
