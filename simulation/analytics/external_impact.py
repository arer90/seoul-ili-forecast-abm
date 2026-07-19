"""외부충격·팬데믹 onset 탐지 + KDCA식 주의경보 단계 + 외생 shock feature.

이 모듈은 사용자 핵심 질문 **"모델/ABM이 팬데믹·외부임팩트(감염병 주의경보 등)를
탐지·반영할 수 있는가?"** 에 대한 전용 layer 다. ILI rate 시계열 위에서 세 가지
독립 기능을 제공하며, **전부 causal (past-only) — 미래 관측을 절대 사용하지 않는다**:

  1. ``detect_regime_shifts``  — 레짐전환/팬데믹 onset changepoint 탐지
       (CUSUM 또는 rolling z-score). 합성 평균-점프를 적중하고, 각 시점에
       shift flag 와 severity 를 반환. 2009 신종플루·COVID 류 구조변화 탐지용.

  2. ``pandemic_alert_level`` — KDCA식 4단계 주의경보 (0=관심·1=주의·2=경계·
       3=심각). 과거 baseline 대비 anomaly z-score 를 단계로 매핑. 심각도가
       올라갈수록 단계가 단조 상승.

  3. ``exogenous_shock_features`` — 외부충격을 feature 로 환산. 한국 휴일/개학
       (calendar 기반)·mobility-drop (NPI proxy, 주어지면)·subtype 우위전환
       (변종 proxy, 주어지면). 모델 입력 feature 로 직접 사용 가능.

[설계 철학 — leak-free]
    감시 시스템에서 가장 흔한 오류는 "현재값을 baseline 에 포함" 시키는 누수다.
    EARS-C2 (Hutwagner 2003) 가 baseline 을 2주 shift 하는 이유와 동일하다. 이
    모듈의 모든 baseline·rolling 통계는 **시점 t 이전 (t 제외 가능) 관측만** 사용해
    on-line 운영에서 그대로 쓸 수 있다.

[학술 배경]
    - Page ES (1954) "Continuous inspection schemes" Biometrika 41:100-115.
      (CUSUM changepoint 탐지의 원전)
    - Hutwagner L, Thompson W, Seeman GM, Treadwell T (2003) J Urban Health
      80(2 Suppl 1):i89-i96. (EARS — shift baseline 으로 self-contamination 회피)
    - Kang SK, Son WS, Kim BI (2024) J Korean Med Sci 39(4):e40.
      (KDCA 공식 epidemic threshold = 비유행기 mean + 2SD)
    - Serfling RE (1963) Public Health Rep 78(6):494-506. (cyclic baseline 조상)

[기존 코드와의 관계]
    이 모듈은 ``simulation.models.ears_models`` (EARS-C1/C2/C3, 점예측용 wrapper)
    및 ``simulation.analytics.kdca_threshold`` (국가단위 단일 threshold) 와 **상보**
    관계다. 그 둘은 모델/threshold 산출이 목적이고, 이 모듈은 **시계열 위에서의
    onset 탐지 + 단계 매핑 + feature 환산**이 목적이다. 기존 코드는 import 만 하고
    수정하지 않는다 (현재는 self-contained 라 import 불필요).

[데이터]
    서울 25구 ILI 337주 (sentinel_influenza all-age aggregate). 데모는 실측 사용.

Test: simulation/tests/test_external_impact.py
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Optional, Sequence, Union

import numpy as np

__all__ = [
    "detect_regime_shifts",
    "pandemic_alert_level",
    "exogenous_shock_features",
]

# ---------------------------------------------------------------------------
# KDCA식 주의경보 단계 라벨 (관심/주의/경계/심각) — 국가 위기경보 4단계 명명 준용
# ---------------------------------------------------------------------------
ALERT_LABELS_KR: tuple[str, str, str, str] = ("관심", "주의", "경계", "심각")
ALERT_LABELS_EN: tuple[str, str, str, str] = ("attention", "caution", "alert", "serious")

# anomaly z-score → 단계 경계 (causal baseline 대비 표준화 점수). KDCA mean+2SD
# (Kang 2024) 의 2σ 를 "주의" 진입선으로 두고, 경계/심각을 단조 상향한 운영 grid.
_DEFAULT_ALERT_THRESHOLDS: tuple[float, float, float] = (1.0, 2.0, 3.0)

DateLike = Union[str, date, datetime, np.datetime64]


def _as_1d_float(y: Sequence[float], *, name: str) -> np.ndarray:
    """1-D float array 로 강제 + 기본 검증 (NaN 위치는 보존).

    Args:
        y: 입력 시퀀스 (list / ndarray / tuple 등).
        name: 오류 메시지용 인자 이름.

    Returns:
        ``(n,)`` float64 ndarray (copy). 음수/0 은 보존.

    Raises:
        ValueError: y 가 비었거나 1-D 로 변환 불가일 때 (0초 fail-fast, G-166).

    Performance: O(n) time, O(n) memory.
    Side effects: 없음 (순수 함수).
    """
    arr = np.asarray(y, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name}: 1-D 시계열이 필요합니다 (got ndim={arr.ndim}).")
    if arr.size == 0:
        raise ValueError(f"{name}: 빈 시계열 (n=0).")
    return arr.copy()


def _causal_rolling_stats(
    y: np.ndarray, *, window: int, shift: int, min_periods: int
) -> tuple[np.ndarray, np.ndarray]:
    """시점 t 이전(t 제외 + shift) 관측만으로 rolling mean/std 계산 (leak-free).

    각 시점 t 의 baseline = ``y[t-shift-window .. t-shift-1]`` (t 및 이후 미사용).
    EARS-C2 (Hutwagner 2003) 의 shift baseline 과 동일한 self-contamination 회피
    원리. 관측이 ``min_periods`` 미만이면 NaN (운영 초기 warm-up).

    Args:
        y: ``(n,)`` 관측 시계열. NaN 은 nan-aware 통계로 무시.
        window: baseline 윈도 길이 (>=1).
        shift: t 와 baseline 끝 사이 gap (>=0). shift=0 이면 t-1 까지 사용.
        min_periods: baseline 유효 관측 최소 개수.

    Returns:
        ``(mean, std)`` 각각 ``(n,)`` float64. warm-up/전부-NaN 구간은 NaN.

    Performance: O(n * window) time, O(n) memory. n<=10⁴ 에서 수 ms.
    Side effects: 없음.
    Caller responsibility: window>=1, shift>=0, min_periods>=1 (아래서 강제).
    """
    n = y.shape[0]
    mean = np.full(n, np.nan, dtype=np.float64)
    std = np.full(n, np.nan, dtype=np.float64)
    for t in range(n):
        end = t - shift  # exclusive: baseline 은 end-1 까지
        start = end - window
        if end <= 0:
            continue
        start = max(0, start)
        seg = y[start:end]
        valid = seg[~np.isnan(seg)]
        if valid.size < min_periods:
            continue
        mean[t] = float(np.mean(valid))
        # ddof=0 (population) — 작은 윈도서 ddof=1 폭주 방지. EARS 도 population σ.
        std[t] = float(np.std(valid))
    return mean, std


# ===========================================================================
# 1. 레짐전환 / 팬데믹 onset 탐지
# ===========================================================================
def detect_regime_shifts(
    y: Sequence[float],
    *,
    method: str = "cusum",
    threshold: float = 5.0,
    window: int = 8,
    shift: int = 1,
    min_periods: int = 4,
    drift: float = 0.5,
) -> dict:
    """ILI 시계열에서 레짐전환/팬데믹 onset (changepoint) 을 탐지한다 (leak-free).

    2009 신종플루·COVID-19 류 구조적 평균-수준 점프를 잡는다. 두 가지 method:

      - ``"cusum"`` (Page 1954): 누적합 관제도. 시점 t 의 standardized residual
        ``z_t = (y_t - μ_{causal}) / σ_{causal}`` 을 누적해 양/음 CUSUM 통계
        ``S⁺, S⁻`` 를 갱신, ``threshold`` 초과 시 changepoint 선언 후 누적 reset.
        지속적 평균 이동에 민감 (점프 후 여러 시점 alarm 유지하지 않고 onset 만 표시).

      - ``"zscore"``: rolling z-score 임계 초과를 changepoint 로. 단발성 spike 에
        민감 (CUSUM 보다 지속성 요구가 약함).

    **leak-free 보장**: 시점 t 의 baseline (μ, σ) 은 ``y[t-shift-window .. t-shift-1]``
    만 사용 — t 및 미래 관측 미사용. on-line 운영에서 그대로 재현된다.

    Args:
        y: ``(n,)`` ILI rate 시계열 (음수/0 허용, NaN 은 미탐지 시점으로 통과).
        method: ``"cusum"`` 또는 ``"zscore"``. 그 외 값은 ValueError.
        threshold: 탐지 임계. cusum=누적 통계 임계 (표준편차 단위 누적, 권장 4-6),
            zscore=|z| 임계 (권장 2-3). 클수록 보수적 (FP↓).
        window: causal baseline 윈도 (주). 기본 8 (약 2달).
        shift: baseline 과 현재 시점 gap (주). 기본 1 (직전 관측 미포함=onset 자체
            오염 회피). >=0.
        min_periods: baseline 최소 관측 수 (warm-up). 미만이면 그 시점 미탐지.
        drift: cusum slack ``k`` (표준편차 단위). 작은 변동을 흡수해 FP 억제.
            기본 0.5 (Page 표준). zscore method 에선 무시.

    Returns:
        dict:
          - ``"changepoints"``: ``list[int]`` — onset 으로 선언된 시점 인덱스 (오름차순).
          - ``"shift_flags"``: ``(n,)`` int8 — changepoint 시점 1, 그 외 0.
          - ``"severity"``: ``(n,)`` float64 — 각 시점 탐지 강도. cusum=alarm 시
            누적통계/threshold 비율(>=1), 비-alarm=0; zscore=|z| (전 시점). NaN
            baseline 구간은 0.
          - ``"zscore"``: ``(n,)`` float64 — causal standardized residual (진단용,
            NaN-warmup 은 0).
          - ``"method"``: 실제 사용 method 문자열.

    Raises:
        ValueError: y 가 비었거나 1-D 아님; method 미지원; window/shift/min_periods
            범위 위반 (0초 fail-fast).

    Performance: O(n * window) time, O(n) memory. 337주 ≈ <2 ms.
    Side effects: 없음 (순수 함수, 전역 상태 미변경).
    Caller responsibility: y 는 단일 region/aggregate 의 시간순 정렬 시계열.
        결측은 NaN 으로 (0 으로 채우면 가짜 onset 유발).
    """
    arr = _as_1d_float(y, name="detect_regime_shifts.y")
    if method not in ("cusum", "zscore"):
        raise ValueError(f"method 는 'cusum'|'zscore' 중 하나 (got {method!r}).")
    if window < 1 or shift < 0 or min_periods < 1:
        raise ValueError(
            f"window>=1, shift>=0, min_periods>=1 필요 "
            f"(got window={window}, shift={shift}, min_periods={min_periods})."
        )

    n = arr.shape[0]
    mean, std = _causal_rolling_stats(
        arr, window=window, shift=shift, min_periods=min_periods
    )
    # standardized residual (causal). std==0 또는 NaN 구간은 z=0 (탐지 보류).
    z = np.zeros(n, dtype=np.float64)
    valid = (~np.isnan(mean)) & (~np.isnan(std)) & (std > 1e-9) & (~np.isnan(arr))
    z[valid] = (arr[valid] - mean[valid]) / std[valid]

    shift_flags = np.zeros(n, dtype=np.int8)
    severity = np.zeros(n, dtype=np.float64)
    changepoints: list[int] = []

    if method == "zscore":
        severity = np.abs(z)
        hits = np.where(valid & (np.abs(z) >= threshold))[0]
        for t in hits:
            shift_flags[t] = 1
            changepoints.append(int(t))
        return {
            "changepoints": changepoints,
            "shift_flags": shift_flags,
            "severity": severity,
            "zscore": z,
            "method": method,
        }

    # --- CUSUM (Page 1954): 양/음 단측 누적, alarm 시 reset ---
    s_pos = 0.0
    s_neg = 0.0
    k = float(drift)
    for t in range(n):
        if not valid[t]:
            # baseline 무효 시 누적 동결 (가짜 누적 방지)
            continue
        s_pos = max(0.0, s_pos + z[t] - k)
        s_neg = max(0.0, s_neg - z[t] - k)
        stat = max(s_pos, s_neg)
        if stat >= threshold:
            shift_flags[t] = 1
            severity[t] = stat / threshold  # >=1
            changepoints.append(int(t))
            s_pos = 0.0
            s_neg = 0.0  # onset 표시 후 reset → 다음 레짐 탐지
    return {
        "changepoints": changepoints,
        "shift_flags": shift_flags,
        "severity": severity,
        "zscore": z,
        "method": method,
    }


# ===========================================================================
# 2. KDCA식 4단계 주의경보 (관심/주의/경계/심각)
# ===========================================================================
def pandemic_alert_level(
    y: Sequence[float],
    *,
    baseline_window: int = 52,
    shift: int = 1,
    min_periods: int = 8,
    thresholds: Sequence[float] = _DEFAULT_ALERT_THRESHOLDS,
    return_labels: bool = False,
) -> Union[np.ndarray, dict]:
    """과거 baseline 대비 anomaly z-score 를 KDCA식 4단계 경보로 매핑한다 (leak-free).

    국가 위기경보 4단계 (관심·주의·경계·심각) 명명을 ILI anomaly 에 적용. 각 시점
    t 의 z-score = ``(y_t - μ_{past}) / σ_{past}`` 를 ``thresholds`` 경계와 비교해
    단계 0/1/2/3 을 부여한다. **심각도(z)가 단조 증가하면 단계도 단조 비감소**.

      level 0 (관심):  z < thresholds[0]
      level 1 (주의):  thresholds[0] <= z < thresholds[1]   (≈ KDCA mean+2SD 진입)
      level 2 (경계):  thresholds[1] <= z < thresholds[2]
      level 3 (심각):  z >= thresholds[2]

    **leak-free 보장**: baseline (μ, σ) 은 시점 t 이전(shift 적용) 관측만 사용.
    warm-up (baseline < min_periods) 구간은 level 0 (관심, 미경보) 으로 안전 처리.

    Args:
        y: ``(n,)`` ILI rate 시계열. NaN 은 level 0 으로 통과.
        baseline_window: 과거 baseline 윈도 (주). 기본 52 (1년, 계절성 1주기 포함).
        shift: 현재 시점과 baseline 끝 gap (주, >=0). 기본 1.
        min_periods: baseline 최소 관측 수. 미만이면 level 0.
        thresholds: 길이-3 오름차순 z 경계 ``(t1, t2, t3)``. 기본 ``(1, 2, 3)``.
            t1 은 KDCA mean+2SD 의 2σ 보다 낮춘 조기 "주의" 선.
        return_labels: True 면 dict (level + 한/영 라벨 + z) 반환, False 면 level
            array 만.

    Returns:
        ``return_labels=False``: ``(n,)`` int8 — 단계 (0..3).
        ``return_labels=True``: dict ``{"level": (n,) int8, "zscore": (n,) float64,
        "label_kr": (n,) <U2, "label_en": (n,) object, "thresholds": tuple}``.

    Raises:
        ValueError: y 비었음/1-D 아님; thresholds 길이≠3 또는 비오름차순;
            baseline_window<1 / shift<0 / min_periods<1 (0초 fail-fast).

    Performance: O(n * baseline_window) time, O(n) memory. 337주 ≈ <3 ms.
    Side effects: 없음.
    Caller responsibility: y 는 시간순 정렬된 단일 series. 단계는 운영 경보용
        — 모델 학습 feature 로 쓸 땐 미래 누수 없음(causal)이라 그대로 안전.
    """
    arr = _as_1d_float(y, name="pandemic_alert_level.y")
    thr = np.asarray(thresholds, dtype=np.float64)
    if thr.shape != (3,):
        raise ValueError(f"thresholds 는 길이-3 (got shape {thr.shape}).")
    if not np.all(np.diff(thr) > 0):
        raise ValueError(f"thresholds 는 순증가해야 함 (got {thr.tolist()}).")
    if baseline_window < 1 or shift < 0 or min_periods < 1:
        raise ValueError(
            f"baseline_window>=1, shift>=0, min_periods>=1 필요 "
            f"(got {baseline_window}, {shift}, {min_periods})."
        )

    n = arr.shape[0]
    mean, std = _causal_rolling_stats(
        arr, window=baseline_window, shift=shift, min_periods=min_periods
    )
    z = np.zeros(n, dtype=np.float64)
    valid = (~np.isnan(mean)) & (~np.isnan(std)) & (std > 1e-9) & (~np.isnan(arr))
    z[valid] = (arr[valid] - mean[valid]) / std[valid]

    # np.searchsorted: z 가 thr 의 어느 구간인지 → 단계 0..3. 단조 비감소 보장.
    # side='right': z==t1 이면 level 1 (경계 포함 진입).
    level = np.searchsorted(thr, z, side="right").astype(np.int8)
    # warm-up/invalid 구간은 안전하게 관심(0). (z=0 이미 0단계지만 명시)
    level[~valid] = 0

    if not return_labels:
        return level

    label_kr = np.array(ALERT_LABELS_KR, dtype="<U2")[level]
    label_en = np.array(ALERT_LABELS_EN, dtype=object)[level]
    return {
        "level": level,
        "zscore": z,
        "label_kr": label_kr,
        "label_en": label_en,
        "thresholds": tuple(thr.tolist()),
    }


# ===========================================================================
# 3. 외생 shock feature (calendar + mobility + subtype)
# ===========================================================================
# 한국 양력 고정 공휴일 (월, 일). 음력 기반(설·추석)은 양력 고정이 아니므로
# 별도 인자(설/추석 회피)로 다루지 않고, 학기 캘린더 + 양력 공휴일로 NPI/접촉
# 변화의 1차 proxy 를 구성한다. (정직: 음력 변동 공휴일은 미포함 — 가짜 매핑 회피)
_KR_FIXED_HOLIDAYS: tuple[tuple[int, int], ...] = (
    (1, 1),    # 신정
    (3, 1),    # 삼일절
    (5, 5),    # 어린이날
    (6, 6),    # 현충일
    (8, 15),   # 광복절
    (10, 3),   # 개천절
    (10, 9),   # 한글날
    (12, 25),  # 성탄절
)

# 한국 학기 개학 근사 (초·중·고). 1학기≈3/2 전후, 2학기≈8/하순. 개학 주변엔
# 학령 접촉↑ → ILI 상승 압력 (causal calendar flag, 관측 불필요).
_SCHOOL_START_WINDOWS: tuple[tuple[int, int, int, int], ...] = (
    # (시작월, 시작일, 끝월, 끝일) — 개학 ±주 윈도
    (3, 2, 3, 9),     # 1학기 개학 (3월 초)
    (8, 16, 9, 1),    # 2학기 개학 (8월 하순)
)


def _to_pydate(d: DateLike) -> date:
    """다양한 날짜 입력을 ``datetime.date`` 로 정규화한다.

    Args:
        d: ISO 문자열("2020-03-02")·date·datetime·np.datetime64 중 하나.

    Returns:
        ``datetime.date``.

    Raises:
        ValueError: 파싱 불가 문자열/타입.

    Performance: O(1).
    Side effects: 없음.
    """
    if isinstance(d, np.datetime64):
        # ns → date
        return np.datetime64(d, "D").astype("datetime64[D]").astype(date)
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    if isinstance(d, str):
        try:
            return datetime.fromisoformat(d[:10]).date()
        except ValueError as e:
            raise ValueError(f"날짜 문자열 파싱 실패: {d!r} ({e}).") from e
    raise ValueError(f"지원하지 않는 날짜 타입: {type(d).__name__}.")


def _is_kr_holiday_week(d: date) -> bool:
    """해당 주(월~일)에 한국 양력 고정 공휴일이 포함되는지 (causal calendar).

    주간 ILI 데이터 정렬에 맞춰, ``d`` 가 속한 ISO 주(월요일~일요일) 안에 고정
    공휴일이 하루라도 있으면 True.

    Args:
        d: 해당 관측 주의 대표 날짜 (어느 요일이든 그 주 전체로 확장).

    Returns:
        공휴일 주이면 True.

    Performance: O(7 * |holidays|) = O(1).
    Side effects: 없음.
    """
    monday = date.fromordinal(d.toordinal() - d.weekday())
    for offset in range(7):
        day = date.fromordinal(monday.toordinal() + offset)
        if (day.month, day.day) in _KR_FIXED_HOLIDAYS:
            return True
    return False


def _is_school_start_week(d: date) -> bool:
    """해당 날짜가 개학 윈도(월/일 범위) 안에 드는지 (causal calendar).

    Args:
        d: 관측 주 대표 날짜.

    Returns:
        개학 윈도 내이면 True.

    Performance: O(|windows|) = O(1).
    Side effects: 없음.
    """
    md = (d.month, d.day)
    for sm, sd, em, ed in _SCHOOL_START_WINDOWS:
        if (sm, sd) <= md <= (em, ed):
            return True
    return False


def _causal_drop_flag(
    x: np.ndarray, *, window: int, shift: int, rel_drop: float, min_periods: int
) -> np.ndarray:
    """시점 t 값이 직전 causal baseline 대비 ``rel_drop`` 이상 급감했는지 flag.

    mobility-drop = NPI(거리두기/봉쇄) proxy. baseline = ``x[t-shift-window..t-shift-1]``
    의 평균 (leak-free). ``x_t <= (1 - rel_drop) * baseline`` 이면 1.

    Args:
        x: ``(n,)`` mobility 류 시계열 (클수록 이동 많음). NaN 통과(flag 0).
        window: baseline 윈도. shift: t 와 baseline gap (>=0).
        rel_drop: 상대 급감 임계 (0..1). 0.3 = 30% 이상 하락 시 NPI 신호.
        min_periods: baseline 최소 관측.

    Returns:
        ``(n,)`` int8 drop flag.

    Performance: O(n * window). Side effects: 없음.
    """
    n = x.shape[0]
    base_mean, _ = _causal_rolling_stats(
        x, window=window, shift=shift, min_periods=min_periods
    )
    flag = np.zeros(n, dtype=np.int8)
    ok = (~np.isnan(base_mean)) & (base_mean > 1e-9) & (~np.isnan(x))
    flag[ok & (x <= (1.0 - rel_drop) * base_mean)] = 1
    return flag


def _causal_share_jump_flag(
    share: np.ndarray, *, window: int, shift: int, abs_jump: float, min_periods: int
) -> np.ndarray:
    """우점 subtype share 가 직전 baseline 대비 급변(우위전환)했는지 flag (변종 proxy).

    subtype_share = 우점 변종(예: 신규 우점 H3N2 fraction) 의 비율. baseline 평균
    대비 절대 변화량이 ``abs_jump`` 이상이면 변종 우위전환 신호 (leak-free).

    Args:
        share: ``(n,)`` 우점 subtype 비율 (0..1 권장). NaN 통과(flag 0).
        window, shift, min_periods: causal baseline 파라미터.
        abs_jump: 절대 변화 임계 (예: 0.2 = 20%p 이동).

    Returns:
        ``(n,)`` int8 변종 우위전환 flag.

    Performance: O(n * window). Side effects: 없음.
    """
    n = share.shape[0]
    base_mean, _ = _causal_rolling_stats(
        share, window=window, shift=shift, min_periods=min_periods
    )
    flag = np.zeros(n, dtype=np.int8)
    ok = (~np.isnan(base_mean)) & (~np.isnan(share))
    flag[ok & (np.abs(share - base_mean) >= abs_jump)] = 1
    return flag


def exogenous_shock_features(
    dates: Sequence[DateLike],
    *,
    mobility: Optional[Sequence[float]] = None,
    subtype_share: Optional[Sequence[float]] = None,
    mobility_window: int = 8,
    mobility_shift: int = 1,
    mobility_rel_drop: float = 0.3,
    subtype_window: int = 8,
    subtype_shift: int = 1,
    subtype_abs_jump: float = 0.2,
    min_periods: int = 4,
) -> dict:
    """외부충격을 causal feature 로 환산한다 (calendar + mobility + subtype).

    세 종류의 외생 shock proxy 를 모델 입력 feature 로 만든다. **전부 past-only**:
    calendar 는 관측 무관(미래 누수 불가능), mobility/subtype 은 causal baseline
    (시점 t 이전) 만 사용.

      - ``holiday_flag``      : 한국 양력 고정 공휴일 주 (접촉 패턴 변화).
      - ``school_start_flag`` : 개학 윈도 (학령 접촉↑, ILI 상승 압력).
      - ``mobility_drop_flag``: mobility 가 직전 baseline 대비 급감 = NPI proxy
          (mobility 주어졌을 때만; 아니면 전부 0).
      - ``variant_shift_flag``: 우점 subtype share 급변 = 변종 우위전환 proxy
          (subtype_share 주어졌을 때만; 아니면 전부 0).

    Args:
        dates: ``(n,)`` 관측 주 날짜 (ISO str / date / datetime / datetime64).
        mobility: ``(n,)`` 이동량 proxy (없으면 None → drop flag 전부 0).
        subtype_share: ``(n,)`` 우점 변종 비율 (없으면 None → variant flag 전부 0).
        mobility_window/mobility_shift/mobility_rel_drop: NPI drop 탐지 파라미터.
            rel_drop=0.3 → baseline 대비 30%+ 하락 시 1.
        subtype_window/subtype_shift/subtype_abs_jump: 변종 우위전환 파라미터.
            abs_jump=0.2 → baseline 대비 20%p+ 이동 시 1.
        min_periods: mobility/subtype baseline 최소 관측 (warm-up).

    Returns:
        dict[str, np.ndarray] — 키별 ``(n,)`` 배열:
          - ``"holiday_flag"``: int8
          - ``"school_start_flag"``: int8
          - ``"mobility_drop_flag"``: int8 (mobility None 이면 전부 0)
          - ``"variant_shift_flag"``: int8 (subtype_share None 이면 전부 0)
        모든 배열 길이 = len(dates) (shape 일치 보장).

    Raises:
        ValueError: dates 비었음; mobility/subtype_share 가 dates 와 길이 불일치;
            날짜 파싱 실패 (0초 fail-fast).

    Performance: O(n * max(window)) time, O(n) memory. 337주 ≈ <5 ms.
    Side effects: 없음 (순수 함수, DB/disk 미접근).
    Caller responsibility: dates 는 ILI series 와 동일 정렬·동일 길이. calendar
        flag 는 양력 고정 공휴일/개학 근사만 — 음력 변동 공휴일(설·추석)은 미포함
        (가짜 매핑 회피). mobility 는 "클수록 이동 많음" 방향.
    """
    if dates is None or len(dates) == 0:
        raise ValueError("exogenous_shock_features.dates: 빈 입력.")
    py_dates = [_to_pydate(d) for d in dates]
    n = len(py_dates)

    holiday = np.fromiter(
        (1 if _is_kr_holiday_week(d) else 0 for d in py_dates),
        dtype=np.int8, count=n,
    )
    school = np.fromiter(
        (1 if _is_school_start_week(d) else 0 for d in py_dates),
        dtype=np.int8, count=n,
    )

    mob_flag = np.zeros(n, dtype=np.int8)
    if mobility is not None:
        mob = _as_1d_float(mobility, name="exogenous_shock_features.mobility")
        if mob.shape[0] != n:
            raise ValueError(
                f"mobility 길이({mob.shape[0]}) != dates 길이({n})."
            )
        mob_flag = _causal_drop_flag(
            mob, window=mobility_window, shift=mobility_shift,
            rel_drop=mobility_rel_drop, min_periods=min_periods,
        )

    var_flag = np.zeros(n, dtype=np.int8)
    if subtype_share is not None:
        sub = _as_1d_float(subtype_share, name="exogenous_shock_features.subtype_share")
        if sub.shape[0] != n:
            raise ValueError(
                f"subtype_share 길이({sub.shape[0]}) != dates 길이({n})."
            )
        var_flag = _causal_share_jump_flag(
            sub, window=subtype_window, shift=subtype_shift,
            abs_jump=subtype_abs_jump, min_periods=min_periods,
        )

    return {
        "holiday_flag": holiday,
        "school_start_flag": school,
        "mobility_drop_flag": mob_flag,
        "variant_shift_flag": var_flag,
    }
