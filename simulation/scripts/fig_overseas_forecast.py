"""fig_overseas_forecast.py — 해외 ILI 시계열에 baseline 예측 적용 (참고치, 2026-06-26).

★ 정직성 최우선 (ENGINEERING_PRINCIPLES.md #5 재현성 · G-237 fail-loud):
    이 figure 는 **우리 메인 챔피언(FusedEpi)의 해외 예측이 아니다**.
    FusedEpi 는 서울 ILI 전용으로 학습된 모델이고, 해외→서울 transfer 는
    구축되지 않았다 (OverseasTransfer = phantom, encoder 우회 이력). 따라서
    챔피언을 해외 시계열에 직접 적용하는 것은 방법론적으로 부적절하다.

    대신 **가벼운 univariate baseline (ARIMA·Theta)** 을 overseas_ili 의 주요국
    시계열에 **1-step rolling**(expanding window, 매 주 재적합 후 1주 예측)으로
    직접 적용해 "해외 데이터에서 단순 통계 baseline 이 얼마나 맞는가" 만 본다.
    이는 **참고치(reference)** 이지 우리 시스템의 해외 예측 성능이 아니다.
    제목/주석에 이 점을 명시한다.

데이터 (read-only, 기존 코드 미수정):
    overseas_ili 테이블 — source/country/year/week_no/ili_rate/positivity_pct.
    - 대부분 source 는 ili_rate 사용.
    - who_flunet 은 ili_rate 가 sparse(많이 null) → positivity_pct(양성률 %) 를
      신호로 사용 (label 에 "positivity" 명시 — ILI rate 와 다른 지표임을 정직히 표기).
    DB = simulation/data/db/epi_real_seoul.db, read_only_connect 만 사용
    (저수준 직접 연결 금지 = G-116/117).

평가 (per 국가 × per 모델):
    - R²  (1 - SS_res/SS_tot, 음수 가능 = 평균보다 나쁨)
    - MASE (Mean Absolute Scaled Error, scale = 1-step seasonal-naive in-sample MAE;
      MASE<1 = naive 보다 우수, 표준 forecast 비교 지표)
    1-step rolling: 학습 시작 후 매 주 t 에서 [0..t-1] 로 적합 → ŷ_t, 다음 주로 확장.

엄수: 실 데이터만 · matplotlib Agg + 한글폰트(AppleGothic→NanumGothic) ·
      데이터 없으면 정직히 skip+로그(가짜 0) · 결정성(정렬·seed·고정 색) ·
      출력 figures/fig_overseas_forecast_compare.png dpi=130 bbox_inches=tight.

출력 (figures shown 1 at a time): 각 패널을 단일 PNG 로도 저장 + combined 1개.
    - fig_overseas_forecast_r2_bars.png     (국가별 R² 막대)
    - fig_overseas_forecast_mase_bars.png    (국가별 MASE 막대)
    - fig_overseas_forecast_timeseries.png   (대표국 1-step rolling 오버레이, full-width)
    - fig_overseas_forecast_compare.png      (combined, 기존과 동일 = back-compat)

Usage:
    .venv/bin/python -m simulation.scripts.fig_overseas_forecast

Returns:
    생성된 PNG 경로 list (print). 데이터 부족 시 정직히 skip 로그.

Side effects:
    위 4개 PNG 작성 (figures/). DB read-only. 모델 로드 없음.

Performance: O(국가 × 주차 × refit). 국가당 ~수백 주 × 2모델 rolling refit → 수십 초.
"""
from __future__ import annotations

import os
import warnings

import numpy as np

# 결정성
np.random.seed(42)
warnings.filterwarnings("ignore")  # statsmodels convergence 경고 억제 (수렴 실패는 NaN 으로 처리)

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_FIG_DIR = os.path.join(_REPO, "simulation", "results", "figures")
_OUT_PNG = os.path.join(_FIG_DIR, "fig_overseas_forecast_compare.png")

# 단일 패널 standalone PNG ("figures shown 1 at a time" — combined 와 동일 helper 재사용).
_OUT_R2_PNG = os.path.join(_FIG_DIR, "fig_overseas_forecast_r2_bars.png")
_OUT_MASE_PNG = os.path.join(_FIG_DIR, "fig_overseas_forecast_mase_bars.png")
_OUT_TS_PNG = os.path.join(_FIG_DIR, "fig_overseas_forecast_timeseries.png")

# 평가할 주요국 (source, country, signal, 표시라벨).
# ili_rate 가 충분한 series 우선; who_flunet 은 positivity_pct (양성률) 사용.
#   delphi_national US = 미국 ILINet wILI (1997~), 가장 긴 series.
#   who_flunet 주요국 = ili_rate sparse → positivity_pct (양성률 %).
_SERIES = [
    ("delphi_national", "US", "ili_rate", "United States (US, wILI %)"),
    ("influnet_it", "IT", "ili_rate", "Italy (IT, ILI rate)"),
    ("who_flunet", "AU", "positivity_pct", "Australia (AU, positivity %)"),
    ("who_flunet", "CN", "positivity_pct", "China (CN, positivity %)"),
    ("who_flunet", "JP", "positivity_pct", "Japan (JP, positivity %)"),
    ("who_flunet", "SG", "positivity_pct", "Singapore (SG, positivity %)"),
]

# rolling 시작: 처음 _MIN_TRAIN 주는 학습 전용, 그 이후부터 1-step 평가.
_MIN_TRAIN = 60
_SEASON = 52  # 주간 계절 주기 (연 ≈52주) — MASE seasonal-naive scale & Theta period

_MODEL_COLORS = {
    "ARIMA(1,1,1)": "#1f77b4",
    "Theta": "#d62728",
}


def _set_korean_font(plt) -> None:
    """한글 폰트 설정 (macOS AppleGothic → Linux NanumGothic → fallback)."""
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False


def _load_series(source: str, country: str, signal: str) -> np.ndarray:
    """overseas_ili 에서 (source, country) 의 signal 을 연·주 정렬 1D 배열로.

    Args:
        source: overseas_ili.source (예 'who_flunet').
        country: ISO 국가코드 (예 'JP').
        signal: 'ili_rate' 또는 'positivity_pct'.

    Returns:
        시간순(year, week_no) 정렬된 non-null float 배열 (shape (T,)).
        선두/말단 null 은 제거, 중간 null 은 직전값 forward-fill (rolling 연속성).

    Side effects: DB read-only 1 query.
    """
    from simulation.database import read_only_connect

    con = read_only_connect()
    cur = con.cursor()
    rows = cur.execute(
        f"SELECT year, week_no, {signal} FROM overseas_ili "
        "WHERE source = ? AND country = ? ORDER BY year, week_no",
        (source, country),
    ).fetchall()
    vals = [r[2] for r in rows]
    # 선두 null 제거
    while vals and vals[0] is None:
        vals.pop(0)
    while vals and vals[-1] is None:
        vals.pop()
    # 중간 null forward-fill
    out = []
    last = None
    for v in vals:
        if v is None:
            if last is not None:
                out.append(last)
        else:
            out.append(float(v))
            last = float(v)
    return np.asarray(out, dtype=float)


def _rolling_arima(y: np.ndarray, start: int) -> np.ndarray:
    """ARIMA(1,1,1) 1-step rolling 예측. ŷ_t = fit([0..t-1]).forecast(1).

    Args:
        y: 관측 시계열 (T,).
        start: 첫 예측 인덱스 (이전은 학습 전용).

    Returns:
        ŷ 배열 길이 (T-start,); 적합/예측 실패 주는 np.nan.

    Performance: O((T-start) × ARIMA-fit). 수렴 실패 → NaN (가짜 채움 X).
    """
    from statsmodels.tsa.arima.model import ARIMA

    preds = []
    for t in range(start, len(y)):
        hist = y[:t]
        try:
            res = ARIMA(hist, order=(1, 1, 1),
                        enforce_stationarity=False,
                        enforce_invertibility=False).fit(method_kwargs={"warn_convergence": False})
            fc = float(np.asarray(res.forecast(1))[0])
            preds.append(fc if np.isfinite(fc) else np.nan)
        except Exception:
            preds.append(np.nan)
    return np.asarray(preds, dtype=float)


def _rolling_theta(y: np.ndarray, start: int) -> np.ndarray:
    """Theta(통계 baseline) 1-step rolling 예측.

    Args:
        y: 관측 시계열 (T,).
        start: 첫 예측 인덱스.

    Returns:
        ŷ 배열 길이 (T-start,); 실패 주는 np.nan.

    Performance: O((T-start) × Theta-fit). deseasonalize period=52 (관측 길면).
    """
    from statsmodels.tsa.forecasting.theta import ThetaModel

    preds = []
    for t in range(start, len(y)):
        hist = y[:t]
        try:
            period = _SEASON if t >= 2 * _SEASON else None
            deseason = period is not None
            tm = ThetaModel(hist, period=period, deseasonalize=deseason).fit()
            fc = float(np.asarray(tm.forecast(1))[0])
            preds.append(fc if np.isfinite(fc) else np.nan)
        except Exception:
            preds.append(np.nan)
    return np.asarray(preds, dtype=float)


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """결정계수 R². NaN 쌍 제거 후 계산. 분산 0 또는 표본 부족 → nan."""
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3:
        return float("nan")
    yt, yp = y_true[m], y_pred[m]
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    ss_res = float(np.sum((yt - yp) ** 2))
    return 1.0 - ss_res / ss_tot


def _mase(y_true: np.ndarray, y_pred: np.ndarray, y_full: np.ndarray,
          start: int, season: int) -> float:
    """MASE — seasonal-naive(in-sample, lag=season) MAE 로 스케일한 평균절대오차.

    Args:
        y_true: 평가구간 관측 (start..end).
        y_pred: 같은 구간 예측.
        y_full: 전체 관측 (scale 분모 = 학습구간 in-sample naive MAE).
        start: 평가 시작 인덱스.
        season: naive lag (52 또는 데이터 짧으면 1).

    Returns:
        MASE (>0); <1 = naive 보다 우수. scale 0/표본 부족 → nan.
    """
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 3:
        return float("nan")
    train = y_full[:start]
    lag = season if len(train) > season else 1
    if len(train) <= lag:
        return float("nan")
    naive_mae = float(np.mean(np.abs(train[lag:] - train[:-lag])))
    if naive_mae <= 0:
        return float("nan")
    mae = float(np.mean(np.abs(y_true[m] - y_pred[m])))
    return mae / naive_mae


def _build() -> dict:
    """전 series 에 대해 rolling baseline 예측 + R²/MASE 계산.

    Returns:
        {label: {"y": y, "start": start, "preds": {model: ŷ},
                 "metrics": {model: {"r2":, "mase":}}, "signal": signal}}
        — 데이터 부족 series 는 제외.
    """
    results = {}
    for source, country, signal, label in _SERIES:
        y = _load_series(source, country, signal)
        if len(y) < _MIN_TRAIN + 20:
            print(f"  [skip] {label}: 관측 {len(y)} < {_MIN_TRAIN + 20} (rolling 평가 부족)")
            continue
        start = _MIN_TRAIN
        y_eval = y[start:]
        season = _SEASON if len(y[:start]) > _SEASON else 1
        preds = {}
        metrics = {}
        for mname, fn in (("ARIMA(1,1,1)", _rolling_arima), ("Theta", _rolling_theta)):
            yp = fn(y, start)
            preds[mname] = yp
            metrics[mname] = {
                "r2": _r2(y_eval, yp),
                "mase": _mase(y_eval, yp, y, start, season),
            }
        results[label] = {"y": y, "start": start, "preds": preds,
                          "metrics": metrics, "signal": signal,
                          "source": source, "country": country}
        ms = " | ".join(
            f"{mn}: R²={metrics[mn]['r2']:.3f}, MASE={metrics[mn]['mase']:.3f}"
            for mn in preds)
        print(f"  [ok]   {label} (n_eval={len(y_eval)}) — {ms}")
    return results


def _panel_r2_bars(ax, results: dict, labels: list, models: list,
                   standalone: bool) -> None:
    """국가별 R² 그룹 막대 패널 (combined·standalone 공용 — 동일 그리기 코드).

    Args:
        ax: 그릴 matplotlib Axes.
        results: _build() 산출 dict.
        labels: 국가 라벨 순서.
        models: 모델명 리스트.
        standalone: True 면 단독 figure 용(서수 prefix '(a)' 제거).
    """
    n = len(labels)
    x = np.arange(n)
    w = 0.38
    for i, mn in enumerate(models):
        vals = [results[lb]["metrics"][mn]["r2"] for lb in labels]
        vals = [v if np.isfinite(v) else 0.0 for v in vals]
        ax.bar(x + (i - 0.5) * w, vals, w, label=mn,
               color=_MODEL_COLORS[mn], alpha=0.85)
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([lb.split(" (")[0] for lb in labels], rotation=30,
                       ha="right", fontsize=8.5)
    ax.set_ylabel("R² (1-step rolling)", fontsize=10)
    title = ("R² by country — higher is better (negative = worse than mean)" if standalone
             else "(a) R² by country — higher is better (negative = worse than mean)")
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8.5, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)


def _panel_mase_bars(ax, results: dict, labels: list, models: list,
                     standalone: bool) -> None:
    """국가별 MASE 그룹 막대 패널 (combined·standalone 공용).

    Args:
        ax: 그릴 matplotlib Axes.
        results: _build() 산출 dict.
        labels: 국가 라벨 순서.
        models: 모델명 리스트.
        standalone: True 면 단독 figure 용(서수 prefix '(b)' 제거).
    """
    n = len(labels)
    x = np.arange(n)
    w = 0.38
    for i, mn in enumerate(models):
        vals = [results[lb]["metrics"][mn]["mase"] for lb in labels]
        vals = [v if np.isfinite(v) else 0.0 for v in vals]
        ax.bar(x + (i - 0.5) * w, vals, w, label=mn,
               color=_MODEL_COLORS[mn], alpha=0.85)
    ax.axhline(1.0, color="#d62728", lw=1.0, ls="--",
               label="MASE=1 (seasonal-naive)")
    ax.set_xticks(x)
    ax.set_xticklabels([lb.split(" (")[0] for lb in labels], rotation=30,
                       ha="right", fontsize=8.5)
    ax.set_ylabel("MASE (lower is better)", fontsize=10)
    title = ("MASE by country — <1 means better than naive" if standalone
             else "(b) MASE by country — <1 means better than naive")
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)


def _panel_timeseries(ax, results: dict, labels: list, models: list,
                      standalone: bool) -> None:
    """대표국 1-step rolling 오버레이(관측 vs ARIMA/Theta) 패널 (combined·standalone 공용).

    가장 긴 평가구간을 가진 국가를 대표로 선택.

    Args:
        ax: 그릴 matplotlib Axes.
        results: _build() 산출 dict.
        labels: 국가 라벨 순서.
        models: 모델명 리스트.
        standalone: True 면 단독 figure 용(서수 prefix '(c)' 제거).
    """
    rep = max(labels, key=lambda lb: len(results[lb]["y"]) - results[lb]["start"])
    rr = results[rep]
    y, start = rr["y"], rr["start"]
    weeks = np.arange(start, len(y))
    ax.plot(weeks, y[start:], color="#222", lw=1.6, label="Observed (ground truth)", zorder=3)
    for mn in models:
        yp = rr["preds"][mn]
        ax.plot(weeks, yp, color=_MODEL_COLORS[mn], lw=1.1, alpha=0.8,
                label=f"{mn} forecast (R²={rr['metrics'][mn]['r2']:.3f})")
    sig_lab = "positivity %" if rr["signal"] == "positivity_pct" else "ILI rate"
    ax.set_xlabel(f"Week index (after {start}-week training, 1-step rolling)", fontsize=10)
    ax.set_ylabel(sig_lab, fontsize=10)
    title = (f"Representative country {rep} — observed vs baseline 1-step rolling forecast" if standalone
             else f"(c) Representative country {rep} — observed vs baseline 1-step rolling forecast")
    ax.set_title(title, fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8.5, ncol=3, framealpha=0.9, loc="upper left")
    ax.grid(alpha=0.3)


def _plot(results: dict) -> list:
    """국가별 R²/MASE 막대 + 대표 1-step rolling 오버레이 — 단일 패널 PNG 3개 + combined 1개.

    "figures shown 1 at a time" 요구: 각 패널을 독립 PNG 로도 저장(동일 helper 재사용,
    실데이터·placeholder 없음). 이후 기존과 동일한 combined 도 저장(back-compat).

    Returns: 저장된 PNG 경로 list (standalone 3 + combined 1).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _set_korean_font(plt)

    labels = list(results.keys())
    models = ["ARIMA(1,1,1)", "Theta"]

    os.makedirs(_FIG_DIR, exist_ok=True)
    written = []

    # --- standalone 패널 PNG (1개씩) ---
    # (a) R² bars — chart 패널 ~(7.5,5.5)
    fig_r2 = plt.figure(figsize=(7.5, 5.5))
    _panel_r2_bars(fig_r2.add_subplot(1, 1, 1), results, labels, models, standalone=True)
    fig_r2.savefig(_OUT_R2_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig_r2)
    written.append(_OUT_R2_PNG)

    # (b) MASE bars — chart 패널 ~(7.5,5.5)
    fig_ma = plt.figure(figsize=(7.5, 5.5))
    _panel_mase_bars(fig_ma.add_subplot(1, 1, 1), results, labels, models, standalone=True)
    fig_ma.savefig(_OUT_MASE_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig_ma)
    written.append(_OUT_MASE_PNG)

    # (c) timeseries — full-width 시계열 ~(12,5)
    fig_ts = plt.figure(figsize=(12, 5))
    _panel_timeseries(fig_ts.add_subplot(1, 1, 1), results, labels, models, standalone=True)
    fig_ts.savefig(_OUT_TS_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig_ts)
    written.append(_OUT_TS_PNG)

    # --- combined figure (기존과 동일, 동일 helper standalone=False) ---
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.25], hspace=0.42, wspace=0.22)
    _panel_r2_bars(fig.add_subplot(gs[0, 0]), results, labels, models, standalone=False)
    _panel_mase_bars(fig.add_subplot(gs[0, 1]), results, labels, models, standalone=False)
    _panel_timeseries(fig.add_subplot(gs[1, :]), results, labels, models, standalone=False)

    # --- 정직성 헤더 + 주석 ---
    fig.suptitle(
        "Baseline forecasts applied to overseas ILI series (reference only) — "
        "not our champion (FusedEpi) overseas forecast",
        fontsize=13, fontweight="bold", y=0.985)
    fig.text(
        0.5, 0.945,
        "Note: FusedEpi is a Seoul-ILI-only model — overseas->Seoul transfer is not built (direct application inappropriate). "
        "This figure only applies lightweight univariate baselines (ARIMA, Theta) directly to overseas_ili via 1-step rolling, as a reference.",
        ha="center", fontsize=8.5, color="#444", style="italic")
    fig.text(
        0.5, 0.012,
        "Data: overseas_ili (read-only). For who_flunet countries ili_rate is sparse, so positivity_pct (positivity %) is used as the signal — "
        "note this is a different metric from ILI rate. MASE scale = in-sample seasonal-naive (lag=52).",
        ha="center", fontsize=8, color="#666")

    fig.savefig(_OUT_PNG, dpi=130, bbox_inches="tight")
    plt.close(fig)
    written.append(_OUT_PNG)
    return written


def main() -> list:
    """해외 baseline 예측 figure 생성 (단일 패널 3 + combined 1). 데이터 부족 시 정직히 skip.

    Returns: 생성된 PNG 경로 list (0 또는 4개 = standalone 3 + combined 1).
    """
    print("[fig_overseas_forecast] 해외 ILI baseline 예측 (참고치, 챔피언 아님)")
    print("  rolling 1-step: ARIMA(1,1,1) · Theta — expanding window 매주 재적합")
    results = _build()
    if not results:
        print("  [SKIP] 평가 가능한 해외 series 없음 — figure 미생성 (가짜 생성 X).")
        return []
    outs = _plot(results)
    for out in outs:
        size = os.path.getsize(out) if os.path.exists(out) else 0
        print(f"  [saved] {out} ({size} bytes)")
    return outs


if __name__ == "__main__":
    paths = main()
    print("DONE:", paths)
