"""fig_overseas_states_full.py — 챔피언 foundation base(TiRex) 전 US 주 ILI 일반화.

챔피언 ``FusedEpiForecaster`` 의 **foundation base = TiRex** (NX-AI xLSTM 35M 시계열
foundation) 만 떼어, 미국 *전* 주(state-level) ILI 시계열에 calendar-locked forward
(전향) rolling 1-step 으로 그대로 적용하여 **Seoul 밖 일반화**를 전수(全數) 실측한다.

★정직 표기 (ENGINEERING_PRINCIPLES.md D-5 gray-box / #5 정직성):
- 본 figure 의 예측기 = **TiRex-core (foundation base) 단독**. 챔피언 풀 FusedEpi 의
  ① TabPFN 잔차보정 ② mechanistic(Rt/FoI) 채널 ③ NegBin·conformal·동적α 는 **빠진** base.
  → 풀 FusedEpi 의 주별 상세판은 ``fig_champion_us_states.png`` 참조 (보완 관계).
- TiRex native 분위 = {0.1,…,0.9} → WIS 는 native (0.1,0.9)·(0.2,0.8) 구간 사용
  (풀 FusedEpi 의 95%/50% conformal 구간과 다름 — 정직 명시).

데이터/방법:
- **실데이터만**: ``overseas_ili_regional`` (country='USA', region = 2-letter state
  code + DC) — read_only_connect, 파라미터화 쿼리(literal-0).
  데이터 없으면 정직 skip (합성/가짜 데이터 절대 생성 X).
  ※ (year,week) 당 복수 source 행(delphi/nssp/nwss 등 ~2–4행)은 **ISO 주 단위 평균**으로
    1주=1값 통합(``_load_state_series``). 통합 안 하면 같은 주가 시계열에 여러 번 끼어
    lag/forward split 이 깨진다.
- **데이터 품질 가드**: max ILI% > ``MAX_PLAUSIBLE_ILI``(=50) 주는 단위/입력오류로 보고 제외.
  제외 주 수·주명은 출력 CSV(SKIPPED 행)·보고에 정직 명시.
- **★calendar-locked forward eval (2026-06-26 개정)**: 옛 ``TEST_FRAC=0.25`` 비율 split
  대신 **공통 캘린더 경계**(in-sample 종료=2026-02-09, forward=2026-02-16 이후)로 split.
  ISO (year, week_no) → 월요일 날짜 ≤ 02-09 = in-sample, 초과 = forward.
  ⇒ 전 주 forward 가 **2026-W08(02-16)** 로 정렬. forward = in-sample 컨텍스트 시드 후 관측 y
  흘리며(``y_observed``) 1주씩 1-step 예측 → R²/WIS. leakage-free (각 t 예측은 y[<t]만).
- **공통 forward 창 (가짜 연장 금지)**: 주별 forward 길이가 데이터 끝 차이로 다르면
  ``common_forward_len`` 로 공통 최소(또는 ``FORWARD_WEEKS_CAP``)에 truncate → figure/CSV 명시.
- **baseline**: persistence(lag1) · seasonal-naive(작년 같은 주, lag52).
- **★Seoul 기준선 = 라이브 forward (하드코드 제거)**: 옛 0.9357(per_model_eval test-slab,
  forward 아님) 삭제 → ``compute_seoul_forward_baseline`` 가 **동일 프로토콜**(feature_cache
  ili_rate+week_start, in-sample≤02-09, 풀 FusedEpi rolling, 공통 창 truncate)로 Seoul forward
  R²/WIS 를 **실시간 계산**해 점선 참조선으로 사용 — '관측'이 아니라 **모델-유래** (figure/CSV
  'source' 명시). (TiRex-core base 와 직접 동급 비교 아님: 풀 FusedEpi 참조선.)
  계산 실패(캐시 부재) 시 기준선 선/행 생략(가짜 값 박제 X).

핵심 메시지:
  "한국 구별=관측 per-gu ILI 부재라 불가 / 해외 US 주별=관측 per-state ILI 존재라 가능
   → 챔피언 foundation(TiRex) 이 N개 주에서 R²>0.5 (Seoul 밖 일반화 실측)."

성능: 주당 TiRex rolling 1-step ~수십 초. CPU. 50+주 전수 ~수 분.
부작용: ``simulation/results/figures/fig_overseas_states_full.png`` (dpi=130) +
        ``…/fig_overseas_states_full.csv`` 작성. DB read-only. 결정성(seed 42).

실행:
    .venv/bin/python -m simulation.scripts.fig_overseas_states_full

smoke (1주만 calendar-locked forward 검증 후 종료, figure/CSV 미작성):
    MPH_OVERSEAS_SMOKE=CA .venv/bin/python -m simulation.scripts.fig_overseas_states_full
    # 또는 'USA:CA' 형식도 허용 (country prefix 무시).
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import numpy as np

from simulation.database import read_only_connect
from simulation.scripts._overseas_forward import (
    common_forward_len,
    compute_seoul_forward_baseline,
    get_in_sample_end,
    isoweek_monday,
    split_forward_by_isoweek,
)

log = logging.getLogger(__name__)

# ── 재현성 (ENGINEERING_PRINCIPLES.md #5) ──────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── 평가기간 (Seoul 챔피언과 겹치는 구간 + forward 2026 포함) ────────────────
YEAR_LO, YEAR_HI = 2019, 2026     # ★forward(2026-02-16 이후) 포함하도록 2026 까지.

# ── 데이터 품질 가드 ───────────────────────────────────────────────────────
MAX_PLAUSIBLE_ILI = 50.0   # ILINet ILI%는 통상 0–40%; >50%는 source 단위/입력오류.
MIN_WEEKS = 120            # TiRex 컨텍스트 시드 + in-sample 여유.
FORWARD_WEEKS_CAP = 18     # 공통 forward 창 상한(주). 데이터 가용 min 과 함께 작은 쪽 사용.
MIN_TRAIN = 70             # TiRex 컨텍스트 시드 하한(min_ctx 52 + 여유).
TIREX_CTX = 256            # TiRex max_context (FusedEpi._tirex_1step 와 동일).
TIREX_REPO = "NX-AI/TiRex"

# 제외할 영토(territory) — 50 주 + DC 만 평가(per-state ILI 의도와 일치).
TERRITORY_EXCLUDE = {"AS", "GU", "MP", "PR", "VI"}

# ── Seoul 기준선 = 라이브 forward (하드코드 제거, 2026-06-26) ──────────────────
#   옛 0.9357(per_model_eval test-slab)은 forward 가 아니라 hold-out 평가 → 비교 부적합.
#   main() 에서 compute_seoul_forward_baseline(공통창) 로 **동일 프로토콜** 실시간 계산해 채운다.
#   (Seoul 기준선은 풀 FusedEpi forward — TiRex-core 와 직접 동급 아닌 참조선.)

# TiRex native 분위 (config.quantiles) — WIS 용 사용 구간.
TIREX_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
WIS_PAIRS = [(0.1, 0.9), (0.2, 0.8)]   # TiRex native 80%/60% 구간 (95%/50% 아님 — 정직).


def _setup_matplotlib():
    """matplotlib Agg + 한글폰트(AppleGothic→NanumGothic) 설정.

    Returns:
        matplotlib.pyplot 모듈 (Agg 백엔드 고정).
    Side effects: rcParams 전역 폰트 설정.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    return plt


def _list_states(con) -> list[str]:
    """평가 대상 US 주(2-letter state + DC, 영토 제외) 목록 — 알파벳순.

    Args:
        con: read_only sqlite 연결.

    Returns:
        주 약자 리스트 (예: ['AK','AL',…,'WY']). 영토(AS/GU/MP/PR/VI) 제외.

    Caller responsibility: con 은 read_only_connect 산출이어야 함.
    """
    cur = con.cursor()
    rows = cur.execute(
        "SELECT DISTINCT region FROM overseas_ili_regional WHERE country = 'USA'"
    ).fetchall()
    out = []
    for (reg,) in rows:
        if reg in TERRITORY_EXCLUDE:
            continue
        if re.fullmatch(r"[A-Z]{2}", reg or ""):   # 2-letter state code + DC
            out.append(reg)
    return sorted(out)


def _load_state_series(
    con, state: str,
) -> tuple[np.ndarray | None, list[tuple[int, int]], str]:
    """US 주 ILI 시계열 로드 (ISO주 단위 중복제거·정렬, 2019–2026) + skip 사유.

    Args:
        con: read_only sqlite 연결.
        state: 주 약자 (예: 'CA').

    Returns:
        (y, yw, reason): y=(T,) float ndarray(주별 ili_rate, ISO 시간순) 또는 None.
        yw=[(year, week_no), …] 동일 순서(skip 시 []). reason="" (정상) 또는 skip 사유
        ("few_weeks:.."/"unit_error:.."). yw 는 calendar-locked forward split 에 사용.
        ※ (year,week) 당 복수 source 행(USA delphi/nssp/nwss 등)은 **평균**으로 1주=1값 통합.

    Caller responsibility: con 은 read_only_connect 산출이어야 함.
    """
    cur = con.cursor()
    rows = cur.execute(
        "SELECT year, week_no, ili_rate FROM overseas_ili_regional "
        "WHERE country = 'USA' AND region = ? AND year BETWEEN ? AND ? "
        "AND ili_rate IS NOT NULL "
        "ORDER BY year ASC, week_no ASC",
        (state, YEAR_LO, YEAR_HI),
    ).fetchall()
    if not rows:
        return None, [], "no_data"
    # (year,week) 당 복수 source 행 평균으로 통합 (1주=1값) → ISO 월요일 날짜 기준 정렬.
    import datetime as _dt
    from collections import defaultdict
    agg: dict[tuple[int, int], list[float]] = defaultdict(list)
    for yy, ww, vv in rows:
        agg[(int(yy), int(ww))].append(float(vv))
    yw = sorted(agg.keys(), key=lambda k: (isoweek_monday(*k) or _dt.date.min, k))
    y = np.asarray([float(np.mean(agg[k])) for k in yw], dtype=float)
    if y.size < MIN_WEEKS:
        return None, [], f"few_weeks({y.size}<{MIN_WEEKS})"
    ymax = float(np.nanmax(y))
    if ymax > MAX_PLAUSIBLE_ILI:
        return None, [], f"unit_error(max={ymax:.0f}%)"
    y = np.nan_to_num(np.clip(y, 0.0, None), nan=0.0)   # 음수/NaN 위생(정상값 보존)
    return y, yw, ""


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, float).ravel(); y_pred = np.asarray(y_pred, float).ravel()
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _wis_native(y_true: np.ndarray, q_dict: dict) -> float:
    """Weighted Interval Score (proper, lower=better) — TiRex native 분위 소비.

    Args:
        y_true: (n,) 관측.
        q_dict: {level: (n,) 분위예측} — 0.5(median) + WIS_PAIRS 의 대칭쌍 포함.

    Returns:
        평균 WIS (float). median 절대오차 + 구간 interval score 의 가중평균.
        ※ TiRex native (0.1,0.9)·(0.2,0.8) 구간 사용 — 풀 FusedEpi 의 95%/50%
          conformal WIS 와 직접 동급 비교 불가(정직 명시).
    """
    y_true = np.asarray(y_true, float).ravel()
    med = np.asarray(q_dict[0.5], float).ravel()
    total = 0.5 * np.abs(y_true - med)            # K=0 (median) 항
    K = 0
    for lo, hi in WIS_PAIRS:
        if lo not in q_dict or hi not in q_dict:
            continue
        l = np.asarray(q_dict[lo], float).ravel()
        u = np.asarray(q_dict[hi], float).ravel()
        alpha = 1.0 - (hi - lo)
        is_score = (u - l) \
            + (2.0 / alpha) * (l - y_true) * (y_true < l) \
            + (2.0 / alpha) * (y_true - u) * (y_true > u)
        total = total + (alpha / 2.0) * is_score
        K += 1
    return float(np.mean(total / (K + 0.5)))


def _tirex_core_rolling(tx, y_train: np.ndarray, y_obs: np.ndarray):
    """TiRex-core(foundation base) rolling 1-step 예측 — TabPFN/mechanistic 없는 순수 base.

    각 forward 시점 t 의 예측은 [y_train, y_obs[:i]] (관측 흘리기) 컨텍스트의 1-step forecast.
    leakage-free: y[<t] 만 사용. FusedEpi._tirex_1step 과 동일 컨텍스트 규약(max_context).

    Args:
        tx: load_model("NX-AI/TiRex") 산출 모델.
        y_train: (n_train,) 관측 시계열.
        y_obs: (forward_len,) forward 구간 관측(rolling 흘리기 용).

    Returns:
        (y_pred, q_dict): y_pred=(forward_len,) median(0.5) 예측,
        q_dict={level:(forward_len,)} TiRex native 분위(WIS 용).

    Performance: forward_len × TiRex 1-step (CPU). Side effects: 없음(모델 forward만).
    """
    import torch

    n_fwd = len(y_obs)
    preds = np.empty(n_fwd, dtype=float)
    qcols = {lv: np.empty(n_fwd, dtype=float) for lv in TIREX_LEVELS}
    med_idx = TIREX_LEVELS.index(0.5)
    for i in range(n_fwd):
        hist = np.concatenate([y_train, y_obs[:i]]) if i > 0 else y_train
        ctx = torch.tensor(hist[-TIREX_CTX:], dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q, _mean = tx.forecast(context=ctx, prediction_length=1)
        qa = np.asarray(q, dtype=float).ravel()      # (9,) native 분위
        preds[i] = qa[med_idx]
        for j, lv in enumerate(TIREX_LEVELS):
            qcols[lv][i] = qa[j]
    return preds, qcols


def _eval_state(tx, state: str, y: np.ndarray, n_train: int, forward_len: int) -> dict | None:
    """한 주에 대해 calendar-locked forward(TiRex-core rolling 1-step) + baseline 평가.

    Args:
        tx: TiRex 모델.
        state: 주 약자.
        y: (T,) 관측 ILI 시계열(ISO 시간순, 1주=1값).
        n_train: in-sample 길이(주 월요일 ≤ in_sample_end; calendar-locked split).
        forward_len: 평가 forward 주수(공통 창으로 truncate 된 값).

    Returns:
        {state, n_train, n_test(=n_forward), base_r2, base_wis, persist_r2, seasonal_r2,
         persist_wis, seasonal_wis, peak} 또는 in-sample 부족/실패 시 None.

    Performance: TiRex rolling(native forward) forward_len×(1-step). CPU.
    """
    if n_train < MIN_TRAIN or forward_len <= 0:
        log.info("  [skip] %s: split 부적합 (n_train=%d forward=%d)", state, n_train, forward_len)
        return None

    y_tr = y[:n_train]
    sl = slice(n_train, n_train + forward_len)
    y_te = y[sl]
    y_obs = y[sl]                                     # rolling: 관측 흘리기(leakage-free 1-step)

    try:
        y_pred, q = _tirex_core_rolling(tx, y_tr, y_obs)
    except Exception as e:                            # fail-honest: skip + 기록
        log.warning("  [skip] %s: TiRex-core 실패 — %s", state, e)
        return None

    base_r2 = _r2(y_te, y_pred)
    base_wis = _wis_native(y_te, q)

    # ── baseline (관측 기반, 모델-유래 아님) — forward 창 안에서 1-step ──
    persist = y[n_train - 1: n_train + forward_len - 1]   # lag1 (persistence)
    seasonal = np.array([y[n_train + i - 52] if n_train + i - 52 >= 0 else y[n_train - 1]
                         for i in range(forward_len)])    # 작년 같은 주 (seasonal-naive)
    persist_r2 = _r2(y_te, persist)
    seasonal_r2 = _r2(y_te, seasonal)
    persist_wis = float(np.mean(0.5 * np.abs(y_te - persist) / 1.5))
    seasonal_wis = float(np.mean(0.5 * np.abs(y_te - seasonal) / 1.5))

    log.info("  [%s] n_tr=%d n_fwd=%d | TiRex-core R2=%.3f WIS=%.3f | persist R2=%.3f seas R2=%.3f",
             state, n_train, forward_len, base_r2, base_wis, persist_r2, seasonal_r2)
    return {
        "state": state, "n_train": n_train, "n_test": forward_len,
        "base_r2": base_r2, "base_wis": base_wis,
        "persist_r2": persist_r2, "seasonal_r2": seasonal_r2,
        "persist_wis": persist_wis, "seasonal_wis": seasonal_wis,
        "peak": float(np.max(y_te)),
    }


def _panel_states_r2(ax, res: list[dict], r2s: np.ndarray, states: list[str],
                     seoul: dict | None, forward_weeks: int, n_gt05: int,
                     standalone: bool) -> None:
    """패널: 전 US 주 R² 막대 (TiRex-core vs persistence vs seasonal-naive) + Seoul 기준선.

    Args:
        ax: 그릴 Axes.
        res: base_r2 내림차순 정렬된 _eval_state dict 리스트.
        r2s: res 순서의 base_r2 ndarray.
        states: res 순서의 주 약자 리스트.
        seoul: Seoul 라이브 기준선 dict 또는 None.
        forward_weeks: 공통 forward 창(주수).
        n_gt05: R²>0.5 주 수 (제목 명시).
        standalone: True 면 단독 figure (제목에서 '(1)' 서수 제거).

    Side effects: ax 에 3-군 막대/기준선 그림.
    """
    n = len(res)
    x = np.arange(n)
    w = 0.27
    colors = ["#2c7fb8" if v > 0.5 else ("#7fcdbb" if v > 0 else "#d9d9d9") for v in r2s]
    ax.bar(x - w, r2s, w, label="TiRex-core (foundation base)", color=colors, edgecolor="none")
    ax.bar(x, [r["persist_r2"] for r in res], w, label="persistence(lag1)", color="#bdbdbd", alpha=0.8)
    ax.bar(x + w, [r["seasonal_r2"] for r in res], w, label="seasonal-naive(lag52)",
           color="#fdae6b", alpha=0.85)
    if seoul is not None:
        ax.axhline(seoul["r2"], ls="--", color="#d7301f", lw=2,
                   label=f"Seoul live forward R²={seoul['r2']:.3f} "
                         f"(full FusedEpi, same protocol, n={seoul['n_test']})")
    ax.axhline(0.5, ls=":", color="#238b45", lw=1.5, label="R²=0.5 (generalization threshold)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x); ax.set_xticklabels(states, fontsize=8, rotation=90)
    ax.set_ylabel(f"R² (calendar-locked forward, common {forward_weeks} weeks)")
    lo = min(-0.5, float(np.min([r["seasonal_r2"] for r in res])) - 0.1)
    ax.set_ylim(max(lo, -3.0), 1.05)
    prefix = "" if standalone else "(1) "
    ax.set_title(f"{prefix}All US states generalization — TiRex-core (champion foundation base) forward R²  "
                 f"[{n_gt05}/{n} states R²>0.5]", fontsize=11)
    ax.legend(fontsize=8, loc="lower left", ncol=2)
    ax.grid(axis="y", alpha=0.3)


def _panel_r2_hist(ax, r2s: np.ndarray, n: int, n_gt05: int, med_r2: float,
                   standalone: bool) -> None:
    """패널: R² 분포 히스토그램 (median·임계선 표기).

    Args:
        ax: 그릴 Axes.
        r2s: base_r2 ndarray.
        n: 평가 주 수.
        n_gt05: R²>0.5 주 수.
        med_r2: median R².
        standalone: True 면 단독 figure (제목에서 '(2)' 서수 제거).

    Side effects: ax 에 히스토그램/수직선 그림.
    """
    bins = np.linspace(min(-1.0, float(r2s.min())), 1.0, 21)
    ax.hist(r2s, bins=bins, color="#2c7fb8", edgecolor="white", alpha=0.85)
    ax.axvline(0.5, ls=":", color="#238b45", lw=2, label="R²=0.5 threshold")
    ax.axvline(med_r2, ls="--", color="#d7301f", lw=2, label=f"median R²={med_r2:.3f}")
    ax.set_xlabel("R² (TiRex-core, calendar-locked forward)")
    ax.set_ylabel("Number of states")
    prefix = "" if standalone else "(2) "
    ax.set_title(f"{prefix}R² distribution across {n} states  (median={med_r2:.3f}, R²>0.5: {n_gt05}/{n})", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)


def _panel_wis_vs_r2(ax, res: list[dict], r2s: np.ndarray, standalone: bool) -> None:
    """패널: 주별 WIS vs R² 산점 (주 라벨; 우하단=우수).

    Args:
        ax: 그릴 Axes.
        res: base_r2 내림차순 정렬된 _eval_state dict 리스트.
        r2s: res 순서의 base_r2 ndarray.
        standalone: True 면 단독 figure (제목에서 '(3)' 서수 제거).

    Side effects: ax 에 산점/주라벨 그림.
    """
    wiss = np.array([r["base_wis"] for r in res], dtype=float)
    sc_colors = ["#2c7fb8" if v > 0.5 else ("#7fcdbb" if v > 0 else "#d9534f") for v in r2s]
    ax.scatter(r2s, wiss, c=sc_colors, s=55, edgecolor="k", linewidth=0.4, zorder=3)
    for r in res:
        ax.annotate(r["state"], (r["base_r2"], r["base_wis"]),
                    fontsize=6.5, ha="center", va="bottom", xytext=(0, 2),
                    textcoords="offset points")
    ax.axvline(0.5, ls=":", color="#238b45", lw=1.5)
    ax.set_xlabel("R² (higher is better)")
    ax.set_ylabel("WIS (lower is better, TiRex native 80/60% intervals)")
    prefix = "" if standalone else "(3) "
    ax.set_title(f"{prefix}WIS vs R² by state  (lower-right = best: high R², low WIS)", fontsize=11)
    ax.grid(alpha=0.3)


def _plot(results: list[dict], skipped: list[tuple[str, str]], out_png: Path,
          seoul: dict | None, forward_weeks: int) -> list[Path]:
    """3-패널 figure — (1) 전 주 R² 막대 vs baseline + Seoul 라이브 기준선,
    (2) R² 분포 히스토그램, (3) WIS vs R² 산점(주 라벨).

    각 패널을 ① 단독 PNG(panel별) 로 먼저 저장한 뒤 ② 동일 헬퍼로 결합 gridspec figure
    (out_png, 기존과 동일) 를 저장한다 (사용자 요구 "한 번에 한 그림씩").

    Args:
        results: _eval_state 산출 dict 리스트(주별, 평가성공).
        skipped: (state, reason) 리스트 — 단위오류/데이터부족 제외 주(주석용).
        out_png: 결합 출력 PNG 경로.
        seoul: compute_seoul_forward_baseline 산출(라이브) 또는 None(선 생략).
        forward_weeks: 공통 forward 창(주수) — 축/제목 명시.

    Returns:
        작성된 PNG 경로 리스트 (단독 3개 + 결합 1개 순서).

    Side effects: 단독 PNG 3개 + 결합 PNG(out_png) 작성 (dpi=130).
    """
    plt = _setup_matplotlib()
    res = sorted(results, key=lambda d: d["base_r2"], reverse=True)
    states = [r["state"] for r in res]
    r2s = np.array([r["base_r2"] for r in res], dtype=float)
    n = len(res)

    n_skip_unit = sum(1 for _, why in skipped if why.startswith("unit_error"))
    n_skip_few = sum(1 for _, why in skipped if why.startswith("few_weeks"))
    n_gt05 = int(np.sum(r2s > 0.5))
    med_r2 = float(np.median(r2s))

    out_dir = out_png.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # ── ① 단독 PNG (panel별) ──
    p_bars = out_dir / "fig_overseas_states_full_states_r2.png"
    fig1 = plt.figure(figsize=(17, 5.5))
    _panel_states_r2(fig1.add_subplot(111), res, r2s, states, seoul, forward_weeks,
                     n_gt05, standalone=True)
    fig1.tight_layout()
    fig1.savefig(p_bars, dpi=130, bbox_inches="tight")
    plt.close(fig1)
    written.append(p_bars)

    p_hist = out_dir / "fig_overseas_states_full_r2_hist.png"
    fig2 = plt.figure(figsize=(7, 5.5))
    _panel_r2_hist(fig2.add_subplot(111), r2s, n, n_gt05, med_r2, standalone=True)
    fig2.tight_layout()
    fig2.savefig(p_hist, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    written.append(p_hist)

    p_scatter = out_dir / "fig_overseas_states_full_wis_vs_r2.png"
    fig3 = plt.figure(figsize=(7, 5.5))
    _panel_wis_vs_r2(fig3.add_subplot(111), res, r2s, standalone=True)
    fig3.tight_layout()
    fig3.savefig(p_scatter, dpi=130, bbox_inches="tight")
    plt.close(fig3)
    written.append(p_scatter)

    # ── ② 결합 gridspec figure (out_png, 기존과 동일·back-compat) ──
    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0], hspace=0.42, wspace=0.22)
    ax1 = fig.add_subplot(gs[0, :])      # (1) 전 주 R² 막대 (가로 폭 전체)
    ax2 = fig.add_subplot(gs[1, 0])      # (2) R² 분포 히스토그램
    ax3 = fig.add_subplot(gs[1, 1])      # (3) WIS vs R² 산점
    _panel_states_r2(ax1, res, r2s, states, seoul, forward_weeks, n_gt05, standalone=False)
    _panel_r2_hist(ax2, r2s, n, n_gt05, med_r2, standalone=False)
    _panel_wis_vs_r2(ax3, res, r2s, standalone=False)

    # ── 전체 제목 + 정직 부제 ──
    skip_note = (f"{n_skip_unit} states excluded for unit/input errors"
                 + (f" · {n_skip_few} states excluded for insufficient data" if n_skip_few else ""))
    seoul_txt = (f"dashed = Seoul live forward R²={seoul['r2']:.3f} (full FusedEpi, same protocol, reference)"
                 if seoul is not None else "Seoul live baseline not computable (line omitted)")
    fig.suptitle(
        "Champion FusedEpi's foundation base = TiRex (for all-state scaling; "
        "base without TabPFN residual correction and mechanistic channel) — US all-states ILI generalization\n"
        f"calendar-locked forward (in-sample ≤ 2026-02-09, forward from 2026-02-16, common {forward_weeks} weeks) · "
        f"{n} states evaluated ({skip_note}) · {seoul_txt}\n"
        "Full FusedEpi per-state detailed version = see fig_champion_us_states.png (complementary)",
        fontsize=12.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
    written.append(out_png)
    return written


def _write_csv(results: list[dict], skipped: list[tuple[str, str]], out_csv: Path,
               seoul: dict | None, forward_weeks: int) -> None:
    """주별 결과 + 제외 주(SKIPPED) CSV 작성 — source 컬럼으로 모델-유래 명시.

    Args:
        results: _eval_state 산출 dict 리스트.
        skipped: (state, reason) 리스트.
        out_csv: 출력 CSV 경로.
        seoul: compute_seoul_forward_baseline 산출(라이브) 또는 None.
        forward_weeks: 공통 forward 창(주수) — CSV 메타.
    Side effects: out_csv 작성.
    """
    import csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["state", "year_lo", "year_hi", "forward_weeks", "n_train", "n_test", "peak_ili",
            "base_r2", "base_wis", "persist_r2", "seasonal_r2",
            "persist_wis", "seasonal_wis", "metric_source"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in sorted(results, key=lambda d: d["base_r2"], reverse=True):
            w.writerow([r["state"], YEAR_LO, YEAR_HI, forward_weeks, r["n_train"], r["n_test"],
                        f"{r['peak']:.2f}",
                        f"{r['base_r2']:.4f}", f"{r['base_wis']:.4f}",
                        f"{r['persist_r2']:.4f}", f"{r['seasonal_r2']:.4f}",
                        f"{r['persist_wis']:.4f}", f"{r['seasonal_wis']:.4f}",
                        "model-derived (TiRex-core foundation base, calendar-locked forward 1-step)"])
        # 제외 주 정직 기록.
        for state, why in sorted(skipped):
            w.writerow([state, YEAR_LO, YEAR_HI, forward_weeks, "", "", "", "", "", "", "", "", "",
                        f"SKIPPED: {why}"])
        # Seoul 풀 FusedEpi 라이브 forward 기준선 1행 (동일 프로토콜; TiRex-core 와 직접 동급 아님).
        if seoul is not None:
            w.writerow(["SEOUL_REF_FULL_FUSEDEPI", YEAR_LO, YEAR_HI, forward_weeks,
                        seoul.get("n_train", ""), seoul["n_test"], "",
                        f"{seoul['r2']:.4f}", f"{seoul['wis']:.4f}",
                        "", "", "", "", seoul["source"]])


def _resolve_smoke(spec: str) -> str:
    """smoke env 문자열을 주 약자로 해석 ('CA' 또는 'USA:CA' 형식 허용).

    Args:
        spec: MPH_OVERSEAS_SMOKE 값.

    Returns:
        대문자 주 약자 (country prefix 는 무시).
    """
    spec = spec.strip()
    if ":" in spec:
        _, _, st = spec.partition(":")
        return st.strip().upper()
    return spec.upper()


def main() -> int:
    """전 US 주 TiRex-core 일반화 평가 entry point (calendar-locked forward).

    Returns:
        0 = 성공(figure+CSV 작성) 또는 smoke 통과, 1 = 실데이터 없음/smoke 실패(정직).

    Side effects: figures/fig_overseas_states_full.{png,csv} 작성. DB read-only.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = Path(__file__).resolve().parents[1] / "results" / "figures"
    out_png = out_dir / "fig_overseas_states_full.png"
    out_csv = out_dir / "fig_overseas_states_full.csv"

    smoke = os.environ.get("MPH_OVERSEAS_SMOKE", "").strip()
    in_sample_end = get_in_sample_end()

    from tirex import load_model
    tx = load_model(TIREX_REPO, device="cpu")        # 챔피언 foundation base 단독

    con = read_only_connect()
    try:
        # ── smoke 모드 (env): 1주만 calendar-locked forward 검증 후 종료 ──
        if smoke:
            state = _resolve_smoke(smoke)
            log.info("=== SMOKE: %s — calendar-locked forward(TiRex-core) 1주 검증 ===", state)
            y, yw, reason = _load_state_series(con, state)
            if y is None:
                log.error("SMOKE: %s 데이터 부족/품질불량 (%s) — 실패", state, reason)
                return 1
            n_train = split_forward_by_isoweek(yw, in_sample_end)
            forward_len = common_forward_len([len(y) - n_train], cap=FORWARD_WEEKS_CAP)
            fwd_first = next((isoweek_monday(*yw[i]) for i in range(n_train, len(yw))), None)
            log.info("  in-sample=%d(≤%s) forward 가용=%d 공통창=%d 시작=%s",
                     n_train, in_sample_end.isoformat(), len(y) - n_train, forward_len, fwd_first)
            r = _eval_state(tx, state, y, n_train, forward_len)
            if r is None:
                log.error("SMOKE: %s forward eval 실패", state)
                return 1
            log.info("=== SMOKE OK: %s 라이브 forward R²=%.4f WIS=%.4f (n_fwd=%d, 시작 %s) ===",
                     state, r["base_r2"], r["base_wis"], r["n_test"], fwd_first)
            return 0

        states = _list_states(con)
        log.info("평가 후보 주: %d개 (영토 제외)", len(states))

        # ── 1패스: 전 주 로드 + calendar-locked split (n_train, 가용 forward 길이) ──
        loaded_all: list[tuple[str, np.ndarray, int, int]] = []
        skipped: list[tuple[str, str]] = []
        for state in states:
            y, yw, reason = _load_state_series(con, state)
            if y is None:
                if reason != "no_data":
                    skipped.append((state, reason))
                    log.info("  [skip] %s — %s", state, reason)
                continue
            n_train = split_forward_by_isoweek(yw, in_sample_end)
            avail_fwd = len(y) - n_train
            if n_train < MIN_TRAIN or avail_fwd <= 0:
                skipped.append((state, "train_short"))
                log.info("  [skip] %s: in-sample=%d forward 가용=%d (부적합)",
                         state, n_train, avail_fwd)
                continue
            loaded_all.append((state, y, n_train, avail_fwd))

        if not loaded_all:
            log.error("실데이터 없음(forward 가용 0) — figure 생성 skip (정직).")
            return 1

        # ── 공통 forward 창 (가짜 연장 금지): 가용 min 과 CAP 중 작은 쪽 ──
        forward_weeks = common_forward_len([t[3] for t in loaded_all], cap=FORWARD_WEEKS_CAP)
        log.info("[공통 forward 창] 평가 %d개 주 가용 forward = %s → 공통 %d주 truncate",
                 len(loaded_all), sorted(t[3] for t in loaded_all), forward_weeks)

        # ── 2패스: 공통 창으로 forward eval ──
        results: list[dict] = []
        for state, y, n_train, _avail in loaded_all:
            r = _eval_state(tx, state, y, n_train, forward_weeks)
            if r is not None:
                results.append(r)
            else:
                skipped.append((state, "train_short"))
    finally:
        con.close()

    if not results:
        log.error("forward eval 전부 실패 — figure 생성 skip (정직).")
        return 1

    # ── Seoul 라이브 forward 기준선 (풀 FusedEpi 동일 프로토콜·동일 공통 창) — 하드코드 대체 ──
    seoul = compute_seoul_forward_baseline(forward_cap=forward_weeks)

    written = _plot(results, skipped, out_png, seoul, forward_weeks)
    _write_csv(results, skipped, out_csv, seoul, forward_weeks)

    # ── 요약 (TiRex-core foundation base, 모델-유래 + 정직 제외) ──
    n = len(results)
    r2s = np.array([r["base_r2"] for r in results], dtype=float)
    med_r2 = float(np.median(r2s))
    n_gt05 = int(np.sum(r2s > 0.5))
    best = max(results, key=lambda d: d["base_r2"])
    worst = min(results, key=lambda d: d["base_r2"])
    n_unit = sum(1 for _, why in skipped if why.startswith("unit_error"))
    n_few = sum(1 for _, why in skipped if why.startswith("few_weeks"))
    n_short = sum(1 for _, why in skipped if why == "train_short")
    log.info("\n=== 요약 (TiRex-core, calendar-locked forward, 공통 %d주) ===", forward_weeks)
    log.info("평가 주 수: %d (in-sample≤%s, forward 2026-02-16~) | 제외: 단위오류 %d · 데이터부족 %d · train부족 %d",
             n, in_sample_end.isoformat(), n_unit, n_few, n_short)
    log.info("median R²: %.4f | R²>0.5: %d/%d 주", med_r2, n_gt05, n)
    log.info("최고: %s R²=%.4f | 최저: %s R²=%.4f",
             best["state"], best["base_r2"], worst["state"], worst["base_r2"])
    if seoul is not None:
        log.info("Seoul 라이브 forward R²=%.4f WIS=%.4f (풀 FusedEpi 동일 프로토콜 참조, n_fwd=%d)",
                 seoul["r2"], seoul["wis"], seoul["n_test"])
    else:
        log.info("Seoul 라이브 기준선 계산 불가(캐시 부재 등) — 선 생략")
    log.info("PNG (결합 + 단독 패널 %d개):", len(written))
    for p in written:
        log.info("  - %s", p)
    log.info("CSV: %s", out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
