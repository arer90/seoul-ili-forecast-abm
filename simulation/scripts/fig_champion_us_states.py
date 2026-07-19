"""fig_champion_us_states.py — 챔피언(FusedEpi) 일반화 검증: US 주별 ILI 적용.

Seoul-개념 챔피언 ``FusedEpiForecaster`` (TiRex+TabPFN foundation 융합)를 미국
대표 ~12개 주의 ILI 시계열에 그대로 적용하여 **Seoul 밖 일반화**를 실측한다.

설계 (ENGINEERING_PRINCIPLES.md D-1~D-5 / K-1~K-4 준수):
- **실데이터만**: ``overseas_ili_regional`` (country='USA') — read_only_connect.
  데이터 없으면 정직 skip (합성/가짜 데이터 절대 생성 X).
  ※ USA regional 은 (year,week) 당 복수 source 행(delphi/nssp/nwss 등 ~2–4행)이 있어
    **ISO 주 단위 평균**으로 1주=1값 통합 후 평가(``_load_state_series``). 통합 안 하면
    같은 주가 시계열에 여러 번 끼어 lag/forward split 이 깨진다.
- **★calendar-locked forward eval (2026-06-26 개정)**: 옛 ``TEST_FRAC=0.25`` 비율 split
  대신 **공통 캘린더 경계**(in-sample 종료=2026-02-09, forward=2026-02-16 이후)로 split.
  ISO (year, week_no) → 월요일 날짜 ≤ 02-09 = in-sample, 초과 = forward.
  ⇒ 전 주 forward 가 **2026-W08(02-16)** 로 정렬. forward = in-sample fit 후 관측 y 흘리며
  (``y_observed``) 1주씩 1-step 예측 → R²/WIS(leak-free). (national/regions 와 동일 프로토콜.)
- **공통 forward 창 (가짜 연장 금지)**: 주별 forward 길이가 데이터 끝 차이로 다르면
  ``common_forward_len`` 로 공통 최소(또는 ``FORWARD_WEEKS_CAP``)에 truncate → figure/CSV 명시.
- **BASIC feature + FusedEpi rolling 1-step**: ``_overseas_forward.fused_epi_forward`` 재사용
  (lag1/2/4/52 + 계절성; in-sample fit → forward 관측 흘리며 1-step). leakage-free.
- **baseline 대비**: persistence(lag1) · seasonal-naive(작년 같은 주, lag52).
- **★Seoul 기준선 = 라이브 forward (하드코드 제거)**: 옛 0.9357(per_model_eval test-slab,
  forward 아님) 삭제 → ``compute_seoul_forward_baseline`` 가 **동일 프로토콜**(feature_cache
  ili_rate+week_start, in-sample≤02-09, FusedEpi rolling, 공통 창 truncate)로 Seoul forward
  R²/WIS 를 **실시간 계산**해 기준선. '관측'이 아니라 **모델-유래** (figure/CSV 'source' 명시).
  계산 실패(캐시 부재) 시 기준선 선/행 생략(가짜 값 박제 X).

성능: 주당 FusedEpi fit(TiRex rolling 캐시) + rolling 1-step test ~수 분. CPU.
부작용: ``simulation/results/figures/fig_champion_us_states.png`` (combined, dpi=120) +
        단일 패널 PNG (figures shown 1 at a time):
          ``fig_champion_us_states_r2.png``  (R² 막대 + Seoul 기준선)
          ``fig_champion_us_states_wis.png`` (WIS 막대 + Seoul 기준선) +
        ``simulation/results/figures/fig_champion_us_states.csv`` 작성.
        DB read-only. 결정성(seed 42, rolling 결정적).

실행:
    .venv/bin/python -m simulation.scripts.fig_champion_us_states

smoke (1주만 calendar-locked forward 검증 후 종료, figure/CSV 미작성):
    MPH_OVERSEAS_SMOKE=CA .venv/bin/python -m simulation.scripts.fig_champion_us_states
    # 또는 'USA:CA' 형식도 허용 (country prefix 무시).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from simulation.database import read_only_connect
from simulation.scripts._overseas_forward import (
    common_forward_len,
    compute_seoul_forward_baseline,
    fused_epi_forward,
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

# ── 대표 US 주 (지역 다양: 서부/동부/남부/중서부/산악·뉴잉글랜드) ─────────
#   데이터 품질 사전검증(min/max plausible ILI%)으로 선별 — 단위오류 주
#   (VA=128706·OK=818·OH=133·GA=119·CO=100 등) 제외. 정직 명시.
CANDIDATE_STATES = [
    "CA",  # West / Pacific
    "WA",  # West / Pacific NW  (max~? 검증 후 가드)
    "TX",  # South Central
    "FL",  # Southeast
    "NY",  # Northeast / Mid-Atlantic
    "PA",  # Northeast / Mid-Atlantic
    "IL",  # Midwest
    "MN",  # Midwest / Upper
    "NC",  # Southeast
    "MA",  # New England
    "TN",  # South
    "CT",  # New England
]

# 데이터 품질 가드: 이 상한을 넘는 max ILI% 주는 단위/입력오류로 보고 제외.
MAX_PLAUSIBLE_ILI = 50.0   # ILINet ILI%는 통상 0–40%; >50%는 source 입력오류.
MIN_WEEKS = 120            # FusedEpi min_data=70 + in-sample 여유.
FORWARD_WEEKS_CAP = 18     # 공통 forward 창 상한(주). 데이터 가용 min 과 함께 작은 쪽 사용.

# ── Seoul 기준선 = 라이브 forward (하드코드 제거, 2026-06-26) ──────────────────
#   옛 0.9357(per_model_eval test-slab)은 forward 가 아니라 hold-out 평가 → 비교 부적합.
#   main() 에서 compute_seoul_forward_baseline(공통창) 로 **동일 프로토콜** 실시간 계산해 채운다.


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


def _load_state_series(
    con, state: str,
) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    """US 주 ILI 시계열 로드 (ISO주 단위 중복제거·정렬, 2019–2026) — calendar-locked 용.

    Args:
        con: read_only sqlite 연결.
        state: 주 약자 (예: 'CA').

    Returns:
        (y, yw) — y=(T,) float ndarray(주별 ili_rate, ISO 시간순), yw=[(year, week_no), …]
        동일 순서. 데이터 부족/품질불량 시 None. yw 는 calendar-locked forward split 에 사용.
        ※ (year,week) 당 복수 source 행(USA delphi/nssp/nwss 등)은 **평균**으로 1주=1값 통합.

    Caller responsibility: con은 read_only_connect 산출이어야 함.
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
        return None
    # (year,week) 당 복수 source 행 평균으로 통합 (1주=1값) → ISO 월요일 날짜 기준 정렬.
    import datetime as _dt
    from collections import defaultdict
    agg: dict[tuple[int, int], list[float]] = defaultdict(list)
    for yy, ww, vv in rows:
        agg[(int(yy), int(ww))].append(float(vv))
    yw = sorted(agg.keys(), key=lambda k: (isoweek_monday(*k) or _dt.date.min, k))
    y = np.asarray([float(np.mean(agg[k])) for k in yw], dtype=float)
    if y.size < MIN_WEEKS:
        log.info("  [skip] %s: %d고유주 < %d (불충분)", state, y.size, MIN_WEEKS)
        return None
    ymax = float(np.nanmax(y))
    if ymax > MAX_PLAUSIBLE_ILI:
        log.info("  [skip] %s: max ILI=%.1f%% > %.0f (단위/입력오류 의심)",
                 state, ymax, MAX_PLAUSIBLE_ILI)
        return None
    # 잔여 음수/NaN 위생 (관측 정상값은 보존).
    y = np.nan_to_num(np.clip(y, 0.0, None), nan=0.0)
    return y, yw


def _eval_state(state: str, y: np.ndarray, n_train: int, forward_len: int) -> dict | None:
    """한 주에 대해 calendar-locked forward(FusedEpi rolling 1-step) + baseline 평가.

    Args:
        state: 주 약자.
        y: (T,) 관측 ILI 시계열(ISO 시간순, 1주=1값).
        n_train: in-sample 길이(주 월요일 ≤ in_sample_end; calendar-locked split).
        forward_len: 평가 forward 주수(공통 창으로 truncate 된 값).

    Returns:
        {state, n_train, n_test(=n_forward), champ_r2, champ_wis, persist_r2,
         seasonal_r2, persist_wis, seasonal_wis, ymax} 또는 fit/부적합 시 None.

    Performance: TiRex rolling fit(캐시) + forward_len×(1-step). CPU.
    """
    if n_train < 70 or forward_len <= 0:
        log.info("  [skip] %s: split 부적합 (n_train=%d forward=%d)", state, n_train, forward_len)
        return None

    r = fused_epi_forward(y, n_train, forward_len)
    if r is None:
        log.warning("  [skip] %s: FusedEpi forward 실패", state)
        return None

    log.info("  [%s] n_tr=%d n_fwd=%d | champ R2=%.3f WIS=%.3f | persist R2=%.3f | seasonal R2=%.3f",
             state, r["n_train"], r["n_forward"],
             r["champ_r2"], r["champ_wis"], r["persist_r2"], r["seasonal_r2"])
    return {
        "state": state, "n_train": r["n_train"], "n_test": r["n_forward"],
        "champ_r2": r["champ_r2"], "champ_wis": r["champ_wis"],
        "persist_r2": r["persist_r2"], "seasonal_r2": r["seasonal_r2"],
        "persist_wis": r["persist_wis"], "seasonal_wis": r["seasonal_wis"],
        "ymax": float(np.nanmax(y)),
    }


def _panel_r2(ax, res: list[dict], states: list, x, seoul: dict | None,
              forward_weeks: int, standalone: bool) -> None:
    """R² 막대(챔피언 vs persistence vs seasonal-naive) + Seoul 기준선 패널 (공용).

    Args:
        ax: 그릴 matplotlib Axes.
        res: champ_r2 내림차순 정렬된 _eval_state dict 리스트.
        states: 주 약자 순서(res 와 동일).
        x: np.arange(len(states)) — 막대 위치.
        seoul: compute_seoul_forward_baseline 산출(라이브) 또는 None(선 생략).
        forward_weeks: 공통 forward 창(주수) — y라벨 명시.
        standalone: True 면 단독 figure 용(제목에 'R²' 명시).
    """
    w = 0.27
    ax.bar(x - w, [r["champ_r2"] for r in res], w, label="FusedEpi champion", color="#2c7fb8")
    ax.bar(x, [r["persist_r2"] for r in res], w, label="persistence(lag1)", color="#bdbdbd")
    ax.bar(x + w, [r["seasonal_r2"] for r in res], w, label="seasonal-naive(lag52)", color="#fdae6b")
    if seoul is not None:
        ax.axhline(seoul["r2"], ls="--", color="#d7301f", lw=2,
                   label=f"Seoul live forward R²={seoul['r2']:.3f} (same protocol, n={seoul['n_test']})")
    ax.set_xticks(x); ax.set_xticklabels(states)
    ax.set_ylabel(f"R² (calendar-locked forward, common {forward_weeks} weeks)")
    ax.set_title("US per-state champion generalization — R²")
    ax.set_ylim(min(-0.2, min(r["seasonal_r2"] for r in res) - 0.1), 1.05)
    ax.axhline(0, color="k", lw=0.6)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)


def _panel_wis(ax, res: list[dict], states: list, x, seoul: dict | None,
               standalone: bool) -> None:
    """WIS 막대(챔피언 vs baseline, 낮을수록 우수) + Seoul 기준선 패널 (공용).

    Args:
        ax: 그릴 matplotlib Axes.
        res: champ_r2 내림차순 정렬된 _eval_state dict 리스트.
        states: 주 약자 순서(res 와 동일).
        x: np.arange(len(states)) — 막대 위치.
        seoul: compute_seoul_forward_baseline 산출(라이브) 또는 None(선 생략).
        standalone: True 면 단독 figure 용(제목에 'WIS' 명시).
    """
    w = 0.27
    ax.bar(x - w, [r["champ_wis"] for r in res], w, label="FusedEpi champion", color="#2c7fb8")
    ax.bar(x, [r["persist_wis"] for r in res], w, label="persistence(lag1)", color="#bdbdbd")
    ax.bar(x + w, [r["seasonal_wis"] for r in res], w, label="seasonal-naive(lag52)", color="#fdae6b")
    if seoul is not None:
        ax.axhline(seoul["wis"], ls="--", color="#d7301f", lw=2,
                   label=f"Seoul live forward WIS={seoul['wis']:.3f} (same protocol)")
    ax.set_xticks(x); ax.set_xticklabels(states)
    ax.set_ylabel("WIS (lower is better)")
    ax.set_title("US per-state champion generalization — WIS")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)


def _plot(results: list[dict], out_png: Path,
          seoul: dict | None, forward_weeks: int) -> list[Path]:
    """주별 챔피언 R²/WIS 막대 + Seoul 기준선 — 단일 패널 PNG 2개 + combined 1개.

    "figures shown 1 at a time" 요구: 각 패널을 독립 PNG 로도 저장(동일 helper 재사용,
    실데이터·placeholder 없음). 이후 기존과 동일한 combined 도 저장(back-compat).

    Args:
        results: _eval_state 산출 dict 리스트(주별).
        out_png: combined 출력 PNG 경로(기존). standalone 경로는 이 stem 에서 파생.
        seoul: compute_seoul_forward_baseline 산출(라이브) 또는 None(선 생략).
        forward_weeks: 공통 forward 창(주수) — 축/제목 명시.

    Returns: 저장된 PNG 경로 list (standalone 2 + combined 1).

    Side effects: out_png + 단일 패널 PNG 2개 작성 (dpi=120).
    """
    plt = _setup_matplotlib()
    res = sorted(results, key=lambda d: d["champ_r2"], reverse=True)
    states = [r["state"] for r in res]
    x = np.arange(len(states))
    n_states = len(states)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_r2 = out_png.with_name("fig_champion_us_states_r2.png")
    out_wis = out_png.with_name("fig_champion_us_states_wis.png")
    written: list[Path] = []

    # ── standalone 패널 PNG (1개씩) — chart 패널 ~(7.5,5.5) ──
    fig_r2, axr = plt.subplots(figsize=(7.5, 5.5))
    _panel_r2(axr, res, states, x, seoul, forward_weeks, standalone=True)
    fig_r2.tight_layout()
    fig_r2.savefig(out_r2, dpi=120)
    plt.close(fig_r2)
    written.append(out_r2)

    fig_wis, axw = plt.subplots(figsize=(7.5, 5.5))
    _panel_wis(axw, res, states, x, seoul, standalone=True)
    fig_wis.tight_layout()
    fig_wis.savefig(out_wis, dpi=120)
    plt.close(fig_wis)
    written.append(out_wis)

    # ── combined figure (기존과 동일, 동일 helper standalone=False) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    _panel_r2(ax1, res, states, x, seoul, forward_weeks, standalone=False)
    _panel_wis(ax2, res, states, x, seoul, standalone=False)

    seoul_txt = (f"dashed = Seoul live forward (same protocol)"
                 if seoul is not None else "Seoul live baseline not computable (line omitted)")
    fig.suptitle(
        f"Champion FusedEpi (TiRex+TabPFN) generalization: US {n_states} states ILI "
        f"(calendar-locked forward, in-sample ≤ 2026-02-09, forward from 2026-02-16, common {forward_weeks} weeks) · {seoul_txt}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    written.append(out_png)
    return written


def _write_csv(results: list[dict], out_csv: Path,
               seoul: dict | None, forward_weeks: int) -> None:
    """주별 결과 CSV 작성 (forward_weeks 공통창 + Seoul 라이브 기준선 행; source 명시).

    Args:
        results: _eval_state 산출 dict 리스트.
        out_csv: 출력 CSV 경로.
        seoul: compute_seoul_forward_baseline 산출(라이브) 또는 None.
        forward_weeks: 공통 forward 창(주수) — CSV 메타.
    Side effects: out_csv 작성.
    """
    import csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["state", "year_lo", "year_hi", "forward_weeks", "n_train", "n_test", "ymax",
            "champ_r2", "champ_wis", "persist_r2", "seasonal_r2",
            "persist_wis", "seasonal_wis", "champ_metric_source"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in sorted(results, key=lambda d: d["champ_r2"], reverse=True):
            w.writerow([r["state"], YEAR_LO, YEAR_HI, forward_weeks, r["n_train"], r["n_test"],
                        f"{r['ymax']:.2f}",
                        f"{r['champ_r2']:.4f}", f"{r['champ_wis']:.4f}",
                        f"{r['persist_r2']:.4f}", f"{r['seasonal_r2']:.4f}",
                        f"{r['persist_wis']:.4f}", f"{r['seasonal_wis']:.4f}",
                        "model-derived (FusedEpi calendar-locked forward 1-step)"])
        # Seoul 라이브 forward 기준선도 1행 기록 (동일 프로토콜, source 명시).
        if seoul is not None:
            w.writerow(["SEOUL_REF", YEAR_LO, YEAR_HI, forward_weeks,
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
    """US 주별 챔피언 일반화 평가 entry point (calendar-locked forward).

    Returns:
        0 = 성공(figure+CSV 작성) 또는 smoke 통과, 1 = 실데이터 없음/smoke 실패(정직).

    Side effects: figures/fig_champion_us_states.{png,csv} 작성. DB read-only.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = Path(__file__).resolve().parents[1] / "results" / "figures"
    out_png = out_dir / "fig_champion_us_states.png"
    out_csv = out_dir / "fig_champion_us_states.csv"

    smoke = os.environ.get("MPH_OVERSEAS_SMOKE", "").strip()
    in_sample_end = get_in_sample_end()

    con = read_only_connect()
    try:
        # ── smoke 모드 (env): 1주만 calendar-locked forward 검증 후 종료 ──
        if smoke:
            state = _resolve_smoke(smoke)
            log.info("=== SMOKE: %s — calendar-locked forward 1주 검증 ===", state)
            loaded = _load_state_series(con, state)
            if loaded is None:
                log.error("SMOKE: %s 데이터 부족/품질불량 — 실패", state)
                return 1
            y, yw = loaded
            n_train = split_forward_by_isoweek(yw, in_sample_end)
            forward_len = common_forward_len([len(y) - n_train], cap=FORWARD_WEEKS_CAP)
            fwd_first = next((isoweek_monday(*yw[i]) for i in range(n_train, len(yw))), None)
            log.info("  in-sample=%d(≤%s) forward 가용=%d 공통창=%d 시작=%s",
                     n_train, in_sample_end.isoformat(), len(y) - n_train, forward_len, fwd_first)
            r = _eval_state(state, y, n_train, forward_len)
            if r is None:
                log.error("SMOKE: %s forward eval 실패", state)
                return 1
            log.info("=== SMOKE OK: %s 라이브 forward R²=%.4f WIS=%.4f (n_fwd=%d, 시작 %s) ===",
                     state, r["champ_r2"], r["champ_wis"], r["n_test"], fwd_first)
            return 0

        # ── 1패스: 전 주 로드 + calendar-locked split (n_train, 가용 forward 길이) ──
        loaded_all: list[tuple[str, np.ndarray, int, int]] = []
        for state in CANDIDATE_STATES:
            loaded = _load_state_series(con, state)
            if loaded is None:
                continue
            y, yw = loaded
            n_train = split_forward_by_isoweek(yw, in_sample_end)
            avail_fwd = len(y) - n_train
            if n_train < 70 or avail_fwd <= 0:
                log.info("  [skip] %s: in-sample=%d forward 가용=%d (부적합)",
                         state, n_train, avail_fwd)
                continue
            loaded_all.append((state, y, n_train, avail_fwd))
    finally:
        con.close()

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
        r = _eval_state(state, y, n_train, forward_weeks)
        if r is not None:
            results.append(r)

    if not results:
        log.error("forward eval 전부 실패 — figure 생성 skip (정직).")
        return 1

    # ── Seoul 라이브 forward 기준선 (동일 프로토콜·동일 공통 창) — 하드코드 대체 ──
    seoul = compute_seoul_forward_baseline(forward_cap=forward_weeks)

    written = _plot(results, out_png, seoul, forward_weeks)
    _write_csv(results, out_csv, seoul, forward_weeks)

    # ── 요약 (모델-유래 강조) ──
    n = len(results)
    mean_r2 = float(np.mean([r["champ_r2"] for r in results]))
    n_beats_persist = sum(r["champ_r2"] > r["persist_r2"] for r in results)
    n_beats_seasonal = sum(r["champ_r2"] > r["seasonal_r2"] for r in results)
    log.info("\n=== 요약 (US 주별, calendar-locked forward, 공통 %d주) ===", forward_weeks)
    log.info("평가 주 수: %d (in-sample≤%s, forward 2026-02-16~)", n, in_sample_end.isoformat())
    if seoul is not None:
        log.info("Seoul 라이브 forward R²=%.4f WIS=%.4f (동일 프로토콜, n_fwd=%d)",
                 seoul["r2"], seoul["wis"], seoul["n_test"])
    else:
        log.info("Seoul 라이브 기준선 계산 불가(캐시 부재 등) — 선 생략")
    log.info("챔피언 평균 R² (US): %.4f", mean_r2)
    log.info("persistence 능가: %d/%d 주 | seasonal-naive 능가: %d/%d 주",
             n_beats_persist, n, n_beats_seasonal, n)
    for p in written:
        log.info("PNG: %s", p)
    log.info("CSV: %s", out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
