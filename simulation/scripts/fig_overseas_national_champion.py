"""fig_overseas_national_champion.py — 챔피언(FusedEpi) national-레벨 다국가 일반화 검증.

Seoul-개념 챔피언 ``FusedEpiForecaster`` (TiRex+TabPFN foundation 융합)를 **national
(국가단위)** ILI 시계열 ~12개국에 그대로 적용하여 **Seoul 밖 일반화**를 실측한다.
기존 ``fig_overseas_regions_champion.py`` (풀 FusedEpi, sub-national 4개국) 와
``fig_champion_us_states.py`` (US 주별) 의 챔피언 fit/predict 방식을 그대로 재사용하되,
대상 테이블을 ``overseas_ili`` (national-레벨) 로 바꾸고 국가 차원으로 확장했다.

설계 (ENGINEERING_PRINCIPLES.md D-1~D-5 / K-1~K-4 준수):
- **★풀 FusedEpi(FusedEpiForecaster, TiRex+TabPFN 융합) 사용** — TiRex-core 아님.
  regions/us-states 스크립트의 ``fit`` / ``predict(X, y_observed=...)`` /
  ``predict_quantiles`` 로직을 1-line 변경 없이 재사용.
- **실데이터만**: ``overseas_ili`` 테이블 (source/country/year/week_no/ili_rate) —
  ``read_only_connect``. 데이터 없으면 정직 skip (합성/가짜 데이터 절대 생성 X, 가짜 0 금지).
- **★국가별 surveillance metric 스케일이 다름 (정직)**: ``overseas_ili.ili_rate`` 는
  source마다 의미가 다르다 — ``delphi_national/US`` = ILI%(~0–8), ``who_flunet/*`` =
  검사 양성률/활동도(positivity-style, 0–100, 국가별 단위 상이). 따라서 단일 가드 부적합
  → regions 스크립트의 **국가별 plausibility band** ``COUNTRY_PLAUSIBLE`` 방식 재사용해
  단위/입력오류만 제외하고 정상 스케일 차이는 보존한다. R²/WIS는 각 국가 **자체 스케일
  내** 평가라 cross-country 비교 OK (수준이 아니라 적합도 비교).
- **★calendar-locked forward eval (2026-06-26 개정)**: 옛 ``TEST_FRAC=0.25`` 비율 split
  대신 **공통 캘린더 경계**(in-sample 종료=2026-02-09, forward=2026-02-16 이후)로 split.
  ISO (year, week_no) → 월요일 날짜 ≤ 02-09 = in-sample, 초과 = forward.
  ⇒ 전 국가 forward 가 **2026-W08(02-16)** 로 정렬. forward = in-sample fit 후 관측 y 흘리며
  (``y_observed``) 1주씩 1-step 예측 → R²/WIS(leak-free).
- **공통 forward 창 (가짜 연장 금지)**: 국가별 forward 길이가 데이터 끝 차이로 다르면
  ``common_forward_len`` 로 공통 최소(또는 ``FORWARD_WEEKS_CAP``)에 truncate → figure/CSV 명시.
- **baseline 대비**: persistence(lag1) · seasonal-naive(작년 같은 주, lag52).
- **★Seoul 기준선 = 라이브 forward (하드코드 제거)**: 옛 0.9357(per_model_eval test-slab,
  forward 아님) 삭제 → ``compute_seoul_forward_baseline`` 가 **동일 프로토콜**(feature_cache
  ili_rate+week_start, in-sample≤02-09, FusedEpi rolling, 공통 창 truncate)로 Seoul forward
  R²/WIS 를 **실시간 계산**해 기준선으로 사용. '관측'이 아니라 **모델-유래** (figure/CSV 'source' 명시).

CONFIG(상단)로 국가목록·eval 연도범위 조절 가능 (detached 실행 시 조정).
기본 = ~12개국, 2019–2026(forward 포함).

성능: 국가당 FusedEpi fit(TiRex rolling 캐시) + rolling 1-step test ~수 분. CPU.
부작용: ``simulation/results/figures/fig_overseas_national_champion.png`` (dpi=120) +
        동 ``.csv`` 작성. DB read-only. 결정성(seed 42, rolling 결정적).

실행 (★전체 = detached로; 본 스크립트 자체는 그냥 entry):
    .venv/bin/python -m simulation.scripts.fig_overseas_national_champion

smoke (1국가만 fit/predict 검증 후 종료, figure/CSV 미작성):
    MPH_OVERSEAS_SMOKE=delphi_national:US \
        .venv/bin/python -m simulation.scripts.fig_overseas_national_champion
    # 또는 source 생략 시 자동 추론: MPH_OVERSEAS_SMOKE=DE
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

# ── CONFIG (detached 실행 시 여기만 조정) ───────────────────────────────────
#   COUNTRIES : 평가 대상 (source, country, 한국어 라벨) — overseas_ili national.
#               delphi_national/US = ILI% ; who_flunet/* = 양성률/활동도(국가별 단위 상이).
#   YEAR_LO/HI: eval 연도범위 (Seoul 챔피언 학습기간과 겹침; national data ~2024+).
#   (source,country) 쌍을 key로 둬 같은 country가 다른 source에 있어도 분리 평가 가능.
COUNTRIES: list[tuple[str, str, str]] = [
    ("delphi_national", "US", "United States (ILI%)"),
    ("who_flunet", "JP", "Japan"),
    ("who_flunet", "DE", "Germany"),
    ("who_flunet", "FR", "France"),
    ("who_flunet", "AU", "Australia"),
    ("who_flunet", "KR", "South Korea"),
    ("who_flunet", "NL", "Netherlands"),
    ("who_flunet", "SE", "Sweden"),
    ("who_flunet", "CN", "China"),
    ("who_flunet", "SG", "Singapore"),
    ("who_flunet", "GB", "United Kingdom"),
    ("who_flunet", "HK", "Hong Kong"),
]
YEAR_LO, YEAR_HI = 2019, 2026     # ★forward(2026-02-16 이후) 포함하도록 2026 까지.

# calendar-locked forward eval 파라미터.
MIN_WEEKS = 120            # FusedEpi min_data=70 + in-sample 여유.
FORWARD_WEEKS_CAP = 18     # 공통 forward 창 상한(주). 데이터 가용 min 과 함께 작은 쪽 사용.
                           # (HWP forward 18주 목표; 데이터 부족 시 공통 min 으로 자동 축소.)

# ── ★국가별 plausibility band (정직: surveillance metric 스케일이 source/국가마다 상이) ──
#   (min_max, max_max): 한 국가 시계열의 MAX 가 이 band 밖이면 단위/입력오류로 보고 제외.
#   값 근거 = overseas_ili 실측 (2019–2024) per-country max 분포:
#     delphi_national/US (ILI%)       : max≈7.4         → 정상 1–15
#     who_flunet/* (양성률/활동도, %)  : max≈30–100      → 정상 5–100 (양성률은 100% 상한)
#   GB는 2019–2024 구간이 109주(<120) + max≈1.0 으로 사실상 결측 → MIN_WEEKS 에서 정직 제외.
COUNTRY_PLAUSIBLE: dict[tuple[str, str], tuple[float, float]] = {
    ("delphi_national", "US"): (1.0, 15.0),
}
# who_flunet 양성률 기본 band (개별 등록 없는 who_flunet 국가에 적용).
WHO_FLUNET_BAND: tuple[float, float] = (5.0, 100.0)

# ── (B) 세계 bubble 지도용 위경도 (lon, lat) — fig_overseas_map.py 와 동일 좌표 ──
#   geopandas 부재 → 위경도 산점 bubble 명시 (지도 윤곽 아님).
COUNTRY_LATLON: dict[str, tuple[float, float]] = {
    "US": (-77.04, 38.91),
    "JP": (139.69, 35.69),
    "DE": (13.40, 52.52),
    "FR": (2.35, 48.86),
    "AU": (149.13, -35.28),
    "KR": (126.98, 37.57),
    "NL": (4.90, 52.37),
    "SE": (18.07, 59.33),
    "CN": (116.40, 39.90),
    "SG": (103.82, 1.35),
    "GB": (-0.13, 51.51),
    "HK": (114.17, 22.32),
}
# 라벨 충돌 회피용 수동 오프셋(경도, 위도) — 밀집 클러스터(서유럽/동아시아) 텍스트 겹침 방지.
LABEL_OFFSET: dict[str, tuple[float, float]] = {
    "GB": (-20.0, 6.0), "FR": (-12.0, -14.0), "NL": (-2.0, 12.0), "DE": (22.0, 4.0),
    "CN": (-2.0, 14.0), "KR": (-12.0, -14.0), "JP": (20.0, 4.0), "HK": (28.0, -8.0),
    "SE": (8.0, 6.0), "US": (0.0, -10.0), "AU": (0.0, 8.0), "SG": (10.0, -6.0),
}

# ── Seoul 기준선 = 라이브 forward (하드코드 제거, 2026-06-26) ──────────────────
#   옛 0.9357(per_model_eval test-slab)은 forward 가 아니라 hold-out 평가 → 비교 부적합.
#   main() 에서 compute_seoul_forward_baseline(공통창) 로 **동일 프로토콜** 실시간 계산해 채운다.
#   계산 실패(캐시 부재 등) 시 기준선 선/점은 생략(가짜 값 박제 X).


def _band_for(source: str, country: str) -> tuple[float, float]:
    """(source, country) 의 plausibility band 조회 (who_flunet 기본 band fallback).

    Args:
        source: overseas_ili.source (예 'delphi_national', 'who_flunet').
        country: overseas_ili.country (예 'US', 'DE').

    Returns:
        (min_max, max_max) — 시계열 MAX 가 이 구간 밖이면 단위/입력오류로 제외.
    """
    if (source, country) in COUNTRY_PLAUSIBLE:
        return COUNTRY_PLAUSIBLE[(source, country)]
    if source == "who_flunet":
        return WHO_FLUNET_BAND
    return (0.0, float("inf"))


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


def _load_national_series(
    con, source: str, country: str,
) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    """한 국가 national ILI 시계열 로드 (year·week_no 정렬, plausibility 검증).

    Args:
        con: read_only sqlite 연결.
        source: overseas_ili.source (예 'delphi_national', 'who_flunet').
        country: overseas_ili.country (예 'US', 'DE').

    Returns:
        (y, yw) — y=(T,) float ndarray(ili_rate, 시간순), yw=[(year, week_no), …] 동일 순서.
        불충분/band-밖 시 None. yw 는 calendar-locked forward split 에 사용.

    Caller responsibility: con은 read_only_connect 산출이어야 함.
    """
    lo, hi = _band_for(source, country)
    cur = con.cursor()
    rows = cur.execute(
        "SELECT year, week_no, ili_rate FROM overseas_ili "
        "WHERE source = ? AND country = ? AND year BETWEEN ? AND ? "
        "AND ili_rate IS NOT NULL "
        "ORDER BY year ASC, week_no ASC",
        (source, country, YEAR_LO, YEAR_HI),
    ).fetchall()
    if not rows:
        log.info("  [skip] %s/%s: 행 없음", source, country)
        return None
    y = np.asarray([r[2] for r in rows], dtype=float)
    yw = [(int(r[0]), int(r[1])) for r in rows]
    if y.size < MIN_WEEKS:
        log.info("  [skip] %s/%s: %d주 < %d (불충분)", source, country, y.size, MIN_WEEKS)
        return None
    ymax = float(np.nanmax(y))
    if not (lo <= ymax <= hi):
        log.info("  [skip] %s/%s: max=%.1f band(%.0f–%.0f) 밖 (단위/입력오류 의심)",
                 source, country, ymax, lo, hi)
        return None
    # 잔여 음수/NaN 위생 (관측 정상값은 보존).
    y = np.nan_to_num(np.clip(y, 0.0, None), nan=0.0)
    return y, yw


def _eval_country(source: str, country: str, label: str,
                  y: np.ndarray, n_train: int, forward_len: int) -> dict | None:
    """한 국가에 대해 calendar-locked forward(FusedEpi rolling 1-step) + baseline 평가.

    Args:
        source: overseas_ili.source.
        country: overseas_ili.country.
        label: 한국어 표시 라벨.
        y: (T,) 관측 ILI 시계열(시간순).
        n_train: in-sample 길이(주 월요일 ≤ in_sample_end 인 주수; calendar-locked split).
        forward_len: 평가 forward 주수(공통 창으로 truncate 된 값).

    Returns:
        {source, country, label, n_train, n_test(=n_forward), champ_r2, champ_wis,
         persist_r2, seasonal_r2, persist_wis, seasonal_wis, ymax,
         y_te(list), y_pred(list)} 또는 fit/부적합 시 None.
        (y_te/y_pred 는 (C) 대표국가 예측 vs 실측 패널용으로만 보존.)

    Performance: TiRex rolling fit(캐시) + forward_len×(1-step). CPU. 국가당 ~수 분.
    Side effects: 없음 (DB write 0).
    """
    if n_train < 70 or forward_len <= 0:
        log.info("  [skip] %s/%s: split 부적합 (n_train=%d forward=%d)",
                 source, country, n_train, forward_len)
        return None

    r = fused_epi_forward(y, n_train, forward_len)
    if r is None:
        log.warning("  [skip] %s/%s: FusedEpi forward 실패", source, country)
        return None

    log.info("  [%s/%s] n_tr=%d n_fwd=%d | champ R2=%.3f WIS=%.3f | persist R2=%.3f | seasonal R2=%.3f",
             source, country, r["n_train"], r["n_forward"],
             r["champ_r2"], r["champ_wis"], r["persist_r2"], r["seasonal_r2"])
    return {
        "source": source, "country": country, "label": label,
        "n_train": r["n_train"], "n_test": r["n_forward"],
        "champ_r2": r["champ_r2"], "champ_wis": r["champ_wis"],
        "persist_r2": r["persist_r2"], "seasonal_r2": r["seasonal_r2"],
        "persist_wis": r["persist_wis"], "seasonal_wis": r["seasonal_wis"],
        "ymax": float(np.nanmax(y)),
        "y_te": r["y_te"], "y_pred": r["y_pred"],
    }



def _panel_country_r2(ax, res: list[dict], norm, cmap, seoul: dict | None,
                      forward_weeks: int, standalone: bool) -> None:
    """패널: 국가별 챔피언 R² 막대 + Seoul 라이브 forward 기준선.

    Args:
        ax: 그릴 Axes.
        res: champ_r2 내림차순 정렬된 _eval_country dict 리스트.
        norm: matplotlib Normalize (R² 색 매핑용).
        cmap: matplotlib colormap.
        seoul: Seoul 라이브 기준선 dict 또는 None.
        forward_weeks: 공통 forward 창(주수).
        standalone: True 면 단독 figure (제목에서 '(A)' 서수 제거).

    Side effects: ax 에 막대/기준선/라벨 그림.
    """
    r2_vals = [r["champ_r2"] for r in res]
    x = np.arange(len(res))
    bar_colors = [cmap(norm(r["champ_r2"])) for r in res]
    ax.bar(x, r2_vals, color=bar_colors, edgecolor="white", linewidth=0.5)
    if seoul is not None:
        ax.axhline(seoul["r2"], ls="--", color="#d7301f", lw=1.8,
                   label=f"Seoul live forward R²={seoul['r2']:.3f} "
                         f"(same protocol, n={seoul['n_test']})")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['label']}\n({r['country']})" for r in res],
                       fontsize=8)
    ax.set_ylabel(f"R² (calendar-locked forward, common {forward_weeks} weeks)")
    lo_y = min([-0.3] + r2_vals) - 0.05
    ax.set_ylim(max(-2.0, lo_y), 1.05)
    for i, r in enumerate(res):                   # 막대 위 값 표기
        ax.text(i, r["champ_r2"] + 0.02 if r["champ_r2"] >= 0 else r["champ_r2"] - 0.07,
                f"{r['champ_r2']:.2f}", ha="center", fontsize=7)
    title = ("National-level champion FusedEpi forward R² by country (calendar-locked) · dashed = Seoul live baseline"
             if standalone
             else "(A) National-level champion FusedEpi forward R² by country (calendar-locked) · dashed = Seoul live baseline")
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(axis="y", alpha=0.3)


def _panel_world_bubble(ax, res: list[dict], norm, cmap, seoul: dict | None,
                        fig, standalone: bool) -> None:
    """패널: 세계 bubble 지도 (위경도 산점; 색/크기∝R², ★=Seoul).

    Args:
        ax: 그릴 Axes.
        res: champ_r2 내림차순 정렬된 _eval_country dict 리스트.
        norm: matplotlib Normalize.
        cmap: matplotlib colormap.
        seoul: Seoul 라이브 기준선 dict 또는 None.
        fig: colorbar attach 용 Figure.
        standalone: True 면 단독 figure (제목에서 '(B)' 서수 제거).

    Side effects: ax 에 bubble/grid/colorbar 그림.
    """
    from matplotlib import cm
    for lon0 in range(-150, 181, 30):
        ax.axvline(lon0, color="#E3E7EC", lw=0.5, zorder=0)
    for lat0 in range(-40, 81, 20):
        ax.axhline(lat0, color="#E3E7EC", lw=0.5, zorder=0)
    for r in res:
        c = r["country"]
        if c not in COUNTRY_LATLON:
            continue
        lon, lat = COUNTRY_LATLON[c]
        size = 120 + 1300 * max(0.0, r["champ_r2"])   # bubble 크기 ∝ R²(≥0)
        ax.scatter(lon, lat, s=size, color=cmap(norm(r["champ_r2"])),
                   alpha=0.85, edgecolor="k", linewidth=0.8, zorder=3)
        ox, oy = LABEL_OFFSET.get(c, (0.0, 6.0))
        ax.annotate(f"{c}\n{r['champ_r2']:.2f}", (lon, lat), xytext=(lon + ox, lat + oy),
                    ha="center", va="center", fontsize=7, zorder=4,
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.5)
                    if (ox or oy) else None)
    # Seoul(서울 sentinel) 라이브 forward 기준선도 한 점으로 표기 (KR national 과 구분: '★Seoul').
    if seoul is not None and "KR" in COUNTRY_LATLON:
        slon, slat = COUNTRY_LATLON["KR"]
        ax.scatter(slon, slat - 0.0, marker="*", s=260,
                   color=cmap(norm(seoul["r2"])), edgecolor="k",
                   linewidth=0.9, zorder=5)
    ax.set_xlim(-160, 180)
    ax.set_ylim(-50, 80)
    ax.set_xlabel("Longitude (lon)")
    ax.set_ylabel("Latitude (lat)")
    title = ("World bubble map (no geopandas → lon/lat bubbles; color/size ∝ R², ★ = Seoul)"
             if standalone
             else "(B) World bubble map (no geopandas → lon/lat bubbles; color/size ∝ R², ★ = Seoul)")
    ax.set_title(title, fontsize=10)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02, label="R²")


def _panel_rep_forecast(ax, res: list[dict], standalone: bool) -> None:
    """패널: 대표 국가(최고 R²) forward 예측 vs 실측.

    Args:
        ax: 그릴 Axes.
        res: champ_r2 내림차순 정렬된 _eval_country dict 리스트(res[0]=대표국가).
        standalone: True 면 단독 figure (제목에서 '(C)' 서수 제거).

    Side effects: ax 에 관측/예측 선 그림.
    """
    rep = res[0]
    yt = np.asarray(rep["y_te"], float)
    yp = np.asarray(rep["y_pred"], float)
    tt = np.arange(len(yt))
    ax.plot(tt, yt, color="#222222", lw=1.6, marker="o", ms=3, label="Observed (ground-truth ILI)")
    ax.plot(tt, yp, color="#2c7fb8", lw=1.6, ls="--", marker="s", ms=3,
            label="FusedEpi forecast (model-derived)")
    ax.set_xlabel("Forward week (from 2026-02-16, 1-step)")
    ax.set_ylabel("ILI (country's own scale)")
    prefix = "" if standalone else "(C) "
    ax.set_title(f"{prefix}Representative country {rep['label']} ({rep['country']}) forward forecast vs observed "
                 f"(R²={rep['champ_r2']:.3f})", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)


def _plot(results: list[dict], out_png: Path,
          seoul: dict | None, forward_weeks: int) -> list[Path]:
    """national 챔피언 검증 figure — (A) 국가별 R² 막대 + Seoul 라이브 기준선,
    (B) 세계 bubble 지도(위경도 산점; 색/크기∝R²), (C) 대표국가 예측 vs 실측.

    각 패널을 ① 단독 PNG(panel별) 로 먼저 저장한 뒤 ② 동일 헬퍼로 결합 gridspec figure
    (out_png, 기존과 동일) 를 저장한다 (사용자 요구 "한 번에 한 그림씩").

    Args:
        results: _eval_country 산출 dict 리스트.
        out_png: 결합 출력 PNG 경로.
        seoul: compute_seoul_forward_baseline 산출(라이브) 또는 None(계산 실패 시 선 생략).
        forward_weeks: 공통 forward 창(주수) — figure 제목/축 명시용.

    Returns:
        작성된 PNG 경로 리스트 (단독 3개 + 결합 1개 순서).

    Side effects: 단독 PNG 3개 + 결합 PNG(out_png) 작성 (dpi=120).
    """
    plt = _setup_matplotlib()
    from matplotlib import cm
    from matplotlib.colors import Normalize

    res = sorted(results, key=lambda d: d["champ_r2"], reverse=True)

    # 색 매핑: R² (음수 가능) → coolwarm (R² 클수록 파랑쪽).
    r2_vals = [r["champ_r2"] for r in res]
    norm = Normalize(vmin=min(-0.5, min(r2_vals)), vmax=1.0)
    cmap = cm.get_cmap("coolwarm_r")

    out_dir = out_png.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # ── ① 단독 PNG (panel별) ──
    p_r2 = out_dir / "fig_overseas_national_champion_country_r2.png"
    fig1 = plt.figure(figsize=(13, 5))
    _panel_country_r2(fig1.add_subplot(111), res, norm, cmap, seoul, forward_weeks, standalone=True)
    fig1.tight_layout()
    fig1.savefig(p_r2, dpi=120, bbox_inches="tight")
    plt.close(fig1)
    written.append(p_r2)

    p_bubble = out_dir / "fig_overseas_national_champion_world_bubble.png"
    fig2 = plt.figure(figsize=(7, 5.5))
    _panel_world_bubble(fig2.add_subplot(111), res, norm, cmap, seoul, fig2, standalone=True)
    fig2.tight_layout()
    fig2.savefig(p_bubble, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    written.append(p_bubble)

    p_rep = out_dir / "fig_overseas_national_champion_rep_forecast.png"
    fig3 = plt.figure(figsize=(7, 5.5))
    _panel_rep_forecast(fig3.add_subplot(111), res, standalone=True)
    fig3.tight_layout()
    fig3.savefig(p_rep, dpi=120, bbox_inches="tight")
    plt.close(fig3)
    written.append(p_rep)

    # ── ② 결합 gridspec figure (out_png, 기존과 동일·back-compat) ──
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], hspace=0.40, wspace=0.20)
    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[1, 1])
    _panel_country_r2(axA, res, norm, cmap, seoul, forward_weeks, standalone=False)
    _panel_world_bubble(axB, res, norm, cmap, seoul, fig, standalone=False)
    _panel_rep_forecast(axC, res, standalone=False)

    cnames = ", ".join(r["country"] for r in res)
    seoul_txt = (f"dashed/★ = Seoul live forward R²={seoul['r2']:.3f} (same protocol)"
                 if seoul is not None else "Seoul live baseline not computable (line omitted)")
    fig.suptitle(
        f"Champion FusedEpi (TiRex+TabPFN) national-level multi-country generalization: "
        f"{len(res)} countries, ILI calendar-locked forward (in-sample ≤ 2026-02-09, "
        f"forward from 2026-02-16, common {forward_weeks} weeks)\n"
        f"Note: metric scale differs by source (delphi = ILI% / who_flunet = positivity/activity, unit within-series) — "
        f"unit errors excluded, R² evaluated on each country's own scale · {seoul_txt} · countries: {cnames}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    written.append(out_png)
    return written


def _write_csv(results: list[dict], out_csv: Path,
               seoul: dict | None, forward_weeks: int) -> None:
    """국가별 결과 CSV 작성 (forward_weeks 공통창 + Seoul 라이브 기준선 행 명시).

    Args:
        results: _eval_country 산출 dict 리스트.
        out_csv: 출력 CSV 경로.
        seoul: compute_seoul_forward_baseline 산출(라이브) 또는 None.
        forward_weeks: 공통 forward 창(주수) — CSV 메타로 기록.

    Side effects: out_csv 작성.
    """
    import csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["source", "country", "label", "year_lo", "year_hi", "forward_weeks",
            "n_train", "n_test", "ymax",
            "champ_r2", "champ_wis", "persist_r2", "seasonal_r2",
            "persist_wis", "seasonal_wis", "champ_metric_source"]
    ordered = sorted(results, key=lambda d: d["champ_r2"], reverse=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in ordered:
            w.writerow([r["source"], r["country"], r["label"], YEAR_LO, YEAR_HI, forward_weeks,
                        r["n_train"], r["n_test"], f"{r['ymax']:.2f}",
                        f"{r['champ_r2']:.4f}", f"{r['champ_wis']:.4f}",
                        f"{r['persist_r2']:.4f}", f"{r['seasonal_r2']:.4f}",
                        f"{r['persist_wis']:.4f}", f"{r['seasonal_wis']:.4f}",
                        "model-derived (FusedEpi calendar-locked forward 1-step; unit within-series)"])
        # Seoul 라이브 forward 기준선도 1행 기록 (동일 프로토콜, source 명시).
        if seoul is not None:
            w.writerow(["seoul_live_forward", "SEOUL_REF", "Seoul(서울 sentinel)",
                        YEAR_LO, YEAR_HI, forward_weeks,
                        seoul.get("n_train", ""), seoul["n_test"], "",
                        f"{seoul['r2']:.4f}", f"{seoul['wis']:.4f}",
                        "", "", "", "", seoul["source"]])


def _resolve_smoke(spec: str) -> tuple[str, str] | None:
    """smoke env 문자열을 (source, country) 로 해석.

    형식:
      "source:country" (예 'who_flunet:DE') — 명시.
      "country"        (예 'DE')           — CONFIG COUNTRIES 에서 source 자동 추론.

    Args:
        spec: MPH_OVERSEAS_SMOKE 값.

    Returns:
        (source, country) 또는 해석 실패 시 None.
    """
    spec = spec.strip()
    if ":" in spec:
        src, _, ctry = spec.partition(":")
        return src.strip(), ctry.strip().upper()
    ctry = spec.upper()
    for src, c, _label in COUNTRIES:
        if c == ctry:
            return src, c
    return None


def main() -> int:
    """national-레벨 챔피언 다국가 일반화 평가 entry point.

    Returns:
        0 = 성공(figure+CSV 작성) 또는 smoke 통과, 1 = 실데이터 없음/smoke 실패(정직).

    Side effects: figures/fig_overseas_national_champion.{png,csv} 작성. DB read-only.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = Path(__file__).resolve().parents[1] / "results" / "figures"
    out_png = out_dir / "fig_overseas_national_champion.png"
    out_csv = out_dir / "fig_overseas_national_champion.csv"

    label_of = {(s, c): lab for s, c, lab in COUNTRIES}

    # ── smoke 모드 (env): 1국가만 fit/predict 검증 후 종료 (figure/CSV 미작성) ──
    smoke = os.environ.get("MPH_OVERSEAS_SMOKE", "").strip()

    in_sample_end = get_in_sample_end()

    con = read_only_connect()
    try:
        if smoke:
            resolved = _resolve_smoke(smoke)
            if resolved is None:
                log.error("SMOKE: '%s' 해석 실패 (형식 source:country 또는 country)", smoke)
                return 1
            source, country = resolved
            label = label_of.get((source, country), country)
            log.info("=== SMOKE: %s/%s (%s) — calendar-locked forward 1국가 검증 ===",
                     source, country, label)
            loaded = _load_national_series(con, source, country)
            if loaded is None:
                log.error("SMOKE: %s/%s 데이터 부족/band-밖 — 실패", source, country)
                return 1
            y, yw = loaded
            n_train = split_forward_by_isoweek(yw, in_sample_end)
            forward_len = common_forward_len([len(y) - n_train], cap=FORWARD_WEEKS_CAP)
            fwd_first = next((isoweek_monday(*yw[i]) for i in range(n_train, len(yw))), None)
            log.info("  in-sample=%d(≤%s) forward 가용=%d 공통창=%d 시작=%s",
                     n_train, in_sample_end.isoformat(), len(y) - n_train, forward_len, fwd_first)
            r = _eval_country(source, country, label, y, n_train, forward_len)
            if r is None:
                log.error("SMOKE: %s/%s forward eval 실패", source, country)
                return 1
            log.info("=== SMOKE OK: %s/%s forward R²=%.4f WIS=%.4f (n_fwd=%d, 시작 %s) ===",
                     source, country, r["champ_r2"], r["champ_wis"], r["n_test"], fwd_first)
            return 0

        # ── 1패스: 전 국가 로드 + calendar-locked split (n_train, 가용 forward 길이) ──
        loaded_all: list[tuple[str, str, str, np.ndarray, int, int]] = []
        for source, country, label in COUNTRIES:
            log.info("[load] %s/%s (%s) …", source, country, label)
            loaded = _load_national_series(con, source, country)
            if loaded is None:
                continue
            y, yw = loaded
            n_train = split_forward_by_isoweek(yw, in_sample_end)
            avail_fwd = len(y) - n_train
            if n_train < 70 or avail_fwd <= 0:
                log.info("  [skip] %s/%s: in-sample=%d forward 가용=%d (부적합)",
                         source, country, n_train, avail_fwd)
                continue
            loaded_all.append((source, country, label, y, n_train, avail_fwd))

        if not loaded_all:
            log.error("실데이터 없음(forward 가용 0) — figure 생성 skip (정직).")
            return 1

        # ── 공통 forward 창 (가짜 연장 금지): 가용 min 과 CAP 중 작은 쪽 ──
        forward_weeks = common_forward_len([t[5] for t in loaded_all], cap=FORWARD_WEEKS_CAP)
        log.info("[공통 forward 창] 평가 %d개국 가용 forward = %s → 공통 %d주 truncate",
                 len(loaded_all), sorted(t[5] for t in loaded_all), forward_weeks)

        # ── 2패스: 공통 창으로 forward eval ──
        results: list[dict] = []
        for source, country, label, y, n_train, _avail in loaded_all:
            log.info("[run] %s/%s (%s) forward …", source, country, label)
            r = _eval_country(source, country, label, y, n_train, forward_weeks)
            if r is not None:
                results.append(r)
    finally:
        con.close()

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
    med_r2 = float(np.median([r["champ_r2"] for r in results]))
    n_beats_persist = sum(r["champ_r2"] > r["persist_r2"] for r in results)
    n_beats_seasonal = sum(r["champ_r2"] > r["seasonal_r2"] for r in results)
    log.info("\n=== 요약 (national, calendar-locked forward, 공통 %d주) ===", forward_weeks)
    log.info("평가 국가 수: %d (in-sample≤%s, forward 2026-02-16~)", n, in_sample_end.isoformat())
    if seoul is not None:
        log.info("Seoul 라이브 forward R²=%.4f WIS=%.4f (동일 프로토콜, n_fwd=%d)",
                 seoul["r2"], seoul["wis"], seoul["n_test"])
    else:
        log.info("Seoul 라이브 기준선 계산 불가(캐시 부재 등) — 선 생략")
    log.info("챔피언 평균 R²=%.4f | median R²=%.4f", mean_r2, med_r2)
    log.info("persistence 능가: %d/%d 국 | seasonal-naive 능가: %d/%d 국",
             n_beats_persist, n, n_beats_seasonal, n)
    log.info("PNG (결합 + 단독 패널 %d개):", len(written))
    for p in written:
        log.info("  - %s", p)
    log.info("CSV: %s", out_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
