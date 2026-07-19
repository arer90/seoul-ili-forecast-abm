"""fig_overseas_regions_champion.py — 챔피언(FusedEpi) 다국가 sub-national 일반화 검증.

Seoul-개념 챔피언 ``FusedEpiForecaster`` (TiRex+TabPFN foundation 융합)를 **4개국**
(USA·JPN·FRA·DEU) 대표 sub-national 지역의 ILI 시계열에 그대로 적용하여 **Seoul 밖
일반화**를 실측한다. 기존 ``fig_champion_us_states.py`` (11 US주, 풀 FusedEpi)를
참조해 같은 챔피언 fit/predict 방식을 4개국으로 확장했다.

설계 (ENGINEERING_PRINCIPLES.md D-1~D-5 / K-1~K-4 준수):
- **풀 FusedEpi(FusedEpiForecaster, TiRex+TabPFN 융합) 사용** — TiRex-core 아님.
  ``fig_champion_us_states.py`` 의 fit/predict/predict_quantiles 로직을 그대로 재사용.
- **실데이터만**: ``overseas_ili_regional`` (country in USA/JPN/FRA/DEU) — read_only_connect.
  데이터 없으면 정직 skip (합성/가짜 데이터 절대 생성 X, 가짜 0 금지).
- **★국가별 surveillance metric 스케일이 다름 (정직)**: 4개국은 동일 'ILI'라는 이름이지만
  실제 단위가 다르다 — USA(nwss_flu) = ILI%(~0–40), JPN(jihs_prefecture) = 정점당
  보고건수(~37–105), FRA(sentiweb) = 10만명당 발생률(~446–976), DEU(rki_bundesland)
  = 진료지수(~19–192). 따라서 단일 50% 가드는 부적합 → **국가별 plausibility band**
  ``COUNTRY_PLAUSIBLE`` 로 단위오류(USA OK=818·OH=133 등 입력오류)만 제외하고
  정상 스케일 차이는 보존한다. R²/WIS는 각 지역 자체 스케일 내 평가라 cross-country 비교 OK.
- **★calendar-locked forward eval (2026-06-26 개정)**: 옛 ``TEST_FRAC=0.25`` 비율 split
  대신 **공통 캘린더 경계**(in-sample 종료=2026-02-09, forward=2026-02-16 이후)로 split.
  ISO (year, week_no) → 월요일 날짜 ≤ 02-09 = in-sample, 초과 = forward.
  ⇒ 전 지역 forward 가 **2026-W08(02-16)** 로 정렬. forward = in-sample fit 후 관측 y 흘리며
  (``y_observed``) 1주씩 1-step 예측 → R²/WIS(leak-free).
- **공통 forward 창 (가짜 연장 금지)**: 지역별 forward 길이가 데이터 끝 차이로 다르면
  ``common_forward_len`` 로 공통 최소(또는 ``FORWARD_WEEKS_CAP``)에 truncate → figure/CSV 명시.
- **baseline 대비**: persistence(lag1) · seasonal-naive(작년 같은 주, lag52).
- **★Seoul 기준선 = 라이브 forward (하드코드 제거)**: 옛 0.9357(per_model_eval test-slab,
  forward 아님) 삭제 → ``compute_seoul_forward_baseline`` 가 **동일 프로토콜**(feature_cache
  ili_rate+week_start, in-sample≤02-09, FusedEpi rolling, 공통 창 truncate)로 Seoul forward
  R²/WIS 를 **실시간 계산**해 기준선. '관측'이 아니라 **모델-유래** (figure/CSV 'source' 명시).

CONFIG(상단)로 국가·국가당 지역수·eval 연도범위 조절 가능 (detached 실행 시 조정).
기본 = 국가당 8지역, 2019–2026(forward 포함).

성능: 지역당 FusedEpi fit(TiRex rolling 캐시) + rolling 1-step test ~수 분. CPU.
부작용: ``simulation/results/figures/fig_overseas_regions_champion.png`` (dpi=120) +
        동 ``.csv`` 작성. DB read-only. 결정성(seed 42, rolling 결정적).

실행 (★전체 = detached):
    .venv/bin/python -m simulation.scripts.fig_overseas_regions_champion

smoke (1지역만 검증, env로 제어):
    MPH_OVERSEAS_SMOKE=USA:CA .venv/bin/python -m simulation.scripts.fig_overseas_regions_champion
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
#   COUNTRIES        : 평가 국가 (DB country 코드).
#   REGIONS_PER_COUNTRY: 국가당 선정 지역 수 (n_weeks 많고 plausibility 통과 우선).
#   YEAR_LO/HI       : eval 연도범위 (Seoul 챔피언 학습기간과 겹침; 지역데이터 ~2024+).
COUNTRIES = ("USA", "JPN", "FRA", "DEU")
REGIONS_PER_COUNTRY = 8
YEAR_LO, YEAR_HI = 2019, 2026     # ★forward(2026-02-16 이후) 포함하도록 2026 까지.

# calendar-locked forward eval 파라미터.
MIN_WEEKS = 120            # FusedEpi min_data=70 + in-sample 여유.
FORWARD_WEEKS_CAP = 18     # 공통 forward 창 상한(주). 데이터 가용 min 과 함께 작은 쪽 사용.

# ── ★국가별 plausibility band (정직: surveillance metric 스케일이 국가마다 다름) ──
#   (min_max, max_max): 한 지역의 시계열 MAX 가 이 band 밖이면 단위/입력오류로 보고 제외.
#   값 근거 = overseas_ili_regional 실측 per-region max 분포(2019–2024):
#     USA nwss(ILI%)         per-region max p25=10.6 med=16.6 p75=34.2  → 정상≤50, OK=818·OH=133 등 입력오류 제외
#     JPN jihs(정점당 보고건수) per-region max p25=57.2 med=65.7 p75=78.1  → 정상 30–150
#     FRA sentiweb(10만명당)   per-region max p25=528  med=644  p75=766   → 정상 200–1200
#     DEU rki(진료지수)        per-region max p25=45.2 med=58.8 p75=127   → 정상 10–250
COUNTRY_PLAUSIBLE: dict[str, tuple[float, float]] = {
    "USA": (2.0, 50.0),
    "JPN": (20.0, 150.0),
    "FRA": (100.0, 1200.0),
    "DEU": (5.0, 250.0),
}

# ── 국가별 표시 색 (figure 국가 색구분) ─────────────────────────────────────
COUNTRY_COLOR: dict[str, str] = {
    "USA": "#2c7fb8",   # blue
    "JPN": "#d7301f",   # red
    "FRA": "#238b45",   # green
    "DEU": "#feb24c",   # amber
}

# ── (C) bubble 지도용 대략 위경도 (geopandas 부재 → bubble 명시) ─────────────
#   국가 수도/중심 근방 — 국가 단위 bubble 1개씩(지역별 좌표 미보유 → 국가 집계 bubble).
COUNTRY_LATLON: dict[str, tuple[float, float]] = {
    "USA": (39.8, -98.6),
    "JPN": (36.2, 138.3),
    "FRA": (46.6, 2.5),
    "DEU": (51.2, 10.4),
}

# ── Seoul 기준선 = 라이브 forward (하드코드 제거, 2026-06-26) ──────────────────
#   옛 0.9357(per_model_eval test-slab)은 forward 가 아니라 hold-out 평가 → 비교 부적합.
#   main() 에서 compute_seoul_forward_baseline(공통창) 로 **동일 프로토콜** 실시간 계산해 채운다.
#   계산 실패(캐시 부재 등) 시 기준선 선/점은 생략(가짜 값 박제 X).


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


def _select_regions(con, country: str, k: int) -> list[str]:
    """국가별 대표 지역 k개 선정 (고유 ISO주 많고 plausibility band 통과).

    Args:
        con: read_only sqlite 연결.
        country: 국가 코드 (USA/JPN/FRA/DEU).
        k: 선정할 지역 수.

    Returns:
        지역명 리스트 (고유 주차수 내림차순; plausibility 통과만). band 밖(단위오류)·
        불충분 지역은 제외하고 진행 로그에 사유 출력.
        ※ USA 등은 (year,week) 당 복수 sub-source 행이 있어 COUNT(*) 가 부풀려짐 →
          DISTINCT (year,week) 로 고유 주차수 산정(중복 제거 후 길이 기준).

    Caller responsibility: con은 read_only_connect 산출이어야 함.
    """
    lo, hi = COUNTRY_PLAUSIBLE.get(country, (0.0, float("inf")))
    cur = con.cursor()
    rows = cur.execute(
        "SELECT region, COUNT(DISTINCT year || '-' || week_no) AS n, MAX(ili_rate) AS mx "
        "FROM overseas_ili_regional "
        "WHERE country = ? AND year BETWEEN ? AND ? AND ili_rate IS NOT NULL "
        "GROUP BY region ORDER BY n DESC",
        (country, YEAR_LO, YEAR_HI),
    ).fetchall()

    picked: list[str] = []
    for region, n, mx in rows:
        if len(picked) >= k:
            break
        if n < MIN_WEEKS:
            continue                                   # 불충분 (조용히 skip)
        if mx is None or not (lo <= float(mx) <= hi):  # 단위/입력오류 band 밖
            log.info("  [%s] [excl] %s: max=%.1f band(%.0f–%.0f) 밖 (단위오류 의심)",
                     country, region, float(mx or 0.0), lo, hi)
            continue
        picked.append(region)
    log.info("[%s] 선정 %d/%d 지역: %s", country, len(picked), k, picked)
    return picked


def _load_region_series(
    con, country: str, region: str,
) -> tuple[np.ndarray, list[tuple[int, int]]] | None:
    """한 지역 ILI 시계열 로드 (ISO주 단위 중복제거·정렬, plausibility 재검증).

    Args:
        con: read_only sqlite 연결.
        country: 국가 코드.
        region: 지역명.

    Returns:
        (y, yw) — y=(T,) float ndarray(주별 ili_rate, ISO 시간순), yw=[(year, week_no), …].
        불충분/band-밖 시 None.
        ※ (year,week) 당 복수 sub-source 행(USA 등)은 **평균**으로 1주=1값 통합 →
          calendar-locked forward split(contiguous tail) 가 성립. ISO 월요일 날짜로 정렬.

    Caller responsibility: con은 read_only_connect 산출이어야 함.
    """
    lo, hi = COUNTRY_PLAUSIBLE.get(country, (0.0, float("inf")))
    cur = con.cursor()
    rows = cur.execute(
        "SELECT year, week_no, ili_rate FROM overseas_ili_regional "
        "WHERE country = ? AND region = ? AND year BETWEEN ? AND ? "
        "AND ili_rate IS NOT NULL "
        "ORDER BY year ASC, week_no ASC",
        (country, region, YEAR_LO, YEAR_HI),
    ).fetchall()
    if not rows:
        return None
    # (year,week) 당 복수 행 평균으로 통합 (1주=1값) → ISO 월요일 날짜 기준 정렬.
    from collections import defaultdict
    agg: dict[tuple[int, int], list[float]] = defaultdict(list)
    for yy, ww, vv in rows:
        agg[(int(yy), int(ww))].append(float(vv))
    import datetime as _dt
    yw = sorted(agg.keys(),
                key=lambda k: (isoweek_monday(*k) or _dt.date.min, k))
    y = np.asarray([float(np.mean(agg[k])) for k in yw], dtype=float)
    if y.size < MIN_WEEKS:
        log.info("  [%s] [skip] %s: %d고유주 < %d (불충분)", country, region, y.size, MIN_WEEKS)
        return None
    ymax = float(np.nanmax(y))
    if not (lo <= ymax <= hi):
        log.info("  [%s] [skip] %s: max=%.1f band(%.0f–%.0f) 밖 (단위오류 의심)",
                 country, region, ymax, lo, hi)
        return None
    # 잔여 음수/NaN 위생 (관측 정상값은 보존).
    y = np.nan_to_num(np.clip(y, 0.0, None), nan=0.0)
    return y, yw


def _eval_region(country: str, region: str, y: np.ndarray,
                 n_train: int, forward_len: int) -> dict | None:
    """한 지역에 대해 calendar-locked forward(FusedEpi rolling 1-step) + baseline 평가.

    Args:
        country: 국가 코드.
        region: 지역명.
        y: (T,) 관측 ILI 시계열(ISO 시간순, 1주=1값).
        n_train: in-sample 길이(주 월요일 ≤ in_sample_end; calendar-locked split).
        forward_len: 평가 forward 주수(공통 창 truncate).

    Returns:
        {country, region, n_train, n_test(=n_forward), champ_r2, champ_wis, persist_r2,
         seasonal_r2, persist_wis, seasonal_wis, ymax} 또는 fit/부적합 시 None.

    Performance: TiRex rolling fit(캐시) + forward_len×(1-step). CPU. 지역당 ~수 분.
    Side effects: 없음 (DB write 0).
    """
    if n_train < 70 or forward_len <= 0:
        log.info("  [%s] [skip] %s: split 부적합 (n_train=%d forward=%d)",
                 country, region, n_train, forward_len)
        return None

    r = fused_epi_forward(y, n_train, forward_len)
    if r is None:
        log.warning("  [%s] [skip] %s: FusedEpi forward 실패", country, region)
        return None

    log.info("  [%s] [%s] n_tr=%d n_fwd=%d | champ R2=%.3f WIS=%.3f | persist R2=%.3f | seasonal R2=%.3f",
             country, region, r["n_train"], r["n_forward"],
             r["champ_r2"], r["champ_wis"], r["persist_r2"], r["seasonal_r2"])
    return {
        "country": country, "region": region,
        "n_train": r["n_train"], "n_test": r["n_forward"],
        "champ_r2": r["champ_r2"], "champ_wis": r["champ_wis"],
        "persist_r2": r["persist_r2"], "seasonal_r2": r["seasonal_r2"],
        "persist_wis": r["persist_wis"], "seasonal_wis": r["seasonal_wis"],
        "ymax": float(np.nanmax(y)),
    }



def _panel_region_r2(ax, plt, ordered: list[dict], by_country: dict[str, list[dict]],
                     seoul: dict | None, forward_weeks: int, standalone: bool) -> None:
    """패널: 4개국 sub-national 지역별 챔피언 R² 막대(국가 색) + Seoul 라이브 기준선.

    Args:
        ax: 그릴 Axes.
        plt: matplotlib.pyplot (legend handle Rectangle 용).
        ordered: COUNTRIES 순·국가 내 R² 내림차순 정렬된 dict 리스트.
        by_country: 국가→지역 dict 리스트 매핑.
        seoul: Seoul 라이브 기준선 dict 또는 None.
        forward_weeks: 공통 forward 창(주수).
        standalone: True 면 단독 figure (제목에서 '(A)' 서수 제거).

    Side effects: ax 에 막대/기준선/이중범례 그림.
    """
    x = np.arange(len(ordered))
    colors = [COUNTRY_COLOR.get(r["country"], "#888888") for r in ordered]
    ax.bar(x, [r["champ_r2"] for r in ordered], color=colors, edgecolor="white", linewidth=0.4)
    if seoul is not None:
        ax.axhline(seoul["r2"], ls="--", color="#000000", lw=1.6,
                   label=f"Seoul live forward R²={seoul['r2']:.3f} "
                         f"(same protocol, n={seoul['n_test']})")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['country']}·{r['region']}" for r in ordered],
                       rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(f"R² (calendar-locked forward, common {forward_weeks} weeks)")
    lo_y = min([-0.2] + [r["champ_r2"] for r in ordered]) - 0.05
    ax.set_ylim(max(-2.0, lo_y), 1.05)
    title = ("4-country sub-national champion FusedEpi forward R² by region (bar color = country) · dashed = Seoul live baseline"
             if standalone
             else "(A) 4-country sub-national champion FusedEpi forward R² by region (bar color = country) · dashed = Seoul live baseline")
    ax.set_title(title, fontsize=11)
    # 국가 색 범례 + Seoul 점선.
    handles = [plt.Rectangle((0, 0), 1, 1, color=COUNTRY_COLOR[c]) for c in COUNTRIES
               if by_country.get(c)]
    labels = [f"{c} (n={len(by_country.get(c, []))})" for c in COUNTRIES if by_country.get(c)]
    leg1 = ax.legend(handles, labels, fontsize=8, loc="lower left", title="Country")
    ax.add_artist(leg1)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)


def _panel_country_median(ax, cnames: list[str], by_country: dict[str, list[dict]],
                          seoul: dict | None, standalone: bool) -> None:
    """패널: 국가별 median R² 요약 막대 + Seoul 기준선.

    Args:
        ax: 그릴 Axes.
        cnames: 데이터 있는 국가 코드 리스트 (CONFIG 순서).
        by_country: 국가→지역 dict 리스트 매핑.
        seoul: Seoul 라이브 기준선 dict 또는 None.
        standalone: True 면 단독 figure (제목에서 '(B)' 서수 제거).

    Side effects: ax 에 막대/값표기/기준선 그림.
    """
    med_r2 = [float(np.median([r["champ_r2"] for r in by_country[c]])) for c in cnames]
    xb = np.arange(len(cnames))
    ax.bar(xb, med_r2, color=[COUNTRY_COLOR[c] for c in cnames], edgecolor="white")
    for i, v in enumerate(med_r2):
        ax.text(i, v + 0.02 if v >= 0 else v - 0.06, f"{v:.3f}",
                ha="center", fontsize=9)
    if seoul is not None:
        ax.axhline(seoul["r2"], ls="--", color="#000000", lw=1.4,
                   label=f"Seoul live R²={seoul['r2']:.3f}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(xb)
    ax.set_xticklabels([f"{c}\n(n={len(by_country[c])})" for c in cnames])
    ax.set_ylabel("median R² (by country)")
    ax.set_ylim(max(-1.0, min([-0.2] + med_r2) - 0.1), 1.05)
    ax.set_title("Median R² by country" if standalone else "(B) Median R² by country", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)


def _panel_country_bubble(ax, cnames: list[str], by_country: dict[str, list[dict]],
                          standalone: bool) -> None:
    """패널: 국가 bubble 지도 (위경도 산점; 크기∝median R²; 국가집계).

    Args:
        ax: 그릴 Axes.
        cnames: 데이터 있는 국가 코드 리스트.
        by_country: 국가→지역 dict 리스트 매핑.
        standalone: True 면 단독 figure (제목에서 '(C)' 서수 제거).

    Side effects: ax 에 bubble/주석 그림.
    """
    for c in cnames:
        lat, lon = COUNTRY_LATLON[c]
        med = float(np.median([r["champ_r2"] for r in by_country[c]]))
        size = 300 + 1400 * max(0.0, med)        # bubble 크기 ∝ median R²(≥0)
        ax.scatter(lon, lat, s=size, color=COUNTRY_COLOR[c],
                   alpha=0.65, edgecolor="k", linewidth=0.8, zorder=3)
        ax.annotate(f"{c}\nR²={med:.2f}\nn={len(by_country[c])}", (lon, lat),
                    ha="center", va="center", fontsize=8, zorder=4)
    ax.set_xlim(-130, 160)
    ax.set_ylim(20, 60)
    ax.set_xlabel("Longitude (lon)")
    ax.set_ylabel("Latitude (lat)")
    title = ("Country bubble map (no geopandas → lon/lat bubbles; size ∝ median R²)"
             if standalone
             else "(C) Country bubble map (no geopandas → lon/lat bubbles; size ∝ median R²)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.25)


def _plot(results: list[dict], out_png: Path,
          seoul: dict | None, forward_weeks: int) -> list[Path]:
    """국가별 그룹 패널 figure — (A) 지역별 R² 막대(국가 색) + Seoul 라이브 기준선,
    (B) 국가별 median R² 요약, (C) 국가 bubble 지도(geopandas 부재→bubble).

    각 패널을 ① 단독 PNG(panel별) 로 먼저 저장한 뒤 ② 동일 헬퍼로 결합 gridspec figure
    (out_png, 기존과 동일) 를 저장한다 (사용자 요구 "한 번에 한 그림씩").

    Args:
        results: _eval_region 산출 dict 리스트.
        out_png: 결합 출력 PNG 경로.
        seoul: compute_seoul_forward_baseline 산출(라이브) 또는 None(선 생략).
        forward_weeks: 공통 forward 창(주수) — 축/제목 명시.

    Returns:
        작성된 PNG 경로 리스트 (단독 3개 + 결합 1개 순서).

    Side effects: 단독 PNG 3개 + 결합 PNG(out_png) 작성 (dpi=120).
    """
    plt = _setup_matplotlib()

    # 국가별 그룹화 (CONFIG COUNTRIES 순서, 국가 내 R² 내림차순).
    by_country: dict[str, list[dict]] = {c: [] for c in COUNTRIES}
    for r in results:
        by_country.setdefault(r["country"], []).append(r)
    for c in by_country:
        by_country[c].sort(key=lambda d: d["champ_r2"], reverse=True)
    ordered = [r for c in COUNTRIES for r in by_country.get(c, [])]
    cnames = [c for c in COUNTRIES if by_country.get(c)]

    out_dir = out_png.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # ── ① 단독 PNG (panel별) ──
    p_r2 = out_dir / "fig_overseas_regions_champion_region_r2.png"
    fig1 = plt.figure(figsize=(13, 5))
    _panel_region_r2(fig1.add_subplot(111), plt, ordered, by_country, seoul,
                     forward_weeks, standalone=True)
    fig1.tight_layout()
    fig1.savefig(p_r2, dpi=120, bbox_inches="tight")
    plt.close(fig1)
    written.append(p_r2)

    p_med = out_dir / "fig_overseas_regions_champion_country_median.png"
    fig2 = plt.figure(figsize=(7, 5.5))
    _panel_country_median(fig2.add_subplot(111), cnames, by_country, seoul, standalone=True)
    fig2.tight_layout()
    fig2.savefig(p_med, dpi=120, bbox_inches="tight")
    plt.close(fig2)
    written.append(p_med)

    p_bubble = out_dir / "fig_overseas_regions_champion_country_bubble.png"
    fig3 = plt.figure(figsize=(7, 5.5))
    _panel_country_bubble(fig3.add_subplot(111), cnames, by_country, standalone=True)
    fig3.tight_layout()
    fig3.savefig(p_bubble, dpi=120, bbox_inches="tight")
    plt.close(fig3)
    written.append(p_bubble)

    # ── ② 결합 gridspec figure (out_png, 기존과 동일·back-compat) ──
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0],
                          hspace=0.42, wspace=0.22)
    axA = fig.add_subplot(gs[0, :])
    axB = fig.add_subplot(gs[1, 0])
    axC = fig.add_subplot(gs[1, 1])
    _panel_region_r2(axA, plt, ordered, by_country, seoul, forward_weeks, standalone=False)
    _panel_country_median(axB, cnames, by_country, seoul, standalone=False)
    _panel_country_bubble(axC, cnames, by_country, standalone=False)

    n_total = len(ordered)
    seoul_txt = (f"dashed = Seoul live forward R²={seoul['r2']:.3f} (same protocol)"
                 if seoul is not None else "Seoul live baseline not computable (line omitted)")
    fig.suptitle(
        f"Champion FusedEpi (TiRex+TabPFN) multi-country sub-national generalization: "
        f"{', '.join(cnames)} {n_total} regions, ILI calendar-locked forward"
        f"(in-sample ≤ 2026-02-09, forward from 2026-02-16, common {forward_weeks} weeks)\n"
        f"Note: surveillance scale differs by country (USA ILI% / JPN reports per sentinel / FRA per 100k / DEU practice index, "
        f"unit within-series) — unit-error regions excluded, R² evaluated on each region's own scale · {seoul_txt}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    written.append(out_png)
    return written


def _write_csv(results: list[dict], out_csv: Path,
               seoul: dict | None, forward_weeks: int) -> None:
    """지역별 결과 CSV 작성 (forward_weeks 공통창 + Seoul 라이브 기준선 행 명시).

    Args:
        results: _eval_region 산출 dict 리스트.
        out_csv: 출력 CSV 경로.
        seoul: compute_seoul_forward_baseline 산출(라이브) 또는 None.
        forward_weeks: 공통 forward 창(주수) — CSV 메타.

    Side effects: out_csv 작성.
    """
    import csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    cols = ["country", "region", "year_lo", "year_hi", "forward_weeks",
            "n_train", "n_test", "ymax",
            "champ_r2", "champ_wis", "persist_r2", "seasonal_r2",
            "persist_wis", "seasonal_wis", "champ_metric_source"]
    ordered = sorted(results, key=lambda d: (d["country"], -d["champ_r2"]))
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in ordered:
            w.writerow([r["country"], r["region"], YEAR_LO, YEAR_HI, forward_weeks,
                        r["n_train"], r["n_test"], f"{r['ymax']:.2f}",
                        f"{r['champ_r2']:.4f}", f"{r['champ_wis']:.4f}",
                        f"{r['persist_r2']:.4f}", f"{r['seasonal_r2']:.4f}",
                        f"{r['persist_wis']:.4f}", f"{r['seasonal_wis']:.4f}",
                        "model-derived (FusedEpi calendar-locked forward 1-step; unit within-series)"])
        # Seoul 라이브 forward 기준선도 1행 기록 (동일 프로토콜, source 명시).
        if seoul is not None:
            w.writerow(["SEOUL_REF", "Seoul", YEAR_LO, YEAR_HI, forward_weeks,
                        seoul.get("n_train", ""), seoul["n_test"], "",
                        f"{seoul['r2']:.4f}", f"{seoul['wis']:.4f}",
                        "", "", "", "", seoul["source"]])


def main() -> int:
    """4개국 sub-national 챔피언 일반화 평가 entry point.

    Returns:
        0 = 성공(figure+CSV 작성), 1 = 실데이터 없음(정직 skip).

    Side effects: figures/fig_overseas_regions_champion.{png,csv} 작성. DB read-only.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out_dir = Path(__file__).resolve().parents[1] / "results" / "figures"
    out_png = out_dir / "fig_overseas_regions_champion.png"
    out_csv = out_dir / "fig_overseas_regions_champion.csv"

    # ── smoke 모드 (env): 1지역만 calendar-locked forward 검증 후 종료 (figure/CSV 미작성) ──
    #   사용: MPH_OVERSEAS_SMOKE=USA:CA python -m simulation.scripts.fig_overseas_regions_champion
    smoke = os.environ.get("MPH_OVERSEAS_SMOKE", "").strip()
    in_sample_end = get_in_sample_end()

    con = read_only_connect()
    try:
        if smoke:
            country, _, region = smoke.partition(":")
            country = country.strip().upper()
            region = region.strip()
            log.info("=== SMOKE: %s / %s — calendar-locked forward 1지역 검증 ===", country, region)
            loaded = _load_region_series(con, country, region)
            if loaded is None:
                log.error("SMOKE: %s/%s 데이터 부족/band-밖 — 실패", country, region)
                return 1
            y, yw = loaded
            n_train = split_forward_by_isoweek(yw, in_sample_end)
            forward_len = common_forward_len([len(y) - n_train], cap=FORWARD_WEEKS_CAP)
            fwd_first = next((isoweek_monday(*yw[i]) for i in range(n_train, len(yw))), None)
            log.info("  in-sample=%d(≤%s) forward 가용=%d 공통창=%d 시작=%s",
                     n_train, in_sample_end.isoformat(), len(y) - n_train, forward_len, fwd_first)
            r = _eval_region(country, region, y, n_train, forward_len)
            if r is None:
                log.error("SMOKE: %s/%s forward eval 실패", country, region)
                return 1
            log.info("=== SMOKE OK: %s/%s forward R²=%.4f WIS=%.4f (n_fwd=%d, 시작 %s) ===",
                     country, region, r["champ_r2"], r["champ_wis"], r["n_test"], fwd_first)
            return 0

        # ── 1패스: 지역 로드 + calendar-locked split (n_train, 가용 forward 길이) ──
        loaded_all: list[tuple[str, str, np.ndarray, int, int]] = []
        for country in COUNTRIES:
            regions = _select_regions(con, country, REGIONS_PER_COUNTRY)
            for region in regions:
                loaded = _load_region_series(con, country, region)
                if loaded is None:
                    continue
                y, yw = loaded
                n_train = split_forward_by_isoweek(yw, in_sample_end)
                avail_fwd = len(y) - n_train
                if n_train < 70 or avail_fwd <= 0:
                    log.info("  [%s] [skip] %s: in-sample=%d forward 가용=%d (부적합)",
                             country, region, n_train, avail_fwd)
                    continue
                loaded_all.append((country, region, y, n_train, avail_fwd))

        if not loaded_all:
            log.error("실데이터 없음(forward 가용 0) — figure 생성 skip (정직).")
            return 1

        # ── 공통 forward 창 (가짜 연장 금지): 가용 min 과 CAP 중 작은 쪽 ──
        forward_weeks = common_forward_len([t[4] for t in loaded_all], cap=FORWARD_WEEKS_CAP)
        log.info("[공통 forward 창] 평가 %d개 지역 가용 forward = %s → 공통 %d주 truncate",
                 len(loaded_all), sorted(t[4] for t in loaded_all), forward_weeks)

        # ── 2패스: 공통 창으로 forward eval ──
        results: list[dict] = []
        for country, region, y, n_train, _avail in loaded_all:
            r = _eval_region(country, region, y, n_train, forward_weeks)
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
    log.info("\n=== 요약 (calendar-locked forward, 공통 %d주, in-sample≤%s) ===",
             forward_weeks, in_sample_end.isoformat())
    for country in COUNTRIES:
        sub = [r for r in results if r["country"] == country]
        if not sub:
            continue
        med = float(np.median([r["champ_r2"] for r in sub]))
        nbp = sum(r["champ_r2"] > r["persist_r2"] for r in sub)
        nbs = sum(r["champ_r2"] > r["seasonal_r2"] for r in sub)
        log.info("[%s] n=%d | median R²=%.4f | persist 능가 %d/%d | seasonal 능가 %d/%d",
                 country, len(sub), med, nbp, len(sub), nbs, len(sub))
    if seoul is not None:
        log.info("Seoul 라이브 forward R²=%.4f WIS=%.4f (동일 프로토콜, n_fwd=%d)",
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
