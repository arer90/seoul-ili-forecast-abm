"""해외 인플루엔자 양성률(positivity) bubble map — 정직한 scatter (경계 없음).

이 스크립트는 ``overseas_ili`` 테이블의 ``who_flunet`` source 를 사용해 12개
국가의 **최근 인플루엔자 검체 양성률(positivity %)** 을 세계 지도 위 bubble
(원) 산점으로 그린다.

★ 정직성 (geojson 부재):
    이 프로젝트에는 **서울(seoul) geojson 만** 있고 해외 국가 경계 polygon 이
    없다. 또한 ``geopandas`` (naturalearth_lowres world) 도 이 venv 에 설치돼
    있지 않다(``import geopandas`` 실패). 따라서 진짜 choropleth(국가 면적 채색)
    는 불가능하다. 대신 각 국가 수도(대략) 위·경도를 코드 내 dict 로 하드코딩하고,
    그 좌표에 **bubble(원)** 을 찍는다. 원 크기·색 = 양성률. 배경은 간단한 위경도
    grid 프레임(진짜 해안선 polygon 아님)이며, 이를 제목/부제에 명시한다.
    → "국가 경계 choropleth 인 척" 절대 금지. bubble/scatter 임을 figure 에 표기.

★ 양성률 계산 (단일 주차 노이즈 회피):
    ``who_flunet`` 의 ``ili_rate`` 와 ``positivity_pct`` 컬럼은 (수집기 aliasing
    으로) 값이 존재할 때 서로 동일하며, 최신 단일 주차는 비수기/소표본으로 매우
    불안정하다(예: KR 2026 w22 = 0.69% vs 최근 12주 집계 7.65%). 따라서 본 figure
    는 **국가별 최근 12주(specimen_total>0) 의 검체 합산 양성률**
    = ``100 × Σ specimen_positive / Σ specimen_total`` 을 사용한다(검체수 가중 →
    소표본 주차의 과대 변동 완화). 이 정의를 부제에 정직히 표기.

★ 해외 forecast 없음:
    OverseasTransfer 모델은 phantom(미배선)이라 해외 예측 산출이 없다. 본 figure
    는 **관측 양성률만** 그린다(예측/모델-유래 값 없음).

[데이터]
    DB ``simulation/data/db/epi_real_seoul.db`` → ``overseas_ili`` (read-only).
    source = ``who_flunet`` (12 국: AU CN DE FR GB HK JP KR NL SE SG US).
    서울(KR) 은 본 프로젝트 메인 타깃이므로 별도 강조 표식(테두리/주석).

[정직성 요약]
    - 진짜 국가 경계 없음 → bubble/scatter (제목 명시).
    - 양성률 = 관측 검체 합산(최근 12주, 검체수 가중) = 요약통계 (부제 명시).
    - 해외 예측 산출 없음 (관측만).
    - 데이터 부재 국가는 가짜 생성 없이 skip + 로그.

[출력]
    ``figures/fig_overseas_map.png`` (프로젝트 루트 기준).
    dpi=130, bbox_inches="tight". Agg backend + 한글폰트(AppleGothic→NanumGothic).

[결정성]
    순수 read + 결정적 통계(난수 미사용). 동일 DB 면 동일 PNG.

실행:
    .venv/bin/python -m simulation.scripts.fig_overseas_map
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless 렌더 (ENGINEERING_PRINCIPLES.md #1 OS 비종속)
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize

from simulation.database import read_only_connect  # G-116/117: 안전 read-only 헬퍼 사용

# ---------------------------------------------------------------------------
# 경로 SSOT (단일 DB / 프롬프트 지정 출력 경로 figures/)
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]  # .../MPH_infection_simulation
FIG_DIR = _ROOT / "simulation" / "results" / "figures"
FIG_PATH = FIG_DIR / "fig_overseas_map.png"

WHO_SOURCE = "who_flunet"
ROLLING_WEEKS = 12  # 최근 N주 검체 합산(단일 주차 노이즈 회피)

# 국가별 대략 좌표(수도 부근) + 한국어 라벨. 진짜 polygon 아님 — bubble 위치용.
# (lon, lat, 한국어 라벨)
COUNTRY_COORDS: dict[str, tuple[float, float, str]] = {
    "AU": (149.13, -35.28, "Australia"),
    "CN": (116.40, 39.90, "China"),
    "DE": (13.40, 52.52, "Germany"),
    "FR": (2.35, 48.86, "France"),
    "GB": (-0.13, 51.51, "United Kingdom"),
    "HK": (114.17, 22.32, "Hong Kong"),
    "JP": (139.69, 35.69, "Japan"),
    "KR": (126.98, 37.57, "South Korea (Seoul)"),
    "NL": (4.90, 52.37, "Netherlands"),
    "SE": (18.07, 59.33, "Sweden"),
    "SG": (103.82, 1.35, "Singapore"),
    "US": (-77.04, 38.91, "United States"),
}

# 라벨 충돌 회피용 수동 오프셋(경도, 위도). bubble 위치는 그대로, 글자만 이동.
# 밀집 클러스터(서유럽 GB/FR/NL/DE, 동아시아 CN/KR/JP/HK) 의 텍스트 겹침 방지.
LABEL_OFFSET: dict[str, tuple[float, float]] = {
    "GB": (-22.0, -2.0),
    "FR": (-14.0, -16.0),
    "NL": (-2.0, 14.0),
    "DE": (24.0, 4.0),
    "CN": (-2.0, 16.0),
    "KR": (-12.0, -14.0),
    "JP": (20.0, 4.0),
    "HK": (28.0, -8.0),
}


# ===========================================================================
# 한글 폰트 (AppleGothic→NanumGothic fallback)
# ===========================================================================
def _set_korean_font() -> str:
    """matplotlib 한글 폰트를 설정한다 (AppleGothic→NanumGothic fallback).

    등록된 폰트 중 우선순위대로 첫 가용 폰트를 ``font.family`` 로 지정하고,
    음수 부호 깨짐(``axes.unicode_minus``)을 끈다.

    Returns:
        선택된 폰트 이름. 한글 폰트 부재 시 ``"DejaVu Sans"`` 반환 + 경고 로그
        (가짜 폰트 생성 금지).

    Side effects: ``plt.rcParams`` 전역 변경 (font.family / axes.unicode_minus).
    """
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans"


# ===========================================================================
# 데이터 로드 (read-only DB)
# ===========================================================================
def load_positivity(weeks: int = ROLLING_WEEKS) -> dict[str, tuple[float, int, str]]:
    """who_flunet 국가별 최근 N주 검체 합산 양성률을 로드한다.

    각 국가의 최신 N주(specimen_total>0) 행을 모아 검체수 가중 양성률
    ``100 × Σ specimen_positive / Σ specimen_total`` 을 계산한다(단일 주차
    소표본 변동 완화).

    Args:
        weeks: 합산할 최근 주차 수 (기본 12). >0.

    Returns:
        ``{country_code: (positivity_pct, total_specimens, latest_label)}`` dict.
        검체가 없는 국가는 제외. positivity_pct 단위 = %, 범위 [0, 100].

    Raises:
        없음 — DB 연결 실패 시 read_only_connect 가 예외 전파.

    Side effects: DB 읽기 전용 open/close. 디스크 쓰기 없음.
    """
    con = read_only_connect()
    try:
        cur = con.cursor()
        cur.execute(
            """
            WITH ranked AS (
              SELECT country, year, week_no, specimen_positive, specimen_total,
                     ROW_NUMBER() OVER (
                       PARTITION BY country ORDER BY year DESC, week_no DESC
                     ) AS rn
              FROM overseas_ili
              WHERE source = ? AND specimen_total > 0
            )
            SELECT country,
                   100.0 * SUM(specimen_positive) / SUM(specimen_total) AS pos_pct,
                   SUM(specimen_total) AS spec,
                   MAX(year) AS y, MAX(week_no) AS w
            FROM ranked
            WHERE rn <= ?
            GROUP BY country
            """,
            (WHO_SOURCE, weeks),
        )
        out: dict[str, tuple[float, int, str]] = {}
        for country, pos, spec, y, w in cur.fetchall():
            if pos is None or spec is None or spec <= 0:
                continue
            out[country] = (float(pos), int(spec), f"~{int(y)} wk {int(w)}")
        return out
    finally:
        con.close()


# ===========================================================================
# Figure
# ===========================================================================
def _draw_world_frame(ax) -> None:
    """위경도 grid 프레임을 그린다 (진짜 해안선 polygon 아님 — 정직성).

    Side effects: ``ax`` 에 grid line + 축 한계/라벨 설정.
    """
    ax.set_xlim(-130, 165)
    ax.set_ylim(-55, 75)
    # 경도/위도 grid (참조선)
    for lon in range(-120, 180, 30):
        ax.axvline(lon, color="#D9DEE4", lw=0.6, zorder=0)
    for lat in range(-40, 80, 20):
        ax.axhline(lat, color="#D9DEE4", lw=0.6, zorder=0)
    ax.axhline(0, color="#B7C0CC", lw=1.0, ls="--", zorder=0)  # 적도
    ax.set_xlabel("Longitude (°E)", fontsize=9)
    ax.set_ylabel("Latitude (°N)", fontsize=9)
    ax.set_facecolor("#F7F9FB")


def build_figure() -> Path:
    """해외 양성률 bubble map figure 를 생성·저장한다.

    Returns:
        저장된 PNG 경로. 데이터가 전혀 없으면 그래도 빈 프레임 대신 None-안전
        ValueError 를 raise (가짜 figure 금지).

    Raises:
        ValueError: who_flunet 양성률 데이터가 0개일 때 (fail-loud, G-237).

    Side effects: ``figures/fig_overseas_map.png`` 쓰기. matplotlib 전역 상태 변경.
    """
    font = _set_korean_font()
    data = load_positivity()
    if not data:
        raise ValueError(
            "who_flunet 양성률 데이터 0개 — figure 생성 중단 (가짜 생성 금지)"
        )

    # 좌표 있는 국가만 플롯 (좌표 dict 가 SSOT)
    plotted = {c: v for c, v in data.items() if c in COUNTRY_COORDS}
    skipped = sorted(set(data) - set(COUNTRY_COORDS))
    if skipped:
        print(f"[fig_overseas_map] INFO: 좌표 미등록 국가 skip: {skipped}")

    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    _draw_world_frame(ax)

    pos_vals = np.array([plotted[c][0] for c in plotted])
    vmax = float(max(pos_vals.max(), 1.0))
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = matplotlib.colormaps["YlOrRd"]

    # bubble 크기: 양성률에 비례 (면적 스케일, 최소 가시성 보장)
    def _size(p: float) -> float:
        return 120.0 + (p / vmax) * 2600.0

    for code, (pos, spec, _lbl) in sorted(
        plotted.items(), key=lambda kv: kv[1][0], reverse=True
    ):
        lon, lat, kr = COUNTRY_COORDS[code]
        is_kr = code == "KR"
        ax.scatter(
            lon,
            lat,
            s=_size(pos),
            c=[cmap(norm(pos))],
            edgecolors="#1F3A5F" if is_kr else "#444444",
            linewidths=2.6 if is_kr else 0.9,
            alpha=0.88,
            zorder=5 if not is_kr else 6,
        )
        # 라벨: 국가 + 양성률. 밀집 클러스터는 수동 오프셋 + 지시선(leader line).
        if code in LABEL_OFFSET:
            ox, oy = LABEL_OFFSET[code]
            tx, ty = lon + ox, lat + oy
            ax.annotate(
                f"{kr}\n{pos:.1f}%",
                xy=(lon, lat),
                xytext=(tx, ty),
                ha="center",
                va="center",
                fontsize=8.2,
                fontweight="bold" if is_kr else "normal",
                color="#1F3A5F" if is_kr else "#222222",
                arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6),
                zorder=7,
            )
        else:
            dy = 6.5 + np.sqrt(_size(pos)) / 14.0
            ax.annotate(
                f"{kr}\n{pos:.1f}%",
                (lon, lat),
                xytext=(lon, lat + dy),
                ha="center",
                va="bottom",
                fontsize=8.2,
                color="#222222",
                zorder=7,
            )

    # 컬러바
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Influenza specimen positivity (%)", fontsize=9)

    # bubble 크기 범례 (대표 3개 값)
    legend_vals = [v for v in (5, 25, 50) if v <= vmax] or [round(vmax, 0)]
    handles = [
        ax.scatter(
            [], [], s=_size(v), c=[cmap(norm(v))], edgecolors="#444",
            linewidths=0.8, alpha=0.8, label=f"{v:.0f}%",
        )
        for v in legend_vals
    ]
    leg = ax.legend(
        handles=handles,
        title="Positivity (bubble size)",
        loc="lower left",
        frameon=True,
        labelspacing=1.6,
        borderpad=1.0,
        fontsize=8,
        title_fontsize=8.5,
    )
    leg.get_frame().set_alpha(0.9)

    n_iso = ROLLING_WEEKS
    # 제목 + 2줄 정직성 부제 (경계 없음 / 양성률 정의 / 예측 없음). 제목과 분리 배치.
    fig.suptitle(
        "Overseas influenza specimen positivity — WHO FluNet (observed, bubble/scatter)",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )
    subtitle = (
        f"Note: no true country-boundary geojson/geopandas available -> bubbles on capital coordinates (not a choropleth). "
        f"Bubble size/color = positivity pooled over the last {n_iso} weeks (specimen-weighted, = Sigma positive / Sigma specimens).\n"
        f"who_flunet {len(plotted)} countries, observed only — no overseas forecast output (OverseasTransfer not wired). "
        f"South Korea (Seoul) = main target (bold outline) highlighted."
    )
    fig.text(
        0.5,
        0.925,
        subtitle,
        ha="center",
        va="top",
        fontsize=8.2,
        color="#555555",
    )

    fig.tight_layout(rect=(0, 0, 1, 0.91))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_PATH, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(
        f"[fig_overseas_map] saved → {FIG_PATH} "
        f"({len(plotted)}개국, font={font}, vmax={vmax:.1f}%)"
    )
    return FIG_PATH


def main() -> int:
    try:
        path = build_figure()
    except Exception as exc:  # fail-loud (G-237): 가짜 success 금지
        print(f"[fig_overseas_map] ERROR: {exc}", file=sys.stderr)
        return 1
    size = path.stat().st_size if path.exists() else 0
    if size <= 0:
        print(f"[fig_overseas_map] ERROR: 출력 파일 0바이트 {path}", file=sys.stderr)
        return 1
    print(f"[fig_overseas_map] OK: {path} ({size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
