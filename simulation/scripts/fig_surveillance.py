"""감염병 감시 figure 2종 — 외부임팩트/경보 타임라인 + Serfling epidemic baseline.

이 스크립트는 서울 표본감시 인플루엔자(``sentinel_influenza``) **전연령 ILI 354주**
시계열 위에서 두 가지 감시(surveillance) figure 를 그린다. **실측 관측만 사용**하며,
DB 를 read-only 로 연다. 모델/가짜/합성 데이터는 일절 만들지 않는다.

  figure 1 (``surveillance_alert_timeline``):
      ``simulation.analytics.external_impact`` 의 ``detect_regime_shifts`` +
      ``pandemic_alert_level`` 을 전연령 ILI 에 적용. ILI 곡선 위에 KDCA식 4단계
      경보(관심/주의/경계/심각) 배경 음영 + CUSUM onset(레짐전환) 마커를 겹친다.
      경보 단계·onset 은 **causal baseline 기반 산출치(모델-유래 진단)** 이므로 제목·
      범례에 그 사실을 명시한다(관측인 척 금지).

  figure 2 (``surveillance_epidemic_curve_serfling``):
      동일 ILI 시계열 + 계절(주차) baseline. 각 ISO 주차의 **평년 평균 + 2SD**
      (KDCA epidemic threshold, Serfling식 주차 회귀의 단순 비모수 버전)을 epidemic
      threshold 로 그리고, threshold 를 초과한 구간을 음영 처리한다. baseline/threshold
      는 관측에서 직접 계산한 **요약통계**이며 제목에 명시한다.

[데이터]
    DB ``simulation/data/db/epi_real_seoul.db`` → ``sentinel_influenza``.
    전연령 ILI = 7개 연령밴드(0세 / 1-6세 / 7-12세 / 13-18세 / 19-49세 / 50-64세 /
    65세 이상) 의 **주차별 평균**. (테이블에 연령별 분모(내원수)가 없어 가중평균 불가
    → 비가중 산술평균. 이 점을 부제에 정직히 표기.)
    시간 순서 = (season_start, week_seq). x축 날짜는 week_label(ISO 주차)→월요일로
    환산(라벨 표기용; 정렬은 season/week 순).

[정직성]
    - 경보 단계·onset = 산출 진단치(모델-유래) → 제목/범례 명시.
    - baseline/threshold = 관측 요약통계 → 제목 명시.
    - 데이터 부재 시 가짜 생성 없이 정직히 skip + 로그.

[출력]
    ``simulation/results/figures/surveillance_alert_timeline.png``
    ``simulation/results/figures/surveillance_epidemic_curve_serfling.png``
    dpi=120, bbox_inches="tight". Agg backend + 한글폰트(AppleGothic→NanumGothic).

[결정성]
    순수 read + 결정적 통계(난수 미사용). 동일 DB 면 동일 PNG.

실행:
    .venv/bin/python -m simulation.scripts.fig_surveillance
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless 렌더 (G-001 OS 비종속)
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

from simulation.analytics.external_impact import (
    ALERT_LABELS_KR,
    detect_regime_shifts,
    pandemic_alert_level,
)

# ---------------------------------------------------------------------------
# 경로 SSOT (ENGINEERING_PRINCIPLES.md #4: 단일 DB / 단일 출력 디렉터리)
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]  # .../MPH_infection_simulation
DB_PATH = _ROOT / "simulation" / "data" / "db" / "epi_real_seoul.db"
FIG_DIR = _ROOT / "simulation" / "results" / "figures"

# KDCA식 4단계 경보 배경색 (관심=무음영 / 주의=노랑 / 경계=주황 / 심각=빨강)
_ALERT_FACECOLORS: tuple[str, str, str, str] = (
    "none",       # 0 관심 (배경 음영 없음)
    "#FFF3B0",    # 1 주의
    "#FBC687",    # 2 경계
    "#F08A8A",    # 3 심각
)


# ===========================================================================
# 한글 폰트
# ===========================================================================
def _set_korean_font() -> str:
    """matplotlib 한글 폰트를 설정한다 (AppleGothic→NanumGothic fallback).

    등록된 폰트 중 우선순위대로 첫 가용 폰트를 ``font.family`` 로 지정하고,
    음수 부호 깨짐(``axes.unicode_minus``)을 끈다.

    Returns:
        실제로 선택된 폰트 이름. 한글 폰트가 하나도 없으면 ``"DejaVu Sans"``
        (기본) 를 반환하고 경고를 로그한다 (가짜 폰트 생성 금지).

    Performance: O(폰트수) 1회 스캔.
    Side effects: ``plt.rcParams`` 전역 변경 (font.family / axes.unicode_minus).
    """
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans"


# ===========================================================================
# 데이터 로드 (read-only DB)
# ===========================================================================
def _label_num_to_date(season_start: int, label_num: int) -> date:
    """season + ISO 주차 라벨을 월요일 날짜로 환산한다 (축 표기용, 결정적).

    표본감시 시즌은 ISO 주차 36 부터 시작해 익년 주차 35 까지 이어진다. 따라서
    라벨번호 >=36 은 ``season_start`` 연도, <36 은 ``season_start+1`` 연도에 속한다.
    해당 ISO 연도에 그 주차가 없으면(예: 53주 미존재) 52주로 안전 fallback.

    Args:
        season_start: 시즌 시작 연도 (예: 2019).
        label_num: ISO 주차 번호 (1..53).

    Returns:
        그 ISO 주차의 월요일 ``datetime.date``.

    Performance: O(1). Side effects: 없음.
    """
    cal_year = season_start if label_num >= 36 else season_start + 1
    try:
        return date.fromisocalendar(cal_year, label_num, 1)
    except ValueError:
        return date.fromisocalendar(cal_year, 52, 1)


def load_all_age_ili(db_path: Path) -> dict:
    """``sentinel_influenza`` 전연령 ILI 시계열을 read-only 로 로드한다.

    7개 연령밴드를 (season_start, week_seq) 별로 산술평균해 전연령 ILI 를 만든다.
    (연령별 분모가 테이블에 없어 가중평균 불가 — 비가중 평균. 호출자가 부제에 명시.)
    시간 순서는 (season_start, week_seq) 오름차순.

    Args:
        db_path: ``epi_real_seoul.db`` 경로.

    Returns:
        dict:
          - ``"ili"``: ``(n,)`` float64 전연령 ILI (주차별 연령밴드 평균).
          - ``"dates"``: ``list[datetime.date]`` 길이 n (라벨→월요일).
          - ``"iso_week"``: ``(n,)`` int — 각 시점 ISO 주차(1..53), Serfling 주차 baseline 용.
          - ``"season"``: ``(n,)`` int — season_start.
          - ``"n_age"``: 평균에 사용된 연령밴드 수 (정직성 표기용).
        데이터가 없으면 빈 ``"ili"`` (n=0) 를 담아 반환 (호출자가 skip 판단).

    Raises:
        FileNotFoundError: DB 파일 부재.

    Performance: 단일 SELECT + O(n) 집계. Side effects: DB read-only open/close.
    Caller responsibility: ili NaN 없음(평균이라 1밴드만 있어도 값 존재); 빈 결과 시 skip.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"DB 없음: {db_path}")

    # read-only 연결 (single-writer 위생, 학습 run 과 충돌 방지; G-116/G-117)
    from simulation.database import read_only_connect
    con = read_only_connect(str(db_path))
    try:
        cur = con.cursor()
        # 주차별 전연령 평균 + 라벨(대표 1개) + 연령밴드 수
        cur.execute(
            """
            SELECT season_start,
                   week_seq,
                   AVG(ili_rate)            AS ili_mean,
                   COUNT(*)                 AS n_age,
                   MIN(week_label)          AS any_label
            FROM sentinel_influenza
            WHERE ili_rate IS NOT NULL
            GROUP BY season_start, week_seq
            ORDER BY season_start, week_seq
            """
        )
        rows = cur.fetchall()
    finally:
        con.close()

    if not rows:
        return {"ili": np.empty(0), "dates": [], "iso_week": np.empty(0, int),
                "season": np.empty(0, int), "n_age": 0}

    ili = np.array([r[2] for r in rows], dtype=np.float64)
    seasons = np.array([r[0] for r in rows], dtype=int)
    n_age = int(max(r[3] for r in rows))

    iso_week: list[int] = []
    dates: list[date] = []
    for season_start, _week_seq, _ili, _nage, any_label in rows:
        # week_label 예: "36주" / "01주" → 숫자 추출
        digits = "".join(ch for ch in any_label if ch.isdigit())
        wk = int(digits) if digits else 1
        iso_week.append(wk)
        dates.append(_label_num_to_date(int(season_start), wk))

    return {
        "ili": ili,
        "dates": dates,
        "iso_week": np.array(iso_week, dtype=int),
        "season": seasons,
        "n_age": n_age,
    }


# ===========================================================================
# Serfling식 주차 baseline (평년 평균 + 2SD epidemic threshold)
# ===========================================================================
def seasonal_baseline(ili: np.ndarray, iso_week: np.ndarray, *, sd_mult: float = 2.0
                      ) -> tuple[np.ndarray, np.ndarray]:
    """ISO 주차별 평년 평균 + (mean + sd_mult·SD) epidemic threshold 를 계산한다.

    KDCA epidemic threshold(비유행기 mean+2SD; Kang 2024) 의 주차 분해 버전이자
    Serfling(1963) cyclic baseline 의 단순 비모수 형태. 각 ISO 주차 w 에 대해 전
    시즌의 그 주차 관측을 모아 평균/표준편차를 구하고, 모든 시점에 그 주차의 값을
    매핑한다.

    Args:
        ili: ``(n,)`` 전연령 ILI 시계열.
        iso_week: ``(n,)`` 각 시점의 ISO 주차(1..53). ili 와 동일 길이.
        sd_mult: threshold 의 SD 배수. 기본 2.0 (KDCA mean+2SD).

    Returns:
        ``(baseline, threshold)`` 각각 ``(n,)`` float64. baseline=주차 평균,
        threshold=주차 평균 + sd_mult·주차 SD. 표본 1개뿐인 주차는 SD=0 처리.

    Raises:
        ValueError: ili 와 iso_week 길이 불일치.

    Performance: O(n). Side effects: 없음 (순수 함수).
    Caller responsibility: 동일 정렬·동일 길이.
    """
    if ili.shape[0] != iso_week.shape[0]:
        raise ValueError(
            f"길이 불일치: ili={ili.shape[0]}, iso_week={iso_week.shape[0]}"
        )
    n = ili.shape[0]
    baseline = np.empty(n, dtype=np.float64)
    threshold = np.empty(n, dtype=np.float64)
    for w in np.unique(iso_week):
        mask = iso_week == w
        vals = ili[mask]
        mu = float(np.mean(vals))
        sd = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
        baseline[mask] = mu
        threshold[mask] = mu + sd_mult * sd
    return baseline, threshold


# ===========================================================================
# figure 1: 외부임팩트 / 경보 타임라인
# ===========================================================================
def _alert_segments(level: np.ndarray) -> list[tuple[int, int, int]]:
    """경보 레벨 배열을 연속 동일-레벨 구간 ``(start, end, lvl)`` 으로 압축한다.

    Args:
        level: ``(n,)`` int 경보 단계(0..3).

    Returns:
        ``list[(start_idx, end_idx_exclusive, lvl)]`` — axvspan 음영용.

    Performance: O(n). Side effects: 없음.
    """
    segs: list[tuple[int, int, int]] = []
    if level.size == 0:
        return segs
    start = 0
    cur = int(level[0])
    for i in range(1, level.size):
        if int(level[i]) != cur:
            segs.append((start, i, cur))
            start = i
            cur = int(level[i])
    segs.append((start, level.size, cur))
    return segs


def make_alert_timeline(data: dict, out_path: Path) -> bool:
    """figure 1 — ILI 곡선 + KDCA 4단계 경보 음영 + CUSUM onset 마커.

    Args:
        data: ``load_all_age_ili`` 결과 dict.
        out_path: 저장 PNG 경로.

    Returns:
        성공 저장 시 True, 데이터 부족 시 False (가짜 생성 없이 skip).

    Performance: O(n). Side effects: PNG 파일 write.
    """
    ili = data["ili"]
    dates = data["dates"]
    if ili.size < 20:
        print(f"[fig_surveillance] SKIP alert_timeline: n={ili.size} (<20).")
        return False

    # 경보 단계 + onset (산출 진단치 — 모델-유래). 전부 causal baseline.
    alert = pandemic_alert_level(ili, return_labels=True)
    level = np.asarray(alert["level"], dtype=int)
    regime = detect_regime_shifts(ili, method="cusum")
    onsets = regime["changepoints"]

    x = np.arange(ili.size)
    fig, ax = plt.subplots(figsize=(13, 5.2))

    # 경보 단계 배경 음영 (0=관심은 무음영)
    used_levels: set[int] = set()
    for s, e, lvl in _alert_segments(level):
        if lvl == 0:
            continue
        ax.axvspan(x[s] - 0.5, x[e - 1] + 0.5,
                   color=_ALERT_FACECOLORS[lvl], alpha=0.75, zorder=0, linewidth=0)
        used_levels.add(lvl)

    # ILI 관측 곡선
    ax.plot(x, ili, color="#1f3b73", linewidth=1.6, zorder=3,
            label="all-age ILI (observed)")

    # CUSUM onset 마커 (레짐전환 — 산출 진단치)
    if onsets:
        ax.plot([x[t] for t in onsets], [ili[t] for t in onsets],
                marker="v", linestyle="none", markersize=9,
                markerfacecolor="#c0392b", markeredgecolor="black",
                markeredgewidth=0.6, zorder=5,
                label=f"CUSUM regime-shift onset (computed, n={len(onsets)})")

    # x축: 시즌 경계마다 연-주 라벨
    _apply_date_ticks(ax, dates, x)

    ax.set_ylabel("ILI rate (ILI proportion, ‰)")
    ax.set_xlabel("sentinel surveillance week (season starts week 36 → week 35 of following year)")
    ax.set_title(
        "Seoul all-age influenza ILI — external impact / alert timeline\n"
        "background = KDCA 4-level alert (Watch/Caution/Alert/Serious), inverted triangle = CUSUM regime-shift onset "
        "· alert & onset are causal-baseline computed values (model-derived), ILI is observed",
        fontsize=11, loc="left",
    )
    _add_alert_legend(ax, used_levels)
    ax.margins(x=0.005)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def _add_alert_legend(ax, used_levels: set[int]) -> None:
    """경보 단계 배경색 + 곡선/마커를 합쳐 범례를 단다.

    Args:
        ax: 대상 Axes.
        used_levels: 실제 음영된 경보 단계 집합 (1..3).

    Side effects: ax 에 legend 추가.
    """
    from matplotlib.patches import Patch

    # English display for KDCA 4-level alert names (data module stays Korean).
    _alert_en = {0: "Watch", 1: "Caution", 2: "Alert", 3: "Serious"}

    handles, labels = ax.get_legend_handles_labels()
    for lvl in (1, 2, 3):
        if lvl in used_levels:
            name_en = _alert_en.get(lvl, ALERT_LABELS_KR[lvl])
            handles.append(Patch(facecolor=_ALERT_FACECOLORS[lvl], alpha=0.75,
                                  label=f"Level {lvl} = {name_en}"))
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.9,
              ncol=2)


# ===========================================================================
# figure 2: epidemic curve + Serfling baseline
# ===========================================================================
def make_epidemic_curve(data: dict, out_path: Path) -> bool:
    """figure 2 — ILI epidemic curve + 주차 평년 baseline + mean+2SD threshold 음영.

    Args:
        data: ``load_all_age_ili`` 결과 dict.
        out_path: 저장 PNG 경로.

    Returns:
        성공 저장 시 True, 데이터 부족 시 False (가짜 생성 없이 skip).

    Performance: O(n). Side effects: PNG 파일 write.
    """
    ili = data["ili"]
    iso_week = data["iso_week"]
    dates = data["dates"]
    if ili.size < 20:
        print(f"[fig_surveillance] SKIP epidemic_curve: n={ili.size} (<20).")
        return False

    baseline, threshold = seasonal_baseline(ili, iso_week, sd_mult=2.0)
    x = np.arange(ili.size)
    above = ili > threshold  # epidemic threshold 초과 구간

    fig, ax = plt.subplots(figsize=(13, 5.2))

    # threshold 초과 음영 (관측이 epidemic threshold 를 넘은 구간)
    ax.fill_between(x, threshold, ili, where=above, interpolate=True,
                    color="#F08A8A", alpha=0.55, zorder=1,
                    label="above epidemic threshold")

    # 관측 ILI
    ax.plot(x, ili, color="#1f3b73", linewidth=1.6, zorder=4,
            label="all-age ILI (observed)")
    # 주차 평년 baseline (관측 요약통계)
    ax.plot(x, baseline, color="#2e8b57", linewidth=1.3, linestyle="--",
            zorder=3, label="weekly seasonal-average baseline (observed mean)")
    # mean + 2SD epidemic threshold (Serfling/KDCA)
    ax.plot(x, threshold, color="#c0392b", linewidth=1.3, linestyle="-.",
            zorder=3, label="epidemic threshold (seasonal-average mean + 2SD)")

    _apply_date_ticks(ax, dates, x)
    ax.set_ylabel("ILI rate (ILI proportion, ‰)")
    ax.set_xlabel("sentinel surveillance week (season starts week 36 → week 35 of following year)")
    ax.set_title(
        "Seoul all-age influenza ILI — epidemic curve + Serfling/KDCA seasonal baseline\n"
        "weekly seasonal-average mean + 2SD = epidemic threshold (observed summary statistic, nonparametric weekly regression) "
        "· exceedance interval shaded",
        fontsize=11, loc="left",
    )
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9, ncol=2)
    ax.margins(x=0.005)
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


# ===========================================================================
# 공통: x축 날짜 틱
# ===========================================================================
def _apply_date_ticks(ax, dates: list[date], x: np.ndarray) -> None:
    """시즌 경계(36주 시작) 근처에 연-월 날짜 틱을 단다.

    표본감시 시즌 시작(9월 초)마다 1틱을 찍어 가독성 확보. 너무 촘촘하면 격년으로.

    Args:
        ax: 대상 Axes.
        dates: 길이 n 의 date 리스트 (시점별).
        x: 길이 n 의 정수 인덱스.

    Side effects: ax 의 xticks/xticklabels 변경.
    """
    if not dates:
        return
    # 시즌 시작 = 9월(month==9) 가장 이른 주차에 틱. 대략 매년 1개.
    tick_idx: list[int] = []
    seen_years: set[int] = set()
    for i, d in enumerate(dates):
        if d.month == 9 and d.year not in seen_years:
            tick_idx.append(i)
            seen_years.add(d.year)
    if len(tick_idx) < 2:  # fallback: 균등 8틱
        step = max(1, len(dates) // 8)
        tick_idx = list(range(0, len(dates), step))
    ax.set_xticks([x[i] for i in tick_idx])
    ax.set_xticklabels([dates[i].strftime("%Y-%m") for i in tick_idx],
                       rotation=0, fontsize=9)


# ===========================================================================
# main
# ===========================================================================
def main() -> int:
    """두 figure 를 생성하고 PNG 존재·비영 크기를 검증한다.

    Returns:
        프로세스 종료코드 (0=성공, 1=데이터 부재로 둘 다 skip, 2=저장 검증 실패).

    Side effects: 한글폰트 rcParams 설정, PNG 2개 write, stdout 로그.
    """
    font = _set_korean_font()
    print(f"[fig_surveillance] font={font}  db={DB_PATH}")

    try:
        data = load_all_age_ili(DB_PATH)
    except FileNotFoundError as e:
        print(f"[fig_surveillance] SKIP (DB 부재): {e}")
        return 1

    n = data["ili"].size
    if n == 0:
        print("[fig_surveillance] SKIP: sentinel_influenza 관측 0행 (가짜 생성 안 함).")
        return 1
    print(f"[fig_surveillance] 전연령 ILI n={n}주, 연령밴드={data['n_age']}개 평균, "
          f"기간={data['dates'][0]}~{data['dates'][-1]}")

    out1 = FIG_DIR / "surveillance_alert_timeline.png"
    out2 = FIG_DIR / "surveillance_epidemic_curve_serfling.png"

    ok1 = make_alert_timeline(data, out1)
    ok2 = make_epidemic_curve(data, out2)

    # 저장 검증 (os.path.getsize > 0)
    rc = 0
    for ok, path in ((ok1, out1), (ok2, out2)):
        if not ok:
            continue
        if path.exists() and path.stat().st_size > 0:
            print(f"[fig_surveillance] OK  {path}  ({path.stat().st_size} bytes)")
        else:
            print(f"[fig_surveillance] FAIL 저장 검증: {path}")
            rc = 2
    if not (ok1 or ok2):
        return 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
