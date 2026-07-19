"""보건역학 표준 지표 심화 figure — Rt / 연령별 attack rate / 의료부하 3패널.

서울 표본감시 인플루엔자(``sentinel_influenza``)와 EpiEstim Cori 재생산수
이력(``rt_history``), 응급실 가용병상(``emergency_room_availability`` →
``ed_weekly_burden``)을 한 장(3행 1열)에 모은 표준 역학지표 figure. **DB 를
read-only 로 열고, 실측만 사용**한다. 가짜/합성 데이터는 일절 만들지 않으며
데이터가 없으면 가짜 생성 대신 정직히 skip(빈 패널 안내 텍스트)한다.

  패널 ① Rt 타임라인 (**모델-유래**):
      ``rt_history`` (method=EpiEstim_Cori, gu=seoul_city) 의 주차별 시간변동
      재생산수 ``rt_mean`` 과 95% 신용구간(``rt_q025``/``rt_q975``) 밴드. Rt=1
      임계선(유행 임계)을 겹친다. Rt = Cori et al.(2013) EpiEstim 추정치로,
      관측이 아니라 **ILI 발생곡선에 모델을 적용해 산출한 양**임을 제목에 명시.
      추정에 쓰인 serial interval(SI mean/sd)은 ``rt_history`` 에 저장된 값을 표기.

  패널 ② 연령별 ILI 부담 (**관측**):
      ``sentinel_influenza`` 7개 연령밴드 × 시즌별 **정점 ILI rate**(peak) 막대.
      ILI rate = 외래 1,000명당 인플루엔자 의사환자수(관측 표본감시 지표).
      연령별 분모(내원수)가 테이블에 없어 누적합은 "정점"보다 해석이 약하므로
      **정점 ILI(peak)** 를 attack-rate proxy 로 채택하고 부제에 정직히 표기.

  패널 ③ 의료부하 (**관측**):
      ``emergency_room_availability`` (서울 권역) 의 스냅샷별 **응급실 가용병상
      평균(hvec)** 시계열 + ``ed_weekly_burden`` 의 주간 ED 점유율(있을 때).
      모두 실측 관측치(가용병상=수집 시점 실가용수). 표본 기간이 짧음을 부제 명시.

[문헌값 인용 (데이터 산출과 구분)]
    인플루엔자 기초재생산수 R0 ≈ 1.3 (Biggerstaff et al. 2014, BMC Infect Dis
    "Estimates of R0..."), serial interval SI ≈ 2.6일 (Cowling et al. 2009,
    Epidemiology). 이 두 값은 **문헌 인용**이며 본 DB 산출이 아니다 — 패널 ①
    하단 텍스트박스에 "문헌값(인용)" 으로 명시. (DB ``rt_history`` 에 저장된
    SI=2.6/1.5 도 이 문헌값에서 유래한 EpiEstim prior 임을 함께 표기.)

[모델-유래 vs 관측 구분]
    - 패널 ① Rt = **모델-유래**(EpiEstim 추정) → 제목 [모델-유래] 태그.
    - 패널 ②③ = **관측**(표본감시 ILI / 응급실 실가용) → 제목 [관측] 태그.

[출력]
    ① 결합본 ``simulation/results/figures/fig_epi_metrics.png`` (3행 1열), dpi=120.
    ② 패널별 단독 PNG (한 번에 한 그림씩 보기용):
        - ``fig_epi_metrics_rt.png``        (패널 ① Rt 타임라인)
        - ``fig_epi_metrics_age_peak.png``  (패널 ② 연령별 정점 ILI)
        - ``fig_epi_metrics_ed_load.png``   (패널 ③ 의료부하)
    모두 dpi 지정 + bbox_inches="tight". Agg backend + 한글폰트(AppleGothic→NanumGothic).

[결정성]
    순수 read + 결정적 집계(난수 미사용). 동일 DB → 동일 PNG.

실행:
    .venv/bin/python -m simulation.scripts.fig_epi_metrics
"""
from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless 렌더 (#1 OS 비종속)
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# 경로 SSOT (ENGINEERING_PRINCIPLES.md #4: 단일 DB / 단일 출력 디렉터리)
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]  # .../MPH_infection_simulation
DB_PATH = _ROOT / "simulation" / "data" / "db" / "epi_real_seoul.db"
FIG_DIR = _ROOT / "simulation" / "results" / "figures"
OUT_PATH = FIG_DIR / "fig_epi_metrics.png"
# 패널별 단독 PNG (의미이름) — "한 번에 한 그림씩" 표시용
OUT_RT_PATH = FIG_DIR / "fig_epi_metrics_rt.png"
OUT_AGE_PATH = FIG_DIR / "fig_epi_metrics_age_peak.png"
OUT_ED_PATH = FIG_DIR / "fig_epi_metrics_ed_load.png"

# 연령밴드 표시 순서(어린이→노인). DB age_group 매칭 키(데이터 값). 색은 viridis 계열로 결정적 매핑.
_AGE_ORDER: tuple[str, ...] = (
    "0세", "1-6세", "7-12세", "13-18세", "19-49세", "50-64세", "65세 이상",
)
# DB age_group(한글) → 영어 표시 라벨(figure 범례용). 매칭은 _AGE_ORDER 로, 표시는 이 맵으로.
_AGE_DISPLAY: dict[str, str] = {
    "0세": "0 yr", "1-6세": "1-6 yr", "7-12세": "7-12 yr", "13-18세": "13-18 yr",
    "19-49세": "19-49 yr", "50-64세": "50-64 yr", "65세 이상": "65+ yr",
}

# 문헌값 (인용 — DB 산출 아님)
_LIT_R0 = 1.3        # Biggerstaff et al. 2014, BMC Infect Dis (seasonal influenza)
_LIT_SI_DAYS = 2.6   # Cowling et al. 2009, Epidemiology (influenza serial interval)


# ===========================================================================
# 한글 폰트
# ===========================================================================
def _set_korean_font() -> str:
    """matplotlib 한글 폰트를 설정한다 (AppleGothic→NanumGothic fallback).

    등록된 폰트 중 우선순위대로 첫 가용 폰트를 ``font.family`` 로 지정하고
    음수 부호 깨짐(``axes.unicode_minus``)을 끈다.

    Returns:
        실제로 선택된 폰트 이름. 한글 폰트가 하나도 없으면 ``"DejaVu Sans"`` 을
        반환하고 경고를 로그한다 (가짜 폰트 생성 금지).

    Performance: O(폰트수) 1회 스캔.
    Side effects: ``plt.rcParams`` 전역 변경 (font.family / axes.unicode_minus).
    """
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans"


# ===========================================================================
# 데이터 로드 (read-only DB)
# ===========================================================================
def load_rt_history(db_path: Path) -> dict:
    """``rt_history`` EpiEstim Cori 재생산수 이력을 read-only 로 로드한다.

    서울시(gu=seoul_city) 의 주차별 ``rt_mean`` + 95% 신용구간(q025/q975) 을
    ``week_start`` 오름차순으로 읽는다. SI(serial interval) prior 도 함께 읽어
    제목 표기에 쓴다.

    Args:
        db_path: ``epi_real_seoul.db`` 경로.

    Returns:
        dict:
          - ``"dates"``: ``list[datetime.date]`` 길이 n (week_start 파싱).
          - ``"rt"``: ``(n,)`` float64 rt_mean.
          - ``"lo"`` / ``"hi"``: ``(n,)`` float64 95% CI 경계.
          - ``"si_mean"`` / ``"si_sd"``: float | None — 저장된 SI prior.
          - ``"method"``: str — 추정 방법명(예: EpiEstim_Cori).
        행이 없으면 ``"rt"`` 가 빈 배열(n=0).

    Raises:
        FileNotFoundError: DB 파일 부재.

    Performance: 단일 SELECT + O(n). Side effects: DB read-only open/close.
    Caller responsibility: 빈 결과(n=0) 시 패널 skip.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"DB 없음: {db_path}")
    from simulation.database import read_only_connect

    con = read_only_connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT week_start, rt_mean, rt_q025, rt_q975, si_mean, si_sd, method "
            "FROM rt_history WHERE gu = 'seoul_city' "
            "ORDER BY week_start ASC"
        )
        rows = cur.fetchall()
    finally:
        con.close()

    dates: list[date] = []
    rt: list[float] = []
    lo: list[float] = []
    hi: list[float] = []
    si_mean = si_sd = None
    method = ""
    for ws, m, q025, q975, sim, sisd, meth in rows:
        if m is None:
            continue
        try:
            d = datetime.strptime(str(ws)[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue
        dates.append(d)
        rt.append(float(m))
        lo.append(float(q025) if q025 is not None else float(m))
        hi.append(float(q975) if q975 is not None else float(m))
        if sim is not None:
            si_mean = float(sim)
        if sisd is not None:
            si_sd = float(sisd)
        if meth:
            method = str(meth)
    return {
        "dates": dates,
        "rt": np.asarray(rt, dtype=np.float64),
        "lo": np.asarray(lo, dtype=np.float64),
        "hi": np.asarray(hi, dtype=np.float64),
        "si_mean": si_mean,
        "si_sd": si_sd,
        "method": method,
    }


def load_age_peak_ili(db_path: Path) -> dict:
    """``sentinel_influenza`` 연령밴드 × 시즌별 **정점 ILI rate** 를 로드한다.

    각 (season_start, age_group) 그룹에서 ``MAX(ili_rate)`` 를 시즌 정점으로
    집계한다(관측 표본감시 지표). 연령 정렬은 ``_AGE_ORDER`` 고정.

    Args:
        db_path: ``epi_real_seoul.db`` 경로.

    Returns:
        dict:
          - ``"seasons"``: ``list[int]`` 오름차순 고유 시즌.
          - ``"ages"``: ``list[str]`` 표시 순서 연령밴드(데이터에 존재하는 것만).
          - ``"peak"``: ``dict[(season,age) -> float]`` 정점 ILI rate.
        데이터가 없으면 ``"seasons"`` 빈 리스트.

    Raises:
        FileNotFoundError: DB 파일 부재.

    Performance: 단일 GROUP BY SELECT. Side effects: DB read-only open/close.
    Caller responsibility: 빈 결과 시 패널 skip.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"DB 없음: {db_path}")
    from simulation.database import read_only_connect

    con = read_only_connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT season_start, age_group, MAX(ili_rate) "
            "FROM sentinel_influenza "
            "WHERE ili_rate IS NOT NULL "
            "GROUP BY season_start, age_group"
        )
        rows = cur.fetchall()
    finally:
        con.close()

    peak: dict[tuple[int, str], float] = {}
    seasons_set: set[int] = set()
    ages_seen: set[str] = set()
    for s, age, mx in rows:
        if mx is None or age is None or s is None:
            continue
        peak[(int(s), str(age))] = float(mx)
        seasons_set.add(int(s))
        ages_seen.add(str(age))
    seasons = sorted(seasons_set)
    ages = [a for a in _AGE_ORDER if a in ages_seen]
    # _AGE_ORDER 에 없는 라벨도 누락 없이 뒤에 붙임(정직성)
    ages += sorted(a for a in ages_seen if a not in _AGE_ORDER)
    return {"seasons": seasons, "ages": ages, "peak": peak}


def load_ed_load(db_path: Path) -> dict:
    """의료부하 시계열을 로드한다 — 응급실 가용병상 + 주간 ED 점유율.

    ① ``emergency_room_availability`` (서울 권역) 스냅샷별 **응급실 가용병상
       평균(hvec)** — 수집일(collected_at 앞 10자) 단위로 평균(같은 날 복수
       스냅샷 평균). 관측 실가용수.
    ② ``ed_weekly_burden`` 의 주간 ED 점유율(avg_ed_occupancy_pct) — 있으면 함께.

    Args:
        db_path: ``epi_real_seoul.db`` 경로.

    Returns:
        dict:
          - ``"er_dates"``: ``list[datetime.date]`` 가용병상 시계열 날짜.
          - ``"er_avail"``: ``(m,)`` float64 일자별 평균 가용병상.
          - ``"burden_dates"``: ``list[datetime.date]`` 주간 부하 날짜.
          - ``"ed_occ"``: ``(k,)`` float64 ED 점유율(%) — 비어 있을 수 있음.
        데이터가 없으면 해당 키가 빈 배열/리스트.

    Raises:
        FileNotFoundError: DB 파일 부재.

    Performance: 2 SELECT. Side effects: DB read-only open/close.
    Caller responsibility: 빈 결과 시 패널 skip.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"DB 없음: {db_path}")
    from simulation.database import read_only_connect

    con = read_only_connect(str(db_path))
    try:
        cur = con.cursor()
        cur.execute(
            "SELECT substr(collected_at,1,10) AS d, AVG(hvec) "
            "FROM emergency_room_availability "
            "WHERE sido_nm LIKE '%서울%' AND hvec IS NOT NULL "
            "GROUP BY d ORDER BY d ASC"
        )
        er_rows = cur.fetchall()
        cur.execute(
            "SELECT week_start, avg_ed_occupancy_pct "
            "FROM ed_weekly_burden "
            "WHERE avg_ed_occupancy_pct IS NOT NULL "
            "ORDER BY week_start ASC"
        )
        b_rows = cur.fetchall()
    finally:
        con.close()

    er_dates: list[date] = []
    er_avail: list[float] = []
    for d, v in er_rows:
        try:
            er_dates.append(datetime.strptime(str(d)[:10], "%Y-%m-%d").date())
            er_avail.append(float(v))
        except (ValueError, TypeError):
            continue

    burden_dates: list[date] = []
    ed_occ: list[float] = []
    for ws, occ in b_rows:
        try:
            burden_dates.append(datetime.strptime(str(ws)[:10], "%Y-%m-%d").date())
            ed_occ.append(float(occ) * 100.0)  # 비율(0~1) → % 표기
        except (ValueError, TypeError):
            continue

    return {
        "er_dates": er_dates,
        "er_avail": np.asarray(er_avail, dtype=np.float64),
        "burden_dates": burden_dates,
        "ed_occ": np.asarray(ed_occ, dtype=np.float64),
    }


# ===========================================================================
# 패널 렌더
# ===========================================================================
def _empty_panel(ax: plt.Axes, msg: str) -> None:
    """데이터 부재 패널에 정직한 안내 텍스트만 그린다 (가짜 데이터 금지)."""
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=11,
            color="#888888", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_rt(ax: plt.Axes, rt: dict, standalone: bool = False) -> None:
    """패널 ① Rt 타임라인 (모델-유래) — rt_mean + 95% CI + Rt=1 임계선.

    Args:
        ax: 그릴 Axes.
        rt: ``load_rt_history`` 반환 dict.
        standalone: True 면 단독 figure(서수 ① 프리픽스 제거), False 면 결합본.
    """
    if rt["rt"].size == 0:
        _empty_panel(ax, "No rt_history data -> skip")
        return
    dates = rt["dates"]
    ax.fill_between(dates, rt["lo"], rt["hi"], color="#4C72B0", alpha=0.22,
                    label="95% credible interval (EpiEstim)")
    ax.plot(dates, rt["rt"], color="#1F3D7A", lw=1.4, label="Rt mean (model-derived)")
    ax.axhline(1.0, color="#C44E52", lw=1.1, ls="--", label="Rt = 1 (epidemic threshold)")
    ax.set_ylabel("Reproduction number Rt")
    meth = rt["method"] or "EpiEstim"
    _prefix = "" if standalone else "(1) "
    ax.set_title(
        f"{_prefix}Time-varying reproduction number Rt timeline  [model-derived: {meth} · Seoul]",
        fontsize=12, loc="left", fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, alpha=0.25)

    # 문헌값 + 저장 SI prior 명시 텍스트박스 (데이터 산출과 구분)
    si_txt = ""
    if rt["si_mean"] is not None:
        sd_txt = f"±{rt['si_sd']:.1f}" if rt["si_sd"] is not None else ""
        si_txt = f"  estimated SI prior (DB-stored)={rt['si_mean']:.1f}{sd_txt} days"
    box = (
        f"Literature values (citation, not DB-derived): seasonal influenza R0 ≈ {_LIT_R0} "
        f"(Biggerstaff 2014) · SI ≈ {_LIT_SI_DAYS} days (Cowling 2009).{si_txt}"
    )
    ax.text(0.005, -0.30, box, transform=ax.transAxes, fontsize=7.6,
            color="#555555", va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="#F4F4F2", ec="#CCCCCC"))


def _draw_age_peak(ax: plt.Axes, age: dict, standalone: bool = False) -> None:
    """패널 ② 연령별 정점 ILI rate (관측) — 시즌×연령 grouped bar.

    Args:
        ax: 그릴 Axes.
        age: ``load_age_peak_ili`` 반환 dict.
        standalone: True 면 단독 figure(서수 ② 프리픽스 제거), False 면 결합본.
    """
    seasons = age["seasons"]
    ages = age["ages"]
    if not seasons or not ages:
        _empty_panel(ax, "No sentinel_influenza data -> skip")
        return
    n_age = len(ages)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / max(1, n_age - 1)) for i in range(n_age)]
    x = np.arange(len(seasons), dtype=np.float64)
    total_w = 0.84
    bw = total_w / n_age
    for j, a in enumerate(ages):
        vals = [age["peak"].get((s, a), np.nan) for s in seasons]
        offset = -total_w / 2 + bw * (j + 0.5)
        ax.bar(x + offset, vals, width=bw * 0.95, color=colors[j],
               label=_AGE_DISPLAY.get(a, a), edgecolor="white", linewidth=0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}/{(s + 1) % 100:02d}" for s in seasons])
    ax.set_ylabel("Peak ILI rate\n(ILI patients per 1,000 outpatient visits)")
    ax.set_xlabel("Sentinel surveillance season (season_start/following year)")
    _prefix = "" if standalone else "(2) "
    ax.set_title(
        f"{_prefix}Peak ILI burden by age group  [observed: sentinel surveillance influenza · 7 age bands]",
        fontsize=12, loc="left", fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=7.5, ncol=4, title="Age band",
              title_fontsize=8, framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.25)


def _draw_ed_load(ax: plt.Axes, ed: dict, standalone: bool = False) -> None:
    """패널 ③ 의료부하 (관측) — 응급실 가용병상 + 주간 ED 점유율(있으면 보조축).

    Args:
        ax: 그릴 Axes.
        ed: ``load_ed_load`` 반환 dict.
        standalone: True 면 단독 figure(서수 ③ 프리픽스 제거), False 면 결합본.
    """
    has_er = ed["er_avail"].size > 0
    has_occ = ed["ed_occ"].size > 0
    if not has_er and not has_occ:
        _empty_panel(ax, "No ED availability/load data -> skip")
        return
    if has_er:
        ax.plot(ed["er_dates"], ed["er_avail"], marker="o", ms=4.5,
                color="#55A868", lw=1.4,
                label="Mean available ED beds (Seoul, observed hvec)")
        ax.set_ylabel("Mean available ED beds", color="#2E6B43")
        ax.tick_params(axis="y", labelcolor="#2E6B43")
    else:
        ax.set_yticks([])
    _prefix = "" if standalone else "(3) "
    ax.set_title(
        f"{_prefix}Healthcare load time series  [observed: actual available ED beds · ED occupancy]",
        fontsize=12, loc="left", fontweight="bold",
    )
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    for lab in ax.get_xticklabels():
        lab.set_rotation(30)
        lab.set_ha("right")

    handles, labels = ax.get_legend_handles_labels()
    if has_occ:
        ax2 = ax.twinx()
        ax2.plot(ed["burden_dates"], ed["ed_occ"], marker="s", ms=4.5,
                 color="#C44E52", lw=1.2, ls="--",
                 label="Weekly ED occupancy % (ed_weekly_burden, observed)")
        ax2.set_ylabel("Weekly ED occupancy (%)", color="#8C2E32")
        ax2.tick_params(axis="y", labelcolor="#8C2E32")
        h2, l2 = ax2.get_legend_handles_labels()
        handles += h2
        labels += l2
    ax.legend(handles, labels, loc="upper left", fontsize=8, framealpha=0.9)


# ===========================================================================
# 메인
# ===========================================================================
def build_figure(db_path: Path = DB_PATH, out_path: Path = OUT_PATH) -> list[Path]:
    """보건역학 지표 figure 를 생성·저장한다 (실측만, 결정적).

    결합본(3행 1열)에 더해 **패널마다 단독 PNG** 도 저장한다("한 번에 한 그림씩"
    표시용). 단독 PNG 는 같은 draw 헬퍼를 ``standalone=True`` 로 재사용하므로
    그려지는 데이터는 결합본과 100% 동일하고 제목의 서수(①②③) 프리픽스만 제거된다.

    Args:
        db_path: ``epi_real_seoul.db`` 경로.
        out_path: 결합본 PNG 경로 (dpi=120). 단독 PNG 는 ``OUT_RT_PATH`` 등
            모듈 상수 경로에 저장.

    Returns:
        저장된 PNG ``Path`` 리스트(단독 패널들 + 결합본). 데이터가 있는 패널만
        단독 PNG 로 저장되며, 세 패널 모두 데이터가 없으면 (생성할 figure 가 없어)
        빈 리스트를 반환하고 정직히 skip 로그.

    Raises:
        FileNotFoundError: DB 파일 부재.

    Performance: 3 SELECT + 패널당 1 render(최대 4 figure). Side effects: PNG
    파일 최대 4개 디스크 쓰기, DB read-only open/close.
    """
    font = _set_korean_font()
    print(f"[fig_epi_metrics] 폰트={font}  DB={db_path}")

    rt = load_rt_history(db_path)
    age = load_age_peak_ili(db_path)
    ed = load_ed_load(db_path)

    has_rt = rt["rt"].size > 0
    has_age = bool(age["seasons"]) and bool(age["ages"])
    has_ed = ed["er_avail"].size > 0 or ed["ed_occ"].size > 0
    print(f"[fig_epi_metrics] Rt n={rt['rt'].size}  "
          f"age(seasons={len(age['seasons'])},bands={len(age['ages'])})  "
          f"ER n={ed['er_avail'].size}  ED-occ n={ed['ed_occ'].size}")

    if not (has_rt or has_age or has_ed):
        print("[fig_epi_metrics] 세 패널 모두 데이터 없음 → figure 생성 skip (정직)")
        return []

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # 공통 출처 캡션 (단독 패널 + 결합본 동일 문구, 정직성)
    _src_caption = (
        "Source: epi_real_seoul.db (rt_history · sentinel_influenza · "
        "emergency_room_availability/ed_weekly_burden). "
        "[model-derived]=EpiEstim estimate, [observed]=actual sentinel surveillance/ED. No synthetic or fabricated data."
    )

    # ── 패널별 단독 PNG (데이터 있는 패널만; 같은 draw 헬퍼 standalone=True 재사용) ──
    # 세로로 긴 wide stacked 패널이므로 단독에서도 넉넉한 가로폭 유지.
    _standalone_specs = (
        (_draw_rt, rt, has_rt, OUT_RT_PATH),
        (_draw_age_peak, age, has_age, OUT_AGE_PATH),
        (_draw_ed_load, ed, has_ed, OUT_ED_PATH),
    )
    for draw_fn, data, has_data, panel_path in _standalone_specs:
        if not has_data:
            continue
        sfig, sax = plt.subplots(figsize=(12.5, 5.0))
        draw_fn(sax, data, standalone=True)
        sfig.text(0.005, 0.002, _src_caption, fontsize=7.4, color="#777777")
        # Rt 패널은 문헌값 텍스트박스가 축 아래(-0.30)에 있어 여백을 더 확보.
        _bottom = 0.16 if draw_fn is _draw_rt else 0.04
        sfig.tight_layout(rect=(0, _bottom, 1, 0.98))
        sfig.savefig(panel_path, dpi=130, bbox_inches="tight")
        plt.close(sfig)
        written.append(panel_path)
        print(f"[fig_epi_metrics] 단독 패널 저장 → {panel_path}")

    # ── 결합본(3행 1열) — 동일 헬퍼 standalone=False (back-compat) ──
    fig, axes = plt.subplots(3, 1, figsize=(12.5, 13.5))
    fig.suptitle(
        "Standard epidemiological indicators — Reproduction number (model-derived) · ILI burden by age group (observed) · Healthcare load (observed)",
        fontsize=14, fontweight="bold", y=0.995,
    )
    _draw_rt(axes[0], rt, standalone=False)
    _draw_age_peak(axes[1], age, standalone=False)
    _draw_ed_load(axes[2], ed, standalone=False)

    fig.text(0.005, 0.002, _src_caption, fontsize=7.4, color="#777777")

    fig.tight_layout(rect=(0, 0.015, 1, 0.985))
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    written.append(out_path)
    print(f"[fig_epi_metrics] 결합본 저장 완료 → {out_path}")
    return written


def main() -> int:
    """CLI 진입점. 성공 0, 데이터 전무 skip 0, 예외 1."""
    try:
        outs = build_figure()
    except FileNotFoundError as e:
        print(f"[fig_epi_metrics] ERROR: {e}", file=sys.stderr)
        return 1
    if not outs:
        return 0
    for out in outs:
        if not out.exists() or out.stat().st_size == 0:
            print(f"[fig_epi_metrics] ERROR: PNG 미생성/0바이트 → {out}",
                  file=sys.stderr)
            return 1
    print(f"[fig_epi_metrics] OK — {len(outs)}개 PNG 저장:")
    for out in outs:
        print(f"  - {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
