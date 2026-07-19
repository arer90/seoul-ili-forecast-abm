"""서울 자치구별(25구) 감염병 figure — 관측 전수감시(2020-2024) + 모델-유래 ILI.

이 스크립트는 **두 종류의 자치구 데이터를 명확히 구분**하여 1개 PNG(5 패널)로
시각화한다. 절대 합성/가짜 데이터를 만들지 않으며, 모델-유래 값을 관측인 것처럼
표기하지 않는다.

데이터 출처 (실재 검증):
  관측 (Panel A·B·C·D) — DB ``seoul_disease_district`` (전수감시 연간, 2020-2024):
    70개 법정감염병 × 25개 자치구 × 5년 × category(발생/사망 × 계/남/여).
    헤드라인 질병 = **제2급감염병**(법정 제2급 신고 집계) — 전 5년 25구 발생_계
    풍부, 발생_남/여(2023-24)·사망_계(4년)까지 보유. 인구 정규화는
    ``kosis_age_district`` (자치구 연령별 주민등록 인구 합)로 10만명당 발생률 계산.
    ※ 2022년은 COVID-19 대유행으로 제2급 발생이 급증(엔데믹 연도의 ~350×) —
      숨기지 않고 log y축 + 주석으로 정직하게 표시.
  모델 (Panel E) — ``web/public/aggregates/ili-local.json`` (key ``gu``):
    도시 표본감시 ILI를 검증된 ABM I_frac 공간패턴으로 25구에 분배한 **모델 산출물**.
    구별 직접 관측이 아니며, snapshot 단일 시점값(주간).

규율: matplotlib Agg + 한글폰트(AppleGothic→NanumGothic fallback). 결정성(정렬·
고정 색상·고정 연도순). 데이터 부재 시 가짜 채우지 않고 정직하게 패널 skip + 사유
표기. read_only_connect (LOCK-FREE read-only) 사용 — 저수준 직접 연결 금지(G-116/117).
"""

from __future__ import annotations

import json
import os
from contextlib import closing
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display backend
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

from simulation.database import read_only_connect  # noqa: E402

# ----------------------------------------------------------------------------
# 경로 (project-relative, OS-비종속)
# ----------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parents[2]  # .../MPH_infection_simulation
ILI_LOCAL_JSON = PROJECT_ROOT / "web" / "public" / "aggregates" / "ili-local.json"
FIG_DIR = PROJECT_ROOT / "simulation" / "results" / "figures"
OUT_PNG = FIG_DIR / "fig_district_seoul.png"

# 단일 패널 PNG (의미이름) — 합본 6패널이 읽기 어려워 "한 번에 하나씩" 보기 위함.
# 합본(OUT_PNG)은 back-compat 으로 그대로 유지.
PANEL_PNGS = {
    "trend": FIG_DIR / "district_seoul_trend.png",
    "ranking": FIG_DIR / "district_seoul_ranking.png",
    "cfr": FIG_DIR / "district_seoul_cfr.png",
    "sex": FIG_DIR / "district_seoul_sex.png",
    "model_ili": FIG_DIR / "district_seoul_model_ili.png",
    "datasource_note": FIG_DIR / "district_seoul_datasource_note.png",
}

# 헤드라인 관측 질병 = 제2급감염병 (전 5년 25구 발생_계 풍부 + 남/여 + 사망).
# (COVID-19 단독은 2022-2023 2년만 존재 → 5년 추세 부적합)
OBSERVED_DISEASE_NM = "제2급감염병"
ALL_YEARS = (2020, 2021, 2022, 2023, 2024)
PANDEMIC_YEAR = 2022  # COVID-19 대유행으로 제2급 발생 급증 — 주석 대상

# 대표 자치구 (지리·인구 대표성 — 강남/영등포/종로/송파/노원/관악 + 강서/마포)
PANEL_REPRESENTATIVE = [
    "강남구",
    "영등포구",
    "종로구",
    "송파구",
    "노원구",
    "관악구",
    "강서구",
    "마포구",
]

# Seoul 25 districts: Korean gu name -> romanized English (figure tick labels;
# DB values stay Korean for matching).
_GU_ENG: dict[str, str] = {
    "종로구": "Jongno", "중구": "Jung", "용산구": "Yongsan", "성동구": "Seongdong",
    "광진구": "Gwangjin", "동대문구": "Dongdaemun", "중랑구": "Jungnang",
    "성북구": "Seongbuk", "강북구": "Gangbuk", "도봉구": "Dobong", "노원구": "Nowon",
    "은평구": "Eunpyeong", "서대문구": "Seodaemun", "마포구": "Mapo",
    "양천구": "Yangcheon", "강서구": "Gangseo", "구로구": "Guro",
    "금천구": "Geumcheon", "영등포구": "Yeongdeungpo", "동작구": "Dongjak",
    "관악구": "Gwanak", "서초구": "Seocho", "강남구": "Gangnam", "송파구": "Songpa",
    "강동구": "Gangdong",
}


def _gu_eng(name: str) -> str:
    """Romanize a Seoul district name (append '-gu'); pass through if unknown."""
    base = _GU_ENG.get(str(name))
    return f"{base}-gu" if base else str(name)


# 색상 (결정성 — 대표 구 강조용 고정 팔레트)
COLOR_OBSERVED = "#C44E52"  # 관측 강조(빨강 계열)
COLOR_MODEL = "#4C72B0"  # 모델(파랑 계열)
COLOR_OTHER = "#B8B8B8"  # 비대표 구(회색)


def _setup_korean_font() -> str:
    """한글 깨짐 방지 폰트를 rcParams 에 설정.

    Returns:
        선택된 폰트 family 이름. 후보가 모두 없으면 ``"DejaVu Sans"`` (한글은
        깨지지만 figure 생성 자체는 진행).

    Side effects: ``matplotlib.rcParams['font.family']`` 및
        ``axes.unicode_minus`` 를 전역 변경한다.
    """
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans"


# ----------------------------------------------------------------------------
# 데이터 로딩 (read_only_connect — DB 1회 연결로 전부)
# ----------------------------------------------------------------------------
def load_observed() -> dict[str, dict]:
    """seoul_disease_district 에서 제2급감염병 관측 집계를 한 번에 로드.

    read_only_connect (LOCK-FREE) — 저수준 직접 연결 금지(G-116/117).

    Returns:
        ``{"incidence": {구: {연도: cases}}, "death": {구: {연도: cases}},``
        ``"sex_m": {구: {연도: cases}}, "sex_f": {구: {연도: cases}}}``.
        '서울시' 합계 행과 NULL/cases 결측은 제외. 테이블 부재 시 4개 전부 빈
        dict (가짜 채우지 않음 → 호출자 정직 skip).

    Side effects: DB read-only 연결 1회 (closing 자동 close).

    Performance: O(자치구×연도×category) 작은 집계 — 수십 ms.
    """
    out = {"incidence": {}, "death": {}, "sex_m": {}, "sex_f": {}}
    cat_map = {
        "발생_계": "incidence",
        "사망_계": "death",
        "발생_남": "sex_m",
        "발생_여": "sex_f",
    }
    with closing(read_only_connect()) as con:
        cur = con.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='seoul_disease_district'"
        )
        if cur.fetchone() is None:
            return out
        cur.execute(
            """
            SELECT category, gu_nm, year, cases
            FROM seoul_disease_district
            WHERE disease_nm = ?
              AND category IN ('발생_계', '사망_계', '발생_남', '발생_여')
              AND gu_nm != '서울시' AND cases IS NOT NULL
            ORDER BY category, gu_nm, year
            """,
            (OBSERVED_DISEASE_NM,),
        )
        for category, gu_nm, year, cases in cur.fetchall():
            bucket = cat_map.get(category)
            if bucket is None:
                continue
            out[bucket].setdefault(gu_nm, {})[int(year)] = int(cases)
    return out


def load_population() -> dict[str, dict[int, int]]:
    """kosis_age_district 에서 자치구×연도 총 주민등록 인구(연령 합)를 로드.

    Returns:
        ``{구이름: {연도: 총인구}}``. 합계/서울특별시 행은 제외. 테이블 부재 시
        빈 dict → 호출자는 rate 정규화 포기하고 raw cases 로 정직 fallback.

    Side effects: DB read-only 연결 1회.
    """
    out: dict[str, dict[int, int]] = {}
    with closing(read_only_connect()) as con:
        cur = con.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='kosis_age_district'"
        )
        if cur.fetchone() is None:
            return {}
        cur.execute(
            """
            SELECT gu_nm, prd_de, SUM(population)
            FROM kosis_age_district
            WHERE gu_nm NOT IN ('서울특별시', '합계', '서울시')
              AND population IS NOT NULL
            GROUP BY gu_nm, prd_de
            """
        )
        for gu_nm, prd_de, pop in cur.fetchall():
            try:
                yr = int(prd_de)
            except (TypeError, ValueError):
                continue
            out.setdefault(gu_nm, {})[yr] = int(pop)
    return out


def load_modeled_ili() -> tuple[dict[str, float], float | None]:
    """ili-local.json 에서 자치구별 모델-유래 ILI snapshot + q70 임계값을 로드.

    Returns:
        ``({구: ili값}, q70_중앙값)``. 파일/키 부재 시 ``({}, None)``
        (가짜 채우지 않음 — 정직 skip 유도). q70 은 구별 미세 차이가 있어
        중앙값을 대표값으로 사용(합성 아님, 실제 모델 산출 통계량).

    Side effects: 디스크에서 JSON 1회 읽기.
    """
    if not ILI_LOCAL_JSON.exists():
        return {}, None
    with open(ILI_LOCAL_JSON, encoding="utf-8") as fh:
        payload = json.load(fh)
    gu = payload.get("gu", {})
    ili = {k: float(v["ili"]) for k, v in gu.items() if "ili" in v}
    q70s = sorted(float(v["q70"]) for v in gu.values() if "q70" in v)
    q70 = q70s[len(q70s) // 2] if q70s else None
    return ili, q70


# ----------------------------------------------------------------------------
# 패널 (각 함수: 데이터 있으면 그리고 True, 부재면 skip-text 후 False)
# ----------------------------------------------------------------------------
def _rate(cases: int | None, pop: int | None) -> float | None:
    """10만명당 발생률. cases/pop 둘 다 유효해야 산출, 아니면 None.

    Args:
        cases: 발생/사망 건수 (음수 없음 가정).
        pop: 주민등록 인구.

    Returns:
        ``cases/pop*100000`` 또는 None.
    """
    if cases is None or not pop:
        return None
    return cases / pop * 100_000


def _panel_trend(ax, incidence, pop, standalone: bool = False) -> bool:
    """Panel A: 대표 자치구 제2급감염병 발생률(10만명당) 2020-2024 추세 (log-y).

    Args:
        ax: matplotlib Axes.
        incidence: ``{구: {연도: cases}}`` 발생_계.
        pop: ``{구: {연도: 인구}}`` (빈 dict 면 raw cases 추세로 fallback).
        standalone: True 면 단일 PNG 용(제목에서 'A.' 서수 접두 제거).

    Returns:
        True=그림 성공, False=데이터 부재 skip.

    정직성: 2022 COVID 대유행 spike 를 숨기지 않고 log y축으로 전 5년 표시 +
        수직 점선·주석. 인구 정규화 가능하면 10만명당, 불가면 raw 로 정직 표기.
    """
    if not incidence:
        ax.text(0.5, 0.5, "No observed incidence data (skip)", ha="center", va="center")
        ax.set_axis_off()
        return False
    use_rate = bool(pop)
    rep = [g for g in PANEL_REPRESENTATIVE if g in incidence]
    cmap = plt.get_cmap("tab10")
    for i, gu in enumerate(rep):
        xs, ys = [], []
        for yr in ALL_YEARS:
            c = incidence[gu].get(yr)
            v = _rate(c, pop.get(gu, {}).get(yr)) if use_rate else (
                float(c) if c is not None else None
            )
            if v is not None and v > 0:  # log축: 0/None 은 점 생략(가짜 0 금지)
                xs.append(yr)
                ys.append(v)
        if xs:
            ax.plot(
                xs, ys, marker="o", ms=5, lw=1.7, label=_gu_eng(gu), color=cmap(i),
                zorder=3,
            )
    ax.set_yscale("log")
    ax.set_xticks(list(ALL_YEARS))
    ax.set_xticklabels([str(y) for y in ALL_YEARS])
    ax.axvline(PANDEMIC_YEAR, color="#999999", ls="--", lw=1.0, zorder=1)
    ax.annotate(
        "2022 COVID-19\npandemic (Group-2 surge)",
        xy=(PANDEMIC_YEAR, ax.get_ylim()[1]),
        xytext=(PANDEMIC_YEAR - 0.05, ax.get_ylim()[1]),
        ha="center",
        va="top",
        fontsize=7,
        color="#777777",
    )
    unit = "Incidence per 100k (log)" if use_rate else "Incidence cases (raw, log)"
    ax.set_ylabel(unit, fontsize=9)
    ax.set_xlabel("Year", fontsize=9)
    ax.grid(alpha=0.3, which="both", zorder=0)
    ax.legend(fontsize=6.8, ncol=2, loc="upper left", framealpha=0.85)
    prefix = "" if standalone else "A. "
    ax.set_title(
        f"{prefix}Group-2 notifiable disease incidence trend, representative districts (2020-2024)\n"
        "— observed full surveillance, annual · log axis (2022 pandemic shown honestly)",
        fontsize=9.5,
        loc="left",
    )
    return True


def _panel_ranking(ax, incidence, pop, standalone: bool = False) -> bool:
    """Panel B: 최신 연도(2024) 25개 자치구 제2급 발생률 ranking.

    Args:
        ax: matplotlib Axes.
        incidence: ``{구: {연도: cases}}`` 발생_계.
        pop: ``{구: {연도: 인구}}`` (빈 dict 면 raw cases ranking).
        standalone: True 면 단일 PNG 용(제목에서 'B.' 서수 접두 제거).

    Returns:
        True=성공, False=skip.

    정직성: 한 구라도 인구 결측이면 전체 raw 로 통일(혼합 단위 금지). 대표 구는
        강조색. 최신 연도 자동 선택(2024 우선).
    """
    if not incidence:
        ax.text(0.5, 0.5, "No observed incidence data (skip)", ha="center", va="center")
        ax.set_axis_off()
        return False
    years = sorted({y for d in incidence.values() for y in d})
    latest = years[-1]
    use_rate = bool(pop)

    rows: list[tuple[str, float]] = []
    for gu, ydict in incidence.items():
        if latest not in ydict:
            continue
        if use_rate:
            r = _rate(ydict[latest], pop.get(gu, {}).get(latest))
            if r is None:  # 한 구라도 인구 결측 → 전체 raw 통일
                use_rate = False
                break
            rows.append((gu, r))
        else:
            rows.append((gu, float(ydict[latest])))
    if not use_rate:  # raw 로 재구성
        rows = [
            (gu, float(yd[latest])) for gu, yd in incidence.items() if latest in yd
        ]

    rows.sort(key=lambda x: x[1], reverse=True)  # 결정성 + 높은 구 위로
    gus = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [COLOR_OBSERVED if g in PANEL_REPRESENTATIVE else COLOR_OTHER for g in gus]
    ax.barh(range(len(gus)), vals, color=colors, zorder=3)
    ax.set_yticks(range(len(gus)))
    ax.set_yticklabels([_gu_eng(g) for g in gus], fontsize=7)
    ax.invert_yaxis()  # 1위가 위
    unit = "Incidence per 100k" if use_rate else "Incidence cases (raw)"
    ax.set_xlabel(unit, fontsize=9)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    norm_note = (
        "population-normalized (KOSIS registry)" if use_rate else "no population data -> raw (honest)"
    )
    prefix = "" if standalone else "B. "
    ax.set_title(
        f"{prefix}Group-2 notifiable disease incidence ranking by district, {latest}\n"
        f"— observed full surveillance · ({norm_note}) · representative districts = red",
        fontsize=9.5,
        loc="left",
    )
    return True


def _panel_cfr(ax, incidence, death, standalone: bool = False) -> bool:
    """Panel C: 제2급감염병 치명률(CFR=사망_계/발생_계) 연도별 (서울 전체).

    Args:
        ax: matplotlib Axes.
        incidence: ``{구: {연도: cases}}`` 발생_계.
        death: ``{구: {연도: cases}}`` 사망_계.
        standalone: True 면 단일 PNG 용(제목에서 'C.' 서수 접두 제거).

    Returns:
        True=성공, False=skip(사망 데이터 부재).

    정직성: 2022 는 사망_계 결측(전수감시 미집계)이라 CFR 산출 불가 → 해당 연도
        막대 생략(가짜 0 금지) + 주석. 구별 합산해 서울 전체 CFR 만 표시(소표본 구별
        CFR 은 불안정).
    """
    if not death:
        ax.text(0.5, 0.5, "No observed death data (skip)", ha="center", va="center")
        ax.set_axis_off()
        return False
    # 서울 전체 = 전 자치구 합 (연도별)
    inc_tot = {y: 0 for y in ALL_YEARS}
    dth_tot = {y: 0 for y in ALL_YEARS}
    inc_has = {y: False for y in ALL_YEARS}
    dth_has = {y: False for y in ALL_YEARS}
    for yd in incidence.values():
        for y, c in yd.items():
            if y in inc_tot:
                inc_tot[y] += c
                inc_has[y] = True
    for yd in death.values():
        for y, c in yd.items():
            if y in dth_tot:
                dth_tot[y] += c
                dth_has[y] = True

    xs, cfr, deaths = [], [], []
    for y in ALL_YEARS:
        if inc_has[y] and dth_has[y] and inc_tot[y] > 0:
            xs.append(y)
            cfr.append(dth_tot[y] / inc_tot[y] * 100.0)
            deaths.append(dth_tot[y])
    if not xs:
        ax.text(0.5, 0.5, "No year with computable CFR (skip)", ha="center", va="center")
        ax.set_axis_off()
        return False

    bars = ax.bar(
        range(len(xs)), cfr, color="#55A868", width=0.6, zorder=3
    )
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([str(y) for y in xs])
    ax.set_ylabel("Case fatality rate CFR (%)", fontsize=9)
    ax.set_xlabel("Year", fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    for b, c, d in zip(bars, cfr, deaths):
        ax.text(
            b.get_x() + b.get_width() / 2,
            c,
            f"{c:.2f}%\n(deaths {d:,})",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    ax.set_ylim(0, max(cfr) * 1.30 if cfr else 1)
    missing = [str(y) for y in ALL_YEARS if y not in xs]
    miss_txt = (
        f" · {', '.join(missing)} omitted (deaths not collected; not a fake 0)" if missing else ""
    )
    prefix = "" if standalone else "C. "
    ax.set_title(
        f"{prefix}Group-2 notifiable disease case fatality rate (CFR) by year — Seoul total\n"
        f"— observed deaths / incidence{miss_txt}",
        fontsize=9.5,
        loc="left",
    )
    return True


def _panel_sex(ax, sex_m, sex_f, pop, standalone: bool = False) -> bool:
    """Panel D: 대표 자치구 제2급 성별 발생률(10만명당), 최신 성별데이터 연도.

    Args:
        ax: matplotlib Axes.
        sex_m: ``{구: {연도: 발생_남}}``.
        sex_f: ``{구: {연도: 발생_여}}``.
        pop: ``{구: {연도: 인구}}`` (없으면 raw cases).
        standalone: True 면 단일 PNG 용(제목에서 'D.' 서수 접두 제거).

    Returns:
        True=성공, False=skip(성별 데이터 부재).

    정직성: 성별 발생은 일부 연도만 집계(2023-24) → 두 성별 모두 존재하는 최신
        연도 자동 선택. 인구는 자치구 전체(성별 인구 분해 미보유 → 동일 분모,
        주석 명시). 남/여 grouped bar.
    """
    if not sex_m or not sex_f:
        ax.text(0.5, 0.5, "No observed sex-disaggregated data (skip)", ha="center", va="center")
        ax.set_axis_off()
        return False
    # 남·여 둘 다 존재하는 최신 연도
    common_years = sorted(
        {y for d in sex_m.values() for y in d} & {y for d in sex_f.values() for y in d}
    )
    if not common_years:
        ax.text(0.5, 0.5, "No common male/female year (skip)", ha="center", va="center")
        ax.set_axis_off()
        return False
    yr = common_years[-1]
    use_rate = bool(pop)

    rep = [g for g in PANEL_REPRESENTATIVE if g in sex_m and g in sex_f]
    rep = [g for g in rep if yr in sex_m[g] and yr in sex_f[g]]
    if not rep:
        ax.text(0.5, 0.5, "No sex data for representative districts (skip)", ha="center", va="center")
        ax.set_axis_off()
        return False

    m_vals, f_vals = [], []
    for g in rep:
        p = pop.get(g, {}).get(yr) if use_rate else None
        if use_rate and not p:
            use_rate = False  # 한 구라도 인구 결측 → 전체 raw 통일
        m_vals.append(_rate(sex_m[g][yr], p) if use_rate else float(sex_m[g][yr]))
        f_vals.append(_rate(sex_f[g][yr], p) if use_rate else float(sex_f[g][yr]))
    if not use_rate:  # raw 재구성
        m_vals = [float(sex_m[g][yr]) for g in rep]
        f_vals = [float(sex_f[g][yr]) for g in rep]

    x = range(len(rep))
    w = 0.4
    ax.bar([i - w / 2 for i in x], m_vals, w, label="Male", color="#4C72B0", zorder=3)
    ax.bar([i + w / 2 for i in x], f_vals, w, label="Female", color="#DD8452", zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels([_gu_eng(g) for g in rep], rotation=30, ha="right", fontsize=8)
    unit = "Incidence per 100k" if use_rate else "Incidence cases (raw)"
    ax.set_ylabel(unit, fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    ax.legend(fontsize=8, loc="upper right")
    pop_note = (
        "denominator = total district population (no sex breakdown)" if use_rate else "no population -> raw"
    )
    prefix = "" if standalone else "D. "
    ax.set_title(
        f"{prefix}Group-2 notifiable disease incidence by sex, {yr} — representative districts\n"
        f"— observed male/female incidence · ({pop_note})",
        fontsize=9.5,
        loc="left",
    )
    return True


def _panel_model_ili(ax, ili, q70, standalone: bool = False) -> bool:
    """Panel E: 자치구별 모델-유래 ILI snapshot (대표 구 막대) — 관측 아님.

    Args:
        ax: matplotlib Axes.
        ili: ``{구: ili}`` 모델 산출 snapshot.
        q70: 경보 임계값(있으면 주석), 없으면 None.
        standalone: True 면 단일 PNG 용(제목에서 'E.' 서수 접두 제거).

    Returns:
        True=성공, False=skip.

    정직성: 제목/주석에 '모델 공간분배, 구별 관측 아님' 명시. 색상도 관측 패널과
        구분(파랑). y축 0 시작 → 구간 ILI 가 거의 균일함을 과장 없이 표시.
    """
    rep = [g for g in PANEL_REPRESENTATIVE if g in ili]
    if not rep:
        ax.text(0.5, 0.5, "No ILI model data (skip)", ha="center", va="center")
        ax.set_axis_off()
        return False
    vals = [ili[g] for g in rep]
    bars = ax.bar(range(len(rep)), vals, color=COLOR_MODEL, width=0.62, zorder=3)
    ax.set_xticks(range(len(rep)))
    ax.set_xticklabels([_gu_eng(g) for g in rep], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Model ILI (%, snapshot)", fontsize=9)
    ax.set_ylim(0, max(vals) * 1.18 if max(vals) > 0 else 1)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=7.5,
        )
    spread = max(ili.values()) - min(ili.values())
    q70_txt = (
        f" · alert threshold ~q70={q70:.1f}% (off axis, no alert)" if q70 is not None else ""
    )
    prefix = "" if standalone else "E. "
    ax.set_title(
        f"{prefix}District-level model-derived ILI (snapshot)\n"
        "— city ILI x validated spatial-weight distribution, not per-district observation",
        fontsize=9.5,
        loc="left",
    )
    ax.text(
        0.0,
        -0.40,
        f"Inter-district ILI spread = {spread:.4f}%p (nearly uniform: flat ABM I_frac spatial pattern){q70_txt}\n"
        "(i) Model-derived — differs in data type/precision from A/B/C/D (observed full surveillance); not directly comparable",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#555555",
    )
    return True


def _panel_datasource_note(ax) -> bool:
    """6번째 셀: 데이터 출처·정직성 텍스트 박스 (그래프 아님).

    Args:
        ax: matplotlib Axes. 축은 끄고 텍스트만 렌더.

    Returns:
        항상 True (텍스트 박스 — 데이터 부재 개념 없음).

    정직성: 관측(A·B·C·D)과 모델(E)의 데이터 종류·정밀도가 달라 직접 비교
        불가함을 명시. 합본/단일 PNG 둘 다 동일 텍스트 사용.
    """
    ax.set_axis_off()
    ax.text(
        0.0,
        0.98,
        "Data sources · Honesty",
        fontsize=11,
        fontweight="bold",
        va="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.0,
        0.86,
        "Observed (A/B/C/D) — DB seoul_disease_district\n"
        "  · Group-2 notifiable disease, full surveillance, annual (2020-2024)\n"
        "  · 70 diseases x 25 districts x 5 years x incidence/death, total/male/female\n"
        "  · population normalization = kosis_age_district registry\n"
        "  · 2022 = COVID-19 pandemic surge of Group-2 diseases\n"
        "    (not hidden; shown with log axis + annotation)\n\n"
        "Model (E) — web/.../ili-local.json\n"
        "  · city ILI distributed across 25 districts via ABM I_frac\n"
        "    spatial pattern (model output, not per-district observation)\n\n"
        "· The two data types differ in kind and precision,\n"
        "  not directly comparable — distinguished by color\n"
        "  (observed = red/green, model = blue)\n"
        "· Panels/years with no data are honestly omitted,\n"
        "  with no fake 0",
        fontsize=8.5,
        va="top",
        transform=ax.transAxes,
        color="#333333",
    )
    return True


def _save_standalone(name: str, draw, *draw_args) -> Path:
    """단일 패널을 자체 figure 로 렌더해 PNG 1개 저장 (한 번에 하나씩 보기용).

    Args:
        name: ``PANEL_PNGS`` 키 (출력 경로 lookup).
        draw: ``_panel_*`` 헬퍼. ``draw(ax, *draw_args, standalone=...)`` 호출.
        draw_args: 헬퍼에 전달할 데이터 인자.

    Returns:
        저장된 PNG 경로.

    Side effects: 디스크에 PNG 1개(dpi=130, bbox_inches='tight') 저장 후 figure
        close. note 패널은 ``standalone`` 키워드를 받지 않으므로 조건 분기.
    """
    out = PANEL_PNGS[name]
    fig, ax = plt.subplots(figsize=(8, 6))
    if name == "datasource_note":
        draw(ax, *draw_args)
    else:
        draw(ax, *draw_args, standalone=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def build_figure() -> list[Path]:
    """6개 패널을 각각 단일 PNG 로 저장 + 합본 6패널 PNG(back-compat) 저장.

    사용자 요청: 합본 6패널이 읽기 어려워 패널을 "한 번에 하나씩" 보고 싶음 →
    각 패널을 의미이름 단일 PNG 로 분리 저장. 합본(OUT_PNG)도 그대로 유지.

    Returns:
        저장된 PNG 경로 리스트 — 단일 6개(``PANEL_PNGS`` 순서) + 합본 1개.

    Side effects: 한글폰트 rcParams 전역 설정, DB read-only 2회(관측·인구), JSON
        1회, 디스크에 PNG 7개(단일 dpi=130 + 합본 dpi=120) 저장. 출력 디렉터리
        없으면 생성.

    Performance: O(자치구×연도×category) 작은 집계 — 1초 미만.

    Raises:
        RuntimeError: 저장된 합본 PNG 가 0 bytes (생성 실패).
    """
    font = _setup_korean_font()
    print(f"[font] 사용 폰트: {font}")

    obs = load_observed()
    pop = load_population()
    ili, q70 = load_modeled_ili()

    print(
        f"[data] 관측 발생 구={len(obs['incidence'])} · 사망 구={len(obs['death'])} · "
        f"성별(남/여) 구={len(obs['sex_m'])}/{len(obs['sex_f'])} · "
        f"인구 구={len(pop)} · 모델 ILI 구={len(ili)} · q70={q70}"
    )

    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ── 단일 패널 PNG (한 번에 하나씩) — 동일 _panel_* 헬퍼, standalone=True ──
    written: list[Path] = []
    p_trend = _save_standalone("trend", _panel_trend, obs["incidence"], pop)
    p_ranking = _save_standalone("ranking", _panel_ranking, obs["incidence"], pop)
    p_cfr = _save_standalone("cfr", _panel_cfr, obs["incidence"], obs["death"])
    p_sex = _save_standalone("sex", _panel_sex, obs["sex_m"], obs["sex_f"], pop)
    p_model = _save_standalone("model_ili", _panel_model_ili, ili, q70)
    p_note = _save_standalone("datasource_note", _panel_datasource_note)
    written.extend([p_trend, p_ranking, p_cfr, p_sex, p_model, p_note])
    for p in written:
        print(f"[ok] 단일 패널 저장: {p} ({os.path.getsize(p):,} bytes)")

    # ── 합본 6패널 PNG (back-compat) — 동일 _panel_* 헬퍼, standalone=False ──
    # 2x3 그리드: A B C / D E (관측) + (모델). 마지막 셀은 정직성 노트 공간.
    fig, axes = plt.subplots(2, 3, figsize=(19, 11.5), constrained_layout=True)
    ok_a = _panel_trend(axes[0][0], obs["incidence"], pop)
    ok_b = _panel_ranking(axes[0][1], obs["incidence"], pop)
    ok_c = _panel_cfr(axes[0][2], obs["incidence"], obs["death"])
    ok_d = _panel_sex(axes[1][0], obs["sex_m"], obs["sex_f"], pop)
    ok_e = _panel_model_ili(axes[1][1], ili, q70)
    _panel_datasource_note(axes[1][2])  # 마지막 셀 = 데이터 출처/정직성 텍스트 박스

    fig.suptitle(
        "Seoul notifiable disease by district: observed full surveillance 2020-2024 (A/B/C/D) vs model-derived ILI (E)",
        fontsize=14,
        fontweight="bold",
    )
    fig.get_layout_engine().set(rect=(0.0, 0.0, 1.0, 0.96))

    fig.savefig(OUT_PNG, dpi=120)
    plt.close(fig)
    written.append(OUT_PNG)

    size = os.path.getsize(OUT_PNG)
    print(f"[ok] panels A/B/C/D/E = {ok_a}/{ok_b}/{ok_c}/{ok_d}/{ok_e}")
    print(f"[ok] 합본 저장: {OUT_PNG} ({size:,} bytes)")
    if size == 0:
        raise RuntimeError("PNG 생성 실패: 0 bytes")
    return written


if __name__ == "__main__":
    paths = build_figure()
    print(f"[done] 총 {len(paths)}개 PNG 저장 (단일 6 + 합본 1):")
    for p in paths:
        print(f"  - {p}")
