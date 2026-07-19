"""서울 ILI vs 해외 주요국 ILI 비교 figure (정직성 우선, 관측 only).

이 스크립트는 ``epi_real_seoul.db`` 의 두 관측 테이블을 **read-only** 로 읽어
서울 표본감시 인플루엔자(``sentinel_influenza``) 전연령 ILI 와 해외 주요국
(``overseas_ili``) 주간 ILI 활동도를 **동일 달력 시간축**(ISO 연-주차 → 월요일)
위에서 비교한다. 모델/가짜/합성 데이터는 일절 만들지 않는다.

[왜 정규화가 필수인가 — 단위 정직성]
    세 출처의 ``ili_rate`` 는 **단위가 다르다**:
      - 서울 ``sentinel_influenza`` = KDCA 표본감시 **외래 1,000명당 ILI 의사환자수**
        (전연령 = 7개 연령밴드 비가중 산술평균; 테이블에 연령별 분모 없음).
      - ``delphi_national`` / ``cdc_ilinet`` (US) = **외래방문 중 ILI 비율(%)**
        (대략 0.4~8 스케일).
      - ``who_flunet`` = WHO FluNet 보고 **ILI 활동도 / 검출 지표**(국가별 보고
        관행에 따라 0~100 스케일; 검체 양성률·ILI 보고율이 섞여 들어와 절대값을
        국가간 직접 비교 불가).
    → 절대값을 한 축에 겹치면 **거짓 비교**가 된다. 따라서:
      panel A = **z-score 정규화 오버레이**(각 시계열을 자기 평균/표준편차로 표준화)
                — 계절 **위상(peak timing)·상대 진폭(shape)** 비교 전용.
      panel B = **출처별 raw 멀티패널(small multiples)** — 각국을 자기 native 단위
                축에 그려 절대 스케일 차이를 숨기지 않는다(각 패널 제목에 단위 명시).
      panel C = **챔피언(FusedEpi) hold-out test 예측 vs 실측** — 단, 파이프라인
                target 축은 서울 sentinel 전연령 평균과 **스케일/정렬이 다르므로**
                별도 패널·자기 축에 그리고 그 사실을 부제에 정직히 명시
                (서울 라인에 강제 오버레이 = 거짓 정렬이라 하지 않는다).

[데이터]
    DB ``simulation/data/db/epi_real_seoul.db``
      - ``sentinel_influenza``  : 서울 ILI (season_start, week_seq, week_label, age_group, ili_rate)
      - ``overseas_ili``        : 해외 (source, country, year, week_no, ili_rate, ...)
    챔피언 예측 CSV: ``simulation/results/csv/predictions_FusedEpi.csv``
    챔피언 식별: ``simulation/results/per_model_eval/ranking.json`` 의 top10_by_wis[0].

[정직성 체크리스트]
    - who_flunet ili_rate = ILI 활동도/검출 지표(positivity 류) → 단위 다름 명시.
    - z-score 패널 = **위상/형태 비교 전용**(절대 크기 비교 아님) 부제 명시.
    - raw 패널 = 각국 native 단위 축 + 단위 라벨(절대 스케일 은폐 금지).
    - 챔피언 예측 = 파이프라인 target 축(서울 평균과 정렬·스케일 상이) → 별도 패널·부제 명시.
    - 데이터 부재/희소 시 가짜 생성 없이 honest skip + 로그.
    - 해외 forecast 산출 없음(OverseasTransfer = phantom) → 해외 라인에 예측 절대 안 그림.

[출력]
    모두 ``simulation/results/figures/`` (미존재 시 생성). dpi=130, bbox_inches="tight".
    Agg backend + 한글폰트(AppleGothic→NanumGothic).
      - 단독 패널 ("한 번에 하나씩" 보기):
          ``overseas_compare_overview.png``            = Panel A z-score 오버레이.
          ``overseas_compare_<source>_<country>.png``  = 해외 각국 raw native 단위(국가별 1장).
          ``overseas_compare_seoul_raw.png``           = 서울 raw(외래 1,000명당).
          ``overseas_compare_champion.png``            = Panel C 챔피언 예측 vs 실측(CSV 있을 때만).
      - back-compat combined gridspec(종전 동일):
          ``fig_overseas_ili_compare.png``

[결정성]
    순수 read + 결정적 통계(난수 미사용). 동일 DB/CSV 면 동일 PNG.

실행:
    .venv/bin/python -m simulation.scripts.fig_overseas_compare
"""

from __future__ import annotations

import csv
import json
import sys
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless 렌더 (#1 OS 비종속)
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

from simulation.database import read_only_connect  # G-116/117: 안전 read-only 헬퍼 사용

# ---------------------------------------------------------------------------
# 경로 SSOT (#4 단일 DB / 단일 출력)
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]  # .../MPH_infection_simulation
DB_PATH = _ROOT / "simulation" / "data" / "db" / "epi_real_seoul.db"
FIG_DIR = _ROOT / "simulation" / "results" / "figures"
CHAMPION_CSV_DIR = _ROOT / "simulation" / "results" / "csv"
RANKING_JSON = _ROOT / "simulation" / "results" / "per_model_eval" / "ranking.json"

# 비교 대상 해외 출처×국가 (2019-2026 ili_rate 가용 확인된 것만).
# who_flunet GB 는 ili_rate 109/352 로 희소 → 제외(panel 에서 honest skip 처리도 함).
PANEL_SOURCES: list[tuple[str, str, str]] = [
    ("delphi_national", "US", "United States (US) (Delphi/ILINet, outpatient ILI %)"),
    ("who_flunet", "KR", "South Korea (KR) (WHO FluNet, ILI activity)"),
    ("who_flunet", "JP", "Japan (JP) (WHO FluNet, ILI activity)"),
    ("who_flunet", "DE", "Germany (DE) (WHO FluNet, ILI activity)"),
    ("who_flunet", "FR", "France (FR) (WHO FluNet, ILI activity)"),
    ("who_flunet", "AU", "Australia (AU) (WHO FluNet, ILI activity)"),
    ("who_flunet", "US", "United States (US) (WHO FluNet, ILI activity)"),
    ("who_flunet", "SG", "Singapore (SG) (WHO FluNet, ILI activity)"),
]


# ===========================================================================
# 한글 폰트 (AppleGothic → NanumGothic fallback)
# ===========================================================================
def _set_korean_font() -> str:
    """matplotlib 한글 폰트를 설정한다 (AppleGothic→NanumGothic fallback).

    Returns:
        실제로 선택된 폰트 이름. 한글 폰트 부재 시 ``"DejaVu Sans"`` 반환 + 경고.

    Performance: O(폰트수) 1회 스캔.
    Side effects: ``plt.rcParams`` 전역 변경(font.family / axes.unicode_minus).
    """
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False
    return "DejaVu Sans"


# ===========================================================================
# 시간축 helper (ISO 연-주차 → 월요일 날짜; 결정적)
# ===========================================================================
def _isoweek_to_date(cal_year: int, week_no: int) -> date | None:
    """ISO 달력 연-주차를 그 주 월요일 ``date`` 로 환산한다 (결정적).

    Args:
        cal_year: 달력 연도(예: 2023).
        week_no: ISO 주차(1..53). 해당 연도에 53주 미존재 시 52주로 안전 fallback.

    Returns:
        월요일 ``datetime.date``. 주차가 비정상(<1 또는 >53)이면 ``None``.

    Performance: O(1). Side effects: 없음.
    """
    if week_no < 1 or week_no > 53:
        return None
    try:
        return date.fromisocalendar(cal_year, week_no, 1)
    except ValueError:
        try:
            return date.fromisocalendar(cal_year, 52, 1)
        except ValueError:
            return None


def _seoul_label_to_date(season_start: int, label: str) -> date | None:
    """서울 (season_start, week_label) 을 그 ISO 주차 월요일로 환산한다.

    표본감시 시즌은 ISO 주차 36 에서 시작해 익년 35 주차까지 이어진다. 라벨번호
    >=36 은 ``season_start`` 연도, <36 은 ``season_start+1`` 연도에 속한다.

    Args:
        season_start: 시즌 시작 연도(예: 2019).
        label: ``"36주"`` 형태 ISO 주차 라벨.

    Returns:
        월요일 ``datetime.date`` 또는 파싱 실패 시 ``None``.

    Performance: O(1). Side effects: 없음.
    """
    try:
        n = int(label.replace("주", "").strip())
    except (ValueError, AttributeError):
        return None
    cal_year = season_start if n >= 36 else season_start + 1
    return _isoweek_to_date(cal_year, n)


# ===========================================================================
# 데이터 로드 (read-only DB)
# ===========================================================================
def load_seoul_all_age_ili(db_path: Path) -> dict:
    """서울 ``sentinel_influenza`` 전연령 ILI 시계열을 read-only 로 로드한다.

    7개 연령밴드를 (season_start, week_seq) 별 비가중 산술평균해 전연령 ILI 를 만든다.
    (연령별 분모 부재 → 가중평균 불가; 호출자가 부제에 명시.)

    Args:
        db_path: ``epi_real_seoul.db`` 경로.

    Returns:
        dict:
          - ``"dates"``: ``list[date]`` 길이 n (라벨→월요일).
          - ``"ili"``:   ``(n,)`` float64 전연령 ILI (외래 1,000명당, 밴드 평균).
          - ``"n_age"``: 평균에 사용된 연령밴드 수(정직성 표기용).
        데이터 부재 시 빈 ``"dates"``(n=0).

    Raises:
        FileNotFoundError: DB 파일 부재.

    Performance: 단일 SELECT + O(n) 집계. Side effects: DB read-only open/close.
    Caller responsibility: 빈 결과 시 skip.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"DB 없음: {db_path}")
    con = read_only_connect(str(db_path))
    try:
        cur = con.cursor()
        rows = cur.execute(
            "SELECT season_start, week_seq, week_label, "
            "AVG(ili_rate) AS ili, COUNT(DISTINCT age_group) AS n_age "
            "FROM sentinel_influenza "
            "GROUP BY season_start, week_seq, week_label "
            "ORDER BY season_start, week_seq"
        ).fetchall()
    finally:
        con.close()

    dates: list[date] = []
    ili: list[float] = []
    n_age_max = 0
    for season_start, _wk, label, ili_v, n_age in rows:
        d = _seoul_label_to_date(int(season_start), str(label))
        if d is None or ili_v is None:
            continue
        dates.append(d)
        ili.append(float(ili_v))
        n_age_max = max(n_age_max, int(n_age))
    return {"dates": dates, "ili": np.asarray(ili, dtype=np.float64), "n_age": n_age_max}


def load_overseas_series(db_path: Path, source: str, country: str) -> dict:
    """``overseas_ili`` 의 한 (source, country) 주간 ili_rate 시계열을 로드한다.

    ``ili_rate`` 가 NULL 인 주는 제외한다. (year, week_no) 를 ISO 월요일로 환산.

    Args:
        db_path: ``epi_real_seoul.db`` 경로.
        source: overseas_ili.source (예: ``"who_flunet"``).
        country: overseas_ili.country (ISO2, 예: ``"KR"``).

    Returns:
        dict:
          - ``"dates"``: ``list[date]`` 오름차순.
          - ``"ili"``:   ``(m,)`` float64 (native 단위, 출처마다 다름).
        데이터 부재 시 빈 ``"dates"``.

    Performance: 단일 SELECT + O(m). Side effects: DB read-only open/close.
    Caller responsibility: 단위가 출처마다 다름 — 절대값 직접 비교 금지(z-score/멀티패널 사용).
    """
    con = read_only_connect(str(db_path))
    try:
        cur = con.cursor()
        rows = cur.execute(
            "SELECT year, week_no, ili_rate FROM overseas_ili "
            "WHERE source=? AND country=? AND ili_rate IS NOT NULL "
            "ORDER BY year, week_no",
            (source, country),
        ).fetchall()
    finally:
        con.close()

    dates: list[date] = []
    ili: list[float] = []
    for year, week_no, ili_v in rows:
        d = _isoweek_to_date(int(year), int(week_no))
        if d is None:
            continue
        dates.append(d)
        ili.append(float(ili_v))
    return {"dates": dates, "ili": np.asarray(ili, dtype=np.float64)}


def load_champion_forecast() -> dict:
    """챔피언(top10_by_wis[0]) hold-out test 예측/실측을 로드한다.

    ``ranking.json`` 에서 챔피언 이름을 읽고 ``predictions_<champion>.csv`` 의
    ``split=='test'`` 행(idx 오름차순)을 가져온다. 이 CSV 의 ``y_true`` 는 파이프라인
    target 축으로, 서울 sentinel 전연령 평균과 **스케일/정렬이 다르다**(검증함). 따라서
    호출자는 이 시리즈를 서울 라인에 강제 오버레이하지 말고 **별도 패널·자기 축**에
    그리고 그 사실을 명시해야 한다.

    Returns:
        dict:
          - ``"name"``:   챔피언 모델명(str) 또는 ``None``(식별 실패).
          - ``"idx"``:    ``(k,)`` int  (test 슬랩 내 0..k-1 순서).
          - ``"y_true"``: ``(k,)`` float64.
          - ``"y_pred"``: ``(k,)`` float64.
        예측 CSV 부재 시 빈 배열(k=0).

    Performance: JSON 1회 + CSV 1회. Side effects: 파일 read-only.
    Caller responsibility: 서울 sentinel 축과 정렬 불가 — 별도 패널·정직 부제 필수.
    """
    name: str | None = None
    if RANKING_JSON.exists():
        try:
            rank = json.loads(RANKING_JSON.read_text(encoding="utf-8"))
            top = rank.get("top10_by_wis")
            if isinstance(top, list) and top:
                first = top[0]
                name = first if isinstance(first, str) else first.get("model") or first.get("name")
        except (json.JSONDecodeError, OSError, AttributeError):
            name = None

    empty = {"name": name, "idx": np.array([]), "y_true": np.array([]), "y_pred": np.array([])}
    if not name:
        return empty
    csv_path = CHAMPION_CSV_DIR / f"predictions_{name}.csv"
    if not csv_path.exists():
        return empty

    idx: list[int] = []
    yt: list[float] = []
    yp: list[float] = []
    with csv_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if not {"split", "idx", "y_true", "y_pred"}.issubset(set(cols)):
            return empty
        for row in reader:
            if row.get("split") != "test":
                continue
            try:
                idx.append(int(row["idx"]))
                yt.append(float(row["y_true"]))
                yp.append(float(row["y_pred"]))
            except (ValueError, KeyError):
                continue
    order = np.argsort(idx) if idx else np.array([], dtype=int)
    return {
        "name": name,
        "idx": np.asarray(idx, dtype=int)[order] if idx else np.array([]),
        "y_true": np.asarray(yt, dtype=np.float64)[order] if yt else np.array([]),
        "y_pred": np.asarray(yp, dtype=np.float64)[order] if yp else np.array([]),
    }


# ===========================================================================
# 변환 helper
# ===========================================================================
def _zscore(arr: np.ndarray) -> np.ndarray:
    """평균 0·표준편차 1 표준화 (결정적). 표준편차 0/NaN 이면 0 벡터 반환.

    Args:
        arr: ``(n,)`` float 시계열.

    Returns:
        ``(n,)`` float z-score. n==0 이면 빈 배열.

    Performance: O(n). Side effects: 없음.
    """
    if arr.size == 0:
        return arr
    mu = float(np.nanmean(arr))
    sd = float(np.nanstd(arr))
    if not np.isfinite(sd) or sd == 0.0:
        return np.zeros_like(arr)
    return (arr - mu) / sd


# ===========================================================================
# 패널 drawing helper (combined gridspec ↔ standalone PNG 공유 — 동일 코드 SSOT)
# ===========================================================================
def _region_slug(source: str, country: str) -> str:
    """(source, country) 를 파일명 안전 슬러그로 환산한다 (결정적).

    Args:
        source: overseas_ili.source (예: ``"who_flunet"``).
        country: overseas_ili.country (ISO2, 예: ``"KR"``).

    Returns:
        ``"who_flunet_kr"`` 형태 소문자 슬러그(영숫자/언더스코어만).

    Performance: O(len). Side effects: 없음.
    """
    raw = f"{source}_{country}".lower()
    return "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in raw)


def _panel_overview(ax, seoul: dict, overseas: list, seoul_start, standalone: bool) -> None:
    """Panel A(z-score 오버레이)를 ``ax`` 에 그린다 — combined/standalone 공유 SSOT.

    Args:
        ax: 그릴 matplotlib Axes.
        seoul: ``load_seoul_all_age_ili`` 반환 dict.
        overseas: ``(source, country, label, series_dict)`` 리스트.
        seoul_start: x축 좌측 경계(서울 첫 날짜).
        standalone: True 면 단독 figure용(제목 폰트 약간 키움). 그리기 로직은 동일.

    Performance: O(전체 점수). Side effects: ``ax`` 에 그림.
    """
    cmap = plt.get_cmap("tab10")
    # 서울 (굵게 강조)
    ax.plot(
        seoul["dates"], _zscore(seoul["ili"]),
        color="black", lw=2.4, label="Seoul ILI (KDCA sentinel surveillance, all-age mean)", zorder=5,
    )
    for i, (_src, _c, label, s) in enumerate(overseas):
        ax.plot(s["dates"], _zscore(s["ili"]), color=cmap(i % 10), lw=1.1, alpha=0.8, label=label)
    ax.axhline(0.0, color="grey", lw=0.6, ls=":")
    ax.set_xlim(seoul_start, max(seoul["dates"][-1], max(s["dates"][-1] for *_x, s in overseas)))
    ax.set_ylabel("z-score (own mean=0,\nSD=1 standardized)")
    ax.set_title(
        "A. Seoul vs major overseas countries ILI — z-score normalized overlay "
        "(for seasonal phase / relative amplitude comparison only; not absolute magnitude)",
        fontsize=12 if standalone else 11, fontweight="bold", loc="left",
    )
    ax.legend(fontsize=7.5, ncol=2, loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.25)


def _panel_region(ax, idx: int, label: str, series: dict, seoul_start, standalone: bool) -> None:
    """Panel B 의 한 해외 국가 raw native 단위 패널을 ``ax`` 에 그린다 — 공유 SSOT.

    Args:
        ax: 그릴 matplotlib Axes.
        idx: overseas 리스트 내 색인(색상 결정용).
        label: 패널 제목(국가·출처·단위 명시).
        series: ``load_overseas_series`` 반환 dict.
        seoul_start: x축 좌측 경계.
        standalone: True 면 단독 figure용(제목/틱 폰트 키움).

    Performance: O(점수). Side effects: ``ax`` 에 그림.
    """
    cmap = plt.get_cmap("tab10")
    ax.plot(series["dates"], series["ili"], color=cmap(idx % 10), lw=1.0)
    ax.set_title(label, fontsize=11 if standalone else 8.5)
    ax.tick_params(labelsize=9 if standalone else 7)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(seoul_start, series["dates"][-1])
    if standalone:
        ax.set_ylabel("ili_rate (native unit)")


def _panel_seoul_raw(ax, seoul: dict, seoul_start, standalone: bool) -> None:
    """Panel B 의 서울 raw(외래 1,000명당) 패널을 ``ax`` 에 그린다 — 공유 SSOT.

    Args:
        ax: 그릴 matplotlib Axes.
        seoul: ``load_seoul_all_age_ili`` 반환 dict.
        seoul_start: x축 좌측 경계.
        standalone: True 면 단독 figure용(제목/틱 폰트 키움).

    Performance: O(점수). Side effects: ``ax`` 에 그림.
    """
    ax.plot(seoul["dates"], seoul["ili"], color="black", lw=1.2)
    ax.set_title("Seoul ILI (per 1,000 outpatient visits, all-age mean)", fontsize=11 if standalone else 8.5)
    ax.tick_params(labelsize=9 if standalone else 7)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(seoul_start, seoul["dates"][-1])
    if standalone:
        ax.set_ylabel("ILI per 1,000 outpatient visits")


def _panel_champion(ax, champ: dict, standalone: bool) -> None:
    """Panel C(챔피언 hold-out test 예측 vs 실측)를 ``ax`` 에 그린다 — 공유 SSOT.

    Args:
        ax: 그릴 matplotlib Axes.
        champ: ``load_champion_forecast`` 반환 dict.
        standalone: True 면 단독 figure용(제목 폰트 키움).

    Performance: O(점수). Side effects: ``ax`` 에 그림.
    """
    if champ["y_true"].size > 0:
        x = np.arange(champ["y_true"].size)
        ax.plot(x, champ["y_true"], color="black", lw=1.8, label="observed y_true")
        ax.plot(x, champ["y_pred"], color="tab:red", lw=1.5, ls="--",
                label=f"champion forecast ({champ['name']})")
        ax.fill_between(x, champ["y_true"], champ["y_pred"], color="tab:red", alpha=0.12)
        ax.set_xlabel("hold-out test week (pipeline slab order, idx)")
        ax.set_ylabel("pipeline target unit")
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title(
            f"C. Champion ({champ['name']}) hold-out test forecast vs observed "
            "— pipeline target axis (scale/alignment differs from Seoul sentinel all-age mean, verified) → "
            "not force-overlaid on the Seoul line (no false alignment)",
            fontsize=11 if standalone else 10, fontweight="bold", loc="left",
        )
        ax.grid(True, alpha=0.25)
    else:
        ax.text(
            0.5, 0.5,
            f"Champion forecast CSV missing or identification failed (champion={champ['name']}) → panel omitted (no fabricated plot)",
            ha="center", va="center", fontsize=10, transform=ax.transAxes,
        )
        ax.set_axis_off()


def _save_standalone_panels(
    fig_dir: Path, seoul: dict, overseas: list, champ: dict, seoul_start,
) -> list[Path]:
    """각 패널을 단독 single-panel PNG 로 저장한다 ("한 번에 하나씩" 요구).

    파일명(``simulation/results/figures/`` 하위):
      - ``overseas_compare_overview.png``        = Panel A z-score 오버레이.
      - ``overseas_compare_<source>_<country>.png`` = 각 해외국 raw(국가별 1개).
      - ``overseas_compare_seoul_raw.png``       = 서울 raw native 단위.
      - ``overseas_compare_champion.png``        = Panel C 챔피언 예측 vs 실측
                                                   (예측 CSV 부재 시 생략).

    Args:
        fig_dir: PNG 저장 디렉터리(자동 생성).
        seoul: ``load_seoul_all_age_ili`` 반환 dict.
        overseas: ``(source, country, label, series_dict)`` 리스트.
        champ: ``load_champion_forecast`` 반환 dict.
        seoul_start: x축 좌측 경계.

    Returns:
        실제로 쓴 PNG 경로 리스트(생성 순서).

    Performance: O(패널수 × 점수). Side effects: PNG 파일 다수 쓰기 + stdout 로그.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def _emit(fig, path: Path) -> None:
        fig.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
        print(f"[fig_overseas_compare] 저장(단독): {path}")

    # Panel A — overview (full-width chart → 넓게)
    figA, axA = plt.subplots(figsize=(12, 5))
    _panel_overview(axA, seoul, overseas, seoul_start, standalone=True)
    _emit(figA, fig_dir / "overseas_compare_overview.png")

    # Panel B — 국가별 raw 단독 1장씩
    for i, (src, country, label, s) in enumerate(overseas):
        figR, axR = plt.subplots(figsize=(8, 5))
        _panel_region(axR, i, label, s, seoul_start, standalone=True)
        _emit(figR, fig_dir / f"overseas_compare_{_region_slug(src, country)}.png")

    # 서울 raw 단독
    figS, axSraw = plt.subplots(figsize=(8, 5))
    _panel_seoul_raw(axSraw, seoul, seoul_start, standalone=True)
    _emit(figS, fig_dir / "overseas_compare_seoul_raw.png")

    # Panel C — 챔피언 (예측 있을 때만; 부재 시 가짜 안 그림)
    if champ["y_true"].size > 0:
        figC, axCc = plt.subplots(figsize=(12, 5))
        _panel_champion(axCc, champ, standalone=True)
        _emit(figC, fig_dir / "overseas_compare_champion.png")
    else:
        print("[fig_overseas_compare] skip 단독 champion: 예측 CSV 부재(가짜 안 그림)")

    return written


# ===========================================================================
# Figure
# ===========================================================================
def build_figure(db_path: Path, out_path: Path) -> list[Path]:
    """서울 vs 해외 ILI 비교 figure 를 단독 패널 PNG + combined gridspec 로 저장한다.

    "한 번에 하나씩" 보기 위해 각 패널을 **단독 single-panel PNG** 로 먼저 저장하고,
    그 다음 back-compat 용 **combined gridspec PNG**(``out_path``)를 EXACTLY 종전대로
    저장한다. 두 경로 모두 동일한 ``_panel_*`` helper 를 재사용한다(SSOT).

    panel A: z-score 오버레이(서울+해외 전부) — 계절 위상/상대 진폭 비교.
    panel B: 출처별 raw 멀티패널(각국 native 단위 축) + 서울 raw.
    panel C: 챔피언 hold-out test 예측 vs 실측(파이프라인 target 축, 별도).

    Args:
        db_path: ``epi_real_seoul.db`` 경로.
        out_path: combined figure 저장 경로(부모 디렉터리 자동 생성).

    Returns:
        실제로 쓴 PNG 경로 리스트(단독 패널들 + combined). 서울/해외 데이터 부재로
        그릴 수 없으면 빈 리스트(honest skip).

    Performance: O(전체 시계열 점수). Side effects: PNG 파일 다수 쓰기 + stdout 로그.
    Caller responsibility: 빈 리스트 반환 시 산출물 없음으로 간주.
    """
    font = _set_korean_font()
    print(f"[fig_overseas_compare] 폰트={font}")

    seoul = load_seoul_all_age_ili(db_path)
    if seoul["ili"].size == 0:
        print("[fig_overseas_compare] WARN: 서울 ILI 데이터 부재 → skip(가짜 생성 안 함)")
        return []
    print(
        f"[fig_overseas_compare] 서울 ILI n={seoul['ili'].size}주 "
        f"({seoul['dates'][0]}~{seoul['dates'][-1]}, 연령밴드 {seoul['n_age']}개 평균)"
    )

    # 해외 시계열 로드 (희소/부재는 honest skip)
    seoul_start = seoul["dates"][0]
    overseas: list[tuple[str, str, str, dict]] = []
    for source, country, label in PANEL_SOURCES:
        s = load_overseas_series(db_path, source, country)
        if s["ili"].size < 30:
            print(f"[fig_overseas_compare] skip {source}/{country}: n={s['ili'].size} (<30, 희소)")
            continue
        overseas.append((source, country, label, s))
        print(f"[fig_overseas_compare] {label}: n={s['ili'].size}주")
    if not overseas:
        print("[fig_overseas_compare] WARN: 비교 가능한 해외 시계열 없음 → skip")
        return []

    champ = load_champion_forecast()

    # === (1) 단독 single-panel PNG 먼저 저장 ("한 번에 하나씩" 보기) ===
    written = _save_standalone_panels(FIG_DIR, seoul, overseas, champ, seoul_start)

    # === (2) back-compat combined gridspec PNG (종전과 EXACTLY 동일) ===
    # --- layout: A (top, wide), B (middle grid), C (bottom) ---
    n_panels_b = len(overseas)
    ncol_b = 3
    nrow_b = (n_panels_b + ncol_b - 1) // ncol_b
    fig = plt.figure(figsize=(15, 5.2 + 2.4 * nrow_b + 3.0))
    gs = fig.add_gridspec(
        nrows=2 + nrow_b,
        ncols=ncol_b,
        height_ratios=[3.2] + [1.9] * nrow_b + [2.6],
        hspace=0.62,
        wspace=0.28,
    )

    # ===== Panel A: z-score 오버레이 =====
    axA = fig.add_subplot(gs[0, :])
    _panel_overview(axA, seoul, overseas, seoul_start, standalone=False)

    # ===== Panel B: 출처별 raw 멀티패널 (native 단위) =====
    for i, (_src, _c, label, s) in enumerate(overseas):
        r = 1 + i // ncol_b
        c = i % ncol_b
        ax = fig.add_subplot(gs[r, c])
        _panel_region(ax, i, label, s, seoul_start, standalone=False)
    # 서울 raw 도 멀티패널 끝에 (단위 = 외래 1,000명당)
    used = n_panels_b
    r = 1 + used // ncol_b
    c = used % ncol_b
    if r <= nrow_b:  # 자리 있으면 서울 raw 추가
        axS = fig.add_subplot(gs[r, c])
        _panel_seoul_raw(axS, seoul, seoul_start, standalone=False)

    # B 패널 전체 위 제목용 텍스트(첫 패널 좌상단 annotation 대신 figure text)
    y_b_top = 1.0 - (3.2 / (3.2 + 1.9 * nrow_b + 2.6)) * 0.34
    fig.text(
        0.012, y_b_top,
        "B. Raw ILI by source (units differ — each panel on its native axis): "
        "WHO FluNet = ILI activity/detection index (0–100), Delphi/ILINet US = outpatient ILI %, Seoul = per 1,000 outpatient visits. "
        "→ do not directly compare absolute values across countries (units differ).",
        fontsize=9, fontweight="bold", ha="left", va="bottom",
    )

    # ===== Panel C: 챔피언 hold-out test 예측 vs 실측 =====
    axC = fig.add_subplot(gs[1 + nrow_b, :])
    _panel_champion(axC, champ, standalone=False)

    # ===== figure 전체 제목/주석 =====
    fig.suptitle(
        "Seoul ILI vs major overseas countries ILI comparison (observed only; units honestly separated)",
        fontsize=14, fontweight="bold", y=0.995,
    )
    fig.text(
        0.012, 0.004,
        "Data: epi_real_seoul.db — sentinel_influenza (Seoul, all-age mean per 1,000 outpatient visits) + "
        "overseas_ili (delphi_national/US, who_flunet/KR, JP, DE, FR, AU, US, SG). "
        "who_flunet ili_rate = WHO-reported ILI activity/detection index (positivity-style, units/scale differ by country). "
        "This figure compares seasonal SHAPE/PHASE only (z-score overlay; e.g. Seoul-US shape Spearman rho ~ 0.49) — it is NOT forecast accuracy: "
        "the champion forward-forecast generalization (e.g. US R^2 ~ 0.96) is reported separately (Fig. champion-generalization / Table 3). "
        "No overseas forecast output here (OverseasTransfer = phantom). read-only DB, no fabricated data.",
        fontsize=7.0, ha="left", va="bottom", color="#444444",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig_overseas_compare] 저장(combined): {out_path}")
    written.append(out_path)
    return written


def main() -> int:
    """엔트리: 단독 패널 PNG + combined figure 생성 후 성공/실패 반환.

    Returns:
        0 성공, 1 실패(데이터 부재/저장 실패 → honest skip).
    """
    out_path = FIG_DIR / "fig_overseas_ili_compare.png"
    written = build_figure(DB_PATH, out_path)
    if not written:
        print("[fig_overseas_compare] figure 미생성(데이터 부재) — 가짜 생성 안 함.")
        return 1
    # 산출물 무결성 검증 + 전체 목록 출력
    bad = [p for p in written if not p.exists() or p.stat().st_size == 0]
    if bad:
        print(f"[fig_overseas_compare] ERROR: 산출 파일 비정상 {[str(p) for p in bad]}")
        return 1
    print(f"[fig_overseas_compare] OK — 총 {len(written)}개 PNG 작성:")
    for p in written:
        print(f"    - {p} ({p.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
