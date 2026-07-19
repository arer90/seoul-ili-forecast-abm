"""대상자(agent) 프로파일 상세 그림 1종 생성기 — 실 합성인구 전용.

"감염병 시뮬레이션의 대상자가 **누구인지**" 를 구체화하는 단일 figure
(``fig_agent_profile.png``). ``simulation.abm.agent_history`` 의 검증된 deep
module(``simulate_with_history`` · ``extract_agent_trajectory`` ·
``population_summary``)만 호출하며, **SEIR 동역학·합성인구를 재구현하지 않는다**
(ENGINEERING_PRINCIPLES.md K-3 / D-4: base 코드 미수정).

데이터 출처(정직성 — 관측 vs 모델-유래 구분):
    - **관측(observed) 분포**: agent 의 정적 속성(연령·성별·기저질환 severity·직업·
      home/work 구)은 ``generate_population`` 이 ``epi_real_seoul.db`` 의 실측
      통계(KOSIS 연령·성별, HIRA 인플루엔자 입원분율, 통근행렬, 사업체 산업분류)를
      그대로 확률화해 표본추출한 것 → **관측 기반**.
    - **모델-유래(model-derived)**: 각 agent 의 S→E→I→R 타임라인·감염 day·집계
      SEIR 곡선은 agent-based SEIR kernel(``run_agent_world``)의 결정적 시뮬
      산출 → **모델-유래**. 그림 제목/패널 주석에 [관측]/[모델] 라벨로 명시.

생성 그림 (단일 PNG, 2×4 grid):
    상단 4패널 = **감염된 대표 agent 1-2명의 상세 프로파일**
        ① 대표 agent A 카드: 연령·성별·기저질환·직업·home/work 구 (텍스트 카드)
        ② 대표 agent A 감염경로 타임라인: S→E→I→R 상태 띠 + 전이 day 주석
        ③ 대표 agent B 카드 (있으면; 없으면 정직히 비움)
        ④ 대표 agent B 감염경로 타임라인
    하단 4패널 = **population 패널 (누가 감염되나)**
        ⑤ 연령×성별 인구 피라미드 (전체 vs 감염자 overlay)
        ⑥ 기저질환(severity) 분포: 전체 vs 감염자 (감염자가 고위험 비중 높은가)
        ⑦ 직업(산업분류) 분포: 전체 vs 감염자
        ⑧ 연령군별 발병률(attack rate) — 감염자/전체 비율 (누가 더 감염되나)

실행:
    .venv/bin/python -m simulation.scripts.fig_agent_profile

출력:
    simulation/results/figures/fig_agent_profile.png  (dpi=120, bbox_inches="tight")

설계 규율:
    - 실 합성인구만 (DB read-only via generate_population). 가짜/임의 합성 금지.
    - DB 직접 접근 필요 시 ``read_only_connect`` 전용 (저수준 연결 금지).
    - matplotlib Agg backend + 한글폰트 (AppleGothic → NanumGothic fallback).
    - 결정성: seed 고정(=42), 임의 난수 없음. 동일 실행 → 비트 동일 그림.
    - 데이터(=감염자) 없으면 정직히 skip + 로그 (가짜 생성 금지).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless 렌더링 (디스플레이 비종속, ENGINEERING_PRINCIPLES.md §1)

import matplotlib.font_manager as fm
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from simulation.abm.agent_history import (
    AGE_BAND_LABELS,
    SEVERITY_LABELS,
    SEX_LABELS,
    STATE_LABELS,
    extract_agent_trajectory,
    population_summary,
    simulate_with_history,
)
from simulation.abm.synthetic_population import INDUSTRY_NAMES

# ---------------------------------------------------------------------------
# 경로 SSOT (단일 코드 루트 — ENGINEERING_PRINCIPLES.md §4 KISS)
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parents[2]  # simulation/scripts/fig_agent_profile.py → repo root
FIG_DIR = PROJECT_ROOT / "simulation" / "results" / "figures"
OUTPUT_PNG = FIG_DIR / "fig_agent_profile.png"  # 통합본(2×4 grid, back-compat)

# 단일패널(standalone) 산출 — "figure 를 한 번에 1장씩" 보고 싶다는 요청 대응.
# 그룹화 결정(아래 build_figure 주석 참고):
#   - 대표 agent 1명 = 카드+타임라인을 한 standalone figure 로 묶음(= 그 agent 의
#     "프로파일" = 누구인가 + 어떻게 감염됐나, 하나의 이야기) → A/B 각각 1장.
#   - 하단 population 패널 4종 = 각각 독립 차트이므로 각각 1장.
OUTPUT_AGENT_A = FIG_DIR / "agent_profile_agent_a.png"   # 카드+타임라인(대표 agent A)
OUTPUT_AGENT_B = FIG_DIR / "agent_profile_agent_b.png"   # 카드+타임라인(대표 agent B)
OUTPUT_PYRAMID = FIG_DIR / "agent_profile_age_sex_pyramid.png"
OUTPUT_SEVERITY = FIG_DIR / "agent_profile_severity_dist.png"
OUTPUT_OCCUPATION = FIG_DIR / "agent_profile_occupation_dist.png"
OUTPUT_ATTACK_RATE = FIG_DIR / "agent_profile_attack_rate_by_age.png"

# 시뮬 설정 (결정성: 모든 값 고정). off-season 확률적 소멸을 막는 작은 import_rate +
# severity 로 변조되는 사망 hazard delta 를 켜서 "기저질환 → 결과" 연결을 표면화한다.
SEED = 42
N_AGENTS = 3000
T_DAYS = 120
SIM_KWARGS = dict(
    beta=0.45,
    sigma=0.3,
    gamma=0.18,
    delta=0.004,        # severity(고위험)로 변조되는 사망 hazard — 기저질환 결과 연결
    import_rate=2e-4,   # off-season 확률적 소멸 방지(초기 점화)
)

# 상태 색상 (S→E→I→R→V→D) — 타임라인 띠 / 범례 공통
STATE_COLORS: dict[str, str] = {
    "S": "#9ecae1",  # 감수성 (연파랑)
    "E": "#fdae6b",  # 잠복 (주황)
    "I": "#de2d26",  # 발병 (적색)
    "R": "#74c476",  # 회복 (녹색)
    "V": "#9e9ac8",  # 백신 (보라)
    "D": "#252525",  # 사망 (흑)
}
STATE_FULL: dict[str, str] = {
    "S": "Susceptible (S)",
    "E": "Exposed (E)",
    "I": "Infectious (I)",
    "R": "Recovered (R)",
    "V": "Vaccinated (V)",
    "D": "Dead (D)",
}

# Compartment ordering for the discrete state-timeline step plot (Fig 13 fix):
# a single agent occupies exactly one compartment per day, so a flat color band
# is unreadable; we render the state as a step line over an ordinal S<E<I<R(<D)
# axis instead, which makes each transition legible.
STATE_STEP_ORDER: dict[str, int] = {"S": 0, "E": 1, "I": 2, "R": 3, "V": 4, "D": 5}
STATE_STEP_LABELS: list[str] = ["S", "E", "I", "R", "V", "D"]


def _setup_korean_font() -> str:
    """Force the default English font for paper figures (English labels only).

    Returns:
        The applied font family name ("DejaVu Sans").

    Side effects: sets plt.rcParams["font.family"], ["axes.unicode_minus"].
    """
    chosen = "DejaVu Sans"
    plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False  # avoid broken minus glyph
    print(f"[font] applied font = {chosen}")
    return chosen


def _confirm_png(path: Path) -> bool:
    """PNG 가 실제로 생성되었고 비어있지 않은지 검증.

    Args:
        path: 검증할 PNG 절대경로.

    Returns:
        존재 + size>0 이면 True, 아니면 False (로그 출력).
    """
    if path.exists() and path.stat().st_size > 0:
        print(f"[OK]   {path}  ({path.stat().st_size:,} bytes)")
        return True
    print(f"[FAIL] {path}  (미생성 또는 0 bytes)")
    return False


# English display labels for the base-module Korean constants (base code is NOT
# modified, per K-3 — figure-side display mapping only; index-aligned to
# synthetic_population.INDUSTRY_NAMES, KSIC major occupational groups).
_INDUSTRY_ENG: dict[str, str] = {
    "관리자 전문가 및 관련종사자": "Managers & professionals",
    "사무 종사자": "Clerical workers",
    "서비스·판매 종사자": "Service & sales workers",
    "농림어업 숙련종사자": "Skilled agriculture/fishery",
    "기능·기계조작·조립 종사자": "Craft/machine operators",
    "단순노무 종사자": "Elementary occupations",
}

# Seoul 25 districts: Korean gu name → romanized English (for card display).
_GU_ENG: dict[str, str] = {
    "종로구": "Jongno-gu", "중구": "Jung-gu", "용산구": "Yongsan-gu",
    "성동구": "Seongdong-gu", "광진구": "Gwangjin-gu", "동대문구": "Dongdaemun-gu",
    "중랑구": "Jungnang-gu", "성북구": "Seongbuk-gu", "강북구": "Gangbuk-gu",
    "도봉구": "Dobong-gu", "노원구": "Nowon-gu", "은평구": "Eunpyeong-gu",
    "서대문구": "Seodaemun-gu", "마포구": "Mapo-gu", "양천구": "Yangcheon-gu",
    "강서구": "Gangseo-gu", "구로구": "Guro-gu", "금천구": "Geumcheon-gu",
    "영등포구": "Yeongdeungpo-gu", "동작구": "Dongjak-gu", "관악구": "Gwanak-gu",
    "서초구": "Seocho-gu", "강남구": "Gangnam-gu", "송파구": "Songpa-gu",
    "강동구": "Gangdong-gu",
}


def _gu_eng(name: str) -> str:
    """Romanize a Seoul district (gu) name to English; pass through if unknown."""
    return _GU_ENG.get(str(name), str(name))


def _occupation_label(code: int) -> str:
    """Map occupation (KSIC major group) integer code to an English label.

    Args:
        code: ``attrs['occupation']`` integer code (``INDUSTRY_NAMES`` index).

    Returns:
        English occupation-group name. "unavailable" or the integer string
        when out of range.
    """
    if 0 <= code < len(INDUSTRY_NAMES):
        raw = str(INDUSTRY_NAMES[code])
        return _INDUSTRY_ENG.get(raw, raw)
    return str(int(code))


def _pick_representative_infected(result: dict, k: int = 2) -> list[int]:
    """감염된(E/I 경험) 대표 agent 인덱스를 결정적으로 선정.

    "대표성"을 위해 감염된 agent 중 **고위험(severity=1) 우선 → 가장 일찍 감염된
    순"**으로 정렬해 상위 k 명을 고른다(둘 다 없으면 일반 감염자). 결정적(tie-break
    = agent_id 오름차순)이라 동일 seed → 동일 선정.

    Args:
        result: ``simulate_with_history`` 반환 dict.
        k: 뽑을 대표 agent 수(>=1).

    Returns:
        agent_id 리스트(길이 0..k). 감염자 자체가 없으면 빈 리스트.

    Performance: O(T_days * N) time(한 번의 마스킹). Side effects: 없음.
    """
    history_state = np.asarray(result["history_state"])
    severity = np.asarray(result["attrs"]["severity"])
    e_idx = STATE_LABELS.index("E")
    i_idx = STATE_LABELS.index("I")

    ever_infected = ((history_state == e_idx) | (history_state == i_idx)).any(axis=0)
    candidates = np.flatnonzero(ever_infected)
    if candidates.size == 0:
        return []

    # First-infection day per candidate (first E or I appearance).
    infect_mask = (history_state == e_idx) | (history_state == i_idx)
    first_day = np.full(history_state.shape[1], history_state.shape[0], dtype=np.int64)
    rows = np.argmax(infect_mask, axis=0)  # first True row (only candidates used)
    first_day[candidates] = rows[candidates]

    # Distinct compartments visited (richer trajectory = more illustrative for the
    # state-timeline step plot; a day-0 seed that jumps straight to I shows fewer).
    n_distinct = np.array(
        [int(np.unique(history_state[:, a]).size) for a in candidates], dtype=np.int64
    )
    distinct_by_id = dict(zip(candidates.tolist(), n_distinct.tolist()))

    # Sort key: high-risk first (-severity), richer path first (-n_distinct so the
    # full S->E->I->R is visible rather than a day-0 seed), then earliest infection,
    # then agent_id (deterministic tie-break).
    order = sorted(
        candidates.tolist(),
        key=lambda a: (
            -int(severity[a]),
            -int(distinct_by_id[a]),
            int(first_day[a]),
            int(a),
        ),
    )
    return order[: max(1, k)]


def _infected_mask(result: dict) -> np.ndarray:
    """전체 agent 중 시뮬 기간 동안 한 번이라도 감염(E/I)된 boolean 마스크.

    Args:
        result: ``simulate_with_history`` 반환 dict.

    Returns:
        (N,) bool — True = 감염 경험. Side effects: 없음.
    """
    history_state = np.asarray(result["history_state"])
    e_idx = STATE_LABELS.index("E")
    i_idx = STATE_LABELS.index("I")
    return ((history_state == e_idx) | (history_state == i_idx)).any(axis=0)


# ---------------------------------------------------------------------------
# 상단: 대표 agent 프로파일 패널 (카드 + 타임라인)
# ---------------------------------------------------------------------------
def _draw_agent_card(ax: plt.Axes, traj: dict, label: str) -> None:
    """한 agent 의 정적 프로파일을 텍스트 카드로 렌더(연령·성별·기저질환·직업·구).

    Args:
        ax: 그릴 축(축선/눈금 숨김 텍스트 카드).
        traj: ``extract_agent_trajectory`` 반환 dict.
        label: 카드 제목 접두(예: "대표 agent A").

    Side effects: ax 에 텍스트/도형 그림. 반환값 없음.
    """
    ax.axis("off")
    a = traj["attrs"]
    sev_en = "Yes (high-risk)" if a["severity"] == 1 else "No (low-risk)"
    commute = (
        "Commuter (works in another gu)"
        if a["is_commuter"]
        else "Non-commuter (within home gu)"
    )
    inf_day = traj["infected_day"]
    inf_txt = f"day {inf_day}" if inf_day is not None else "never infected"

    lines = [
        (f"{label}  (agent #{traj['agent_id']})", True),
        (f"Age group       {a['age_band_label']} yr", False),
        (f"Sex             {'Male' if a['sex'] == 0 else 'Female'}", False),
        (f"Comorbidity     {sev_en}", False),
        (f"Occupation      {_occupation_label(a['occupation'])}", False),
        (f"Home district   {_gu_eng(a['home_gu_name'])}", False),
        (f"Work district   {_gu_eng(a['work_gu_name'])}", False),
        (f"Mobility        {commute}", False),
        (f"First infection {inf_txt}", False),
    ]
    y = 0.95
    for text, is_title in lines:
        ax.text(
            0.04,
            y,
            text,
            transform=ax.transAxes,
            fontsize=12.5 if is_title else 11,
            fontweight="bold" if is_title else "normal",
            va="top",
            ha="left",
            color="#222222" if not is_title else "#08306b",
        )
        y -= 0.135 if is_title else 0.105
    # 카드 테두리
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (0.01, 0.01),
            0.98,
            0.98,
            transform=ax.transAxes,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=1.3,
            edgecolor="#bbbbbb",
            facecolor="#f7fbff",
            zorder=-1,
        )
    )


def _draw_agent_timeline(ax: plt.Axes, traj: dict, label: str) -> None:
    """Render one agent's S->E->I->R path as a discrete state-timeline step plot.

    Fix (Fig 13): a single agent occupies exactly ONE compartment per day, so the
    previous flat color band was unreadable. We instead plot the compartment as a
    step line over an ordinal axis (S<E<I<R[<D]) with state-colored markers and
    a faint state-colored background band, so every transition day is legible.

    Args:
        ax: target axes.
        traj: ``extract_agent_trajectory`` return dict.
        label: panel title prefix.

    Side effects: draws on ax. Title flags [model-derived] (simulation dynamics).
    """
    states = traj["state_labels"]
    T = len(states)
    days = np.arange(T)
    levels = np.array([STATE_STEP_ORDER.get(s, 0) for s in states], dtype=float)

    # Faint per-day state-colored background band (orientation aid, low alpha).
    for day, s in enumerate(states):
        ax.axvspan(
            day, day + 1, color=STATE_COLORS.get(s, "#cccccc"), alpha=0.16, linewidth=0
        )

    # Step line of the compartment level over time (post-step = state held until
    # next transition) + state-colored markers so each day's compartment is clear.
    ax.step(days, levels, where="post", color="#333333", linewidth=1.6, zorder=3)
    for day, s in enumerate(states):
        ax.plot(
            day,
            STATE_STEP_ORDER.get(s, 0),
            marker="o",
            markersize=3.2,
            color=STATE_COLORS.get(s, "#cccccc"),
            markeredgecolor="#333333",
            markeredgewidth=0.4,
            zorder=4,
        )

    # Transition markers: vertical line + "<state> day N" annotation (skip day-0
    # seed). Annotations are placed at the post-transition compartment level with a
    # small alternating vertical offset so adjacent transitions do not overlap.
    real_transitions = [t for t in traj["transitions"] if t[0] != 0]
    for j, (day, _code, lab) in enumerate(real_transitions):
        ax.axvline(day, color="black", linewidth=0.7, alpha=0.45, zorder=2)
        y_to = STATE_STEP_ORDER.get(lab, levels[min(day, T - 1)])
        # stagger label above/below the post-transition level to reduce overlap
        y_text = y_to + (0.55 if (j % 2 == 0) else -0.55)
        ax.annotate(
            f"{lab} @ day {day}",
            xy=(day, y_to),
            xytext=(day + max(2.0, T * 0.015), y_text),
            ha="left",
            va="center",
            fontsize=8.0,
            color="#111111",
            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.6),
        )

    ax.set_xlim(0, T)
    ax.set_ylim(-0.5, len(STATE_STEP_LABELS) - 0.5)
    ax.set_yticks(list(range(len(STATE_STEP_LABELS))))
    ax.set_yticklabels(
        [STATE_FULL[s] for s in STATE_STEP_LABELS], fontsize=8.5
    )
    ax.set_xlabel("Simulation day", fontsize=10)
    ax.set_title(f"{label} - infection-state timeline [model-derived]", fontsize=11.5)
    ax.grid(axis="x", alpha=0.18, zorder=0)


# ---------------------------------------------------------------------------
# 하단: population 패널 (누가 감염되나)
# ---------------------------------------------------------------------------
def _draw_age_sex_pyramid(ax: plt.Axes, result: dict, inf_mask: np.ndarray) -> None:
    """연령×성별 인구 피라미드 (전체 vs 감염자 overlay).

    좌측=남성, 우측=여성. 옅은 막대=전체 인구(관측 분포), 진한 막대=감염자(모델).
    """
    age = np.asarray(result["attrs"]["age_band"])
    sex = np.asarray(result["attrs"]["sex"])
    n_band = len(AGE_BAND_LABELS)
    ypos = np.arange(n_band)

    def _counts(mask_extra: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        m = np.ones(age.shape[0], bool) if mask_extra is None else mask_extra
        male = np.array([int(((age == b) & (sex == 0) & m).sum()) for b in range(n_band)])
        female = np.array(
            [int(((age == b) & (sex == 1) & m).sum()) for b in range(n_band)]
        )
        return male, female

    tot_m, tot_f = _counts(None)
    inf_m, inf_f = _counts(inf_mask)

    ax.barh(ypos, -tot_m, color="#c6dbef", label="All male (observed)")
    ax.barh(ypos, tot_f, color="#fcbba1", label="All female (observed)")
    ax.barh(ypos, -inf_m, color="#3182bd", label="Infected male (model)")
    ax.barh(ypos, inf_f, color="#de2d26", label="Infected female (model)")

    ax.set_yticks(ypos)
    ax.set_yticklabels(AGE_BAND_LABELS, fontsize=9)
    ax.set_xlabel("<- Male     count     Female ->", fontsize=9.5)
    ax.set_title("Age x sex pyramid: all vs infected", fontsize=11)
    ax.axvline(0, color="#444444", linewidth=0.8)
    # x 라벨을 절대값으로 (FixedLocator 로 tick 고정 후 라벨 — set_ticklabels 경고 회피)
    ticks = ax.get_xticks()
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{abs(int(t)):,}" for t in ticks], fontsize=8)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.85)


def _draw_severity_dist(ax: plt.Axes, result: dict, inf_mask: np.ndarray) -> None:
    """기저질환(severity) 분포: 전체 vs 감염자 (비율, %).

    감염자가 전체보다 고위험 비중이 높은지 비교(이질 ABM 의 핵심 — 누가 감염되나).
    """
    severity = np.asarray(result["attrs"]["severity"])
    n_lab = len(SEVERITY_LABELS)
    x = np.arange(n_lab)
    width = 0.38

    tot = np.array([int((severity == v).sum()) for v in range(n_lab)], float)
    inf = np.array(
        [int(((severity == v) & inf_mask).sum()) for v in range(n_lab)], float
    )
    tot_pct = 100.0 * tot / max(tot.sum(), 1.0)
    inf_pct = 100.0 * inf / max(inf.sum(), 1.0)

    labels_en = {"low": "Low-risk (no comorbidity)", "high": "High-risk (comorbidity)"}
    xt = [labels_en.get(SEVERITY_LABELS[v], SEVERITY_LABELS[v]) for v in range(n_lab)]

    ax.bar(x - width / 2, tot_pct, width, color="#9ecae1", label="All (observed)")
    ax.bar(x + width / 2, inf_pct, width, color="#de2d26", label="Infected (model)")
    for xi, val in zip(x - width / 2, tot_pct):
        ax.text(xi, val + 0.5, f"{val:.0f}%", ha="center", fontsize=8)
    for xi, val in zip(x + width / 2, inf_pct):
        ax.text(xi, val + 0.5, f"{val:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(xt, fontsize=8.5)
    ax.set_ylabel("Proportion (%)", fontsize=9.5)
    ax.set_title("Comorbidity distribution: all vs infected", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")


def _draw_occupation_dist(ax: plt.Axes, result: dict, inf_mask: np.ndarray) -> None:
    """직업(산업분류) 분포: 전체 vs 감염자 (비율, %), 수평 막대."""
    occ = np.asarray(result["attrs"]["occupation"])
    n_occ = len(INDUSTRY_NAMES)
    if n_occ == 0 or (n_occ == 1 and INDUSTRY_NAMES[0] == "unavailable"):
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "No occupation data (DB unavailable) - honest skip",
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.set_title("Occupation (industry) distribution", fontsize=11)
        return

    y = np.arange(n_occ)
    height = 0.38
    tot = np.array([int((occ == c).sum()) for c in range(n_occ)], float)
    inf = np.array([int(((occ == c) & inf_mask).sum()) for c in range(n_occ)], float)
    tot_pct = 100.0 * tot / max(tot.sum(), 1.0)
    inf_pct = 100.0 * inf / max(inf.sum(), 1.0)

    ax.barh(y - height / 2, tot_pct, height, color="#a1d99b", label="All (observed)")
    ax.barh(y + height / 2, inf_pct, height, color="#de2d26", label="Infected (model)")
    ax.set_yticks(y)
    ax.set_yticklabels(
        [_occupation_label(c) for c in range(n_occ)], fontsize=8.5
    )
    ax.invert_yaxis()
    ax.set_xlabel("Proportion (%)", fontsize=9.5)
    ax.set_title("Occupation distribution: all vs infected", fontsize=11)
    ax.legend(fontsize=8, loc="lower right")


def _draw_attack_rate_by_age(ax: plt.Axes, result: dict, inf_mask: np.ndarray) -> None:
    """연령군별 발병률(attack rate) = 감염자/전체 (%) — 누가 더 감염되나 [모델]."""
    age = np.asarray(result["attrs"]["age_band"])
    n_band = len(AGE_BAND_LABELS)
    x = np.arange(n_band)
    rates = []
    for b in range(n_band):
        denom = int((age == b).sum())
        num = int(((age == b) & inf_mask).sum())
        rates.append(100.0 * num / denom if denom > 0 else 0.0)
    rates = np.array(rates)

    bars = ax.bar(x, rates, color="#fd8d3c", edgecolor="#7f2704", linewidth=0.6)
    for xi, val in zip(x, rates):
        ax.text(xi, val + 0.4, f"{val:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(AGE_BAND_LABELS, fontsize=8.5, rotation=0)
    ax.set_ylabel("Attack rate (%)", fontsize=9.5)
    ax.set_xlabel("Age group", fontsize=9.5)
    ax.set_title("Attack rate by age group [model-derived]", fontsize=11)
    ax.set_ylim(0, max(rates.max() * 1.18, 1.0))


def _state_legend_handles() -> list[mpatches.Patch]:
    """SEIR 상태 색상 범례 핸들(타임라인 띠 해석용)."""
    return [
        mpatches.Patch(color=STATE_COLORS[s], label=STATE_FULL[s])
        for s in ("S", "E", "I", "R", "D")
    ]


# ---------------------------------------------------------------------------
# 단일패널(standalone) figure 렌더 — "1장씩" 보고 싶다는 요청 대응.
#   각 helper 는 combined(2×4) 와 **동일한** _draw_* 코드를 재사용한다(SSOT).
#   card+timeline 은 한 agent 의 "프로파일" 한 장(2 subplot)으로 묶는다.
# ---------------------------------------------------------------------------
def _save_agent_profile_standalone(traj: dict, label: str, out_path: Path) -> bool:
    """한 대표 agent 의 카드+타임라인을 단일 figure(1행 2열)로 저장.

    카드(정적 프로파일=누구인가)와 타임라인(S→E→I→R 경로=어떻게 감염됐나)은
    한 agent 를 설명하는 한 쌍이라 **하나의 standalone figure** 로 묶는다.
    상태 색상 범례(타임라인 해석용)를 figure 하단에 부착.

    Args:
        traj: ``extract_agent_trajectory`` 반환 dict.
        label: 카드/타임라인 제목 접두(예: "대표 agent A").
        out_path: 저장할 PNG 절대경로.

    Returns:
        PNG 생성 성공(size>0)이면 True.

    Side effects: out_path 에 PNG write. figure close.
    """
    fig = plt.figure(figsize=(13, 5.2))
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.0, 1.25], wspace=0.16,
        left=0.04, right=0.985, top=0.86, bottom=0.18,
    )
    ax_card = fig.add_subplot(gs[0, 0])
    ax_tl = fig.add_subplot(gs[0, 1])
    _draw_agent_card(ax_card, traj, label)
    _draw_agent_timeline(ax_tl, traj, label)
    fig.legend(
        handles=_state_legend_handles(),
        loc="lower center",
        ncol=5,
        fontsize=9,
        frameon=True,
        framealpha=0.9,
        title="State color (timeline)",
        title_fontsize=9,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        f"{label} profile - who they are (static attrs = observed) "
        "+ infection path (SEIR = model-derived)",
        fontsize=12.5,
        fontweight="bold",
        y=0.975,
    )
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return _confirm_png(out_path)


def _save_population_panel_standalone(
    draw_fn, result: dict, inf_mask: np.ndarray, out_path: Path
) -> bool:
    """하단 population 패널 1종을 단일 figure 로 저장(combined 와 동일 _draw_* 재사용).

    Args:
        draw_fn: ``_draw_age_sex_pyramid`` 등 ``(ax, result, inf_mask)`` 시그니처 helper.
        result: ``simulate_with_history`` 반환 dict.
        inf_mask: (N,) bool 감염 마스크.
        out_path: 저장할 PNG 절대경로.

    Returns:
        PNG 생성 성공(size>0)이면 True.

    Side effects: out_path 에 PNG write. figure close.
    """
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    draw_fn(ax, result, inf_mask)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return _confirm_png(out_path)


def build_figure() -> bool:
    """실 합성인구로 per-agent SEIR 시뮬 후 대표 agent 프로파일 + population 패널 렌더.

    파이프라인:
        1) ``simulate_with_history`` 로 서울 25구 합성 인구(관측 분포) + per-agent
           SEIR 궤적(모델-유래) 생성 (seed 고정 → 결정적).
        2) 감염자가 0명이면 정직히 skip(가짜 생성 금지) — 빈 리스트 반환.
        3) 대표 감염 agent 1-2명 추출(고위험·조기감염 우선) + population 요약.
        4) **단일패널(standalone) figure 들을 먼저 1장씩 저장**(요청="1장씩 보고 싶다"):
           - 대표 agent A 프로파일(카드+타임라인 한 장), 있으면 agent B 도.
           - population 4종(연령×성별 피라미드·기저질환·직업·연령별 발병률) 각 1장.
        5) 이어서 combined 2×4 grid figure 를 **동일 _draw_* helper 로** 렌더 →
           ``fig_agent_profile.png`` 저장(dpi=120, back-compat).

    Returns:
        생성에 성공한 PNG 절대경로(``Path``) 리스트. 데이터 부재로 skip 하면 빈 리스트.

    Performance: O(T_days * N) time, ~수십 MB peak. Side effects: FIG_DIR 에 여러 PNG
        write; ``generate_population`` 이 epi_real_seoul.db 를 read-only 로 연다(write 없음).
    Caller responsibility: 없음(단독 실행 가능).
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _setup_korean_font()

    print(
        f"[sim] simulate_with_history(N={N_AGENTS}, T_days={T_DAYS}, seed={SEED}) "
        f"— 서울 25구 합성인구(관측분포) + per-agent SEIR(모델)"
    )
    result = simulate_with_history(N_AGENTS, T_DAYS, seed=SEED, **SIM_KWARGS)

    inf_mask = _infected_mask(result)
    n_infected = int(inf_mask.sum())
    if n_infected == 0:
        print(
            "[SKIP] 시뮬 기간 동안 감염자(E/I)가 0명 — 대표 agent/감염자 분포를 "
            "정직히 그릴 수 없음(가짜 생성 금지). figure 미생성."
        )
        return []

    summary = population_summary(result)
    print(
        f"[sim] 감염자={n_infected:,}/{summary['n_agents']:,} "
        f"(attack_rate={summary['attack_rate']:.3f}, "
        f"peak_prevalence={summary['peak_prevalence']:.3f})"
    )

    reps = _pick_representative_infected(result, k=2)
    print(f"[sim] 대표 감염 agent = {reps}")
    traj_a = extract_agent_trajectory(result, reps[0]) if len(reps) >= 1 else None
    traj_b = extract_agent_trajectory(result, reps[1]) if len(reps) >= 2 else None

    # ---- 단일패널(standalone) figure 들을 먼저 1장씩 저장 ----
    written: list[Path] = []
    if traj_a is not None and _save_agent_profile_standalone(
        traj_a, "Representative agent A", OUTPUT_AGENT_A
    ):
        written.append(OUTPUT_AGENT_A)
    if traj_b is not None and _save_agent_profile_standalone(
        traj_b, "Representative agent B", OUTPUT_AGENT_B
    ):
        written.append(OUTPUT_AGENT_B)
    for draw_fn, out_path in (
        (_draw_age_sex_pyramid, OUTPUT_PYRAMID),
        (_draw_severity_dist, OUTPUT_SEVERITY),
        (_draw_occupation_dist, OUTPUT_OCCUPATION),
        (_draw_attack_rate_by_age, OUTPUT_ATTACK_RATE),
    ):
        if _save_population_panel_standalone(draw_fn, result, inf_mask, out_path):
            written.append(out_path)

    # ---- figure: 2 행 × 4 열 (상단=대표 agent, 하단=population) — back-compat ----
    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(
        2,
        4,
        height_ratios=[1.0, 1.05],
        hspace=0.42,
        wspace=0.32,
        left=0.05,
        right=0.985,
        top=0.90,
        bottom=0.07,
    )

    # Top: representative agent A (card + timeline)
    ax_a_card = fig.add_subplot(gs[0, 0])
    ax_a_tl = fig.add_subplot(gs[0, 1])
    _draw_agent_card(ax_a_card, traj_a, "Representative agent A")
    _draw_agent_timeline(ax_a_tl, traj_a, "Representative agent A")

    # Top: representative agent B (if present; else honestly left blank)
    ax_b_card = fig.add_subplot(gs[0, 2])
    ax_b_tl = fig.add_subplot(gs[0, 3])
    if traj_b is not None:
        _draw_agent_card(ax_b_card, traj_b, "Representative agent B")
        _draw_agent_timeline(ax_b_tl, traj_b, "Representative agent B")
    else:
        for ax in (ax_b_card, ax_b_tl):
            ax.axis("off")
        ax_b_card.text(
            0.5,
            0.5,
            "No second representative infected agent\n(only one infected agent)",
            ha="center",
            va="center",
            fontsize=11,
        )

    # 하단: population 패널
    ax_pyr = fig.add_subplot(gs[1, 0])
    ax_sev = fig.add_subplot(gs[1, 1])
    ax_occ = fig.add_subplot(gs[1, 2])
    ax_atk = fig.add_subplot(gs[1, 3])
    _draw_age_sex_pyramid(ax_pyr, result, inf_mask)
    _draw_severity_dist(ax_sev, result, inf_mask)
    _draw_occupation_dist(ax_occ, result, inf_mask)
    _draw_attack_rate_by_age(ax_atk, result, inf_mask)

    # 상태 색상 범례 (figure 레벨, 타임라인 해석용)
    fig.legend(
        handles=_state_legend_handles(),
        loc="upper right",
        ncol=5,
        fontsize=9.5,
        bbox_to_anchor=(0.985, 0.965),
        frameon=True,
        framealpha=0.9,
        title="State color (timeline)",
        title_fontsize=9.5,
    )

    fig.suptitle(
        "Agent profile in the epidemic simulation - 'who are the agents'\n"
        f"Seoul 25-district synthetic population N={summary['n_agents']:,} | "
        f"T={T_DAYS} days | seed={SEED}  |  "
        "static attrs = observed (KOSIS/HIRA/commute/business) / "
        "SEIR path = model-derived (agent-based kernel)",
        fontsize=15,
        fontweight="bold",
        y=0.985,
    )

    fig.savefig(OUTPUT_PNG, dpi=120, bbox_inches="tight")
    plt.close(fig)
    if _confirm_png(OUTPUT_PNG):
        written.append(OUTPUT_PNG)
    return written


def main() -> int:
    """엔트리포인트: 단일패널+통합 그림 생성 + 산출 검증. 성공 0 / skip·실패 1.

    Returns:
        프로세스 종료 코드(int). 0 = PNG ≥1장 생성 성공, 1 = skip 또는 전부 실패.
    """
    written = build_figure()
    if written:
        print(f"[done] fig_agent_profile 생성 완료 — {len(written)}개 PNG:")
        for p in written:
            print(f"        - {p}")
    else:
        print("[done] fig_agent_profile skip/실패")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
