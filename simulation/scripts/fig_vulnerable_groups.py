"""취약계층(vulnerable group) 위험 계층화 그림 생성기 — 실 데이터 전용.

"누가 더 위험한가"를 **관측(observed)** 과 **모델-유래(model-derived)** 두 축으로
계층화해 단일 figure(``fig_vulnerable_groups.png``, 3-panel)로 렌더링한다. 합성/가짜
데이터를 절대 만들지 않으며, 데이터가 없으면 정직히 skip(빈 panel + 사유 텍스트)한다.

panel ① — **관측** 연령군별 ILI rate (고위험 vs 청장년)
    SSOT = DB table ``sentinel_influenza`` (KDCA 표본감시, 7 연령군 × 주, 실측).
    영유아(0세·1-6세)·고령(65세 이상) 등 고위험군과 청장년(19-49세)의 시즌 평균
    ILI rate 를 막대로 비교. 고위험군은 색으로 강조. **관측 데이터** — 모델 아님.

panel ② — **모델-유래** 기저질환 severity별 결과 차등
    SSOT = ABM ``simulate_with_history`` (per-agent SEIR + severity 속성).
    기저질환 高(severity=high) vs 低(severity=low) agent 의 ① 누적 감염률(attack rate)
    ② 사망률(death proxy) 을 비교. kernel ``HIGH_SEVERITY_DELTA_SCALE=3.0`` vs
    ``_LOW_SEVERITY_DELTA_SCALE=0.5`` (= 6× 사망압 차등) 가 결과로 발현. **모델 산출** —
    관측 사망 데이터 아님(가짜 사망률 주장 금지, "model death proxy" 명시).

panel ③ — **모델-유래** 연령 × 기저질환 교차 위험
    SSOT = ABM ``simulate_with_history`` (per-agent age_band × severity 결합).
    연령군 × severity(low/high) 셀별 사망률(death proxy) 히트맵. severity 자체가
    DB 의 sex×age 기저질환 확률로 배정되므로(나이 들수록 高), 연령과 severity 의 복합
    위험이 한 그림에 드러난다. **모델 산출** 명시.

실행:
    .venv/bin/python -m simulation.scripts.fig_vulnerable_groups

출력 (조합 panel 이 구분 어려워 단일 panel 도 1개씩 별도 저장):
    simulation/results/figures/vulnerable_groups_observed_age_ili.png    (단일, dpi=130)
    simulation/results/figures/vulnerable_groups_severity_outcome.png    (단일, dpi=130)
    simulation/results/figures/vulnerable_groups_age_severity_cross.png  (단일, dpi=130)
    simulation/results/figures/fig_vulnerable_groups.png  (조합 3-panel, dpi=120, back-compat)

설계 규율 (ENGINEERING_PRINCIPLES.md):
    - DB read = ``read_only_connect`` 만 (저수준 직접 연결 금지). 가짜/합성 0.
    - ABM = ``simulate_with_history`` import만 (base 동역학 미재구현, 검증된 kernel 호출).
    - matplotlib Agg backend + 한글폰트 (AppleGothic → NanumGothic fallback).
    - 결정성: ORDER BY 명시 + 고정 seed(42). 동일 실행 → 비트 동일 figure.
    - 관측 vs 모델-유래 구분을 panel 제목/주석에 명시 (정직성).
    - 데이터 없으면 빈 panel + 사유 텍스트 (가짜 생성 금지).
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless 렌더링 (디스플레이 비종속)

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

from simulation.abm.agent_history import simulate_with_history
from simulation.abm.agent_kernel import (
    HIGH_SEVERITY_DELTA_SCALE,
    _LOW_SEVERITY_DELTA_SCALE,
)
from simulation.database import read_only_connect

# ---------------------------------------------------------------------------
# 경로 SSOT (단일 코드 루트 — ENGINEERING_PRINCIPLES.md §4 KISS)
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parents[2]  # simulation/scripts/<this>.py → repo root
FIG_DIR = PROJECT_ROOT / "simulation" / "results" / "figures"
OUT_PNG = FIG_DIR / "fig_vulnerable_groups.png"
# 단일 panel 산출(조합 panel 이 구분 어려워 1개씩 별도 저장 — 의미 이름)
OUT_OBSERVED_AGE = FIG_DIR / "vulnerable_groups_observed_age_ili.png"
OUT_SEVERITY_OUTCOME = FIG_DIR / "vulnerable_groups_severity_outcome.png"
OUT_AGE_SEVERITY_CROSS = FIG_DIR / "vulnerable_groups_age_severity_cross.png"

# 모델 상태 코드 (agent_history STATE_LABELS = S,E,I,R,V,D)
_STATE_D = 5  # 사망(death)
_STATE_E, _STATE_I, _STATE_R = 1, 2, 3  # ever-infected 판별용(E/I/R/D)

# 관측 연령군 분류 (sentinel_influenza age_group SSOT)
# 고위험 = 영유아(0세·1-6세) + 고령(65세 이상); 청장년 reference = 19-49세
_AGE_GROUP_ORDER = (
    "0세",
    "1-6세",
    "7-12세",
    "13-18세",
    "19-49세",
    "50-64세",
    "65세 이상",
)
_HIGH_RISK_AGE_GROUPS = frozenset({"0세", "1-6세", "65세 이상"})

# Display-only English mapping for age-group tick labels (DB values stay Korean).
_AGE_DISPLAY = {
    "0세": "0 yr",
    "1-6세": "1-6 yr",
    "7-12세": "7-12 yr",
    "13-18세": "13-18 yr",
    "19-49세": "19-49 yr",
    "50-64세": "50-64 yr",
    "65세 이상": "65+ yr",
}

# 모델 연령 밴드 라벨 (agent_history AGE_BAND_LABELS SSOT, 10년 단위 7밴드)
_MODEL_AGE_LABELS = ("0-9", "10-19", "20-29", "30-39", "40-49", "50-59", "60+")
_MODEL_HIGH_RISK_AGE_IDX = frozenset({0, 6})  # 0-9(영유아) + 60+(고령)

# 색상 (취약계층 강조)
_C_HIGH = "#c0392b"  # 고위험 — 적색
_C_REF = "#7f8c8d"   # 청장년/저위험 — 회색
_C_HIGH_SEV = "#e74c3c"
_C_LOW_SEV = "#3498db"


def _setup_korean_font() -> str:
    """matplotlib 전역 한글폰트 설정 (AppleGothic → NanumGothic fallback).

    Returns:
        실제 적용된 폰트 패밀리 이름 (str). 한글폰트 미발견 시 "DejaVu Sans"
        (한글 깨짐 경고 로그 출력).

    Side effects: ``plt.rcParams["font.family"]``, ``["axes.unicode_minus"]``
        전역 변경.
    """
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False  # 음수 부호 깨짐 방지
    print("[font] 적용 폰트 = DejaVu Sans")
    return "DejaVu Sans"


def _confirm_png(path: Path) -> bool:
    """PNG 가 실제로 생성되었고 비어있지 않은지 검증.

    Args:
        path: 검증할 PNG 절대경로.

    Returns:
        존재 + ``size > 0`` 이면 True, 아니면 False (로그 출력).

    Side effects: stdout 로그.
    """
    if path.exists() and path.stat().st_size > 0:
        print(f"[OK]   {path}  ({path.stat().st_size:,} bytes)")
        return True
    print(f"[FAIL] {path}  (미생성 또는 0 bytes)")
    return False


# ---------------------------------------------------------------------------
# 데이터 로더
# ---------------------------------------------------------------------------
def load_observed_age_ili() -> dict[str, float]:
    """관측 연령군별 평균 ILI rate 를 DB(sentinel_influenza)에서 로드.

    KDCA 표본감시 ``sentinel_influenza`` (7 연령군 × 주, 전 시즌)의 연령군별 평균
    ILI rate 를 계산한다. **관측 데이터** — 모델 산출 아님.

    Returns:
        {age_group: mean_ili_rate} dict. 데이터가 전혀 없으면 빈 dict.
        age_group 키는 DB 의 한국어 라벨 그대로("0세", "65세 이상" 등).

    Raises:
        OperationalError: DB 파일 부재 시 (read_only_connect 가 fail-loud).

    Performance: O(rows) 단일 GROUP BY 쿼리.
    Side effects: read-only DB fd 1개 open/close (write 없음).
    Caller responsibility: 없음 (read-only).
    """
    out: dict[str, float] = {}
    with contextlib.closing(read_only_connect()) as con:
        cur = con.cursor()
        rows = cur.execute(
            "SELECT age_group, AVG(ili_rate) "
            "FROM sentinel_influenza "
            "WHERE ili_rate IS NOT NULL "
            "GROUP BY age_group "
            "ORDER BY age_group"
        ).fetchall()
    for age_group, mean_rate in rows:
        if mean_rate is not None:
            out[str(age_group)] = float(mean_rate)
    return out


def run_abm_severity(
    *, N: int = 6000, T_days: int = 160, seed: int = 42
) -> dict | None:
    """ABM per-agent SEIR 시뮬레이션 → severity × age 결과 집계 (모델-유래).

    ``simulate_with_history`` (검증된 kernel) 를 호출해 per-agent severity(low/high)·
    age_band 별 누적 감염률·사망률(death proxy)을 산출한다. 사망압은 kernel 의
    ``HIGH_SEVERITY_DELTA_SCALE`` (high) vs ``_LOW_SEVERITY_DELTA_SCALE`` (low) 로
    severity 에 의해 변조된다. **모델 산출** — 관측 사망 데이터 아님.

    Args:
        N: agent 수. 결과 안정성 위해 수천 권장(severity high 비율이 낮아 충분 표본 필요).
        T_days: 시뮬 일수. 유행이 충분히 전개되도록(감염→사망) 충분히 길게.
        seed: 결정성 시드(인구 생성 + kernel 체이닝). 동일 seed → 비트 동일.

    Returns:
        dict 또는 None(시뮬에 사망/감염이 전혀 없어 차등이 정의 불가하면 None):
          - ``n``: int — 총 agent 수.
          - ``sev_n``: (2,) — [low, high] 각 agent 수.
          - ``sev_attack``: (2,) — [low, high] 누적 감염률(E/I/R/D ever / n).
          - ``sev_death``: (2,) — [low, high] 사망률(최종 D / n) — model death proxy.
          - ``age_labels``: list[str] — 모델 연령 밴드 라벨(7개).
          - ``age_sev_death``: (7, 2) — [age_band, sev] 사망률 행렬(빈 셀=NaN).
          - ``age_sev_n``: (7, 2) — [age_band, sev] agent 수(표본 크기 annotation).
          - ``death_scale``: (low, high) 사망압 배수 튜플(주석용).

    Side effects: ``simulate_with_history`` 가 epi_real_seoul.db 를 read-only 로 연다
        (DB write 없음). 파일시스템 write 없음. stderr 에 kernel 진행 로그.
    Caller responsibility: 없음.
    """
    res = simulate_with_history(
        N=N,
        T_days=T_days,
        seed=seed,
        beta=0.45,
        sigma=0.3,
        gamma=0.2,
        delta=0.01,        # 기저 사망 hazard (severity 로 ×3.0 / ×0.5 변조)
        import_rate=1e-4,  # off-season 확률적 소멸 방지
    )
    hist = np.asarray(res["history_state"])  # (T_days, N)
    attrs = res["attrs"]
    severity = np.asarray(attrs["severity"], dtype=np.int64)  # 0=low, 1=high
    age_band = np.asarray(attrs["age_band"], dtype=np.int64)   # 0..6
    n = int(severity.shape[0])

    final = hist[-1]  # (N,) 종료 시점 상태
    ever_inf = ((hist == _STATE_E) | (hist == _STATE_I) | (hist == _STATE_R) | (hist == _STATE_D)).any(axis=0)
    died = final == _STATE_D

    # 차등이 정의 불가(감염·사망 전무)하면 정직 skip
    if not ever_inf.any():
        print("[SKIP] ABM: 감염 발생 0 — severity 차등 정의 불가")
        return None

    n_age = len(_MODEL_AGE_LABELS)
    sev_n = np.array([int((severity == s).sum()) for s in (0, 1)], dtype=np.int64)
    sev_attack = np.full(2, np.nan)
    sev_death = np.full(2, np.nan)
    for s in (0, 1):
        m = severity == s
        if m.any():
            sev_attack[s] = float(ever_inf[m].mean())
            sev_death[s] = float(died[m].mean())

    age_sev_death = np.full((n_age, 2), np.nan)
    age_sev_n = np.zeros((n_age, 2), dtype=np.int64)
    for a in range(n_age):
        for s in (0, 1):
            m = (age_band == a) & (severity == s)
            cnt = int(m.sum())
            age_sev_n[a, s] = cnt
            if cnt > 0:
                age_sev_death[a, s] = float(died[m].mean())

    return {
        "n": n,
        "sev_n": sev_n,
        "sev_attack": sev_attack,
        "sev_death": sev_death,
        "age_labels": list(_MODEL_AGE_LABELS),
        "age_sev_death": age_sev_death,
        "age_sev_n": age_sev_n,
        "death_scale": (float(_LOW_SEVERITY_DELTA_SCALE), float(HIGH_SEVERITY_DELTA_SCALE)),
    }


# ---------------------------------------------------------------------------
# Panel 렌더러
# ---------------------------------------------------------------------------
def _panel_observed_age(ax, observed: dict[str, float], *, standalone: bool = False) -> None:
    """panel ① — 관측 연령군별 평균 ILI rate (고위험 강조).

    Args:
        ax: 그릴 matplotlib Axes.
        observed: {age_group: mean_ili_rate}. 빈 dict 면 사유 텍스트만 표시.
        standalone: True 면 단독 figure 용 — "①" prefix 제거(조합 panel 구분용 X).
    """
    title = "Observed: ILI rate by age group" if standalone else "(1) Observed: ILI rate by age group"
    if not observed:
        ax.text(
            0.5, 0.5,
            "No observed age-specific ILI data\n(sentinel_influenza is empty)",
            ha="center", va="center", transform=ax.transAxes, fontsize=11,
        )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")
        return

    labels = [g for g in _AGE_GROUP_ORDER if g in observed]
    # 정의에 없는 라벨도 누락 없이 뒤에 붙임(정렬 결정성)
    labels += sorted(g for g in observed if g not in _AGE_GROUP_ORDER)
    vals = [observed[g] for g in labels]
    colors = [_C_HIGH if g in _HIGH_RISK_AGE_GROUPS else _C_REF for g in labels]

    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels([_AGE_DISPLAY.get(g, g) for g in labels], fontsize=10)
    ax.invert_yaxis()  # 0세가 위
    ax.set_xlabel("Mean ILI rate (full-season average)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    for yi, v in zip(y, vals):
        ax.text(v, yi, f" {v:.1f}", va="center", fontsize=9)
    # 범례(고위험 vs 청장년)
    from matplotlib.patches import Patch
    ax.legend(
        handles=[
            Patch(color=_C_HIGH, label="High-risk (infants/young children, elderly)"),
            Patch(color=_C_REF, label="Working-age adults / other"),
        ],
        loc="lower right", fontsize=8, framealpha=0.9,
    )
    ax.text(
        0.01, -0.16,
        "Source: KDCA sentinel surveillance (sentinel_influenza). Observed data.",
        transform=ax.transAxes, fontsize=7.5, color="#555555",
    )


def _panel_severity_outcome(ax, abm: dict | None, *, standalone: bool = False) -> None:
    """panel ② — 모델 severity(기저질환)별 감염률·사망률 차등.

    Args:
        ax: 그릴 matplotlib Axes.
        abm: ``run_abm_severity`` 산출 dict 또는 None(감염 0 → 사유 텍스트).
        standalone: True 면 단독 figure 용 — "②" prefix 제거.
    """
    title = "Model: outcome by comorbidity severity" if standalone else "(2) Model: outcome by comorbidity severity"
    ax.set_title(title, fontsize=12, fontweight="bold")
    if abm is None:
        ax.text(
            0.5, 0.5, "No ABM output (zero infections)",
            ha="center", va="center", transform=ax.transAxes, fontsize=11,
        )
        ax.axis("off")
        return

    attack = abm["sev_attack"]   # [low, high]
    death = abm["sev_death"]     # [low, high]
    sev_n = abm["sev_n"]
    lo_scale, hi_scale = abm["death_scale"]

    groups = ["Attack rate", "Death rate\n(death proxy)"]
    x = np.arange(len(groups))
    w = 0.36
    low_vals = [attack[0], death[0]]
    high_vals = [attack[1], death[1]]

    ax.bar(x - w / 2, low_vals, w, label=f"Comorbidity low (n={sev_n[0]})", color=_C_LOW_SEV)
    ax.bar(x + w / 2, high_vals, w, label=f"Comorbidity high (n={sev_n[1]})", color=_C_HIGH_SEV)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=9)
    ax.set_ylabel("Proportion", fontsize=10)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    for xi, lv, hv in zip(x, low_vals, high_vals):
        if np.isfinite(lv):
            ax.text(xi - w / 2, lv, f"{lv:.3f}", ha="center", va="bottom", fontsize=8)
        if np.isfinite(hv):
            ax.text(xi + w / 2, hv, f"{hv:.3f}", ha="center", va="bottom", fontsize=8)

    # 사망압 차등 비(高/低) 주석
    ratio_txt = ""
    if np.isfinite(death[0]) and death[0] > 0 and np.isfinite(death[1]):
        ratio_txt = f"  Death-rate ratio (high/low) ≈ {death[1] / death[0]:.1f}×"
    ax.text(
        0.01, -0.18,
        f"Model output (death proxy, not observed deaths). "
        f"Kernel mortality-pressure multiplier high={hi_scale:.1f} vs low={lo_scale:.1f} (={hi_scale / lo_scale:.0f}×).{ratio_txt}",
        transform=ax.transAxes, fontsize=7.5, color="#555555",
    )


def _panel_age_severity_cross(ax, fig, abm: dict | None, *, standalone: bool = False) -> None:
    """panel ③ — 모델 연령 × 기저질환 교차 사망률 히트맵.

    Args:
        ax: 그릴 matplotlib Axes.
        fig: colorbar 부착용 Figure(ax 의 부모).
        abm: ``run_abm_severity`` 산출 dict 또는 None(감염 0 → 사유 텍스트).
        standalone: True 면 단독 figure 용 — "③" prefix 제거.
    """
    title = "Model: age x comorbidity cross risk" if standalone else "(3) Model: age x comorbidity cross risk"
    ax.set_title(title, fontsize=12, fontweight="bold")
    if abm is None:
        ax.text(
            0.5, 0.5, "No ABM output (zero infections)",
            ha="center", va="center", transform=ax.transAxes, fontsize=11,
        )
        ax.axis("off")
        return

    mat = np.asarray(abm["age_sev_death"])  # (7, 2) 사망률
    cnt = np.asarray(abm["age_sev_n"])      # (7, 2) 표본
    age_labels = abm["age_labels"]

    masked = np.ma.masked_invalid(mat)
    cmap = plt.get_cmap("Reds").copy()
    cmap.set_bad("#dddddd")  # 표본 없는 셀 = 회색
    im = ax.imshow(masked, aspect="auto", cmap=cmap, origin="upper")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Comorbidity low", "Comorbidity high"], fontsize=10)
    ax.set_yticks(np.arange(len(age_labels)))
    ax.set_yticklabels(age_labels, fontsize=9)
    ax.set_ylabel("Age band (model, 10-year bins)", fontsize=9)

    # 고위험 연령(0-9, 60+) y축 라벨 강조
    for i, lbl in enumerate(ax.get_yticklabels()):
        if i in _MODEL_HIGH_RISK_AGE_IDX:
            lbl.set_color(_C_HIGH)
            lbl.set_fontweight("bold")

    # 셀에 사망률 + 표본 수 텍스트
    for a in range(mat.shape[0]):
        for s in range(2):
            v = mat[a, s]
            n_cell = cnt[a, s]
            if np.isfinite(v):
                txt = f"{v:.3f}\n(n={n_cell})"
                # 명도에 따라 텍스트 색 결정
                vmax = np.nanmax(mat) if np.isfinite(np.nanmax(mat)) else 1.0
                tcol = "white" if (vmax > 0 and v > 0.6 * vmax) else "black"
            else:
                txt = f"n={n_cell}"
                tcol = "#666666"
            ax.text(s, a, txt, ha="center", va="center", fontsize=7.5, color=tcol)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Death rate (death proxy)", fontsize=8)
    ax.text(
        0.01, -0.16,
        "Model output. Severity assigned by DB sex x age comorbidity probability (the older, the higher).",
        transform=ax.transAxes, fontsize=7.5, color="#555555",
    )


# ---------------------------------------------------------------------------
# 메인
# ---------------------------------------------------------------------------
def make_figure() -> list[Path] | None:
    """취약계층 위험 계층화 figure 생성 → 단일 panel 3개 + 조합 1개 (총 4 PNG).

    panel ①(관측 연령 ILI) + ②(모델 severity 결과) + ③(모델 연령×severity 교차)를
    ① 먼저 각각 단독 figure 로 저장(조합 panel 이 구분 어려워 "1개씩"), ② 그다음
    기존 3-panel 조합 figure 를 동일하게 저장(back-compat)한다. ① 데이터 부재 시 해당
    panel 만 정직 skip(빈 panel + 사유), ②③ 둘 다 ABM 산출이면 함께 skip 가능. 최소
    1개 panel 이라도 실데이터가 있으면 figure 를 저장한다.

    Returns:
        저장된 PNG 경로 리스트(단일 3개 + 조합 1개 = 4개, 모두 절대경로) 또는
        None(모든 panel 데이터 부재로 저장 안 함).

    Side effects: ``FIG_DIR`` 에 PNG 4개 write. read-only DB read(관측 + ABM 인구).
        stdout 로그.
    Caller responsibility: 없음.
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("[load] 관측 연령별 ILI (sentinel_influenza)...")
    observed = load_observed_age_ili()
    print(f"       연령군 {len(observed)}개 로드")

    print("[run ] ABM per-agent SEIR (severity × age)...")
    abm = run_abm_severity()
    if abm is not None:
        print(
            f"       N={abm['n']}  severity[low,high]={abm['sev_n'].tolist()}  "
            f"attack={np.round(abm['sev_attack'], 3).tolist()}  "
            f"death={np.round(abm['sev_death'], 4).tolist()}"
        )

    if not observed and abm is None:
        print("[SKIP] 관측·모델 양쪽 데이터 모두 부재 — figure 미생성 (정직 skip)")
        return None

    written: list[Path] = []

    # ── 1) 단일 panel 3개 (1개씩 별도 저장 — 조합 panel 구분 어려움 해소) ──
    fig1, ax1 = plt.subplots(figsize=(7.5, 5.6))
    _panel_observed_age(ax1, observed, standalone=True)
    fig1.tight_layout()
    fig1.savefig(OUT_OBSERVED_AGE, dpi=130, bbox_inches="tight")
    plt.close(fig1)
    written.append(OUT_OBSERVED_AGE)

    fig2, ax2 = plt.subplots(figsize=(7.5, 5.6))
    _panel_severity_outcome(ax2, abm, standalone=True)
    fig2.tight_layout()
    fig2.savefig(OUT_SEVERITY_OUTCOME, dpi=130, bbox_inches="tight")
    plt.close(fig2)
    written.append(OUT_SEVERITY_OUTCOME)

    fig3, ax3 = plt.subplots(figsize=(7.5, 5.6))
    _panel_age_severity_cross(ax3, fig3, abm, standalone=True)
    fig3.tight_layout()
    fig3.savefig(OUT_AGE_SEVERITY_CROSS, dpi=130, bbox_inches="tight")
    plt.close(fig3)
    written.append(OUT_AGE_SEVERITY_CROSS)

    # ── 2) 조합 3-panel figure (back-compat — 기존과 동일) ──
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.6))
    fig.suptitle(
        "Vulnerable-group risk stratification: who is more at risk  "
        "((1) observed age ILI  vs  (2)(3) model severity outcome)",
        fontsize=14, fontweight="bold",
    )

    _panel_observed_age(axes[0], observed)
    _panel_severity_outcome(axes[1], abm)
    _panel_age_severity_cross(axes[2], fig, abm)

    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    plt.close(fig)
    written.append(OUT_PNG)

    return written


def main() -> int:
    """엔트리포인트: 한글폰트 설정 → figure 생성 → 산출 검증.

    단일 panel 3개 + 조합 1개(총 4개)를 생성하고 각각 size>0 검증한다.

    Returns:
        0 = 성공(생성된 PNG 전부 size>0 확인), 1 = 데이터 부재 skip, 2 = 일부 생성 실패.
    """
    _setup_korean_font()
    paths = make_figure()
    if paths is None:
        print("[DONE] 데이터 부재로 figure 미생성 (정직 skip)")
        return 1
    all_ok = all(_confirm_png(p) for p in paths)
    print(f"[DONE] 총 {len(paths)}개 figure 작성 (단일 panel 3 + 조합 1)")
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
