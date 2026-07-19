"""중재/정책 디테일 그림 생성기 — ABM counterfactual 산출 재사용 (실 산출 전용).

서울 감염병 ABM 의 **중재/정책 counterfactual** 을 한 장으로 시각화한다. 세 패널 모두
`simulation.abm` 의 개입 모듈(counterfactual.py · behavioural.py · adaptive_allocation.py)이
이미 산출한 **실 ABM 결과 JSON/npz 만** 소비한다. 합성/가짜 데이터를 절대 만들지 않으며,
필요한 산출이 없으면 정직히 skip + 로그 후 종료한다.

⚠ 모든 곡선·막대는 **모델-유래(ABM counterfactual 시나리오)** 이며 관측이 아니다 —
   "counterfactual = 모델 what-if" 임을 제목/주석에 명시한다 (held-out 예측 아님).

패널 구성 (fig_intervention_policy.png):
    A) 백신 시나리오 — uniform vs 표적(고접촉/고령) attack rate 차등 (오차막대=95% CI).
       이질(heterogeneous) ABM 은 표적-고접촉이 attack rate 를 가장 낮추는 반면,
       동질(homogeneous, mean-field) 대조군은 배분 전략이 결과를 가르지 못함(구조적 무차별)
       을 나란히 보여 ABM 의 정책적 부가가치를 드러낸다.
       SSOT = abm_v1/counterfactual.json (simulation/abm/counterfactual.py 산출).

    B) NPI/행동(compliance) 효과 — 행동 OFF(정적 베이스라인) vs 행동 ON(위험인지·피로
       기반 자발적 거리두기) 의 도시 전역 감염자 곡선 I(t) + 정점 이동(%) 주석.
       SSOT = abm_v1/trajectory.npz (city_I_off/on) + regime_rebound.json (앙상블 정점이동 CI).
       simulation/abm/behavioural.py run_rebound_scenario 산출.

    C) 표적개입 vs 무개입 counterfactual — 같은 dose budget 에서 전략별 dose 당 회피
       감염수(infections averted/dose), 이질 vs 동질 대조. 0(무개입 대비 이득 없음) 기준선과
       함께 표적-고접촉의 우위(전파차단·간접보호; Medlock-Galvani 2009 Science)를 정량화.
       SSOT = abm_v1/counterfactual.json 의 analysis 블록.

실행:
    .venv/bin/python -m simulation.scripts.fig_intervention_policy

출력 ("한 번에 하나씩" 보기용 단일-패널 PNG 3종 + 기존 조합 PNG):
    simulation/results/figures/intervention_policy_vaccination.png  (Panel A, dpi=130)
    simulation/results/figures/intervention_policy_behaviour.png    (Panel B, dpi=130)
    simulation/results/figures/intervention_policy_averted.png      (Panel C, dpi=130)
    simulation/results/figures/fig_intervention_policy.png          (조합 3-패널, dpi=120, back-compat)

설계 규율 (ENGINEERING_PRINCIPLES.md):
    - 실 ABM 산출만 (기존 JSON/npz 재사용). 가짜/합성 금지. 데이터 없으면 skip + 로그.
    - DB 접근 필요 시 read_only_connect 만 (저수준 연결 직접 금지).
    - matplotlib Agg backend + 한글폰트 (AppleGothic → NanumGothic fallback).
    - 결정성: 임의 seed 없음. 입력 JSON/npz 가 결정성 산출 → 그림도 결정적.
    - 모델-유래(관측 아님) 명시.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")  # headless 렌더링 (디스플레이 비종속)

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# 경로 SSOT (단일 코드 루트 — ENGINEERING_PRINCIPLES.md §4 KISS)
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parents[2]  # simulation/scripts/fig_intervention_policy.py → repo root
FIG_DIR = PROJECT_ROOT / "simulation" / "results" / "figures"
OUT_PNG = FIG_DIR / "fig_intervention_policy.png"

# 단일 패널(standalone) PNG — "한 번에 하나씩" 보기용. 의미 있는 이름으로 분리 저장.
# 조합 그림(OUT_PNG)은 그대로 유지(back-compat) + 패널별 독립 PNG 추가.
OUT_VACCINATION = FIG_DIR / "intervention_policy_vaccination.png"   # Panel A
OUT_BEHAVIOUR = FIG_DIR / "intervention_policy_behaviour.png"       # Panel B
OUT_AVERTED = FIG_DIR / "intervention_policy_averted.png"           # Panel C

# ABM counterfactual 산출 후보 경로 (최신 active 우선 → archive fallback).
# 데이터 기반 결정: 존재하는 첫 디렉터리를 SSOT 로 사용한다(고정 경로 hardcode 회피).
_ABM_DIR_CANDIDATES = (
    PROJECT_ROOT / "simulation" / "results" / "abm_v1",
    PROJECT_ROOT / "simulation" / "results" / "_archive_fullrun_20260624_140351" / "abm_v1",
    PROJECT_ROOT / "_archive_stale_sweep_20260618_205715" / "results" / "abm_v1",
)

# 전략 라벨 (ubiquitous language: counterfactual.py STRATEGIES 와 동일 어휘)
_STRATEGY_KO = {
    "none": "No intervention",
    "uniform": "Uniform allocation",
    "target_elderly": "Targeted: elderly",
    "target_high_contact": "Targeted: high-contact",
}
# 색상 (전략별 일관)
_STRATEGY_COLOR = {
    "none": "#9e9e9e",
    "uniform": "#1f77b4",
    "target_elderly": "#ff7f0e",
    "target_high_contact": "#2ca02c",
}


def _setup_korean_font() -> str:
    """matplotlib 전역 한글폰트 설정 (AppleGothic → NanumGothic fallback).

    Returns:
        실제 적용된 폰트 패밀리 이름 (str). 한글폰트 미발견 시 "DejaVu Sans"
        (한글 깨짐 경고 로그 출력).

    Side effects: plt.rcParams["font.family"], ["axes.unicode_minus"] 전역 변경.
    """
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False  # 음수 부호 깨짐 방지
    print(f"[font] 적용 폰트 = DejaVu Sans")
    return "DejaVu Sans"


def _find_abm_dir() -> Optional[Path]:
    """ABM counterfactual 산출이 있는 첫 후보 디렉터리 반환 (없으면 None).

    Returns:
        counterfactual.json 을 포함한 첫 후보 Path, 또는 None (모두 부재 시).
    """
    for d in _ABM_DIR_CANDIDATES:
        if (d / "counterfactual.json").exists():
            return d
    return None


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    """JSON 안전 로드 (부재/파손 시 None + 로그)."""
    if not path.exists():
        print(f"[skip] 부재: {path}")
        return None
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[skip] 로드 실패 {path}: {exc}")
        return None


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


# ---------------------------------------------------------------------------
# Panel A — 백신 시나리오: attack rate ± 95% CI (이질 vs 동질)
# ---------------------------------------------------------------------------
def _panel_vaccination(ax, cf: dict[str, Any], standalone: bool = False) -> bool:
    """백신 배분 전략별 attack rate (95% CI) 막대 — 이질 vs 동질 ABM.

    Args:
        ax: matplotlib Axes.
        cf: counterfactual.json 전체 dict (results/metadata/analysis).
        standalone: True 면 단독 PNG 용(제목 "A." 접두어 제거), False 면 조합 그림용.

    Returns:
        그렸으면 True, 필요한 키 부재 시 False.

    Side effects: ax 에 그룹 막대 + 오차막대 렌더.
    """
    res = cf.get("results", {})
    het = res.get("heterogeneous")
    hom = res.get("homogeneous")
    if not het or not hom:
        print("[skip] Panel A: results.heterogeneous/homogeneous 부재")
        return False

    strategies = [s for s in ("none", "uniform", "target_elderly", "target_high_contact")
                  if s in het and s in hom]
    x = np.arange(len(strategies), dtype=float)
    width = 0.38

    def _vals(cell: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        ar = np.array([cell[s]["attack_rate"] for s in strategies], dtype=float)
        # 95% CI 를 비대칭 오차막대로 (yerr=[lower, upper])
        lo = np.array([cell[s]["attack_rate"] - cell[s]["attack_rate_ci95"][0]
                       for s in strategies], dtype=float)
        hi = np.array([cell[s]["attack_rate_ci95"][1] - cell[s]["attack_rate"]
                       for s in strategies], dtype=float)
        return ar * 100.0, np.vstack([lo, hi]) * 100.0  # % 단위

    het_ar, het_err = _vals(het)
    hom_ar, hom_err = _vals(hom)

    ax.bar(x - width / 2, het_ar, width, yerr=het_err, capsize=3,
           color="#2166ac", alpha=0.92, label="Heterogeneous ABM (realistic population)",
           error_kw={"elinewidth": 1.1, "ecolor": "#222"})
    ax.bar(x + width / 2, hom_ar, width, yerr=hom_err, capsize=3,
           color="#b2b2b2", alpha=0.92, label="Homogeneous control (mean-field)",
           error_kw={"elinewidth": 1.1, "ecolor": "#222"})

    # 이질 ABM 최소 attack rate 전략 강조
    best_i = int(np.argmin(het_ar))
    ax.annotate("Lowest attack rate", xy=(x[best_i] - width / 2, het_ar[best_i]),
                xytext=(x[best_i] - width / 2, het_ar[best_i] + 9),
                ha="center", fontsize=8, color="#1a5276",
                arrowprops={"arrowstyle": "->", "color": "#1a5276", "lw": 1.0})

    ax.set_xticks(x)
    ax.set_xticklabels([_STRATEGY_KO.get(s, s) for s in strategies], fontsize=8.5)
    ax.set_ylabel("Attack rate (%)", fontsize=9)
    budget = cf.get("metadata", {}).get("budget")
    K = cf.get("metadata", {}).get("K")
    sub = f"Same vaccine budget={budget:,} doses, K={K} seeds" if budget else ""
    prefix = "" if standalone else "A. "
    ax.set_title(f"{prefix}Vaccine allocation scenario — attack rate ±95% CI\n{sub}", fontsize=9.5, loc="left")
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(het_ar.max(), hom_ar.max()) * 1.28)
    return True


# ---------------------------------------------------------------------------
# Panel B — NPI/행동(compliance) 효과: I(t) off vs on + 정점이동
# ---------------------------------------------------------------------------
def _panel_behaviour(ax, abm_dir: Path, standalone: bool = False) -> bool:
    """행동 OFF vs ON 의 도시 전역 감염자 곡선 I(t) + 정점이동(%) 주석.

    Args:
        ax: matplotlib Axes.
        abm_dir: trajectory.npz / rebound_summary.json 이 위치한 디렉터리.
        standalone: True 면 단독 PNG 용(제목 "B." 접두어 제거), False 면 조합 그림용.

    Returns:
        그렸으면 True, 궤적 부재 시 False.

    Side effects: ax 에 두 곡선 + 정점이동 화살표 렌더.
    """
    traj_path = abm_dir / "trajectory.npz"
    if not traj_path.exists():
        print(f"[skip] Panel B: 부재 {traj_path}")
        return False
    try:
        # behavioural.py 가 numpy.savez 로 저장한 순수 float 배열 (object array 아님).
        z = np.load(traj_path, allow_pickle=False)
        days = np.asarray(z["days"], dtype=float)
        I_off = np.asarray(z["city_I_off"], dtype=float)
        I_on = np.asarray(z["city_I_on"], dtype=float)
    except (OSError, KeyError, ValueError) as exc:
        print(f"[skip] Panel B: trajectory.npz 로드 실패: {exc}")
        return False

    ax.plot(days, I_off / 1e3, color="#9e9e9e", lw=1.8,
            label="Behavior OFF (static baseline)")
    ax.plot(days, I_on / 1e3, color="#c0392b", lw=1.8,
            label="Behavior ON (voluntary distancing)")
    ax.fill_between(days, I_on / 1e3, I_off / 1e3,
                    where=(I_off >= I_on), color="#c0392b", alpha=0.10,
                    label="Burden averted by intervention")

    peak_off = float(I_off.max())
    peak_on = float(I_on.max())
    d_off = float(days[int(np.argmax(I_off))])
    d_on = float(days[int(np.argmax(I_on))])
    shift_pct = 100.0 * (peak_on - peak_off) / peak_off if peak_off > 0 else 0.0

    # 정점 이동 주석 + 앙상블 CI (regime_rebound.json 있으면 병기)
    summ = _load_json(abm_dir / "regime_rebound.json")
    ci_txt = ""
    if summ and "rebound" in summ:
        rb = summ["rebound"]
        mean = rb.get("peak_shift_pct_mean")
        ci = rb.get("peak_shift_pct_ci95")
        if mean is not None and ci:
            ci_txt = (f"\nEnsemble peak reduction {mean:+.0f}% "
                      f"[95% CI {ci[0]:+.0f}, {ci[1]:+.0f}]")

    ax.annotate(
        f"Peak {shift_pct:+.0f}%{ci_txt}",
        xy=(d_off, peak_off / 1e3),
        xytext=(d_off + 18, peak_off / 1e3 * 0.74),
        fontsize=8, color="#7b241c", ha="left",
        arrowprops={"arrowstyle": "->", "color": "#7b241c", "lw": 1.1},
    )
    ax.scatter([d_off, d_on], [peak_off / 1e3, peak_on / 1e3],
               color=["#9e9e9e", "#c0392b"], s=22, zorder=5)

    ax.set_xlabel("Epidemic time (days)", fontsize=9)
    ax.set_ylabel("City-wide infections I(t)  (thousands)", fontsize=9)
    prefix = "" if standalone else "B. "
    ax.set_title(f"{prefix}NPI/behavior (compliance) effect — behavior OFF vs ON curves",
                 fontsize=9.5, loc="left")
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    ax.grid(alpha=0.3)
    return True


# ---------------------------------------------------------------------------
# Panel C — 표적개입 vs 무개입 counterfactual: dose 당 회피 감염수
# ---------------------------------------------------------------------------
def _panel_averted(ax, cf: dict[str, Any], standalone: bool = False) -> bool:
    """전략별 dose 당 회피 감염수 (이질 vs 동질) — 무개입 대비 counterfactual 이득.

    Args:
        ax: matplotlib Axes.
        cf: counterfactual.json 전체 dict.
        standalone: True 면 단독 PNG 용(제목 "C." 접두어 제거), False 면 조합 그림용.

    Returns:
        그렸으면 True, analysis 부재 시 False.

    Side effects: ax 에 그룹 막대 + 0 기준선 렌더.
    """
    analysis = cf.get("analysis", {})
    het = analysis.get("heterogeneous", {}).get("per_strategy")
    hom = analysis.get("homogeneous", {}).get("per_strategy")
    if not het or not hom:
        print("[skip] Panel C: analysis.per_strategy 부재")
        return False

    strategies = [s for s in ("uniform", "target_elderly", "target_high_contact")
                  if s in het and s in hom]
    x = np.arange(len(strategies), dtype=float)
    width = 0.38

    het_v = np.array([het[s]["infections_averted_per_dose"] for s in strategies], dtype=float)
    hom_v = np.array([hom[s]["infections_averted_per_dose"] for s in strategies], dtype=float)

    ax.bar(x - width / 2, het_v, width, color="#2166ac", alpha=0.92,
           label="Heterogeneous ABM")
    ax.bar(x + width / 2, hom_v, width, color="#b2b2b2", alpha=0.92,
           label="Homogeneous control")
    ax.axhline(0.0, color="#444", lw=0.8)

    # 이질 ABM 최대 (표적-고접촉이 보통 1위) 강조
    best_i = int(np.argmax(het_v))
    ax.annotate("Optimal transmission blocking\n(indirect protection)",
                xy=(x[best_i] - width / 2, het_v[best_i]),
                xytext=(x[best_i] - width / 2, het_v[best_i] + 0.28),
                ha="center", fontsize=7.5, color="#1a5276",
                arrowprops={"arrowstyle": "->", "color": "#1a5276", "lw": 1.0})

    ax.set_xticks(x)
    ax.set_xticklabels([_STRATEGY_KO.get(s, s) for s in strategies], fontsize=8.5)
    ax.set_ylabel("Infections averted per dose\n(vs no intervention)", fontsize=9)
    prefix = "" if standalone else "C. "
    ax.set_title(f"{prefix}Targeted intervention vs no intervention — infections averted per dose",
                 fontsize=9.5, loc="left")
    ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(het_v.max(), hom_v.max()) * 1.32)
    return True


def build_figure() -> list[Path]:
    """3-패널 중재/정책 그림을 ABM 산출에서 빌드 후 PNG 저장.

    "한 번에 하나씩" 보기 요구에 맞춰 각 패널을 **독립 단일-패널 PNG** 로 먼저
    저장하고(의미 있는 이름), 이어서 기존 3-패널 조합 그림(OUT_PNG)을 같은 헬퍼로
    저장한다(back-compat).

    Returns:
        실제 생성+검증된 PNG 절대경로 리스트 (산출 부재로 skip 시 빈 리스트).

    Performance: O(1) (수백 점 곡선 + 막대 12개). 가벼운 plotting only —
        ABM 커널 재실행 없음(기존 결정성 산출 재사용).
    Side effects: simulation/results/figures/ 에 단일-패널 PNG 3종 +
        조합 PNG(fig_intervention_policy.png) 기록.
    Caller responsibility: ABM counterfactual.json/trajectory.npz 가 존재해야 함.
    """
    _setup_korean_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    abm_dir = _find_abm_dir()
    if abm_dir is None:
        print("[SKIP] ABM counterfactual 산출(counterfactual.json) 미발견 — "
              "가짜 데이터 생성 거부, 정직 skip. "
              "후보: " + " | ".join(str(d) for d in _ABM_DIR_CANDIDATES))
        return []
    print(f"[data] ABM 산출 디렉터리 = {abm_dir}")

    cf = _load_json(abm_dir / "counterfactual.json")
    if cf is None:
        print("[SKIP] counterfactual.json 로드 실패 — skip")
        return []

    written: list[Path] = []

    # --- (1) 단일-패널 PNG (한 번에 하나씩) — 같은 드로잉 헬퍼 재사용(standalone=True) ---
    # (drawer, out_path, panel-label) — 데이터 부재 패널은 정직히 skip(PNG 미생성).
    panel_specs = (
        (lambda ax: _panel_vaccination(ax, cf, standalone=True), OUT_VACCINATION, "A"),
        (lambda ax: _panel_behaviour(ax, abm_dir, standalone=True), OUT_BEHAVIOUR, "B"),
        (lambda ax: _panel_averted(ax, cf, standalone=True), OUT_AVERTED, "C"),
    )
    for drawer, out_path, label in panel_specs:
        fig1, ax1 = plt.subplots(figsize=(7.0, 5.0))
        ok = drawer(ax1)
        if not ok:
            print(f"[skip] 단일 패널 {label}: 데이터 부재 — {out_path.name} 미생성")
            plt.close(fig1)
            continue
        fig1.tight_layout()
        fig1.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig1)
        if _confirm_png(out_path):
            written.append(out_path)

    # --- (2) 조합 3-패널 그림(OUT_PNG) — 기존과 동일(standalone=False, back-compat) ---
    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.7))
    ok_a = _panel_vaccination(axes[0], cf)
    ok_b = _panel_behaviour(axes[1], abm_dir)
    ok_c = _panel_averted(axes[2], cf)

    if not any((ok_a, ok_b, ok_c)):
        print("[SKIP] 세 패널 모두 데이터 부재 — 조합 PNG 미생성")
        plt.close(fig)
        return written

    meta = cf.get("metadata", {})
    n_agents = meta.get("n_agents")
    year = meta.get("year")
    src = (f"ABM model-derived counterfactual scenario (not observed) · "
           f"N={n_agents:,} agents · reference year {year} · "
           f"Heterogeneous ABM = population structure (age, occupation, severity) preserved, "
           f"homogeneous = mean-field control")
    fig.suptitle(
        "Epidemic intervention/policy counterfactual — vaccine allocation · NPI (behavior) · "
        "targeted intervention (ABM what-if, not a forecast)",
        fontsize=12, y=1.02, fontweight="bold",
    )
    fig.text(0.5, -0.04, src, ha="center", fontsize=7.6, color="#555")

    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(OUT_PNG, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[done] panels: A={ok_a} B={ok_b} C={ok_c}")
    if _confirm_png(OUT_PNG):
        written.append(OUT_PNG)
    return written


def main() -> int:
    """엔트리: 그림 빌드 후 성공=0 / skip·실패=1 반환."""
    written = build_figure()
    if not written:
        return 1
    print(f"[written] {len(written)} PNG:")
    for p in written:
        print(f"  - {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
