"""다층 접촉망 개념검증(proof-of-concept) 산출기 — 접촉구조 이질성 데모.

`simulation.abm.contact_network` 의 명시적 다층 접촉망(가구·직장·학교·지역사회)이
평균장(mean-field) 가정을 어떻게 넘어서는지 **한 번** 산출해 보인다. 메인 ABM 코어
kernel 은 자기 구(區)의 평균 prevalence 에 모두가 동일하게 노출되는 평균장을 쓴다.
본 스크립트는 그 평균장과, edge 기반 per-agent 감염력(FoI)을 같은 합성 인구·같은
감염 상태 위에서 나란히 산출해 **접촉구조가 FoI 이질성(분산·tail)을 만든다**는 것을
실측·시각화한다.

⚠ 정직 framing(honest): 본 산출은 **mechanism 데모**일 뿐 **예측력 개선 주장이 아니다**.
   "다층 접촉망을 쓰면 ILI 예측이 좋아진다"는 어떤 주장도 하지 않는다. 보이는 것은
   오직 "평균장은 한 구 안 모든 agent 에게 같은 FoI 를 주지만, 명시적 접촉망은
   누가 누구와 접촉하느냐에 따라 agent 마다 다른 FoI 를 만든다(이질성·tail 실재)"이다.

비교 설계(공정한 mean-field 대조):
    - network FoI = ``contact_network.network_foi`` (per-agent, edge 기반).
    - mean-field FoI = 각 agent 의 network FoI 를 **자기 home_gu 평균으로 대체**.
      → 구별 평균은 정확히 동일하게 보존되고, 차이는 **구 내 이질성(within-gu
      dispersion)** 하나로 격리된다. 평균장이 평탄화하는 바로 그 성분만 비교된다.

패널 구성 (contact_network_demo.png):
    A) per-agent FoI 분포 — network(edge 기반) vs mean-field(구 평균) 히스토그램.
       평균장은 구당 소수의 막대(구 평균값)로 뭉치고, network 는 0(감염 이웃 없음)부터
       다수 감염 이웃 보유 agent 까지 넓은 우향 tail 을 갖는다.
    B) layer 별 degree 분포 — 가구/직장/학교/지역사회 각 layer 의 per-agent degree
       히스토그램(접촉구조의 층별 이질성).
    C) within-gu FoI 표준편차 — 구별로 network FoI 의 구 내 표준편차 막대.
       평균장은 모두 0(구 내 균질). network 만 양(+)의 within-gu 분산을 가짐.

실행:
    .venv/bin/python -m simulation.scripts.fig_contact_network_demo

출력:
    simulation/results/figures/contact_network_demo.png  (dpi=120, bbox_inches="tight")
    stdout 에 요약 통계(분산비·tail·degree summary·within-gu σ·sqlite write=0).

설계 규율 (ENGINEERING_PRINCIPLES.md):
    - 라이브 코드 무수정: contact_network / synthetic_population 을 import 만(편집 X).
    - DB 접근은 synthetic_population.generate_population(내부 read-only SQLite)만 —
      본 스크립트는 저수준 SQLite 리터럴/연결을 직접 쓰지 않는다(write=0).
    - matplotlib Agg backend(headless). 결정성: 단일 seed 고정.
    - 가벼운 1회 산출(합성 인구 N=4000, 1 snapshot). 학습/파이프라인 무관.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless 렌더링 (디스플레이 비종속)

import matplotlib.pyplot as plt
import numpy as np

# 라이브 모듈 import 만 — 편집 없음(가법 데모).
from simulation.abm.agent_kernel import STATE_I, STATE_S
from simulation.abm.contact_network import (
    build_multilayer_network,
    degree_summary,
    network_foi,
)
from simulation.abm.synthetic_population import generate_population

# ---------------------------------------------------------------------------
# 경로 SSOT (단일 코드 루트 — ENGINEERING_PRINCIPLES.md §4 KISS)
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parents[2]  # simulation/scripts/<this> → repo root
FIG_DIR = PROJECT_ROOT / "simulation" / "results" / "figures"
OUT_PATH = FIG_DIR / "contact_network_demo.png"  # combined (kept for back-compat)
# 단일 패널 산출물 (사용자 요청: figure 1개씩) — 의미있는 이름.
OUT_FOI = FIG_DIR / "contact_network_foi.png"
OUT_DEGREE = FIG_DIR / "contact_network_degree.png"
OUT_WITHIN_GU = FIG_DIR / "contact_network_within_gu.png"

# 데모 파라미터 — 가벼운 1회 산출, 단일 seed 로 결정성.
N_AGENTS = 4000
POP_SEED = 20260626
NET_SEED = 20260626
PREVALENCE = 0.05  # 초기 감염 비율(5%): 고정 I-상태 snapshot.
_LAYERS = ("household", "workplace", "school", "community")
# 층별 전파율(per-day hazard 단위, 데모 스케일) — 가구>학교>직장>지역사회.
# 캘리브레이션 값이 아님(상대 크기만으로 이질성 mechanism 을 보임).
BETA_BY_LAYER = {
    "household": 0.30,
    "workplace": 0.10,
    "school": 0.20,
    "community": 0.05,
}


def _fixed_infection_state(home_gu: np.ndarray, *, seed: int) -> np.ndarray:
    """고정 prevalence 의 감염 snapshot 을 결정적으로 만든다.

    Args:
        home_gu: 길이 N 의 home_gu 코드(구별 층화 표집용).
        seed: 결정적 감염자 선택 seed.

    Returns:
        길이 N int8 state 벡터. ``PREVALENCE`` 비율을 STATE_I, 나머지 STATE_S.
        구별 균등하게 감염을 배치해 평균장 비교를 공정하게 한다.

    Side effects: 없음(순수 함수).
    """
    n = home_gu.shape[0]
    rng = np.random.default_rng(seed)
    state = np.full(n, STATE_S, dtype=np.int8)
    # 구별 층화 감염: 각 구에서 동일 비율을 감염시켜 구 평균 prevalence 를 균질화
    # → mean-field 대조가 "구 평균 노출"이라는 가정과 정합.
    for gu in np.unique(home_gu):
        idx = np.flatnonzero(home_gu == gu)
        k = int(round(idx.shape[0] * PREVALENCE))
        if k > 0:
            chosen = rng.choice(idx, size=k, replace=False)
            state[chosen] = STATE_I
    return state


def _meanfield_foi(network_foi_vec: np.ndarray, home_gu: np.ndarray) -> np.ndarray:
    """공정한 평균장 대조 FoI — network FoI 를 구 평균으로 평탄화.

    각 agent 의 network FoI 를 자기 home_gu 의 평균으로 대체한다. 구별 평균은
    정확히 보존되므로 두 분포의 **유일한** 차이는 구 내 이질성(within-gu
    dispersion)이다 — 평균장이 평탄화해 버리는 바로 그 성분.

    Args:
        network_foi_vec: 길이 N 의 per-agent network FoI(>=0).
        home_gu: 길이 N 의 home_gu 코드.

    Returns:
        길이 N float64. 같은 구 agent 는 모두 그 구의 평균 network FoI 값(구 내 분산 0).

    Side effects: 없음(순수 함수).
    """
    mf = np.zeros_like(network_foi_vec, dtype=np.float64)
    for gu in np.unique(home_gu):
        idx = np.flatnonzero(home_gu == gu)
        mf[idx] = float(network_foi_vec[idx].mean())
    return mf


def _layer_degrees(layers: dict, name: str) -> np.ndarray:
    """한 layer 의 per-agent degree 벡터(행 합 = 이웃 수)."""
    return np.asarray(layers[name].sum(axis=1)).ravel()


def _within_gu_std(foi_vec: np.ndarray, home_gu: np.ndarray) -> np.ndarray:
    """구별 FoI 의 구 내 표준편차 벡터(평균장이면 전부 0)."""
    gus = np.unique(home_gu)
    out = np.empty(gus.shape[0], dtype=np.float64)
    for i, gu in enumerate(gus):
        out[i] = float(foi_vec[np.flatnonzero(home_gu == gu)].std())
    return out


def _panel_foi(ax, foi_net: np.ndarray, foi_mf: np.ndarray, *, standalone: bool) -> None:
    """패널 A — per-agent FoI 분포(network vs mean-field). 단일/합본 공용."""
    hi = float(max(foi_net.max(), foi_mf.max(), 1e-9))
    bins = np.linspace(0.0, hi, 40)
    ax.hist(
        foi_net, bins=bins, color="#1f77b4", alpha=0.65,
        label=f"network (edge-based)\nvar={foi_net.var():.3f}",
    )
    ax.hist(
        foi_mf, bins=bins, color="#d62728", alpha=0.55,
        label=f"mean-field (gu avg)\nvar={foi_mf.var():.3f}",
    )
    title = ("per-agent force-of-infection distribution\n(mechanism demo — not a forecast)"
             if standalone else
             "A) per-agent FoI distribution\n(mechanism demo — not a forecast)")
    ax.set_title(title)
    ax.set_xlabel("force of infection (per-agent)")
    ax.set_ylabel("agent count")
    ax.legend(fontsize=9 if standalone else 8, loc="upper right")


def _panel_degree(ax, layers: dict, deg: dict, *, standalone: bool) -> None:
    """패널 B — layer 별 degree 분포. 단일/합본 공용."""
    colors = {"household": "#2ca02c", "workplace": "#ff7f0e",
              "school": "#9467bd", "community": "#8c564b"}
    max_deg = 0
    for name in _LAYERS:
        d = _layer_degrees(layers, name)
        max_deg = max(max_deg, int(d.max()) if d.size else 0)
    dbins = np.arange(0, max_deg + 2) - 0.5
    for name in _LAYERS:
        d = _layer_degrees(layers, name)
        ax.hist(
            d, bins=dbins, histtype="step", linewidth=1.8, color=colors[name],
            label=f"{name} (mean deg={deg[name]:.2f})",
        )
    title = ("per-layer degree distribution\n(contact-structure heterogeneity)"
             if standalone else
             "B) per-layer degree distribution\n(contact-structure heterogeneity)")
    ax.set_title(title)
    ax.set_xlabel("degree (number of contacts)")
    ax.set_ylabel("agent count")
    ax.set_yscale("log")
    ax.legend(fontsize=9 if standalone else 8, loc="upper right")


def _panel_within_gu(ax, foi_net: np.ndarray, foi_mf: np.ndarray,
                     home_gu: np.ndarray, *, standalone: bool) -> None:
    """패널 C — within-gu FoI 표준편차. 단일/합본 공용."""
    sd_net = _within_gu_std(foi_net, home_gu)
    sd_mf = _within_gu_std(foi_mf, home_gu)
    gus = np.arange(sd_net.shape[0])
    width = 0.4
    ax.bar(gus - width / 2, sd_net, width=width, color="#1f77b4",
           label=f"network (mean σ={sd_net.mean():.3f})")
    ax.bar(gus + width / 2, sd_mf, width=width, color="#d62728",
           label=f"mean-field (mean σ={sd_mf.mean():.3f})")
    title = ("within-gu FoI std dev\n(mean-field collapses to 0 by design)"
             if standalone else
             "C) within-gu FoI std dev\n(mean-field collapses to 0 by design)")
    ax.set_title(title)
    ax.set_xlabel("gu index (0-24)")
    ax.set_ylabel("within-gu std dev of FoI")
    ax.legend(fontsize=9 if standalone else 8, loc="upper right")


def _draw(
    foi_net: np.ndarray,
    foi_mf: np.ndarray,
    layers: dict,
    home_gu: np.ndarray,
    deg: dict,
) -> list[Path]:
    """3개 **단일 패널** figure(우선) + 합본 1개(back-compat) 를 그려 저장.

    사용자 요청("figure 는 가능한 1개씩"): 각 패널을 독립 figure 로 따로 저장한다.
    합본(OUT_PATH)도 유지해 기존 참조가 깨지지 않게 한다(placeholder 없음, 동일 코드 재현).

    Returns:
        저장한 모든 figure 경로 리스트(단일 3개 + 합본 1개).
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # --- 단일 패널 (우선) ---
    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    _panel_foi(ax, foi_net, foi_mf, standalone=True)
    fig.tight_layout()
    fig.savefig(OUT_FOI, dpi=130, bbox_inches="tight")
    plt.close(fig)
    written.append(OUT_FOI)

    fig, ax = plt.subplots(figsize=(7.5, 5.4))
    _panel_degree(ax, layers, deg, standalone=True)
    fig.tight_layout()
    fig.savefig(OUT_DEGREE, dpi=130, bbox_inches="tight")
    plt.close(fig)
    written.append(OUT_DEGREE)

    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    _panel_within_gu(ax, foi_net, foi_mf, home_gu, standalone=True)
    fig.tight_layout()
    fig.savefig(OUT_WITHIN_GU, dpi=130, bbox_inches="tight")
    plt.close(fig)
    written.append(OUT_WITHIN_GU)

    # --- 합본 (back-compat 유지) ---
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    _panel_foi(axes[0], foi_net, foi_mf, standalone=False)
    _panel_degree(axes[1], layers, deg, standalone=False)
    _panel_within_gu(axes[2], foi_net, foi_mf, home_gu, standalone=False)
    fig.suptitle(
        "Multi-layer contact network — contact structure makes FoI heterogeneous "
        "(mean-field cannot). Mechanism proof-of-concept, NOT a prediction-skill claim.",
        fontsize=11, y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)
    written.append(OUT_PATH)
    return written


def main() -> None:
    """합성 인구로 다층 접촉망을 한 번 산출하고 figure + 요약 통계를 낸다."""
    print("[fig_contact_network_demo] 다층 접촉망 개념검증 (mechanism demo)")
    print(f"  population: N={N_AGENTS} (DB-grounded synthetic, read-only SQLite)")

    # 1) DB 기반 합성 인구(내부 read-only SQLite — 저수준 리터럴 직접 X).
    pop = generate_population(N_AGENTS, seed=POP_SEED)
    home_gu = np.asarray(pop["home_gu"], dtype=np.int64)

    # 2) 다층 접촉망 구축(라이브 모듈, 편집 X).
    layers = build_multilayer_network(pop, seed=NET_SEED)
    deg = degree_summary(layers)

    # 3) 고정 감염 snapshot → network FoI 와 mean-field FoI.
    state = _fixed_infection_state(home_gu, seed=NET_SEED)
    foi_net = network_foi(state, layers, BETA_BY_LAYER)
    foi_mf = _meanfield_foi(foi_net, home_gu)

    # 4) figure(코드 재현, placeholder 없음) — 단일 패널 우선 + 합본 back-compat.
    written = _draw(foi_net, foi_mf, layers, home_gu, deg)

    # 5) 요약 통계(보고용).
    n_inf = int((state == STATE_I).sum())
    var_ratio = (foi_net.var() / foi_mf.var()) if foi_mf.var() > 0 else float("inf")
    sd_net = _within_gu_std(foi_net, home_gu)
    p95 = float(np.percentile(foi_net, 95))
    print(f"  infected snapshot: {n_inf}/{N_AGENTS} ({PREVALENCE:.0%}) fixed I-state")
    print("  --- degree summary (per-layer mean degree) ---")
    for name in _LAYERS:
        print(f"    {name:10s}: {deg[name]:.3f}")
    print(f"    {'_total':10s}: {deg['_total']:.3f}")
    print("  --- network FoI vs mean-field FoI (per-agent) ---")
    print(f"    network   : mean={foi_net.mean():.4f}  var={foi_net.var():.4f}  "
          f"max={foi_net.max():.4f}  p95={p95:.4f}")
    print(f"    mean-field: mean={foi_mf.mean():.4f}  var={foi_mf.var():.4f}  "
          f"max={foi_mf.max():.4f}")
    print(f"    variance ratio (network / mean-field) = {var_ratio:.2f}x")
    print(f"    network FoI == 0 (no infected neighbor): "
          f"{int((foi_net == 0).sum())}/{N_AGENTS} agents "
          f"({(foi_net == 0).mean():.1%})")
    print(f"    within-gu FoI std dev: network mean={sd_net.mean():.4f}  "
          f"mean-field=0.0000 (by design)")
    print("  figures (single-panel preferred + combined back-compat):")
    for p in written:
        print(f"    {p.name}")
    print("  honest framing: mechanism demo only — NO prediction-skill claim.")
    print("  sqlite writes by this script = 0 (generate_population = read-only).")


if __name__ == "__main__":
    main()
