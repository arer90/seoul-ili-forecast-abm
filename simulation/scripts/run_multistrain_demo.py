"""Multi-strain ABM feasibility demo grounded in WHO FluNet Korea subtype data.

석사논문 범위의 *descriptive* multi-strain overlay 데모. 라이브 ABM 커널
(``simulation.abm.multi_strain.run_multistrain``)을 **무수정** 재사용하고, DB는
``read_only_connect`` 로만 읽는다(쓰기 없음). 새 산출:

    simulation/results/multi_strain_demo/result.json
    simulation/results/multi_strain_demo/strain_dynamics.png

정직성 경계(과대주장 금지)
--------------------------
- **데이터로 grounding 되는 것(실측)**: strain 초기분배 비율. WHO FluNet
  ``who_flunet`` 테이블의 ``country='Republic of Korea'`` subtype-분해 양성수
  (A/H1 = inf_a_h1n1pdm09+inf_a_h1, A/H3 = inf_a_h3, B = inf_b)에서 한 시즌의
  관측 비율을 그대로 ``initial_infected`` 로 주입한다.
- **데이터로 grounding 되지 *않는* 것(placeholder)**: strain별 전파율 betas와
  교차면역 행렬 cross_immunity. 한국 strain별 Rt/교차면역 추정치가 DB·문헌에
  부재하므로 일반 인플루엔자 문헌값(homologous 차단 + 약한 heterosubtypic
  교차보호)을 placeholder 로 둔다. ⇒ 이 데모는 **calibrated multi-strain 이
  아니라 descriptive subtype overlay** 다. 우점/소멸 동역학은 정성적 시연이며
  관측 시계열에 fit 된 것이 아니다.

실행:
    .venv/bin/python -m simulation.scripts.run_multistrain_demo
sqlite write = 0 (read-only). 가벼운 1회(N=20000, T=180일).
"""
from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path

import numpy as np

from simulation.abm.multi_strain import run_multistrain, strain_competition_summary
from simulation.database import read_only_connect

# 데모 파라미터(전부 명시 — placeholder 와 실측을 분리 보고)
_KOREA = "Republic of Korea"
_SEASON = (2023, 2024)  # 3-strain 공존이 가장 뚜렷한 최근 시즌(A/H1·A/H3·B)
_N = 20_000             # agent 수(가벼운 1회)
_T_DAYS = 180           # 약 6개월(한 인플루엔자 시즌 길이)
_SEED = 42
_STRAIN_NAMES = ["A/H1", "A/H3", "B"]
_OUT_DIR = Path("simulation/results/multi_strain_demo")

# ── placeholder 역학 파라미터(문헌값, calibration 아님) ─────────────────────
# 일별 hazard 단위(커널 관례). 인플루엔자 잠복≈2일, 감염≈4일.
_SIGMA = 1.0 / 2.0      # E->I (잠복기 2일)
_GAMMA = 1.0 / 4.0      # I->R (감염기 4일)
# strain별 전파율: A/H3 가 통상 가장 transmissible, B 가 가장 낮다는 일반 패턴.
# (절대값이 아니라 *상대* 우열만 정성 시연 — 관측 fit 아님.)
_BETAS = [0.34, 0.40, 0.30]
# 교차면역 행렬(placeholder, [i][j]=strain i 회복자의 strain j 보호확률):
# 대각=1(homologous 재감염 차단), A/H1↔A/H3 약한 heterosubtypic(0.10),
# A↔B 사실상 무교차(0.02). 문헌 정성값이며 한국 추정치 아님.
_CROSS = [
    [1.00, 0.10, 0.02],
    [0.10, 1.00, 0.02],
    [0.02, 0.02, 1.00],
]
# stochastic 소멸 방지용 미세 외부유입(전 strain 동일).
_IMPORT = 1e-4


def _read_korea_subtype_shares() -> dict:
    """WHO FluNet Korea 한 시즌의 관측 subtype 양성수 → 정규화 비율(실측 grounding).

    Returns:
        dict — ``{"counts": {name: int}, "shares": {name: float}, "total": int,
        "season": [start, end], "n_weeks": int}``. counts 는 시즌 누적 양성수,
        shares 는 합=1 정규화 비율(strain 초기분배에 주입).

    Side effects: read-only DB open/close. 쓰기 없음.
    """
    start, end = _SEASON
    sql = (
        "SELECT "
        "  COALESCE(SUM(COALESCE(inf_a_h1n1pdm09,0)+COALESCE(inf_a_h1,0)),0) AS a_h1, "
        "  COALESCE(SUM(COALESCE(inf_a_h3,0)),0)                            AS a_h3, "
        "  COALESCE(SUM(COALESCE(inf_b,0)),0)                               AS b, "
        "  COUNT(*) AS n_weeks "
        "FROM who_flunet "
        "WHERE country = ? "
        "  AND ((year = ? AND week_no >= 30) OR (year = ? AND week_no < 30))"
    )
    with closing(read_only_connect()) as con:
        cur = con.execute(sql, (_KOREA, start, end))
        a_h1, a_h3, b, n_weeks = cur.fetchone()
    counts = {"A/H1": int(a_h1), "A/H3": int(a_h3), "B": int(b)}
    total = sum(counts.values())
    if total <= 0:
        raise RuntimeError(
            f"Korea subtype positives = 0 for season {start}-{end}; cannot ground demo"
        )
    shares = {k: v / total for k, v in counts.items()}
    return {
        "counts": counts,
        "shares": shares,
        "total": total,
        "season": [start, end],
        "n_weeks": int(n_weeks),
    }


def _shares_to_initial(shares: dict, n_seed_total: int) -> list[int]:
    """관측 비율 → strain별 초기 감염 agent 수(largest-remainder 로 합 보존)."""
    raw = np.array([shares[name] * n_seed_total for name in _STRAIN_NAMES])
    floor = np.floor(raw).astype(int)
    deficit = n_seed_total - int(floor.sum())
    if deficit > 0:
        order = np.argsort(-(raw - floor))  # 잔여 큰 순으로 +1
        for i in order[:deficit]:
            floor[i] += 1
    return [int(x) for x in floor]


def _plot(result: dict, summary: dict, grounding: dict, out_png: Path) -> None:
    """strain별 일별 감염력(I) 곡선 figure. placeholder/실측 경계를 caption 에 명시."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    I = np.asarray(result["I"])  # (T, S)
    days = np.arange(I.shape[0])
    names = result["strain_names"]
    colors = {"A/H1": "#1f77b4", "A/H3": "#d62728", "B": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(9, 5))
    for k, name in enumerate(names):
        ax.plot(days, I[:, k], label=name, color=colors.get(name, None), lw=2)
    ax.set_xlabel("Day")
    ax.set_ylabel("Infectious agents (I)")
    s0, s1 = grounding["season"]
    sh = grounding["shares"]
    ax.set_title(
        f"Multi-strain ABM (descriptive subtype overlay)\n"
        f"Seed shares from WHO FluNet Korea {s0}-{s1}: "
        f"A/H1={sh['A/H1']:.2f}  A/H3={sh['A/H3']:.2f}  B={sh['B']:.2f}"
    )
    ax.legend(title="strain")
    dom = summary["dominant_strain"]
    fig.text(
        0.5, -0.04,
        "Seed proportions = OBSERVED (WHO FluNet KR). "
        "betas / cross-immunity = LITERATURE PLACEHOLDER (not calibrated). "
        f"Dominant strain (qualitative): {dom}.",
        ha="center", fontsize=8, style="italic", wrap=True,
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    grounding = _read_korea_subtype_shares()
    n_seed_total = max(len(_STRAIN_NAMES), int(round(0.01 * _N)))  # ~1% seeded
    initial_infected = _shares_to_initial(grounding["shares"], n_seed_total)

    result = run_multistrain(
        N=_N,
        T_days=_T_DAYS,
        seed=_SEED,
        betas=_BETAS,
        cross_immunity=_CROSS,
        sigma=_SIGMA,
        gamma=_GAMMA,
        initial_infected=initial_infected,
        strain_names=_STRAIN_NAMES,
        import_rate=_IMPORT,
    )
    summary = strain_competition_summary(result)

    # 보존 불변식 sanity(S + ΣE + ΣI + R == N) 마지막 행에서 확인.
    last = (
        int(result["S"][-1])
        + int(np.asarray(result["E"])[-1].sum())
        + int(np.asarray(result["I"])[-1].sum())
        + int(result["R"][-1])
    )
    conservation_ok = last == _N

    out_json = {
        "demo": "multi_strain_descriptive_subtype_overlay",
        "honesty": {
            "grounded_by_data": "strain seed proportions (WHO FluNet Korea subtype positives)",
            "placeholder_not_calibrated": ["betas", "cross_immunity", "sigma", "gamma"],
            "claim_level": "descriptive subtype overlay (NOT calibrated multi-strain)",
        },
        "data_grounding": grounding,
        "params": {
            "N": _N,
            "T_days": _T_DAYS,
            "seed": _SEED,
            "strain_names": _STRAIN_NAMES,
            "betas": _BETAS,
            "cross_immunity": _CROSS,
            "sigma": _SIGMA,
            "gamma": _GAMMA,
            "import_rate": _IMPORT,
            "n_seed_total": n_seed_total,
            "initial_infected": initial_infected,
        },
        "summary": summary,
        "timeseries": {
            "I": np.asarray(result["I"]).tolist(),
            "incidence": np.asarray(result["incidence"]).tolist(),
            "cumulative_incidence": np.asarray(result["cumulative_incidence"]).tolist(),
            "S": np.asarray(result["S"]).tolist(),
            "R": np.asarray(result["R"]).tolist(),
        },
        "conservation_invariant_ok": conservation_ok,
    }

    (_OUT_DIR / "result.json").write_text(
        json.dumps(out_json, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _plot(result, summary, grounding, _OUT_DIR / "strain_dynamics.png")

    print("=== multi-strain demo (descriptive subtype overlay) ===")
    print(f"season {grounding['season']}  observed seed shares: {grounding['shares']}")
    print(f"observed positives: {grounding['counts']} (total {grounding['total']})")
    print(f"initial_infected (seeded): {dict(zip(_STRAIN_NAMES, initial_infected))}")
    print(f"dominant strain (qualitative): {summary['dominant_strain']}")
    print(f"attack_rate: {summary['attack_rate']}")
    print(f"peak_infectious: {summary['peak_infectious']}")
    print(f"conservation S+E+I+R==N: {conservation_ok}")
    print(f"wrote: {_OUT_DIR / 'result.json'}")
    print(f"wrote: {_OUT_DIR / 'strain_dynamics.png'}")


if __name__ == "__main__":
    main()
