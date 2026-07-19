"""Simulation-Based Calibration (SBC) posterior validation for ABM behavior params.

Q7(교란통제 약함) 직격 산출물 — PLOS CB 리뷰어 필수. NPE 사후가 *보정*되었는지
(posterior coverage 정직) Talts et al. 2018 (arXiv:1804.06788) 절차로 검증한다.

원리
----
prior 에서 θ* 를 뽑아 데이터 x* 를 생성 → 같은 NPE 사후를 x* 에 조건화 → 사후표본 중
θ* 의 rank 를 계산. 사후가 보정되면 (좌표별) rank 가 **균등분포** 여야 한다(Talts 정리 1).
rank 히스토그램이 비균등(KS p < 0.05) = 사후 미보정(과신/과소신).

설계 규율 (ENGINEERING_PRINCIPLES.md)
--------------------
- **라이브 코드 무수정**: ``sbi_calibration.run_sbi`` 의 prior/simulator/NPE 패턴을
  *재사용*하되, ``run_sbc`` 는 학습된 posterior 객체를 필요로 한다(run_sbi 는 표본만 반환).
  따라서 이 스크립트가 동일 패턴으로 NPE 를 인라인 재구성한다 — 편집 아님, 신규.
- **read-only DB**: ``simulate_response`` 는 metapop 로드만(쓰기 없음). 별도 DB 커넥션 없음.
- **결정성**: ``torch.manual_seed`` / ``np.random.seed`` 고정. 같은 입력 → 같은 rank.
- **toy-first**: 잘 식별된 Gaussian toy 로 SBC 머신을 먼저 검증(파이프라인 sanity).
  toy 가 균등 rank 를 못 내면 ABM 결과를 신뢰할 수 없으므로 toy_passed 를 verdict 에 게이트.
- **예산**: ABM 단일 sim ≈ 0.3s. 학습 sims + SBC sims 를 작게 잡고 명시(아래 상수).

Run:
    .venv/bin/python scripts/abm_sbc_check.py
    .venv/bin/python scripts/abm_sbc_check.py --abm-sbc 120 --abm-train 400
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Callable

import numpy as np

warnings.filterwarnings("ignore")

# ── 산출물 경로 (단일 source) ────────────────────────────────────────────────
_REPO = Path(__file__).resolve().parent.parent
RESULT_JSON = _REPO / "simulation" / "results" / "abm_sbc" / "result.json"
FIG_PATH = _REPO / "simulation" / "results" / "figures" / "abm_sbc_ranks.png"

# ── 예산 상수 (실측: ABM 단일 sim ≈ 0.3s) ───────────────────────────────────
SEED = 42
TOY_TRAIN, TOY_SBC, TOY_POST = 600, 300, 400      # toy = 싸다 → 넉넉히
ABM_TRAIN, ABM_SBC, ABM_POST = 400, 120, 500      # ABM ≈ (400+120)×0.3s ≈ 2.6min + NPE
NPE_MAX_EPOCHS = 200

# ABM behavior params + uniform-prior bounds (sbi_posterior_calibration 와 동일 SSOT).
ABM_PRIORS: dict[str, tuple[float, float]] = {
    "alpha": (0.5, 3.5), "kappa": (0.05, 0.40),
    "tau": (40.0, 140.0), "theta": (0.03, 0.30),
}


def _build_npe_and_ranks(simulator: Callable[[np.ndarray], np.ndarray],
                         lows, highs, *, n_train: int, n_sbc: int,
                         n_post: int, seed: int,
                         density_estimator: str | None = None) -> dict:
    """NPE 사후 학습 → SBC rank 계산 → KS 균등성 검정 (Talts 2018).

    run_sbi(sbi_calibration.py) 의 prior/NPE 구성과 동일 패턴을 인라인 재현한다.
    run_sbc 가 학습된 posterior 객체를 소비하므로 표본만 반환하는 run_sbi 로는 부족 → 재구성.

    Args:
        simulator: callable(theta_row 1-D np) → x_row 1-D np (요약통계). finite 보장 불필요.
        lows, highs: 좌표별 uniform-prior 경계 (길이 D).
        n_train: NPE 학습용 prior 시뮬레이션 수.
        n_sbc: SBC rank 계산용 (prior→data) 쌍 수 (Talts 의 L).
        n_post: rank 산출 시 사후표본 수.
        seed: RNG seed (결정성).

    Returns:
        ``{names_len D, ks_pvals (list D), ranks (n_kept, D) np, n_train_kept,
        n_sbc_kept, n_post}``. ks_pvals[i] = i-번째 좌표 rank 의 KS-uniform p-value.

    Raises:
        RuntimeError: finite 학습 시뮬레이션이 20 미만(식별 불가/시뮬 폭발).

    Performance: O(n_train + n_sbc) simulator 호출 + 1 NPE train. Side effects: none.
    Caller responsibility: simulator 가 NaN 행을 내도 됨(내부에서 finite 필터).
    """
    import torch
    from sbi.diagnostics import check_sbc, run_sbc
    from sbi.inference import NPE
    from sbi.utils import BoxUniform

    torch.manual_seed(seed)
    np.random.seed(seed)
    lows = np.asarray(lows, dtype=np.float64)
    highs = np.asarray(highs, dtype=np.float64)
    D = len(lows)
    prior = BoxUniform(low=torch.tensor(lows, dtype=torch.float32),
                       high=torch.tensor(highs, dtype=torch.float32))

    def _simulate(theta_t):
        rows = np.array([np.asarray(simulator(t.numpy()), dtype=np.float64) for t in theta_t])
        x = torch.tensor(rows, dtype=torch.float32)
        finite = torch.isfinite(x).all(dim=1) & torch.isfinite(theta_t).all(dim=1)
        return theta_t[finite], x[finite], int(finite.sum())

    # ① NPE 사후 학습 (run_sbi 패턴) ──────────────────────────────────────────
    theta_tr = prior.sample((n_train,))
    theta_tr, x_tr, kept_tr = _simulate(theta_tr)
    if kept_tr < 20:
        raise RuntimeError(f"too few finite training simulations ({kept_tr})")
    inference = (NPE(prior=prior) if not density_estimator
                 else NPE(prior=prior, density_estimator=density_estimator))
    inference.append_simulations(theta_tr, x_tr).train(max_num_epochs=NPE_MAX_EPOCHS)
    posterior = inference.build_posterior()

    # ② SBC: 새 prior 표본 → 데이터 생성 → rank 계산 (Talts 2018) ─────────────
    theta_sbc = prior.sample((n_sbc,))
    theta_sbc, x_sbc, kept_sbc = _simulate(theta_sbc)
    if kept_sbc < 20:
        raise RuntimeError(f"too few finite SBC simulations ({kept_sbc})")
    ranks, dap = run_sbc(theta_sbc, x_sbc, posterior,
                         num_posterior_samples=n_post, show_progress_bar=False)

    # ③ rank 균등성 검정 (KS, 좌표별) ─────────────────────────────────────────
    prior_for_check = prior.sample((kept_sbc,))
    chk = check_sbc(ranks, prior_for_check, dap, num_posterior_samples=n_post)
    ks_pvals = [float(v) for v in chk["ks_pvals"]]
    return {"D": D, "ks_pvals": ks_pvals, "ranks": ranks.numpy(),
            "n_train_kept": kept_tr, "n_sbc_kept": kept_sbc, "n_post": n_post,
            "density_estimator": density_estimator or "maf"}


def _toy_simulator_factory(seed: int) -> Callable[[np.ndarray], np.ndarray]:
    """잘 식별된 2-D→4-D Gaussian toy (test_sbi_calibration 와 동일 구조).

    x = [θ0, θ1, θ0·θ1, θ0+θ1] + N(0, 0.03). prior=[0,1]². 잘 식별 → SBC rank 균등 기대.
    """
    rng = np.random.default_rng(seed)

    def simulator(theta: np.ndarray) -> np.ndarray:
        n = rng.normal(0, 0.03, size=4)
        return np.array([theta[0], theta[1], theta[0] * theta[1],
                         theta[0] + theta[1]]) + n
    return simulator


def _abm_summary(traj) -> np.ndarray:
    """scale-invariant shape 통계 (sbi_posterior_calibration._summary 와 동일).

    peak 위치·정규화 rise/fall·peak/mean — 전국 ILI vs 시뮬 prevalence 스케일차 제거.
    """
    t = np.asarray(traj, dtype=np.float64)
    t = t[np.isfinite(t)]
    if len(t) < 6 or t.max() <= 0:
        return np.array([np.nan] * 4)
    pk, n = int(np.argmax(t)), len(t)
    rise = (t[pk] - t[0]) / (t[pk] + 1e-9) / max(pk, 1)
    fall = (t[pk] - t[-1]) / (t[pk] + 1e-9) / max(n - pk, 1)
    return np.array([pk / n, rise * 52, fall * 52, t[pk] / (t.mean() + 1e-9)])


def _abm_simulator_factory(names: list[str]):
    """ABM behavior-param simulator: θ(alpha,kappa,tau,theta) → 4-D shape 요약.

    simulate_response(metapop, kw)["prevalence"] 를 재사용(라이브 코드 무수정).
    예외/폭발 시 NaN 행 반환(상위 finite 필터가 처리).
    """
    from simulation.abm.sim_vs_observed import load_seoul_metapop, simulate_response

    mp = load_seoul_metapop(days=180)

    def simulator(theta: np.ndarray) -> np.ndarray:
        kw = {names[j]: float(theta[j]) for j in range(len(names))}
        try:
            return _abm_summary(simulate_response(mp, kw)["prevalence"])
        except Exception:
            return np.array([np.nan] * 4)
    return simulator


_N_BINS = 12


def _draw_rank_panel(ax, tag: str, label: str, res: dict, col_index: int) -> None:
    """한 좌표의 rank 히스토그램 1개를 ``ax`` 에 그린다 (단일 셀 = 단일 figure 공용).

    균등 기대선 + 95% binomial 일관성 밴드(Talts Fig 회색 밴드) + KS verdict 제목.
    조합 grid 와 standalone figure 가 같은 코드 경로를 쓰도록 추출 (드로잉 SSOT).

    Args:
        ax: matplotlib Axes (그릴 대상).
        tag: 그룹 태그 ("toy" / "ABM").
        label: 파라미터 이름 (예: "theta0", "alpha").
        res: ``_build_npe_and_ranks`` 반환 dict ({ranks, ks_pvals, n_post, D, ...}).
        col_index: 좌표 인덱스 (0-based). ``res["D"]`` 미만이어야 함.

    Side effects: ``ax`` 에 hist/axhspan/axhline/title/label 그림. 파일 쓰기 없음.
    Caller responsibility: ``col_index < res["D"]`` (off-cell 은 호출 전 skip).
    """
    from scipy.stats import binom

    ranks = res["ranks"]
    n_sbc = ranks.shape[0]
    n_post = res["n_post"]
    # 균등 기대 빈도 + 95% binomial 일관성 밴드 (Talts Fig 의 회색 밴드).
    expected = n_sbc / _N_BINS
    lo = binom.ppf(0.005, n_sbc, 1.0 / _N_BINS)
    hi = binom.ppf(0.995, n_sbc, 1.0 / _N_BINS)
    ax.hist(ranks[:, col_index], bins=_N_BINS, range=(0, n_post + 1),
            color="#3b6ea5", edgecolor="white", alpha=0.85)
    ax.axhspan(lo, hi, color="0.80", alpha=0.6, zorder=0)
    ax.axhline(expected, color="0.45", ls="--", lw=1)
    pv = res["ks_pvals"][col_index]
    verdict = "uniform" if pv >= 0.05 else "NON-uniform"
    ax.set_title(f"{tag}: {label}\nKS p={pv:.3f} ({verdict})", fontsize=9)
    ax.set_xlabel("rank")
    ax.set_ylabel("count")


def _plot_rank_histograms(toy: dict | None, abm: dict | None,
                          run_label: str | None = None) -> list[Path]:
    """좌표별 rank 히스토그램 — 파라미터별 단독 PNG + 조합 PNG (back-compat).

    "한 번에 1개" 요구: 각 좌표(toy:theta*, ABM:alpha/kappa/tau/theta) 를 독립
    figure 로 저장(``abm_sbc_ranks_<tag>_<label>.png``). 동시에 기존 조합 figure
    (``abm_sbc_ranks.png``) 도 유지. off-cell(col ≥ res["D"]) 은 skip.

    Returns:
        쓰여진 figure 절대경로 list (단독 figure 들 + 조합 figure 마지막).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = []
    if toy is not None:
        panels.append(("toy", ["theta0", "theta1"], toy))
    if abm is not None:
        panels.append(("ABM", list(ABM_PRIORS), abm))

    fig_base = (FIG_PATH if not run_label
                else FIG_PATH.with_name(f"abm_sbc_ranks_{run_label}.png"))
    fig_base.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # ── ① 파라미터별 단독 figure (한 번에 1개) ───────────────────────────────
    for tag, labels, res in panels:
        for c in range(res["D"]):                       # off-cell 자동 skip (range=D)
            label = labels[c]
            single, ax = plt.subplots(figsize=(5.0, 4.0))
            _draw_rank_panel(ax, tag, label, res, c)
            single.tight_layout()
            out = fig_base.with_name(f"{fig_base.stem}_{tag}_{label}.png")
            single.savefig(out, dpi=130)
            plt.close(single)
            written.append(out)

    # ── ② 조합 figure (back-compat: abm_sbc_ranks.png) ───────────────────────
    n_cols = max(p[2]["D"] for p in panels)
    n_rows = len(panels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.0 * n_rows),
                             squeeze=False)
    for r, (tag, labels, res) in enumerate(panels):
        for c in range(n_cols):
            ax = axes[r][c]
            if c >= res["D"]:
                ax.axis("off")
                continue
            _draw_rank_panel(ax, tag, labels[c], res, c)
            if c != 0:
                ax.set_ylabel("")                       # grid 는 좌측 열만 ylabel
    fig.suptitle("Simulation-Based Calibration — rank histograms (uniform = calibrated)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(fig_base, dpi=130)
    plt.close(fig)
    written.append(fig_base)
    return written


def _verdict(ks_pvals: list[float]) -> str:
    """모든 좌표 KS p ≥ 0.05 → 'posterior 보정됨', 하나라도 < 0.05 → '미보정'."""
    return "posterior 보정됨" if all(p >= 0.05 for p in ks_pvals) else "posterior 미보정"


def main(abm_train: int = ABM_TRAIN, abm_sbc: int = ABM_SBC,
         abm_post: int = ABM_POST, skip_abm: bool = False,
         density_estimator: str | None = None, run_label: str | None = None) -> int:
    """toy-first SBC → ABM behavior-param SBC → result.json + rank figure.

    Returns: 0 (toy 통과 시) / 2 (toy 미통과 = SBC 머신 자체 의심).
    """
    report: dict = {
        "method": "Simulation-Based Calibration (Talts et al. 2018, arXiv:1804.06788)",
        "library": "sbi.diagnostics.run_sbc + check_sbc (sbi 0.26.1)",
        "seed": SEED,
        "rank_uniform_test": "Kolmogorov-Smirnov (per-parameter marginal), alpha=0.05",
        "toy": None, "abm": None, "toy_passed": None, "verdict": None,
    }

    # ── ① TOY-FIRST: SBC 머신이 잘 식별된 toy 에서 균등 rank 를 내는가 ────────
    print(f"[1/2] toy SBC  (train={TOY_TRAIN}, sbc={TOY_SBC}, post={TOY_POST}) ...")
    toy = _build_npe_and_ranks(_toy_simulator_factory(SEED), [0.0, 0.0], [1.0, 1.0],
                               n_train=TOY_TRAIN, n_sbc=TOY_SBC, n_post=TOY_POST,
                               seed=SEED)
    toy_passed = all(p >= 0.05 for p in toy["ks_pvals"])
    report["toy_passed"] = bool(toy_passed)
    report["toy"] = {
        "params": ["theta0", "theta1"],
        "rank_uniform_pvalue": {f"theta{i}": round(p, 4)
                                for i, p in enumerate(toy["ks_pvals"])},
        "n_sims": toy["n_sbc_kept"], "n_train": toy["n_train_kept"],
        "n_posterior_samples": toy["n_post"],
        "verdict": _verdict(toy["ks_pvals"]),
    }
    print(f"      toy ks_pvals={[round(p,3) for p in toy['ks_pvals']]} "
          f"→ {'PASS (uniform)' if toy_passed else 'FAIL (non-uniform)'}")

    # ── ② ABM behavior params: alpha/kappa/tau/theta SBC ─────────────────────
    abm = None
    if not skip_abm:
        names = list(ABM_PRIORS)
        lows = [ABM_PRIORS[k][0] for k in names]
        highs = [ABM_PRIORS[k][1] for k in names]
        print(f"[2/2] ABM SBC  (train={abm_train}, sbc={abm_sbc}, post={abm_post}, "
              f"params={names}) ...")
        abm = _build_npe_and_ranks(_abm_simulator_factory(names), lows, highs,
                                   n_train=abm_train, n_sbc=abm_sbc, n_post=abm_post,
                                   seed=SEED, density_estimator=density_estimator)
        report["abm"] = {
            "params": names,
            "density_estimator": abm.get("density_estimator", "maf"),
            "priors": {k: list(ABM_PRIORS[k]) for k in names},
            "rank_uniform_pvalue": {k: round(abm["ks_pvals"][i], 4)
                                    for i, k in enumerate(names)},
            "per_param_verdict": {
                k: ("보정됨" if abm["ks_pvals"][i] >= 0.05 else "미보정")
                for i, k in enumerate(names)},
            "n_sims": abm["n_sbc_kept"], "n_train": abm["n_train_kept"],
            "n_posterior_samples": abm["n_post"],
            "verdict": _verdict(abm["ks_pvals"]),
        }
        print(f"      ABM ks_pvals="
              f"{ {k: round(abm['ks_pvals'][i],3) for i,k in enumerate(names)} }")
    else:
        print("[2/2] ABM SBC skipped (--skip-abm)")

    # ── ③ 종합 verdict (toy 게이트) ──────────────────────────────────────────
    if not toy_passed:
        report["verdict"] = "INVALID — toy SBC 미통과(머신 의심), ABM 해석 보류"
    elif abm is not None:
        report["verdict"] = _verdict(abm["ks_pvals"])
    else:
        report["verdict"] = "toy only (ABM skipped)"

    # ── ④ figure + JSON ──────────────────────────────────────────────────────
    figs = _plot_rank_histograms(toy, abm, run_label=run_label)
    out_json = (RESULT_JSON if not run_label
                else RESULT_JSON.with_name(f"result_{run_label}.json"))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"\n→ {out_json}")
    print(f"→ figures ({len(figs)} written):")
    for p in figs:
        print(f"    {p}")
    print(f"verdict: {report['verdict']}  (toy_passed={report['toy_passed']})")
    return 0 if toy_passed else 2


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SBC posterior validation for ABM behavior params")
    ap.add_argument("--abm-train", type=int, default=ABM_TRAIN, help="NPE training sims for ABM")
    ap.add_argument("--abm-sbc", type=int, default=ABM_SBC, help="SBC datasets for ABM")
    ap.add_argument("--abm-post", type=int, default=ABM_POST, help="posterior samples per rank")
    ap.add_argument("--skip-abm", action="store_true", help="toy-only (debug)")
    ap.add_argument("--density-estimator", default=None,
                    help="NPE flow: omitted=maf (default) | nsf (neural spline flow)")
    ap.add_argument("--label", default=None,
                    help="output label -> result_<label>.json + abm_sbc_ranks_<label>*.png")
    args = ap.parse_args()
    raise SystemExit(main(abm_train=args.abm_train, abm_sbc=args.abm_sbc,
                          abm_post=args.abm_post, skip_abm=args.skip_abm,
                          density_estimator=args.density_estimator, run_label=args.label))
