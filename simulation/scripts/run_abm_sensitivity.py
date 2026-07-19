"""ABM 전역 민감도(LHS-PRCC) + 확률 앙상블을 실행하고 **영속화**한다 (research integrity).

배경 (외부 reviewer supplement, 2026-06-24): `simulation/abm/sensitivity.py` 는 `run_ensemble`
(Lee 2015 stochastic 앙상블) · `global_sensitivity`(Marino 2008 LHS-PRCC)를 구현하나, `__main__`
이 print 만 하고 산출물을 저장하지 않아 **docx [713] PRCC 수치(+0.92/−0.83)가 어떤 artifact 에도
결박되지 않은 orphan**(= orphan R²=0.884 와 동일 무결성 패턴, desk-reject 위험). 이 스크립트가
SA 를 1회 실행해 ``results/abm_v1/sensitivity.json`` 으로 결박한다.

Usage:
    python -m simulation.scripts.run_abm_sensitivity                    # 기본(200 seeds, 500 LHS)
    python -m simulation.scripts.run_abm_sensitivity --n-seeds 200 --n-samples 500

Performance: agent-world ~0.04 s/run → (n_seeds + n_samples) × 0.04 s (~30 s for 200+500).
Side effects: writes results/abm_v1/sensitivity.json (mkdir -p). 라이브 학습/DB 무관.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from simulation.abm.sensitivity import (
    BASE, SA_RANGES, run_ensemble, global_sensitivity,
)


def main() -> None:
    ap = argparse.ArgumentParser(description="ABM LHS-PRCC SA + stochastic ensemble → persisted JSON")
    ap.add_argument("--n-seeds", type=int, default=200, help="stochastic 앙상블 replicate 수 (Lee 2015)")
    ap.add_argument("--n-samples", type=int, default=500, help="LHS 표본수 (Marino 2008 global SA)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="simulation/results/abm_v1/sensitivity.json")
    args = ap.parse_args()

    print(f"[ensemble] {args.n_seeds} seeds (Lee 2015 stochastic 앙상블) ...")
    ens = run_ensemble(n_seeds=args.n_seeds, seed0=args.seed)
    print(f"[global-SA] LHS {args.n_samples} × PRCC (Marino 2008, 7 params) ...")
    sa = global_sensitivity(n_samples=args.n_samples, seed=args.seed)

    payload = {
        "schema": "abm_sensitivity_v1",
        "method": {
            "ensemble": "stochastic replicate ensemble (Lee et al. 2015, JASSS 18(4):4)",
            "global_sa": "Latin-Hypercube + Partial Rank Correlation Coefficient (Marino et al. 2008, J Theor Biol 254:178)",
        },
        "base_config": BASE,
        "sa_ranges": {k: list(v) for k, v in SA_RANGES.items()},
        "n_seeds": args.n_seeds,
        "n_samples": args.n_samples,
        "seed": args.seed,
        "ensemble": ens,
        "global_sensitivity": sa,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # 결박 확인 — 핵심 PRCC 상위 인자 출력(docx 결박 대상)
    print(f"\n✓ persisted → {out}")
    for metric in ("attack_rate", "peak_infected", "peak_day"):
        ranked = sorted(sa["prcc"][metric].items(), key=lambda kv: -abs(kv[1]["prcc"]))
        top = ", ".join(f"{p}={d['prcc']:+.2f}(p={d['p']:.0e})" for p, d in ranked[:3])
        print(f"  PRCC[{metric:14}] top3: {top}")
    vs = ens["variance_stabilization"]
    print(f"  ensemble CV(attack) n={vs['n'][0]}→{vs['n'][-1]}: "
          f"{vs['running_cv_attack'][0]:.3f} → {vs['running_cv_attack'][-1]:.3f} (replicate 수 정당화)")


if __name__ == "__main__":
    main()
