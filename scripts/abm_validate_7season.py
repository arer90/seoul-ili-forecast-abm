"""ABM agent-world 7-season 검증 재실행 (user 2026-06-13 "다 해" — ABM 캠페인 ①).

목적: thesis 의 orphan R²=0.884(어떤 artifact 에도 없음) 를 **실측으로 대체**하기 위해, 진짜 agent-world
(run_agent_world rich-movement)를 7개 실측 KDCA ILI 시즌 전수에 보정 → R²/WIS/corr + **edge-pinning 해소**
(기본 grid 가 경계에 핀 → 양끝 넓힌 grid 로 optimum bracket) → active SSOT(simulation/results/abm/) 저장.

진단(abm-diagnosis-20260613): 코어 SOLID, 단 R²≈0.95 가 2023 1시즌만·grid-edge pinned·artifact _trash 빈json.
→ 사용자 원칙 "주장강등 말고 코드실행으로 사실화" 대로 진짜 7시즌 숫자를 만든다.

용법:
  .venv/bin/python scripts/abm_validate_7season.py --smoke   # 1시즌·소 grid·N=2000 (메커니즘 검증)
  .venv/bin/python scripts/abm_validate_7season.py           # 실: 7시즌 wide grid (수십분~시간)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

# edge-pinning 해소: 기본(beta .12-.22 / amp .45-.90 / phase 90-135)보다 양끝 넓힘 → optimum bracket.
# v2(2026-06-13): 1차 wide grid서도 2019/2022/2024 가 상단 핀(amp>1.3·β>0.3) → upper bound 더 확장.
WIDE_BETA = (0.08, 0.15, 0.22, 0.30, 0.38, 0.46)
WIDE_AMP = (0.30, 0.65, 1.00, 1.40, 1.80, 2.20)
WIDE_PHASE = (40.0, 78.0, 116.0, 154.0, 192.0, 230.0)


def _is_edge(val, grid) -> bool:
    return val <= min(grid) + 1e-9 or val >= max(grid) - 1e-9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--grid-agents", type=int, default=15000, help="grid-search 시 N (빠른 랭킹)")
    ap.add_argument("--refine-agents", type=int, default=37500, help="best forcing 정밀화 N")
    ap.add_argument("--out", default="simulation/results/abm/agent_world_7season.json")
    args = ap.parse_args()

    from simulation.abm.epi_proof import _load_ili_seasons
    from simulation.abm.agent_world_fit import calibrate_agent_world, evaluate_agent_world_full

    db = Path("simulation/data/db/epi_real_seoul.db")
    seasons = [s for s in _load_ili_seasons(db) if len(s.ili_rate) >= 50]
    seasons.sort(key=lambda s: s.season)

    if args.smoke:
        seasons = [s for s in seasons if s.season == 2023][:1] or seasons[:1]
        beta_g, amp_g, phase_g = (0.14, 0.19), (0.55, 0.80), (90.0, 120.0)
        grid_agents, refine_agents, grid_seeds, refine_seeds = 2000, 2000, (1,), (1,)
        print(f"[abm-7season] SMOKE — 1시즌(season={seasons[0].season}) 2×2×2 grid N=2000")
    else:
        beta_g, amp_g, phase_g = WIDE_BETA, WIDE_AMP, WIDE_PHASE
        grid_agents, refine_agents = args.grid_agents, args.refine_agents
        grid_seeds, refine_seeds = (1, 2), (1, 2, 3, 4, 5)
        print(f"[abm-7season] 실 — {len(seasons)}시즌 × {len(beta_g)}×{len(amp_g)}×{len(phase_g)} "
              f"grid (N={grid_agents}→refine {refine_agents}). 시즌: {[s.season for s in seasons]}")

    results = []
    t_all = time.time()
    for s in seasons:
        t0 = time.time()
        cal = calibrate_agent_world(s, n_agents=grid_agents, seeds=grid_seeds,
                                    beta_grid=beta_g, amp_grid=amp_g, phase_grid=phase_g)
        if "error" in cal:
            print(f"  season {s.season}: {cal['error']}"); results.append({"season": s.season, **cal}); continue
        f = cal["forcing"]
        edge = {"beta": _is_edge(f["beta"], beta_g), "amp": _is_edge(f["beta_amp"], amp_g),
                "phase": _is_edge(f["beta_phase"], phase_g)}
        cal["edge_pinned"] = bool(any(edge.values()))
        cal["edge_detail"] = edge
        # 정밀화: best forcing 으로 134-metric (headline 수치)
        full = evaluate_agent_world_full(s, f, n_agents=refine_agents, seeds=refine_seeds)
        cal["refine_surface"] = full.get("surface") if "error" not in full else {"error": full.get("error")}
        cal["secs"] = round(time.time() - t0, 1)
        results.append(cal)
        rs = cal["refine_surface"]
        rs_r2 = rs.get("r2") if isinstance(rs, dict) else None
        print(f"  ✓ season {s.season}: grid R²={cal['r2']:.4f} wis={cal['wis']:.3f} corr={cal['corr']:.3f} "
              f"hit0.8={cal['hit_0p8']} edge={cal['edge_pinned']} | refine R²={rs_r2}  ({cal['secs']:.0f}s)")

    # 요약
    ok = [r for r in results if "r2" in r]
    summary = {
        "n_seasons": len(results),
        "n_hit_0p8": sum(1 for r in ok if r.get("hit_0p8")),
        "n_edge_pinned": sum(1 for r in ok if r.get("edge_pinned")),
        "best_season": max(ok, key=lambda r: r["r2"])["season"] if ok else None,
        "best_r2": max((r["r2"] for r in ok), default=None),
        "grid": {"beta": list(beta_g), "amp": list(amp_g), "phase": list(phase_g)},
        "per_season": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(summary, open(args.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"\n[abm-7season] {summary['n_hit_0p8']}/{summary['n_seasons']} 시즌 R²≥0.8, "
          f"edge-pinned {summary['n_edge_pinned']}, best={summary['best_season']}(R²={summary['best_r2']}). "
          f"전체 {time.time()-t_all:.0f}s\n→ {args.out}")


if __name__ == "__main__":
    main()
