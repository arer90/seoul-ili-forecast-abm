#!/usr/bin/env python
"""Exploration harness for nov_shift_conformal — leak-free config search.

Sweeps the shift-aware conformal (scale family x ACI coverage-boost x asym
widener x cold-start seed) and, for each of two leak-free selection objectives
(sharpness-first vs coverage-targeted), reports the chosen config's full-68 and
truly-unseen last-34 WIS/PICP95 vs the deployed additive online conformal.

All selection uses ONLY the first 34 test weeks. No live code modified.
"""
from __future__ import annotations

import json
import sys
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for p in (str(REPO), str(REPO / "scripts")):
    if p not in sys.path:
        sys.path.insert(0, p)

from simulation.analytics.adaptive_conformal import online_conformal_bounds  # noqa: E402
from simulation.analytics.hub_metrics import FLUSIGHT_ALPHAS  # noqa: E402
from fusedepi_fusion_wis import load_frozen  # noqa: E402
from nov_shift_conformal import (  # noqa: E402
    PRIMARY_KI,
    PRIMARY_WINDOW,
    compute_scale,
    score,
    shift_aware_conformal_bounds,
)

REPO = Path(__file__).resolve().parents[1]


def insample_seed_u(model: str, pred: np.ndarray, y: np.ndarray, kind: str, **kw):
    """Normalized in-sample-residual seed |r|/s, leak-free (pre-test). None if absent."""
    j = json.loads((REPO / f"simulation/results/per_model_optimal/{model}.json").read_text())
    ir = (j.get("val_metrics") or {}).get("insample_residuals")
    if not ir:
        return None
    ir = np.asarray(ir, float).ravel()
    # scale seed by the median in-sample scale proxy (level of |r| already absolute);
    # normalize by a representative early-regime scale so units match the buffer
    s0 = float(np.median(compute_scale(pred[:10], y[:10], kind, **kw)))
    return np.abs(ir) / max(s0, 1e-9)


def build_grid():
    g = []
    for kind, kappa_opts, eta_opts in [
        ("sqrt", (0.0,), (0.0,)),
        ("level", (0.0,), (0.0,)),
        ("level_vol", (0.5, 1.0), (0.0, 0.5, 1.0)),
    ]:
        for floor in (1.0, 3.0):
            for kappa in kappa_opts:
                for eta in eta_opts:
                    for ts in (1.0, 0.6, 0.4):
                        for lam in (0.0, 0.4):
                            g.append(dict(kind=kind, floor=floor, kappa=kappa,
                                          eta=eta, target_scale=ts, asym_lambda=lam))
    return g


def make_bounds(pred, y, cfg, seed_u=None):
    sc = compute_scale(pred, y, cfg["kind"], floor=cfg["floor"],
                       kappa=cfg["kappa"], eta=cfg["eta"])
    return shift_aware_conformal_bounds(
        pred, y, FLUSIGHT_ALPHAS, sc,
        window=PRIMARY_WINDOW, ki=PRIMARY_KI,
        asym_lambda=cfg["asym_lambda"], target_scale=cfg["target_scale"],
        seed_u=seed_u,
    )


def main():
    out = {}
    for model in ("TiRex", "FusedEpi"):
        pred, y, _ = load_frozen(model)
        pred = np.asarray(pred, float).ravel()
        y = np.asarray(y, float).ravel()
        n = len(y)
        cut = n // 2
        first = np.zeros(n, bool); first[:cut] = True
        last = np.zeros(n, bool); last[cut:] = True
        full = np.ones(n, bool)
        peak = y >= float(np.quantile(y, 0.75))
        last_peak = last & (y >= float(np.quantile(y[cut:], 0.75)))

        base = online_conformal_bounds(pred, y, FLUSIGHT_ALPHAS,
                                       window=PRIMARY_WINDOW, ki=PRIMARY_KI)
        base_full = score(base, pred, y, full)
        base_last = score(base, pred, y, last)
        base_peak = score(base, pred, y, peak)

        seed_u = None  # keep primary search seed-free (TiRex has no in-sample resid)
        grid = build_grid()
        rows = []
        for cfg in grid:
            b = make_bounds(pred, y, cfg, seed_u=seed_u)
            sf = score(b, pred, y, first)
            rows.append((cfg, b, sf))

        # objective 1: sharpness-first (min first-34 WIS)
        o1 = min(rows, key=lambda r: r[2]["wis"])
        # objective 2: coverage-targeted (min first-34 WIS s.t. first-34 picp95>=0.93)
        feas = [r for r in rows if r[2]["picp95"] >= 0.93]
        o2 = min(feas, key=lambda r: r[2]["wis"]) if feas else o1

        def report(sel):
            cfg, b, _ = sel
            return {
                "config": cfg,
                "full68": score(b, pred, y, full),
                "last34_unseen": score(b, pred, y, last),
                "peak": score(b, pred, y, peak),
                "last34_peak": score(b, pred, y, last_peak),
            }

        # seeded cold-start variant (only where in-sample residuals exist)
        seeded = None
        su = insample_seed_u(model, pred, y, "sqrt", floor=1.0)
        if su is not None:
            b_seed = shift_aware_conformal_bounds(
                pred, y, FLUSIGHT_ALPHAS, compute_scale(pred, y, "sqrt", floor=1.0),
                window=PRIMARY_WINDOW, ki=PRIMARY_KI, seed_u=su,
            )
            seeded = {
                "seed_n": int(len(su)),
                "full68": score(b_seed, pred, y, full),
                "last34_unseen": score(b_seed, pred, y, last),
                "peak": score(b_seed, pred, y, peak),
            }

        out[model] = {
            "baseline_additive": {"full68": base_full, "last34_unseen": base_last, "peak": base_peak},
            "obj1_sharpness_first": report(o1),
            "obj2_coverage_targeted_picp95_ge0.93": report(o2),
            "sqrt_seeded_coldstart": seeded,
        }

    OUTP = Path(
        os.environ.get("MPH_SCRATCH", str(Path(__file__).resolve().parents[1] / "_scratch")) + "/elevate/nov_shift_explore.json"
    )
    OUTP.parent.mkdir(parents=True, exist_ok=True)
    OUTP.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # compact print
    for model, r in out.items():
        print("=" * 74)
        print(model)
        b = r["baseline_additive"]
        print(f"  additive baseline : full WIS={b['full68']['wis']:.3f} PICP95={b['full68']['picp95']:.3f}"
              f" | last34 WIS={b['last34_unseen']['wis']:.3f} PICP95={b['last34_unseen']['picp95']:.3f}"
              f" | peak WIS={b['peak']['wis']:.3f} PICP95={b['peak']['picp95']:.3f}")
        for key in ("obj1_sharpness_first", "obj2_coverage_targeted_picp95_ge0.93"):
            s = r[key]
            c = s["config"]
            print(f"  {key}")
            print(f"    cfg: kind={c['kind']} floor={c['floor']} kappa={c['kappa']} eta={c['eta']}"
                  f" target_scale={c['target_scale']} asym={c['asym_lambda']}")
            print(f"    full WIS={s['full68']['wis']:.3f} PICP95={s['full68']['picp95']:.3f} w95={s['full68']['mean_width95']:.1f}"
                  f" | last34 WIS={s['last34_unseen']['wis']:.3f} PICP95={s['last34_unseen']['picp95']:.3f}"
                  f" | peak WIS={s['peak']['wis']:.3f} PICP95={s['peak']['picp95']:.3f}")
        if r["sqrt_seeded_coldstart"]:
            s = r["sqrt_seeded_coldstart"]
            print(f"  sqrt+seed coldstart: full WIS={s['full68']['wis']:.3f} PICP95={s['full68']['picp95']:.3f}"
                  f" | last34 WIS={s['last34_unseen']['wis']:.3f} PICP95={s['last34_unseen']['picp95']:.3f}"
                  f" | peak PICP95={s['peak']['picp95']:.3f}")
    print("\nwrote", OUTP)
    return out


if __name__ == "__main__":
    main()
