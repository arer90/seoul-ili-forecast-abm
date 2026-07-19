"""simulation.abm.variant_ablation
====================================
Realism-ablation comparison: score ABM variants A / B (and later C / D) against
the SAME real observed + forward Seoul ILI, apples-to-apples.

Variant A = mean-field district transmission (the current baseline). Variant B =
agent-to-agent contact-network transmission (:func:`run_agent_world` with
``transmission_mode='network'``), which additionally yields person-like outputs
(who-infected-whom tree, offspring / superspreading distribution, per-layer and
occupation attack shares) that the mean-field model structurally cannot produce.

Fairness: both variants use the SAME fixed DB-grounded Seoul population, the SAME
seeds, and the SAME season-tail initial state; each variant's transmission scale
(beta) is calibrated LEAK-FREE on the in-sample window only, then run forward and
scored against the real ``sentinel_influenza`` forward window via the harness's
own affine map + R². The forecaster (FusedEpi) remains the accuracy anchor — this
ablation measures which mechanism level best serves the ABM's mechanism/spatial
role, NOT whether the ABM beats the forecaster.

Honest MVP note: the initial season-tail state (infectious + recovered fractions)
is a structural assumption, not calibrated; this is a mechanism comparison, and
absolute R² should be read against variant A on the same footing, not against the
forecast-anchored 0.722 headline (a different, anchored arm).
"""
from __future__ import annotations

import numpy as np

from simulation.abm.agent_kernel import STATE_I, STATE_R, STATE_S, run_agent_world
from simulation.abm.synthetic_population import generate_population
from simulation.scripts.run_abm_forward_validation import (
    _fit_linear_map,
    _r2,
    _rmse,
    _weekly_from_daily_I,
    load_real_ili_split,
)

__all__ = ["compare_variants", "compare_anchored_variants", "tree_metrics"]

# NOTE: a single constant-beta epidemic is sharper than a real ILI season, and the
# harness's fitted seasonal forcing (beta_phase in calendar days) does not transfer
# to this simplified day-0-relative run — so ABSOLUTE aggregate forward-fit here is
# limited and, honestly, negative for BOTH A and B. That is itself the point: the
# ABM (either mechanism) is NOT the aggregate accuracy anchor (the forecaster is);
# variant B's VALUE is the person-like metrics it uniquely produces. A calendar-
# aligned anchored comparison is future work (the existing 0.722 harness machinery).
_FORWARD_KW = dict(sigma=0.45, gamma=0.18, delta=0.002, nu=0.0002)


def _initial_state(n: int, *, infectious_frac: float, recovered_frac: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Season-tail initial state: mostly R (post-peak immunity) + some I, rest S."""
    state = np.full(n, STATE_S, dtype=np.int8)
    idx = rng.permutation(n)
    n_i = int(round(infectious_frac * n))
    n_r = int(round(recovered_frac * n))
    state[idx[:n_i]] = STATE_I
    state[idx[n_i:n_i + n_r]] = STATE_R
    return state


def _full_curve(variant: str, pop: dict, *, beta: float, n_weeks_total: int,
                seeds, init_frac: float) -> tuple[np.ndarray, list]:
    """Mean weekly infected curve over ONE continuous in-sample→forward run.

    Starting from a small seed, a constant-beta epidemic rises, peaks, and declines
    by endogenous susceptible depletion — so the forward-window decline is generated
    by the mechanism, not imposed by a guessed immunity level. Returns the full
    length-``n_weeks_total`` weekly curve (in-sample + forward) and B's trees.
    """
    n = pop["home_gu"].size
    curves, trees = [], []
    for seed in seeds:
        init = _initial_state(n, infectious_frac=init_frac, recovered_frac=0.0,
                              rng=np.random.default_rng(1000 + int(seed)))
        kw = dict(N=n, T_days=n_weeks_total * 7, beta=beta, population=pop,
                  global_seed=int(seed), import_rate=3.0e-4,
                  initial_state=init, **_FORWARD_KW)
        if variant == "B":
            kw["transmission_mode"] = "network"
        out = run_agent_world(**kw)
        curves.append(_weekly_from_daily_I(out["I"], n_weeks_total))
        if variant == "B":
            trees.append(out.get("transmission_tree"))
    return np.mean(curves, axis=0), trees


def _calibrate_beta(variant: str, pop: dict, in_sample_ili: np.ndarray,
                    n_weeks_total: int, *, seeds, beta_grid, init_frac: float) -> dict:
    """Leak-free: pick the beta whose IN-SAMPLE portion best fits in-sample ILI."""
    n_in = len(in_sample_ili)
    # (offset, scale) — the order _fit_linear_map returns. Identity is (0.0, 1.0):
    # the old (1.0, 0.0) default mapped every curve to the constant 1.0.
    best = {"beta": beta_grid[0], "r2": -np.inf, "curve": None, "trees": None,
            "affine": (0.0, 1.0)}
    for beta in beta_grid:
        curve, trees = _full_curve(variant, pop, beta=beta, n_weeks_total=n_weeks_total,
                                   seeds=seeds, init_frac=init_frac)
        ins = curve[:n_in]
        # _fit_linear_map returns (offset, scale) — applying it as scale*x+offset
        # was transposed here, which is why the shipped in_sample_r2 was -766.65.
        # An affine fit scored on the data it was fitted to cannot be negative.
        offset, scale = _fit_linear_map(ins, in_sample_ili)   # fit on IN-SAMPLE only
        r2 = _r2(in_sample_ili, offset + scale * ins)
        if r2 > best["r2"]:
            best = {"beta": float(beta), "r2": float(r2), "curve": curve,
                    "trees": trees, "affine": (float(offset), float(scale))}
    return best


def tree_metrics(trees: list, pop: dict) -> dict:
    """Person-like metrics from variant-B transmission trees (aggregated over seeds).

    Returns offspring/superspreading dispersion, per-layer transmission share,
    occupation-specific attack counts, and a district-to-district OD summary — the
    outputs a mean-field model cannot produce.
    """
    # CRITICAL: agent index j in seed 0 and seed 1 are DIFFERENT simulated people.
    # Everything that keys on an agent index (offspring bincount, occupation attack)
    # MUST be counted PER SEED and only then pooled — concatenating raw indices and
    # bincount()-ing merges distinct agents, inflating offspring/superspreader counts;
    # a seed-pooled numerator over a single-population denominator inflates the
    # occupation attack by ~n_seeds. Pooling-invariant ratios (layer share, cross-
    # district share) are accumulated as counts and normalized once at the end.
    occ = np.asarray(pop["occupation"])
    home = np.asarray(pop["home_gu"])
    uniq_occ = np.unique(occ)
    pop_o = {int(o): int((occ == o).sum()) for o in uniq_occ}
    layer_names: list = []
    k_pool: list = []                                   # per-seed offspring counts, pooled
    occ_attack_seeds: dict = {int(o): [] for o in uniq_occ}
    layer_counts = None
    n_edges = n_a2a = n_imported = n_cross = n_seeds_used = 0
    for tt in trees:
        if not tt or tt["infectee"].size == 0:
            continue
        n_seeds_used += 1
        infector = np.asarray(tt["infector"])
        infectee = np.asarray(tt["infectee"])
        layer = np.asarray(tt["layer"])
        layer_names = tt["layer_names"]
        net = infector >= 0
        n_edges += int(infector.size)
        n_a2a += int(net.sum())
        n_imported += int((~net).sum())
        if net.any():
            ks = np.bincount(infector[net])             # THIS seed's offspring only
            k_pool.append(ks[ks > 0])
            if layer_counts is None:
                layer_counts = np.zeros(len(layer_names))
            lc = np.bincount(layer[net], minlength=len(layer_names))
            layer_counts[:len(layer_names)] += lc[:len(layer_names)]
            n_cross += int((home[infector[net]] != home[infectee[net]]).sum())
        inf_occ = occ[infectee]                         # THIS seed's per-occupation share
        for o in uniq_occ:
            occ_attack_seeds[int(o)].append(
                float((inf_occ == o).sum()) / max(pop_o[int(o)], 1))
    if n_seeds_used == 0:
        return {"n_edges": 0}
    k = np.concatenate(k_pool) if k_pool else np.array([], dtype=float)
    k_mean = float(k.mean()) if k.size else 0.0
    disp = float(k.var() / k_mean) if k_mean > 0 else 0.0
    total_layer = float(layer_counts.sum()) if layer_counts is not None else 0.0
    layer_share = {name: round(float(layer_counts[li]) / total_layer, 3) if total_layer > 0 else 0.0
                   for li, name in enumerate(layer_names)}
    occ_attack = {o: round(float(np.mean(v)), 4) for o, v in occ_attack_seeds.items()}
    return {
        "n_edges": n_edges,
        "n_agent_to_agent": n_a2a,
        "n_imported": n_imported,
        "n_seeds_used": n_seeds_used,
        "offspring_k_mean": round(k_mean, 3),
        "offspring_k_dispersion": round(disp, 3),
        # pooled superspreader count scales with seeds; the per-run mean is the
        # config-robust figure to quote in prose.
        "superspreaders_k_ge5": int((k >= 5).sum()) if k.size else 0,
        "mean_superspreaders_per_run_k_ge5": round(float((k >= 5).sum()) / n_seeds_used, 2)
        if k.size else 0.0,
        "superspreader_share_k_ge5": round(float((k >= 5).mean()), 4) if k.size else 0.0,
        "layer_share": layer_share,
        "occupation_attack_rate": occ_attack,
        "cross_district_transmission_share": round(float(n_cross) / max(n_a2a, 1), 3),
    }


def entity_metrics(pop: dict, network_kwargs: dict, agents_final_state: np.ndarray,
                   *, seed: int = 0) -> dict:
    """Per-ENTITY (household/workplace/school) attack-rate distribution (variant C).

    The contact network's household/workplace/school layers ARE persistent
    entities: each connected component is one household, workplace, or class. This
    rebuilds the (deterministic) layers and reports, per layer, the distribution of
    per-entity attack rate (share of members ever infected) — the "which households
    and workplaces had outbreaks" view a mean-field model cannot give.
    """
    from scipy.sparse.csgraph import connected_components

    from simulation.abm.agent_kernel import STATE_D, STATE_E, STATE_I, STATE_R
    from simulation.abm.contact_network import build_multilayer_network

    nk = {k: v for k, v in (network_kwargs or {}).items() if k != "provenance"}
    layers = build_multilayer_network(pop, seed=int(seed), **nk)
    ever = np.isin(np.asarray(agents_final_state), [STATE_E, STATE_I, STATE_R, STATE_D])
    out = {}
    for name in ("household", "workplace", "school"):
        _, labels = connected_components(layers[name], directed=False)
        n_ent = int(labels.max()) + 1
        members = np.bincount(labels, minlength=n_ent).astype(float)
        infected = np.bincount(labels, weights=ever.astype(float), minlength=n_ent)
        real = members >= 2                         # ignore singletons
        ar = np.divide(infected[real], members[real],
                       out=np.zeros(int(real.sum())), where=members[real] > 0)
        # "≥50% infected" is SIZE-CONFOUNDED — trivial for a 2-person household, near
        # impossible for a 25-person class — so it wrongly reads ~0 for schools even
        # though schools have the HIGHEST mean attack. The size-fair outbreak metric
        # is the share of units with a within-unit outbreak (≥2 members infected =
        # transmission spread beyond a single seed), the standard household-SAR quantity.
        infected_real = infected[real]
        out[name] = {
            "n_entities": int(real.sum()),
            "mean_attack_rate": round(float(ar.mean()) if ar.size else 0.0, 3),
            "share_entities_with_within_spread": round(
                float((infected_real >= 2).mean()) if ar.size else 0.0, 3),
            "share_entities_majority_infected": round(
                float((ar >= 0.5).mean()) if ar.size else 0.0, 3),
            "max_attack_rate": round(float(ar.max()) if ar.size else 0.0, 3),
        }
    return out


def compare_variants(*, variants=("A", "B"), n_agents: int = 6000, n_seeds: int = 3,
                     beta_grid=None, init_frac: float = 0.004, pop_seed: int = 0,
                     season_weeks: int = 26) -> dict:
    """Score each variant against the real observed+forward Seoul ILI, apples-to-apples.

    One continuous in-sample→forward run per (variant, beta); beta is calibrated
    LEAK-FREE on the in-sample portion, then the forward portion is scored against
    the real forward window. Returns per-variant {calibrated_beta, in_sample_r2,
    forward_r2, forward_rmse} plus, for B, a person_like block. Reads real ILI
    read-only; writes nothing.
    """
    if beta_grid is None:
        beta_grid = [0.12, 0.15, 0.18, 0.22]  # slow-season range (peak near cutoff)
    split = load_real_ili_split()
    # Calibrate on the CURRENT season's rise only (a single constant-beta epidemic
    # reproduces one season; the full multi-season in-sample would need seasonal
    # forcing). The forward window is that season's declining tail.
    in_sample_ili = split["in_sample_ili"][-season_weeks:]
    forward_ili = split["forward_ili"]
    n_in = len(in_sample_ili)
    n_fwd = len(forward_ili)
    n_total = n_in + n_fwd
    seeds = list(range(n_seeds))
    pop = generate_population(n_agents, seed=pop_seed)

    out = {"n_agents": n_agents, "n_seeds": n_seeds, "n_in_sample_weeks": n_in,
           "n_forward_weeks": n_fwd, "forward_dates": split["forward_dates"],
           "note": "forecaster (FusedEpi) stays the accuracy anchor; this compares "
                   "which ABM mechanism best serves the mechanism/person-like role.",
           "variants": {}}
    for v in variants:
        cal = _calibrate_beta(v, pop, in_sample_ili, n_total, seeds=seeds,
                              beta_grid=beta_grid, init_frac=init_frac)
        fwd_curve = cal["curve"][n_in:n_in + n_fwd]
        # STRICTLY LEAK-FREE: apply the IN-SAMPLE-fit affine to the forward curve —
        # the forward observations never touch calibration OR the count→rate map.
        offset, scale = cal["affine"]
        pred = offset + scale * fwd_curve
        block = {"calibrated_beta": cal["beta"], "in_sample_r2": round(cal["r2"], 4),
                 "forward_r2": round(_r2(forward_ili, pred), 4),
                 "forward_rmse": round(_rmse(forward_ili, pred), 4),
                 "leak_free": "beta + affine calibrated on in-sample only; forward "
                              "observations used solely for scoring"}
        if v == "B":
            block["person_like"] = tree_metrics(cal["trees"], pop)
        out["variants"][v] = block
    return out


def compare_anchored_variants(*, variants=("A", "B"), n_agents: int = 8000,
                              n_seeds: int = 3, network_kwargs: dict | None = None,
                              beta_by_layer: dict | None = None,
                              db_derived: bool = True) -> dict:
    """The AGENT-SIMULATION-PREDICTION track: anchor each variant to the champion
    FusedEpi forecast (leak-free — the forecast is available at forecast time),
    then score the anchored ABM trajectory against the REAL forward window.

    This is how the ABM becomes a genuine forward predictor: FusedEpi supplies the
    accuracy anchor and the agent world (variant A mean-field or B agent-to-agent)
    is calibrated to track it, then run forward. Variant B additionally carries the
    person-like machinery. Returns per-variant {forward_r2, forward_rmse,
    anchor_corr, fitted_forcing}. Reads real ILI + champion forecast read-only.
    """
    from simulation.abm.forecast_anchor import anchor_abm_to_forecast
    from simulation.scripts.run_abm_forward_validation import load_champion_forward_forecast

    mode = {"A": "meanfield", "B": "network", "H": "hybrid"}
    champ = np.asarray(load_champion_forward_forecast(), dtype=float)
    split = load_real_ili_split()
    fwd_obs = split["forward_ili"]
    n_cmp = min(len(champ), len(fwd_obs))
    forecast, obs = champ[:n_cmp], fwd_obs[:n_cmp]
    out = {"n_cmp": n_cmp, "n_agents": n_agents, "n_seeds": n_seeds,
           "anchor_target": "champion FusedEpi forecast (leak-free)",
           "scored_against": "real sentinel_influenza forward window",
           "variants": {}}
    net_provenance = None
    trajs, corrs = {}, {}
    for v in variants:
        needs_net = v in ("B", "H")          # network + hybrid both need contact layers
        nkw = network_kwargs if needs_net else None
        bbl = beta_by_layer if needs_net else None
        if needs_net and nkw is None and db_derived:
            # DATA-DERIVED, leak-free contact structure (per-gu community degree from
            # mobility + cited census/survey constants) — NOT tuned to the forward score.
            from simulation.abm.network_params_from_db import derive_network_kwargs
            nkw = derive_network_kwargs()
            net_provenance = nkw.get("provenance")
        anc = anchor_abm_to_forecast(forecast, n_agents=n_agents,
                                     seeds=range(n_seeds), transmission_mode=mode[v],
                                     network_kwargs=nkw, beta_by_layer=bbl)
        traj = np.asarray(anc["anchored_trajectory"], dtype=float)[:n_cmp]
        trajs[v] = traj
        corrs[v] = float(anc["corr_sim_vs_forecast"])
        out["variants"][v] = {
            "forward_r2": round(_r2(obs, traj), 4),
            "forward_rmse": round(_rmse(obs, traj), 4),
            "anchor_corr_sim_vs_forecast": round(anc["corr_sim_vs_forecast"], 4),
            "fitted_forcing": anc["fitted_forcing"],
            "degenerate": bool(anc["degenerate"]),
        }
    # ── A+B reinforcement: prediction-level ensemble ──────────────────────────
    # Blend the two anchored forecasts with a LEAK-FREE weight = each variant's
    # correlation to the FORECAST (never the forward truth). Ensembling averages
    # out A's over-smoothness and B's stochastic structure — often beating both.
    if "A" in trajs and "B" in trajs:
        cA, cB = max(corrs["A"], 0.0), max(corrs["B"], 0.0)
        w = cA / (cA + cB) if (cA + cB) > 0 else 0.5
        blend = w * trajs["A"] + (1.0 - w) * trajs["B"]
        half = 0.5 * (trajs["A"] + trajs["B"])
        out["ensemble_AB"] = {
            "weight_A": round(w, 3), "weight_B": round(1 - w, 3),
            "weight_source": "anchor corr-share vs FORECAST (leak-free; forward untouched)",
            "forward_r2": round(_r2(obs, blend), 4),
            "forward_rmse": round(_rmse(obs, blend), 4),
            "forward_r2_equal_weight": round(_r2(obs, half), 4),
        }
    if net_provenance:
        out["network_provenance"] = net_provenance
    return out


def enkf_couple_forward(*, variant: str = "H", n_agents: int = 20000,
                        n_seeds: int = 8, obs_var: float | None = None) -> dict:
    """Real-time EnKF: assimilate the champion FORECAST into the variant's ABM
    forward ensemble week by week, then score against the real forward window.

    This is the closed-loop real-time track: at each forward week the champion's
    nowcast (available at forecast time — NOT the real forward truth) corrects the
    ABM ensemble state via a stochastic Ensemble Kalman analysis (Burgers 1998).
    Leak-free: only the forecast enters the update; the real forward ILI is used
    solely for scoring. ``obs_var`` defaults to the median ensemble variance — a
    parameter-free choice never tuned to the truth.

    Returns per-variant {variant_alone_r2, variant_plus_enkf_r2, champion_alone_r2}.
    """
    from simulation.abm.forecast_anchor import (
        BEHAVIOUR_OFF, DEFAULT_DISEASE, SeasonSeries, _latest_year, _seed_tuple,
        anchor_abm_to_forecast)
    from simulation.abm.epi_proof import _simulate_replicates
    from simulation.abm.enkf_assimilation import ensemble_kalman_update
    from simulation.database.config import DB_PATH
    from simulation.scripts.run_abm_forward_validation import load_champion_forward_forecast

    mode = {"A": "meanfield", "B": "network", "H": "hybrid"}[variant]
    champ = np.asarray(load_champion_forward_forecast(), dtype=float)
    split = load_real_ili_split()
    fwd_obs = split["forward_ili"]
    n_cmp = min(len(champ), len(fwd_obs))
    forecast, obs_real = champ[:n_cmp], fwd_obs[:n_cmp]

    nk = None
    if variant in ("B", "H"):
        from simulation.abm.network_params_from_db import derive_network_kwargs
        nk = derive_network_kwargs()

    seeds = list(range(n_seeds))
    anc = anchor_abm_to_forecast(forecast, n_agents=n_agents, seeds=seeds,
                                 transmission_mode=mode, network_kwargs=nk)
    disease = {**DEFAULT_DISEASE, **anc["fitted_forcing"]}
    affine = anc["affine"]
    variant_mean = np.asarray(anc["anchored_trajectory"], dtype=float)[:n_cmp]

    # per-seed forward ensemble (m × n_cmp) → ILI scale
    season = SeasonSeries(season=_latest_year(DB_PATH),
                          week_seq=np.arange(n_cmp, dtype=np.int16),
                          ili_rate=forecast)
    reps = _simulate_replicates(season, seeds=_seed_tuple(seeds), n_agents=n_agents,
                                disease=disease, behaviour=BEHAVIOUR_OFF,
                                population_kind="rich_movement",
                                transmission_mode=mode, network_kwargs=nk)
    ens = np.asarray(reps, dtype=float)[:, :n_cmp] * affine["scale"] + affine["offset"]

    if obs_var is None:                     # parameter-free, never tuned to truth
        obs_var = float(np.median(np.var(ens, axis=0)))
    H = np.array([[1.0]])
    R = np.array([[max(obs_var, 1e-9)]])
    assimilated = np.empty(n_cmp)
    assim_ens = np.empty((ens.shape[0], n_cmp))     # per-member assimilated trajectory
    for t in range(n_cmp):
        col = ens[:, t:t + 1]
        if col.var() <= 0:
            assimilated[t] = float(col.mean())
            assim_ens[:, t] = col.ravel()
            continue
        Xa = ensemble_kalman_update(col, np.array([forecast[t]]), H, R, seed=42 + t)
        assimilated[t] = float(Xa.mean())
        assim_ens[:, t] = Xa.ravel()

    return {
        "variant": variant, "n_cmp": n_cmp, "obs_var": round(obs_var, 4),
        # per-member assimilated ensemble (m × n_cmp) — the mechanistic uncertainty
        # spread used for mechanism-informed interval calibration (leak-free: anchored
        # to the forecast, never the forward truth).
        "assimilated_ensemble": assim_ens.tolist(),
        "variant_alone_forward_r2": round(_r2(obs_real, variant_mean), 4),
        "variant_plus_enkf_forward_r2": round(_r2(obs_real, assimilated), 4),
        "variant_plus_enkf_forward_rmse": round(_rmse(obs_real, assimilated), 4),
        "champion_alone_forward_r2": round(_r2(obs_real, forecast), 4),
        "trajectories": {
            "forward_dates": split["forward_dates"][:n_cmp],
            "real": obs_real.tolist(),
            "champion_forecast": forecast.tolist(),
            "variant_alone": variant_mean.tolist(),
            "variant_plus_enkf": assimilated.tolist(),
        },
        "leak_free_verified": True,     # machine-checkable: obs = forecast[t], never obs_real
        "leak_free": "EnKF assimilates the champion FORECAST (forecast-time nowcast), "
                     "never the real forward truth; obs_var=median ensemble variance (untuned)",
    }


def main() -> None:  # pragma: no cover - CLI
    import argparse
    import json
    from pathlib import Path

    ap = argparse.ArgumentParser(description="ABM realism-ablation A/B comparison vs real ILI")
    ap.add_argument("--n-agents", type=int, default=6000)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--variants", nargs="+", default=["A", "B"])
    ap.add_argument("--out", default="simulation/results/abm_variant_ablation.json")
    args = ap.parse_args()

    res = compare_variants(variants=tuple(args.variants), n_agents=args.n_agents,
                           n_seeds=args.n_seeds)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
