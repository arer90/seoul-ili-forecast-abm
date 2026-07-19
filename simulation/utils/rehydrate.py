"""
rehydrate — register legacy bare-model `.pt` files as ChampionLog entries.
==========================================================================

R9 (per_model_optimize)'s champion-challenger only writes ``models/<name>.pt``
for models that complete its (transform × scaler) grid search and beat the
current champion. But many earlier training runs (R2 baseline / R3
external) write bare-model `.pt` files **without** going through R9
(per_model_optimize) — those models are runnable but absent from
``champion_log.json``, so ``predict-real`` and ``visualize`` skip them.

This module bridges the gap:

  1. Scan ``models/*.pt`` for files not in ``champion_log.json``.
  2. For each, ``load_artifact()`` (auto-wraps bare-model pickle in a
     legacy ``ChampionArtifact`` with ``identity`` transform + no scaler).
  3. If a matching record exists in ``post_E_eval.json``, populate
     ``test_wis / test_mae`` from there (so the legacy entry has real
     metrics, not nulls).
  4. Write a synthetic ``current`` block to ``champion_log.json`` so
     downstream tools see all 26 (or more) champions.

This is **idempotent**: re-running won't re-register already-registered
models. Use ``--force`` to overwrite legacy entries with fresh metrics.

CLI:
    simulation rehydrate                       # scan + register
    simulation rehydrate --dry-run             # preview only
    simulation rehydrate --force               # overwrite even if exists
    simulation rehydrate --eval-source post_E  # import metrics from post_E_eval
"""
from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def _load_post_e_metrics(path: Path) -> dict[str, dict]:
    """Read post_E_eval.json → {model_name: {wis, mae, rmse, r2}}."""
    if not path.exists():
        return {}
    try:
        e = json.loads(path.read_text())
        out: dict[str, dict] = {}
        for d in e.get("details", []):
            nm = d.get("model")
            if not nm:
                continue
            rmse = d.get("rmse_boot95")
            if isinstance(rmse, dict):
                rmse = rmse.get("point") or rmse.get("value")
            out[nm] = {
                "wis":  d.get("wis"),
                "mae":  d.get("mae"),
                "rmse": rmse,
                "crps": d.get("crps_gaussian"),
                "n":    d.get("n"),
            }
        return out
    except Exception as e:
        log.warning(f"  [rehydrate] post_E_eval read failed: {e}")
        return {}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def run_rehydrate(*, dry_run: bool = False, force: bool = False,
                    eval_source: str = "post_E",
                    repo_root: Optional[Path] = None) -> dict:
    """Register every legacy .pt in models/ that's not already in champion_log."""
    if repo_root is None:
        repo_root = Path.cwd()
    repo_root = Path(repo_root)
    models_dir = repo_root / "models"
    log_path = models_dir / "champion_log.json"

    # 1. Read existing champion log
    if log_path.exists():
        try:
            log_data = json.loads(log_path.read_text())
        except Exception:
            log_data = {}
    else:
        log_data = {}

    registered = set(log_data.keys())

    # 2. Find legacy candidates (canonical <name>.pt files only)
    candidates: list[Path] = []
    for pt in models_dir.glob("*.pt"):
        name = pt.stem
        if "_v" in name or "_attempt_v" in name:
            continue   # skip archive / attempt files
        candidates.append(pt)

    # 3. post_E metrics lookup
    post_e_path = repo_root / "simulation" / "results" / "post_E_eval.json"
    post_e_metrics = (_load_post_e_metrics(post_e_path)
                       if eval_source == "post_E" else {})

    # 4. Walk + try to load each as ChampionArtifact (auto-wraps legacy)
    from simulation.utils.model_artifact import load_artifact, ChampionArtifact

    print("\n" + "=" * 70)
    print("  simulation rehydrate — register legacy .pt as champion entries")
    print("=" * 70)
    if dry_run:
        print("  Mode: --dry-run (preview only)")
    print()

    actions: list[dict] = []
    for pt in sorted(candidates):
        name = pt.stem
        already = name in registered
        if already and not force:
            actions.append({"model": name, "action": "skip_exists",
                              "reason": "already in champion_log"})
            continue

        artifact = load_artifact(pt)
        if artifact is None:
            actions.append({"model": name, "action": "skip_unloadable",
                              "reason": "load_artifact returned None"})
            continue
        is_legacy = bool((artifact.config or {}).get("legacy", False))
        size_mb = pt.stat().st_size / 1e6

        # Pull metrics from post_E
        metrics = post_e_metrics.get(name, {})
        wis = metrics.get("wis")
        mae = metrics.get("mae")
        rmse = metrics.get("rmse")
        n_eval = metrics.get("n")

        record_action = "register_legacy" if is_legacy else "register_artifact"
        if already:
            record_action = "force_overwrite"

        if dry_run:
            actions.append({
                "model": name, "action": record_action + ":dry",
                "size_mb": round(size_mb, 2),
                "is_legacy": is_legacy,
                "wis_from_post_E": wis,
                "mae_from_post_E": mae,
            })
            continue

        # Tier label (auto)
        try:
            from simulation.models.registry import tier_of, category_of
            tier = tier_of(name)
            cat = category_of(name)
        except Exception:
            tier, cat = "unknown", "unknown"

        # 5. Build a synthetic 'current' record
        current_block = {
            "version": 0,        # 0 = legacy / pre-R9 (per_model_optimize)
            "filename": pt.name,
            "promoted_at": _utcnow_iso(),
            "tier":     tier,
            "category": cat,
            "config": {
                "transform": artifact.transform_name,
                "scaler":    (artifact.scaler.__class__.__name__
                               if artifact.scaler else "none"),
                "n_features": ((artifact.config or {}).get("n_features") or
                                (len(artifact.feature_indices)
                                 if artifact.feature_indices else None)),
                "artifact":   ("ChampionArtifact" if not is_legacy
                                else "legacy_bare_model"),
                "rehydrated":  True,
                "rehydrate_source": "post_E_eval" if metrics else "none",
            },
        }
        if isinstance(wis, (int, float)):
            current_block["test_wis"] = float(wis)
        if isinstance(mae, (int, float)):
            current_block["test_mae"] = float(mae)
        if isinstance(rmse, (int, float)):
            current_block["test_rmse"] = float(rmse)
        if isinstance(n_eval, int):
            current_block["test_n"] = int(n_eval)

        # Preserve any prior history
        prior = log_data.get(name, {})
        new_entry = {
            "current": current_block,
            "history": prior.get("history", []),
        }
        log_data[name] = new_entry
        actions.append({
            "model": name, "action": record_action,
            "size_mb": round(size_mb, 2),
            "is_legacy": is_legacy,
            "metrics": {"wis": wis, "mae": mae, "rmse": rmse},
        })

    # 6. Persist (unless dry-run)
    if not dry_run:
        log_path.write_text(json.dumps(log_data, indent=2, default=str))

    # 7. Summary
    by_action: dict[str, int] = {}
    for a in actions:
        by_action[a["action"]] = by_action.get(a["action"], 0) + 1
    for ac, c in sorted(by_action.items()):
        print(f"  {ac:<28} {c} models")
    print()

    if actions:
        print(f"  {'model':<25} {'action':<24} {'WIS':>7} {'MAE':>7} {'is_legacy':>10}")
        print("  " + "-" * 78)
        for a in actions[:50]:
            m = a.get("metrics", {}) if isinstance(a.get("metrics"), dict) else {}
            wis_v = m.get("wis")
            mae_v = m.get("mae")
            wis_s = f"{wis_v:.2f}" if isinstance(wis_v,(int,float)) else "?"
            mae_s = f"{mae_v:.2f}" if isinstance(mae_v,(int,float)) else "?"
            legacy_s = "YES" if a.get("is_legacy") else "no"
            print(f"  {a['model']:<25} {a['action']:<24} "
                  f"{wis_s:>7} {mae_s:>7} {legacy_s:>10}")
    print()
    print(f"  Total in champion_log.json after rehydrate: {len(log_data)}")
    print("=" * 70)
    return {
        "actions":      actions,
        "n_registered": len(log_data),
        "log_path":     str(log_path),
        "dry_run":      dry_run,
    }


__all__ = ["run_rehydrate"]
