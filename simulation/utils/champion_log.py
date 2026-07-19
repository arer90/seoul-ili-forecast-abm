"""
Champion–challenger logic for trained model checkpoints (.pt files).
=====================================================================

Problem: every R2 (baseline) run + every R9 (per_model_optimize) refit overwrote `models/<M>.pt`
even when the new model was *worse* than the prior version on the test slab.
Under the user's request:
  • If the newly-trained model has BETTER test-slab score → promote (replace)
  • Else → keep the current champion, archive the new attempt

This module provides:

  ChampionLog(models_dir, log_path)
    .propose(name, new_pickle_bytes, new_score, *, metric="wis", lower_better=True)
        → "promoted" | "kept_current" | "archived_only"
    .current_score(name, metric)  → float | None
    .history(name) → list[record]
    .summary() → {model: {current_version, current_score, n_versions}}

Layout:

  models/
    XGBoost.pt                    ← current champion
    XGBoost_v3_20260425_174300.pt ← previous champion (archived on demotion)
    XGBoost_v2_20260425_153100.pt
    XGBoost_v1_20260425_120000.pt
  models/champion_log.json        ← single source of truth for promotions

JSON schema (champion_log.json):
  {
    "XGBoost": {
        "current": {"version": 4, "filename": "XGBoost.pt",
                     "test_wis": 3.42, "test_mae": 4.12,
                     "promoted_at": "2026-04-25T17:50:00Z",
                     "config": {"transform":"boxcox","scaler":"robust"}},
        "history": [
            {"version": 1, "filename": "XGBoost_v1_*.pt", "test_wis": 5.1,
              "promoted_at": "...", "demoted_at": "..."},
            ...
        ]
    },
    ...
  }
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ChampionLog:
    """Manages .pt promotion / archive based on test-slab metrics."""

    def __init__(self, models_dir: Path, log_path: Optional[Path] = None):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_path or (self.models_dir / "champion_log.json"))
        self.log: dict = self._load()

    def _load(self) -> dict:
        if self.log_path.exists():
            try:
                return json.loads(self.log_path.read_text())
            except Exception as e:
                log.warning(f"  [champion-log] load failed ({e}); starting fresh")
        return {}

    def _save(self) -> None:
        self.log_path.write_text(json.dumps(self.log, indent=2, default=str))

    def current_score(self, name: str, metric: str = "wis") -> Optional[float]:
        rec = self.log.get(name, {}).get("current")
        if not isinstance(rec, dict):
            return None
        return rec.get(f"test_{metric}")

    def current_filename(self, name: str) -> Optional[Path]:
        rec = self.log.get(name, {}).get("current")
        if not isinstance(rec, dict):
            return None
        fn = rec.get("filename")
        if fn:
            p = self.models_dir / fn
            return p if p.exists() else None
        return None

    def history(self, name: str) -> list[dict]:
        return self.log.get(name, {}).get("history", [])

    def propose(
        self,
        name: str,
        pickle_bytes: bytes,
        new_score: float,
        *,
        metric: str = "wis",
        lower_better: bool = True,
        config: Optional[dict] = None,
        extra_metrics: Optional[dict] = None,
        db_fingerprint: Optional[dict] = None,
    ) -> str:
        """Compare new model to current champion. Promote if better.

        Args:
            name:           Model name (key in champion_log.json).
            pickle_bytes:   Serialised ChampionArtifact (.pt bytes).
            new_score:      Primary metric value (compared against champion).
            metric:         Name of primary metric (default "wis").
            lower_better:   True if lower score = better (default True).
            config:         Free-form config dict stored verbatim in log.
            extra_metrics:  Additional metrics to store (mae, r2, rmse …).
            db_fingerprint: Output of compute_db_fingerprint() — embedded in
                            the champion record so downstream tools can verify
                            that two runs used the same DB state.  Pass None
                            to omit (back-compat).

        Returns one of:
          "promoted"       — current champion replaced
          "kept_current"   — current is better, new attempt archived (history)
          "no_current"     — first time, promoted automatically

        Side effects:
          - Writes models/<name>.pt (champion) on promotion
          - Writes models/<name>_vN_<ts>.pt (historical) on demotion
          - Updates champion_log.json
        """
        ts = _utcnow_iso()
        ts_short = ts.replace("-", "").replace(":", "").replace("Z", "").replace("T", "_")
        current = self.log.get(name, {})
        current_score = self.current_score(name, metric)

        # Decide promotion
        if current_score is None:
            decision = "no_current"
            promote = True
        else:
            if lower_better:
                promote = float(new_score) < float(current_score)
            else:
                promote = float(new_score) > float(current_score)
            decision = "promoted" if promote else "kept_current"

        cur_record = current.get("current") if isinstance(current.get("current"), dict) else None
        history = list(current.get("history", []))
        next_version = (cur_record.get("version", 0) if cur_record else 0) + 1

        # Tier auto-label (paper / extra / negative / unknown)
        try:
            from simulation.models.registry import tier_of, category_of
            tier = tier_of(name)
            cat = category_of(name)
        except Exception:
            tier, cat = "unknown", "unknown"

        if promote:
            # 1. Demote current → archive its file with version suffix
            if cur_record is not None and cur_record.get("filename"):
                old_fn = cur_record["filename"]
                old_path = self.models_dir / old_fn
                if old_path.exists():
                    archive_name = f"{name}_v{cur_record.get('version', 0)}_{cur_record.get('promoted_at','').replace(':','').replace('-','').replace('Z','').replace('T','_')}.pt"
                    archive_path = self.models_dir / archive_name
                    try:
                        shutil.move(str(old_path), str(archive_path))
                    except Exception as e:
                        log.warning(f"  [champion-log] archive {old_path} → {archive_path} failed: {e}")
                        archive_name = old_fn  # fallback: original filename
                    cur_record["demoted_at"] = ts
                    cur_record["filename_archived"] = archive_name
                history.append(cur_record)

            # 2. Write new champion
            champion_path = self.models_dir / f"{name}.pt"
            champion_path.write_bytes(pickle_bytes)

            new_record = {
                "version": next_version,
                "filename": f"{name}.pt",
                "promoted_at": ts,
                "config": config or {},
                "tier":     tier,        # ← paper / extra / negative
                "category": cat,         # ← sub-category
                f"test_{metric}": float(new_score),
            }
            if extra_metrics:
                for k, v in extra_metrics.items():
                    new_record[f"test_{k}"] = float(v) if isinstance(v, (int, float)) else v
            # DB fingerprint — compact hash of training data state at promotion time.
            # Used by compare_v1_v2 to guard against same-name comparisons on
            # different data snapshots.
            if db_fingerprint is not None:
                new_record["db_fingerprint"] = {
                    "combined_sha256": db_fingerprint.get("combined_sha256", "unknown"),
                    "computed_at":     db_fingerprint.get("computed_at"),
                    "db_path":         db_fingerprint.get("db_path"),
                }
            self.log[name] = {"current": new_record, "history": history}
            self._save()
            log.info(
                f"  [champion-log] {name}: PROMOTED to v{next_version} "
                f"(test_{metric}={new_score:.4f}"
                + (f" vs prev {current_score:.4f}" if current_score is not None else " — first")
                + ")"
            )
            return decision

        # Not promoted → archive new attempt for traceability
        archive_name = f"{name}_attempt_v{next_version}_{ts_short}.pt"
        (self.models_dir / archive_name).write_bytes(pickle_bytes)
        attempt_record = {
            "version": next_version,
            "filename": archive_name,
            "attempted_at": ts,
            "config": config or {},
            "tier":     tier,
            "category": cat,
            f"test_{metric}": float(new_score),
            "outcome": "kept_current",
        }
        if extra_metrics:
            for k, v in extra_metrics.items():
                attempt_record[f"test_{k}"] = float(v) if isinstance(v, (int, float)) else v
        history.append(attempt_record)
        self.log[name]["history"] = history
        self._save()
        log.info(
            f"  [champion-log] {name}: KEPT current (champion test_{metric}="
            f"{current_score:.4f}, new={new_score:.4f}, archived as {archive_name})"
        )
        return decision

    def summary(self) -> dict:
        out = {}
        for name, rec in self.log.items():
            cur = rec.get("current", {})
            hist = rec.get("history", [])
            # Tier — prefer stored, else resolve on the fly
            tier = cur.get("tier")
            cat = cur.get("category")
            if tier is None:
                try:
                    from simulation.models.registry import tier_of, category_of
                    tier = tier_of(name)
                    cat = category_of(name)
                except Exception:
                    tier, cat = "unknown", "unknown"
            out[name] = {
                "tier":     tier,         # ← paper / extra / negative
                "category": cat,
                "current_version": cur.get("version"),
                "current_test_wis": cur.get("test_wis"),
                "current_test_mae": cur.get("test_mae"),
                "current_test_r2":  cur.get("test_r2"),
                "config": cur.get("config", {}),
                "promoted_at": cur.get("promoted_at"),
                "n_versions_total": len(hist) + (1 if cur else 0),
                "n_archived_attempts": sum(
                    1 for h in hist if h.get("outcome") == "kept_current"
                ),
            }
        return out

    def write_summary_md(self, path: Path) -> None:
        path = Path(path)
        out = self.summary()
        md = ["# Champion-Challenger Log",
              "",
              "_Each model's current best `.pt` is the champion. New training_",
              "_attempts are promoted only if their test-slab score beats the_",
              "_current champion; otherwise archived as `<name>_attempt_v*.pt`._",
              "",
              "| Model | v# | test_WIS | test_MAE | test_R² | promoted_at | total versions | failed attempts |",
              "|---|---|---|---|---|---|---|---|"]
        for name in sorted(out):
            s = out[name]
            md.append(
                f"| {name} | {s['current_version']} | "
                f"{s.get('current_test_wis', '?')} | "
                f"{s.get('current_test_mae', '?')} | "
                f"{s.get('current_test_r2', '?')} | "
                f"{s.get('promoted_at', '?')} | "
                f"{s['n_versions_total']} | "
                f"{s['n_archived_attempts']} |"
            )
        path.write_text("\n".join(md))
