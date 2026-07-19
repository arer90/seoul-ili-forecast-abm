"""Data import + ensemble orchestrate CLI commands — extracted from __main__.py.

Phase C2 partial (2026-05-12 cont.): 2 data-pipeline handlers (import-external,
orchestrate) moved here.
"""
from __future__ import annotations

import logging
import sys


log = logging.getLogger(__name__)


def cmd_import_external(args) -> None:
    """`python -m simulation import-external` — WHO FluNet / KOSIS / commuter import.

    Targets:
      --all (default if no flags)  : FluNet + commuter + KOSIS-gender + KOSIS-registry
      --flunet                     : WHO FluNet + metadata
      --commuter                   : KOSIS commuter-matrix
      --kosis-gender               : KOSIS disease-by-gender
      --kosis-registry             : KOSIS source registry
      --scan                       : scan available files only (no import)
    """
    from simulation.collectors import import_external as ie
    from simulation.database.config import DB_PATH

    db = str(DB_PATH)
    if getattr(args, "scan", False):
        ie.scan_available(db)
        return

    any_specific = any([
        getattr(args, "flunet", False),
        getattr(args, "commuter", False),
        getattr(args, "kosis_gender", False),
        getattr(args, "kosis_registry", False),
    ])

    if getattr(args, "all_", False) or not any_specific:
        total = ie.import_all(db)
        print(f"\nimport-external --all : {total} rows imported")
        return

    total = 0
    if args.flunet:
        total += ie.import_flunet(db)
        total += ie.import_flunet_metadata(db)
    if args.commuter:
        total += ie.import_commuter_matrix(db)
    if args.kosis_gender:
        total += ie.import_kosis_disease_gender(db)
    if args.kosis_registry:
        total += ie.import_kosis_source_registry(db)
    print(f"\nimport-external : {total} rows imported")


def cmd_orchestrate(args) -> None:
    """`python -m simulation orchestrate` — 3-stage tournament on OOF predictions."""
    import json
    from pathlib import Path

    import numpy as np

    from simulation.ensembles import TournamentOrchestrator
    from simulation.models.registry import PAPER_PRIMARY_11

    oof_path = Path(args.oof_json)
    cat_path = Path(args.categories_json)
    out_path = Path(args.out)

    if not oof_path.exists():
        print(f"ERROR: OOF file not found: {oof_path}")
        sys.exit(2)
    if not cat_path.exists():
        print(f"ERROR: categories file not found: {cat_path}")
        sys.exit(2)

    payload = json.loads(oof_path.read_text(encoding="utf-8"))
    cats = json.loads(cat_path.read_text(encoding="utf-8"))
    if "y_true" not in payload:
        print("ERROR: OOF JSON must contain 'y_true' key")
        sys.exit(2)
    y_true = np.asarray(payload.pop("y_true"), dtype=float)
    oof = {k: np.asarray(v, dtype=float) for k, v in payload.items()}

    paper_names = [n for n, _ in PAPER_PRIMARY_11]

    orch = TournamentOrchestrator(
        top_k_per_category=args.top_k,
        caruana_steps=args.caruana_steps,
        artifacts_dir=out_path.parent,
    )
    result = orch.run(
        oof_predictions=oof,
        y_true=y_true,
        model_categories=cats,
        paper_primary=paper_names,
    )
    result.save_trace(out_path)
    print(f"[orchestrate] champion={result.final_ensemble_name}  "
          f"R²={result.final_r2:.4f}  MAE={result.final_mae:.4f}")
    print(f"  trace: {out_path}")


__all__ = [
    "cmd_import_external",
    "cmd_orchestrate",
]
