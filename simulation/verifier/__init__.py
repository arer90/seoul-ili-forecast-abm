"""
simulation.verifier — SSOT for runtime validity gates
======================================================

Single source of truth for ALL runtime correctness checks in the
forecasting + simulation pipeline. Audited as the canonical home for:

- **AstChecker** (`ast_checker.py`) — static code gate.
  Forbidden patterns: bare `except:`, broad `except Exception: pass`,
  `eval(...)`, `os.system(...)`. Used in CI + verify_before hooks.

- **check_epi_validity** (`epi_validity.py`) — SEIR/Metapop output gate.
  Bounds: Rt ∈ [0.3, 8], S+E+I+R+V+D = N conservation (tol 1e-3),
  seasonal phase coherence. Used by Stage 5 SEIR + ARIA scenario eval.

- **LeakagePerFoldHook** (`leakage_per_fold.py`) — train/test contamination gate.
  Detects target leakage and per-fold X distribution drift. Used by R9
  (per_model_optimize) Optuna trial loop + WF-CV (R4).

- **@verify_before / @verify_after** (`decorators.py`) — hook ABI.
  Decorator-layer (not subprocess) per §9.4 of RECOMMENDED_PIPELINE.md.
  Overhead ~1ms per hook. Persists every check to `verifier_audit` table.

- **RuntimeMonitor** (`runtime_monitor.py`) — long-run health.
  Memory/CPU drift detection during 6-24h training runs.

- **integration** (`integration.py`) — pipeline-side wiring helpers.

Public API:
    from simulation.verifier import verify_before, verify_after
    from simulation.verifier import AstChecker, FORBIDDEN_PATTERNS
    from simulation.verifier import check_epi_validity
    from simulation.verifier import LeakagePerFoldHook
    from simulation.verifier import audit_write

When adding a new runtime check, it belongs HERE — not in pipeline/ or
models/. The `verifier_audit` table is the audit log of every check
ever fired (run_id + caller + result + timestamp).
"""
from __future__ import annotations

from .decorators import (
    verify_before,
    verify_after,
    VerifierError,
    audit_write,
    get_current_run_id,
    set_current_run_id,
)
from .ast_checker import (
    AstChecker,
    FORBIDDEN_PATTERNS,
    scan_file,
    scan_source,
)
from .epi_validity import (
    check_epi_validity,
    EpiValidityError,
    EPI_RANGE,
)
from .leakage_per_fold import LeakagePerFoldHook
from .runtime_monitor import RuntimeMonitor
from .integration import (
    verify_pipeline,
    verified_run_expanding_cv,
    verified_run_pipeline,
    check_db_healthy,
    check_seed_fixed,
    check_predictions_valid,
)

__all__ = [
    "verify_before", "verify_after", "VerifierError", "audit_write",
    "get_current_run_id", "set_current_run_id",
    "AstChecker", "FORBIDDEN_PATTERNS", "scan_file", "scan_source",
    "check_epi_validity", "EpiValidityError", "EPI_RANGE",
    "LeakagePerFoldHook",
    "RuntimeMonitor",
    "verify_pipeline",
    "verified_run_expanding_cv", "verified_run_pipeline",
    "check_db_healthy", "check_seed_fixed", "check_predictions_valid",
]
