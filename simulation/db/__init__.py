# simulation.db — REMOVED ( 2026-04-17)
# The shim previously re-exported simulation.database.* for backward
# compatibility with pre-callers. Grep across simulation/ and the
# project-wide yaml/sh/ps1 files confirms there are no remaining imports
# of `simulation.db.*`. This file is kept as a tombstone to force a loud
# ImportError if any caller is still using the old path.
raise ImportError(
    "simulation.db was removed in . "
    "Use `from simulation.database import ...` instead."
)
