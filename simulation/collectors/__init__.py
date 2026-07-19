"""
simulation.collectors — Data collection orchestrator
Wraps the active collectors (in simulation/collectors/legacy/) with a
unified interface. The `legacy/` directory label is historical — these
modules are the runtime collectors that orchestrator.py imports
(see DEFAULT_ORDER in orchestrator.py). New collectors can be added
there directly.

Usage:
    from simulation.collectors import run_collection, list_groups
    run_collection(groups=["E", "D", "B"])  # collect specific groups
    run_collection()                         # collect all groups
"""
from .orchestrator import run_collection, run_collection_parallel, list_groups, print_status

__all__ = ["run_collection", "run_collection_parallel", "list_groups", "print_status"]
