"""simulation.cli — extracted command handlers from __main__.py.

Phase C2 partial (2026-05-12): __main__.py was a 3,153-line monolith with
25 subcommand handlers inline. This package starts the split — each
module groups related cmd_X handlers by domain (db, maintenance, utility).

Re-imported by simulation/__main__.py so the dispatch table stays unchanged.
"""
