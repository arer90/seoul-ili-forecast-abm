"""Per-architecture DL forecaster sub-package — namespace facade (NOT a deep-module split).

Sprint β Item 6 final decision (Codex audit 2026-05-26):
Unlike `_loaders/` (D-4 deep split done) and `overseas/` (D-4 deep split done),
`dl_models.py` is INTENTIONALLY KEPT as the single-file source of truth. This
sub-package is a documentation/navigation aid — NOT a real D-4 split.

Why dl_models.py CANNOT be safely split (3 hard ABI constraints):

1. **ChampionArtifact pickle ABI** — every existing `.pt` champion file
   embeds the literal string `simulation.models.dl_models.<ClassName>` as
   the unpickler's module path. Moving classes to per-arch modules makes
   their `__module__` attribute change → ALL existing champion artifacts
   fail to load. Breaks: `Pinf inference (구 phase14-inference)`, `server/mcp_epi.py:1164`
   model load, `utils/rehydrate.py:110`, and 7+ downstream consumers.

2. **package_c literal-path patches** — `simulation/scripts/package_c/apply.py`
   lines 195, 199, 258, 259 patch `simulation/models/dl_models.py` by
   literal filesystem path. Moving classes silently skips the patch (no
   error → A-3/A-4 autocast+torch.compile + B-C pinball/huber loss menu
   never apply in production runs).

3. **11+ helper-importer files** — `simulation/models/modern_ts/*.py`
   (nbeats, mamba, timesnet, itransformer, patchtst, tide, nhits) and
   `simulation/models/graph_models*.py` import the 4 internal helpers
   `_apply_weight_init`, `_make_sequences`, `_train_loop`,
   `_predict_torch` directly from `simulation.models.dl_models`. A real
   split would require updating all 11 callers in the same commit, AND
   keeping `dl_models.py` as a re-export shim (which only works if helper
   `__module__` is forced to dl_models — defeats the purpose).

## What this sub-package provides

Per-architecture re-export shims for cleaner caller paths (documentation +
future search-grep). The classes themselves still live in `dl_models.py`
and have `__module__ == "simulation.models.dl_models"`.

- ``dnn``     : DNNForecaster, OptunaDNNForecaster, TinyMLPForecaster
- ``tcn``     : TCNForecaster, OptunaTCNForecaster
- ``tabular`` : TabularDNNForecaster, TabularDNNLiteForecaster

If the 3 ABI constraints are ever removed (champion artifact format change
+ package_c retirement + helper migration to `dl_utils`), the real split
becomes safe. Until then: dl_models.py is the SoT, this sub-package is a
navigation aid only. This is honest D-4 application — "small interface,
rich implementation" requires that the implementation can ACTUALLY live
in the module exposing the interface, which the pickle ABI blocks here.
"""
from simulation.models.dl import dnn, tcn, tabular

__all__ = ["dnn", "tcn", "tabular"]
