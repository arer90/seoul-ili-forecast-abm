"""Per-source overseas collector sub-package (deep modules — Sprint β Item 4 full migration).

Sprint β Item 4 (2026-05-26, Codex analysis):
`group_i_overseas.py` was 2657 lines with 9 sources + 12 shared helpers inline.
Bodies are now split into 9 per-source deep modules + `_common.py` shared
infrastructure. `group_i_overseas.py` is a thin facade that re-exports for
back-compat + provides `run()` (orchestrator dispatch) + CLI.

## Module layout

- ``_common``  : 12 shared helpers (DB connect, retry, upsert, 4 table DDL)
- ``who``      : 2 fns — WHO FluNet (global virological) + WHO FluID (EU ILI)
- ``cdc``      : 2 fns — CDC ILINet/FluSurv-NET + Delphi FluView (US national)
- ``jihs``     : 3 fns — JP national aggregate + parser + JIHS historical
                ⚠ Cross-module data dep on Group O (`overseas_ili_regional`)
- ``ecdc``     : 1 fn  — ECDC ERVISS GitHub mirror (28 EU/EEA countries)
- ``influnet`` : 1 fn  — Italy InfluNet (2003-present)
- ``sentiweb`` : 3 fns — France national CSV + regional JSON + parser
- ``openmeteo``: 2 fns — Open-Meteo ERA5 (33 cities × 6 countries)
- ``nndss``    : 2 fns — AU NNDSS Excel parser + collector
                ⚠ Optional `openpyxl` dep (returns empty on missing)
- ``brightsky``: 2 fns — Germany DWD/Bright Sky (16 Bundesland cities)

## Usage

Per-source (preferred):
    from simulation.collectors.overseas.who import _fetch_who_flunet

Flat namespace (also works):
    from simulation.collectors.overseas import _fetch_who_flunet

Back-compat facade (kept working for legacy callers):
    from simulation.collectors.group_i_overseas import _fetch_who_flunet
"""
# Sub-module exposure
from simulation.collectors.overseas import (
    _common,
    who,
    cdc,
    jihs,
    ecdc,
    influnet,
    sentiweb,
    openmeteo,
    nndss,
    brightsky,
)

# Flat re-exports for `from simulation.collectors.overseas import X` style
from simulation.collectors.overseas._common import (
    _safe_connect_import, _resolve_db, _connect,
    _safe_float, _safe_int, _retry_get,
    _ensure_overseas_ili_table,
    _ensure_overseas_ili_regional_table,
    _ensure_overseas_weather_regional_table,
    _ensure_overseas_flu_state_table,
    _upsert_rows, _upsert_regional_ili_rows,
    _upsert_weather_rows, _upsert_flu_state_rows,
)
from simulation.collectors.overseas.who import (
    _fetch_who_flunet, _fetch_who_fluid_eu,
    WHO_TARGET_COUNTRIES,
)
from simulation.collectors.overseas.cdc import (
    _fetch_cdc_ilinet, _fetch_delphi_national_us,
)
from simulation.collectors.overseas.jihs import (
    _aggregate_jp_national_from_regional,
    _parse_jihs_national_total,
    _fetch_jihs_national_historical,
)
from simulation.collectors.overseas.ecdc import _fetch_ecdc_erviss_github
from simulation.collectors.overseas.influnet import _fetch_influnet_it
from simulation.collectors.overseas.sentiweb import (
    _fetch_sentiweb_fr, _parse_sentiweb_json, collect_sentiweb_france,
)
from simulation.collectors.overseas.openmeteo import (
    _fetch_openmeteo_one_year, collect_openmeteo_regional,
)
from simulation.collectors.overseas.nndss import (
    _parse_nndss_excel, collect_au_nndss,
)
from simulation.collectors.overseas.brightsky import (
    _fetch_brightsky_window, collect_brightsky_germany,
)


__all__ = [
    # Sub-modules
    "_common", "who", "cdc", "jihs", "ecdc", "influnet",
    "sentiweb", "openmeteo", "nndss", "brightsky",

    # _common (14 names)
    "_safe_connect_import", "_resolve_db", "_connect",
    "_safe_float", "_safe_int", "_retry_get",
    "_ensure_overseas_ili_table",
    "_ensure_overseas_ili_regional_table",
    "_ensure_overseas_weather_regional_table",
    "_ensure_overseas_flu_state_table",
    "_upsert_rows", "_upsert_regional_ili_rows",
    "_upsert_weather_rows", "_upsert_flu_state_rows",

    # source-specific
    "WHO_TARGET_COUNTRIES",
    "_fetch_who_flunet", "_fetch_who_fluid_eu",
    "_fetch_cdc_ilinet", "_fetch_delphi_national_us",
    "_aggregate_jp_national_from_regional",
    "_parse_jihs_national_total", "_fetch_jihs_national_historical",
    "_fetch_ecdc_erviss_github",
    "_fetch_influnet_it",
    "_fetch_sentiweb_fr", "_parse_sentiweb_json", "collect_sentiweb_france",
    "_fetch_openmeteo_one_year", "collect_openmeteo_regional",
    "_parse_nndss_excel", "collect_au_nndss",
    "_fetch_brightsky_window", "collect_brightsky_germany",
]
