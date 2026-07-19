"""Per-domain SQL loaders for the feature engineering pipeline.

Sprint β Item 5 full migration (2026-05-26, Gemini analysis):
`loaders.py` was 2064 lines, 39 `_load_*` functions. This package now holds
the REAL implementations split by domain. `loaders.py` (sibling module) is a
thin re-export shim for backward compatibility with the 7 existing importers
(`builder.py` + 6 scripts).

## Domain modules (8 + _common)

- ``_common``       — 4 shared congestion-text helpers (encoding-safe `\\uXXXX` fallback)
- ``sentinel``      — 6 fns: ILI / ARI / SARI / HFMD / Enterovirus / weekly_disease
- ``weather``       — 2 fns: historical (KMA stn 108) + forecast
- ``vaccination``   — 2 fns: adult flu coverage + childhood vaccine coverage
- ``hira``          — 3 fns: gender/age + inpat/opat + Seoul region
- ``infrastructure`` — 5 fns: schools + hospitals + employment (workplace/monthly/residence)
- ``mobility``      — 8 fns: daily population/subway/bus + hotspot + gu-hourly + dong agg + monthly hourly
- ``realtime``      — 9 fns: rt_population/air/road/subway/popdet/fcst/sdot + spatial/temporal agg
- ``external``      — 4 fns: Google Trends + school closure + WHO FluNet + Korean holidays

Total: 39 `_load_*` functions across 8 domain files (no cross-domain edges
inside loaders — every loader is a leaf; composition lives in `builder.py`).

## Usage

Per-domain (preferred for new callers):
    from simulation.models.feature_engine._loaders.sentinel import _load_sentinel_ili

Flat namespace (also works, package-level re-exports):
    from simulation.models.feature_engine._loaders import _load_sentinel_ili

Backward-compat (existing 7 importers — `loaders.py` shim re-exports):
    from simulation.models.feature_engine.loaders import _load_sentinel_ili
"""

# Flat package-level re-exports — all 39 _load_* + 4 congestion symbols
from ._common import (
    _CONG_MAP_PRIMARY,
    _CONG_MAP_ROAD,
    _CONG_MAP_UNICODE,
    _safe_congestion_score,
)
from .sentinel import (
    _load_sentinel_ili,
    _load_sentinel_ari,
    _load_weekly_disease,
    _load_sentinel_sari,
    _load_sentinel_hfmd,
    _load_sentinel_enterovirus,
)
from .weather import (
    _load_weather,
    _load_weather_forecast,
)
from .vaccination import (
    _load_vaccination,
    _load_childhood_vaccination,
)
from .hira import (
    _load_hira_gender_age,
    _load_hira_inpat_opat,
    _load_hira_region_seoul,
)
from .infrastructure import (
    _load_school_info,
    _load_hospitals,
    _load_employment_workplace,
    _load_employment_monthly,
    _load_employment_residence,
)
from .mobility import (
    _load_daily_population_district,
    _load_daily_subway,
    _load_daily_bus,
    _load_daily_population_hotspot,
    _load_daily_population_gu_hourly,
    _load_daily_population_dong_agg,
    _load_monthly_subway_hourly,
    _load_monthly_bus_hourly,
)
from .realtime import (
    _load_rt_population,
    _load_rt_air_quality,
    _load_rt_road_traffic,
    _load_rt_subway_crowd,
    _load_rt_population_detail,
    _load_rt_population_forecast,
    _load_rt_sdot_env,
    _load_rt_spatial_aggregation,
    _load_rt_temporal_patterns,
)
from .external import (
    _load_google_search_trends,
    _load_school_closure,
    _load_flunet_positivity,
    _load_korean_holiday,
)

# Sub-modules also exposed for namespaced access
from . import (
    _common,
    sentinel,
    weather,
    vaccination,
    hira,
    infrastructure,
    mobility,
    realtime,
    external,
)


__all__ = [
    # Sub-modules
    "_common", "sentinel", "weather", "vaccination", "hira",
    "infrastructure", "mobility", "realtime", "external",

    # _common helpers
    "_CONG_MAP_PRIMARY", "_CONG_MAP_ROAD", "_CONG_MAP_UNICODE",
    "_safe_congestion_score",

    # sentinel (6)
    "_load_sentinel_ili", "_load_sentinel_ari", "_load_weekly_disease",
    "_load_sentinel_sari", "_load_sentinel_hfmd", "_load_sentinel_enterovirus",

    # weather (2)
    "_load_weather", "_load_weather_forecast",

    # vaccination (2)
    "_load_vaccination", "_load_childhood_vaccination",

    # hira (3)
    "_load_hira_gender_age", "_load_hira_inpat_opat", "_load_hira_region_seoul",

    # infrastructure (5)
    "_load_school_info", "_load_hospitals",
    "_load_employment_workplace", "_load_employment_monthly", "_load_employment_residence",

    # mobility (8)
    "_load_daily_population_district", "_load_daily_subway", "_load_daily_bus",
    "_load_daily_population_hotspot", "_load_daily_population_gu_hourly",
    "_load_daily_population_dong_agg",
    "_load_monthly_subway_hourly", "_load_monthly_bus_hourly",

    # realtime (9)
    "_load_rt_population", "_load_rt_air_quality", "_load_rt_road_traffic",
    "_load_rt_subway_crowd", "_load_rt_population_detail",
    "_load_rt_population_forecast", "_load_rt_sdot_env",
    "_load_rt_spatial_aggregation", "_load_rt_temporal_patterns",

    # external (4)
    "_load_google_search_trends", "_load_school_closure",
    "_load_flunet_positivity", "_load_korean_holiday",
]
