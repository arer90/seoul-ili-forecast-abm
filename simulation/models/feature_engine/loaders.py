"""Backward-compat shim — real implementations live in `_loaders/<domain>.py`.

Sprint β Item 5 full migration (2026-05-26, Gemini analysis):
This file was 2064 lines with all 39 `_load_*` functions inline. They are
now split by domain into `_loaders/<domain>.py` (deep modules per ENGINEERING_PRINCIPLES.md
D-4). This shim re-exports all 39 names + the 4 `_CONG_MAP_*` congestion
helpers so the 7 existing importers (`builder.py` + 6 scripts +
`_archive/tools_archive/*`) keep working unchanged.

For new code, prefer importing from the per-domain module:
    from simulation.models.feature_engine._loaders.sentinel import _load_sentinel_ili

instead of:
    from simulation.models.feature_engine.loaders import _load_sentinel_ili

See `_loaders/__init__.py` for the full inventory.
"""
from __future__ import annotations

# Re-export every public symbol from _loaders package (flat namespace)
from simulation.models.feature_engine._loaders import (
    # _common helpers
    _CONG_MAP_PRIMARY,
    _CONG_MAP_ROAD,
    _CONG_MAP_UNICODE,
    _safe_congestion_score,
    # sentinel (6)
    _load_sentinel_ili,
    _load_sentinel_ari,
    _load_weekly_disease,
    _load_sentinel_sari,
    _load_sentinel_hfmd,
    _load_sentinel_enterovirus,
    # weather (2)
    _load_weather,
    _load_weather_forecast,
    # vaccination (2)
    _load_vaccination,
    _load_childhood_vaccination,
    # hira (3)
    _load_hira_gender_age,
    _load_hira_inpat_opat,
    _load_hira_region_seoul,
    # infrastructure (5)
    _load_school_info,
    _load_hospitals,
    _load_employment_workplace,
    _load_employment_monthly,
    _load_employment_residence,
    # mobility (8)
    _load_daily_population_district,
    _load_daily_subway,
    _load_daily_bus,
    _load_daily_population_hotspot,
    _load_daily_population_gu_hourly,
    _load_daily_population_dong_agg,
    _load_monthly_subway_hourly,
    _load_monthly_bus_hourly,
    # realtime (9)
    _load_rt_population,
    _load_rt_air_quality,
    _load_rt_road_traffic,
    _load_rt_subway_crowd,
    _load_rt_population_detail,
    _load_rt_population_forecast,
    _load_rt_sdot_env,
    _load_rt_spatial_aggregation,
    _load_rt_temporal_patterns,
    # external (4)
    _load_google_search_trends,
    _load_school_closure,
    _load_flunet_positivity,
    _load_korean_holiday,
)


__all__ = [
    # _common
    "_CONG_MAP_PRIMARY", "_CONG_MAP_ROAD", "_CONG_MAP_UNICODE",
    "_safe_congestion_score",
    # sentinel
    "_load_sentinel_ili", "_load_sentinel_ari", "_load_weekly_disease",
    "_load_sentinel_sari", "_load_sentinel_hfmd", "_load_sentinel_enterovirus",
    # weather
    "_load_weather", "_load_weather_forecast",
    # vaccination
    "_load_vaccination", "_load_childhood_vaccination",
    # hira
    "_load_hira_gender_age", "_load_hira_inpat_opat", "_load_hira_region_seoul",
    # infrastructure
    "_load_school_info", "_load_hospitals",
    "_load_employment_workplace", "_load_employment_monthly", "_load_employment_residence",
    # mobility
    "_load_daily_population_district", "_load_daily_subway", "_load_daily_bus",
    "_load_daily_population_hotspot", "_load_daily_population_gu_hourly",
    "_load_daily_population_dong_agg",
    "_load_monthly_subway_hourly", "_load_monthly_bus_hourly",
    # realtime
    "_load_rt_population", "_load_rt_air_quality", "_load_rt_road_traffic",
    "_load_rt_subway_crowd", "_load_rt_population_detail",
    "_load_rt_population_forecast", "_load_rt_sdot_env",
    "_load_rt_spatial_aggregation", "_load_rt_temporal_patterns",
    # external
    "_load_google_search_trends", "_load_school_closure",
    "_load_flunet_positivity", "_load_korean_holiday",
]
