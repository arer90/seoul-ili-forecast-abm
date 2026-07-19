"""Centralized external API endpoints for overseas + multi-country collectors.

Sprint α Item 9 (2026-05-26): consolidate ~20 hardcoded URLs from individual
collector files into a single registry so URL rot (commit 917a191 — JP 2-digit
year URL fix) only needs to be patched in one place.

## Registry structure

``URLS`` is a flat dict keyed by short token. Each value is the full URL string
(or a base URL when path templating happens in the caller).

``URL_FALLBACK_CHAINS`` holds ordered lists for collectors that try multiple
endpoints on failure (e.g. Sentiweb FR has 3 fallback variants).

## Migration policy

Legacy module-level constants are KEPT as deprecation aliases that import from
this module — caller code (`requests.get(_WHO_BASE_URL, ...)`) keeps working
verbatim. Future PRs can migrate caller imports to `endpoints.URLS["who_flunet"]`
for explicitness, but no urgency.

## Scope (current — 2026-05-26)

Covers the 4 top-level overseas/multi-country collectors:
- group_i_overseas.py        (WHO FluNet/FluID, CDC ILINet, Sentiweb, ECDC,
                              Delphi, JIHS historical, Italy InfluNet)
- group_o_regional_ili.py    (NIID, RKI DE, Delphi, NHSN, NWSS, JIHS CSV/historical)
- group_w_overseas_weather.py (Open-Meteo)
- group_t_commuter_flows.py   (not URL-heavy — left in module)

Legacy collectors (`collectors/legacy/group_*.py`) are NOT migrated yet —
they're still dynamically loaded via `orchestrator.py:191`. Mass URL migration
there can happen separately when the legacy collectors are themselves audited.
"""
from __future__ import annotations


# ─────────────────────────────────────────────────────────────────────
# Single-URL endpoints
# ─────────────────────────────────────────────────────────────────────

URLS: dict[str, str] = {
    # WHO global surveillance (xMart public)
    "who_flunet":         "https://xmart-api-public.who.int/FLUMART/VIW_FNT",
    "who_fluid":          "https://xmart-api-public.who.int/FLUMART/VIW_FID_EPI",

    # US CDC public datasets
    "cdc_ilinet_kvib":    "https://data.cdc.gov/resource/kvib-3txy.json",
    "cdc_nhsn_hrd":       "https://data.cdc.gov/resource/vdzy-6i9v.json",
    "cdc_nwss_flu":       "https://data.cdc.gov/resource/atcp-73re.json",

    # Delphi (CMU) FluView / COVIDcast
    "delphi_fluview":     "https://api.delphi.cmu.edu/epidata/fluview/",
    "delphi_covidcast":   "https://api.delphi.cmu.edu/epidata/covidcast/",

    # Japan NIID + JIHS
    "niid_api":           "https://id.niid.go.jp/api/idwr_weekly_aggregate/",
    "jihs_csv_base":      "https://id-info.jihs.go.jp/en/surveillance/idwr/rapid",
    "jihs_hist_base":     "https://id-info.jihs.go.jp/niid/images/idwr/data-e",

    # France Sentiweb (primary CSV)
    "sentiweb_fr_csv":    "https://www.sentiweb.fr/datasets/incidence-PAY-3.csv",

    # EU ECDC ERVISS (GitHub-hosted weekly TSV)
    "ecdc_erviss":        ("https://raw.githubusercontent.com/EU-ECDC/"
                            "respiratory-viruses-weekly-bulletin-data/main/"
                            "data/erviss_data.csv"),

    # Italy InfluNet (GitHub mirror)
    "influnet_it":        ("https://raw.githubusercontent.com/fbranda/influnet/main/"
                            "season/season-incidence.csv"),

    # Germany RKI Bundesland TSV
    "de_rki_bundesland":  ("https://raw.githubusercontent.com/robert-koch-institut/"
                            "SurvStat-Influenza-Inzidenzdaten/main/Daten/"
                            "InzidenzdatenBundeslandWoche.tsv"),

    # Open-Meteo historical archive
    "open_meteo_archive": "https://archive-api.open-meteo.com/v1/archive",

    # Australia AIHW Influenza CSV (single fallback after NNDSS chain)
    "aihw_au_csv":        ("https://www.aihw.gov.au/getmedia/"
                            "f8b5c7d8-0b2e-4f82-8f9d-9d57d5e2e4c1/"
                            "aihw-phe-5.xlsx.aspx"),

    # Germany Bright Sky weather (DWD station data)
    "brightsky_de":       "https://api.brightsky.dev/weather",
}


# ─────────────────────────────────────────────────────────────────────
# Fallback chains — ordered, caller tries each until one succeeds
# ─────────────────────────────────────────────────────────────────────

URL_FALLBACK_CHAINS: dict[str, list[str]] = {
    # Sentiweb FR regional incidence — 3 endpoint variants
    # (used by group_i_overseas._fetch_sentiweb_fr_regional)
    "sentiweb_fr_regional": [
        "https://www.sentiweb.fr/api/v1/inc?format=json&geo=reg&indicator=incidence&periods=all",
        "https://www.sentiweb.fr/api/v1/indicators/incidence?format=json&geo=reg",
        "https://www.sentiweb.fr/datasets/incidence-PAY-3.csv",
    ],

    # Sentiweb FR national fallback (after regional fails)
    "sentiweb_fr_national": [
        "https://www.sentiweb.fr/api/v1/inc?format=json&geo=fr&indicator=incidence&periods=all",
    ],

    # Australia NNDSS Excel — 3 yearly fallbacks (2024 → 2023 → 2022 release)
    # (used by group_i_overseas._fetch_au_nndss)
    "au_nndss_yearly": [
        "https://www.health.gov.au/sites/default/files/documents/2024/05/nndss-data-collection-2023.xlsx",
        "https://www.health.gov.au/sites/default/files/documents/2023/05/nndss-data-collection-2022.xlsx",
        "https://www.health.gov.au/sites/default/files/documents/2022/06/nndss-data-collection-2021.xlsx",
    ],
}


# ─────────────────────────────────────────────────────────────────────
# Backward-compat module-level constants (used by current callers)
# ─────────────────────────────────────────────────────────────────────
# group_i_overseas.py
_WHO_BASE_URL              = URLS["who_flunet"]
_WHO_FLUID_URL             = URLS["who_fluid"]
_CDC_URL                   = URLS["cdc_ilinet_kvib"]
_SENTIWEB_FR_URL           = URLS["sentiweb_fr_csv"]
_ECDC_ERVISS_URL           = URLS["ecdc_erviss"]
_INFLUNET_IT_URL           = URLS["influnet_it"]
_DELPHI_FLUVIEW_URL        = URLS["delphi_fluview"]
_JIHS_HIST_BASE            = URLS["jihs_hist_base"]
_SENTIWEB_REGIONAL_ENDPOINTS = URL_FALLBACK_CHAINS["sentiweb_fr_regional"]
_SENTIWEB_NATIONAL_ENDPOINT  = URL_FALLBACK_CHAINS["sentiweb_fr_national"][0]

# group_o_regional_ili.py
_NIID_API_BASE             = URLS["niid_api"]
_DE_RKI_TSV_URL            = URLS["de_rki_bundesland"]
_DELPHI_COVIDCAST_URL      = URLS["delphi_covidcast"]
_NHSN_HRD_URL              = URLS["cdc_nhsn_hrd"]
_NWSS_FLU_URL              = URLS["cdc_nwss_flu"]
_JIHS_CSV_BASE             = URLS["jihs_csv_base"]

# group_w_overseas_weather.py
_OPEN_METEO_URL            = URLS["open_meteo_archive"]

# group_i_overseas.py NEW COLLECTOR 2/3/4 (Sprint α Item 9 follow-up 2026-05-26)
_OPENMETEO_ARCHIVE_URL     = URLS["open_meteo_archive"]
_NNDSS_URLS                = URL_FALLBACK_CHAINS["au_nndss_yearly"]
_AIHW_CSV_URL              = URLS["aihw_au_csv"]
_BRIGHTSKY_URL             = URLS["brightsky_de"]


__all__ = [
    "URLS", "URL_FALLBACK_CHAINS",
    # Backward-compat aliases
    "_WHO_BASE_URL", "_WHO_FLUID_URL", "_CDC_URL", "_SENTIWEB_FR_URL",
    "_ECDC_ERVISS_URL", "_INFLUNET_IT_URL", "_DELPHI_FLUVIEW_URL",
    "_JIHS_HIST_BASE", "_SENTIWEB_REGIONAL_ENDPOINTS", "_SENTIWEB_NATIONAL_ENDPOINT",
    "_NIID_API_BASE", "_DE_RKI_TSV_URL", "_DELPHI_COVIDCAST_URL",
    "_NHSN_HRD_URL", "_NWSS_FLU_URL", "_JIHS_CSV_BASE",
    "_OPEN_METEO_URL",
    "_OPENMETEO_ARCHIVE_URL", "_NNDSS_URLS", "_AIHW_CSV_URL", "_BRIGHTSKY_URL",
]
