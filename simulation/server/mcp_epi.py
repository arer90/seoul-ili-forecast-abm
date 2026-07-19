"""
simulation.server.mcp_epi
=========================
ARIA — epidemiology MCP server (Stage 6a foundation).

Exposes **10 read-mostly tools** that the LLM consultation layer
(Claude / Gemini / OpenAI / Ollama) can call to ground its numeric
claims in real project data:

  1.  ``epi.query_db``            — read-only DuckDB SQL over ``epi_real_seoul.db``
  2.  ``epi.forecast``            — ensemble point + 95% PI for a gu
  3.  ``epi.model_compare``       — Diebold-Mariano tests by regime
  4.  ``epi.shap_features``       — Top-N SHAP for a (gu, week)
  5.  ``epi.rt_estimate``         — EpiEstim bayesian Rt sliding window
  6.  ``epi.lead_time_analysis``  — skill vs horizon for a model
  7.  ``epi.outbreak_detect``     — EARS-C1 / CUSUM flagging
  8.  ``epi.validity_check``      — run epi-validity gate on a claim
  9.  ``epi.literature_rag``      — vector-RAG over project PDFs
  10. ``epi.scenario_run``        — metapop SEIR-V-D run (Stage 5)

Of these, 6 are **fully wired** to existing subsystems
(``query_db``, ``rt_estimate``, ``outbreak_detect``, ``validity_check``,
``scenario_run`` + the schema-level ``list_tools``); the remaining 4
depend on artifacts produced by Stage 3 training (``forecast``,
``model_compare``, ``shap_features``, ``lead_time_analysis``) or Stage 6
RAG index (``literature_rag``). Those return graceful
``status="not_available"`` payloads with their contracts locked in, so
the UI and provider adapters can be built against them today and hooked
to real data once the artifacts exist.

Handlers are **pure Python**: ``EpiMCPServer().call_tool(name, args)``
works without any transport. The stdio JSON-RPC layer lives in
``mcp_stdio.py`` and dispatches here.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .sql_guard import SqlGuardError, validate_read_only


def _load_champion_names(log_path: Path) -> list[str]:
    """Read champion_log.json and return the list of current champion names."""
    try:
        j = json.loads(log_path.read_text())
        return [n for n, rec in j.items()
                if isinstance(rec, dict) and rec.get("current")]
    except Exception:
        return []


log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# Public types
# ══════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ToolSpec:
    """Declarative schema for one MCP tool. Mirrors MCP 1.0 shape."""
    name: str
    title: str
    description: str
    input_schema: dict
    wired: bool = True

    def to_mcp(self) -> dict:
        """Serialise to the MCP ``tools/list`` item shape."""
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "inputSchema": self.input_schema,
            "_meta": {"wired": self.wired},
        }


@dataclass
class CallResult:
    """Server-side tool result before JSON-RPC wrapping."""
    content: Any
    is_error: bool = False
    meta: dict = field(default_factory=dict)

    def to_mcp(self) -> dict:
        """Serialise to MCP ``tools/call`` response shape.

        MCP expects ``content`` to be a list of text/image/resource blocks.
        We always return a single ``text`` block with the JSON-serialised
        payload — trivial to consume on the client side.
        """
        text = (
            self.content
            if isinstance(self.content, str)
            else json.dumps(self.content, ensure_ascii=False, default=_ser)
        )
        return {
            "content": [{"type": "text", "text": text}],
            "isError": self.is_error,
            "_meta": self.meta,
        }


# ── JSON-safe NaN/None coercer for optional numeric fields ────────────
def _nanable(v: Any) -> Optional[float]:
    """Return ``None`` for NaN / missing, else ``float(v)``.

    Used in tool payloads so JSON serialises as ``null`` instead of
    ``NaN`` (which is not valid JSON and trips strict clients).
    """
    try:
        if v is None:
            return None
        f = float(v)
        if f != f:  # NaN check that works without importing math
            return None
        return f
    except (TypeError, ValueError):
        return None


# ── JSON serialiser tolerant of numpy scalars + pandas ────────────────
def _ser(o: Any) -> Any:
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (bytes, bytearray)):
        return o.decode("utf-8", errors="replace")
    try:
        # pandas Timestamp, datetime, etc.
        return o.isoformat()
    except AttributeError:
        return str(o)


# ══════════════════════════════════════════════════════════════════════
# Tool schemas  (11 tools, MCP 1.0 shape)
# ══════════════════════════════════════════════════════════════════════
_GU_NAMES_HINT = (
    "Spatial key. Currently the only supported value for weekly-incidence "
    "lookups (rt_estimate, outbreak_detect) is 'seoul_city' — the "
    "city-aggregate Seoul ILI series used to train the 53-model pipeline. "
    "Per-gu panels (e.g. '강남구') will land once Stage 3b runs the "
    "gu-level forecaster; until then they return an empty series with a "
    "graceful warning so callers can fall back to seoul_city."
)

TOOL_SPECS: tuple[ToolSpec, ...] = (
    ToolSpec(
        name="epi.query_db",
        title="Read-only SQL over epi_real_seoul.db",
        description=(
            "Execute a read-only SQL query against the project's SQLite DB "
            "through DuckDB (via ATTACH ... READ_ONLY). Only SELECT / WITH "
            "/ EXPLAIN / DESCRIBE / SHOW / PRAGMA are accepted; any DDL or "
            "DML is rejected. Use this to ground numeric claims in actual "
            "surveillance data."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "Read-only SQL statement."},
                "limit": {
                    "type": "integer",
                    "description": "Hard row cap applied AFTER the query runs. "
                                   "Default 500 to keep responses small.",
                    "default": 500,
                    "minimum": 1,
                    "maximum": 10000,
                },
            },
            "required": ["sql"],
            "additionalProperties": False,
        },
        wired=True,
    ),
    ToolSpec(
        name="epi.forecast",
        title="Ensemble forecast with 95% PI",
        description=(
            "Return the ensemble point forecast + 95% prediction interval "
            "for a given gu across the next N weeks. Uses the tournament "
            "ensemble frozen in the latest Stage 3 run. (Returns "
            "status='not_available' until Stage 3 artifacts exist.)"
        ),
        input_schema={
            "type": "object",
            "properties": {
                "gu": {"type": "string", "description": _GU_NAMES_HINT},
                "horizon": {
                    "type": "integer", "minimum": 1, "maximum": 12,
                    "default": 4,
                    "description": "Forecast horizon in weeks.",
                },
                "model_id": {
                    "type": "string", "default": "ensemble",
                    "description": (
                        "'ensemble' or a specific model name "
                        "(e.g. 'XGBoost', 'PatchTST', 'SEIR-V-D')."
                    ),
                },
            },
            "required": ["gu"],
            "additionalProperties": False,
        },
        wired=True,
    ),
    ToolSpec(
        name="epi.model_compare",
        title="Diebold-Mariano by regime",
        description=(
            "Compare multiple models' forecast loss over a target week or "
            "regime (pre-COVID / during / post / global). Returns DM test "
            "p-values and loss diffs. Requires Stage 4 leaderboard."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "week": {"type": "string",
                         "description": "ISO date 'YYYY-MM-DD' (Monday) or regime tag."},
                "models": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                    "description": "Model names to compare pairwise.",
                },
                "metric": {
                    "type": "string",
                    "enum": ["mae", "rmse", "mape", "pinball"],
                    "default": "mae",
                },
            },
            "required": ["models"],
            "additionalProperties": False,
        },
        wired=True,
    ),
    ToolSpec(
        name="epi.shap_features",
        title="Top-N SHAP feature importance",
        description=(
            "Return the Top-N SHAP features driving the prediction at "
            "(gu, week), plus the week-over-week SHAP delta. Requires "
            "R11 (shap) SHAP artifacts (Stage 3)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "gu": {"type": "string", "description": _GU_NAMES_HINT},
                "week": {"type": "string",
                         "description": "ISO date 'YYYY-MM-DD' (Monday)."},
                "top_n": {"type": "integer", "minimum": 1, "maximum": 50,
                          "default": 10},
                "model": {"type": "string", "default": "XGBoost"},
            },
            "required": ["gu", "week"],
            "additionalProperties": False,
        },
        wired=True,
    ),
    ToolSpec(
        name="epi.rt_estimate",
        title="EpiEstim bayesian Rt",
        description=(
            "Estimate R_t using EpiEstim (Cori et al. 2013) bayesian "
            "sliding-window posterior over the Seoul city-aggregate ILI "
            "series. Returns mean + 95% CI for each time point inside "
            "the estimation range. **Only ``gu='seoul_city'`` is currently "
            "supported** (aliases: 'seoul', '서울', '서울특별시'); per-gu "
            "panels land in Stage 3b."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "gu": {"type": "string", "description": _GU_NAMES_HINT},
                "window_weeks": {
                    "type": "integer", "minimum": 2, "maximum": 14, "default": 7,
                    "description": "Sliding window width (weeks).",
                },
                "serial_interval_mean": {
                    "type": "number", "default": 2.6,
                    "description": "Flu serial interval mean (days).",
                },
                "serial_interval_sd": {
                    "type": "number", "default": 1.5,
                    "description": "Flu serial interval standard deviation (days).",
                },
                "lookback_weeks": {
                    "type": "integer", "minimum": 10, "maximum": 520, "default": 104,
                    "description": (
                        "How many past weeks to load. Source: "
                        "epi.weekly_disease (Seoul flu totals) with "
                        "predictions_*.csv y_true as fallback."
                    ),
                },
            },
            "required": ["gu"],
            "additionalProperties": False,
        },
        wired=True,
    ),
    ToolSpec(
        name="epi.lead_time_analysis",
        title="Forecast skill vs horizon (operational proxy)",
        description=(
            "Lead-time analysis for outbreak-alerting. Returns onset + "
            "peak lead times (weeks AHEAD of observed truth) drawn from "
            "the post-E ``peak_onset.csv`` artifact — each model has one "
            "post-E evaluation point, so this is a single-horizon "
            "operational proxy rather than the full R²/MAPE-vs-horizon "
            "curve promised in the Stage 4 schema. When ``model`` is "
            "given, returns that model's lead times + rank among all "
            "post-E models. Without ``model``, returns the top-10 "
            "earliest-alerters. The richer per-horizon curve will "
            "replace this payload once ``stage4_lead_time.json`` is "
            "produced by the WF-CV rerun."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "model": {
                    "type": "string",
                    "description": (
                        "Model name as it appears in peak_onset.csv "
                        "(e.g. 'Ensemble-NNLS', 'NegBinGLM', 'TabularDNN-Lite'). "
                        "Optional — omit to get the top-10 ranking."
                    ),
                },
                "max_horizon": {
                    "type": "integer", "minimum": 1, "maximum": 12,
                    "default": 4,
                    "description": (
                        "Ignored in the proxy (single-horizon); "
                        "honoured once Stage 4 artifacts land."
                    ),
                },
                "top_k": {
                    "type": "integer", "minimum": 1, "maximum": 66,
                    "default": 10,
                    "description": (
                        "Number of top-ranked models to return when "
                        "``model`` is omitted."
                    ),
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        wired=True,
    ),
    ToolSpec(
        name="epi.outbreak_detect",
        title="Outbreak flag (EARS-C1 / CUSUM)",
        description=(
            "Flag outbreak weeks using EARS-C1 (CDC early aberration "
            "baseline z-score over 7 prior weeks) or CUSUM. Returns "
            "per-week boolean flags + threshold. **Only "
            "``gu='seoul_city'`` is currently supported** (aliases: "
            "'seoul', '서울', '서울특별시'); per-gu panels land in Stage 3b."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "gu": {"type": "string", "description": _GU_NAMES_HINT},
                "method": {"type": "string",
                           "enum": ["EARS-C1", "CUSUM"], "default": "EARS-C1"},
                "lookback_weeks": {"type": "integer", "minimum": 14, "maximum": 520,
                                    "default": 104},
                "z_threshold": {"type": "number", "default": 2.0,
                                "description": "EARS-C1 z cutoff; ignored for CUSUM."},
            },
            "required": ["gu"],
            "additionalProperties": False,
        },
        wired=True,
    ),
    ToolSpec(
        name="epi.validity_check",
        title="Epi-validity gate on a claim",
        description=(
            "Run the epi-validity gate (Rt bounds, compartment "
            "conservation, seasonal peak, SEIR param ranges) on a claim "
            "JSON. Returns pass/warn/fail + per-check detail so the "
            "LLM can attach a badge to the rendered answer."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "params": {
                    "type": "object",
                    "description": (
                        "Optional SEIR params, e.g. {R0, gamma, sigma, VE, ifr}."
                    ),
                },
                "predictions": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Optional forecast array (ILI rate).",
                },
            },
            "additionalProperties": False,
        },
        wired=True,
    ),
    ToolSpec(
        name="epi.literature_rag",
        title="Literature / KDCA guideline lookup",
        description=(
            "Retrieve the most relevant citations for a query over the "
            "project's bibliography. When a vector index exists at "
            "``<artifacts_dir>/rag_index/``, full RAG is used; otherwise "
            "falls back to keyword-overlap matching against a curated "
            "static catalogue (~20 entries: KDCA guidelines, Kermack-"
            "McKendrick, Hansen MCS, Murphy Brier decomposition, "
            "Cori EpiEstim, Gneiting WIS, FluSight, etc.). Each match "
            "includes a one-line ``relevance`` note explaining why it "
            "matters to MPH . The ``status`` field in the payload "
            "tells you which path was taken."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Free-text question or topic. Split into "
                        "keyword tokens for the static fallback; "
                        "passed whole to the vector index when "
                        "present."
                    ),
                },
                "k": {
                    "type": "integer", "minimum": 1, "maximum": 20,
                    "default": 5,
                    "description": "Number of citations to return.",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
        wired=True,
    ),
    ToolSpec(
        name="epi.scenario_run",
        title="Metapop SEIR-V-D scenario",
        description=(
            "Run one of the Stage 5 scenarios (baseline, npi_lockdown, "
            "vaccination_campaign, antiviral_prophylaxis, combined_response, "
            "sensitivity_strain_mismatch) and return trajectories + "
            "peak/final D / epi-validity status. "
            "Set include_international=true to attach international ILI "
            "reference data (US/JP/EU) for SEIRVD curve comparison."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "scenario": {
                    "type": "string",
                    "description": "Scenario name (see epi.scenario_list).",
                },
                "days": {"type": "integer", "minimum": 30, "maximum": 540,
                          "default": 200},
                "seed_district": {"type": "string", "default": "강남구"},
                "seed_infected": {"type": "number", "default": 10.0, "minimum": 0},
                "use_db": {"type": "boolean", "default": True,
                           "description": "Load populations + commuter matrix from DB."},
                "include_international": {
                    "type": "boolean", "default": False,
                    "description": (
                        "Attach current-season ILI positivity for US/JP/DE/FR/NL/SE "
                        "as comparison reference alongside the SEIRVD simulation."
                    ),
                },
            },
            "required": ["scenario"],
            "additionalProperties": False,
        },
        wired=True,
    ),
    ToolSpec(
        name="epi.international_compare",
        title="International ILI comparison",
        description=(
            "Compare Korea's ILI positivity against US (CDC ILINet), Japan (JIHS), "
            "and Europe (DE/FR/GB/NL/SE via WHO FluNet). "
            "Returns: Pearson correlation + lead-lag (weeks), season peak week diff, "
            "normalised ILI series for plotting, and an LLM-ready context summary. "
            "Use this when the user asks 'how does Korea compare to other countries?' "
            "or when SEIRVD simulation needs international epidemic context."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "countries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "default": ["US", "JP", "DE", "NL", "SE"],
                    "description": "Country codes from overseas_ili (US/JP/KR/DE/FR/GB/NL/SE).",
                },
                "start_year": {
                    "type": "integer", "default": 2019,
                    "description": "Analysis start year (post-2015 recommended).",
                },
                "season": {
                    "type": "string",
                    "description": "Specific flu season e.g. '2023/2024'. If omitted, uses all data.",
                },
                "metric": {
                    "type": "string",
                    "enum": ["positivity", "ili_rate"],
                    "default": "positivity",
                    "description": "positivity = INF/spec (WHO FluNet), ili_rate = CLI% (CDC ILINet for US).",
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        wired=True,
    ),
    ToolSpec(
        name="epi.coupled_forward",
        title="Forecast-coupled behavioural ABM forward",
        description=(
            "Run the COUPLED forward: the champion FusedEpi forecast anchors a hybrid "
            "behavioural agent model over Seoul's 25 districts and an EnKF assimilates "
            "the forecast nowcast week by week (leak-free). Returns the coupled "
            "trajectory (champion / ABM-alone / ABM+EnKF), a mechanistic uncertainty "
            "band from the assimilated ensemble, and per-district commuter import "
            "fraction + target load + R_eff from the commuter-coupled next-generation "
            "matrix. This exposes the COUPLED agent model with district resolution — "
            "NOT the standalone deterministic metapop (epi.scenario_run)."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "n_agents": {"type": "integer", "minimum": 2000, "maximum": 30000,
                             "default": 6000},
                "n_seeds": {"type": "integer", "minimum": 2, "maximum": 8, "default": 4},
            },
            "required": [],
            "additionalProperties": False,
        },
        wired=True,
    ),
)

TOOL_BY_NAME: dict[str, ToolSpec] = {t.name: t for t in TOOL_SPECS}


# ══════════════════════════════════════════════════════════════════════
# Server
# ══════════════════════════════════════════════════════════════════════
class EpiMCPServer:
    """Pure-Python epi MCP server.

    Usage::

        srv = EpiMCPServer()
        tools = srv.list_tools()
        result = srv.call_tool("epi.rt_estimate", {"gu": "강남구"})
        print(result.to_mcp())

    The stdio JSON-RPC layer in ``mcp_stdio.py`` composes ``list_tools``
    and ``call_tool`` into protocol responses.
    """

    SERVER_NAME = "epi-mcp"
    SERVER_VERSION = "0.1.0"
    #: D2 (M7): an artifact is flagged STALE when the source DB received data
    #: more than this many days after the artifact was generated.
    STALE_DAYS = 1.0

    def __init__(self, *, artifacts_dir: Optional[Path] = None):
        self.artifacts_dir = (
            Path(artifacts_dir) if artifacts_dir else _default_artifacts_dir()
        )
        # Handler registry is built lazily so import-time failures in
        # heavy deps (torch, duckdb) don't break schema listing.
        self._handlers: dict[str, Callable[[dict], CallResult]] = {
            "epi.query_db":              self._h_query_db,
            "epi.forecast":              self._h_forecast,
            "epi.model_compare":         self._h_model_compare,
            "epi.shap_features":         self._h_shap_features,
            "epi.rt_estimate":           self._h_rt_estimate,
            "epi.lead_time_analysis":    self._h_lead_time,
            "epi.outbreak_detect":       self._h_outbreak_detect,
            "epi.validity_check":        self._h_validity_check,
            "epi.literature_rag":        self._h_literature_rag,
            "epi.scenario_run":          self._h_scenario_run,
            "epi.coupled_forward":       self._h_coupled_forward,
            "epi.international_compare": self._h_international_compare,
        }
        # Per-server cache for the heavy enriched-feature build. Repeated
        # forecast / model_compare / shap_features calls otherwise rebuild the
        # full feature matrix from the DB every time — the measured latency
        # weakness. Built once, reused; drop with ``reset_feature_cache()``
        # after a DB refresh.
        self._enriched_cache: Optional[tuple] = None

    def _get_enriched_features(self) -> tuple:
        """Cached ``build_enriched_features`` result ``(feat_df, meta)``.

        Built once per server instance and reused across forecast /
        model_compare / shap_features calls. The returned ``feat_df`` is used
        read-only by the handlers (``select`` / column access), so sharing the
        cached object is safe. Call ``reset_feature_cache()`` after a DB update.
        """
        if self._enriched_cache is None:
            from simulation.models.feature_engine import build_enriched_features
            from simulation.database.config import DB_PATH
            self._enriched_cache = build_enriched_features(db_path=str(DB_PATH))
        return self._enriched_cache

    def reset_feature_cache(self) -> None:
        """Drop the cached enriched features (call after a DB refresh)."""
        self._enriched_cache = None

    # ── Public MCP surface ────────────────────────────────────────────
    def _probe_wired(self) -> dict:
        """Live wired-status per tool, by probing artifact presence.

        The 5 DB/code tools (query_db, rt_estimate, outbreak_detect,
        validity_check, scenario_run) and the static-citation literature_rag
        always return; the 4 Stage-3/4 artifact tools are callable *now* only if
        their artifact exists. The TOOL_SPECS hardcode ``wired=True``, which
        misleads an MCP/LLM planner into calling artifact-gated tools that will
        degrade to ``not_available`` — so ``list_tools`` reports the live flag.
        """
        ad = self.artifacts_dir
        models = Path("models")

        def _exists(*rel: str) -> bool:
            return any((base / r).exists() for base in (ad, models) for r in rel)

        gated = {
            "epi.forecast": _exists("champion_log.json",
                                    "real_eval/predictions.csv", "stage3_forecasts.json"),
            "epi.model_compare": _exists("per_model_eval/per_model_metrics.csv",
                                         "stage4_dm_results.json"),
            "epi.shap_features": _exists("shap/_summary.json", "stage3_shap/summary.json"),
            "epi.lead_time_analysis": _exists("per_model_eval/per_model_metrics.csv",
                                              "stage4_lead_time.json", "peak_onset.csv"),
        }
        always = {
            "epi.query_db", "epi.rt_estimate", "epi.outbreak_detect",
            "epi.validity_check", "epi.scenario_run", "epi.literature_rag",
            "epi.international_compare",
        }
        return {
            t.name: (True if t.name in always else gated.get(t.name, t.wired))
            for t in TOOL_SPECS
        }

    def list_tools(self) -> list[dict]:
        """Return all tool schemas (MCP ``tools/list`` response shape).

        The ``_meta.wired`` flag reflects **live artifact presence** rather than
        the hardcoded default, so a client can tell which tools are callable now.
        """
        wired_now = self._probe_wired()
        out = []
        for t in TOOL_SPECS:
            d = t.to_mcp()
            d.setdefault("_meta", {})["wired"] = bool(wired_now.get(t.name, t.wired))
            out.append(d)
        return out

    def call_tool(self, name: str, arguments: Optional[dict] = None) -> CallResult:
        """Dispatch one tool call. Never raises; errors land in ``CallResult``."""
        args = arguments or {}
        if name not in self._handlers:
            return CallResult(
                content={"error": f"unknown tool: {name!r}",
                         "known_tools": sorted(self._handlers)},
                is_error=True,
            )
        t0 = time.perf_counter()
        try:
            result = self._handlers[name](args)
        except SqlGuardError as e:
            return CallResult(
                content={"error": "sql_guard", "message": str(e)},
                is_error=True,
            )
        except Exception as e:
            log.exception("tool %s failed", name)
            return CallResult(
                content={"error": type(e).__name__, "message": str(e)},
                is_error=True,
            )
        # Attach timing for observability
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        result.meta.setdefault("elapsed_ms", elapsed_ms)
        result.meta.setdefault("tool", name)
        # Sprint 2026-05-06 top3 ROI #1 (Codex optimization #1): emit
        # metrics via simulation.server.metrics — structured stderr log
        # + in-memory counter for p95/p99. Picked up by Vercel/Datadog/Loki
        # scraping. Metrics must NEVER break the tool call.
        try:
            from simulation.server.metrics import get_metrics
            get_metrics().record_call(
                name, elapsed_ms, is_error=bool(result.is_error),
            )
        except Exception:
            pass
        # D8 (M7): immutable audit record (turn→tool→args→status) — the
        # accountability trail a clinical-advisory system is expected to keep.
        self._audit_call(name, args, result, elapsed_ms)
        # D1 (M7): uniform provenance envelope — every advisory traceable to a
        # versioned artifact + data vintage (auditability).
        result = self._attach_provenance(result)
        return result

    def _audit_call(self, name: str, args: dict, result: "CallResult",
                    elapsed_ms: int) -> None:
        """Append one ``mcp_audit.jsonl`` record per tool call (D8/M7).

        Record = ``{ts, tool, args_hash, status, elapsed_ms, request_id}``. Lets
        any advisory be reconstructed after the fact (turn → tool → args → result).
        Best-effort — auditing must never break the call.
        """
        try:
            import hashlib
            from datetime import datetime
            rec = {
                "ts": datetime.now().isoformat(timespec="seconds"),
                "tool": name,
                "args_hash": hashlib.sha256(
                    json.dumps(args, sort_keys=True, default=str).encode("utf-8")
                ).hexdigest()[:16],
                "status": "error" if result.is_error else "ok",
                "elapsed_ms": int(elapsed_ms),
                "request_id": result.meta.get("request_id") or result.meta.get("tool"),
            }
            with (self.artifacts_dir / "mcp_audit.jsonl").open(
                    "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    # ── D1 (M7): provenance envelope ──────────────────────────────────────
    def _attach_provenance(self, result: "CallResult") -> "CallResult":
        """Inject ``content['provenance']`` so every result is traceable.

        Adds ``{server_version, db_vintage_ts}`` to every successful dict result,
        plus ``{artifact_path, artifact_mtime_iso, artifact_sha256,
        config_sha256?}`` when the handler declared a file ``source`` under
        ``artifacts_dir``. Error results / non-dict content / already-stamped
        results are left untouched. Never raises (provenance must not break a call).
        """
        try:
            content = result.content
            if (not isinstance(content, dict)
                    or content.get("error")
                    or "provenance" in content):
                return result
            prov = {
                "server_version": self.SERVER_VERSION,
                "db_vintage_ts": self._db_vintage(),
            }
            src = content.get("source")
            if isinstance(src, str) and src:
                p = self.artifacts_dir / src
                if p.exists() and p.is_file():
                    prov.update(self._artifact_provenance(p))
            content["provenance"] = prov
            # D2 (M7): staleness guard — flag when the artifact predates the DB's
            # latest data so a stale snapshot is never served *silently*. Keep
            # status:'live' but surface freshness so Hermes/UI can caveat it.
            if "artifact_mtime_iso" in prov:
                fr = self._freshness(prov)
                if fr:
                    content["freshness"] = fr["status"]
                    if fr["status"] != "LIVE":
                        content.setdefault("freshness_detail", fr)
        except Exception:   # provenance is advisory — never break the call
            pass
        return result

    def _db_data_time(self):
        """Latest data-collection time in the DB as a datetime (cached, lock-free).

        ``MAX(weekly_disease.collected_at)`` parsed to a ``datetime``, or None.
        Same read-only / busy-timeout discipline as :meth:`_db_vintage` (never
        blocks behind a writer). Never raises.
        """
        cache = self.__dict__.setdefault("_prov_cache", {})
        if "db_data_time" in cache:
            return cache["db_data_time"]
        from datetime import datetime
        dt = None
        try:
            from simulation.database import read_only_connect
            con = read_only_connect()  # lock-free; never blocks a training writer
            try:
                row = con.execute("SELECT MAX(collected_at) FROM weekly_disease").fetchone()
                if row and row[0]:
                    dt = datetime.fromisoformat(str(row[0]).strip().replace(" ", "T")[:19])
            except Exception:
                dt = None
            finally:
                con.close()
        except Exception:
            dt = None
        cache["db_data_time"] = dt
        return dt

    def _freshness(self, prov: dict) -> Optional[dict]:
        """Verdict comparing an artifact's mtime to the DB's latest data time (D2).

        Args:
            prov: a provenance dict that may carry ``artifact_mtime_iso``.

        Returns:
            ``{status: 'LIVE'|'STALE'|'UNKNOWN', db_ahead_days?, reason?}`` or
            None when there is no artifact to judge. STALE = the DB received data
            more than ``STALE_DAYS`` after the artifact was generated, so the
            advisory may not reflect the newest surveillance week. Never raises.
        """
        mtime_iso = prov.get("artifact_mtime_iso")
        if not mtime_iso:
            return None
        from datetime import datetime
        try:
            art = datetime.fromisoformat(mtime_iso)
        except Exception:
            return None
        db = self._db_data_time()
        if db is None:
            return {"status": "UNKNOWN", "reason": "no DB data-time available"}
        ahead_days = (db - art).total_seconds() / 86400.0
        if ahead_days > self.STALE_DAYS:
            return {"status": "STALE", "db_ahead_days": round(ahead_days, 2),
                    "reason": (f"source DB has data {ahead_days:.1f} days newer than "
                               f"this artifact — advisory may not reflect the latest week")}
        return {"status": "LIVE", "db_ahead_days": round(ahead_days, 2)}

    def _db_vintage(self) -> Optional[str]:
        """Latest data vintage in the source DB (cached per process).

        Returns ``weekly_disease.MAX(vintage_ts)`` (else ``MAX(collected_at)``),
        or None. Opens a **read-only** connection (``mode=ro`` + 2 s busy_timeout)
        so a concurrent collect/train run holding the write lock can NEVER hang
        this read — provenance must never block a tool call. (Before: a plain
        ``safe_connect()`` read-write open blocked ~60 s behind a training run.)
        Never raises.
        """
        cache = self.__dict__.setdefault("_prov_cache", {})
        if "db_vintage" in cache:
            return cache["db_vintage"]
        val: Optional[str] = None
        try:
            from simulation.database import read_only_connect
            con = read_only_connect()  # lock-free; never blocks a training writer
            try:
                for q in ("SELECT MAX(vintage_ts) FROM weekly_disease",
                          "SELECT MAX(collected_at) FROM weekly_disease"):
                    try:
                        row = con.execute(q).fetchone()
                        if row and row[0] is not None:
                            val = str(row[0]); break
                    except Exception:
                        continue
            finally:
                con.close()
        except Exception:
            val = None
        cache["db_vintage"] = val
        return val

    def _config_sha256(self) -> Optional[str]:
        """Run config hash from ``run_manifest.json`` (D7), if present. Cached."""
        cache = self.__dict__.setdefault("_prov_cache", {})
        if "config_sha256" in cache:
            return cache["config_sha256"]
        val: Optional[str] = None
        try:
            mp = self.artifacts_dir / "run_manifest.json"
            if mp.exists():
                val = json.loads(mp.read_text(encoding="utf-8")).get("config_sha256")
        except Exception:
            val = None
        cache["config_sha256"] = val
        return val

    def _artifact_provenance(self, path: "Path") -> dict:
        """``{artifact_path, artifact_mtime_iso, artifact_sha256, config_sha256?}``.

        sha256 is cached by (path, mtime). ``artifact_path`` is relative to
        ``artifacts_dir`` when possible. Never raises.
        """
        import hashlib
        from datetime import datetime

        out: dict = {}
        try:
            st = path.stat()
            try:
                out["artifact_path"] = str(path.relative_to(self.artifacts_dir))
            except ValueError:
                out["artifact_path"] = str(path)
            out["artifact_mtime_iso"] = datetime.fromtimestamp(
                st.st_mtime).isoformat(timespec="seconds")
            sha_cache = self.__dict__.setdefault("_prov_cache", {}).setdefault("sha", {})
            key = (str(path), st.st_mtime)
            sha = sha_cache.get(key)
            if sha is None:
                h = hashlib.sha256()
                with path.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(65536), b""):
                        h.update(chunk)
                sha = h.hexdigest()
                sha_cache[key] = sha
            out["artifact_sha256"] = sha
            cfg = self._config_sha256()
            if cfg:
                out["config_sha256"] = cfg
        except Exception:
            pass
        return out

    # ══════════════════════════════════════════════════════════════════
    # Fully wired handlers
    # ══════════════════════════════════════════════════════════════════
    def _h_query_db(self, args: dict) -> CallResult:
        sql = str(args.get("sql", "")).strip()
        # D10 (M7): hard row cap (DuckDB has no statement_timeout; the LLM composes
        # the SQL, so bound the result rows regardless of the requested limit).
        # Safety boundary = read-only (validate_read_only rejects DDL/DML) +
        # READ_ONLY DuckDB attach + this output cap; tables exposed are aggregate.
        limit = max(1, min(int(args.get("limit", 500)), 10000))
        validate_read_only(sql).raise_if_bad()

        # Defence-in-depth: open DuckDB with READ_ONLY attach.
        from simulation.database.analytics import duckdb_conn

        with duckdb_conn(read_only=True) as con:
            cur = con.execute(sql)
            cols = [d[0] for d in (cur.description or [])]
            rows = cur.fetchmany(limit)

        data = [dict(zip(cols, r)) for r in rows]
        return CallResult(
            content={
                "status": "ok",
                "columns": cols,
                "rows": data,
                "row_count": len(data),
                "truncated": len(data) == limit,
            },
        )

    def _h_rt_estimate(self, args: dict) -> CallResult:
        gu = str(args["gu"])
        window_weeks = int(args.get("window_weeks", 7))
        si_mean = float(args.get("serial_interval_mean", 2.6))
        si_sd = float(args.get("serial_interval_sd", 1.5))
        lookback = int(args.get("lookback_weeks", 104))

        series, weeks = _load_weekly_incidence(gu, lookback)
        if series.size < window_weeks + 2:
            if not _is_seoul_city_key(gu):
                msg = "gu-panel data not yet available; call with gu='seoul_city'"
            else:
                msg = f"need ≥ {window_weeks + 2} weeks, got {series.size}"
            return CallResult(
                content={
                    "status": "insufficient_data",
                    "message": msg,
                    "gu": gu,
                },
            )

        from simulation.models.rt_estimator import RtEstimator

        est = RtEstimator(window_size=window_weeks)
        # rt_estimator operates on "per-day-equivalent" cadence but the
        # serial_interval params are in days, so feed weekly series with
        # SI expressed in the same "unit" (weeks). We convert SI from days
        # to the incident-cadence by dividing by 7. This mirrors what
        # compute_rt_features does in the ML feature engine.
        df = est.estimate(
            ili_series=series,
            serial_interval_mean=si_mean / 7.0,
            serial_interval_sd=si_sd / 7.0,
        )
        out = df.to_dict(orient="records")
        # Align the t index back to calendar weeks.
        for rec in out:
            idx = int(rec["t"])
            if 0 <= idx < len(weeks):
                rec["week_start"] = weeks[idx]
        # Surface (do NOT silently pass) any epidemiologically implausible Rt. The
        # shared RtEstimator has a known off-by-one that can inflate Rt far above a
        # sane reproduction-number band; this flag lets the ARIA layer / any MCP
        # client see the anomaly. NOTE: this only ANNOTATES — it does not change the
        # Rt values (the estimator feeds forecasting features and is not touched here).
        _LO, _HI = 0.3, 8.0
        rt_vals = [rec.get("Rt_mean") for rec in out]
        rt_vals = [v for v in rt_vals if isinstance(v, (int, float)) and v == v]
        n_out = sum(1 for v in rt_vals if not (_LO <= v <= _HI))
        validity = {"rt_plausible_band": [_LO, _HI], "n_evaluated": len(rt_vals),
                    "n_out_of_band": n_out, "all_in_band": n_out == 0}
        if n_out:
            validity["warning"] = (
                f"{n_out}/{len(rt_vals)} Rt estimates fall outside the plausible "
                f"[{_LO}, {_HI}] reproduction-number band — the shared RtEstimator "
                "is likely mis-scaled (off-by-one in rt_estimator.estimate); treat "
                "these Rt values as unreliable pending the coordinated fix.")
        return CallResult(
            content={
                "status": "ok",
                "gu": gu,
                "window_weeks": window_weeks,
                "n_points": len(out),
                "validity": validity,
                "series": out,
            }
        )

    def _h_outbreak_detect(self, args: dict) -> CallResult:
        gu = str(args["gu"])
        method = str(args.get("method", "EARS-C1"))
        lookback = int(args.get("lookback_weeks", 104))
        z_thr = float(args.get("z_threshold", 2.0))

        series, weeks = _load_weekly_incidence(gu, lookback)
        if series.size == 0 and not _is_seoul_city_key(gu):
            return CallResult(
                content={
                    "status": "insufficient_data",
                    "message": "gu-panel data not yet available; call with gu='seoul_city'",
                    "gu": gu,
                    "method": method,
                    "n_weeks": 0,
                },
            )
        if method.upper() == "EARS-C1":
            flags, z_scores = _ears_c1(series, z_thr=z_thr)
        elif method.upper() == "CUSUM":
            flags, z_scores = _cusum(series, k=0.5, h=5.0)
        else:
            return CallResult(
                content={"error": "unknown method", "known": ["EARS-C1", "CUSUM"]},
                is_error=True,
            )

        records = [
            {"week_start": w, "value": float(v),
             "z": float(z) if z == z else None,  # NaN → None
             "flagged": bool(f)}
            for w, v, z, f in zip(weeks, series, z_scores, flags)
        ]
        return CallResult(
            content={
                "status": "ok",
                "gu": gu,
                "method": method,
                "threshold": z_thr if method.upper() == "EARS-C1" else None,
                "n_weeks": len(records),
                "n_flagged": int(sum(flags)),
                "series": records,
            }
        )

    def _h_validity_check(self, args: dict) -> CallResult:
        from simulation.verifier.epi_validity import check_epi_validity

        params = args.get("params")
        preds = args.get("predictions")
        if preds is not None:
            preds_arr = np.asarray(preds, dtype=float)
        else:
            preds_arr = None
        result = check_epi_validity(
            params=params,
            predictions=preds_arr,
            raise_on_fail=False,
        )
        return CallResult(
            content={
                "status": result.status,
                "checker": result.checker,
                "details": result.details,
            }
        )

    def _h_scenario_run(self, args: dict) -> CallResult:
        from simulation.sim import SCENARIO_REGISTRY, run_scenario
        from simulation.sim.io import load_metapop_params

        scenario = str(args["scenario"])
        days = int(args.get("days", 200))
        seed_district = str(args.get("seed_district", "강남구"))
        seed_infected = float(args.get("seed_infected", 10.0))
        use_db = bool(args.get("use_db", True))

        if scenario not in SCENARIO_REGISTRY:
            return CallResult(
                content={
                    "error": f"unknown scenario: {scenario!r}",
                    "known": sorted(SCENARIO_REGISTRY),
                },
                is_error=True,
            )

        base = None
        if use_db:
            try:
                base = load_metapop_params(
                    seed_infected=seed_infected,
                    seed_district=seed_district,
                    days=days,
                )
            except Exception as e:  # pragma: no cover
                log.warning("scenario_run: DB load failed (%s); using synthetic", e)
                base = None

        result = run_scenario(scenario, base, overrides={"days": days})

        I_total = result.city_total("I")
        D_total = result.city_total("D")
        V_total = result.city_total("V")
        # ``result.days`` is the (T+1,) time-axis array, not a scalar —
        # expose it as ``t_axis`` and return the scalar horizon under
        # ``days`` so the LLM / UI can consume both without ambiguity.
        t_axis = np.asarray(result.days).tolist()
        out = {
            "status": "ok",
            "scenario": scenario,
            "days": int(len(t_axis) - 1),
            "t_axis": t_axis,
            "peak_I": float(I_total.max()),
            "peak_day": int(I_total.argmax()),
            "final_D": float(D_total[-1]),
            "final_V": float(V_total[-1]),
            "epi_validity": result.epi_validity.get("metapop_seirvd", {}),
            "I_city_series": I_total.tolist(),
            "D_city_series": D_total.tolist(),
            "district_names": list(result.district_names),
        }

        # International ILI reference for SEIRVD comparison (optional)
        if args.get("include_international", False):
            try:
                intl = self._load_international_context(lookback_weeks=52)
                out["international_reference"] = intl
            except Exception as e:
                log.warning("scenario_run: international reference failed (%s)", e)
                out["international_reference"] = {"error": str(e)}

        return CallResult(content=out)

    def _h_coupled_forward(self, args: dict) -> CallResult:
        """Expose the forecast-anchored + EnKF-coupled behavioural ABM (with district
        resolution) to ARIA — the coupled agent model, not the standalone metapop.

        Leak-free: enkf_couple_forward assimilates the champion FORECAST nowcast (never
        the forward truth); the commuter NGM / import-fraction are pure functions of the
        KOSIS OD matrix + populations. Every returned number is a tool receipt, so the
        provenance-gated blackboard can ground the advisory on the coupled forward.
        """
        import numpy as np

        from simulation.abm.variant_ablation import enkf_couple_forward
        from simulation.sim.commuter_ngm import commuter_ngm, import_fraction
        from simulation.sim.io import load_metapop_params

        n_agents = int(args.get("n_agents", 6000))
        n_seeds = int(args.get("n_seeds", 4))
        try:
            enkf = enkf_couple_forward(variant="H", n_agents=n_agents, n_seeds=n_seeds)
        except Exception as e:  # pragma: no cover
            return CallResult(content={"error": f"coupled forward failed: {e}"},
                              is_error=True)

        t = enkf["trajectories"]
        ens = np.asarray(enkf.get("assimilated_ensemble", []), dtype=float)
        band_lo = np.percentile(ens, 2.5, axis=0).tolist() if ens.size else None
        band_hi = np.percentile(ens, 97.5, axis=0).tolist() if ens.size else None

        # district-resolved commuter-coupled transmission (Clause-2), leak-free
        p = load_metapop_params()
        M = np.asarray(p.mobility, float); pops = np.asarray(p.populations, float)
        gu = list(p.district_names)
        gamma, beta = 0.18, 1.3 * 0.18
        ngm = commuter_ngm(M, pops, beta=beta, gamma=gamma)
        imp = import_fraction(M, pops)
        order = np.argsort(-imp)

        out = {
            "status": "ok",
            "coupling": "champion FusedEpi forecast → anchor + EnKF → hybrid "
                        "behavioural ABM over 25 districts (leak-free)",
            "n_agents": n_agents, "n_seeds": n_seeds,
            "forward_dates": t.get("forward_dates"),
            "champion_forecast": t.get("champion_forecast"),
            "abm_alone": t.get("variant_alone"),
            "abm_plus_enkf": t.get("variant_plus_enkf"),
            "ensemble_band_lo": band_lo, "ensemble_band_hi": band_hi,
            "champion_forward_r2": enkf.get("champion_alone_forward_r2"),
            "abm_alone_forward_r2": enkf.get("variant_alone_forward_r2"),
            "abm_plus_enkf_forward_r2": enkf.get("variant_plus_enkf_forward_r2"),
            "commuter_r_eff": round(float(ngm["r_eff"]), 4),
            "district_import_fraction": {gu[i]: round(float(imp[i]), 4) for i in range(len(gu))},
            "district_target_load": {gu[i]: round(float(ngm["district_in"][i]), 4)
                                     for i in range(len(gu))},
            "top_import_districts": [gu[int(i)] for i in order[:5]],
            "leak_free": True,
        }
        return CallResult(content=out)

    def _load_international_context(self, lookback_weeks: int = 52) -> dict:
        """현재 시즌 국제 ILI positivity 로드 (SEIRVD 비교용).

        Args:
            lookback_weeks: 최근 N주 데이터 로드

        Returns:
            {country: {label, series: [(week_label, value), ...], peak_week, mean}} dict

        Side effects: 없음 (read-only DB query)
        Raises:
            sqlite3.Error: DB 접근 실패 시
        """
        import sqlite3
        from simulation.database import safe_connect

        COUNTRY_LABELS = {
            "US": "미국 (CDC ILINet)",
            "JP": "일본 (JIHS/FluNet)",
            "KR": "한국 (WHO FluNet)",
            "DE": "독일", "FR": "프랑스",
            "GB": "영국(잉글랜드)", "NL": "네덜란드", "SE": "스웨덴",
        }
        db_path = str(Path(__file__).resolve().parent.parent / "data/db/epi_real_seoul.db")

        con = safe_connect(db_path)
        cur = con.cursor()

        # 각 국가별 최근 lookback_weeks 데이터
        result = {}
        for country, label in COUNTRY_LABELS.items():
            source_pref = "cdc_ilinet" if country == "US" else "who_flunet"
            if country == "JP":
                # JP: JIHS 우선 (실제 ILI rate), fallback who_flunet
                cur.execute('''
                    SELECT year, week_no, ili_rate FROM overseas_ili
                    WHERE country=? AND source='japan_jihs' AND ili_rate IS NOT NULL AND ili_rate>0
                    ORDER BY year DESC, week_no DESC LIMIT ?
                ''', (country, lookback_weeks))
                rows = cur.fetchall()
                if len(rows) < 4:
                    source_pref = "who_flunet"
            if country != "JP" or len(rows) < 4:
                cur.execute('''
                    SELECT year, week_no, ili_rate FROM overseas_ili
                    WHERE country=? AND source=? AND ili_rate IS NOT NULL AND ili_rate>0
                    ORDER BY year DESC, week_no DESC LIMIT ?
                ''', (country, source_pref, lookback_weeks))
                rows = cur.fetchall()

            if not rows:
                continue
            rows = sorted(rows, key=lambda r: (r[0], r[1]))
            vals = [r[2] for r in rows]
            result[country] = {
                "label": label,
                "series": [(f"{r[0]}-W{r[1]:02d}", r[2]) for r in rows],
                "peak_week": f"{rows[int(np.argmax(vals))][0]}-W{rows[int(np.argmax(vals))][1]:02d}",
                "mean": round(float(np.mean(vals)), 4),
                "max": round(float(np.max(vals)), 4),
                "n_weeks": len(rows),
            }

        con.close()
        return result

    def _h_international_compare(self, args: dict) -> CallResult:
        """국제 ILI 비교 — KR vs US/JP/EU positivity 상관·리드래그·피크 분석.

        Args:
            args: {countries, start_year, season, metric}

        Returns:
            CallResult with {correlations, peak_comparison, series, context_summary}

        Side effects: 없음 (read-only DB)
        Performance: O(n_countries × n_weeks) — 일반적으로 <200ms
        """
        import sqlite3
        from simulation.database import safe_connect

        countries = args.get("countries", ["US", "JP", "DE", "NL", "SE"])
        start_year = int(args.get("start_year", 2019))
        season_filter = args.get("season", None)   # e.g. "2023/2024"
        metric = args.get("metric", "positivity")

        COUNTRY_LABELS = {
            "US": "미국 (CDC ILINet)", "JP": "일본", "KR": "한국",
            "DE": "독일", "FR": "프랑스", "GB": "영국", "NL": "네덜란드", "SE": "스웨덴",
        }

        db_path = str(Path(__file__).resolve().parent.parent / "data/db/epi_real_seoul.db")
        con = safe_connect(db_path)
        cur = con.cursor()

        # 시즌 필터 → year 범위로 변환
        if season_filter:
            try:
                y1, y2 = [int(y) for y in season_filter.split("/")]
                year_cond = f"AND ((year={y1} AND week_no>=27) OR (year={y2} AND week_no<27))"
            except Exception:
                year_cond = f"AND year >= {start_year}"
        else:
            year_cond = f"AND year >= {start_year}"

        # 국가별 데이터 로드 (KR 포함 강제)
        all_countries = list(set(["KR"] + [c.upper() for c in countries]))
        data: dict[str, dict] = {}

        for country in all_countries:
            source = "cdc_ilinet" if (country == "US" and metric == "ili_rate") else "who_flunet"
            if country == "JP":
                # JP: JIHS 있으면 우선
                cur.execute(f'''
                    SELECT year, week_no, ili_rate FROM overseas_ili
                    WHERE country=? AND source='japan_jihs'
                      AND ili_rate IS NOT NULL AND ili_rate>0 {year_cond}
                    ORDER BY year, week_no
                ''', (country,))
                rows = cur.fetchall()
                if len(rows) < 8:
                    cur.execute(f'''
                        SELECT year, week_no, ili_rate FROM overseas_ili
                        WHERE country=? AND source='who_flunet'
                          AND ili_rate IS NOT NULL AND ili_rate>0 {year_cond}
                        ORDER BY year, week_no
                    ''', (country,))
                    rows = cur.fetchall()
            else:
                cur.execute(f'''
                    SELECT year, week_no, ili_rate FROM overseas_ili
                    WHERE country=? AND source=?
                      AND ili_rate IS NOT NULL AND ili_rate>0 {year_cond}
                    ORDER BY year, week_no
                ''', (country, source))
                rows = cur.fetchall()

            if rows:
                data[country] = {
                    (r[0], r[1]): r[2] for r in rows
                }

        con.close()

        if "KR" not in data:
            return CallResult(
                content={"error": "KR 데이터 없음 — overseas_ili 확인 필요"},
                is_error=True,
            )

        kr_map = data["KR"]

        # ── 상관 + 리드래그 분석 ──────────────────────────────────────
        correlations = []
        for country in all_countries:
            if country == "KR" or country not in data:
                continue
            oc_map = data[country]
            common = sorted(set(kr_map) & set(oc_map))
            if len(common) < 12:
                continue
            kr_vals = np.array([kr_map[k] for k in common])
            oc_vals = np.array([oc_map[k] for k in common])

            try:
                from scipy.stats import pearsonr
                r0, p0 = pearsonr(kr_vals, oc_vals)
            except ImportError:
                r0 = float(np.corrcoef(kr_vals, oc_vals)[0, 1])
                p0 = float("nan")

            # 최적 lag ±8주
            best_lag, best_r = 0, r0
            for lag in range(-8, 9):
                if lag == 0:
                    continue
                if lag > 0:
                    a, b = kr_vals[lag:], oc_vals[:-lag]
                else:
                    a, b = kr_vals[:lag], oc_vals[-lag:]
                if len(a) < 12:
                    continue
                try:
                    lr = float(np.corrcoef(a, b)[0, 1])
                    if abs(lr) > abs(best_r):
                        best_r, best_lag = lr, lag
                except Exception:
                    pass

            correlations.append({
                "country": country,
                "label": COUNTRY_LABELS.get(country, country),
                "n_overlap": len(common),
                "pearson_r": round(float(r0), 4),
                "pearson_p": round(float(p0), 6) if not np.isnan(p0) else None,
                "best_lag_weeks": best_lag,
                "lagged_r": round(float(best_r), 4),
                "interpretation": (
                    f"한국이 {abs(best_lag)}주 {'선행' if best_lag < 0 else '후행' if best_lag > 0 else '동시'}"
                    if best_lag != 0 else "동시 피크"
                ),
            })

        correlations.sort(key=lambda x: -abs(x["pearson_r"]))

        # ── 피크 주차 비교 (최근 3시즌) ──────────────────────────────
        def get_season(yr, wk):
            return f"{yr}/{yr+1}" if wk >= 27 else f"{yr-1}/{yr}"

        peak_comparison = []
        # KR 피크
        kr_by_season: dict[str, tuple] = {}
        for (yr, wk), val in kr_map.items():
            s = get_season(yr, wk)
            if s not in kr_by_season or val > kr_by_season[s][1]:
                kr_by_season[s] = (wk, val)
        recent_seasons = sorted(kr_by_season)[-3:]
        for country in all_countries:
            if country == "KR" or country not in data:
                continue
            oc_map = data[country]
            oc_by_season: dict[str, tuple] = {}
            for (yr, wk), val in oc_map.items():
                s = get_season(yr, wk)
                if s not in oc_by_season or val > oc_by_season[s][1]:
                    oc_by_season[s] = (wk, val)
            for s in recent_seasons:
                if s in kr_by_season and s in oc_by_season:
                    diff = oc_by_season[s][0] - kr_by_season[s][0]
                    peak_comparison.append({
                        "season": s,
                        "country": country,
                        "label": COUNTRY_LABELS.get(country, country),
                        "kr_peak_week": kr_by_season[s][0],
                        "country_peak_week": oc_by_season[s][0],
                        "diff_weeks": diff,
                        "direction": "동시" if diff == 0 else (
                            f"한국 {abs(diff)}주 선행" if diff > 0 else f"한국 {abs(diff)}주 후행"
                        ),
                    })

        # ── 정규화 시계열 (최근 52주) ────────────────────────────────
        series_out = {}
        for country in (["KR"] + [c for c in all_countries if c != "KR"]):
            if country not in data:
                continue
            recent_keys = sorted(data[country].keys())[-52:]
            vals = [data[country][k] for k in recent_keys]
            if not vals:
                continue
            mx = max(vals) if max(vals) > 0 else 1.0
            series_out[country] = {
                "label": COUNTRY_LABELS.get(country, country),
                "weeks": [f"{k[0]}-W{k[1]:02d}" for k in recent_keys],
                "values_raw": [round(v, 5) for v in vals],
                "values_norm": [round(v / mx, 4) for v in vals],  # 0-1 정규화
            }

        # ── LLM 컨텍스트 요약 ────────────────────────────────────────
        top = correlations[0] if correlations else {}
        summary_lines = [
            f"[국제 ILI 비교] 분석 기간: {start_year}년~현재, {metric} 지표",
            f"KR 데이터: {len(kr_map)}주",
        ]
        for c in correlations[:5]:
            lag_str = f"KR {abs(c['best_lag_weeks'])}주 {'선행' if c['best_lag_weeks']<0 else '후행'}" if c['best_lag_weeks'] != 0 else "동시"
            summary_lines.append(
                f"  {c['label']}: r={c['pearson_r']:.3f} (lag={lag_str}, lagged_r={c['lagged_r']:.3f})"
            )
        if peak_comparison:
            summary_lines.append("피크 주차 (최근 시즌):")
            for p in peak_comparison[-6:]:
                summary_lines.append(
                    f"  {p['season']} {p['label']}: KR W{p['kr_peak_week']} vs 대상 W{p['country_peak_week']} ({p['direction']})"
                )
        context_summary = "\n".join(summary_lines)

        return CallResult(content={
            "status": "ok",
            "metric": metric,
            "start_year": start_year,
            "season_filter": season_filter,
            "n_countries": len(correlations) + 1,
            "correlations": correlations,
            "peak_comparison": peak_comparison,
            "series_last_52w": series_out,
            "context_summary": context_summary,
        })

    # ══════════════════════════════════════════════════════════════════
    # Graceful stubs — contract locked, artifacts TBD
    # ══════════════════════════════════════════════════════════════════
    def _h_forecast(self, args: dict) -> CallResult:
        """Live forecast using ChampionArtifact .pt files.

        Priority:
          1. **Live** — load champions from ``models/`` and run
             ``run_inference`` for the requested horizon. h=1 (next week)
             is the operational KPI.
          2. **Cached** — pre-computed manifest at
             ``stage3_forecasts.json`` (legacy / offline UI).
          3. **Schema-only** — neither champion nor manifest available.

        Resilience: enriched features are built once and cached per server
        (``_get_enriched_features`` load-once), and any live-path failure is
        caught and degrades to the cached manifest or a ``not_available`` schema
        stub (graceful circuit-break — never propagates an exception to the
        caller). NOTE: there is **no hard per-call timeout** on the live path —
        on-demand ``.pt`` loading + ``run_inference`` could in principle hang on a
        pathological artifact. A wall-clock timeout is omitted deliberately for
        portability: ``signal.alarm`` is Unix-only, and a watchdog thread cannot
        interrupt CPU-bound numpy/torch work. Operators needing a hard bound
        should wrap the MCP call at the transport layer.
        """
        # 1. Try live champion-based forecast (preferred)
        models_dir = Path("models")
        log_path = models_dir / "champion_log.json"
        if log_path.exists():
            try:
                horizon = int(args.get("horizon", 4))
                model_filter = args.get("model_id")
                wanted: Optional[list[str]] = None
                if model_filter and model_filter != "ensemble":
                    wanted = [m.strip() for m in str(model_filter).split(",") if m.strip()]
                # Lazy import — keep MCP startup fast
                from simulation.pipeline.inference import run_inference
                import polars as pl
                import numpy as np

                # Cached enriched-feature build (rebuilt per call before this fix)
                feat_df, meta = self._get_enriched_features()
                target_col = meta.get("target_col", "ili_rate")
                dates_arr = meta.get("dates")
                schema = feat_df.schema
                num_dt = (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8,
                            pl.UInt16, pl.UInt32, pl.UInt64,
                            pl.Float32, pl.Float64, pl.Boolean)
                feat_cols = [c for c in feat_df.columns
                             if c != target_col and schema[c] in num_dt]
                X_full = feat_df.select(feat_cols).to_numpy().astype(np.float64)
                y_full = (feat_df[target_col].to_numpy().astype(np.float64)
                            if target_col in feat_df.columns else None)
                # auto-pad covid_era_indicator if needed
                from simulation.utils.model_artifact import load_artifact
                expected = 0
                for nm in (wanted or list(_load_champion_names(log_path))):
                    art = load_artifact(models_dir / f"{nm}.pt")
                    if art and art.scaler is not None:
                        try:
                            expected = max(expected, int(art.scaler.n_features_in_))
                        except Exception:
                            pass
                        if art.feature_indices:
                            expected = max(expected, max(art.feature_indices) + 1)
                if expected > X_full.shape[1]:
                    pad = expected - X_full.shape[1]
                    # D9 (M7): disclose the silent zero-padding — a missing REAL
                    # covariate (not the benign covid_era_indicator) would be
                    # filled with zeros and served as 'live' without any signal.
                    log.warning(
                        "  [forecast] feature_pad: %d missing feature column(s) "
                        "zero-filled (expected %d, got %d) — forecast computed on "
                        "fabricated zeros for those columns", pad, expected, X_full.shape[1])
                    self.__dict__["_last_feature_pad"] = {
                        "n_padded": int(pad), "expected": int(expected),
                        "available": int(X_full.shape[1])}
                    X_full = np.hstack([X_full,
                                          np.zeros((len(X_full), pad), dtype=np.float64)])

                i0 = max(0, len(X_full) - horizon)
                X_inf = X_full[i0:]
                y_inf = y_full[i0:] if y_full is not None else None
                dates_inf = dates_arr[i0:] if dates_arr is not None else None

                res = run_inference(
                    X_inference=X_inf,
                    inference_dates=dates_inf,
                    actuals=y_inf,
                    model_names=wanted,
                    models_dir=models_dir,
                    log_path=log_path,
                )
                # Format MCP-style series
                series = []
                preds = res.get("predictions", {})
                champions = res.get("champions_used", {})
                for h, _ in enumerate(X_inf):
                    week_start = (str(dates_inf[h])[:10] if dates_inf is not None
                                    else f"h+{h+1}")
                    row = {"week_start": week_start, "horizon": h + 1}
                    if y_inf is not None and h < len(y_inf):
                        row["actual"] = float(y_inf[h])
                    for nm, ypreds in preds.items():
                        if h < len(ypreds):
                            row[nm] = float(ypreds[h])
                    series.append(row)
                return CallResult(content={
                    "status": "live",
                    "source": "ChampionArtifact .pt files",
                    "horizon": horizon,
                    "n_models": len(preds),
                    "models_used": list(preds.keys()),
                    "primary_kpi": "h=1 (next week)",
                    "series": series,
                    "metrics_per_model": res.get("metrics_per_model", {}),
                    "champions_meta": {nm: {
                        "version": c.get("version"),
                        "test_wis_at_promotion": c.get("test_wis_at_promotion"),
                        "promoted_at": c.get("promoted_at"),
                    } for nm, c in champions.items()},
                })
            except Exception as e:
                # fall through to cached / schema-only
                logging.getLogger(__name__).warning(
                    f"epi.forecast live path failed: {e}")

        # 2. Cached pre-computed manifest
        manifest = self.artifacts_dir / "stage3_forecasts.json"
        if manifest.exists():
            return self._read_artifact_view(
                manifest,
                args_key=("gu", "horizon", "model_id"),
                args=args,
            )

        # 3. Schema-only fallback
        return CallResult(
            content={
                "status": "not_available",
                "message": (
                    "epi.forecast requires either champion .pt files in "
                    "`models/` (preferred — run `simulation train --per-model-optimize`) "
                    f"OR a cached manifest at {manifest}."
                ),
                "expected_artifact": str(manifest),
                "expected_schema": {
                    "gu": "str", "horizon": "int", "model_id": "str",
                    "series": [{
                        "week_start": "YYYY-MM-DD",
                        "point": "float", "pi_lo": "float", "pi_hi": "float",
                    }],
                },
                "request_echo": {
                    "gu": args.get("gu"),
                    "horizon": args.get("horizon", 4),
                    "model_id": args.get("model_id", "ensemble"),
                },
            },
        )

    def _h_model_compare(self, args: dict) -> CallResult:
        # Live: per_model_eval/per_model_metrics.csv — per-model accuracy + DM test.
        csv_path = self.artifacts_dir / "per_model_eval" / "per_model_metrics.csv"
        if csv_path.exists():
            try:
                import csv as _csv

                def _f(x):
                    try:
                        return round(float(x), 6)
                    except (TypeError, ValueError):
                        return None

                with csv_path.open(encoding="utf-8") as fh:
                    rows = list(_csv.DictReader(fh))
                metric = args.get("metric", "wis")
                comp = [{
                    "model": r.get("model"),
                    "wis": _f(r.get("wis")), "mae": _f(r.get("mae")), "r2": _f(r.get("r2")),
                    "dm_p_value": _f(r.get("dm_p_value")),
                    "dm_p_value_bh": _f(r.get("dm_p_value_bh")),
                } for r in rows]
                comp.sort(key=lambda d: (d.get(metric) if d.get(metric) is not None else 1e18))
                return CallResult(content={
                    "status": "live",
                    "source": "per_model_eval/per_model_metrics.csv",
                    "metric": metric, "n_models": len(comp), "models": comp,
                })
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"epi.model_compare live read failed: {e}")
        manifest = self.artifacts_dir / "stage4_dm_results.json"
        if manifest.exists():
            return self._read_artifact_view(
                manifest, args_key=("week", "models", "metric"), args=args)
        return CallResult(
            content={
                "status": "not_available",
                "message": (
                    "epi.model_compare requires per_model_eval/per_model_metrics.csv "
                    f"(live) or legacy {manifest}."
                ),
                "expected_artifact": str(csv_path),
                "request_echo": {
                    "week": args.get("week"),
                    "models": args.get("models", []),
                    "metric": args.get("metric", "wis"),
                },
            },
        )

    def _h_shap_features(self, args: dict) -> CallResult:
        # Live: R11 (shap) SHAP output (shap/_summary.json — all-family permutation
        # + native importance). Falls back to the retired stage3 path.
        manifest = self.artifacts_dir / "shap" / "_summary.json"
        if not manifest.exists():
            manifest = self.artifacts_dir / "stage3_shap" / "summary.json"
        if manifest.exists():
            return self._read_artifact_view(
                manifest,
                args_key=("gu", "week", "model"),
                args=args,
            )
        return CallResult(
            content={
                "status": "not_available",
                "message": (
                    "epi.shap_features requires R11 (shap) SHAP artifacts "
                    f"at {manifest}. Contract unchanged."
                ),
                "expected_artifact": str(manifest),
                "request_echo": {
                    "gu": args.get("gu"),
                    "week": args.get("week"),
                    "top_n": args.get("top_n", 10),
                    "model": args.get("model", "XGBoost"),
                },
            },
        )

    def _h_lead_time(self, args: dict) -> CallResult:
        # Live: per_model_eval/per_model_metrics.csv (lead_time_weeks per model).
        csv_pm = self.artifacts_dir / "per_model_eval" / "per_model_metrics.csv"
        if csv_pm.exists():
            try:
                import csv as _csv
                with csv_pm.open(encoding="utf-8") as fh:
                    rows = list(_csv.DictReader(fh))
                out = []
                for r in rows:
                    lt = r.get("lead_time_weeks")
                    if lt in (None, ""):
                        continue
                    try:
                        out.append({"model": r.get("model"), "lead_time_weeks": round(float(lt), 4)})
                    except (TypeError, ValueError):
                        pass
                if out:
                    return CallResult(content={
                        "status": "live",
                        "source": "per_model_eval/per_model_metrics.csv",
                        "models": out,
                    })
            except Exception as e:
                logging.getLogger(__name__).warning(
                    f"epi.lead_time live read failed: {e}")
        # Preferred: full per-horizon WF-CV artifact (+).
        manifest = self.artifacts_dir / "stage4_lead_time.json"
        if manifest.exists():
            return self._read_artifact_view(
                manifest,
                args_key=("model", "max_horizon"),
                args=args,
            )

        # proxy: single post-E evaluation point per model.
        csv_path = (
            self.artifacts_dir / "pi_v22_6_epi_eval" / "peak_onset.csv"
        )
        if csv_path.exists():
            return self._lead_time_from_peak_onset_csv(csv_path, args)

        return CallResult(
            content={
                "status": "not_available",
                "message": (
                    "epi.lead_time_analysis requires either the full "
                    f"Stage 4 artifact at {manifest} or the "
                    f"proxy at {csv_path}. Neither is present."
                ),
                "expected_artifacts": [str(manifest), str(csv_path)],
                "request_echo": {
                    "model": args.get("model"),
                    "max_horizon": args.get("max_horizon", 4),
                },
            },
        )

    def _lead_time_from_peak_onset_csv(
        self, csv_path: Path, args: dict,
    ) -> CallResult:
        """Compute lead times from post-E peak_onset.csv.

        Semantics
        ---------
        * `onset_week_error_weeks` : forecast onset - observed onset
            (positive = LATE alert; negative = EARLY alert).
        * `peak_week_error_weeks`  : same sign convention for the peak.
        * Lead time = -error, so positive lead time = earlier than truth.

        Output shape
        ------------
        * When ``model`` is given: single record + rank (1 = earliest).
        * Otherwise: top-k models sorted by onset lead time descending
            (ties broken by peak lead time).
        """
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
        except Exception as e:
            return CallResult(
                content={"error": f"peak_onset.csv unreadable: {e}",
                         "path": str(csv_path)},
                is_error=True,
            )

        required = {
            "model", "category",
            "peak_week_error_weeks", "onset_week_error_weeks",
        }
        missing = required - set(df.columns)
        if missing:
            return CallResult(
                content={
                    "error": "peak_onset.csv is missing expected columns",
                    "missing_columns": sorted(missing),
                    "path": str(csv_path),
                },
                is_error=True,
            )

        # Coerce "NA" strings / NaN to missing; keep only numeric rows.
        df["onset_week_error_weeks"] = pd.to_numeric(
            df["onset_week_error_weeks"], errors="coerce",
        )
        df["peak_week_error_weeks"] = pd.to_numeric(
            df["peak_week_error_weeks"], errors="coerce",
        )
        df["lead_time_onset_weeks"] = -df["onset_week_error_weeks"]
        df["lead_time_peak_weeks"] = -df["peak_week_error_weeks"]

        # Rank: higher lead_time_onset = better (earlier alerter).
        df_sorted = df.sort_values(
            ["lead_time_onset_weeks", "lead_time_peak_weeks"],
            ascending=[False, False],
            kind="stable",
            na_position="last",
        ).reset_index(drop=True)
        df_sorted["rank_by_onset_lead_time"] = df_sorted.index + 1

        target_model = args.get("model")
        if target_model:
            hit = df_sorted[df_sorted["model"] == target_model]
            if hit.empty:
                available = df_sorted["model"].tolist()
                return CallResult(
                    content={
                        "status": "model_not_found",
                        "message": (
                            f"model {target_model!r} not in peak_onset.csv"
                        ),
                        "available_models": available,
                    },
                    is_error=True,
                )
            row = hit.iloc[0].to_dict()
            record = {
                "model": row["model"],
                "category": row["category"],
                "onset_lead_time_weeks":
                    _nanable(row["lead_time_onset_weeks"]),
                "peak_lead_time_weeks":
                    _nanable(row["lead_time_peak_weeks"]),
                "rank_by_onset_lead_time":
                    int(row["rank_by_onset_lead_time"]),
                "n_models_ranked": int(len(df_sorted)),
            }
            return CallResult(
                content={
                    "status": "csv_proxy",
                    "note": (
                        "Single-horizon operational proxy from post-E "
                        "peak_onset.csv . Full per-horizon curve "
                        "pending WF-CV rerun."
                    ),
                    "source_artifact": str(csv_path),
                    "record": record,
                    "request_echo": {
                        "model": target_model,
                        "max_horizon": args.get("max_horizon", 4),
                    },
                },
            )

        top_k = int(args.get("top_k", 10))
        top_k = max(1, min(top_k, len(df_sorted)))
        top_rows = df_sorted.head(top_k)
        ranking = [
            {
                "rank": int(r["rank_by_onset_lead_time"]),
                "model": r["model"],
                "category": r["category"],
                "onset_lead_time_weeks":
                    _nanable(r["lead_time_onset_weeks"]),
                "peak_lead_time_weeks":
                    _nanable(r["lead_time_peak_weeks"]),
            }
            for _, r in top_rows.iterrows()
        ]
        return CallResult(
            content={
                "status": "csv_proxy",
                "note": (
                    "Single-horizon operational proxy from post-E "
                    "peak_onset.csv . Higher onset_lead_time = "
                    "earlier alerter (positive = ahead of observed "
                    "onset)."
                ),
                "source_artifact": str(csv_path),
                "n_models_ranked": int(len(df_sorted)),
                "top_k": top_k,
                "ranking": ranking,
                "request_echo": {
                    "model": None,
                    "max_horizon": args.get("max_horizon", 4),
                    "top_k": top_k,
                },
            },
        )

    def _h_literature_rag(self, args: dict) -> CallResult:
        query = args.get("query") or ""
        k = int(args.get("k", 5))
        k = max(1, min(k, 20))

        # Flag-gated hybrid GraphRAG + optional LLM-judge rerank (additive, default
        # OFF; on any failure it falls through to the vector RAG below).
        try:
            from simulation.server.rag.hybrid_rerank import (
                hybrid_rag_search, graph_rag_enabled)
            if graph_rag_enabled():
                ghits = hybrid_rag_search(query, k=k)
                if ghits:
                    reranked = any("rerank_score" in h for h in ghits)
                    return CallResult(
                        content={
                            "status": "graph_rag_hybrid",
                            "backend": (
                                "GraphRAG (TF-IDF + dense + RRF + mesh) over PubMed"
                                + (" + LLM-judge rerank" if reranked else "")
                            ),
                            "results": ghits,
                            "request_echo": {"query": query, "k": k},
                        },
                    )
        except Exception:
            pass

        # Preferred: LanceDB vector RAG when available + indexed.
        try:
            from simulation.server.rag import semantic_search, rag_info
            info = rag_info()
            if info.get("lancedb_available") and info.get("table_exists"):
                hits = semantic_search(query, k=k)
                if hits:
                    return CallResult(
                        content={
                            "status": "vector_rag",
                            "backend": "lancedb + sentence-transformers (all-MiniLM-L6-v2)",
                            "results": hits,
                            "request_echo": {"query": query, "k": k},
                        },
                    )
        except Exception as e:
            # Fall through to static fallback; never crash the tool call
            pass

        # Fallback: curated static catalogue + keyword overlap match.
        return CallResult(
            content={
                "status": "static_fallback",
                "note": (
                    f"vector RAG unavailable — returning top-{k} keyword-overlap "
                    "matches from the curated project bibliography "
                    "(~20 entries). Build the vector index via "
                    "`python -c 'from simulation.server.rag import build_index; build_index()'` "
                    "to enable semantic search."
                ),
                "results": self._static_citation_results(query, k),
                "source": "simulation.server.static_citations.STATIC_CITATIONS",
                "request_echo": {"query": query, "k": k},
            },
        )

    def _static_citation_results(self, query: str, k: int) -> list[dict]:
        """Score + serialise the static citation catalogue."""
        # Local import so the catalogue module is only loaded when needed.
        from .static_citations import score_citations

        matches = score_citations(query, k=k)
        return [
            {"score": score, **cit.to_dict()}
            for score, cit in matches
        ]

    # ── Shared artifact helper ────────────────────────────────────────
    def _read_artifact_view(
        self, manifest: Path, *, args_key: tuple[str, ...], args: dict,
    ) -> CallResult:
        """Read a JSON manifest and pass-through the relevant slice.

        Intentionally minimal — the manifest itself defines its own
        shape. Handlers that need richer filtering can override.
        """
        try:
            with manifest.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception as e:
            return CallResult(
                content={"error": f"artifact unreadable: {e}", "path": str(manifest)},
                is_error=True,
            )
        return CallResult(
            content={
                "status": "ok",
                "artifact": str(manifest),
                "request_echo": {k: args.get(k) for k in args_key},
                "payload": payload,
            }
        )


# ══════════════════════════════════════════════════════════════════════
# Helpers — data loaders + outbreak algorithms
# ══════════════════════════════════════════════════════════════════════
def _default_artifacts_dir() -> Path:
    env = os.getenv("EPI_ARTIFACTS_DIR")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    return here.parent.parent / "results"


_SEOUL_ALIASES = frozenset(
    {"seoul_city", "seoul", "서울", "서울시", "서울특별시", "all", ""}
)


def _is_seoul_city_key(gu: Optional[str]) -> bool:
    """Return True iff ``gu`` refers to the city-aggregate Seoul series."""
    if gu is None:
        return True
    gu_key = gu.strip().lower()
    return gu_key in _SEOUL_ALIASES or gu in _SEOUL_ALIASES


def _load_weekly_incidence(
    gu: str, lookback_weeks: int
) -> tuple[np.ndarray, list[str]]:
    """Return (incidence_series, week_start_iso_list) for the given gu.

    The project DB has no per-gu weekly ILI panel yet — the 53-model
    pipeline trains on **city-aggregate Seoul ILI rate**. This function
    therefore only supports ``gu='seoul_city'`` (plus common aliases).

    For the Seoul aliases it first tries ``epi.weekly_disease`` filtered
    to ``sido_nm IN ('서울', '서울특별시') AND disease_cd LIKE 'J%'`` (the
    contract requested in the Stage 6 MCP spec). If that returns
    insufficient rows — which is the common case on this DB, since
    ``weekly_disease`` is a legally-notifiable-diseases table without
    continuous ILI coverage — it falls back to the training target
    ``y_true`` series stored in every ``predictions_*.csv`` so callers
    still get a usable weekly series.

    For specific gus (e.g. ``'강남구'``), returns an empty array with a
    clear log message; downstream handlers surface
    ``"gu-panel data not yet available; call with gu='seoul_city'"``.
    """
    if not _is_seoul_city_key(gu):
        log.info(
            "_load_weekly_incidence: gu=%r — gu-panel data not yet "
            "available; call with gu='seoul_city'.", gu,
        )
        return np.array([]), []

    # Primary path: DB query against epi.weekly_disease for Seoul totals.
    series_db, weeks_db = _load_seoul_weekly_from_db(lookback_weeks)
    min_points = 10  # rt_estimate.lookback_weeks has minimum 10
    if series_db.size >= min_points:
        return series_db, weeks_db

    if series_db.size > 0:
        log.info(
            "weekly_disease returned only %d rows for Seoul flu series; "
            "falling back to predictions_*.csv y_true.", series_db.size,
        )
    else:
        log.info(
            "weekly_disease has no Seoul flu rows matching the Stage 6 "
            "filter; falling back to predictions_*.csv y_true.",
        )

    return _load_seoul_weekly_from_csv(lookback_weeks)


def _load_seoul_weekly_from_db(
    lookback_weeks: int,
) -> tuple[np.ndarray, list[str]]:
    """Query ``epi.weekly_disease`` for the Seoul flu weekly series.

    Uses ``sido_nm IN ('서울', '서울특별시')`` to tolerate both the short
    and canonical sido encodings (the live DB stores '서울', but the
    Stage 6 spec referenced '서울특별시'). ``disease_cd LIKE 'J%'`` keeps
    the tool contract aligned with ICD-J respiratory filtering — it is a
    no-op on this DB today (codes are Korean-prefixed) but forward-
    compatible with future re-coded ingests.
    """
    sql = (
        "SELECT year, week_no, SUM(cases) AS cases "
        "FROM epi.weekly_disease "
        "WHERE sido_nm IN ('서울', '서울특별시') "
        "  AND (disease_cd LIKE 'J%' OR disease_nm LIKE '%인플루엔자%') "
        "  AND week_no IS NOT NULL "
        "GROUP BY year, week_no "
        "ORDER BY year, week_no"
    )
    try:
        from simulation.database.analytics import duckdb_conn
        with duckdb_conn(read_only=True) as con:
            rows = con.execute(sql).fetchall()
    except Exception as e:  # pragma: no cover — defensive
        log.warning("weekly_disease DB query failed: %s", e)
        return np.array([]), []

    if not rows:
        return np.array([]), []

    from datetime import date, timedelta

    def _iso_week_monday(year: int, week: int) -> str:
        try:
            return date.fromisocalendar(int(year), int(week), 1).isoformat()
        except Exception:
            # Out-of-range ISO week — synthesize "Jan-1 + 7*(week-1)"
            return (date(int(year), 1, 1)
                    + timedelta(days=7 * (int(week) - 1))).isoformat()

    series = np.asarray([float(r[2] or 0.0) for r in rows], dtype=float)
    weeks = [_iso_week_monday(r[0], r[1]) for r in rows]

    if lookback_weeks and lookback_weeks > 0 and series.size > lookback_weeks:
        series = series[-lookback_weeks:]
        weeks = weeks[-lookback_weeks:]

    return series, weeks


def _load_seoul_weekly_from_csv(
    lookback_weeks: int,
) -> tuple[np.ndarray, list[str]]:
    """Fallback: read the training target y_true from predictions_*.csv.

    All 53 models share the same target, so any predictions file works;
    prefer Seasonal-Naive / NegBinGLM / ElasticNet as stable sources.
    """
    root = Path(__file__).resolve().parents[2]
    csv_dir = root / "simulation" / "results" / "csv"
    candidates = sorted(csv_dir.glob("predictions_*.csv"))
    if not candidates:
        # Live csv/ empty (likely mid-training). Fall back to latest backup.
        for bp in sorted(
            (root / "simulation" / "results").glob("backup_*"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        ):
            nested = list(bp.rglob("predictions_*.csv"))
            if nested:
                candidates = sorted(nested)
                log.info("using predictions from backup %s", bp.name)
                break
    if not candidates:
        log.warning("no predictions_*.csv found for Seoul city ILI series")
        return np.array([]), []

    preferred = [c for c in candidates if any(
        tag in c.stem for tag in ("Seasonal-Naive", "NegBinGLM", "ElasticNet")
    )]
    src = preferred[0] if preferred else candidates[0]

    try:
        import pandas as pd  # local import to keep module-load light
        df = pd.read_csv(src).sort_values("idx").drop_duplicates("idx")
    except Exception as e:
        log.warning("failed to read %s: %s", src, e)
        return np.array([]), []

    if "y_true" not in df.columns:
        log.warning("%s has no y_true column", src)
        return np.array([]), []

    y_true = np.asarray(df["y_true"].astype(float).values)
    if y_true.size == 0:
        return np.array([]), []

    from datetime import date, timedelta
    today = date.today()
    offset = (today.weekday() - 0) % 7
    last_monday = today - timedelta(days=offset)
    n = y_true.size
    weeks = [
        (last_monday - timedelta(days=7 * (n - 1 - i))).isoformat()
        for i in range(n)
    ]

    if lookback_weeks and lookback_weeks > 0 and y_true.size > lookback_weeks:
        y_true = y_true[-lookback_weeks:]
        weeks = weeks[-lookback_weeks:]

    return y_true, weeks


def _ears_c1(
    series: np.ndarray, *, baseline_weeks: int = 7, z_thr: float = 2.0
) -> tuple[list[bool], list[float]]:
    """EARS-C1 early aberration: flag if (x_t − mean_baseline) / sd_baseline > z_thr.

    Baseline = the 7 weeks immediately before t (as per CDC EARS-C1).
    The first ``baseline_weeks`` points cannot be scored; they're
    returned as non-flagged with NaN z.
    """
    n = series.size
    flags = [False] * n
    zs = [float("nan")] * n
    for t in range(baseline_weeks, n):
        base = series[t - baseline_weeks:t]
        mean = float(base.mean())
        sd = float(base.std(ddof=1)) if baseline_weeks > 1 else 0.0
        if sd <= 0:
            zs[t] = 0.0
            flags[t] = False
            continue
        z = (float(series[t]) - mean) / sd
        zs[t] = z
        flags[t] = z > z_thr
    return flags, zs


def _cusum(
    series: np.ndarray, *, k: float = 0.5, h: float = 5.0,
) -> tuple[list[bool], list[float]]:
    """One-sided upper CUSUM (standardised). Flag when ``S_t > h``.

    ``k`` is the slack parameter (half the shift we want to detect, in
    standard-deviation units). ``h`` is the decision threshold.
    """
    n = series.size
    if n < 3:
        return [False] * n, [0.0] * n
    # Standardise on a rolling basis (first 7-week initialisation)
    m = min(7, n)
    mu0 = float(series[:m].mean())
    sd0 = float(series[:m].std(ddof=1)) or 1.0
    S = 0.0
    flags: list[bool] = []
    out: list[float] = []
    for v in series:
        z = (float(v) - mu0) / sd0
        S = max(0.0, S + z - k)
        flags.append(S > h)
        out.append(S)
    return flags, out


__all__ = [
    "ToolSpec",
    "CallResult",
    "EpiMCPServer",
    "TOOL_SPECS",
    "TOOL_BY_NAME",
]
