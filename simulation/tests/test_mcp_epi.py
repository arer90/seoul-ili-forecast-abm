"""
simulation.tests.test_mcp_epi
=============================
Unit tests for Stage 6a — ARIA MCP server foundation.

Covers three layers independently so they can fail locally without
cascading into each other:

1. ``sql_guard.validate_read_only`` — accept / reject table.
2. ``EpiMCPServer.list_tools`` — schema shape, 11 tools, ``_meta.wired``.
3. ``EpiMCPServer.call_tool`` — dispatcher behaviour for:
     - unknown tool → ``CallResult.is_error=True``
     - sql guard violation caught
     - graceful stubs return ``status="not_available"`` when artifacts
       are absent
     - scenario_run end-to-end using synthetic params (no DB needed)

4. ``mcp_stdio.StdioServer.handle_line`` — pure-string dispatch covers
   the JSON-RPC envelope (initialize, tools/list, tools/call, bad JSON,
   unknown method, notifications).

These tests do **not** require DuckDB, torch, or the real ``epi_real_seoul.db``;
every path that would touch them either uses a stub or skips cleanly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from simulation.server import (
    CallResult,
    EpiMCPServer,
    TOOL_BY_NAME,
    TOOL_SPECS,
    SqlGuardError,
    ToolSpec,
    validate_read_only,
)
from simulation.server.mcp_stdio import (
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    StdioServer,
    _ShutdownSignal,
)


# ══════════════════════════════════════════════════════════════════════
# 1. SQL guard
# ══════════════════════════════════════════════════════════════════════
class TestSqlGuard:
    def test_accepts_plain_select(self):
        r = validate_read_only("SELECT * FROM weekly_disease LIMIT 5")
        assert r.ok
        assert r.leading_keyword == "SELECT"

    def test_accepts_with_cte(self):
        sql = """
            WITH top AS (
                SELECT gu_nm, SUM(cases) AS c
                FROM epi.weekly_disease
                GROUP BY gu_nm
            )
            SELECT * FROM top ORDER BY c DESC
        """
        assert validate_read_only(sql).ok

    def test_accepts_explain_and_pragma(self):
        assert validate_read_only("EXPLAIN SELECT 1").ok
        assert validate_read_only("PRAGMA table_info(weekly_disease)").ok
        assert validate_read_only("DESCRIBE weekly_disease").ok
        assert validate_read_only("SUMMARIZE weekly_disease").ok

    def test_rejects_empty(self):
        assert not validate_read_only("").ok
        assert not validate_read_only("   \n").ok

    def test_rejects_insert_update_delete(self):
        for sql in [
            "INSERT INTO t VALUES (1)",
            "UPDATE t SET a=1",
            "DELETE FROM t",
            "DROP TABLE t",
            "CREATE TABLE foo(x)",
            "ALTER TABLE t ADD COLUMN x",
            "ATTACH DATABASE 'x.db' AS x",
        ]:
            assert not validate_read_only(sql).ok, f"should reject: {sql!r}"

    def test_rejects_forbidden_verb_mid_query(self):
        r = validate_read_only(
            "SELECT * FROM t WHERE 1=1; DROP TABLE t"
        )
        assert not r.ok

    def test_rejects_stacked_statements(self):
        r = validate_read_only("SELECT 1; SELECT 2")
        assert not r.ok
        assert "multiple statements" in r.reason.lower()

    def test_trailing_semicolon_ok(self):
        assert validate_read_only("SELECT 1;").ok
        assert validate_read_only("SELECT 1  ;  ").ok

    def test_keyword_inside_string_does_not_spoof(self):
        # DELETE inside a string literal must not trigger the guard.
        sql = "SELECT 'DELETE me' AS label FROM t"
        assert validate_read_only(sql).ok

    def test_keyword_inside_identifier_does_not_spoof(self):
        # Double-quoted identifier with a forbidden word inside.
        sql = 'SELECT "drop_zone" FROM t'
        assert validate_read_only(sql).ok

    def test_keyword_inside_line_comment_does_not_spoof(self):
        sql = "SELECT 1 -- DROP TABLE t"
        assert validate_read_only(sql).ok

    def test_keyword_inside_block_comment_does_not_spoof(self):
        sql = "SELECT 1 /* DROP TABLE t */"
        assert validate_read_only(sql).ok

    def test_forbidden_function_prefix_rejected(self):
        assert not validate_read_only(
            "SELECT * FROM read_csv_auto('x.csv')"
        ).ok
        assert not validate_read_only(
            "SELECT * FROM read_parquet('x.pq')"
        ).ok
        assert not validate_read_only(
            "SELECT pragma_user_version()"
        ).ok

    def test_raise_if_bad(self):
        with pytest.raises(SqlGuardError):
            validate_read_only("DROP TABLE t").raise_if_bad()
        # OK case: no raise.
        validate_read_only("SELECT 1").raise_if_bad()


# ══════════════════════════════════════════════════════════════════════
# 2. Schema
# ══════════════════════════════════════════════════════════════════════
class TestToolSpecs:
    def test_tool_count_is_ten(self):
        assert len(TOOL_SPECS) == 12   # + epi.coupled_forward (ABM→ARIA coupling)

    def test_tool_names_unique(self):
        names = [t.name for t in TOOL_SPECS]
        assert len(set(names)) == len(names)

    def test_tool_names_are_namespaced(self):
        for t in TOOL_SPECS:
            assert t.name.startswith("epi."), t.name

    def test_tool_by_name_maps_every_spec(self):
        for t in TOOL_SPECS:
            assert TOOL_BY_NAME[t.name] is t

    @pytest.mark.parametrize("spec", TOOL_SPECS, ids=[t.name for t in TOOL_SPECS])
    def test_spec_mcp_shape(self, spec: ToolSpec):
        s = spec.to_mcp()
        for key in ("name", "title", "description", "inputSchema", "_meta"):
            assert key in s, f"{spec.name} missing {key}"
        assert s["inputSchema"]["type"] == "object"
        assert "properties" in s["inputSchema"]
        assert isinstance(s["_meta"]["wired"], bool)

    def test_wired_flag_matches_intent(self, server: EpiMCPServer):
        # TOOL_SPECS hardcode wired=True for every tool; the live, artifact-aware
        # flag is applied by list_tools via _probe_wired (see mcp_epi). The
        # always-on DB/code tools (and static-citation literature_rag +
        # international_compare) report wired regardless of artifact presence;
        # the artifact-gated tools (forecast/model_compare/shap_features/
        # lead_time_analysis) are environment-dependent and not asserted here.
        wired = {t["name"] for t in server.list_tools() if t["_meta"]["wired"]}
        assert {
            "epi.query_db",
            "epi.rt_estimate",
            "epi.outbreak_detect",
            "epi.validity_check",
            "epi.scenario_run",
            "epi.literature_rag",
            "epi.international_compare",
        } <= wired


# ══════════════════════════════════════════════════════════════════════
# 3. Dispatcher behaviour
# ══════════════════════════════════════════════════════════════════════
@pytest.fixture(autouse=True)
def _no_live_heavy_features(monkeypatch):
    # epi.forecast (and model_compare / shap_features) resolve champion .pt from
    # ``models/`` (cwd-relative, present even in CI) and build the full enriched
    # feature matrix from the 80M-row DB — slow enough to look like a hang. These
    # unit tests exercise the MCP dispatch + the graceful fallback contract, not
    # the live model, so force the heavy build unavailable; the handlers then fall
    # through to their deterministic not_available stub. Live forecasting is
    # covered separately as an integration test, not here.
    def _raise(self):
        raise RuntimeError("live enriched features disabled in unit tests")
    monkeypatch.setattr(EpiMCPServer, "_get_enriched_features", _raise)


@pytest.fixture
def server(tmp_path: Path) -> EpiMCPServer:
    # Point artifacts_dir somewhere empty so graceful stubs report
    # not_available instead of accidentally reading stale manifests.
    return EpiMCPServer(artifacts_dir=tmp_path)


class TestDispatcher:
    def test_list_tools_returns_all_ten(self, server: EpiMCPServer):
        tools = server.list_tools()
        assert len(tools) == 12

    def test_unknown_tool_is_error(self, server: EpiMCPServer):
        r = server.call_tool("epi.nope", {})
        assert r.is_error
        assert "unknown tool" in json.dumps(r.content)

    def test_query_db_catches_sql_guard(self, server: EpiMCPServer):
        r = server.call_tool("epi.query_db", {"sql": "DROP TABLE t"})
        assert r.is_error
        assert r.content["error"] == "sql_guard"

    def test_query_db_catches_empty_sql(self, server: EpiMCPServer):
        r = server.call_tool("epi.query_db", {"sql": ""})
        assert r.is_error

    def test_forecast_graceful_stub(self, server: EpiMCPServer):
        r = server.call_tool(
            "epi.forecast", {"gu": "강남구", "horizon": 4}
        )
        assert not r.is_error
        assert r.content["status"] == "not_available"
        assert "expected_artifact" in r.content
        assert r.content["request_echo"]["gu"] == "강남구"

    def test_model_compare_graceful_stub(self, server: EpiMCPServer):
        r = server.call_tool(
            "epi.model_compare",
            {"models": ["XGBoost", "LSTM"], "metric": "mae"},
        )
        assert not r.is_error
        assert r.content["status"] == "not_available"

    def test_shap_features_graceful_stub(self, server: EpiMCPServer):
        r = server.call_tool(
            "epi.shap_features",
            {"gu": "강남구", "week": "2024-01-01"},
        )
        assert not r.is_error
        assert r.content["status"] == "not_available"

    def test_lead_time_graceful_stub(self, server: EpiMCPServer):
        r = server.call_tool(
            "epi.lead_time_analysis", {"model": "XGBoost"},
        )
        assert not r.is_error
        assert r.content["status"] == "not_available"

    def test_literature_rag_graceful_stub(self, server: EpiMCPServer):
        r = server.call_tool("epi.literature_rag", {"query": "VE"})
        assert not r.is_error
        # literature_rag was upgraded from a bare stub to a static-citation
        # fallback (returns curated references when the RAG corpus is absent).
        assert r.content["status"] == "static_fallback"

    def test_scenario_run_unknown_scenario(self, server: EpiMCPServer):
        r = server.call_tool(
            "epi.scenario_run", {"scenario": "does_not_exist"},
        )
        assert r.is_error
        assert "unknown scenario" in r.content["error"]

    def test_scenario_run_synthetic(self, server: EpiMCPServer):
        # use_db=False → synthetic 25-gu uniform mixing, no DB touch.
        r = server.call_tool(
            "epi.scenario_run",
            {
                "scenario": "baseline",
                "days": 30,
                "use_db": False,
                "seed_infected": 5.0,
            },
        )
        assert not r.is_error, r.content
        assert r.content["status"] == "ok"
        assert r.content["scenario"] == "baseline"
        assert isinstance(r.content["peak_I"], float)
        assert isinstance(r.content["final_D"], float)
        assert len(r.content["I_city_series"]) == len(r.content["D_city_series"])
        assert r.content["days"] == 30

    def test_validity_check_empty_ok(self, server: EpiMCPServer):
        # No params, no predictions → should still return a structured
        # result (pass or warn) rather than erroring.
        r = server.call_tool("epi.validity_check", {})
        assert not r.is_error
        assert "status" in r.content

    def test_call_result_mcp_shape(self):
        r = CallResult(content={"hello": "world"})
        m = r.to_mcp()
        assert isinstance(m["content"], list)
        assert m["content"][0]["type"] == "text"
        parsed = json.loads(m["content"][0]["text"])
        assert parsed == {"hello": "world"}
        assert m["isError"] is False

    def test_call_meta_records_tool_and_elapsed(self, server: EpiMCPServer):
        r = server.call_tool("epi.forecast", {"gu": "강남구"})
        assert r.meta["tool"] == "epi.forecast"
        assert "elapsed_ms" in r.meta


# ══════════════════════════════════════════════════════════════════════
# 4. Stdio JSON-RPC transport (pure-string dispatch)
# ══════════════════════════════════════════════════════════════════════
@pytest.fixture
def stdio(tmp_path: Path) -> StdioServer:
    return StdioServer.create(EpiMCPServer(artifacts_dir=tmp_path))


class TestStdioDispatcher:
    def test_initialize_ok(self, stdio: StdioServer):
        req = {
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"clientInfo": {"name": "pytest", "version": "0"}},
        }
        resp = stdio.handle_line(json.dumps(req))
        assert resp["id"] == 1
        assert "result" in resp
        assert resp["result"]["serverInfo"]["name"] == "epi-mcp"
        assert "capabilities" in resp["result"]

    def test_tools_list_ok(self, stdio: StdioServer):
        req = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}
        resp = stdio.handle_line(json.dumps(req))
        assert "result" in resp
        assert len(resp["result"]["tools"]) == 12

    def test_tools_call_ok(self, stdio: StdioServer):
        req = {
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {
                "name": "epi.forecast",
                "arguments": {"gu": "강남구"},
            },
        }
        resp = stdio.handle_line(json.dumps(req))
        assert "result" in resp
        assert resp["result"]["isError"] is False
        parsed = json.loads(resp["result"]["content"][0]["text"])
        assert parsed["status"] == "not_available"

    def test_bad_json_yields_parse_error(self, stdio: StdioServer):
        resp = stdio.handle_line("{not json")
        assert resp["error"]["code"] == PARSE_ERROR
        assert resp["id"] is None

    def test_bad_envelope_yields_invalid_request(self, stdio: StdioServer):
        resp = stdio.handle_line(json.dumps({"hello": "world"}))
        assert resp["error"]["code"] == INVALID_REQUEST

    def test_unknown_method_yields_not_found(self, stdio: StdioServer):
        req = {"jsonrpc": "2.0", "id": 99, "method": "does/not/exist"}
        resp = stdio.handle_line(json.dumps(req))
        assert resp["error"]["code"] == METHOD_NOT_FOUND
        assert "known_methods" in resp["error"]["data"]

    def test_notification_has_no_response(self, stdio: StdioServer):
        # No id field → notification. Must return None.
        req = {"jsonrpc": "2.0", "method": "notifications/initialized"}
        resp = stdio.handle_line(json.dumps(req))
        assert resp is None

    def test_empty_line_returns_none(self, stdio: StdioServer):
        assert stdio.handle_line("   ") is None
        assert stdio.handle_line("\n") is None

    def test_shutdown_raises_signal(self, stdio: StdioServer):
        req = {"jsonrpc": "2.0", "id": 4, "method": "shutdown"}
        with pytest.raises(_ShutdownSignal):
            stdio.handle_line(json.dumps(req))

    def test_tools_call_invalid_params(self, stdio: StdioServer):
        req = {
            "jsonrpc": "2.0", "id": 5, "method": "tools/call",
            "params": {"name": 42},  # not a string
        }
        resp = stdio.handle_line(json.dumps(req))
        assert "error" in resp
        assert "string" in resp["error"]["message"].lower()

    def test_ping_ok(self, stdio: StdioServer):
        req = {"jsonrpc": "2.0", "id": 6, "method": "ping"}
        resp = stdio.handle_line(json.dumps(req))
        assert resp["result"] == {}


def test_coupled_forward_registered_and_allowlisted():
    """epi.coupled_forward wires the forecast-anchored + EnKF-coupled behavioural ABM
    (with district resolution) to ARIA — the ABM→ARIA hand-off the integration audit
    found broken (ARIA previously saw only the deterministic metapop). Structural guard
    (fast); the full coupled run is verified separately (~30s enkf)."""
    assert "epi.coupled_forward" in TOOL_BY_NAME
    spec = TOOL_BY_NAME["epi.coupled_forward"]
    assert spec.wired
    props = spec.input_schema["properties"]
    assert {"n_agents", "n_seeds"} <= set(props)
    from simulation.llm_compare.specialists import SPECIALISTS
    allow = {s.name: s.tools for s in SPECIALISTS}
    # consumed by the two specialists that need the coupled forward + district signal
    assert "epi.coupled_forward" in allow["simulation_experiment"]
    assert "epi.coupled_forward" in allow["spatial_transmission"]
    # bounded-tool boundary preserved — unrelated specialists cannot reach it
    assert "epi.coupled_forward" not in allow["deep_evidence_research"]
    assert "epi.coupled_forward" not in allow["statistical_verification"]
