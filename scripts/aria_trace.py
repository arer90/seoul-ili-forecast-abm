"""ARIA end-to-end grounded+audited trace — Pillar 3 통합 입증 (④).

논문 중심주장(통합 시스템)의 Pillar 1+2→3 배선 증거: LLM 자문층(ARIA/MCP)이 실 DB·forecast·ABM 산출을
**grounded(provenance db_vintage_ts) + freshness(LIVE/STALE) 봉투 + 감사 ledger(mcp_audit.jsonl)** 로
소비함을 한 trace 로 산출. CallResult.content(dict) 에서 정확히 추출(to_mcp 아님).

산출 → simulation/results/abm_v1/aria_trace.json + 콘솔 요약. server init = DB connectivity ~30s.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logging.disable(logging.WARNING)   # 서버 init 의 pipeline-replay 로그 억제

from simulation.server.mcp_epi import EpiMCPServer  # noqa: E402

ARTIFACTS = Path("simulation/results")
# 3기둥 횡단: 실 DB(Rt·경보) · forecast(P1) · scenario(P2 ABM) · validity · RAG(근거)
TOOL_CALLS = [
    ("epi.rt_estimate", {"gu": "seoul_city"}),       # 실 surveillance DB → Rt
    ("epi.outbreak_detect", {"gu": "seoul_city"}),   # 실 DB → 발생 경보
    ("epi.forecast", {}),                            # Pillar 1 forecast 산출
    ("epi.scenario_run", {"scenario": "vaccination_campaign"}),  # Pillar 2 ABM(SEIR-V-D)
    ("epi.validity_check", {}),                      # 검증 게이트
    ("epi.literature_rag", {"query": "post-COVID influenza rebound Seoul vaccination"}),
]


def main():
    srv = EpiMCPServer(artifacts_dir=ARTIFACTS)
    wired = srv._probe_wired()
    n_wired = sum(1 for v in wired.values() if v) if isinstance(wired, dict) else 0
    print(f"[ARIA] EpiMCPServer — wired 도구 {n_wired}/{len(wired)}")

    trace = []
    for name, args in TOOL_CALLS:
        try:
            r = srv.call_tool(name, args)
            c = r.content if isinstance(r.content, dict) else {"_raw": str(r.content)[:160]}
            prov = c.get("provenance") if isinstance(c, dict) else None
            rec = {
                "tool": name,
                "is_error": bool(getattr(r, "is_error", False)),
                "status": c.get("status") if isinstance(c, dict) else None,
                "freshness": c.get("freshness") if isinstance(c, dict) else None,
                "db_vintage_ts": (prov or {}).get("db_vintage_ts"),
                "server_version": (prov or {}).get("server_version"),
                "grounded": bool(prov is not None),
                "result_keys": [k for k in c if k not in
                                ("provenance", "freshness", "status", "freshness_detail")][:5]
                if isinstance(c, dict) else [],
            }
        except Exception as e:
            rec = {"tool": name, "is_error": True, "exception": f"{type(e).__name__}: {str(e)[:120]}"}
        trace.append(rec)
        fr = rec.get("freshness") or "-"
        g = "grounded✓" if rec.get("grounded") else ("err" if rec.get("is_error") else "no-prov")
        print(f"  [{name:24s}] status={str(rec.get('status')):10s} freshness={fr:5s} {g}  "
              f"keys={rec.get('result_keys', rec.get('exception',''))}")

    # 감사 ledger (D8): 위 호출이 mcp_audit.jsonl 에 기록됐나
    ledger = ARTIFACTS / "mcp_audit.jsonl"
    audit_n = 0
    audit_tail = []
    if ledger.exists():
        lines = ledger.read_text(encoding="utf-8").splitlines()
        audit_n = len(lines)
        for ln in lines[-len(TOOL_CALLS):]:
            try:
                e = json.loads(ln)
                audit_tail.append({k: e.get(k) for k in ("tool", "ts", "ok", "latency_ms") if k in e})
            except Exception:
                pass
    print(f"\n[ARIA] 감사 ledger {ledger.name}: {audit_n} 기록 (D8 accountability), 최근 {len(audit_tail)}건:")
    for a in audit_tail:
        print(f"    {a}")

    n_grounded = sum(1 for t in trace if t.get("grounded"))
    summary = {
        "setup": "ARIA(EpiMCPServer) end-to-end trace — Pillar 1+2→3 통합 (grounded+freshness+audit ledger)",
        "n_tools_wired": n_wired, "n_tools_total": len(wired) if isinstance(wired, dict) else None,
        "n_calls": len(trace), "n_grounded": n_grounded,
        "audit_ledger_entries": audit_n,
        "trace": trace, "audit_tail": audit_tail,
    }
    out = ARTIFACTS / "abm_v1" / "aria_trace.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=str)
    print(f"\n[ARIA] {n_grounded}/{len(trace)} 호출 grounded(provenance 봉투), 감사 {audit_n}건 → {out}")


if __name__ == "__main__":
    main()
