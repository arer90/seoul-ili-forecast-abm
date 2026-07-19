"""
simulation.scripts.aria_add_cli_backends
==========================================
Add CLI-tier LLM backends (Codex / Gemini) to the EXISTING ARIA multi-LLM
grounding evaluation, reusing the shipped test set + scoring verbatim, and merge
the new rows into ``aria_grounding_multi_llm.json`` (preserving the 6 backends
already measured: Claude CLI + 5 Ollama).

WHY a thin driver (not a new harness):
  The grounding test set (two REAL thesis contexts with gold numeric facts) and
  the scoring (``grounding_eval`` numeric axis + ``self_ask_grounding`` Self-Ask
  axis) are the SSOT in ``simulation.llm_compare.aria_grounding``. This driver
  only adds backend objects and re-runs the SAME two functions over them, then
  appends rows in the exact ``comparison_table`` schema the figure reads. No
  metric is reimplemented — fabrication-proof by construction.

Test set (identical to the shipped 6-backend run, read-only from disk):
  • P4_identifiability  (abm_forward_validation/result.json, 8 gold facts)
  • ABM_fit             (abm_real_validation/result.json,     5 gold facts)
  → 2 contexts × {numeric grounding, Self-Ask} = 4 live LLM calls per backend.
  The test set is SMALL, so every backend is run on the FULL set (no subset).

Quota policy (Gemini):
  Gemini free quota is intermittent. ``--with-gemini`` probes ``gemini -p PONG``
  first; on quota/availability failure the backend is SKIPPED and the skip is
  recorded in the payload under ``added_backends_skipped`` (honest future-work
  note — never a fabricated row).

Run:
  .venv/bin/python -m simulation.scripts.aria_add_cli_backends            # Codex only
  .venv/bin/python -m simulation.scripts.aria_add_cli_backends --with-gemini
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from simulation.llm_compare.aria_grounding import grounding_eval, self_ask_grounding
from simulation.llm_compare.backends import CodexCliBackend, GeminiCliBackend
from simulation.scripts.aria_sci_common import real_contexts

DATA_JSON = Path("simulation/results/aria_grounding_multi_llm.json")

# Display labels for the added CLI backends (task-specified for Codex).
CLI_LABELS = {
    "cli:codex:codex-default": "Codex (OpenAI GPT)",
    "cli:gemini:gemini-default": "Gemini CLI",
}


def _row_for_backend(bid: str, ng: dict, sa: dict) -> dict:
    """Assemble one ``comparison_table`` row in the EXACT shipped schema.

    Args:
        bid: backend_id (key into both per_backend dicts).
        ng: ``grounding_eval`` output (numeric axis).
        sa: ``self_ask_grounding`` output (Self-Ask axis).

    Returns:
        A dict with the same keys the existing 6 rows use, so the figure reader
        and the multipass-CI consumer treat the new backend identically.
    """
    n = ng["per_backend"][bid]
    s = sa["per_backend"][bid]
    return {
        "backend_id": bid,
        "label": CLI_LABELS.get(bid, bid),
        "tier": "cli",
        "numeric_fact_recall": n["fact_recall"],
        "numeric_faithfulness": n["faithfulness"],
        "numeric_n_spurious": n["n_spurious_total"],
        "numeric_n_errors": n["n_errors"],
        "selfask_subq_recall": s["subq_fact_recall"],
        "selfask_faithfulness": s["faithfulness"],
        "selfask_mean_n_subq": s["mean_n_subq"],
        "selfask_n_errors": s["n_errors"],
    }


def _gemini_quota_ok(timeout_s: int = 90) -> tuple[bool, str]:
    """Probe Gemini free quota with ``gemini -p PONG`` (task step 3).

    Returns:
        ``(ok, note)`` — ok True only if the CLI returns a non-empty answer
        WITHOUT a quota/availability error. The note is a short human string for
        the payload's skip record.

    Side effects: one ``gemini`` subprocess call. Never raises.
    """
    try:
        r = subprocess.run(["gemini", "-p", "PONG"], capture_output=True,
                           text=True, timeout=timeout_s, stdin=subprocess.DEVNULL)
    except Exception as e:  # noqa: BLE001
        return False, f"gemini probe failed: {e}"
    combined = ((r.stdout or "") + " " + (r.stderr or "")).lower()
    quota_markers = ("exhausted your daily quota", "quota", "429", "resource_exhausted",
                     "503", "unavailable", "high demand")
    if any(m in combined for m in quota_markers):
        # extract a short reason
        reason = "429/quota or 503 unavailable"
        if "exhausted your daily quota" in combined:
            reason = "TerminalQuotaError: daily quota exhausted"
        return False, f"gemini quota unavailable ({reason}) — skipped (future work)"
    if r.returncode != 0 or not (r.stdout or "").strip():
        return False, f"gemini probe rc={r.returncode}/empty — skipped"
    return True, "gemini quota OK"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Add Codex/Gemini CLI backends to ARIA multi-LLM eval")
    ap.add_argument("--with-gemini", action="store_true",
                    help="also add Gemini CLI (quota-probed first; skipped on 429)")
    ap.add_argument("--data", default=str(DATA_JSON))
    args = ap.parse_args(argv)

    data_path = Path(args.data)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    existing_ids = {r["backend_id"] for r in data["comparison_table"]}
    contexts = real_contexts()
    print(f"[test set] {len(contexts)} contexts, "
          f"{sum(len(c['facts']) for c in contexts)} gold facts (FULL set, no subset)")
    for c in contexts:
        print(f"    {c['id']:18s} {c['source']:42s} n_facts={len(c['facts'])}")

    # ── Build the list of CLI backends to ADD (Codex always; Gemini if --with-gemini & quota).
    to_add: list = [CodexCliBackend()]
    skipped: list[dict] = []
    if args.with_gemini:
        ok, note = _gemini_quota_ok()
        print(f"[gemini] {note}")
        if ok:
            to_add.append(GeminiCliBackend())
        else:
            skipped.append({"backend_id": "cli:gemini:gemini-default",
                            "label": CLI_LABELS["cli:gemini:gemini-default"],
                            "reason": note})
    else:
        skipped.append({"backend_id": "cli:gemini:gemini-default",
                        "label": CLI_LABELS["cli:gemini:gemini-default"],
                        "reason": "not requested (--with-gemini not passed)"})

    # Drop any backend already present (idempotent re-run replaces the row below).
    to_add = [b for b in to_add if b.is_available()]
    if not to_add:
        print("[abort] no CLI backend available to add (codex not in PATH?)")
        return 1
    print(f"[adding] {', '.join(b.backend_id for b in to_add)}")

    # ── Run the SHIPPED scoring over the new backends (same functions as the 6).
    t0 = time.time()
    ng = grounding_eval(to_add, contexts)
    sa = self_ask_grounding(to_add, contexts)
    elapsed = round(time.time() - t0, 1)

    new_rows = []
    for b in to_add:
        bid = b.backend_id
        row = _row_for_backend(bid, ng, sa)
        new_rows.append(row)
        # merge into the per_backend score blocks too (keep JSON self-consistent)
        data["numeric_grounding"]["per_backend"][bid] = ng["per_backend"][bid]
        data["self_ask"]["per_backend"][bid] = sa["per_backend"][bid]
        if bid not in data["backends"]:
            data["backends"].append(bid)
        print(f"  {row['label']:22s} num_recall={row['numeric_fact_recall']} "
              f"num_faith={row['numeric_faithfulness']} "
              f"sa_recall={row['selfask_subq_recall']} "
              f"sa_faith={row['selfask_faithfulness']} "
              f"mean_subq={row['selfask_mean_n_subq']} "
              f"(num_err={row['numeric_n_errors']}, sa_err={row['selfask_n_errors']})")

    # ── Merge comparison_table: replace existing CLI rows with same id, else append.
    new_by_id = {r["backend_id"] for r in new_rows}
    merged = [r for r in data["comparison_table"] if r["backend_id"] not in new_by_id]
    merged.extend(new_rows)
    data["comparison_table"] = merged

    # ── Honest provenance bookkeeping.
    data.setdefault("added_backends", [])
    for r in new_rows:
        data["added_backends"] = [x for x in data["added_backends"]
                                  if x.get("backend_id") != r["backend_id"]]
        data["added_backends"].append({
            "backend_id": r["backend_id"], "label": r["label"],
            "added_on": time.strftime("%Y-%m-%d"),
            "method": "grounding_eval + self_ask_grounding (same SSOT scoring); "
                      "live CLI single pass, codex reasoning_effort=low",
            "n_calls": len(contexts) * 2,
        })
    data["added_backends_skipped"] = skipped
    data["added_elapsed_s"] = elapsed
    data["description"] = (data.get("description", "")
                           + " | EXTENDED: CLI backends "
                           + ", ".join(sorted(new_by_id))
                           + " added (same test set + scoring).")

    data_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[OK] merged {len(new_rows)} backend(s) into {data_path} "
          f"(now {len(data['comparison_table'])} backends total, elapsed {elapsed}s)")
    if skipped:
        for s in skipped:
            print(f"[skipped] {s['label']}: {s['reason']}")
    print(f"[preserved] original 6: {sorted(existing_ids)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
