"""Merge multiple factual_bench reports into one combined comparison.

Use case: reuse expensive cloud-CLI results from a prior run while swapping the
local roster. Run the new local models cheaply (--no-cli), then merge their
per_item with the CLOUD per_item from the prior report — the items match (same
kr_epi + KorMedMCQA n, deterministic), so the totals are directly comparable.
Recomputes the ranking + SCI stats (compare_backends) + renders figures.

Each --report may carry an optional ``|prefix`` filter on backend_id, e.g.
``old.json|cli:`` keeps only cloud backends from the old run.

Usage:
  python scripts/merge_bench_reports.py \
    --report simulation/results/llm_compare_combined/factual_report.json|cli: \
    --report simulation/results/llm_compare_local/factual_report.json \
    --out simulation/results/llm_compare_latest
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simulation.llm_compare.comparison import compare_backends  # noqa: E402
from simulation.llm_compare.visualize import make_figures  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Merge factual_bench reports")
    ap.add_argument("--report", action="append", required=True,
                    help="PATH or PATH|backend_prefix (e.g. report.json|cli:)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    merged: list = []
    meta: dict = {}
    sources: list = []
    for spec in args.report:
        path, _, prefix = spec.partition("|")
        rep = json.loads(Path(path).read_text(encoding="utf-8"))
        sources.append(Path(path).name)
        tier = {b["backend_id"]: b.get("tier", "") for b in rep.get("backends", [])}
        for pi in rep["per_item"]:
            bid = pi["backend_id"]
            if prefix and not bid.startswith(prefix):
                continue
            if bid in meta and bid not in {p["backend_id"] for p in merged}:
                pass  # first time seeing across reports is fine
            merged.append(pi)
            meta.setdefault(bid, {"backend_id": bid, "model": pi.get("model", ""),
                                  "tier": tier.get(bid, "")})

    # recompute ranking (exclude errored) + unavailable
    ok: dict = {}
    errs: dict = {}
    for pi in merged:
        bid = pi["backend_id"]
        if pi.get("error"):
            errs[bid] = errs.get(bid, 0) + 1
        else:
            ok.setdefault(bid, []).append(pi["total"])
    ranking = sorted(
        [{"backend_id": b, "accuracy": round(sum(v) / len(v), 4), "n_items": len(v),
          "n_errors": errs.get(b, 0), "tier": meta[b]["tier"]} for b, v in ok.items()],
        key=lambda r: -r["accuracy"])
    unavailable = [{"backend_id": b, "tier": meta[b]["tier"], "n_errors": errs[b],
                    "reason": "all responses errored"} for b in errs if b not in ok]
    scored_ok = [pi for pi in merged if not pi.get("error")]
    n_items = max((len(v) for v in ok.values()), default=0)
    item_ids = {pi["item_id"] for pi in merged}
    n_kr = sum(1 for i in item_ids if not i.startswith("KMQ"))
    n_mq = sum(1 for i in item_ids if i.startswith("KMQ"))

    report = {
        "generated_at": "merged", "env": {"api_keys_present": {}},
        "backends": list(meta.values()),
        "n_items": n_items, "n_kr_epi": n_kr, "n_kormedmcqa": n_mq,
        "repetitions": 1, "ranking": ranking, "unavailable": unavailable,
        "statistical_comparison": compare_backends(scored_ok) if len(ok) >= 2 else {},
        "repro_manifest": {"config_sha256": "merged",
                           "merged_from": sources,
                           "confound_control": "cloud=reused-CLI(as-deployed) + local=fresh"},
        "per_item": merged,
    }
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "factual_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    make_figures(out / "factual_report.json", out / "figures")
    print(f"merged {len(meta)} models (kr_epi {n_kr} + KorMedMCQA {n_mq}) → {out}")
    for i, r in enumerate(ranking, 1):
        print(f"  {i}. {r['backend_id'].split('@')[0]:48s} {r['accuracy']:.4f} ({r['tier']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
