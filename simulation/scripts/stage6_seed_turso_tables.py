"""Stage 6b — Seed the ARIA edge tables inside ``epi_real_seoul.db``.

The export script ``web/scripts/export-turso.py`` dumps 8 tables for
Turso. Four of them don't exist yet:
  - ``forecast_runs`` / ``forecast_points`` — from ``stage3_forecasts.json``
  - ``rt_history``                          — from R ``06_rt_epiestim.csv``
  - ``shap_delta_snapshots``                — from phase8 MI ranking
  - ``scenario_runs``                       — from ``sim_runs/_manifest.json``

This script creates the DDL (idempotent) and populates them with data from
the post-E + Stage 5 artifacts. Running ``export-turso.py`` afterwards
produces a ``turso_seed.sql`` with real content.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from simulation.database import safe_connect, bulk_insert

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger("stage6.seed")

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "simulation" / "results"

# ── Schemas (idempotent DDL) ─────────────────────────────────────────────
DDL = {
    "forecast_runs": """
        CREATE TABLE IF NOT EXISTS forecast_runs (
            run_id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            generated_at TEXT NOT NULL,
            gu TEXT NOT NULL,
            horizon INTEGER NOT NULL,
            n_points INTEGER,
            wis REAL,
            wis_source TEXT,
            crps_gaussian REAL,
            pi_coverage_95 REAL
        );
    """,
    "forecast_points": """
        CREATE TABLE IF NOT EXISTS forecast_points (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            week_start TEXT,
            week_idx INTEGER,
            y_true REAL,
            y_pred REAL,
            pi_lo_95 REAL,
            pi_hi_95 REAL,
            pi_lo_90 REAL,
            pi_hi_90 REAL,
            FOREIGN KEY (run_id) REFERENCES forecast_runs(run_id)
        );
    """,
    "rt_history": """
        CREATE TABLE IF NOT EXISTS rt_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            gu TEXT NOT NULL,
            week_start TEXT NOT NULL,
            method TEXT NOT NULL,
            rt_mean REAL,
            rt_q025 REAL,
            rt_q975 REAL,
            si_mean REAL,
            si_sd REAL
        );
    """,
    "shap_delta_snapshots": """
        CREATE TABLE IF NOT EXISTS shap_delta_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_ts TEXT,
            model TEXT NOT NULL,
            feature TEXT NOT NULL,
            rank INTEGER,
            mi_score REAL,
            source TEXT
        );
    """,
    "scenario_runs": """
        CREATE TABLE IF NOT EXISTS scenario_runs (
            scenario_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            generated_at TEXT,
            days_run INTEGER,
            peak_day INTEGER,
            peak_i_city REAL,
            attack_rate_pct REAL,
            cumulative_deaths REAL,
            n_interventions INTEGER,
            epi_validity_ok INTEGER,
            compartment_conservation_err REAL,
            interventions_json TEXT
        );
    """,
}


def create_tables(con):
    cur = con.cursor()
    for tbl, ddl in DDL.items():
        cur.execute(ddl)
        log.info("DDL ok: %s", tbl)
    con.commit()


# ── Population ──────────────────────────────────────────────────────────
def seed_forecast_runs_and_points(con):
    """From stage3_forecasts.json — 66 forecast runs, ~4500 forecast points."""
    path = RES / "stage3_forecasts.json"
    if not path.exists():
        log.warning("skip forecast_* seed: %s missing", path)
        return
    data = json.load(path.open(encoding="utf-8"))
    generated_at = data.get("generated_at", pd.Timestamp.now().isoformat())

    runs_rows, point_rows = [], []
    for model, f in data["forecasts"].items():
        run_id = f"stage3/{model}/h{f['horizon']}/seoul_city"
        runs_rows.append({
            "run_id": run_id,
            "model_id": model,
            "generated_at": generated_at,
            "gu": f.get("gu", "seoul_city"),
            "horizon": int(f.get("horizon", 1)),
            "n_points": int(f.get("n_points", 0)),
            "wis": f.get("wis"),
            "wis_source": f.get("wis_source"),
            "crps_gaussian": f.get("crps_gaussian"),
            "pi_coverage_95": f.get("pi_coverage_95"),
        })
        for rec in f["series"]:
            point_rows.append({
                "run_id": run_id,
                "week_start": rec.get("week_start"),
                "week_idx": rec.get("week_idx"),
                "y_true": rec.get("y_true"),
                "y_pred": rec.get("y_pred"),
                "pi_lo_95": rec.get("pi_lo_95"),
                "pi_hi_95": rec.get("pi_hi_95"),
                "pi_lo_90": rec.get("pi_lo_90"),
                "pi_hi_90": rec.get("pi_hi_90"),
            })

    cur = con.cursor()
    # FK: forecast_points → forecast_runs, delete points first.
    cur.execute("DELETE FROM forecast_points WHERE run_id LIKE 'stage3/%'")
    cur.execute("DELETE FROM forecast_runs WHERE run_id LIKE 'stage3/%'")
    con.commit()
    bulk_insert("forecast_runs", runs_rows, conn=con, chunk_size=200)
    bulk_insert("forecast_points", point_rows, conn=con, chunk_size=2000)
    log.info("seeded forecast_runs=%d, forecast_points=%d",
             len(runs_rows), len(point_rows))


def seed_rt_history(con):
    path = ROOT / "simulation" / "r_verification" / "results" / "06_rt_epiestim.csv"
    if not path.exists():
        log.warning("skip rt_history seed: %s missing", path)
        return
    df = pd.read_csv(path)
    rows = []
    for _, r in df.iterrows():
        if pd.notna(r["rt_cori_mean"]):
            rows.append({
                "gu": "seoul_city",
                "week_start": str(r["week_start"]),
                "method": "EpiEstim_Cori",
                "rt_mean": float(r["rt_cori_mean"]),
                "rt_q025": float(r["rt_cori_q025"]),
                "rt_q975": float(r["rt_cori_q975"]),
                "si_mean": 2.6, "si_sd": 1.5,
            })
        if pd.notna(r.get("rt_seir_v2")):
            rows.append({
                "gu": "seoul_city",
                "week_start": str(r["week_start"]),
                "method": "SEIR-V2",
                "rt_mean": float(r["rt_seir_v2"]),
                "rt_q025": None, "rt_q975": None,
                "si_mean": 2.6, "si_sd": 1.5,
            })
    cur = con.cursor()
    cur.execute("DELETE FROM rt_history "
                "WHERE method IN ('EpiEstim_Cori', 'SEIR-V2')")
    con.commit()
    bulk_insert("rt_history", rows, conn=con, chunk_size=500)
    log.info("seeded rt_history=%d", len(rows))


def seed_shap_delta_snapshots(con):
    summary = RES / "stage3_shap" / "summary.json"
    if not summary.exists():
        log.warning("skip shap_delta_snapshots seed: %s missing", summary)
        return
    data = json.load(summary.open(encoding="utf-8"))
    ts = data.get("generated_at", pd.Timestamp.now().isoformat())
    rows = []

    # MI global ranking → baseline rows for "any model"
    for rec in data.get("global_ranking_mi", [])[:20]:
        rows.append({
            "snapshot_ts": ts,
            "model": "_global",
            "feature": rec["feature"],
            "rank": rec["rank"],
            "mi_score": rec.get("mi_score"),
            "source": "mi_global",
        })
    # Per-model top features
    for model, feats in data.get("per_model_top_features", {}).items():
        for rec in feats:
            rows.append({
                "snapshot_ts": ts,
                "model": model,
                "feature": rec["feature"],
                "rank": rec["rank"],
                "mi_score": None,
                "source": "per_model_recommended",
            })

    cur = con.cursor()
    cur.execute("DELETE FROM shap_delta_snapshots "
                "WHERE source IN ('mi_global', 'per_model_recommended')")
    con.commit()
    bulk_insert("shap_delta_snapshots", rows, conn=con, chunk_size=200)
    log.info("seeded shap_delta_snapshots=%d", len(rows))


def seed_scenario_runs(con):
    m_path = RES / "sim_runs" / "_manifest.json"
    if not m_path.exists():
        log.warning("skip scenario_runs seed: %s missing", m_path)
        return
    manifest = json.load(m_path.open(encoding="utf-8"))
    ts = manifest.get("generated_at")
    rows = []
    for name, s in manifest["scenarios"].items():
        rows.append({
            "scenario_id": f"stage5/{name}",
            "name": name,
            "generated_at": ts,
            "days_run": int(s.get("days_run", 0)),
            "peak_day": int(s.get("peak_day", 0)),
            "peak_i_city": float(s.get("peak_I_city", 0)),
            "attack_rate_pct": float(s.get("attack_rate_pct", 0)),
            "cumulative_deaths": float(s.get("cumulative_deaths", 0)),
            "n_interventions": int(s.get("n_interventions", 0)),
            "epi_validity_ok": int(bool(s.get("epi_validity_ok", False))),
            "compartment_conservation_err": (
                float(s["compartment_conservation_err"])
                if s.get("compartment_conservation_err") is not None else None
            ),
            "interventions_json": json.dumps(s.get("interventions", []),
                                             ensure_ascii=False),
        })

    cur = con.cursor()
    cur.execute("DELETE FROM scenario_runs WHERE scenario_id LIKE 'stage5/%'")
    con.commit()
    bulk_insert("scenario_runs", rows, conn=con, chunk_size=50)
    log.info("seeded scenario_runs=%d", len(rows))


# ── Entrypoint ──────────────────────────────────────────────────────────
def main() -> None:
    with safe_connect() as con:
        create_tables(con)
        seed_forecast_runs_and_points(con)
        seed_rt_history(con)
        seed_shap_delta_snapshots(con)
        seed_scenario_runs(con)
    log.info("Stage 6b Turso table seeding complete")


if __name__ == "__main__":
    main()
