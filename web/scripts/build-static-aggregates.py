#!/usr/bin/env python
"""
build-static-aggregates.py — pre-compute JSON blobs that the Next.js
edge runtime can serve from its CDN layer without hitting Turso on
every request.

Three outputs land in ``web/public/``:

  * ``seoul-gu.geojson``         — vendored from ``simulation/data/external``
                                   (our stub is a bounding-box fallback);
                                   this script picks the real file when
                                   present, else warns.
  * ``aggregates/latest-choropleth.json`` — latest observed ILI per gu
  * ``aggregates/commuter-edges.json``    — top-50 weighted commuter edges
                                            (for the relationship overlay)

These are the three things the map needs *before* a first MCP round-
trip, so pre-baking keeps the cold-start snappy on the demo tablet.

Run from the project root:
    .venv/bin/python web/scripts/build-static-aggregates.py
"""

from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sqlite3
import sys
from pathlib import Path

SEOUL_GU = [
    "강남구", "강동구", "강북구", "강서구", "관악구",
    "광진구", "구로구", "금천구", "노원구", "도봉구",
    "동대문구", "동작구", "마포구", "서대문구", "서초구",
    "성동구", "성북구", "송파구", "양천구", "영등포구",
    "용산구", "은평구", "종로구", "중구", "중랑구",
]


def copy_geojson(src_dir: Path, out_dir: Path) -> None:
    """Prefer the real Seoul-gu GeoJSON if present in data/external."""
    candidates = [
        src_dir / "external" / "seoul_gu.geojson",
        src_dir / "external" / "seoul-gu.geojson",
    ]
    for c in candidates:
        if c.is_file():
            shutil.copyfile(c, out_dir / "seoul-gu.geojson")
            print(f"copied geojson: {c} → {out_dir/'seoul-gu.geojson'}", file=sys.stderr)
            return
    print(
        "! seoul_gu.geojson not found in simulation/data/external — "
        "falling back to the bounding-box stub already in web/public/",
        file=sys.stderr,
    )


def build_choropleth(con: sqlite3.Connection, out_path: Path) -> None:
    """Latest observed notifiable-disease burden per gu.

    NOTE (2026-04-21 audit): Korean surveillance does NOT report ILI at
    gu level — KDCA sentinel is national only. The original query against
    ``weekly_disease`` (where column is ``sido_nm`` not ``gu_nm``, and
    ``disease_cd`` is Korean legal class like '제1급' not ICD-10 J10)
    returns 0 rows. We fall back to ``seoul_disease_district`` 2024
    with ``disease_nm='제2급감염병'`` (legal class 2 aggregate — 22,789
    cases across 25 gus) as a population-level disease burden proxy that
    renders meaningfully on the map. The frontend labels this as
    "Notifiable disease burden (class 2)" to remain honest about the
    flu-vs-aggregate distinction.
    """
    # Preferred: real per-gu flu (does not exist in this DB) — try first,
    # fall through if coverage < 10 gus. The LIKE '%인플루엔자%' also
    # matches 'b형헤모필루스인플루엔자' (Hib — bacterial, not influenza
    # virus); a coverage gate is safer than a brittle NOT LIKE list.
    rows = []
    try:
        rows = con.execute(
            """
            SELECT gu_nm, SUM(cases) AS cases
              FROM seoul_disease_district
             WHERE disease_nm LIKE '%인플루엔자%'
               AND disease_nm NOT LIKE '%헤모필루스%'
               AND category = '발생_계'
               AND cases IS NOT NULL
               AND gu_nm != '서울시'
               AND year = (SELECT MAX(year) FROM seoul_disease_district
                            WHERE disease_nm LIKE '%인플루엔자%'
                              AND disease_nm NOT LIKE '%헤모필루스%'
                              AND cases IS NOT NULL)
             GROUP BY gu_nm
            """
        ).fetchall()
    except sqlite3.OperationalError as e:
        print(f"! flu probe failed: {e}", file=sys.stderr)

    metric_label = "ILI cases (per-gu, latest year)"
    # Require ≥10 gus to be usable as a Seoul choropleth; otherwise fall
    # through to the class-2 aggregate which covers all 25.
    if len(rows) < 10:
        if rows:
            print(
                f"! flu probe returned only {len(rows)} gu(s) — "
                "falling through to 제2급감염병 aggregate",
                file=sys.stderr,
            )
        # Fallback: 제2급감염병 aggregate — actually populated per-gu.
        rows = con.execute(
            """
            SELECT gu_nm, SUM(cases) AS cases
              FROM seoul_disease_district
             WHERE disease_nm = '제2급감염병'
               AND category = '발생_계'
               AND cases IS NOT NULL
               AND gu_nm != '서울시'
               AND year = (SELECT MAX(year) FROM seoul_disease_district
                            WHERE disease_nm = '제2급감염병')
             GROUP BY gu_nm
             ORDER BY gu_nm
            """
        ).fetchall()
        metric_label = "Notifiable disease burden (class 2, latest year)"

    data = [{"gu_nm": r[0], "cases": int(r[1])} for r in rows]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({
            "metric": metric_label,
            "disclaimer": (
                "KDCA sentinel ILI is national-only; this layer shows the "
                "nearest available per-gu notifiable-disease signal."
            ),
            "rows": data,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out_path} ({len(data)} rows, metric={metric_label!r})",
          file=sys.stderr)


def build_commuter_edges(con: sqlite3.Connection, out_path: Path, topk: int = 50) -> None:
    """Top-k weighted commuter edges (origin != dest).

    Schema note: ``commuter_matrix`` has columns ``origin_gu``, ``dest_gu``,
    ``coupling`` (0–1 share; diagonals ≈ 0.80 self-stay, off-diagonals small).
    The integer ``commuters`` column is mostly 0 in the KOSIS feed, so we
    use ``coupling`` as the edge weight.
    """
    try:
        rows = con.execute(
            """
            SELECT origin_gu, dest_gu, coupling
              FROM commuter_matrix
             WHERE origin_gu != dest_gu
               AND coupling IS NOT NULL
             ORDER BY coupling DESC
             LIMIT ?
            """,
            (topk,),
        ).fetchall()
    except sqlite3.OperationalError as e:
        print(f"! commuter_matrix unavailable: {e}", file=sys.stderr)
        return
    data = [{"src": r[0], "dst": r[1], "weight": float(r[2])} for r in rows]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"edges": data}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out_path} ({len(data)} edges)", file=sys.stderr)


def build_ili_local(con: sqlite3.Connection, abm_path: Path, out_path: Path) -> None:
    """Generate per-gu ILI JSON from local sources only (no Turso).

    Algorithm
    ---------
    1. City-level ILI = average of age-group rates for the latest week in
       ``sentinel_influenza`` (KDCA표본감시, 서울 ILI rate /1k).
    2. Per-gu modulation = use relative I_frac pattern from the latest day
       of the ABM baseline scenario to weight the city value across 25 gus.
       If ABM data is absent or all zeros, broadcast city value uniformly.
    3. Per-gu q70 = rolling city-level q70 (52 most-recent weekly averages)
       broadcast to every gu (sentinel is city-only, no gu-level history).
    4. alert flag = value > q70.

    Args:
        con: read-only SQLite connection to epi_real_seoul.db.
        abm_path: path to web/public/aggregates/abm-scenarios.json.
        out_path: destination path for ili-local.json.

    Returns:
        None. Writes JSON to out_path. Idempotent.

    Raises:
        Nothing — catches internal errors and writes a fallback JSON with
        a note field explaining the problem.

    Side effects:
        Writes out_path to disk.
    """
    try:
        _build_ili_local_inner(con, abm_path, out_path)
    except Exception as exc:  # noqa: BLE001
        note = f"build_ili_local error: {exc}"
        print(f"! {note}", file=sys.stderr)
        # Write a minimal valid file so the Edge layer has something to
        # fetch rather than a 404 → fallback chain degrades gracefully.
        payload = {
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "observed_at": "1970-01-01T00:00:00Z",
            "source": "local-db",
            "note": note,
            "gu": {},
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_ili_local_inner(con: sqlite3.Connection, abm_path: Path, out_path: Path) -> None:
    """Inner implementation — see build_ili_local for contract."""
    # ── Step 1: latest week city-level ILI ───────────────────────────
    latest = con.execute(
        """
        SELECT season_start, week_seq
          FROM sentinel_influenza
         GROUP BY season_start, week_seq
         ORDER BY season_start DESC, week_seq DESC
         LIMIT 1
        """
    ).fetchone()
    if latest is None:
        raise RuntimeError("sentinel_influenza is empty")
    season_start, week_seq = latest

    city_ili_row = con.execute(
        """
        SELECT AVG(ili_rate) AS avg_ili
          FROM sentinel_influenza
         WHERE season_start = ? AND week_seq = ?
        """,
        (season_start, week_seq),
    ).fetchone()
    city_ili = float(city_ili_row[0]) if city_ili_row and city_ili_row[0] is not None else 0.0

    # Derive ISO date for the observed week.
    # Korean flu season (season_start = Y) begins at ISO week 36 of year Y.
    # week_seq=1 → ISO week 36, week_seq=k → ISO week 36+(k-1).
    base_date = datetime.date.fromisocalendar(season_start, 36, 1)
    obs_date = base_date + datetime.timedelta(weeks=week_seq - 1)
    observed_at = f"{obs_date.isoformat()}T00:00:00Z"

    # ── Step 2: rolling 52-week city q70 ────────────────────────────
    hist_rows = con.execute(
        """
        SELECT AVG(ili_rate) AS avg_ili
          FROM sentinel_influenza
         GROUP BY season_start, week_seq
         ORDER BY season_start DESC, week_seq DESC
         LIMIT 52
        """
    ).fetchall()
    hist_values = sorted(float(r[0]) for r in hist_rows if r[0] is not None)
    if hist_values:
        q70_idx = max(0, int(0.70 * len(hist_values)) - 1)
        city_q70 = hist_values[q70_idx]
    else:
        city_q70 = city_ili * 1.5  # safe fallback

    # ── Step 3: per-gu relative weights from ABM I_frac ─────────────
    abm_weights: dict[str, float] = {}
    broadcast_note = "서울 표본감시 ILI를 자치구로 분배(모델)"
    if abm_path.is_file():
        try:
            abm = json.loads(abm_path.read_text("utf-8"))
            gu_names: list[str] = abm.get("gu_names", [])
            i_frac_series: list[list[float]] = abm.get("scenarios", {}).get(
                "baseline", {}
            ).get("I_frac", [])
            if gu_names and i_frac_series:
                last_day = i_frac_series[-1]
                total = sum(last_day)
                if total > 0 and len(last_day) == len(gu_names):
                    for gu, frac in zip(gu_names, last_day):
                        abm_weights[gu] = frac / total * len(gu_names)
                    broadcast_note = "서울 표본감시 ILI를 ABM I_frac 패턴으로 자치구 분배(모델)"
        except Exception as exc:  # noqa: BLE001
            print(f"! abm-scenarios parse error: {exc}", file=sys.stderr)

    # ── Step 4: assemble per-gu payload ─────────────────────────────
    gu_dict: dict[str, dict] = {}
    for gu in SEOUL_GU:
        weight = abm_weights.get(gu, 1.0)  # uniform = 1.0 when no ABM
        gu_ili = round(city_ili * weight, 4)
        # q70 is city-level broadcast (no gu-level history exists)
        gu_q70 = round(city_q70 * weight, 4)
        gu_dict[gu] = {
            "ili": gu_ili,
            "q70": gu_q70,
            "alert": gu_ili > gu_q70,
        }

    payload = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "observed_at": observed_at,
        "source": "local-db",
        "note": broadcast_note,
        "gu": gu_dict,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"wrote {out_path} ({len(gu_dict)} gus, "
        f"city_ili={city_ili:.2f}/1k, q70={city_q70:.2f}, "
        f"observed_at={observed_at})",
        file=sys.stderr,
    )


def build_ili_forecast(
    summary_csv: Path,
    predictions_dir: Path,
    abm_path: Path,
    out_path: Path,
) -> None:
    """Generate per-gu ILI forecast JSON from the best trained model.

    Algorithm (2026-06-09 update: production refit path)
    -----------------------------------------------------
    If ``web/scripts/build_production_forecast.py`` is available, delegate to
    it for a proper production refit forecast (NegBinGLM full-data refit →
    future 1-step, climatology weather, conformal PI, gate-checked).

    Legacy path (fallback only):
    1. Champion model = model with the lowest WIS (or best test_r2 as proxy)
       from ``summary_metrics.csv`` in ``predictions_dir``.
    2. Seoul forecast value = latest ``y_pred`` in the champion's
       ``predictions_<model>.csv`` (test split, highest idx row).
    3. Approximate PI = ``y_pred ± 2 × test_rmse`` (95 % normal approx).
    4. Per-gu distribution = ABM baseline I_frac pattern.

    Args:
        summary_csv: path to ``simulation/results/csv/summary_metrics.csv``.
        predictions_dir: directory containing ``predictions_<model>.csv`` files.
        abm_path: path to ``web/public/aggregates/abm-scenarios.json``.
        out_path: destination path for ``ili-forecast.json``.

    Returns:
        None.  Writes JSON to out_path.  Idempotent.

    Raises:
        Nothing — catches internal errors and writes a fallback JSON with
        a ``note`` field so the Edge layer has a valid file to fetch.

    Side effects:
        Writes out_path to disk.
    """
    # ── Production refit path (preferred) ───────────────────────────────
    prod_script = Path(__file__).parent / "build_production_forecast.py"
    if prod_script.is_file():
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "build_production_forecast", str(prod_script)
            )
            _mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(_mod)
            rc = _mod.main()
            if rc == 0:
                print(
                    "build_ili_forecast: production refit complete "
                    f"(wrote {out_path})",
                    file=sys.stderr,
                )
                return
            print(
                f"! build_production_forecast.main() returned {rc} — "
                "falling back to legacy path",
                file=sys.stderr,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"! production refit failed ({type(exc).__name__}: {exc}) — "
                "falling back to legacy path",
                file=sys.stderr,
            )
    # ── Legacy path (fallback) ───────────────────────────────────────────
    try:
        _build_ili_forecast_inner(summary_csv, predictions_dir, abm_path, out_path)
    except Exception as exc:  # noqa: BLE001
        note = f"build_ili_forecast error: {exc}"
        print(f"! {note}", file=sys.stderr)
        payload = {
            "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
            "observed_at": "1970-01-01T00:00:00Z",
            "source": "model-forecast",
            "model": "unknown",
            "horizon_weeks": 1,
            "note": note,
            "gu": {},
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_ili_forecast_inner(
    summary_csv: Path,
    predictions_dir: Path,
    abm_path: Path,
    out_path: Path,
) -> None:
    """Inner implementation — see build_ili_forecast for contract."""
    import csv as _csv

    # ── Step 1: identify champion model ─────────────────────────────────
    if not summary_csv.is_file():
        raise FileNotFoundError(f"summary_metrics.csv not found: {summary_csv}")

    with summary_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(_csv.DictReader(fh))
    if not rows:
        raise ValueError("summary_metrics.csv is empty")

    # Prefer WIS column; fall back to test_r2 (higher = better).
    if "wis" in rows[0]:
        # Lower WIS = better champion.
        rows_sorted = sorted(rows, key=lambda r: float(r["wis"] or "inf"))
    else:
        # Higher test_r2 = better.
        rows_sorted = sorted(rows, key=lambda r: float(r.get("test_r2") or "-inf"), reverse=True)

    champion_name = rows_sorted[0]["name"]
    champion_rmse = float(rows_sorted[0].get("test_rmse") or 0.0)
    print(f"champion: {champion_name}  rmse={champion_rmse:.4f}", file=sys.stderr)

    # ── Step 2: model relative error (for the prediction interval) ──────
    # The champion's raw test predictions span winter peaks, so the *level*
    # of "last test y_pred" is NOT a valid summer 1-week forecast — it would
    # show ~47/1k in June (≈9× the ~5/1k observed). We anchor the level to the
    # CURRENT observed ILI in Step 3 and use the model only for attribution
    # plus a relative-error prediction interval.
    pred_path = predictions_dir / f"predictions_{champion_name}.csv"
    rel_rmse = 0.25  # fallback ±25 %
    if pred_path.is_file():
        with pred_path.open(newline="", encoding="utf-8") as fh:
            pred_rows = list(_csv.DictReader(fh))
        test_pred = [
            float(r["y_pred"])
            for r in pred_rows
            if r.get("split") == "test" and r.get("y_pred")
        ]
        test_mean = sum(test_pred) / len(test_pred) if test_pred else 0.0
        if test_mean > 0 and champion_rmse > 0:
            rel_rmse = min(0.5, max(0.1, champion_rmse / test_mean))
    print(f"champion rel-rmse={rel_rmse:.3f}", file=sys.stderr)

    # ── Step 3: anchor to CURRENT observed ILI + forecast week ──────────
    # Persistence 1-week-ahead anchored to the latest observed city ILI
    # (ili-local.json, sentinel-derived). Epidemiologically sane: summer ILI
    # does not jump 9× in a week. Superseded by the model's live inference
    # once the fresh 53-model run lands.
    anchor = 0.0
    forecast_date = datetime.datetime.utcnow().isoformat() + "Z"
    ili_local_path = abm_path.parent.joinpath("ili-local.json")
    if ili_local_path.is_file():
        try:
            ili_local = json.loads(ili_local_path.read_text("utf-8"))
            gu_obs = ili_local.get("gu", {})
            vals = [
                float(v.get("ili", 0))
                for v in gu_obs.values()
                if isinstance(v, dict)
            ]
            if vals:
                anchor = sum(vals) / len(vals)
            last_obs = ili_local.get("observed_at", "1970-01-01T00:00:00Z")
            last_obs_dt = datetime.date.fromisoformat(last_obs[:10])
            forecast_date = (
                f"{(last_obs_dt + datetime.timedelta(weeks=1)).isoformat()}T00:00:00Z"
            )
        except Exception:  # noqa: BLE001
            pass
    if anchor <= 0:
        anchor = 5.0  # safe summer-baseline fallback
    city_forecast = round(anchor, 4)
    city_lo = max(0.0, round(anchor * (1 - 2 * rel_rmse), 4))
    city_hi = round(anchor * (1 + 2 * rel_rmse), 4)
    print(
        f"anchor={anchor:.2f}/1k  forecast={city_forecast}  "
        f"PI=[{city_lo},{city_hi}]  week={forecast_date}",
        file=sys.stderr,
    )

    # ── Step 4: per-gu weights from ABM I_frac ───────────────────────────
    abm_weights: dict[str, float] = {}
    broadcast_note = (
        f"현 관측 {round(anchor, 1)}/1k anchored 1주 전망 · {champion_name} "
        f"상대오차 ±{round(rel_rmse * 200)}% PI · ABM baseline 패턴 자치구 분배 "
        f"(53-모델 재학습 시 모델 직접예측으로 갱신)"
    )
    if abm_path.is_file():
        try:
            abm = json.loads(abm_path.read_text("utf-8"))
            gu_names: list[str] = abm.get("gu_names", [])
            i_frac_series: list[list[float]] = (
                abm.get("scenarios", {}).get("baseline", {}).get("I_frac", [])
            )
            if gu_names and i_frac_series:
                last_day = i_frac_series[-1]
                total = sum(last_day)
                if total > 0 and len(last_day) == len(gu_names):
                    for gu, frac in zip(gu_names, last_day):
                        abm_weights[gu] = frac / total * len(gu_names)
        except Exception as exc:  # noqa: BLE001
            print(f"! abm-scenarios parse error: {exc}", file=sys.stderr)
            broadcast_note = (
                f"챔피언 {champion_name} 서울 예측을 자치구로 균등 분배(ABM 오류)"
            )
    else:
        broadcast_note = (
            f"챔피언 {champion_name} 서울 예측을 자치구로 균등 분배(ABM 없음)"
        )

    # ── Step 5: assemble per-gu payload ─────────────────────────────────
    gu_dict: dict[str, dict] = {}
    for gu in SEOUL_GU:
        weight = abm_weights.get(gu, 1.0)
        gu_dict[gu] = {
            "ili": round(city_forecast * weight, 4),
            "lo": round(city_lo * weight, 4),
            "hi": round(city_hi * weight, 4),
        }

    payload = {
        "generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "observed_at": forecast_date,
        "source": "model-forecast",
        "model": champion_name,
        "horizon_weeks": 1,
        "note": broadcast_note,
        "gu": gu_dict,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"wrote {out_path} ({len(gu_dict)} gus, "
        f"champion={champion_name}, city_forecast={city_forecast:.4f}/1k, "
        f"observed_at={forecast_date})",
        file=sys.stderr,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="simulation/data/db/epi_real_seoul.db")
    ap.add_argument("--data-dir", default="simulation/data")
    ap.add_argument("--out", default="web/public")
    args = ap.parse_args()

    db_path = Path(args.db)
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "aggregates").mkdir(exist_ok=True)

    copy_geojson(data_dir, out_dir)

    if not db_path.is_file():
        print(f"! db not found: {db_path} — skipping aggregates", file=sys.stderr)
        return 1

    con = sqlite3.connect(str(db_path))
    con.execute("PRAGMA query_only = ON")
    build_choropleth(con, out_dir / "aggregates" / "latest-choropleth.json")
    build_commuter_edges(con, out_dir / "aggregates" / "commuter-edges.json")
    build_ili_local(
        con,
        out_dir / "aggregates" / "abm-scenarios.json",
        out_dir / "aggregates" / "ili-local.json",
    )
    con.close()

    # ILI forecast from best trained model (idempotent; no DB needed).
    build_ili_forecast(
        summary_csv=Path("simulation/results/csv/summary_metrics.csv"),
        predictions_dir=Path("simulation/results/csv"),
        abm_path=out_dir / "aggregates" / "abm-scenarios.json",
        out_path=out_dir / "aggregates" / "ili-forecast.json",
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
