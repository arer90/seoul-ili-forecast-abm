"""Forecast-anchored ABM integration for Seoul ILI model forecasts."""
from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from simulation.abm.counterfactual import run_counterfactual
from simulation.abm.epi_proof import (
    BEHAVIOUR_OFF,
    DB_PATH,
    DEFAULT_DISEASE,
    FORCING_GRID,
    SeasonSeries,
    _apply_affine,
    _fit_affine,
    _load_ili_seasons,
    _make_population,
    _simulate_replicates,
    _wis_per_week,
)
from simulation.database import safe_connect


DEFAULT_MODEL = "NegBinGLM-V7"
DEFAULT_PATH = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "eda"
    / "phase11_per_model_eval"
    / "predictions_per_model.csv"
)
RESULT_PATH = (
    Path(__file__).resolve().parents[2]
    / "paper"
    / "_thesis_revision_20260604"
    / "real_runs"
    / "forecast_anchored.json"
)
_POPULATION_HELPER = _make_population


def load_forecast(
    model_name: str = DEFAULT_MODEL,
    predictions_csv: str | Path = DEFAULT_PATH,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one model's ordered forecast series.

    Args:
        model_name: Forecast model identifier in the CSV ``model`` column.
        predictions_csv: CSV with ``week_idx``, ``model``, and ``y_pred``
            columns. The default is the R10 per_model_eval prediction file.

    Returns:
        Tuple ``(weeks, y_pred)`` as one-dimensional NumPy arrays sorted by
        ``week_idx``. ``weeks`` has integer dtype and ``y_pred`` has float dtype.

    Raises:
        FileNotFoundError: If ``predictions_csv`` does not exist.
        ValueError: If required columns are missing, the selected model has no
            rows, or selected values are non-finite.

    Performance: O(rows) time and O(selected rows) memory.
    Side effects: Reads a local CSV only; no network, DB, or writes.
    Caller responsibility: ``model_name`` must match the CSV exactly.
    """
    path = Path(predictions_csv)
    rows: list[tuple[int, float]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"week_idx", "model", "y_pred"}
        columns = set(reader.fieldnames or ())
        missing = required - columns
        if missing:
            raise ValueError(f"forecast CSV missing columns: {sorted(missing)}")
        for row in reader:
            if row["model"] != model_name:
                continue
            rows.append((int(row["week_idx"]), float(row["y_pred"])))
    if not rows:
        raise ValueError(f"model {model_name!r} not found in {path}")
    rows.sort(key=lambda item: item[0])
    weeks = np.array([w for w, _ in rows], dtype=np.int16)
    y_pred = np.array([v for _, v in rows], dtype=np.float64)
    if not np.all(np.isfinite(y_pred)):
        raise ValueError(f"forecast for {model_name!r} contains non-finite values")
    return weeks, y_pred


DEFAULT_REAL_DIR = (
    Path(__file__).resolve().parents[1] / "results" / "real_eval"
)


def load_real_forecast(
    real_eval_dir: str | Path = DEFAULT_REAL_DIR,
    model_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the real-slab operational forecast from P1 real_forecaster output.

    This is the **live SSOT** source for forecast→ABM anchoring. The post-cutoff
    real forecast (the ABM's true input) is produced by P1 real_forecaster as the
    best *operational* model (``summary.json['best_model']``) — NOT the test
    champion, which extrapolation-collapses on the real slab
    (REAL_FORECAST_STABILITY). The legacy R10 per_model_eval CSV
    ``load_forecast`` is back-compat only.

    Args:
        real_eval_dir: ``simulation/results/real_eval`` directory.
        model_name: operational model name; ``None`` → ``summary.json`` best_model.

    Returns:
        ``(weeks, y_pred)`` — ``weeks = arange(real_n)`` (int16), ``y_pred`` the
        model's per-week real forecast (float64, finite).

    Raises:
        FileNotFoundError: ``real_eval_dir/summary.json`` missing.
        ValueError: model has no ``predictions`` array, or values non-finite.

    Side effects: reads local JSON only; no DB/network/writes.
    """
    real_eval_dir = Path(real_eval_dir)

    def _read_preds(name: str):
        pm = real_eval_dir / "per_model" / f"{name}.json"
        if pm.exists():
            p = json.loads(pm.read_text(encoding="utf-8")).get("predictions")
            if p:
                return p
        mf = real_eval_dir / "metrics_full.json"
        if mf.exists():
            p = (json.loads(mf.read_text(encoding="utf-8")).get(name) or {}).get(
                "predictions"
            )
            if p:
                return p
        return None

    # candidate order: explicit → summary best_model → every available model
    #   (best_model can be a degenerate value like "individual_results" on tiny
    #   runs, so fall through to the real per-model predictions).
    candidates: list[str] = []
    if model_name:
        candidates.append(model_name)
    else:
        sp = real_eval_dir / "summary.json"
        if sp.exists():
            _summ = json.loads(sp.read_text(encoding="utf-8"))
            # A1 (M7): prefer the gated DEPLOYMENT forecast (champion → stable
            # fallback on contract violation) so the ABM is never anchored to an
            # extrapolation-collapsed champion. Falls through if absent/non-finite.
            _dep = _summ.get("deployment")
            _dep_fc = _dep.get("forecast") if isinstance(_dep, dict) else None
            if isinstance(_dep_fc, list) and _dep_fc:
                _arr = np.asarray(_dep_fc, dtype=np.float64)
                if _arr.size and np.all(np.isfinite(_arr)):
                    return np.arange(len(_arr), dtype=np.int16), _arr
            bm = _summ.get("best_model")
            if bm:
                candidates.append(bm)
        mfp = real_eval_dir / "metrics_full.json"
        if mfp.exists():
            for k in json.loads(mfp.read_text(encoding="utf-8")):
                if k not in candidates:
                    candidates.append(k)
        pmd = real_eval_dir / "per_model"
        if pmd.exists():
            for p in sorted(pmd.glob("*.json")):
                if p.stem not in candidates:
                    candidates.append(p.stem)
    preds = resolved = None
    for name in candidates:
        preds = _read_preds(name)
        if preds:
            resolved = name
            break
    if not preds:
        raise ValueError(
            f"no real forecast predictions found in {real_eval_dir} "
            f"(tried {candidates})"
        )
    y_pred = np.asarray(preds, dtype=np.float64)
    weeks = np.arange(len(y_pred), dtype=np.int16)
    if not np.all(np.isfinite(y_pred)):
        raise ValueError(
            f"real forecast for {resolved!r} contains non-finite values"
        )
    return weeks, y_pred


def anchor_abm_to_forecast(
    forecast: np.ndarray,
    *,
    n_agents: int,
    seeds: Sequence[int] | None = None,
    year: int | None = None,
    transmission_mode: str = "meanfield",
    network_kwargs: dict | None = None,
    beta_by_layer: dict | None = None,
) -> dict[str, Any]:
    """Calibrate seasonal ABM forcing to a forecast trajectory.

    Args:
        forecast: One-dimensional weekly ILI rate forecast. Values must be
            finite and non-negative.
        n_agents: Synthetic population size. Use at least 10,000 agents for a
            meaningful seasonal epidemic; smaller runs can collapse to a
            degenerate affine scale.
        seeds: Replicate seeds. Defaults to ``range(5)``.
        year: Synthetic-population reference year. Defaults to the latest
            available ILI season in the local DB.

    Returns:
        Dict with fitted forcing, affine map, correlation, WIS, affine-mapped
        mean trajectory, and ``degenerate``. ``degenerate`` is true exactly when
        ``abs(scale) < 1e-9``; callers must not treat such a run as a successful
        epidemiological anchor.

    Raises:
        ValueError: If inputs are empty, non-finite, negative, or no seeds are
            supplied.
        RuntimeError: If the forcing grid has no valid candidate.

    Performance: O(len(FORCING_GRID product) * len(seeds) * weeks * 7 *
        n_agents) time and O(len(seeds) * weeks) score memory, plus the shared
        synthetic-population cache used by ``epi_proof``.
    Side effects: Reads the local DB only when ``year`` is omitted; no writes or
        network/API calls.
    Caller responsibility: Inspect ``degenerate`` before reporting the result
        as an anchored ABM success.
    """
    y = _validate_forecast(forecast)
    seed_tuple = _seed_tuple(seeds)
    season_year = _latest_year(DB_PATH) if year is None else int(year)
    season = SeasonSeries(
        season=season_year,
        week_seq=np.arange(y.size, dtype=np.int16),
        ili_rate=y,
    )

    best: tuple[float, dict[str, float], Any, np.ndarray, float] | None = None
    tried = 0
    for beta, amp, phase in itertools.product(
        FORCING_GRID["beta"],
        FORCING_GRID["beta_amp"],
        FORCING_GRID["beta_phase"],
    ):
        disease = {
            **DEFAULT_DISEASE,
            "beta": float(beta),
            "beta_amp": float(amp),
            "beta_phase": float(phase),
        }
        reps = _simulate_replicates(
            season,
            seeds=seed_tuple,
            n_agents=int(n_agents),
            disease=disease,
            behaviour=BEHAVIOUR_OFF,
            population_kind="rich_movement",
            transmission_mode=transmission_mode,
            network_kwargs=network_kwargs,
            beta_by_layer=beta_by_layer,
        )
        affine = _fit_affine(reps.mean(axis=0), y)
        mapped = _apply_affine(reps, affine)
        wis = float(np.mean(_wis_per_week(y, mapped)))
        corr = _finite_corr(mapped.mean(axis=0), y)
        tried += 1
        if best is None or wis < best[0]:
            best = (wis, disease, affine, mapped.mean(axis=0), corr)

    if best is None:
        raise RuntimeError("forecast anchoring produced no valid candidate")

    wis, disease, affine, anchored, corr = best
    scale = float(affine.scale)
    degenerate = bool(abs(scale) < 1.0e-9)
    return {
        "fitted_forcing": {
            "beta": float(disease["beta"]),
            "beta_amp": float(disease["beta_amp"]),
            "beta_phase": float(disease["beta_phase"]),
            "import_rate": float(disease["import_rate"]),
        },
        "affine": {"offset": float(affine.offset), "scale": scale},
        "corr_sim_vs_forecast": float(corr),
        "wis": float(wis),
        "anchored_trajectory": [float(v) for v in anchored],
        "degenerate": degenerate,
        "metadata": {
            "n_agents": int(n_agents),
            "seeds": [int(s) for s in seed_tuple],
            "year": int(season_year),
            "weeks": int(y.size),
            "grid_size": int(tried),
            "behaviour": dict(BEHAVIOUR_OFF),
            "objective": "forecast WIS after affine map with behaviour off",
        },
    }


def n_sweep(
    forecast: np.ndarray,
    *,
    n_values: Sequence[int] = (10_000, 20_000, 37_500, 75_000),
    seeds: Sequence[int] | None = None,
    year: int | None = None,
) -> list[dict[str, Any]]:
    """Run forecast anchoring across population sizes.

    Args:
        forecast: One-dimensional weekly ILI rate forecast.
        n_values: Agent counts to evaluate.
        seeds: Replicate seeds. Defaults to ``range(5)``.
        year: Synthetic-population reference year. Defaults to the latest local
            ILI season.

    Returns:
        List of dicts, one per ``n_values`` entry, with ``n_agents``,
        ``corr_sim_vs_forecast``, ``wis``, ``peak``, and ``degenerate``.

    Raises:
        ValueError: If ``n_values`` is empty or contains non-positive values.

    Performance: O(len(n_values) * anchor_abm_to_forecast). Shared simulation
        cache reuses identical N/seed/grid runs within a process.
    Side effects: Reads the local DB only when ``year`` is omitted; no writes or
        network/API calls.
    Caller responsibility: Treat rows with ``degenerate=True`` as failed ABM
        anchors, even if point metrics are finite.
    """
    values = tuple(int(n) for n in n_values)
    if not values:
        raise ValueError("n_values must contain at least one N")
    if any(n <= 0 for n in values):
        raise ValueError("all n_values must be positive")
    rows: list[dict[str, Any]] = []
    for n in values:
        result = anchor_abm_to_forecast(
            forecast,
            n_agents=n,
            seeds=seeds,
            year=year,
        )
        trajectory = np.asarray(result["anchored_trajectory"], dtype=np.float64)
        rows.append(
            {
                "n_agents": int(n),
                "corr_sim_vs_forecast": float(result["corr_sim_vs_forecast"]),
                "wis": float(result["wis"]),
                "peak": float(np.max(trajectory)) if trajectory.size else float("nan"),
                "degenerate": bool(result["degenerate"]),
            }
        )
    return rows


def run_forecast_anchored(
    *,
    model_name: str = DEFAULT_MODEL,
    n_agents: int = 37_500,
    n_sweep_values: Sequence[int] | None = None,
    K: int = 20,
    seeds: Sequence[int] | None = None,
    year: int | None = None,
    output_path: str | Path = RESULT_PATH,
) -> dict[str, Any]:
    """Run the forecast-anchored ABM and counterfactual package.

    Args:
        model_name: Forecast model to load from the R10 per_model_eval
            predictions CSV.
        n_agents: Agent count for the primary anchored baseline.
        n_sweep_values: Optional population-size sweep. Defaults to
            ``(10000, 20000, 37500, 75000)``.
        K: Replicates for the downstream vaccination counterfactual.
        seeds: Anchor replicate seeds. Defaults to ``range(5)``.
        year: Synthetic-population reference year. Defaults to the latest local
            ILI season.
        output_path: JSON path for the integrated result.

    Returns:
        Dict containing the loaded forecast metadata, anchored baseline,
        population-size sweep, counterfactual result, and output metadata.

    Raises:
        FileNotFoundError: If the forecast CSV is missing.
        ValueError: If forecast data or run arguments are invalid.

    Performance: O(anchor + n_sweep + counterfactual). With defaults this is a
        large local ABM run; anchor replicates default to five seeds while
        counterfactual replicates use ``K``.
    Side effects: Reads the local forecast CSV and DB, runs local ABM
        simulations, writes ``output_path`` and a sibling counterfactual JSON;
        no network/API calls.
    Caller responsibility: Check ``anchor['degenerate']`` and every sweep row's
        ``degenerate`` before interpreting the ABM as epidemiologically valid.
    """
    if K <= 0:
        raise ValueError("K must be positive")
    output = Path(output_path)
    # forecast→ABM SSOT: prefer the live P1 real_forecaster forecast (operational
    # best_model on the real slab). The default champion does NOT forecast the
    # real slab (extrapolation collapse → REAL_FORECAST_STABILITY), so the
    # champion default maps to real_eval best_model (None). Fall back to the
    # legacy R10 per_model_eval CSV only if real_eval output is absent.
    try:
        weeks, forecast = load_real_forecast(
            model_name=None if model_name == DEFAULT_MODEL else model_name
        )
    except (FileNotFoundError, ValueError):
        weeks, forecast = load_forecast(model_name)
    seed_tuple = _seed_tuple(seeds)
    season_year = _latest_year(DB_PATH) if year is None else int(year)
    anchor = anchor_abm_to_forecast(
        forecast,
        n_agents=int(n_agents),
        seeds=seed_tuple,
        year=season_year,
    )
    sweep_values = (
        (10_000, 20_000, 37_500, 75_000)
        if n_sweep_values is None
        else tuple(int(n) for n in n_sweep_values)
    )
    sweep = n_sweep(
        forecast,
        n_values=sweep_values,
        seeds=seed_tuple,
        year=season_year,
    )
    disease = {
        **DEFAULT_DISEASE,
        **{k: float(v) for k, v in anchor["fitted_forcing"].items()},
    }
    counterfactual_path = output.with_name(f"{output.stem}_counterfactual.json")
    counterfactual = run_counterfactual(
        K=int(K),
        n_agents=int(n_agents),
        year=season_year,
        disease=disease,
        behaviour=BEHAVIOUR_OFF,
        output_path=counterfactual_path,
    )
    result = {
        "model_name": model_name,
        "forecast": {
            "weeks": [int(w) for w in weeks],
            "y_pred": [float(v) for v in forecast],
        },
        "anchor": anchor,
        "n_sweep": sweep,
        "counterfactual": counterfactual,
        "metadata": {
            # ★ behavior provenance (외부평가 권고): 이 산출이 내생 행동(behavioural.py
            # dR/dt=α·I/N)이 아니라 *forecast-anchored 운영 forcing*임을 명시 태깅 →
            # 독자가 행동 메커니즘과 예측-앵커 forcing을 혼동하지 않게.
            "behavior_mode": "forecast_anchored",
            "anchor_model": str(model_name),
            "n_agents": int(n_agents),
            "K": int(K),
            "seeds": [int(s) for s in seed_tuple],
            "year": int(season_year),
            "predictions_csv": str(DEFAULT_PATH),
            "output_path": str(output),
            "counterfactual_output_path": str(counterfactual_path),
            "local_only": True,
        },
    }
    _write_json(result, output)
    return result


def _seed_tuple(seeds: Sequence[int] | None) -> tuple[int, ...]:
    seed_tuple = tuple(range(5)) if seeds is None else tuple(int(s) for s in seeds)
    if not seed_tuple:
        raise ValueError("at least one seed is required")
    return seed_tuple


def _validate_forecast(forecast: np.ndarray) -> np.ndarray:
    y = np.asarray(forecast, dtype=np.float64)
    if y.ndim != 1 or y.size == 0:
        raise ValueError("forecast must be a non-empty 1D array")
    if not np.all(np.isfinite(y)):
        raise ValueError("forecast contains non-finite values")
    if np.any(y < 0.0):
        raise ValueError("forecast must be non-negative")
    return y


def _finite_corr(x: np.ndarray, y: np.ndarray) -> float:
    xv = np.asarray(x, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(xv) & np.isfinite(yv)
    if int(mask.sum()) < 2:
        return 0.0
    xs = float(np.std(xv[mask]))
    ys = float(np.std(yv[mask]))
    if xs <= 1.0e-12 or ys <= 1.0e-12:
        return 0.0
    corr = float(np.corrcoef(xv[mask], yv[mask])[0, 1])
    return corr if np.isfinite(corr) else 0.0


def _latest_year(db_path: Path) -> int:
    try:
        seasons = _load_ili_seasons(db_path)
        return int(max(s.season for s in seasons))
    except Exception:
        with safe_connect(str(db_path), verify=False) as conn:
            row = conn.execute(
                "SELECT MAX(season_start) AS season_start FROM sentinel_influenza"
            ).fetchone()
        if row is None or row["season_start"] is None:
            raise ValueError("could not infer latest season from DB")
        return int(row["season_start"])


def _write_json(obj: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True)


if __name__ == "__main__":
    print(json.dumps(run_forecast_anchored()["anchor"], indent=2))
