"""simulation.scripts.fig_multidisease.

Two observational, deterministic publication figures on respiratory/enteric
pathogen co-circulation, built **only** from real data in the project DB
(``simulation/data/db/epi_real_seoul.db``, opened read-only). No synthetic or
model-derived values are introduced — every series is a direct sentinel /
laboratory observation.

Figures
-------
1. ``fig_multidisease_cocirculation.png``
   Weekly multi-pathogen co-circulation overlay. Five sentinel/lab streams
   (ARI total, SARI, HFMD, enterovirus, influenza ILI) are placed on a common
   ISO ``(year, week)`` axis. Because the streams carry **different native
   units** (case counts vs. rates), each is z-score normalized **per series**
   so the *timing* of seasonal waves is comparable; the y-axis is therefore
   "standardized weekly level (z-score)", not an absolute magnitude.

2. ``fig_multidisease_flu_composition.png``
   Influenza (sub)type composition stacked-area chart for **Korea at the
   national level** (WHO FluNet, ``country='Republic of Korea'``; *not* Seoul).
   Weekly subtype counts (A/H1N1pdm09, A/H3, A/H1(seasonal), B/Victoria,
   B/Yamagata, B/undetermined, A/other-or-unsubtyped) are converted to within-
   week fractions of typed+counted detections, exposing dominant-strain
   turnover across seasons.

Honesty
-------
- The 'influenza ILI' stream in Fig 1 is the **mean across 7 age bands** of the
  KDCA sentinel ILI rate (the source has no single all-age aggregate); this is
  stated in the figure annotation.
- Fig 2 is **national (Korea)**, explicitly labelled, distinct from the rest of
  the Seoul-focused project.
- If a required table/column is missing or empty, the affected figure is
  skipped with a logged reason rather than fabricated.

CLI
---
    .venv/bin/python -m simulation.scripts.fig_multidisease
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from sqlite3 import Connection
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ----------------------------------------------------------------------------
# Determinism + portable CJK font (AppleGothic → NanumGothic → others) (#5, #1).
# ----------------------------------------------------------------------------
np.random.seed(42)
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

logging.basicConfig(level=logging.INFO, format="[fig_multidisease] %(message)s")
log = logging.getLogger("fig_multidisease")

# Project-rooted paths (portable — no cwd assumptions) (#1).
_THIS = Path(__file__).resolve()
_ROOT = _THIS.parents[2]  # .../MPH_infection_simulation
DB_PATH = _ROOT / "simulation" / "data" / "db" / "epi_real_seoul.db"
FIG_DIR = _ROOT / "simulation" / "results" / "figures"

# Colour-blind-safe palette (Okabe–Ito ordering) for the five pathogens.
_PATHOGEN_COLORS = {
    "ARI (acute respiratory, total)": "#0072B2",
    "SARI (severe acute respiratory)": "#D55E00",
    "HFMD (hand-foot-mouth)": "#009E73",
    "Enterovirus": "#CC79A7",
    "Influenza ILI": "#E69F00",
}


def _connect_readonly(db_path: Path) -> Connection:
    """Open the project DB strictly read-only via a SQLite file URI.

    Args:
        db_path: Absolute path to the SQLite database file.

    Returns:
        An open ``Connection`` in ``mode=ro`` (writes raise).

    Raises:
        FileNotFoundError: If ``db_path`` does not exist on disk.

    Side effects: opens a read-only DB handle (caller must close).
    Caller responsibility: close the returned connection.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    from simulation.database import read_only_connect
    return read_only_connect(str(db_path))


def _iso_week_from_label(label: str) -> int:
    """Extract the ISO week integer from a Korean week label (e.g. '36주').

    Args:
        label: Week label string such as ``'36주'`` or ``'01주'``.

    Returns:
        The integer ISO week (1–53); 0 if no digits are present.

    Performance: O(len(label)).
    """
    digits = re.sub(r"[^0-9]", "", label or "")
    return int(digits) if digits else 0


def _zscore(values: np.ndarray) -> np.ndarray:
    """Z-score normalize, returning zeros if the series has no spread.

    Args:
        values: 1-D float array of a single pathogen's weekly levels.

    Returns:
        Array of identical shape: ``(values - mean) / std`` (std>0), else zeros.

    Performance: O(n). Side effects: none (pure).
    """
    vals = np.asarray(values, dtype=float)
    sd = float(np.nanstd(vals))
    if not np.isfinite(sd) or sd == 0.0:
        return np.zeros_like(vals)
    return (vals - float(np.nanmean(vals))) / sd


# ----------------------------------------------------------------------------
# Data loaders — each returns {(year, iso_week): value} from real observations.
# ----------------------------------------------------------------------------
def _load_ari_total(con: Connection) -> dict[tuple[int, int], float]:
    """Weekly total ARI sentinel count (``pathogen_group='계'``, the row total).

    Returns:
        Mapping ``(year, week_no) -> total count`` (empty if table absent).
    """
    cur = con.execute(
        "SELECT year, week_no, count FROM sentinel_ari "
        "WHERE pathogen_nm='계' AND count IS NOT NULL"
    )
    return {(int(y), int(w)): float(c) for y, w, c in cur.fetchall()}


def _load_sari(con: Connection) -> dict[tuple[int, int], float]:
    """Weekly SARI (severe acute respiratory infection) sentinel count."""
    cur = con.execute(
        "SELECT year, week_no, count FROM sentinel_sari WHERE count IS NOT NULL"
    )
    return {(int(y), int(w)): float(c) for y, w, c in cur.fetchall()}


def _load_hfmd(con: Connection) -> dict[tuple[int, int], float]:
    """Weekly hand-foot-and-mouth disease sentinel rate (per-sentinel rate)."""
    cur = con.execute(
        "SELECT year, week_no, rate FROM sentinel_hfmd WHERE rate IS NOT NULL"
    )
    return {(int(y), int(w)): float(r) for y, w, r in cur.fetchall()}


def _load_enterovirus(con: Connection) -> dict[tuple[int, int], float]:
    """Weekly enterovirus surveillance count."""
    cur = con.execute(
        "SELECT year, week_no, count FROM sentinel_enterovirus WHERE count IS NOT NULL"
    )
    return {(int(y), int(w)): float(c) for y, w, c in cur.fetchall()}


def _load_influenza_ili(con: Connection) -> dict[tuple[int, int], float]:
    """Weekly influenza ILI rate, averaged across the 7 KDCA age bands.

    The source (``sentinel_influenza``) stores season-relative ``week_seq`` with
    ``season_start``; the calendar year is recovered from the ISO week label
    (weeks ≥36 belong to ``season_start``, weeks <36 roll into ``season_start+1``).

    Returns:
        Mapping ``(year, iso_week) -> mean ILI rate over age groups``.
    """
    cur = con.execute(
        "SELECT season_start, week_seq, week_label, ili_rate "
        "FROM sentinel_influenza WHERE ili_rate IS NOT NULL"
    )
    bucket: dict[tuple[int, int], list[float]] = defaultdict(list)
    for season_start, _week_seq, label, rate in cur.fetchall():
        iso = _iso_week_from_label(label)
        if iso == 0:
            continue
        year = int(season_start) if iso >= 36 else int(season_start) + 1
        bucket[(year, iso)].append(float(rate))
    return {k: float(np.mean(v)) for k, v in bucket.items() if v}


def _series_to_arrays(
    series: dict[tuple[int, int], float],
) -> tuple[np.ndarray, np.ndarray]:
    """Sort a ``(year, week)->value`` mapping into a continuous-index series.

    Args:
        series: Mapping keyed by ``(year, iso_week)``.

    Returns:
        ``(x, y)`` where ``x`` is a fractional year axis (year + week/52) and
        ``y`` the matching values, both sorted ascending by time.
    """
    keys = sorted(series.keys())
    x = np.array([yr + (wk - 1) / 52.0 for (yr, wk) in keys], dtype=float)
    y = np.array([series[k] for k in keys], dtype=float)
    return x, y


# ----------------------------------------------------------------------------
# Figure 1 — multi-pathogen co-circulation overlay.
# ----------------------------------------------------------------------------
def make_cocirculation_figure(con: Connection, out_path: Path) -> bool:
    """Render the weekly multi-pathogen co-circulation overlay (Fig 1).

    Each of five sentinel/lab streams is z-score normalized per series so that
    seasonal *timing* (not absolute magnitude, which differs by unit) is
    comparable on one axis.

    Args:
        con: Read-only DB connection.
        out_path: Destination PNG path.

    Returns:
        True if the figure was written; False if no usable data (skipped).

    Side effects: writes one PNG to ``out_path`` on success.
    """
    loaders = {
        "ARI (acute respiratory, total)": _load_ari_total,
        "SARI (severe acute respiratory)": _load_sari,
        "HFMD (hand-foot-mouth)": _load_hfmd,
        "Enterovirus": _load_enterovirus,
        "Influenza ILI": _load_influenza_ili,
    }
    streams: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, loader in loaders.items():
        raw = loader(con)
        if raw:
            streams[name] = _series_to_arrays(raw)
            log.info("co-circulation: '%s' loaded n=%d weeks", name, len(raw))
        else:
            log.warning("co-circulation: '%s' has no rows — omitted", name)

    if len(streams) < 2:
        log.warning(
            "co-circulation SKIPPED — only %d usable stream(s) (need >=2)",
            len(streams),
        )
        return False

    fig, ax = plt.subplots(figsize=(13, 6.2))
    for name, (x, y) in streams.items():
        ax.plot(
            x,
            _zscore(y),
            label=f"{name} (n={len(x)})",
            color=_PATHOGEN_COLORS.get(name, "#444444"),
            linewidth=1.6,
            alpha=0.9,
        )
    ax.axhline(0.0, color="#999999", linewidth=0.7, linestyle="--", alpha=0.6)

    ax.set_title(
        "Multi-disease co-circulation (Seoul region sentinel/lab surveillance) "
        "— weekly standardized level\n"
        "Multi-pathogen co-circulation, weekly z-score (observed surveillance, "
        "not model output)",
        fontsize=12.5,
        pad=12,
    )
    ax.set_xlabel("Year (ISO week, year + week/52)", fontsize=10.5)
    ax.set_ylabel("Standardized weekly level (z-score, per-series normalized)", fontsize=10.5)
    ax.legend(loc="upper left", fontsize=8.5, ncol=2, framealpha=0.9)
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.margins(x=0.01)

    note = (
        "Series with different units (counts vs rates) are normalized per series "
        "via z-score — for comparing the 'timing' of seasonal waves, not amplitude.\n"
        "Influenza ILI = mean over 7 age bands of the KDCA sentinel ILI rate (raw "
        "data has no all-age single value). SARI was collected for 2017–2020 only."
    )
    fig.text(0.012, 0.012, note, fontsize=7.6, color="#555555", va="bottom")
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


# ----------------------------------------------------------------------------
# Figure 2 — influenza subtype composition (WHO FluNet, Korea national).
# ----------------------------------------------------------------------------
# Display label -> DB column (only columns with non-trivial KOR signal kept).
_FLU_SUBTYPES: list[tuple[str, str]] = [
    ("A/H3", "inf_a_h3"),
    ("A/H1N1pdm09", "inf_a_h1n1pdm09"),
    ("A/H1 (seasonal)", "inf_a_h1"),
    ("A (other/unsubtyped)", "_a_other"),  # synthetic of remainder, see below
    ("B/Victoria", "inf_b_victoria"),
    ("B/Yamagata", "inf_b_yamagata"),
    ("B (undetermined)", "inf_b_notdetermined"),
]


def make_flu_composition_figure(con: Connection, out_path: Path) -> bool:
    """Render the influenza subtype composition stacked-area chart (Fig 2).

    National-level (Korea) WHO FluNet weekly subtype counts are converted to
    within-week fractions of all typed/counted detections, revealing dominant-
    strain turnover. ``A (other/unsubtyped)`` is the non-negative remainder of
    ``inf_a`` minus its resolved subtypes (so A always sums consistently).

    Args:
        con: Read-only DB connection.
        out_path: Destination PNG path.

    Returns:
        True if the figure was written; False if no usable data (skipped).

    Side effects: writes one PNG to ``out_path`` on success.
    """
    cur = con.execute(
        "SELECT year, week_no, "
        "inf_a, inf_a_h3, inf_a_h1n1pdm09, inf_a_h1, "
        "inf_b_victoria, inf_b_yamagata, inf_b_notdetermined "
        "FROM who_flunet WHERE country='Republic of Korea' "
        "ORDER BY year, week_no"
    )
    rows = cur.fetchall()
    if not rows:
        log.warning("flu_composition SKIPPED — no WHO FluNet KOR rows")
        return False

    # Aggregate by (year, week) — collapse multiple origin_source rows by sum.
    agg: dict[tuple[int, int], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for (yr, wk, a, a_h3, a_h1pdm, a_h1, b_vic, b_yam, b_und) in rows:
        k = (int(yr), int(wk))
        a = float(a or 0)
        a_h3 = float(a_h3 or 0)
        a_h1pdm = float(a_h1pdm or 0)
        a_h1 = float(a_h1 or 0)
        # Non-negative remainder of A not attributed to resolved subtypes.
        a_other = max(a - (a_h3 + a_h1pdm + a_h1), 0.0)
        d = agg[k]
        d["inf_a_h3"] += a_h3
        d["inf_a_h1n1pdm09"] += a_h1pdm
        d["inf_a_h1"] += a_h1
        d["_a_other"] += a_other
        d["inf_b_victoria"] += float(b_vic or 0)
        d["inf_b_yamagata"] += float(b_yam or 0)
        d["inf_b_notdetermined"] += float(b_und or 0)

    keys = sorted(agg.keys())
    x = np.array([yr + (wk - 1) / 52.0 for (yr, wk) in keys], dtype=float)

    # Build fraction matrix; weeks with zero total are dropped (no detections).
    raw_counts = np.zeros((len(_FLU_SUBTYPES), len(keys)), dtype=float)
    for j, k in enumerate(keys):
        for i, (_lbl, col) in enumerate(_FLU_SUBTYPES):
            raw_counts[i, j] = agg[k][col]
    totals = raw_counts.sum(axis=0)
    keep = totals > 0
    if not keep.any():
        log.warning("flu_composition SKIPPED — all KOR weekly totals are zero")
        return False
    x = x[keep]
    fracs = raw_counts[:, keep] / totals[keep]
    log.info(
        "flu_composition: %d KOR weeks with detections (of %d)",
        int(keep.sum()),
        len(keys),
    )

    # Influenza A = warm hues, B = cool hues (visually separates the two types).
    flu_colors = [
        "#7F0000",  # A/H3
        "#D7301F",  # A/H1N1pdm09
        "#FC8D59",  # A/H1 seasonal
        "#FDCC8A",  # A other/unsubtyped
        "#08519C",  # B/Victoria
        "#3182BD",  # B/Yamagata
        "#9ECAE1",  # B undetermined
    ]
    labels = [lbl for lbl, _ in _FLU_SUBTYPES]

    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.stackplot(x, fracs, labels=labels, colors=flu_colors, alpha=0.92)
    ax.set_ylim(0, 1)
    ax.margins(x=0.01)

    ax.set_title(
        "Influenza subtype composition — national level (Republic of Korea, WHO FluNet)\n"
        "Influenza subtype composition, NATIONAL (Republic of Korea) — "
        "not Seoul; observed lab detections",
        fontsize=12.5,
        pad=12,
    )
    ax.set_xlabel("Year (ISO week, year + week/52)", fontsize=10.5)
    ax.set_ylabel("Weekly detection fraction (fraction of typed/counted detections)", fontsize=10.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=4, fontsize=8.5,
              framealpha=0.9)
    ax.grid(True, axis="y", alpha=0.2, linewidth=0.5)

    note = (
        "Source: WHO FluNet, country='Republic of Korea' (national level; not Seoul). "
        "'A (other/unsubtyped)' = non-negative remainder of inf_a − (H3+H1N1pdm09+H1).\n"
        "Weeks with zero detections excluded. All values are observed lab detections "
        "(not model output)."
    )
    fig.text(0.012, 0.012, note, fontsize=7.6, color="#555555", va="bottom")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return True


def main() -> None:
    """Generate both multi-disease figures from real DB observations.

    Side effects: writes up to two PNGs under
    ``simulation/results/figures/``; logs a skip reason for any figure whose
    source data is unavailable. Never fabricates data.
    """
    log.info("DB: %s", DB_PATH)
    log.info("OUT: %s", FIG_DIR)
    con = _connect_readonly(DB_PATH)
    try:
        f1 = FIG_DIR / "fig_multidisease_cocirculation.png"
        f2 = FIG_DIR / "fig_multidisease_flu_composition.png"
        ok1 = make_cocirculation_figure(con, f1)
        ok2 = make_flu_composition_figure(con, f2)
    finally:
        con.close()

    for ok, path in ((ok1, f1), (ok2, f2)):
        if ok and path.exists() and path.stat().st_size > 0:
            log.info("WROTE %s (%d bytes)", path, path.stat().st_size)
        elif ok:
            log.error("CLAIMED-OK but missing/empty: %s", path)
        else:
            log.warning("SKIPPED (no data): %s", path)


if __name__ == "__main__":
    main()
