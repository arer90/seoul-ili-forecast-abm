"""fig17_overseas_split.py — Figure 17 split + enlarged (Seoul vs overseas ILI shape).

PURPOSE
    The thesis Figure 17 ("Seoul ILI shares its seasonal shape with comparable
    surveillance regions") stacked a full-width z-score overlay, a small 3x3
    grid of per-region raw panels, and an (omitted) champion panel into one
    image, leaving a large empty band and tiny, unreadable region panels. This
    regenerator SPLITS the same real-data SSOT into two enlarged figures:

      Figure 24.1  Seasonal-SHAPE overlay (Panel A): z-score normalized weekly
                   curves for Seoul and each comparable region on one axis
                   (phase / relative-amplitude comparison only).
      Figure 24.2  Per-region RAW series grid (Panel B): each region on its own
                   native-unit axis, enlarged, plus the Seoul raw panel. Units
                   differ by source so absolute values are NOT compared.

DATA SSOT (read-only, measured — no fabrication; champion overlay deliberately
omitted because units/alignment differ, exactly as the source figure):
    simulation/data/db/epi_real_seoul.db
      sentinel_influenza (Seoul all-age) + overseas_ili (8 source/country series)

This reuses the loaders + panel helpers from the original source figure
``simulation/scripts/fig_overseas_compare.py`` (single SSOT). Only the LAYOUT
changes (split + enlarge). No values are altered.

Output (PNG, white bg, dpi=160):
    paper/results_assets/fig17_1_seasonal_shape_overlay.png
    paper/results_assets/fig17_2_per_region_raw_grid.png
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from simulation.scripts.fig_overseas_compare import (  # noqa: E402
    DB_PATH,
    PANEL_SOURCES,
    _panel_overview,
    _panel_region,
    _panel_seoul_raw,
    _set_korean_font,
    load_overseas_series,
    load_seoul_all_age_ili,
)

OUT_DIR = _THIS.parent
OUT_1 = OUT_DIR / "fig17_1_seasonal_shape_overlay.png"
OUT_2 = OUT_DIR / "fig17_2_per_region_raw_grid.png"


def _load() -> tuple[dict, list, object]:
    seoul = load_seoul_all_age_ili(DB_PATH)
    if seoul["ili"].size == 0:
        raise SystemExit("[fig17] Seoul ILI absent — honest skip (no fabricated data).")
    seoul_start = seoul["dates"][0]
    overseas: list[tuple[str, str, str, dict]] = []
    for source, country, label in PANEL_SOURCES:
        s = load_overseas_series(DB_PATH, source, country)
        if s["ili"].size < 30:
            print(f"[fig17] skip {source}/{country}: n={s['ili'].size} (<30, sparse)")
            continue
        overseas.append((source, country, label, s))
        print(f"[fig17] {label}: n={s['ili'].size} weeks")
    if not overseas:
        raise SystemExit("[fig17] no comparable overseas series — honest skip.")
    return seoul, overseas, seoul_start


def make_overlay(seoul: dict, overseas: list, seoul_start) -> None:
    """Figure 24.1 — enlarged z-score seasonal-shape overlay (Panel A)."""
    fig, ax = plt.subplots(figsize=(15.5, 7.5))
    _panel_overview(ax, seoul, overseas, seoul_start, standalone=True)
    # Enlarge fonts that the shared helper sets small.
    ax.title.set_fontsize(14)
    ax.yaxis.label.set_fontsize(13)
    ax.tick_params(labelsize=11)
    for lh in ax.get_legend().get_texts():
        lh.set_fontsize(10)
    fig.suptitle(
        "Figure 24.1  Seoul vs comparable overseas regions: seasonal-shape overlay "
        "(z-score normalized, observed only)",
        fontsize=15, fontweight="bold")
    fig.text(0.5, 0.005,
             "Phase / relative-amplitude comparison only - NOT absolute magnitude and NOT forecast accuracy. "
             "No model is transferred across regions.",
             ha="center", va="bottom", fontsize=10, color="#555555")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(OUT_1, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {OUT_1}  (n_regions={len(overseas)})")


def _seoul_shape_corr(seoul: dict, region: dict):
    """z-normalized shape correlation vs Seoul on overlapping dates (transferability characteristic)."""
    try:
        import numpy as _np
        import pandas as _pd
        a = _pd.Series(_np.asarray(seoul["ili"], float),
                       index=_pd.to_datetime(seoul["dates"])).resample("W").mean()
        b = _pd.Series(_np.asarray(region["ili"], float),
                       index=_pd.to_datetime(region["dates"])).resample("W").mean()
        j = _pd.concat([a, b], axis=1, join="inner").dropna()
        if len(j) < 12 or j.iloc[:, 0].std() == 0 or j.iloc[:, 1].std() == 0:
            return None
        return float(_np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1])
    except Exception:
        return None


def make_grid(seoul: dict, overseas: list, seoul_start) -> None:
    """Figure 24.2 — enlarged per-region raw grid (Panel B) + Seoul raw."""
    import numpy as _np
    n = len(overseas) + 1  # +1 for Seoul raw
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 4.4 * nrow))
    axes = axes.ravel()
    for i, (_src, _c, label, s) in enumerate(overseas):
        _panel_region(axes[i], i, label, s, seoul_start, standalone=True)
        r = _seoul_shape_corr(seoul, s)
        try:
            peak = float(_np.nanmax(_np.asarray(s["ili"], float)))
        except Exception:
            peak = None
        extra = []
        if r is not None:
            extra.append(f"shape r vs Seoul = {r:+.2f}")
        if peak is not None:
            extra.append(f"peak {peak:.0f}")
        if extra:
            axes[i].set_title(label + "\n" + "   ".join(extra), fontsize=10.5)
        axes[i].title.set_fontsize(11)
    _panel_seoul_raw(axes[len(overseas)], seoul, seoul_start, standalone=True)
    axes[len(overseas)].title.set_fontsize(12)
    for k in range(n, len(axes)):  # hide unused cells
        axes[k].axis("off")
    fig.suptitle(
        "Figure 24.2  Per-region raw ILI series (each on its native-unit axis; observed only)\n"
        "WHO FluNet = ILI activity/detection index (0-100); Delphi/ILINet US = outpatient ILI %; "
        "Seoul = per 1,000 outpatient visits - do NOT compare absolute values across regions",
        fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(OUT_2, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[OK] {OUT_2}  (n_panels={n})")


def main() -> int:
    font = _set_korean_font()
    print(f"[fig17] font={font}")
    seoul, overseas, seoul_start = _load()
    make_overlay(seoul, overseas, seoul_start)
    make_grid(seoul, overseas, seoul_start)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
