"""Seoul 25-district choropleth maps of the agent-based spatial outputs.

Makes the ABM's SPATIAL resolution visible as a MAP (not a ranked bar chart): each
district is coloured by a per-district quantity the commuter-coupled agent model
produces — commuter import fraction and target reproduction load — so the spatial
structure (central-business-district transmission hubs vs self-contained residential
districts) is legible at a glance.

Pure matplotlib polygons from ``seoul_gu.geojson`` (no geopandas — keeps the plot
dependency-light and portable). Run:
  .venv/bin/python -m simulation.scripts.fig_seoul_choropleth
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon as MplPolygon

GEO = Path("simulation/data/external/seoul_gu.geojson")
OUT = Path("simulation/results/figures")
_ROM = {"강남구": "Gangnam", "강동구": "Gangdong", "강북구": "Gangbuk", "강서구": "Gangseo",
        "관악구": "Gwanak", "광진구": "Gwangjin", "구로구": "Guro", "금천구": "Geumcheon",
        "노원구": "Nowon", "도봉구": "Dobong", "동대문구": "Dongdaemun", "동작구": "Dongjak",
        "마포구": "Mapo", "서대문구": "Seodaemun", "서초구": "Seocho", "성동구": "Seongdong",
        "성북구": "Seongbuk", "송파구": "Songpa", "양천구": "Yangcheon", "영등포구": "Yeongdeungpo",
        "용산구": "Yongsan", "은평구": "Eunpyeong", "종로구": "Jongno", "중구": "Jung",
        "중랑구": "Jungnang"}


def _rings(geom: dict):
    """Yield each polygon's exterior ring as an (N,2) array (Polygon + MultiPolygon)."""
    t, c = geom["type"], geom["coordinates"]
    polys = [c] if t == "Polygon" else c
    for poly in polys:
        yield np.asarray(poly[0], dtype=float)      # exterior ring


def _load_districts():
    feats = json.loads(GEO.read_text(encoding="utf-8"))["features"]
    out = []
    for f in feats:
        name = f["properties"]["name"]
        rings = list(_rings(f["geometry"]))
        out.append((name, rings))
    return out


def choropleth(ax, values: dict, title: str, cmap="OrRd", label_top: int = 6):
    """Fill each Seoul district by ``values[name]`` on ``ax``; label the top districts."""
    districts = _load_districts()
    vals = np.array([values.get(n, np.nan) for n, _ in districts])
    vmin, vmax = np.nanmin(vals), np.nanmax(vals)
    norm = plt.Normalize(vmin, vmax)
    cm = matplotlib.colormaps[cmap]
    patches, pcolors, centroids = [], [], []
    for (name, rings), v in zip(districts, vals):
        col = cm(norm(v)) if np.isfinite(v) else (0.9, 0.9, 0.9, 1)
        big = max(rings, key=lambda r: r.shape[0])
        centroids.append((name, big[:, 0].mean(), big[:, 1].mean(), v))
        for ring in rings:
            patches.append(MplPolygon(ring, closed=True))
            pcolors.append(col)
    pc = PatchCollection(patches, facecolors=pcolors, edgecolors="white", linewidths=0.6)
    ax.add_collection(pc)
    # label the highest-value districts (the hubs the reader cares about)
    top = sorted([c for c in centroids if np.isfinite(c[3])], key=lambda c: -c[3])[:label_top]
    topset = {c[0] for c in top}
    for name, x, y, v in centroids:
        if name in topset:
            ax.text(x, y, f"{_ROM.get(name, name)}\n{v:.2f}", ha="center", va="center",
                    fontsize=6.5, fontweight="bold", color="#111")
    ax.autoscale_view(); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title, fontsize=11, fontweight="bold")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    plt.colorbar(sm, ax=ax, fraction=0.038, pad=0.02)


_AGE_LABELS = {0: "0-9", 1: "10-19", 2: "20-29", 3: "30-39", 4: "40-49",
               5: "50-59", 6: "60+"}


def _age_panel(ax, age_attack: dict) -> None:
    keys = sorted(int(k) for k in age_attack)
    labels = [_AGE_LABELS.get(k, str(k)) for k in keys]
    vals = [age_attack[str(k)] if str(k) in age_attack else age_attack[k] for k in keys]
    peak = int(np.argmax(vals))
    colors = ["#C44E52" if i == peak else "#4C72B0" for i in range(len(keys))]
    ax.bar(labels, vals, color=colors, edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.002, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_title("(c) Attack rate by age band\n(school-age 10-19 highest, elderly 60+ lowest)",
                 fontsize=11, fontweight="bold")
    ax.set_ylabel("share ever infected"); ax.tick_params(axis="x", rotation=30)
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    cn = json.loads(Path("simulation/results/commuter_ngm.json").read_text(encoding="utf-8"))
    age = json.loads(Path("simulation/results/abm_age_attack.json").read_text(
        encoding="utf-8"))["age_band_attack_rate"]
    imp = cn["import_fraction"]
    tgt = cn["district_in"]
    fig = plt.figure(figsize=(17.5, 6.2))
    ax0 = fig.add_subplot(1, 3, 1); ax1 = fig.add_subplot(1, 3, 2); ax2 = fig.add_subplot(1, 3, 3)
    choropleth(ax0, imp,
               "(a) Commuter import fraction by district\n(infection pressure from other districts)",
               cmap="OrRd")
    choropleth(ax1, tgt,
               "(b) Target reproduction load by district\n(secondary infections received per infectious)",
               cmap="YlGnBu")
    _age_panel(ax2, age)
    fig.suptitle("Seoul agent-based transmission surface — space × demographics "
                 f"(25-district commuter-coupled NGM, R_eff={cn['r_eff']:.2f}; leak-free)",
                 fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "seoul_district_choropleth.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("map →", OUT / "seoul_district_choropleth.png")


if __name__ == "__main__":
    main()
