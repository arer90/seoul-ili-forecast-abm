"""Figure 11 — Observed vs predicted Seoul ILI (champion FusedEpi, 1-step-ahead) with a DATE x-axis.

The held-out test span is the last 68 in-sample weeks (2024-11 .. 2026-02); the original figure
showed only a week index, so the calendar dates are now on the axis and named in the caption.

Data: simulation/results/_archive_fullrun_20260701_024145/csv/predictions_FusedEpi.csv (test split).
Run:  .venv/bin/python paper/methods_assets/fig_observed_predicted.py
Out:  paper/methods_assets/fig_observed_predicted.png
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 200})
NAVY = "#1e3a5f"; TEAL = "#2f8f7d"

ROOT = Path(__file__).resolve().parents[2]
d = pd.read_csv(ROOT / "simulation/results/_archive_fullrun_20260701_024145/csv/predictions_FusedEpi.csv")
t = d[d["split"] == "test"].reset_index(drop=True)
# held-out test = last 68 in-sample weeks; calendar span 2024-11-03 .. 2026-02 (weekly, Sunday)
dates = pd.date_range("2024-11-03", periods=len(t), freq="W-SUN")

fig, ax = plt.subplots(figsize=(13, 6.4))
ax.plot(dates, t["y_true"], "-o", color=NAVY, ms=4.2, lw=1.8, label="Observed ILI")
ax.plot(dates, t["y_pred"], "-s", color=TEAL, ms=4.2, lw=1.8, label="FusedEpi 1-step forecast")

ax.set_title("Observed vs predicted Seoul ILI — champion FusedEpi 1-step-ahead forecast",
             fontsize=14, fontweight="bold", color="#1f2937", pad=10)
ax.set_ylabel("ILI rate (per 1,000 outpatient visits)", fontsize=11)
ax.set_xlabel("Held-out test week  (2024-11 to 2026-02,  n = 68 weeks)", fontsize=11)

ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9.5)

ax.text(0.015, 0.94, "R-squared = 0.936    MAE = 3.90    RMSE = 6.59    WIS = 3.28",
        transform=ax.transAxes, fontsize=10.5, va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor=TEAL, linewidth=1.3))
ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
ax.grid(alpha=0.25, lw=0.7)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
ax.set_ylim(0, None)

fig.tight_layout()
OUT = Path(__file__).resolve().parent / "fig_observed_predicted.png"
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"wrote {OUT}")
