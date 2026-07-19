#!/usr/bin/env python3
"""Swap thesis figures found BY THEIR CAPTION, letterboxing each to the frame it already occupies.

A block index drifts as the document is edited; a caption ("Figure 24.1. …") does not. For each
(caption-prefix, replacement PNG) this finds the caption in the body, takes the image block just
above it, and overwrites that image part's bytes — leaving ``wp:extent`` untouched so the printed
size, and therefore the page map, cannot move. If the new PNG's aspect differs from the frame it is
padded with white (letterboxed), never stretched, so no drawn pixel is distorted.

Run:
    .venv/bin/python scripts/thesis_swap_by_caption.py --dry-run
    .venv/bin/python scripts/thesis_swap_by_caption.py --tag uniformity
"""
from __future__ import annotations

import argparse
import io
import re
import shutil
import sys
from pathlib import Path

import docx
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
_DOCX = _ROOT / "paper" / "고려대_보건석사학위논문_이승진_2026.docx"
_W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_WP = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"
_TOL = 0.005

# caption prefix (must match the body caption's start) -> replacement PNG
SWAPS: list[tuple[str, str]] = [
    ("Figure 1. System overview",          "paper/methods_assets/fig_system_overview.png"),
    ("Figure 3. SEIR-V-D",                 "paper/methods_assets/fig_seirvd.png"),
    ("Figure 19.1. Down-scaling",          "paper/results_assets/fig12_1_downscale_mechanism.png"),
    ("Figure 19.2. Indirect",              "paper/results_assets/fig12_2_downscale_check.png"),
    ("Figure 23.1. Two representative",    "paper/results_assets/fig16_1_representative_agents.png"),
    ("Figure 23.2. Infected-population",   "paper/results_assets/fig16_2_population_distributions.png"),
    ("Figure 24.1. Seasonal-shape",        "paper/results_assets/fig17_1_seasonal_shape_overlay.png"),
    ("Figure 24.2. Per-region",            "paper/results_assets/fig17_2_per_region_raw_grid.png"),
    ("Figure 26.1. ARIA numeric",          "paper/results_assets/fig19_1_aria_numeric_grounding.png"),
    ("Figure 26.2. ARIA Self-Ask",         "paper/results_assets/fig19_2_aria_selfask.png"),
    ("Figure 28.1. Profile likelihoods",   "paper/results_assets/fig20_1_profile_likelihoods.png"),
    ("Figure 28.2. ABC-SMC",               "paper/results_assets/fig20_2_posterior_correlation.png"),
]


def _text(b) -> str:
    return "".join(t.text or "" for t in b.iter(_W + "t")).strip()


def _fit(png: Path, box: float) -> tuple[bytes, float]:
    img = Image.open(png).convert("RGB")
    w, h = img.size
    dist = box / (w / h) - 1.0
    if abs(dist) <= _TOL:
        return png.read_bytes(), dist
    nw, nh = (int(round(h * box)), h) if (w / h) < box else (w, int(round(w / box)))
    canvas = Image.new("RGB", (nw, nh), "white")
    canvas.paste(img, ((nw - w) // 2, (nh - h) // 2))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue(), dist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tag", default="uniformity")
    args = ap.parse_args()

    d = docx.Document(str(_DOCX))
    kids = list(d.element.body.iterchildren())

    plan = []
    for prefix, png in SWAPS:
        src = _ROOT / png
        if not src.exists():
            print(f"✗ missing PNG: {png}")
            return 1
        # find the body caption (skip the LoF cache: those lines end in a page number)
        cap_i = None
        for i, b in enumerate(kids):
            if i < 300:
                continue
            t = _text(b)
            if t.startswith(prefix) and not re.search(r"\d{1,3}$", t.replace(".", "")):
                cap_i = i
                break
            if t.startswith(prefix):        # accept even with trailing digits if nothing better
                cap_i = cap_i or i
        if cap_i is None:
            print(f"✗ caption not found: {prefix!r}")
            return 1
        # image block just above the caption
        img_blk = None
        for j in range(cap_i - 1, cap_i - 5, -1):
            if list(kids[j].iter(_A + "blip")):
                img_blk = kids[j]
                break
        if img_blk is None:
            print(f"✗ no image above caption: {prefix!r}")
            return 1
        part = d.part.related_parts[next(img_blk.iter(_A + "blip")).get(_R + "embed")]
        ext = next(img_blk.iter(_WP + "extent"))
        box = int(ext.get("cx")) / int(ext.get("cy"))
        blob, dist = _fit(src, box)
        plan.append((prefix, part, blob, box, dist, len(part.blob)))

    print(f"{'캡션':<34}{'프레임비':>8}{'letterbox':>10}  {'KB':>10}")
    for prefix, part, blob, box, dist, oldn in plan:
        lb = f"{dist * 100:+.1f}%" if abs(dist) > _TOL else "—"
        print(f"{prefix[:33]:<34}{box:>8.3f}{lb:>10}  {oldn // 1024}→{len(blob) // 1024}")

    if args.dry_run:
        print("\n(dry run — 저장 안 함)")
        return 0

    shutil.copy2(_DOCX, _DOCX.with_name(f"{_DOCX.stem}_pre_{args.tag}.docx"))
    for _p, part, blob, _b, _d, _o in plan:
        part._blob = blob
    d.save(str(_DOCX))
    print(f"\n✅ {len(plan)}장 교체. wp:extent 불변. 백업 _pre_{args.tag}.docx")
    print("   next: .venv/bin/python scripts/thesis_toc_check.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
