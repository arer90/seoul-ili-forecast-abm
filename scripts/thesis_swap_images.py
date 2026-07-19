#!/usr/bin/env python3
"""Swap embedded thesis images in place, keeping the printed size byte-for-byte.

A figure is replaced by overwriting the bytes of its image *part* — never by deleting the
drawing and inserting a new one. The ``wp:extent`` that decides how large Word prints the
picture lives in the paragraph, not in the part, so leaving the drawing untouched is what
guarantees the page lock survives: the picture occupies exactly the same box it did before,
whatever the new PNG's pixel dimensions are.

Two things this refuses to do, because both silently break a thesis:

  * touch a part that more than one drawing references — overwriting it would change every
    figure that shares it;
  * proceed if the block does not hold exactly one image, which means the caller's block index
    is wrong and some other figure is about to be destroyed.

Run:
    .venv/bin/python scripts/thesis_swap_images.py --dry-run
    .venv/bin/python scripts/thesis_swap_images.py --tag figs
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
from pathlib import Path

import docx
from PIL import Image

_ROOT = Path(__file__).resolve().parents[1]
_DOCX = _ROOT / "paper" / "고려대_보건석사학위논문_이승진_2026.docx"

_A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"
_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_WP = "{http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing}"

# (body block index, replacement PNG, what it is)
SWAPS: list[tuple[int, str, str]] = [
    (487, "simulation/results/figures/fig4_forest_plot_wis.png",
     "Figure 12 — WIS forest; drops the stale joint-title-holder axis label"),
    (532, "paper/ch4_new_assets/fig_B_dm_tieset_forest.png",
     "Figure 14 — DM tie-set forest; NegBinGLM ΔL recomputed on the log-link GLM"),
    (648, "paper/methods_assets/fig_integrated_architecture.png",
     "Figure 27 — architecture; champion podium showed NegBinGLM, and 'gu' survived"),
    (1346, "simulation/results/plots_matplotlib/pred_vs_actual_NegBinGLM.png",
     "Appendix P card — NegBinGLM; title carried the archive R²/RMSE"),
    (1349, "simulation/results/plots_matplotlib/pred_vs_actual_PoissonAutoreg.png",
     "Appendix P card — PoissonAutoreg; same"),
]


def _blips(block):
    return list(block.iter(_A + "blip"))


def _extent(block) -> tuple[int, int] | None:
    for e in block.iter(_WP + "extent"):
        return int(e.get("cx")), int(e.get("cy"))
    return None


_TOL = 0.005  # 0.5% — below this the stretch is invisible in print


def _fit_to_box(png: Path, box_ratio: float) -> tuple[bytes, float]:
    """Return PNG bytes whose aspect equals ``box_ratio``, letterboxing with white if needed.

    Word scales the picture to fill ``wp:extent`` regardless of the file's pixel dimensions, so a
    replacement whose aspect differs from the box is silently *stretched* — Figure 27's rebuild
    came out 4% squatter than the frame it had to live in, which reads as a subtly wrong font
    everywhere in the panel. Padding rather than resizing keeps every drawn pixel where the plot
    code put it; only white margin is added, and the plots already sit on white.

    Returns:
        (png_bytes, distortion_before) — distortion as a signed fraction, for reporting.
    """
    img = Image.open(png)
    w, h = img.size
    ratio = w / h
    dist = box_ratio / ratio - 1.0
    if abs(dist) <= _TOL:
        return png.read_bytes(), dist

    if ratio < box_ratio:                     # too tall/narrow -> widen
        new_w, new_h = int(round(h * box_ratio)), h
    else:                                     # too wide -> heighten
        new_w, new_h = w, int(round(w / box_ratio))

    canvas = Image.new("RGB", (new_w, new_h), "white")
    canvas.paste(img.convert("RGB"), ((new_w - w) // 2, (new_h - h) // 2))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue(), dist


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--tag", default="figs")
    args = ap.parse_args()

    d = docx.Document(str(_DOCX))
    kids = list(d.element.body.iterchildren())

    # how many drawings point at each image part — a shared part must not be overwritten
    users: dict[str, int] = {}
    for b in kids:
        for blip in _blips(b):
            rid = blip.get(_R + "embed")
            try:
                users[d.part.related_parts[rid].partname] = \
                    users.get(d.part.related_parts[rid].partname, 0) + 1
            except KeyError:
                pass

    plan = []
    for block, png, what in SWAPS:
        src = _ROOT / png
        if not src.exists():
            print(f"✗ block {block}: missing {png}")
            return 1
        blips = _blips(kids[block])
        if len(blips) != 1:
            print(f"✗ block {block}: expected exactly 1 image, found {len(blips)} — wrong block")
            return 1
        part = d.part.related_parts[blips[0].get(_R + "embed")]
        if users.get(part.partname, 0) != 1:
            print(f"✗ block {block}: {part.partname} is shared by "
                  f"{users[part.partname]} drawings — refusing to overwrite")
            return 1
        ext = _extent(kids[block])
        if ext is None:
            print(f"✗ block {block}: no wp:extent — cannot guarantee the printed size")
            return 1
        blob, dist = _fit_to_box(src, ext[0] / ext[1])
        plan.append((block, part, blob, what, ext, len(part.blob), dist))

    print(f"{'블록':>6}  {'파트':<14} {'표시크기 (EMU)':<20} {'KB':<10} {'왜곡보정':>8}  내용")
    for block, part, blob, what, ext, old_n, dist in plan:
        e = f"{ext[0]}×{ext[1]}"
        pad = f"{dist * 100:+.1f}%" if abs(dist) > _TOL else "—"
        print(f"{block:>6}  {part.partname.split('/')[-1]:<14} {e:<20} "
              f"{old_n // 1024}→{len(blob) // 1024:<6} {pad:>8}  {what}")

    if args.dry_run:
        print("\n(dry run — 아무것도 안 씀)")
        return 0

    backup = _DOCX.with_name(f"{_DOCX.stem}_pre_{args.tag}.docx")
    shutil.copy2(_DOCX, backup)

    for _block, part, blob, _what, _ext, _o, _d in plan:
        part._blob = blob

    d.save(str(_DOCX))
    print(f"\n✅ {len(plan)}장 교체. 표시 크기(wp:extent) 불변 → 페이지 영향 없음.")
    print(f"   백업: {backup.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
