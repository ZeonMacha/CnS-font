"""
OpenType CID-Keyed CFF Font Packer & Anti-Hallucination QA Validator.

Injects newly vectorized cubic Bézier glyphs directly into FOT-SkipPro-B.OTF
while preserving all 15,446 original glyphs, vertical metrics, kana, and kerning.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.t2CharStringPen import T2CharStringPen
from fontTools.ttLib import TTFont
from scipy.ndimage import binary_dilation

from src.vectorizer import DrawCmd


def evaluate_topology_qa(
    pred_sdf: np.ndarray,
    cond_x: np.ndarray,
    trunc_px: float = 16.0,
    max_allowed_drift_px: float = 8.0,
) -> Dict[str, float]:
    """
    QA Gate from PLAN.md Section 4.1:
      1. Out-of-envelope phantom ink rate: fraction of predicted ink pixels lying outside
         `max_allowed_drift_px` dilation of (Noto Serif ∪ Noto Sans). Must be ~0.0%!
      2. Skeleton recall: fraction of core skeleton (`sdf_serif > 0.25`) covered by predicted ink.
      3. Ink density ratio vs Noto Sans reference.
    """
    pred_mask = pred_sdf > 0.0
    union_ref = np.maximum(cond_x[0], cond_x[1]) > 0.0
    core_skel = (cond_x[0] * trunc_px) > 1.8

    dilate_iters = max(1, int(round(max_allowed_drift_px)))
    allowed_envelope = binary_dilation(union_ref, iterations=dilate_iters)
    pred_area = max(int(pred_mask.sum()), 1)

    phantom_pixels = int((pred_mask & (~allowed_envelope)).sum())
    phantom_rate = phantom_pixels / float(pred_area)

    skel_total = max(int(core_skel.sum()), 1)
    skel_covered = int((pred_mask & core_skel).sum())
    skeleton_recall = skel_covered / float(skel_total)

    ref_sans_area = max(int((cond_x[1] > 0.0).sum()), 1)
    ink_ratio = pred_area / float(ref_sans_area)

    return {
        "phantom_ink_rate": phantom_rate,
        "skeleton_recall": skeleton_recall,
        "ink_density_ratio": ink_ratio,
    }


def pack_glyphs_into_otf(
    base_otf_path: str | Path,
    out_otf_path: str | Path,
    new_glyph_cmds: Dict[int, List[DrawCmd]],
    kanji_fd_index: int = 11,
    family_suffix: str = "CnS SC",
) -> int:
    """
    Pack `{unicode_codepoint: DrawCmd_list}` into a copy of `FOT-SkipPro-B.OTF`.
    Handles CID-keyed CFF (`FDArray`, `FDSelect`, `CharStrings`, `hmtx`, `vmtx`, `cmap`).
    """
    tt = TTFont(str(base_otf_path))
    cff = tt["CFF "].cff
    top = cff[0]

    # Switch ROS to Adobe-Identity-0 so custom CIDs >= 23060 render universally across engines
    top.ROS = ("Adobe", "Identity", 0)
    private_dict = top.FDArray[kanji_fd_index].Private

    glyph_order = list(tt.getGlyphOrder())
    next_cid = max(top.CIDCount, 23060)

    added_map: Dict[int, str] = {}
    hmtx = tt["hmtx"].metrics
    vmtx = tt["vmtx"].metrics if "vmtx" in tt else None
    cjk_ascender = 880

    for codepoint, cmds in sorted(new_glyph_cmds.items()):
        if not cmds:
            continue

        pen = T2CharStringPen(width=1000, glyphSet=None)
        for op, pts in cmds:
            if op == "moveTo":
                pen.moveTo(pts[0])
            elif op == "lineTo":
                pen.lineTo(pts[0])
            elif op == "curveTo":
                pen.curveTo(*pts)
            elif op == "closePath":
                pen.closePath()

        charstring = pen.getCharString(private=private_dict, globalSubrs=cff.GlobalSubrs)

        # Compute exact bounding box for lsb & tsb metrics
        bpen = BoundsPen(None)
        for op, pts in cmds:
            if op == "moveTo":
                bpen.moveTo(pts[0])
            elif op == "lineTo":
                bpen.lineTo(pts[0])
            elif op == "curveTo":
                bpen.curveTo(*pts)
            elif op == "closePath":
                bpen.closePath()
        bounds = bpen.bounds or (50, -50, 950, 830)
        lsb = int(round(bounds[0]))
        tsb = int(round(cjk_ascender - bounds[3]))

        gname = f"cid{next_cid:05d}"
        next_cid += 1

        new_gid = len(top.CharStrings.charStringsIndex)
        top.CharStrings.charStringsIndex.append(charstring)
        top.CharStrings.charStrings[gname] = new_gid
        glyph_order.append(gname)
        top.charset.append(gname)
        top.FDSelect.append(kanji_fd_index)

        hmtx[gname] = (1000, lsb)
        if vmtx is not None:
            vmtx[gname] = (1000, tsb)
        added_map[codepoint] = gname

    top.CIDCount = next_cid + 1
    tt.setGlyphOrder(glyph_order)

    # Update Unicode cmap subtables (Format 4 for BMP, Format 12 for full Unicode)
    for subtable in tt["cmap"].tables:
        if subtable.isUnicode():
            if subtable.format == 4:
                for cp, gname in added_map.items():
                    if cp <= 0xFFFF:
                        subtable.cmap[cp] = gname
            else:
                for cp, gname in added_map.items():
                    subtable.cmap[cp] = gname

    out_path = Path(out_otf_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tt.save(str(out_path))
    tt.close()
    return len(added_map)
