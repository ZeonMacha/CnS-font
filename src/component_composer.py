"""
Route 3 Native CFF Vector Sub-Glyph Extractor & Bounding-Box Affine Composer.

Extracts connected vector contour groups directly from `FOT-SkipPro-B.OTF` (or a
curated simplified radical bank) and affinely aligns them into target component
bounding boxes measured from `Noto Serif/Sans CJK SC`.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from fontTools.misc.transform import Transform
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.recordingPen import RecordingPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

from src.vectorizer import DrawCmd

BBox = Tuple[float, float, float, float]  # (xmin, ymin, xmax, ymax)


def extract_glyph_contours(tt: TTFont, codepoint: int) -> List[Tuple[BBox, List[DrawCmd]]]:
    """
    Extract individual closed contours and their 1000-UPEM bounding boxes from `tt`
    for `codepoint`.
    """
    cmap = tt.getBestCmap() or {}
    if codepoint not in cmap:
        return []

    gname = cmap[codepoint]
    gs = tt.getGlyphSet()
    rec = RecordingPen()
    gs[gname].draw(rec)

    contours: List[List[DrawCmd]] = []
    curr: List[DrawCmd] = []
    for op, pts in rec.value:
        curr.append((op, list(pts)))
        if op in ("closePath", "endPath"):
            contours.append(curr)
            curr = []
    if curr:
        contours.append(curr)

    result: List[Tuple[BBox, List[DrawCmd]]] = []
    for c_cmds in contours:
        bp = BoundsPen(None)
        for op, pts in c_cmds:
            getattr(bp, op)(*pts)
        if bp.bounds is not None:
            result.append((bp.bounds, c_cmds))
    return result


def split_left_right_component_cmds(
    tt: TTFont,
    donor_codepoint: int,
    side: str = "right",
    split_x_upem: float = 420.0,
) -> Optional[Tuple[BBox, List[DrawCmd]]]:
    """
    Extract either the left (`xmax <= split_x`) or right (`xmin >= split_x`) vector
    component from an existing left-right donor glyph in `FOT-SkipPro-B.OTF`
    (e.g., extracting native CFF `讠`-equivalent or phonetic right side).
    """
    contours = extract_glyph_contours(tt, donor_codepoint)
    if not contours:
        return None

    selected: List[DrawCmd] = []
    xs_min, ys_min, xs_max, ys_max = [], [], [], []
    for (xmin, ymin, xmax, ymax), cmds in contours:
        cx = 0.5 * (xmin + xmax)
        if (side == "left" and cx < split_x_upem) or (side == "right" and cx >= split_x_upem):
            selected.extend(cmds)
            xs_min.append(xmin)
            ys_min.append(ymin)
            xs_max.append(xmax)
            ys_max.append(ymax)

    if not selected:
        return None
    union_bbox: BBox = (min(xs_min), min(ys_min), max(xs_max), max(ys_max))
    return union_bbox, selected


def transform_cmds_to_bbox(
    cmds: List[DrawCmd],
    src_bbox: BBox,
    dst_bbox: BBox,
) -> List[DrawCmd]:
    """Affinely map `cmds` from `src_bbox` into `dst_bbox` in 1000-UPEM space."""
    sx0, sy0, sx1, sy1 = src_bbox
    dx0, dy0, dx1, dy1 = dst_bbox
    sw = max(sx1 - sx0, 1.0)
    sh = max(sy1 - sy0, 1.0)
    dw = max(dx1 - dx0, 1.0)
    dh = max(dy1 - dy0, 1.0)

    scale_x = dw / sw
    scale_y = dh / sh
    tx = dx0 - sx0 * scale_x
    ty = dy0 - sy0 * scale_y

    t = Transform(scale_x, 0.0, 0.0, scale_y, tx, ty)
    rec = RecordingPen()
    tpen = TransformPen(rec, t)
    for op, pts in cmds:
        getattr(tpen, op)(*pts)

    out: List[DrawCmd] = []
    for op, pts in rec.value:
        out.append((op, [(round(float(x), 1), round(float(y), 1)) for x, y in pts]))
    return out
