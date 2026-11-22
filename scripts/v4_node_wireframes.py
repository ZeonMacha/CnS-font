#!/usr/bin/env python3
"""
Generate v4 node wireframes comparison image (like v2_m2_node_wireframes.png).

3 rows:
  Row 0: SkipPro-B native (or traditional variant if SC char missing)
  Row 1: v3 synthesized output
  Row 2: v4 synthesized output

Each cell shows control points, Bezier handles, and node counts.
Uses fontTools directly (no uharfbuzz/pathops dependency chain).

Usage:
    python scripts/v4_node_wireframes.py \
        --skip_otf FOT-SkipPro-B.OTF \
        --v3_missing outputs/CnS-SkipPro-B-v3-Missing.otf \
        --v4_missing outputs/CnS-SkipPro-B-v4-Missing.otf \
        --out_png outputs/v4_node_wireframes.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
from fontTools.pens.recordingPen import RecordingPen


DrawCmd = Tuple[str, List[Tuple[float, float]]]


def extract_glyph_cmds(font_path: Path, char: str) -> Optional[List[DrawCmd]]:
    """Extract 1000-UPEM DrawCmd outlines from an OTF font for a single character."""
    try:
        tt = TTFont(str(font_path))
        cmap = tt.getBestCmap()
        cp = ord(char)
        if cp not in cmap:
            tt.close()
            return None
        glyph_name = cmap[cp]
        glyf = tt.getGlyphSet()
        glyph = glyf[glyph_name]
        pen = RecordingPen()
        glyph.draw(pen)
        tt.close()

        cmds: List[DrawCmd] = []
        for op, args in pen.value:
            if op in ("moveTo", "lineTo"):
                # CFF: args = ((x, y),)  ->  args[0] is a tuple
                if isinstance(args[0], tuple):
                    pt = (float(args[0][0]), float(args[0][1]))
                else:
                    pt = (float(args[0]), float(args[1]))
                cmds.append((op, [pt]))
            elif op == "curveTo":
                # CFF: args = ((x1,y1), (x2,y2), (x3,y3))
                pts = []
                for a in args:
                    if isinstance(a, tuple):
                        pts.append((float(a[0]), float(a[1])))
                    else:
                        # flat: args = (x1, y1, x2, y2, x3, y3)
                        break
                if len(pts) == 3:
                    cmds.append(("curveTo", pts))
                else:
                    # flat format fallback
                    cmds.append(("curveTo", [
                        (float(args[0]), float(args[1])),
                        (float(args[2]), float(args[3])),
                        (float(args[4]), float(args[5])),
                    ]))
            elif op in ("closePath", "endPath"):
                cmds.append(("closePath", []))
            elif op == "qCurveTo":
                pts = []
                for a in args:
                    if isinstance(a, tuple):
                        pts.append((float(a[0]), float(a[1])))
                if pts:
                    for p in pts:
                        cmds.append(("lineTo", [p]))
                else:
                    for i in range(0, len(args), 2):
                        if i + 1 < len(args):
                            cmds.append(("lineTo", [(float(args[i]), float(args[i+1]))]))
        return cmds if cmds else None
    except Exception:
        return None


def split_cmds_into_contours(cmds: List[DrawCmd]) -> List[List[DrawCmd]]:
    """Split a flat DrawCmd sequence into closed contour DrawCmd lists."""
    contours: List[List[DrawCmd]] = []
    curr: List[DrawCmd] = []
    for op, pts in cmds:
        curr.append((op, list(pts)))
        if op in ("closePath", "endPath"):
            if curr:
                contours.append(curr)
                curr = []
    if curr:
        contours.append(curr)
    return contours


def _flatten_contour_poly(c_cmds: List[DrawCmd], step_upem: float = 16.0) -> Tuple[np.ndarray, float]:
    """Flatten a closed contour DrawCmd list to dense polyline + exact signed area in UPEM."""
    pts: List[Tuple[float, float]] = []
    curr = (0.0, 0.0)
    for op, cpts in c_cmds:
        if op == "moveTo":
            curr = cpts[0]
            pts.append(curr)
        elif op == "lineTo":
            curr = cpts[0]
            pts.append(curr)
        elif op == "curveTo":
            p0 = np.array(curr, dtype=np.float64)
            p1 = np.array(cpts[0], dtype=np.float64)
            p2 = np.array(cpts[1], dtype=np.float64)
            p3 = np.array(cpts[2], dtype=np.float64)
            chord = float(np.linalg.norm(p3 - p0))
            n_seg = max(3, min(16, int(chord / step_upem) + 1))
            for t in np.linspace(0.0, 1.0, n_seg + 1)[1:]:
                u = 1.0 - t
                q = (u ** 3) * p0 + 3.0 * (u ** 2) * t * p1 + 3.0 * u * (t ** 2) * p2 + (t ** 3) * p3
                pts.append((float(q[0]), float(q[1])))
            curr = cpts[2]
    if len(pts) < 3:
        return np.zeros((0, 2), dtype=np.float64), 0.0
    arr = np.array(pts, dtype=np.float64)
    if np.linalg.norm(arr[0] - arr[-1]) < 1e-3 and len(arr) > 3:
        arr = arr[:-1]
    x, y = arr[:, 0], arr[:, 1]
    area = 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return arr, area


def count_outline_stats(cmds: List[DrawCmd]) -> Dict[str, int]:
    """Count on-curve nodes, line/curve ops, contours, mid-stem nodes."""
    on_curve = 0
    n_line = 0
    n_curve = 0
    contours = split_cmds_into_contours(cmds)
    mid_stem_nodes = 0
    segs: List[Tuple[Tuple[float, float], Tuple[float, float], str]] = []
    for c_cmds in contours:
        curr: Tuple[float, float] = (0.0, 0.0)
        for op, pts in c_cmds:
            if op == "moveTo":
                on_curve += 1
                curr = pts[0]
            elif op == "lineTo":
                on_curve += 1
                n_line += 1
                segs.append((curr, pts[0], "line"))
                curr = pts[0]
            elif op == "curveTo":
                on_curve += 1
                n_curve += 1
                segs.append((curr, pts[2], "curve"))
                curr = pts[2]
    # Check consecutive collinear near-axis long segments
    for i in range(len(segs)):
        p0, p1, _ = segs[i]
        p1b, p2, _ = segs[(i + 1) % len(segs)]
        v1 = np.array(p1) - np.array(p0)
        v2 = np.array(p2) - np.array(p1b)
        L1, L2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if L1 >= 45.0 and L2 >= 45.0:
            ang1 = abs(np.degrees(np.arctan2(v1[1], v1[0])))
            ang2 = abs(np.degrees(np.arctan2(v2[1], v2[0])))
            cos_a = float(np.dot(v1, v2) / (L1 * L2))
            same_h = (ang1 < 1.5 or abs(ang1 - 180) < 1.5) and (ang2 < 1.5 or abs(ang2 - 180) < 1.5)
            same_v = (abs(ang1 - 90) < 1.5) and (abs(ang2 - 90) < 1.5)
            if cos_a > 0.99985 and (same_h or same_v):
                mid_stem_nodes += 1
    return {
        "on_curve_nodes": on_curve,
        "line_ops": n_line,
        "curve_ops": n_curve,
        "contours": len(contours),
        "mid_stem_nodes": mid_stem_nodes,
    }


def render_wireframe_cell(cmds: List[DrawCmd], size: int = 260, title: str = "") -> Image.Image:
    """
    Render a dark-grid FontCreator/Glyphs-style control-point wireframe image:
      - gray glyph fill + crisp outline
      - green/red on-curve points, yellow off-curve Bezier handles
      - header badge with on-curve node count & mid-stem count
    """
    img = np.full((size, size, 3), 18, dtype=np.uint8)
    # Subtle grid
    for g in range(0, size, max(1, size // 8)):
        img[g, :, :] = 28
        img[:, g, :] = 28

    pad = 80.0
    scale = size / (1000.0 + 2.0 * pad)

    def to_xy(pt: Tuple[float, float]) -> Tuple[float, float]:
        return (pt[0] + pad) * scale, (880.0 + pad - pt[1]) * scale

    contours = split_cmds_into_contours(cmds)
    poly_items: List[Tuple[float, float, np.ndarray]] = []
    for c_cmds in contours:
        pts_u, area = _flatten_contour_poly(c_cmds, step_upem=12.0)
        if len(pts_u) < 3:
            continue
        pts_px = np.array([to_xy(p) for p in pts_u], dtype=np.int32)
        poly_items.append((area, 0.0, pts_px))

    # Fill polygons (even-odd rule via alternating signs)
    if poly_items:
        # Sort by area descending for proper fill
        poly_items.sort(key=lambda x: abs(x[0]), reverse=True)
        for _area, _sign, pts_px in poly_items:
            cv2.fillPoly(img, [pts_px], (140, 140, 140))
            cv2.polylines(img, [pts_px], True, (180, 180, 180), 1)

    # Draw control points
    for c_cmds in contours:
        curr: Tuple[float, float] = (0.0, 0.0)
        for op, pts in c_cmds:
            if op == "moveTo":
                px, py = to_xy(pts[0])
                cv2.circle(img, (int(px), int(py)), 3, (0, 255, 0), -1)  # green on-curve
                curr = pts[0]
            elif op == "lineTo":
                px, py = to_xy(pts[0])
                cv2.circle(img, (int(px), int(py)), 3, (0, 200, 0), -1)
                # Draw line
                x0, y0 = to_xy(curr)
                cv2.line(img, (int(x0), int(y0)), (int(px), int(py)), (100, 100, 100), 1)
                curr = pts[0]
            elif op == "curveTo":
                # Off-curve handles (yellow)
                for cp in pts[:2]:
                    hx, hy = to_xy(cp)
                    cv2.circle(img, (int(hx), int(hy)), 2, (255, 255, 0), -1)
                    x0, y0 = to_xy(curr)
                    cv2.line(img, (int(x0), int(y0)), (int(hx), int(hy)), (80, 80, 40), 1)
                # On-curve endpoint (green)
                px, py = to_xy(pts[2])
                cv2.circle(img, (int(px), int(py)), 3, (0, 255, 0), -1)
                x1, y1 = to_xy(pts[1])
                cv2.line(img, (int(x1), int(y1)), (int(px), int(py)), (80, 80, 40), 1)
                curr = pts[2]

    # Header badge
    if title:
        stats = count_outline_stats(cmds)
        badge = f"{title} | pts={stats['on_curve_nodes']} mid={stats['mid_stem_nodes']}"
        cv2.putText(img, badge, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def main() -> None:
    parser = argparse.ArgumentParser(description="v4 node wireframes")
    parser.add_argument("--skip_otf", type=Path, default=Path("FOT-SkipPro-B.OTF"))
    parser.add_argument("--v3_missing", type=Path, default=Path("outputs/CnS-SkipPro-B-v3-Missing.otf"))
    parser.add_argument("--v4_missing", type=Path, default=Path("outputs/CnS-SkipPro-B-v4-Missing.otf"))
    parser.add_argument("--out_png", type=Path, default=Path("outputs/v4_node_wireframes.png"))
    parser.add_argument("--n_chars", type=int, default=20)
    parser.add_argument("--cell_size", type=int, default=260)
    args = parser.parse_args()

    # Use chars that are in the Missing set (synthesized by v3/v4)
    # These are chars NOT in SkipPro-B but in TG8105
    sample_chars = list("红语银饭伞丌习乡卫专书东言直令骨微道龙丝")
    if len(sample_chars) > args.n_chars:
        sample_chars = sample_chars[: args.n_chars]

    # Traditional equivalents for native reference row
    orig_equiv = {
        "红": "紅", "语": "語", "银": "銀", "饭": "飯",
        "卫": "衛", "专": "專", "书": "書", "东": "東",
        "言": "言", "直": "直", "令": "令", "骨": "骨",
        "微": "微", "道": "道", "龙": "龍", "丝": "絲",
        "伞": "傘", "丌": "丌", "习": "習", "乡": "鄉",
    }

    cell = args.cell_size
    cols = len(sample_chars)
    n_rows = 3
    sheet = Image.new("RGB", (cols * cell, n_rows * cell), (12, 12, 16))

    print(f"Generating node wireframes for {len(sample_chars)} chars...")
    for i, ch in enumerate(sample_chars):
        # Row 0: SkipPro-B native (use traditional equivalent if available)
        eq = orig_equiv.get(ch, ch)
        cmds_native = extract_glyph_cmds(args.skip_otf, eq)
        if cmds_native is None:
            cmds_native = extract_glyph_cmds(args.skip_otf, ch)
        if cmds_native is not None:
            title_native = f"ORIG:{eq}"
            sheet.paste(render_wireframe_cell(cmds_native, cell, title_native), (i * cell, 0))
        else:
            empty = Image.new("RGB", (cell, cell), (18, 18, 22))
            dr = ImageDraw.Draw(empty)
            dr.text((cell // 2 - 30, cell // 2), "N/A", fill=(80, 80, 80))
            sheet.paste(empty, (i * cell, 0))

        # Row 1: v3 synthesized
        cmds_v3 = extract_glyph_cmds(args.v3_missing, ch) if args.v3_missing.exists() else None
        if cmds_v3 is not None:
            title_v3 = f"v3:{ch}"
            sheet.paste(render_wireframe_cell(cmds_v3, cell, title_v3), (i * cell, cell))
        else:
            empty = Image.new("RGB", (cell, cell), (18, 18, 22))
            dr = ImageDraw.Draw(empty)
            dr.text((cell // 2 - 30, cell // 2), "N/A", fill=(80, 80, 80))
            sheet.paste(empty, (i * cell, cell))

        # Row 2: v4 synthesized
        cmds_v4 = extract_glyph_cmds(args.v4_missing, ch) if args.v4_missing.exists() else None
        if cmds_v4 is not None:
            title_v4 = f"v4:{ch}"
            sheet.paste(render_wireframe_cell(cmds_v4, cell, title_v4), (i * cell, 2 * cell))
        else:
            empty = Image.new("RGB", (cell, cell), (18, 18, 22))
            dr = ImageDraw.Draw(empty)
            dr.text((cell // 2 - 30, cell // 2), "N/A", fill=(80, 80, 80))
            sheet.paste(empty, (i * cell, 2 * cell))

        n_pts_v3 = count_outline_stats(cmds_v3)["on_curve_nodes"] if cmds_v3 else 0
        n_pts_v4 = count_outline_stats(cmds_v4)["on_curve_nodes"] if cmds_v4 else 0
        ch_hex = f"U+{ord(ch):04X}"
        print(f"  [{i+1}/{len(sample_chars)}] {ch_hex}: native={'OK' if cmds_native else 'N/A'}, "
              f"v3={'OK' if cmds_v3 else 'N/A'} (pts={n_pts_v3}), "
              f"v4={'OK' if cmds_v4 else 'N/A'} (pts={n_pts_v4})")

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.out_png)
    print(f"\nNode wireframes saved to {args.out_png}")
    print(f"  Size: {sheet.size[0]}x{sheet.size[1]}, {len(sample_chars)} chars x {n_rows} rows")


if __name__ == "__main__":
    main()
