"""
Vector Boolean Composition (skia-pathops), Contour Winding Verification,
Primitive-Aware Curve Cleaner (§4.1a & §4.2), and Wireframe Node Renderer.

Guarantees:
  - 0 mid-stem staircase wave nodes on straight / horizontal / vertical stems
  - Single-segment symmetric concave waist (0.85 UPEM) on long near-axis stems
  - Strict PostScript CFF winding (even containment depth = CCW, odd depth = CW)
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pathops
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon
from skimage.measure import approximate_polygon

from src.v2.hb_outlines import DrawCmd, split_cmds_into_contours


def cmds_to_pathops_path(cmds: List[DrawCmd]) -> pathops.Path:
    """Convert 1000-UPEM DrawCmd list into a skia-pathops Path."""
    p = pathops.Path()
    pen = p.getPen()
    for op, pts in cmds:
        getattr(pen, op)(*pts)
    return p


def pathops_path_to_contours(path: pathops.Path) -> List[List[DrawCmd]]:
    """Extract clean closed contour DrawCmd lists from a skia-pathops Path."""
    from fontTools.pens.recordingPen import RecordingPen

    out: List[List[DrawCmd]] = []
    for contour in path.contours:
        rec = RecordingPen()
        contour.draw(rec)
        c_cmds = [(op, [(float(x), float(y)) for x, y in pts]) for op, pts in rec.value]
        if c_cmds:
            if c_cmds[-1][0] != "closePath":
                c_cmds.append(("closePath", []))
            out.append(c_cmds)
    return out


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


def reverse_contour_cmds(c_cmds: List[DrawCmd]) -> List[DrawCmd]:
    """Reverse the winding direction of a single closed contour DrawCmd list cleanly."""
    if not c_cmds:
        return c_cmds
    from fontTools.pens.pointPen import ReverseContourPointPen
    from fontTools.pens.recordingPen import RecordingPen, RecordingPointPen
    from fontTools.pens.pointPen import SegmentToPointPen, PointToSegmentPen

    rp = RecordingPointPen()
    s2p = SegmentToPointPen(rp)
    for op, pts in c_cmds:
        getattr(s2p, op)(*pts)

    rec = RecordingPen()
    p2s = PointToSegmentPen(rec)
    rev = ReverseContourPointPen(p2s)
    rp.replay(rev)
    out = [(op, [(float(x), float(y)) for x, y in pts]) for op, pts in rec.value]
    if out and out[-1][0] != "closePath":
        out.append(("closePath", []))
    return out


def ensure_cff_winding(contours_cmds: List[List[DrawCmd]]) -> List[List[DrawCmd]]:
    """
    Guarantee OpenType CFF PostScript winding via topological polygon containment depth:
      even nesting depth -> CCW (signed area > 0)
      odd nesting depth  -> CW  (signed area < 0)
    """
    polys: List[Optional[Polygon]] = []
    areas: List[float] = []
    valid_contours: List[List[DrawCmd]] = []

    for c_cmds in contours_cmds:
        pts, area = _flatten_contour_poly(c_cmds)
        if len(pts) < 3 or abs(area) < 10.0:
            continue
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = poly.buffer(0)
        polys.append(poly)
        areas.append(area)
        valid_contours.append(c_cmds)

    n = len(valid_contours)
    if n == 0:
        return []

    fixed: List[List[DrawCmd]] = []
    for i in range(n):
        pi = polys[i]
        rep_pt = pi.representative_point() if pi is not None and not pi.is_empty else None
        depth = 0
        if rep_pt is not None:
            for j in range(n):
                if i == j or abs(areas[j]) <= abs(areas[i]):
                    continue
                pj = polys[j]
                if pj is not None and pj.contains(rep_pt):
                    depth += 1
        want_ccw = (depth % 2 == 0)
        is_ccw = areas[i] > 0.0
        fixed.append(valid_contours[i] if (want_ccw == is_ccw) else reverse_contour_cmds(valid_contours[i]))
    return fixed


def boolean_union_cmds(contour_lists: Sequence[List[DrawCmd]]) -> List[DrawCmd]:
    """Clean Skia boolean union across component DrawCmd lists, preserving holes & CFF winding."""
    merged_path = pathops.Path()
    pen = merged_path.getPen()
    for cmds in contour_lists:
        for op, pts in cmds:
            getattr(pen, op)(*pts)
    merged_path.simplify(fix_winding=True, keep_starting_points=True, clockwise=False)
    contours = ensure_cff_winding(pathops_path_to_contours(merged_path))
    flat: List[DrawCmd] = []
    for c in contours:
        flat.extend(c)
    return flat


def _detect_corners(pts: np.ndarray, angle_thresh_deg: float = 27.0) -> List[int]:
    n = len(pts)
    if n <= 4:
        return list(range(n))

    la = min(2, max(1, n // 10))
    ang1 = np.zeros(n, dtype=np.float64)
    angles = np.zeros(n, dtype=np.float64)
    adj_min_len = np.zeros(n, dtype=np.float64)
    for i in range(n):
        v1 = pts[i] - pts[(i - 1) % n]
        v2 = pts[(i + 1) % n] - pts[i]
        l1, l2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        adj_min_len[i] = min(l1, l2)
        a1 = 0.0
        if l1 > 1e-2 and l2 > 1e-2:
            a1 = float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0))))
        ang1[i] = a1

        # Also inspect 2-step chord for smooth staircase immunity
        w1 = pts[i] - pts[(i - la) % n]
        w2 = pts[(i + la) % n] - pts[i]
        best_a = a1
        nw1, nw2 = float(np.linalg.norm(w1)), float(np.linalg.norm(w2))
        if nw1 > 1e-2 and nw2 > 1e-2:
            a2 = float(np.degrees(np.arccos(np.clip(np.dot(w1, w2) / (nw1 * nw2), -1.0, 1.0))))
            if a2 > best_a:
                best_a = a2
        angles[i] = best_a

    corners: List[int] = []
    for i in range(n):
        # Always preserve genuine 1-step sharp corners (>= 42 deg) with >= 8u incident edges
        # so adjacent 90 deg stroke junctions (like 丌 underside) never suppress each other!
        if ang1[i] >= 42.0 and adj_min_len[i] >= 8.0:
            corners.append(i)
        elif angles[i] >= angle_thresh_deg:
            if angles[i] >= max(angles[(i - 1) % n], angles[(i + 1) % n]) - 1e-5:
                corners.append(i)

    # Ensure at least 3 split anchors so smooth teardrops/loops never degenerate
    if len(corners) == 0:
        c0 = int(np.argmax(angles))
        corners = sorted({c0, (c0 + n // 3) % n, (c0 + 2 * n // 3) % n})
    elif len(corners) == 1:
        c0 = corners[0]
        corners = sorted({c0, (c0 + n // 3) % n, (c0 + 2 * n // 3) % n})
    elif len(corners) == 2:
        c0, c1 = corners
        mid = (c0 + (c1 - c0) // 2) % n if (c1 - c0) >= (n - (c1 - c0)) else (c1 + (n - (c1 - c0)) // 2) % n
        corners = sorted({c0, c1, mid})
    return corners


def _fit_bounded_cubic(segment: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    p0, p3 = segment[0], segment[-1]
    chord = p3 - p0
    chord_len = float(np.linalg.norm(chord))
    m = len(segment)
    if m <= 2 or chord_len < 1e-2:
        return p0 + chord / 3.0, p0 + 2.0 * chord / 3.0, 0.0

    diffs = np.linalg.norm(np.diff(segment, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(diffs)])
    arc_len = float(cum[-1])
    if arc_len < 1e-2:
        return p0 + chord / 3.0, p0 + 2.0 * chord / 3.0, 0.0

    t = cum / arc_len
    b0 = (1.0 - t) ** 3
    b1 = 3.0 * ((1.0 - t) ** 2) * t
    b2 = 3.0 * (1.0 - t) * (t ** 2)
    b3 = t ** 3
    p1_lin, p2_lin = p0 + chord / 3.0, p0 + 2.0 * chord / 3.0

    if m >= 5:
        rhs = segment[1:-1] - b0[1:-1, None] * p0 - b3[1:-1, None] * p3
        a_mat = np.stack([b1[1:-1], b2[1:-1]], axis=1)
        reg = 1e-2
        ata = a_mat.T @ a_mat + reg * np.eye(2)
        atb = a_mat.T @ rhs + reg * np.stack([p1_lin, p2_lin], axis=0)
        sol = np.linalg.solve(ata, atb)
        p1, p2 = sol[0], sol[1]
        max_h = 0.60 * arc_len
        if np.linalg.norm(p1 - p0) > max_h or np.linalg.norm(p2 - p3) > max_h:
            ts = segment[min(2, m - 1)] - p0
            te = p3 - segment[max(0, m - 3)]
            p1 = p0 + (ts / max(float(np.linalg.norm(ts)), 1e-3)) * (arc_len / 3.0)
            p2 = p3 - (te / max(float(np.linalg.norm(te)), 1e-3)) * (arc_len / 3.0)
    else:
        p1, p2 = p1_lin, p2_lin

    fit = b0[:, None] * p0 + b1[:, None] * p1 + b2[:, None] * p2 + b3[:, None] * p3
    return p1, p2, float(np.linalg.norm(fit - segment, axis=1).max())


def _fit_curve_recursive(segment: np.ndarray, max_err: float = 1.8, depth: int = 0) -> List[DrawCmd]:
    p0, p3 = segment[0], segment[-1]
    chord = p3 - p0
    L = float(np.linalg.norm(chord))
    if L < 1.0:
        return []
    if len(segment) == 2:
        return [("lineTo", [(float(p3[0]), float(p3[1]))])]

    u = chord / L
    perp = np.abs((segment - p0)[:, 0] * (-u[1]) + (segment - p0)[:, 1] * u[0])
    if float(perp.max()) <= 1.8:
        return [("lineTo", [(float(p3[0]), float(p3[1]))])]

    p1, p2, err = _fit_bounded_cubic(segment)
    if err <= max_err or depth >= 4 or len(segment) <= 2:
        return [("curveTo", [(float(p1[0]), float(p1[1])), (float(p2[0]), float(p2[1])), (float(p3[0]), float(p3[1]))])]

    mid = len(segment) // 2
    return _fit_curve_recursive(segment[: mid + 1], max_err, depth + 1) + _fit_curve_recursive(
        segment[mid:], max_err, depth + 1
    )


def _prune_collinear_corners(pts: np.ndarray, corners: List[int], max_angle_deg: float = 12.0, max_perp_u: float = 7.5) -> List[int]:
    """Drop redundant mid-stem corner anchors whose adjacent chords align within `max_angle_deg` and `max_perp_u`."""
    n = len(pts)
    cos_thresh = float(np.cos(np.radians(max_angle_deg)))
    corners = list(corners)
    changed = True
    while changed and len(corners) > 3:
        changed = False
        for k in range(len(corners)):
            c_prev = corners[(k - 1) % len(corners)]
            c_cur = corners[k]
            c_next = corners[(k + 1) % len(corners)]
            v1 = pts[c_cur] - pts[c_prev]
            v2 = pts[c_next] - pts[c_cur]
            L1 = float(np.linalg.norm(v1))
            L2 = float(np.linalg.norm(v2))
            if L1 < 1e-2 or L2 < 1e-2:
                corners.pop(k)
                changed = True
                break
            if float(np.dot(v1, v2) / (L1 * L2)) < cos_thresh:
                continue
            chord = pts[c_next] - pts[c_prev]
            Lc = float(np.linalg.norm(chord))
            if Lc < 1e-2:
                continue
            span = [idx % n for idx in range(c_prev, (c_next + 1) if c_next > c_prev else (c_next + n + 1))]
            u = chord / Lc
            perp = np.abs((pts[span] - pts[c_prev])[:, 0] * (-u[1]) + (pts[span] - pts[c_prev])[:, 1] * u[0])
            if float(perp.max()) <= max_perp_u:
                corners.pop(k)
                changed = True
                break
    return corners


def _make_stem_primitive(
    p0: Tuple[float, float],
    p3: Tuple[float, float],
    stem_concave_sagitta: float,
) -> DrawCmd:
    """Create 1 `lineTo` or 1 single inward-concave `curveTo` between two endpoints on a CCW-ink-left loop."""
    arr0 = np.array(p0, dtype=np.float64)
    arr3 = np.array(p3, dtype=np.float64)
    v = arr3 - arr0
    L = float(np.linalg.norm(v))
    if L >= 110.0 and stem_concave_sagitta > 0.0:
        u = v / L
        inward_n = np.array([-u[1], u[0]])
        offset = (4.0 / 3.0) * stem_concave_sagitta * inward_n
        p1 = arr0 + v / 3.0 + offset
        p2 = arr0 + 2.0 * v / 3.0 + offset
        return (
            "curveTo",
            [
                (round(float(p1[0]), 1), round(float(p1[1]), 1)),
                (round(float(p2[0]), 1), round(float(p2[1]), 1)),
                (round(float(p3[0]), 1), round(float(p3[1]), 1)),
            ],
        )
    return ("lineTo", [(round(float(p3[0]), 1), round(float(p3[1]), 1))])


def _weld_collinear_segments(
    c_cmds: List[DrawCmd],
    stem_concave_sagitta: float,
) -> List[DrawCmd]:
    """Merge consecutive collinear near-axis segments around the closed contour (including moveTo wrap-around)."""
    if not c_cmds:
        return c_cmds
    segs: List[Tuple[Tuple[float, float], str, List[Tuple[float, float]], Tuple[float, float]]] = []
    curr = (0.0, 0.0)
    for op, pts in c_cmds:
        if op == "moveTo":
            curr = pts[0]
        elif op == "lineTo":
            segs.append((curr, "lineTo", pts, pts[0]))
            curr = pts[0]
        elif op == "curveTo":
            segs.append((curr, "curveTo", pts, pts[2]))
            curr = pts[2]

    def _is_axis_aligned_collinear(s1, s2) -> bool:
        v1 = np.array(s1[3]) - np.array(s1[0])
        v2 = np.array(s2[3]) - np.array(s2[0])
        L1, L2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
        if L1 < 35.0 or L2 < 35.0:
            return False
        if float(np.dot(v1, v2) / (L1 * L2)) <= 0.9995:
            return False
        a1 = abs(np.degrees(np.arctan2(v1[1], v1[0])))
        a2 = abs(np.degrees(np.arctan2(v2[1], v2[0])))
        same_h = (a1 < 2.5 or abs(a1 - 180.0) < 2.5) and (a2 < 2.5 or abs(a2 - 180.0) < 2.5)
        same_v = (abs(a1 - 90.0) < 2.5) and (abs(a2 - 90.0) < 2.5)
        return same_h or same_v

    merged = True
    while merged and len(segs) >= 3:
        merged = False
        m = len(segs)
        for i in range(m):
            j = (i + 1) % m
            if _is_axis_aligned_collinear(segs[i], segs[j]):
                s0 = segs[i][0]
                e1 = segs[j][3]
                new_op, new_pts = _make_stem_primitive(s0, e1, stem_concave_sagitta)
                if j == i + 1:
                    segs[i] = (s0, new_op, new_pts, e1)
                    segs.pop(j)
                else:
                    # Wrap-around merge: last + first -> keep start at segs[i].start, pop segs[0]
                    segs[i] = (s0, new_op, new_pts, e1)
                    segs.pop(0)
                merged = True
                break

    out: List[DrawCmd] = [("moveTo", [segs[0][0]])]
    for _, op, pts, _ in segs:
        out.append((op, pts))
    out.append(("closePath", []))
    return out


def optimize_synthesized_contours(raw_cmds: List[DrawCmd], stem_concave_sagitta: float = 0.85) -> List[DrawCmd]:
    """
    Apply plan2.0 §4.1(a) & §4.2 primitive-aware simplification to NEWLY synthesized contours:
      - Eliminates all sub-pixel staircase ripples via 2.1 UPEM Douglas-Peucker smoothing
        and prunes redundant mid-stem collinear corners.
      - Snaps long near-axis stems (`>= 55u`, `±3.2°`, `max_perp <= 8.0u`) to 0 mid-stem nodes:
        either 1 `lineTo` or 1 symmetric inward-concave `curveTo` (sagitta 0.85u matching SkipPro-B!).
      - Welds any remaining consecutive collinear near-axis segments around the closed loop.
      - Enforces exact CFF winding.
    """
    contours = split_cmds_into_contours(raw_cmds)
    out_contours: List[List[DrawCmd]] = []

    for c_cmds in ensure_cff_winding(contours):
        pts_dense, area = _flatten_contour_poly(c_cmds, step_upem=10.0)
        if len(pts_dense) < 4 or abs(area) < 45.0:
            continue

        # Smooth sub-pixel staircase noise (in 1000 UPEM)
        ring = np.vstack([pts_dense, pts_dense[0:1]])
        poly = approximate_polygon(ring, tolerance=2.1)
        if len(poly) > 3 and np.linalg.norm(poly[0] - poly[-1]) < 0.5:
            pts = poly[:-1]
        else:
            pts = poly
        if len(pts) < 3:
            continue

        corners = _detect_corners(pts, angle_thresh_deg=28.0)
        corners = _prune_collinear_corners(pts, corners, max_angle_deg=12.0, max_perp_u=7.5)

        pts_adj = pts.copy()
        nc = len(corners)
        n = len(pts_adj)

        def seg_indices(i0: int, i1: int) -> List[int]:
            return [k % n for k in range(i0, (i1 + 1) if i1 > i0 else (i1 + n + 1))]

        snapped_pts = pts_adj.copy()

        # Pass 1: Snap exact horizontal / vertical coordinates along straight stems
        for k in range(nc):
            idx = seg_indices(corners[k], corners[(k + 1) % nc])
            seg = snapped_pts[idx]
            v = seg[-1] - seg[0]
            L = float(np.linalg.norm(v))
            if L < 55.0:
                continue
            u = v / L
            perp = np.abs((seg - seg[0])[:, 0] * (-u[1]) + (seg - seg[0])[:, 1] * u[0])
            if float(perp.max()) > 8.0:
                continue
            ang = abs(np.degrees(np.arctan2(v[1], v[0])))
            if ang <= 3.2 or abs(ang - 180.0) <= 3.2:
                y_med = round(float(np.median(seg[:, 1])), 1)
                snapped_pts[idx[0], 1] = y_med
                snapped_pts[idx[-1], 1] = y_med
            elif abs(ang - 90.0) <= 3.2:
                x_med = round(float(np.median(seg[:, 0])), 1)
                snapped_pts[idx[0], 0] = x_med
                snapped_pts[idx[-1], 0] = x_med

        # Pass 2: Fit 1 primitive per segment (0 mid-stem nodes!)
        c_out: List[DrawCmd] = []
        p_start = snapped_pts[corners[0]]
        c_out.append(("moveTo", [(round(float(p_start[0]), 1), round(float(p_start[1]), 1))]))

        for k in range(nc):
            idx = seg_indices(corners[k], corners[(k + 1) % nc])
            seg = snapped_pts[idx]
            p0, p3 = seg[0], seg[-1]
            v = p3 - p0
            L = float(np.linalg.norm(v))
            if L < 0.8:
                continue
            u = v / L
            perp = np.abs((seg - p0)[:, 0] * (-u[1]) + (seg - p0)[:, 1] * u[0])
            max_perp = float(perp.max())
            ang = abs(np.degrees(np.arctan2(v[1], v[0])))
            is_ortho = (ang <= 3.2 or abs(ang - 180.0) <= 3.2 or abs(ang - 90.0) <= 3.2)

            if is_ortho and L >= 55.0 and max_perp <= 8.0:
                c_out.append(
                    _make_stem_primitive(
                        (float(p0[0]), float(p0[1])),
                        (float(p3[0]), float(p3[1])),
                        stem_concave_sagitta,
                    )
                )
            elif max_perp <= 2.2:
                c_out.append(("lineTo", [(round(float(p3[0]), 1), round(float(p3[1]), 1))]))
            elif max_perp <= 6.5:
                p1, p2, _ = _fit_bounded_cubic(seg)
                c_out.append(
                    (
                        "curveTo",
                        [
                            (round(float(p1[0]), 1), round(float(p1[1]), 1)),
                            (round(float(p2[0]), 1), round(float(p2[1]), 1)),
                            (round(float(p3[0]), 1), round(float(p3[1]), 1)),
                        ],
                    )
                )
            else:
                for op, cpts in _fit_curve_recursive(seg, max_err=1.8):
                    c_out.append((op, [(round(x, 1), round(y, 1)) for x, y in cpts]))

        c_out.append(("closePath", []))
        c_out = _weld_collinear_segments(c_out, stem_concave_sagitta=stem_concave_sagitta)
        out_contours.append(c_out)

    final_cmds: List[DrawCmd] = []
    for c in ensure_cff_winding(out_contours):
        final_cmds.extend(c)
    return final_cmds


def count_outline_stats(cmds: List[DrawCmd]) -> dict:
    """Count on-curve nodes, curveTo ops, lineTo ops, and mid-stem nodes on long near-axis edges."""
    on_curve = 0
    n_line = 0
    n_curve = 0
    mid_stem_nodes = 0
    contours = split_cmds_into_contours(cmds)

    for c in contours:
        curr = (0.0, 0.0)
        segs = []
        for op, pts in c:
            if op == "moveTo":
                curr = pts[0]
            elif op == "lineTo":
                on_curve += 1; n_line += 1
                segs.append((curr, pts[0], "line"))
                curr = pts[0]
            elif op == "curveTo":
                on_curve += 1; n_curve += 1
                segs.append((curr, pts[2], "curve"))
                curr = pts[2]
        # Check consecutive genuinely collinear near-axis long segments (staircase mid-stem split)
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


def render_wireframe_cell(
    cmds: List[DrawCmd],
    size: int = 260,
    title: str = "",
) -> Image.Image:
    """
    Render a dark-grid FontCreator/Glyphs-style control-point wireframe image:
      - gray glyph fill + crisp outline
      - green/red on-curve points, yellow off-curve Bézier handles
      - header badge with on-curve node count & mid-stem count
    """
    import cv2

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
        pts_px = np.round(np.array([to_xy((x, y)) for x, y in pts_u])).astype(np.int32)
        poly_items.append((abs(area), area, pts_px))

    poly_items.sort(key=lambda item: -item[0])
    for _, area, pts_px in poly_items:
        cv2.fillPoly(img, [pts_px], (125, 125, 125) if area >= 0.0 else (18, 18, 18))
    for _, _, pts_px in poly_items:
        cv2.polylines(img, [pts_px], True, (220, 220, 220), 1, lineType=cv2.LINE_AA)

    # Draw Bézier handles & on-curve anchor nodes
    for c_cmds in contours:
        curr = (0.0, 0.0)
        for op, pts in c_cmds:
            if op == "moveTo":
                curr = pts[0]
            elif op == "lineTo":
                px, py = to_xy(pts[0])
                cv2.rectangle(img, (int(px) - 2, int(py) - 2), (int(px) + 2, int(py) + 2), (40, 235, 80), -1)
                curr = pts[0]
            elif op == "curveTo":
                x0, y0 = to_xy(curr)
                x1, y1 = to_xy(pts[0])
                x2, y2 = to_xy(pts[1])
                x3, y3 = to_xy(pts[2])
                cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (120, 120, 40), 1)
                cv2.line(img, (int(x3), int(y3)), (int(x2), int(y2)), (120, 120, 40), 1)
                cv2.circle(img, (int(x1), int(y1)), 2, (0, 220, 255), -1)
                cv2.circle(img, (int(x2), int(y2)), 2, (0, 220, 255), -1)
                cv2.circle(img, (int(x3), int(y3)), 3, (60, 80, 255), -1)
                curr = pts[2]

    pil_im = Image.fromarray(img[:, :, ::-1])  # BGR -> RGB
    stats = count_outline_stats(cmds)
    badge = f"{title} pts={stats['on_curve_nodes']} mid={stats['mid_stem_nodes']}".strip()
    draw = ImageDraw.Draw(pil_im)
    draw.rectangle([0, 0, size, 20], fill=(10, 10, 14))
    draw.text((6, 4), badge, fill=(235, 240, 245))
    return pil_im
