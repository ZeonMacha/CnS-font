"""
v3.0 Curvature-Constrained Curve Optimizer (`Plan3.0.md §2.2`).

Key safeguards preventing v2 straight-line reward hacking:
  1. Builds `no_snap_mask` from `Noto Sans Light SC` smooth curved spans (`6 deg .. 42 deg` over ~40u,
     dilated by 28u) -> hooks, sweeps, teardrops (`竖弯钩/撇/捺/点`) are NEVER straightened!
  2. Tightens collinear corner merging to `6.0 deg` (`max_perp <= 4.0u`).
  3. Enforces per-contour bidirectional Hausdorff <= 4.0 UPEM AND relative area delta <= 1.5%;
     any contour violating either limit reverts cleanly to the un-snapped bounded cubic curve!
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize
from skimage.measure import approximate_polygon

from src.v2.curve_optimizer import (
    _detect_corners,
    _fit_curve_recursive,
    _flatten_contour_poly,
    _make_stem_primitive,
    _prune_collinear_corners,
    ensure_cff_winding,
)
from src.v2.hb_outlines import DrawCmd, rasterize_cmds, split_cmds_into_contours


def build_no_snap_mask(
    ref_sc_cmds: Optional[List[DrawCmd]],
    canvas_size: int = 160,
    padding: float = 80.0,
    dilate_upem: float = 28.0,
) -> Optional[np.ndarray]:
    """
    Detect genuinely curved/hooked stroke regions in `ref_sc_cmds` and dilate by `dilate_upem`.
    Smooth turning vertices (6..42 deg over ~40u step) mark curved strokes without flagging
    sharp 90 deg T/L corners at straight bar ends.
    """
    if not ref_sc_cmds:
        return None
    scale = canvas_size / (1000.0 + 2.0 * padding)
    mask = np.zeros((canvas_size, canvas_size), dtype=np.uint8)

    for c_cmds in split_cmds_into_contours(ref_sc_cmds):
        pts, _ = _flatten_contour_poly(c_cmds, step_upem=20.0)
        n = len(pts)
        if n < 5:
            continue
        for i in range(n):
            v1 = pts[i] - pts[(i - 2) % n]
            v2 = pts[(i + 2) % n] - pts[i]
            l1, l2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
            if l1 < 1e-2 or l2 < 1e-2:
                continue
            ang = float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0))))
            if 6.0 <= ang <= 42.0:
                px = int(np.clip(round((pts[i, 0] + padding) * scale), 0, canvas_size - 1))
                py = int(np.clip(round((880.0 + padding - pts[i, 1]) * scale), 0, canvas_size - 1))
                mask[py, px] = 255

    if not mask.any():
        return mask > 0
    rad = max(1, int(round(dilate_upem * scale)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rad + 1, 2 * rad + 1))
    return cv2.dilate(mask, k) > 0


def _segment_hits_no_snap(
    seg: np.ndarray,
    no_snap_mask: Optional[np.ndarray],
    canvas_size: int = 160,
    padding: float = 80.0,
) -> bool:
    if no_snap_mask is None or len(seg) == 0:
        return False
    scale = canvas_size / (1000.0 + 2.0 * padding)
    ts = np.linspace(0.0, 1.0, min(9, max(3, len(seg))))
    p0, p1 = seg[0], seg[-1]
    hits = 0
    for t in ts:
        pt = p0 + t * (p1 - p0)
        px = int(np.clip(round((pt[0] + padding) * scale), 0, canvas_size - 1))
        py = int(np.clip(round((880.0 + padding - pt[1]) * scale), 0, canvas_size - 1))
        if no_snap_mask[py, px]:
            hits += 1
    return (hits / float(len(ts))) >= 0.25


def _bidirectional_hausdorff(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
    if len(pts_a) == 0 or len(pts_b) == 0:
        return 999.0
    da, _ = cKDTree(pts_b).query(pts_a)
    db, _ = cKDTree(pts_a).query(pts_b)
    return float(max(float(da.max()), float(db.max())))


def _fit_unconstrained_contour(pts_dense: np.ndarray) -> List[DrawCmd]:
    """Produce baseline un-snapped bounded cubic Bézier contour without straight-stem snapping."""
    ring = np.vstack([pts_dense, pts_dense[0:1]])
    poly = approximate_polygon(ring, tolerance=1.35)
    pts = poly[:-1] if (len(poly) > 3 and np.linalg.norm(poly[0] - poly[-1]) < 0.5) else poly
    if len(pts) < 3:
        return []
    corners = _detect_corners(pts, angle_thresh_deg=28.0)
    n = len(pts)
    nc = len(corners)
    out: List[DrawCmd] = [("moveTo", [(round(float(pts[corners[0], 0]), 1), round(float(pts[corners[0], 1]), 1))])]
    for k in range(nc):
        i0, i1 = corners[k], corners[(k + 1) % nc]
        idx = [m % n for m in range(i0, (i1 + 1) if i1 > i0 else (i1 + n + 1))]
        seg = pts[idx]
        for op, cpts in _fit_curve_recursive(seg, max_err=1.35):
            out.append((op, [(round(float(x), 1), round(float(y), 1)) for x, y in cpts]))
    out.append(("closePath", []))
    return out


def optimize_synthesized_contours_v3(
    raw_cmds: List[DrawCmd],
    ref_sc_cmds: Optional[List[DrawCmd]] = None,
    stem_concave_sagitta: float = 0.85,
    max_hausdorff_upem: float = 4.0,
    max_area_rel: float = 0.015,
) -> Tuple[List[DrawCmd], Dict[str, int]]:
    """
    Apply curvature-masked, Hausdorff-guarded straight-stem refinement (`Plan3.0.md §2.2`).
    Returns `(final_cmds, stats)` where `stats` tracks kept vs reverted contours.
    """
    contours = split_cmds_into_contours(raw_cmds)
    no_snap = build_no_snap_mask(ref_sc_cmds)
    out_contours: List[List[DrawCmd]] = []
    snapped_cnt = 0
    reverted_cnt = 0

    for c_cmds in ensure_cff_winding(contours):
        pts_dense, area_raw = _flatten_contour_poly(c_cmds, step_upem=6.0)
        if len(pts_dense) < 4 or abs(area_raw) < 45.0:
            continue

        baseline_cmds = _fit_unconstrained_contour(pts_dense)
        if not baseline_cmds:
            continue
        pts_base, area_base = _flatten_contour_poly(baseline_cmds, step_upem=6.0)

        ring = np.vstack([pts_dense, pts_dense[0:1]])
        poly = approximate_polygon(ring, tolerance=1.7)
        pts = poly[:-1] if (len(poly) > 3 and np.linalg.norm(poly[0] - poly[-1]) < 0.5) else poly
        if len(pts) < 3:
            out_contours.append(baseline_cmds)
            continue

        corners = _detect_corners(pts, angle_thresh_deg=28.0)
        # Tightened 6.0 deg / 4.0u collinear pruning (Plan 3.0 §2.2)
        corners = _prune_collinear_corners(pts, corners, max_angle_deg=6.0, max_perp_u=4.0)
        n = len(pts)
        nc = len(corners)

        def seg_idx(i0: int, i1: int) -> List[int]:
            return [m % n for m in range(i0, (i1 + 1) if i1 > i0 else (i1 + n + 1))]

        snapped_pts = pts.copy()
        for k in range(nc):
            idx = seg_idx(corners[k], corners[(k + 1) % nc])
            seg = snapped_pts[idx]
            if _segment_hits_no_snap(seg, no_snap):
                continue
            v = seg[-1] - seg[0]
            L = float(np.linalg.norm(v))
            if L < 55.0:
                continue
            u = v / L
            perp = np.abs((seg - seg[0])[:, 0] * (-u[1]) + (seg - seg[0])[:, 1] * u[0])
            if float(perp.max()) > 4.0:
                continue
            ang = abs(np.degrees(np.arctan2(v[1], v[0])))
            if ang <= 2.8 or abs(ang - 180.0) <= 2.8:
                ym = round(float(np.median(seg[:, 1])), 1)
                snapped_pts[idx[0], 1] = ym
                snapped_pts[idx[-1], 1] = ym
            elif abs(ang - 90.0) <= 2.8:
                xm = round(float(np.median(seg[:, 0])), 1)
                snapped_pts[idx[0], 0] = xm
                snapped_pts[idx[-1], 0] = xm

        c_snap: List[DrawCmd] = [
            ("moveTo", [(round(float(snapped_pts[corners[0], 0]), 1), round(float(snapped_pts[corners[0], 1]), 1))])
        ]
        for k in range(nc):
            idx = seg_idx(corners[k], corners[(k + 1) % nc])
            seg = snapped_pts[idx]
            p0, p3 = seg[0], seg[-1]
            v = p3 - p0
            L = float(np.linalg.norm(v))
            if L < 0.8:
                continue
            u = v / L
            max_perp = float(np.abs((seg - p0)[:, 0] * (-u[1]) + (seg - p0)[:, 1] * u[0]).max())
            ang = abs(np.degrees(np.arctan2(v[1], v[0])))
            is_ortho = ang <= 2.8 or abs(ang - 180.0) <= 2.8 or abs(ang - 90.0) <= 2.8
            hits_curve = _segment_hits_no_snap(seg, no_snap)

            if (not hits_curve) and is_ortho and L >= 55.0 and max_perp <= 4.0:
                c_snap.append(_make_stem_primitive((float(p0[0]), float(p0[1])), (float(p3[0]), float(p3[1])), stem_concave_sagitta))
            elif (not hits_curve) and max_perp <= 1.8:
                c_snap.append(("lineTo", [(round(float(p3[0]), 1), round(float(p3[1]), 1))]))
            else:
                for op, cpts in _fit_curve_recursive(seg, max_err=1.4):
                    c_snap.append((op, [(round(float(x), 1), round(float(y), 1)) for x, y in cpts]))
        c_snap.append(("closePath", []))

        pts_snap, area_snap = _flatten_contour_poly(c_snap, step_upem=6.0)
        hd = _bidirectional_hausdorff(pts_base, pts_snap)
        rel_a = abs(abs(area_snap) - abs(area_base)) / max(abs(area_base), 1.0)
        if hd <= max_hausdorff_upem and rel_a <= max_area_rel:
            out_contours.append(c_snap)
            snapped_cnt += 1
        else:
            out_contours.append(baseline_cmds)
            reverted_cnt += 1

    final_cmds: List[DrawCmd] = []
    for c in ensure_cff_winding(out_contours):
        final_cmds.extend(c)
    return final_cmds, {"snapped_contours": snapped_cnt, "reverted_contours": reverted_cnt}
