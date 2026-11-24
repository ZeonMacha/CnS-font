"""
v4.5 Segment-Level Curve Optimizer & Orphan/Floating Island Filter.

Fixes over v3/v4.3 `curve_constrain.py` (from 审计v4.md §2.3 #4, #6 & Phase A #2, #4):
  1. Per-segment Hausdorff guard (`max_seg_hd = 5.8 UPEM`) instead of whole-contour revert:
     eliminates the 98.13% whole-contour revert bug (`snapped=479 / reverted=25,151`)
     and restores clean zero-mid straight stems while preserving Taubin-smoothed arcs.
  2. Orphan & floating speckle island filter (`filter_orphan_islands_v45`):
     removes detached specks `< 400 UPEM^2` with zero overlap on 12u-dilated SC ref, AND
     floating noise blobs `< 1,000 UPEM^2` whose min distance to SC skeleton is `> 35 UPEM`
     (fixes `孩 106.5u²` and `孓 686.1u²` without ever touching real dots like `义`'s `丶`).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt as edt
from scipy.spatial import cKDTree
from skimage.measure import approximate_polygon
from skimage.morphology import skeletonize

from src.v2.component_bank import cmds_to_islands
from src.v2.curve_optimizer import (
    _detect_corners,
    _fit_curve_recursive,
    _flatten_contour_poly,
    _make_stem_primitive,
    _prune_collinear_corners,
    ensure_cff_winding,
)
from src.v2.hb_outlines import DrawCmd, rasterize_cmds, split_cmds_into_contours
from src.v3.curve_constrain import (
    _fit_core_straight_line,
    _fit_unconstrained_contour,
    _segment_hits_no_snap,
    _taubin_smooth_open_segment,
    build_no_snap_mask,
)


def filter_orphan_islands_v45(
    cmds: List[DrawCmd],
    ref_sc_cmds: Optional[List[DrawCmd]],
    canvas_size: int = 256,
    padding: float = 96.0,
) -> Tuple[List[DrawCmd], int]:
    """
    Remove detached micro-speckles (<400 UPEM^2 outside 12u-dilated SC ref) and
    floating noise blobs (<1,000 UPEM^2 with min distance >35 UPEM to SC skeleton).
    Never removes real strokes/dots that sit on or near the SC reference skeleton.
    """
    if not cmds or not ref_sc_cmds:
        return cmds, 0

    isls = cmds_to_islands(cmds)
    if len(isls) <= 1:
        return cmds, 0

    upem_px = (1000.0 + 2.0 * padding) / float(canvas_size)
    sc_bin = rasterize_cmds(ref_sc_cmds, canvas_size=canvas_size, padding=padding)
    if not sc_bin.any():
        return cmds, 0

    r_chk = max(1, int(round(12.0 / upem_px)))
    sc_dil = cv2.dilate(
        sc_bin.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_chk + 1, 2 * r_chk + 1)),
    ) > 0
    sc_skel = skeletonize(sc_bin)
    dt_sc_u = (edt(~sc_skel) * upem_px).astype(np.float32) if sc_skel.any() else np.full((canvas_size, canvas_size), 999.0, dtype=np.float32)

    kept_cmds: List[DrawCmd] = []
    dropped = 0
    for isl in isls:
        area = abs(float(isl.area))
        if area < 1000.0:
            m_isl = rasterize_cmds(isl.cmds, canvas_size=canvas_size, padding=padding)
            if not m_isl.any():
                dropped += 1
                continue
            if area < 400.0 and not (m_isl & sc_dil).any():
                dropped += 1
                continue
            min_dist_u = float(dt_sc_u[m_isl].min())
            if min_dist_u > 35.0:
                dropped += 1
                continue
        kept_cmds.extend(isl.cmds)

    return (kept_cmds if kept_cmds else cmds), dropped


def _seg_hausdorff(seg_pts: np.ndarray, cmd_list: List[DrawCmd]) -> float:
    """Bidirectional Hausdorff distance (in UPEM) between original polyline segment and fitted commands."""
    if len(seg_pts) < 2 or not cmd_list:
        return 999.0
    # Sample fitted commands densely
    fake = [("moveTo", [(float(seg_pts[0, 0]), float(seg_pts[0, 1]))])] + list(cmd_list)
    pts_fit, _ = _flatten_contour_poly(fake, step_upem=6.0)
    if len(pts_fit) < 2:
        return 999.0
    d1, _ = cKDTree(pts_fit).query(seg_pts)
    d2, _ = cKDTree(seg_pts).query(pts_fit)
    return float(max(float(d1.max()), float(d2.max())))


def smooth_unconstrained_cmds_v45(raw_cmds: List[DrawCmd]) -> List[DrawCmd]:
    """Taubin-smoothed unconstrained cubic Bézier fallback (never emits jagged raw polygons)."""
    out_contours: List[List[DrawCmd]] = []
    for c_cmds in ensure_cff_winding(split_cmds_into_contours(raw_cmds)):
        pts_dense, area_raw = _flatten_contour_poly(c_cmds, step_upem=6.0)
        if len(pts_dense) < 4 or abs(area_raw) < 45.0:
            continue
        c_fit = _fit_unconstrained_contour(pts_dense)
        if c_fit:
            out_contours.append(c_fit)
    final: List[DrawCmd] = []
    for c in ensure_cff_winding(out_contours):
        final.extend(c)
    return final


def optimize_synthesized_contours_v45(
    raw_cmds: List[DrawCmd],
    ref_sc_cmds: Optional[List[DrawCmd]] = None,
    stem_concave_sagitta: float = 0.85,
    max_seg_hausdorff_upem: float = 5.8,
    max_area_rel: float = 0.025,
) -> Tuple[List[DrawCmd], Dict[str, int]]:
    """
    Segment-level curvature-masked straight-stem snapping + Taubin arc smoothing.
    Each segment between sharp corners is validated independently so one curved corner
    never causes straight horizontal/vertical stems in the same contour to revert.
    """
    clean_raw, dropped_speckles = filter_orphan_islands_v45(raw_cmds, ref_sc_cmds)
    contours = split_cmds_into_contours(clean_raw)
    no_snap = build_no_snap_mask(ref_sc_cmds)
    out_contours: List[List[DrawCmd]] = []
    snapped_segs = 0
    curved_segs = 0
    reverted_contours = 0

    for c_cmds in ensure_cff_winding(contours):
        pts_dense, area_raw = _flatten_contour_poly(c_cmds, step_upem=6.0)
        if len(pts_dense) < 4 or abs(area_raw) < 45.0:
            continue

        baseline_cmds = _fit_unconstrained_contour(pts_dense)
        if not baseline_cmds:
            continue
        _, area_base = _flatten_contour_poly(baseline_cmds, step_upem=6.0)

        ring = np.vstack([pts_dense, pts_dense[0:1]])
        poly = approximate_polygon(ring, tolerance=1.7)
        pts = poly[:-1] if (len(poly) > 3 and np.linalg.norm(poly[0] - poly[-1]) < 0.5) else poly
        if len(pts) < 3:
            out_contours.append(baseline_cmds)
            continue

        corners = _detect_corners(pts, angle_thresh_deg=28.0)
        corners = _prune_collinear_corners(pts, corners, max_angle_deg=6.0, max_perp_u=4.0)
        n = len(pts)
        nc = len(corners)

        def seg_idx(i0: int, i1: int) -> List[int]:
            return [m % n for m in range(i0, (i1 + 1) if i1 > i0 else (i1 + n + 1))]

        # Build contour segment-by-segment with per-segment Hausdorff verification
        c_out: List[DrawCmd] = [
            ("moveTo", [(round(float(pts[corners[0], 0]), 1), round(float(pts[corners[0], 1]), 1))])
        ]
        for k in range(nc):
            idx = seg_idx(corners[k], corners[(k + 1) % nc])
            seg = pts[idx]
            p0, p3 = seg[0], seg[-1]
            v = p3 - p0
            L = float(np.linalg.norm(v))
            if L < 0.8:
                continue

            _, _, max_perp, ang = _fit_core_straight_line(seg)
            is_ortho_or_tilt = (ang <= 3.5 or abs(ang - 180.0) <= 3.5 or abs(ang - 90.0) <= 3.2)
            hits_curve = _segment_hits_no_snap(seg, no_snap)

            cand_seg: Optional[List[DrawCmd]] = None
            if (not hits_curve) and is_ortho_or_tilt and L >= 52.0 and max_perp <= 4.5:
                cand_seg = [_make_stem_primitive((float(p0[0]), float(p0[1])), (float(p3[0]), float(p3[1])), stem_concave_sagitta)]
            elif (not hits_curve) and max_perp <= 2.2:
                cand_seg = [("lineTo", [(round(float(p3[0]), 1), round(float(p3[1]), 1))])]

            if cand_seg is not None and _seg_hausdorff(seg, cand_seg) <= max_seg_hausdorff_upem:
                c_out.extend(cand_seg)
                snapped_segs += 1
            else:
                seg_smooth = _taubin_smooth_open_segment(seg, n_iters=2)
                for op, cpts in _fit_curve_recursive(seg_smooth, max_err=2.2):
                    c_out.append((op, [(round(float(x), 1), round(float(y), 1)) for x, y in cpts]))
                curved_segs += 1

        c_out.append(("closePath", []))

        _, area_out = _flatten_contour_poly(c_out, step_upem=6.0)
        rel_a = abs(abs(area_out) - abs(area_base)) / max(abs(area_base), 1.0)
        if rel_a <= max_area_rel:
            out_contours.append(c_out)
        else:
            out_contours.append(baseline_cmds)
            reverted_contours += 1

    final_cmds: List[DrawCmd] = []
    for c in ensure_cff_winding(out_contours):
        final_cmds.extend(c)
    return final_cmds, {
        "snapped_segs": snapped_segs,
        "curved_segs": curved_segs,
        "reverted_contours": reverted_contours,
        "dropped_speckles": dropped_speckles,
    }
