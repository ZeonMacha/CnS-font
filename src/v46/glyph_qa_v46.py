"""
v4.6 High-Resolution Glyph QA Gate, Phantom-Island Repair & Soft Scorer (`src/v46/glyph_qa_v46.py`).

Implements 审计v46_进度.md §5.1 on a 384x384 canvas (~3.10 UPEM/px), operating on vector
outlines (compatible with both existing OTF glyphs and newly vectorized SDF candidates):
  1. Phantom island repair (`area < 1500 u^2` & ref ink overlap `< 30%` -> delete; cures 讠).
  2. Hard structural/detail checks:
       - `extra_stroke` (large unaligned island >= 1500 u^2)
       - `broken_stroke` (one ref island skeleton covered >= 15% by >= 2 cand islands)
       - `merged_gap` (two ref islands >= 30u apart merged into one cand island)
       - `filled_counter` (ref hole inscribed radius >= 30u disappears; cures 纟 filled triangle)
       - `spurious_counter` (cand adds new hole inscribed radius >= 12u)
       - optional whole-glyph `StructuralGateV45` check
  3. Stroke-width bulge (`blob_ratio = p98 / median`) & SkipPro median stroke-width ratio.
  4. Vector quality: G1 kink count (6-25 deg), curvature sign-flip frequency, node density,
     and max deviation vs raw SDF zero level set.
  5. SDF confidence (when SDF is available): zero-level-set gradient-deficit (`fog_ratio`),
     fog area ratio, and optional model-predicted uncertainty (`pred_logvar`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from fontTools.misc.transform import Transform
from scipy.ndimage import distance_transform_edt as edt
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from src.v2.component_bank import cmds_to_islands, transform_cmds
from src.v2.hb_outlines import BBox, DrawCmd, cmds_to_bbox, rasterize_cmds
from src.v3.hb_light import align_cmds_to_bbox
from src.v46.vectorize_v46 import VecParams, _signed_area, cmds_to_polylines, sdf_iso_contours_upem

QA_CANVAS = 384
QA_PADDING = 96.0
QA_UPX = (1000.0 + 2.0 * QA_PADDING) / float(QA_CANVAS)  # ~3.104 UPEM/px


@dataclass
class QAConfig:
    canvas_size: int = QA_CANVAS
    padding: float = QA_PADDING
    phantom_max_area_u2: float = 1500.0
    phantom_max_overlap: float = 0.30
    extra_stroke_min_area_u2: float = 1500.0
    extra_stroke_max_overlap: float = 0.08
    broken_cov_thresh: float = 0.15
    broken_dist_u: float = 16.0
    merged_gap_min_dist_u: float = 30.0
    filled_counter_min_rad_u: float = 30.0
    spurious_counter_min_rad_u: float = 12.0
    skippro_median_half_width_u: float = 38.0
    # Soft scoring weights (calibrated for 0..100 scale, tunable via logistic regression)
    w_blob: float = 14.0
    w_width_dev: float = 12.0
    w_kink: float = 1.8
    w_curv_flip: float = 1.2
    w_node_density: float = 0.15
    w_sdf_dev: float = 2.0
    w_fog: float = 25.0
    w_uncert: float = 10.0
    w_consensus: float = 20.0
    w_filled_counter: float = 18.0
    w_spurious_counter: float = 12.0


@dataclass
class QAResult:
    passed: bool
    score: float
    hard_reasons: List[str] = field(default_factory=list)
    repaired_cmds: List[DrawCmd] = field(default_factory=list)
    repaired_phantoms: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)


def repair_phantom_islands_v46(
    cand_cmds: List[DrawCmd],
    ref_sc_cmds: List[DrawCmd],
    cfg: Optional[QAConfig] = None,
) -> Tuple[List[DrawCmd], int, int]:
    """
    Inspect candidate islands against bbox-aligned Noto SC reference at 384 px.
    Returns `(repaired_cmds, n_repaired_phantoms, n_extra_large_strokes)`.
    """
    cfg = cfg or QAConfig()
    if not cand_cmds or not ref_sc_cmds:
        return cand_cmds, 0, 0

    isls = cmds_to_islands(cand_cmds)
    if len(isls) <= 1:
        return cand_cmds, 0, 0

    # Compute outlier-resistant whole-glyph alignment from major islands (>= phantom_max_area_u2)
    # so far-away phantom specks cannot stretch `b_cand`, while bounded scale/shift preserves valid dots.
    ref_mask = rasterize_cmds(ref_sc_cmds, canvas_size=cfg.canvas_size, padding=cfg.padding)
    if not ref_mask.any():
        return cand_cmds, 0, 0

    major_cand_cmds: List[DrawCmd] = []
    for isl in isls:
        if abs(float(isl.area)) >= cfg.phantom_max_area_u2:
            major_cand_cmds.extend(isl.cmds)
    ref_isls = cmds_to_islands(ref_sc_cmds)
    major_ref_cmds: List[DrawCmd] = []
    for r_isl in ref_isls:
        if abs(float(r_isl.area)) >= cfg.phantom_max_area_u2:
            major_ref_cmds.extend(r_isl.cmds)

    b_cand = cmds_to_bbox(major_cand_cmds or cand_cmds)
    b_ref = cmds_to_bbox(major_ref_cmds or ref_sc_cmds)
    align_tf: Optional[Transform] = None
    if b_cand is not None and b_ref is not None:
        w1, h1 = max(b_cand[2] - b_cand[0], 1.0), max(b_cand[3] - b_cand[1], 1.0)
        w2, h2 = max(b_ref[2] - b_ref[0], 1.0), max(b_ref[3] - b_ref[1], 1.0)
        sx = float(np.clip(w2 / w1, 0.85, 1.15))
        sy = float(np.clip(h2 / h1, 0.85, 1.15))
        cx1, cy1 = 0.5 * (b_cand[0] + b_cand[2]), 0.5 * (b_cand[1] + b_cand[3])
        cx2, cy2 = 0.5 * (b_ref[0] + b_ref[2]), 0.5 * (b_ref[1] + b_ref[3])
        dx_c = float(np.clip(cx2 - cx1, -45.0, 45.0))
        dy_c = float(np.clip(cy2 - cy1, -45.0, 45.0))
        tx = (cx1 + dx_c) - cx1 * sx
        ty = (cy1 + dy_c) - cy1 * sy
        align_tf = Transform(sx, 0.0, 0.0, sy, tx, ty)

    upx = (1000.0 + 2.0 * cfg.padding) / float(cfg.canvas_size)
    r_dil = max(1, int(round(10.0 / upx)))
    ref_dil = cv2.dilate(
        ref_mask.astype(np.uint8) * 255,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_dil + 1, 2 * r_dil + 1)),
    ) > 0

    kept: List[DrawCmd] = []
    repaired = 0
    extra_large = 0

    for isl in isls:
        area = abs(float(isl.area))
        isl_for_mask = transform_cmds(isl.cmds, align_tf) if align_tf is not None else isl.cmds
        m_isl = rasterize_cmds(isl_for_mask, canvas_size=cfg.canvas_size, padding=cfg.padding)
        n_px = int(m_isl.sum())
        if n_px == 0:
            repaired += 1
            continue
        overlap_strict = float((m_isl & ref_mask).sum()) / float(n_px)
        overlap_dil = float((m_isl & ref_dil).sum()) / float(n_px)

        if area < cfg.phantom_max_area_u2 and overlap_strict < cfg.phantom_max_overlap and overlap_dil < 0.45:
            repaired += 1
            continue
        if area >= cfg.extra_stroke_min_area_u2 and overlap_dil < cfg.extra_stroke_max_overlap:
            extra_large += 1
        kept.extend(isl.cmds)

    return (kept if kept else cand_cmds), repaired, extra_large


def _extract_holes(mask_u8: np.ndarray, upx: float, min_rad_u: float) -> List[Tuple[np.ndarray, float, Tuple[float, float]]]:
    """Return list of `(hole_mask_bool, inscribed_radius_u, (cy, cx))` for interior holes."""
    h, w = mask_u8.shape
    inv = (mask_u8 == 0).astype(np.uint8)
    n_cc, labels = cv2.connectedComponents(inv, connectivity=4)
    # Identify background labels touching image border
    border_labels = set(np.unique(labels[0, :]).tolist()) | set(np.unique(labels[-1, :]).tolist())
    border_labels |= set(np.unique(labels[:, 0]).tolist()) | set(np.unique(labels[:, -1]).tolist())

    holes: List[Tuple[np.ndarray, float, Tuple[float, float]]] = []
    for lb in range(1, n_cc):
        if lb in border_labels:
            continue
        hm = labels == lb
        dt_h = edt(hm) * upx
        rad = float(dt_h.max())
        if rad >= min_rad_u:
            ys, xs = np.where(hm)
            holes.append((hm, rad, (float(ys.mean()), float(xs.mean()))))
    return holes


def analyze_topology_and_bulge_v46(
    cand_cmds: List[DrawCmd],
    ref_sc_cmds: List[DrawCmd],
    cfg: Optional[QAConfig] = None,
) -> Dict[str, float]:
    """
    Compute 384 px topology & stroke-width metrics:
      - broken_stroke_count, merged_gap_count
      - filled_counter_count, spurious_counter_count
      - blob_ratio (p98 / median stroke half-width), median_half_width_u, width_ratio
    """
    cfg = cfg or QAConfig()
    s = cfg.canvas_size
    pad = cfg.padding
    upx = (1000.0 + 2.0 * pad) / float(s)

    cand_aligned = align_cmds_to_bbox(cand_cmds, ref_sc_cmds)
    ref_bin = rasterize_cmds(ref_sc_cmds, canvas_size=s, padding=pad).astype(np.uint8)
    cand_bin = rasterize_cmds(cand_aligned, canvas_size=s, padding=pad).astype(np.uint8)

    if not ref_bin.any() or not cand_bin.any():
        return {
            "broken_stroke_count": 0,
            "merged_gap_count": 0,
            "filled_counter_count": 0,
            "spurious_counter_count": 0,
            "blob_ratio": 1.0,
            "median_half_width_u": 0.0,
            "width_ratio": 0.0,
        }

    # 1. Connected components on ref and cand
    n_ref, ref_lbl = cv2.connectedComponents(ref_bin, connectivity=8)
    n_cand, cand_lbl = cv2.connectedComponents(cand_bin, connectivity=8)

    # Precompute distance transform per candidate island (only for islands with >= 12 px)
    cand_isl_dts: Dict[int, np.ndarray] = {}
    for ci in range(1, n_cand):
        cm = cand_lbl == ci
        if int(cm.sum()) >= 12:
            cand_isl_dts[ci] = edt(~cm) * upx

    # 2. Broken stroke check & dominant candidate owner per ref island
    broken_count = 0
    ref_owner: Dict[int, int] = {}
    ref_masks: Dict[int, np.ndarray] = {}

    for ri in range(1, n_ref):
        rm = ref_lbl == ri
        if int(rm.sum()) < 16:
            continue
        ref_masks[ri] = rm
        r_skel = skeletonize(rm)
        n_sk = int(r_skel.sum())
        if n_sk < 6:
            continue

        covs: List[Tuple[int, float]] = []
        for ci, c_dt in cand_isl_dts.items():
            cov = float((c_dt[r_skel] <= cfg.broken_dist_u).mean())
            if cov >= cfg.broken_cov_thresh:
                covs.append((ci, cov))
        covs.sort(key=lambda t: t[1], reverse=True)
        if covs:
            ref_owner[ri] = covs[0][0]
        if len(covs) >= 2:
            # Ensure the 2nd candidate island covers pixels NOT already covered by the 1st
            ci0 = covs[0][0]
            uncovered = r_skel & (cand_isl_dts[ci0] > cfg.broken_dist_u)
            n_unc = int(uncovered.sum())
            if n_unc >= max(4, int(0.12 * n_sk)):
                for ci_other, _ in covs[1:]:
                    extra_cov = float((cand_isl_dts[ci_other][uncovered] <= cfg.broken_dist_u).sum()) / float(n_sk)
                    if extra_cov >= cfg.broken_cov_thresh:
                        broken_count += 1
                        break

    # 3. Bulge / stroke width consistency on candidate (computed before merged-gap so Bold expansion is known)
    dt_in_u = edt(cand_bin > 0) * upx
    c_skel = skeletonize(cand_bin > 0)
    blob_ratio = 1.0
    med_hw = 0.0
    if c_skel.any():
        k3 = np.ones((3, 3), dtype=np.uint8)
        neigh = cv2.filter2D(c_skel.astype(np.uint8), -1, k3, borderType=cv2.BORDER_CONSTANT) - 1
        ep_jc = (c_skel & ((neigh == 1) | (neigh >= 3))).astype(np.uint8)
        r_excl = max(1, int(round(12.0 / upx)))
        excl = cv2.dilate(
            ep_jc * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_excl + 1, 2 * r_excl + 1)),
        ) > 0
        clean_sk = c_skel & (~excl)
        widths = dt_in_u[clean_sk] if int(clean_sk.sum()) >= 12 else dt_in_u[c_skel]
        if len(widths) >= 4:
            med_hw = float(np.median(widths))
            p98_hw = float(np.percentile(widths, 98))
            blob_ratio = p98_hw / max(med_hw, 1e-3)

    # 4. Bold-aware merged gap check: compensate for stroke thickening from Light ref (wght=300) to Bold cand
    merged_count = 0
    ref_ids = sorted(ref_masks.keys())
    if len(ref_ids) >= 2:
        ref_dt_in_u = edt(ref_bin > 0) * upx
        ref_all_skel = skeletonize(ref_bin > 0)
        ref_med_hw = float(np.median(ref_dt_in_u[ref_all_skel])) if ref_all_skel.any() else 20.0
        # Cap target half-width to canonical SkipPro Bold (~40u) so bloated/merged blobs never excuse themselves
        cap_hw = cfg.skippro_median_half_width_u * 1.05
        target_hw = min(cap_hw, max(med_hw, cfg.skippro_median_half_width_u)) if med_hw > 0 else cfg.skippro_median_half_width_u
        bold_expand_u = 2.0 * max(0.0, target_hw - ref_med_hw)
        eff_gap_thresh_u = cfg.merged_gap_min_dist_u + 0.65 * bold_expand_u

        ref_dts = {ri: edt(~ref_masks[ri]) * upx for ri in ref_ids}
        for idx_a in range(len(ref_ids)):
            ra = ref_ids[idx_a]
            oa = ref_owner.get(ra)
            if oa is None:
                continue
            for idx_b in range(idx_a + 1, len(ref_ids)):
                rb = ref_ids[idx_b]
                if ref_owner.get(rb) != oa:
                    continue
                min_gap_u = float(ref_dts[ra][ref_masks[rb]].min())
                if min_gap_u >= eff_gap_thresh_u:
                    merged_count += 1

    # 5. Counter / hole checks
    ref_holes = _extract_holes(ref_bin, upx, cfg.filled_counter_min_rad_u)
    cand_holes = _extract_holes(cand_bin, upx, cfg.spurious_counter_min_rad_u)
    cand_holes_all = _extract_holes(cand_bin, upx, 6.0)

    filled_counters = 0
    for rhm, _, _ in ref_holes:
        matched = any(float((rhm & chm).sum()) / max(float(rhm.sum()), 1.0) >= 0.12 for chm, _, _ in cand_holes_all)
        if not matched:
            filled_counters += 1

    ref_holes_all = _extract_holes(ref_bin, upx, 6.0)
    spurious_counters = 0
    for chm, _, _ in cand_holes:
        matched = any(float((chm & rhm).sum()) / max(float(chm.sum()), 1.0) >= 0.12 for rhm, _, _ in ref_holes_all)
        if not matched:
            spurious_counters += 1

    width_ratio = med_hw / max(cfg.skippro_median_half_width_u, 1e-3) if med_hw > 0 else 0.0

    return {
        "broken_stroke_count": int(broken_count),
        "merged_gap_count": int(merged_count),
        "filled_counter_count": int(filled_counters),
        "spurious_counter_count": int(spurious_counters),
        "blob_ratio": round(float(blob_ratio), 3),
        "median_half_width_u": round(float(med_hw), 2),
        "width_ratio": round(float(width_ratio), 3),
    }


def _unit_vec(v: np.ndarray) -> Optional[np.ndarray]:
    n = float(np.linalg.norm(v))
    return (v / n) if n > 1e-6 else None


def compute_vector_quality_v46(
    cmds: Sequence[DrawCmd],
    sdf_2d: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Measure outline craftsmanship:
      - `g1_kinks`: joins with tangent discontinuity in [6 deg, 25 deg]
      - `curv_sign_flips`: inflection / curvature sign-flip count per 1000u
      - `node_density`: on-curve nodes per 1000u contour length
      - `sdf_max_dev_u` / `sdf_p95_dev_u`: deviation against raw SDF zero level set (if `sdf_2d` given)
    """
    contours: List[List[Tuple[str, Tuple[float, float], Optional[Tuple[float, float]], Optional[Tuple[float, float]], Tuple[float, float]]]] = []
    cur_segs = []
    pos = (0.0, 0.0)
    start = (0.0, 0.0)
    on_curve = 0
    total_len = 0.0
    curv_flips = 0

    for op, cp in cmds:
        if op == "moveTo":
            if cur_segs:
                contours.append(cur_segs)
            cur_segs = []
            pos = cp[0]
            start = cp[0]
        elif op == "lineTo":
            p0, p3 = pos, cp[0]
            d = math.hypot(p3[0] - p0[0], p3[1] - p0[1])
            if d > 1e-4:
                cur_segs.append(("L", p0, None, None, p3))
                total_len += d
                on_curve += 1
            pos = p3
        elif op == "curveTo":
            p0, c1, c2, p3 = pos, cp[0], cp[1], cp[2]
            d = math.hypot(c1[0] - p0[0], c1[1] - p0[1]) + math.hypot(c2[0] - c1[0], c2[1] - c1[1]) + math.hypot(p3[0] - c2[0], p3[1] - c2[1])
            if d > 1e-4:
                cur_segs.append(("C", p0, c1, c2, p3))
                total_len += 0.75 * d
                on_curve += 1
                # Detect S-curve inflection inside cubic via cross products (c1-p0)x(c2-c1) vs (c2-c1)x(p3-c2)
                v1 = (c1[0] - p0[0], c1[1] - p0[1])
                v2 = (c2[0] - c1[0], c2[1] - c1[1])
                v3 = (p3[0] - c2[0], p3[1] - c2[1])
                cr1 = v1[0] * v2[1] - v1[1] * v2[0]
                cr2 = v2[0] * v3[1] - v2[1] * v3[0]
                if cr1 * cr2 < -1e-2:
                    curv_flips += 1
            pos = p3
        elif op == "closePath":
            if cur_segs:
                if math.hypot(start[0] - pos[0], start[1] - pos[1]) > 1e-3:
                    cur_segs.append(("L", pos, None, None, start))
                    total_len += math.hypot(start[0] - pos[0], start[1] - pos[1])
                    on_curve += 1
                contours.append(cur_segs)
                cur_segs = []
    if cur_segs:
        contours.append(cur_segs)

    g1_kinks = 0
    for segs in contours:
        m = len(segs)
        if m < 2:
            continue
        for i in range(m):
            s_prev = segs[i]
            s_next = segs[(i + 1) % m]
            t_in = np.array(s_prev[4]) - np.array(s_prev[3] if s_prev[0] == "C" else s_prev[1])
            t_out = np.array(s_next[2] if s_next[0] == "C" else s_next[4]) - np.array(s_next[1])
            u_in = _unit_vec(t_in)
            u_out = _unit_vec(t_out)
            if u_in is None or u_out is None:
                continue
            cos_a = float(np.clip(np.dot(u_in, u_out), -1.0, 1.0))
            ang = math.degrees(math.acos(cos_a))
            if 6.0 <= ang <= 25.0:
                g1_kinks += 1

    len_k = max(total_len / 1000.0, 0.1)
    res: Dict[str, float] = {
        "on_curve_nodes": int(on_curve),
        "contour_len_u": round(float(total_len), 1),
        "node_density": round(float(on_curve / len_k), 2),
        "g1_kinks": int(g1_kinks),
        "curv_sign_flips_per_k": round(float(curv_flips / len_k), 2),
        "sdf_max_dev_u": 0.0,
        "sdf_p95_dev_u": 0.0,
    }

    if sdf_2d is not None:
        raw_prm = VecParams(pre_sigma_px=0.0, upsample=2)
        iso_polys_raw = sdf_iso_contours_upem(sdf_2d, raw_prm)
        cand_polys = cmds_to_polylines(cmds, step=2.0)
        if iso_polys_raw and cand_polys:
            p_cand = np.vstack(cand_polys)
            cand_tree = cKDTree(p_cand)
            # Filter out dropped specks/pinholes and intentionally repaired phantom islands from iso_polys
            # so repaired candidates are measured ONLY on surviving contours (never penalized 200u+ for repair).
            kept_iso: List[np.ndarray] = []
            for poly in iso_polys_raw:
                if abs(_signed_area(poly)) < raw_prm.min_hole_u2:
                    continue
                d_poly, _ = cand_tree.query(poly)
                if float(np.median(d_poly)) <= 18.0:
                    kept_iso.append(poly)
            if kept_iso:
                p_iso = np.vstack(kept_iso)
                d1, _ = cand_tree.query(p_iso)
                d2, _ = cKDTree(p_iso).query(p_cand)
                d_all = np.concatenate([d1, d2])
                res["sdf_max_dev_u"] = round(float(d_all.max()), 3)
                res["sdf_p95_dev_u"] = round(float(np.percentile(d_all, 95)), 3)

    return res


def compute_sdf_confidence_v46(
    sdf_2d: np.ndarray,
    pred_logvar_2d: Optional[np.ndarray] = None,
    trunc_px: float = 16.0,
) -> Dict[str, float]:
    """
    Measure R5 SDF fog instability directly from a 2D [-1, 1] SDF array (numpy, no torch required).
    """
    s = sdf_2d.astype(np.float32)
    gx = cv2.Sobel(s, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(s, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    gnorm = np.sqrt(gx * gx + gy * gy + 1e-8)

    expected = 1.0 / max(trunc_px, 1.0)
    near_zero = np.abs(s) < 0.15
    low_grad = gnorm < (0.55 * expected)
    fog_px = int((near_zero & low_grad).sum())
    band_px = max(int(near_zero.sum()), 1)
    ink_or_band = max(int((s > -0.15).sum()), 1)

    out = {
        "sdf_fog_ratio": round(float(fog_px / band_px), 4),
        "sdf_fog_area_ratio": round(float(fog_px / ink_or_band), 4),
        "sdf_uncert_mean": 0.0,
    }
    if pred_logvar_2d is not None:
        lv = np.clip(pred_logvar_2d.astype(np.float32), -6.0, 2.0)
        if near_zero.any():
            out["sdf_uncert_mean"] = round(float(np.exp(lv[near_zero]).mean()), 4)
    return out


class GlyphQAGateV46:
    """
    Combined v4.6 Repair + Hard Gate + Soft Scorer.
    Wraps optional `StructuralGateV45` for whole-glyph structure while adding 384 px
    radical-level defect detection (phantom islands, broken strokes, merged gaps,
    filled counters, stroke-width bulges, vector kinks, and SDF fog).
    """

    def __init__(
        self,
        structural_gate: Optional[object] = None,
        cfg: Optional[QAConfig] = None,
    ) -> None:
        self.structural_gate = structural_gate
        self.cfg = cfg or QAConfig()

    def evaluate(
        self,
        cand_cmds: List[DrawCmd],
        ch: str,
        ref_sc_cmds: List[DrawCmd],
        *,
        sdf_2d: Optional[np.ndarray] = None,
        pred_logvar_2d: Optional[np.ndarray] = None,
        consensus_iou: Optional[float] = None,
        exempt_boxes: Optional[Sequence[BBox]] = None,
        check_jp_diff: bool = True,
        repair_phantoms: bool = True,
    ) -> QAResult:
        if not cand_cmds:
            return QAResult(passed=False, score=-100.0, hard_reasons=["empty_cmds"], repaired_cmds=[])

        # 1. Repair phantom islands (< 1500 u^2, < 30% ref overlap)
        if repair_phantoms:
            cmds_use, n_rep, n_extra = repair_phantom_islands_v46(cand_cmds, ref_sc_cmds, self.cfg)
        else:
            cmds_use, n_rep, n_extra = cand_cmds, 0, 0

        hard_reasons: List[str] = []
        if n_extra > 0:
            hard_reasons.append(f"extra_stroke={n_extra}")

        # 2. Optional whole-glyph StructuralGateV45 check
        sg_metrics: Dict[str, float] = {}
        if self.structural_gate is not None:
            ok_sg, gm_sg = self.structural_gate.check(
                cmds_use, ch, exempt_boxes=exempt_boxes, check_jp_diff=check_jp_diff
            )
            if not ok_sg:
                for r in gm_sg.get("reasons", [gm_sg.get("reason", "structural_gate_fail")]):
                    hard_reasons.append(f"sg:{r}")
            for k in ("chamfer", "p95", "miss_sc", "miss_cand"):
                if k in gm_sg:
                    sg_metrics[f"sg_{k}"] = float(gm_sg[k])

        # 3. 384 px topology & bulge metrics
        topo = analyze_topology_and_bulge_v46(cmds_use, ref_sc_cmds, self.cfg)
        if topo["broken_stroke_count"] > 0:
            hard_reasons.append(f"broken_stroke={topo['broken_stroke_count']}")
        if topo["merged_gap_count"] > 0:
            hard_reasons.append(f"merged_gap={topo['merged_gap_count']}")
        if topo["filled_counter_count"] > 0:
            hard_reasons.append(f"filled_counter={topo['filled_counter_count']}")

        # 4. Vector quality metrics
        vec_q = compute_vector_quality_v46(cmds_use, sdf_2d=sdf_2d)

        # 5. SDF confidence metrics (when SDF provided)
        sdf_q = (
            compute_sdf_confidence_v46(sdf_2d, pred_logvar_2d=pred_logvar_2d)
            if sdf_2d is not None
            else {"sdf_fog_ratio": 0.0, "sdf_fog_area_ratio": 0.0, "sdf_uncert_mean": 0.0}
        )

        # 6. Soft quality score (higher is better; baseline 100.0 minus calibrated penalties).
        # Hinged SDF/fog/uncertainty tolerances ensure clean neural SDFs are never unfairly penalized
        # relative to OTF candidates that lack `sdf_2d` / `pred_logvar_2d` telemetry.
        c = self.cfg
        pen_blob = c.w_blob * max(0.0, topo["blob_ratio"] - 1.35)
        pen_width = c.w_width_dev * abs(topo["width_ratio"] - 1.0) if topo["width_ratio"] > 0 else 15.0
        pen_kink = c.w_kink * float(vec_q["g1_kinks"])
        pen_flip = c.w_curv_flip * float(vec_q["curv_sign_flips_per_k"])
        pen_node = c.w_node_density * max(0.0, float(vec_q["node_density"]) - 28.0)
        sdf_p95_eff = float(vec_q["sdf_p95_dev_u"]) if sdf_2d is not None else 3.2
        fog_eff = float(sdf_q["sdf_fog_ratio"]) if sdf_2d is not None else 0.035
        unc_eff = float(sdf_q["sdf_uncert_mean"]) if pred_logvar_2d is not None else 0.05
        pen_sdf_dev = c.w_sdf_dev * max(0.0, sdf_p95_eff - 3.2)
        pen_fog = c.w_fog * max(0.0, fog_eff - 0.035)
        pen_unc = c.w_uncert * max(0.0, unc_eff - 0.05)
        pen_counter = (
            c.w_filled_counter * float(topo["filled_counter_count"])
            + c.w_spurious_counter * float(topo["spurious_counter_count"])
        )
        bonus_cons = c.w_consensus * float(consensus_iou) if consensus_iou is not None else 0.0

        score = 100.0 + bonus_cons - (
            pen_blob + pen_width + pen_kink + pen_flip + pen_node
            + pen_sdf_dev + pen_fog + pen_unc + pen_counter
        )
        if hard_reasons:
            score -= 50.0 * len(hard_reasons)

        metrics: Dict[str, float] = {
            **sg_metrics,
            **topo,
            **vec_q,
            **sdf_q,
            "repaired_phantoms": int(n_rep),
            "extra_large_strokes": int(n_extra),
        }
        if consensus_iou is not None:
            metrics["consensus_iou"] = round(float(consensus_iou), 4)

        return QAResult(
            passed=(len(hard_reasons) == 0),
            score=round(float(score), 3),
            hard_reasons=hard_reasons,
            repaired_cmds=cmds_use,
            repaired_phantoms=n_rep,
            metrics=metrics,
        )
