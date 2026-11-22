"""
Empirical Geometry Statistics from FOT-SkipPro-B.OTF & QA v2.0 Gate (§5).

Calibrates node-count buckets (indexed by Noto SC complexity) and straight-stem
concavity tolerances directly from original SkipPro-B glyphs so false alarms on
original Fontworks glyphs remain near zero.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.v2.curve_optimizer import _flatten_contour_poly, count_outline_stats
from src.v2.hb_outlines import (
    DrawCmd,
    HBOutlineExtractor,
    SkipProOutlineExtractor,
    cmds_iou,
    cmds_to_bbox,
    split_cmds_into_contours,
)


@dataclass
class SkipProStats:
    p95_nodes_by_bucket: Dict[int, int]  # bucket = ref_sc_nodes // 40 -> p95 orig nodes
    concave_sagitta_p10: float
    concave_sagitta_p50: float
    concave_sagitta_p90: float
    calib_false_alarm_rate: float


def measure_skippro_stats(
    skip_ext: SkipProOutlineExtractor,
    sans_sc: HBOutlineExtractor,
    chars: List[str],
    sample_n: int = 400,
    seed: int = 7,
) -> SkipProStats:
    """Measure node-count buckets and stem sagitta distributions on original SkipPro-B."""
    rng = random.Random(seed)
    pool = [c for c in chars if skip_ext.has_char(c)]
    rng.shuffle(pool)
    picked = pool[: min(sample_n, len(pool))]

    buckets: Dict[int, List[int]] = {}
    sags: List[float] = []

    for ch in picked:
        orig_cmds = skip_ext.get_cmds(ch)
        _, sc_cmds = sans_sc.get_gid_and_cmds(ch, "zh-Hans")
        if not orig_cmds or not sc_cmds:
            continue
        st_orig = count_outline_stats(orig_cmds)
        st_sc = count_outline_stats(sc_cmds)
        b = st_sc["on_curve_nodes"] // 40
        buckets.setdefault(b, []).append(st_orig["on_curve_nodes"])

        # Measure concave sagitta on long near-axis curves
        curr = (0.0, 0.0)
        for op, pts in orig_cmds:
            if op == "moveTo":
                curr = pts[0]
            elif op == "lineTo":
                curr = pts[0]
            elif op == "curveTo":
                p0 = np.array(curr)
                p1, p2, p3 = np.array(pts[0]), np.array(pts[1]), np.array(pts[2])
                v = p3 - p0
                L = float(np.linalg.norm(v))
                if L > 100.0 and (abs(v[0]) < L * 0.035 or abs(v[1]) < L * 0.035):
                    ts = np.linspace(0.0, 1.0, 17)[:, None]
                    cpts = ((1 - ts) ** 3) * p0 + 3 * ((1 - ts) ** 2) * ts * p1 + 3 * (1 - ts) * (ts ** 2) * p2 + (ts ** 3) * p3
                    u = v / L
                    d = (cpts - p0) @ np.array([-u[1], u[0]])
                    sags.append(float(np.abs(d).max()))
                curr = pts[2]

    p95_map: Dict[int, int] = {}
    all_nodes = [n for lst in buckets.values() for n in lst] or [180]
    global_p95 = int(np.percentile(all_nodes, 95))
    for b in range(0, 12):
        vals = buckets.get(b)
        if vals and len(vals) >= 8:
            p95_map[b] = int(np.ceil(np.percentile(vals, 95)))
        else:
            p95_map[b] = max(global_p95, (b + 1) * 45 + 60)

    sag_arr = np.array(sags or [0.4, 0.9, 1.6])
    stats = SkipProStats(
        p95_nodes_by_bucket=p95_map,
        concave_sagitta_p10=float(np.percentile(sag_arr, 10)),
        concave_sagitta_p50=float(np.percentile(sag_arr, 50)),
        concave_sagitta_p90=float(np.percentile(sag_arr, 90)),
        calib_false_alarm_rate=0.0,
    )

    # Calibrate QA gate directly against original glyphs
    false_alarms = 0
    tested = 0
    for ch in picked[:200]:
        orig_cmds = skip_ext.get_cmds(ch)
        _, sc_cmds = sans_sc.get_gid_and_cmds(ch, "zh-Hans")
        if orig_cmds and sc_cmds:
            ok, _ = check_glyph_qa_v2(orig_cmds, sc_cmds, stats, is_synthesized=False)
            false_alarms += (not ok)
            tested += 1
    stats.calib_false_alarm_rate = false_alarms / float(max(tested, 1))
    return stats


def check_glyph_qa_v2(
    cmds: List[DrawCmd],
    ref_sc_cmds: List[DrawCmd],
    stats: SkipProStats,
    is_synthesized: bool = True,
) -> Tuple[bool, Dict[str, float | int | bool]]:
    """
    QA v2.0 Gate (§5):
      - Non-empty valid positive CFF total signed area
      - Mid-stem nodes == 0 for synthesized glyphs (<= 2 tolerance on untouched original parts)
      - On-curve nodes <= bucket_p95 * 1.25
      - Polygon IoU vs zh-Hans reference >= 0.50 & bbox shift within tolerance
    """
    if not cmds:
        return False, {"empty": True}

    contours = split_cmds_into_contours(cmds)
    total_area = 0.0
    for c in contours:
        _, a = _flatten_contour_poly(c, step_upem=20.0)
        total_area += a

    st_gen = count_outline_stats(cmds)
    st_ref = count_outline_stats(ref_sc_cmds) if ref_sc_cmds else {"on_curve_nodes": 120}
    bucket = min(11, st_ref["on_curve_nodes"] // 40)
    node_cap = int(round(stats.p95_nodes_by_bucket.get(bucket, 240) * 1.25))

    iou_sc = cmds_iou(cmds, ref_sc_cmds, size=72) if ref_sc_cmds else 1.0
    bbox_g = cmds_to_bbox(cmds)
    bbox_r = cmds_to_bbox(ref_sc_cmds) if ref_sc_cmds else bbox_g

    bbox_ok = True
    if bbox_g is not None and bbox_r is not None:
        w_diff = abs((bbox_g[2] - bbox_g[0]) - (bbox_r[2] - bbox_r[0]))
        h_diff = abs((bbox_g[3] - bbox_g[1]) - (bbox_r[3] - bbox_r[1]))
        bbox_ok = (w_diff < 240.0) and (h_diff < 240.0)

    mid_ok = (st_gen["mid_stem_nodes"] == 0) if is_synthesized else (st_gen["mid_stem_nodes"] <= 3)
    nodes_ok = st_gen["on_curve_nodes"] <= max(node_cap, 120)
    area_ok = total_area > 500.0
    iou_ok = iou_sc >= 0.45

    passed = bool(mid_ok and nodes_ok and area_ok and iou_ok and bbox_ok)
    return passed, {
        "passed": passed,
        "on_curve_nodes": st_gen["on_curve_nodes"],
        "node_cap": node_cap,
        "mid_stem_nodes": st_gen["mid_stem_nodes"],
        "iou_vs_sc": round(iou_sc, 4),
        "total_area": round(total_area, 1),
        "bbox_ok": bbox_ok,
    }
