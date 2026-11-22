"""
Deskinned Noto Sans Light Reference Extractor, BBox-Aligned Medial Skeletonizer,
Uniform-Width SDF & Stroke-Orientation Field Builders (Plan 3.0 §1).

Zero Noto Serif contours or serif difference channels are used anywhere in v3.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from fontTools.misc.transform import Transform
from scipy.ndimage import distance_transform_edt as edt
from skimage.morphology import skeletonize

from src.sdf_utils import mask_to_normalized_sdf
from src.v2.component_bank import transform_cmds
from src.v2.hb_outlines import (
    BBox,
    DrawCmd,
    HBOutlineExtractor,
    SkipProOutlineExtractor,
    cmds_to_bbox,
    rasterize_cmds,
)


class SansLightExtractor(HBOutlineExtractor):
    """Extracts 1000-UPEM `Noto Sans CJK` Light (`wght=300`) outlines via HarfBuzz."""

    def __init__(self, sans_font_path: str | Path, language: str = "zh-Hans", weight: int = 300) -> None:
        ttc_index = 0 if language == "ja" else 2
        super().__init__(sans_font_path, ttc_index=ttc_index, weight=weight)
        self.default_language = language

    def get_cmds(self, char: str, unioned: bool = False) -> List[DrawCmd]:
        if unioned:
            return self.get_unioned_cmds(char, self.default_language)
        _, cmds = self.get_gid_and_cmds(char, self.default_language)
        return cmds


def align_cmds_to_bbox(cmds: List[DrawCmd], ref_cmds: List[DrawCmd]) -> List[DrawCmd]:
    """Affinely map `cmds` bounding box onto `ref_cmds` bounding box for translation/scale invariance."""
    b1 = cmds_to_bbox(cmds)
    b2 = cmds_to_bbox(ref_cmds)
    if b1 is None or b2 is None:
        return cmds
    w1, h1 = max(b1[2] - b1[0], 1.0), max(b1[3] - b1[1], 1.0)
    w2, h2 = max(b2[2] - b2[0], 1.0), max(b2[3] - b2[1], 1.0)
    sx, sy = w2 / w1, h2 / h1
    tx = b2[0] - b1[0] * sx
    ty = b2[1] - b1[1] * sy
    return transform_cmds(cmds, Transform(sx, 0.0, 0.0, sy, tx, ty))


@dataclass
class CleanSkeleton:
    """BBox-independent medial skeleton and topological summary at `(S x S)` grid."""

    skel: np.ndarray        # bool (S, S)
    dt_upem: np.ndarray     # float32 (S, S), Euclidean distance in UPEM to nearest skeleton pixel
    hole_count: int
    island_count: int
    junction_count: int
    endpoint_count: int
    canvas_size: int
    padding: float


def extract_clean_skeleton(
    cmds: List[DrawCmd],
    canvas_size: int = 160,
    padding: float = 80.0,
) -> CleanSkeleton:
    """
    Extract stabilized medial-axis skeleton + distance transform (in UPEM) + topological invariants.
    Using interior EDT core erosion (`>= 0.38 * median_radius`) suppresses thick serif corner spurs.
    """
    raw = rasterize_cmds(cmds, canvas_size=canvas_size, padding=padding).astype(np.uint8)
    if not raw.any():
        empty_b = np.zeros((canvas_size, canvas_size), dtype=bool)
        empty_f = np.full((canvas_size, canvas_size), 999.0, dtype=np.float32)
        return CleanSkeleton(empty_b, empty_f, 0, 0, 0, 0, canvas_size, padding)

    dt_in = edt(raw)
    med = float(np.median(dt_in[dt_in > 0]))
    if med > 2.2:
        core = (dt_in >= max(0.9, med * 0.38)).astype(np.uint8)
        if int(core.sum()) < 12:
            core = raw
    else:
        core = raw

    skel = skeletonize(core > 0)
    upem_per_px = (1000.0 + 2.0 * padding) / float(canvas_size)
    dt_upem = (edt(~skel) * upem_per_px).astype(np.float32) if skel.any() else np.full((canvas_size, canvas_size), 999.0, dtype=np.float32)

    # Topological components (holes on eroded core prevents micro serif notches forming fake enclosed loops)
    num_bg, _ = cv2.connectedComponents(1 - core, connectivity=4)
    hole_count = max(0, int(num_bg) - 2)
    num_fg, _ = cv2.connectedComponents(raw, connectivity=8)
    island_count = max(0, int(num_fg) - 1)

    if skel.any():
        k3 = np.ones((3, 3), dtype=np.uint8)
        neigh = cv2.filter2D(skel.astype(np.uint8), -1, k3, borderType=cv2.BORDER_CONSTANT) - 1
        ep_mask = (skel & (neigh == 1)).astype(np.uint8)
        jc_mask = (skel & (neigh >= 3)).astype(np.uint8)
        endpoint_count = max(0, int(cv2.connectedComponents(ep_mask, connectivity=8)[0]) - 1)
        junction_count = max(0, int(cv2.connectedComponents(jc_mask, connectivity=8)[0]) - 1)
    else:
        endpoint_count, junction_count = 0, 0

    return CleanSkeleton(
        skel=skel,
        dt_upem=dt_upem,
        hole_count=hole_count,
        island_count=island_count,
        junction_count=junction_count,
        endpoint_count=endpoint_count,
        canvas_size=canvas_size,
        padding=padding,
    )


def upem_box_to_mask(box: BBox, canvas_size: int = 160, padding: float = 80.0, margin_u: float = 18.0) -> np.ndarray:
    """Convert a 1000-UPEM BBox `(x0, y0, x1, y1)` into a boolean mask on `(canvas_size, canvas_size)`."""
    scale = canvas_size / (1000.0 + 2.0 * padding)
    x0 = int(np.clip(np.floor((box[0] - margin_u + padding) * scale), 0, canvas_size - 1))
    x1 = int(np.clip(np.ceil((box[2] + margin_u + padding) * scale), 0, canvas_size))
    # Y-up UPEM -> top-down image row
    y_top = int(np.clip(np.floor((880.0 + padding - (box[3] + margin_u)) * scale), 0, canvas_size - 1))
    y_bot = int(np.clip(np.ceil((880.0 + padding - (box[1] - margin_u)) * scale), 0, canvas_size))
    m = np.zeros((canvas_size, canvas_size), dtype=bool)
    if x1 > x0 and y_bot > y_top:
        m[y_top:y_bot, x0:x1] = True
    return m


def build_v3_deskinned_channels(
    cmds_light: List[DrawCmd],
    canvas_size: int = 256,
    padding: float = 96.0,
    trunc_px: float = 16.0,
    uniform_half_width_u: float = 40.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build raw spatial maps for Plan 3.0 §1.2 deskinned inputs:
      Returns `(sdf_light, dt_skel_upem, cos2t, sin2t)` as `float32` `(S, S)` arrays.
      At inference/training with `(wght=300, half_width=r_u)`:
        ch0 = sdf_light (in [-1, 1])
        ch1 = clip((r_u - dt_skel_upem) / trunc_upem, -1, 1)  # Uniform-width skeleton SDF
        ch2 = cos2t (tangent orientation field)
        ch3 = sin2t
    """
    s_hi = canvas_size * 2
    mask_hi = rasterize_cmds(cmds_light, canvas_size=s_hi, padding=padding)
    sdf_light = mask_to_normalized_sdf(mask_hi, canvas_size, trunc_px=trunc_px).astype(np.float32)

    mask_lo = rasterize_cmds(cmds_light, canvas_size=canvas_size, padding=padding)
    skel = skeletonize(mask_lo)
    upem_per_px = (1000.0 + 2.0 * padding) / float(canvas_size)
    if skel.any():
        dt_skel_u = (edt(~skel) * upem_per_px).astype(np.float32)
    else:
        dt_skel_u = np.full((canvas_size, canvas_size), 255.0, dtype=np.float32)

    # Smooth distance field slightly for clean stroke tangent vector orientation
    dt_smooth = cv2.GaussianBlur(dt_skel_u, (5, 5), sigmaX=1.2)
    gx = cv2.Sobel(dt_smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(dt_smooth, cv2.CV_32F, 0, 1, ksize=3)
    norm2 = gx * gx + gy * gy + 1e-6
    decay = np.exp(-dt_skel_u / 65.0).astype(np.float32)
    # Normal to skeleton is (gx, gy), so tangent is (-gy, gx) -> double-angle cos(2θ), sin(2θ)
    cos2t = (((gy * gy) - (gx * gx)) / norm2 * decay).astype(np.float32)
    sin2t = ((-2.0 * gx * gy) / norm2 * decay).astype(np.float32)
    return sdf_light, dt_skel_u, cos2t, sin2t


def pack_v3_input_tensor(
    sdf_light: np.ndarray,
    dt_skel_u: np.ndarray,
    cos2t: np.ndarray,
    sin2t: np.ndarray,
    uniform_half_width_u: float = 40.0,
    canvas_size: int = 256,
    padding: float = 96.0,
    trunc_px: float = 16.0,
    sdf_shift: float = 0.0,
) -> np.ndarray:
    """Assemble the exact `(4, S, S)` float32 input tensor for `LocalStrokeSDFNetV3`."""
    upem_per_px = (1000.0 + 2.0 * padding) / float(canvas_size)
    trunc_u = trunc_px * upem_per_px
    ch0 = np.clip(sdf_light + sdf_shift, -1.0, 1.0)
    ch1 = np.clip((uniform_half_width_u - dt_skel_u) / max(trunc_u, 1e-3), -1.0, 1.0)
    return np.stack([ch0, ch1, cos2t, sin2t], axis=0).astype(np.float32)


def measure_skippro_hv_halfwidths(
    skip_ext: SkipProOutlineExtractor,
    chars: List[str],
    sample_n: int = 150,
) -> Tuple[float, float]:
    """
    Measure calibrated SkipPro-B median horizontal & vertical half-widths (in UPEM)
    for Plan 3.0 §1.3 Light-skeleton box expansion (zero Sans-620 width inheritance!).
    """
    h_vals: List[float] = []
    v_vals: List[float] = []
    sobel_x = cv2.Sobel
    S = 192
    pad = 96.0
    upem_per_px = (1000.0 + 2.0 * pad) / S
    picked = [c for c in chars if skip_ext.has_char(c)][:sample_n]
    for ch in picked:
        m = rasterize_cmds(skip_ext.get_cmds(ch), canvas_size=S, padding=pad).astype(np.uint8)
        if not m.any():
            continue
        dt = edt(m) * upem_per_px
        skel = skeletonize(m > 0)
        ys, xs = np.where(skel)
        if len(ys) < 10:
            continue
        gx = sobel_x(dt.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = sobel_x(dt.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        for y, x in zip(ys, xs):
            r = float(dt[y, x])
            if r < 12.0 or r > 90.0:
                continue
            ax, ay = abs(float(gx[y, x])), abs(float(gy[y, x]))
            if ay > 2.2 * ax:
                # Normal points vertically -> horizontal bar
                h_vals.append(r)
            elif ax > 2.2 * ay:
                # Normal points horizontally -> vertical stem
                v_vals.append(r)
    h_med = float(np.median(h_vals)) if h_vals else 28.0
    v_med = float(np.median(v_vals)) if v_vals else 39.0
    return h_med, v_med
