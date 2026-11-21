"""
Signed Distance Field (SDF) Computation & 4-Channel Complementary Dual-Skeleton Builder.

Constructs pixel-aligned input tensors:
  - Channel 0: Normalized SDF of Noto Serif CJK Bold (curves, sweeps, hooks, dots)
  - Channel 1: Normalized SDF of Noto Sans CJK SemiBold (flat sans horizontal/vertical bars)
  - Channel 2: Difference SDF (Serif - Sans), explicitly isolating triangular serifs (ウロコ)
  - Channel 3: Normalized internal stroke-depth field (medial thickness cue)
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt

from src.glyph_renderer import AlignedFontRenderer


def mask_to_normalized_sdf(
    hi_mask: np.ndarray,
    target_size: int,
    trunc_px: float = 16.0,
) -> np.ndarray:
    """
    Compute sub-pixel accurate Signed Distance Field from a supersampled binary mask,
    downsample to `target_size x target_size`, and normalize to [-1.0, +1.0].

    Convention:
      +1.0 = deep inside black stroke ink (>= trunc_px inside)
       0.0 = exact vector outline boundary
      -1.0 = far outside stroke in white background (>= trunc_px outside)
    """
    h_hi, w_hi = hi_mask.shape
    ss = h_hi / float(target_size)

    if not np.any(hi_mask):
        return np.full((target_size, target_size), -1.0, dtype=np.float32)

    dist_in = distance_transform_edt(hi_mask)
    dist_out = distance_transform_edt(~hi_mask)
    # Signed distance in supersampled pixels, centered on pixel boundary
    sdf_hi = (dist_in - 0.5) * hi_mask - (dist_out - 0.5) * (~hi_mask)
    # Convert distance unit to target canvas pixels
    sdf_hi = (sdf_hi / ss).astype(np.float32)

    if h_hi != target_size:
        ss_int = int(round(ss))
        sdf = sdf_hi.reshape(target_size, ss_int, target_size, ss_int).mean(axis=(1, 3))
    else:
        sdf = sdf_hi

    return np.clip(sdf / trunc_px, -1.0, 1.0).astype(np.float32)


def build_dual_skeleton_input(
    char: str,
    serif_renderer: AlignedFontRenderer,
    sans_renderer: AlignedFontRenderer,
    trunc_px: float = 16.0,
) -> np.ndarray:
    """
    Build 4-channel complementary input tensor (4, S, S) in [-1.0, +1.0]:
      [0] Serif Bold SDF
      [1] Sans SemiBold SDF
      [2] (Serif SDF - Sans SDF) clipped to [-1, 1] -> highlights horizontal triangular serifs
      [3] Internal stroke-depth map -> distinguishes thin hook tips from thick main stems
    """
    size = serif_renderer.spec.canvas_size
    mask_serif = serif_renderer.render_binary_hi(char)
    mask_sans = sans_renderer.render_binary_hi(char)

    sdf_serif = mask_to_normalized_sdf(mask_serif, size, trunc_px=trunc_px)
    sdf_sans = mask_to_normalized_sdf(mask_sans, size, trunc_px=trunc_px)
    sdf_diff = np.clip((sdf_serif - sdf_sans) * 2.0, -1.0, 1.0)
    stroke_depth = np.clip(sdf_serif, 0.0, 1.0) * 2.0 - 1.0

    return np.stack([sdf_serif, sdf_sans, sdf_diff, stroke_depth], axis=0).astype(np.float32)


def sdf_to_antialiased_gray(sdf: np.ndarray, trunc_px: float = 16.0) -> np.ndarray:
    """Convert normalized SDF [-1, 1] to crisp anti-aliased grayscale image [0=black, 255=white]."""
    px_dist = sdf * trunc_px
    # 1-pixel smooth step across zero level-set
    alpha = np.clip(px_dist + 0.5, 0.0, 1.0)
    return ((1.0 - alpha) * 255.0).round().astype(np.uint8)
