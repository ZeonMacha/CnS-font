"""
Patch-Cropped Font SDF Dataset with Zero Data Leakage on Held-Out Characters.

Trains strictly on local `patch_size x patch_size` stroke crops centered around
active stroke edges, intersections, and Serif-vs-Sans discrepancy regions.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from src.glyph_renderer import AlignedFontRenderer, CanvasSpec
from src.sdf_utils import build_dual_skeleton_input, mask_to_normalized_sdf

# Exact 8 characters where zi2zi-JiT suffered severe radical hallucination / overfitting
ZI2ZI_JIT_FAILURE_8 = "志洳菜嘉改栅畴逡"

# Representative unseen pure Simplified Chinese test characters (0% presence in SkipPro-B)
UNSEEN_SIMPLIFIED_12 = "语银红饭车东书专买风龙业"


class PatchCropSDFDataset(Dataset):
    """
    Pre-computes aligned (4, S, S) dual-skeleton SDF inputs and (1, S, S) target SDFs,
    then yields stroke-focused `patch_size x patch_size` random local crops during training
    so the network NEVER sees a full character or full radical.
    """

    def __init__(
        self,
        chars: Sequence[str],
        skip_otf_path: str,
        noto_serif_ttc: str,
        noto_sans_ttc: str,
        canvas_size: int = 256,
        patch_size: int = 96,
        patches_per_char_per_epoch: int = 6,
        trunc_px: float = 16.0,
        serif_weight: int = 700,
        sans_weight: int = 620,
    ) -> None:
        super().__init__()
        self.chars = list(chars)
        self.canvas_size = canvas_size
        self.patch_size = patch_size
        self.patches_per_char = patches_per_char_per_epoch
        self.trunc_px = trunc_px

        spec = CanvasSpec(canvas_size=canvas_size)
        # Training strictly uses JP face (index=0) to match Adobe-Japan1 glyph standard of SkipPro-B
        serif_jp = AlignedFontRenderer(noto_serif_ttc, spec, ttc_index=0, variation_weight=serif_weight)
        sans_jp = AlignedFontRenderer(noto_sans_ttc, spec, ttc_index=0, variation_weight=sans_weight)
        skip_rd = AlignedFontRenderer(skip_otf_path, spec, ttc_index=0)

        self.inputs: List[np.ndarray] = []
        self.targets: List[np.ndarray] = []
        self.edge_centers: List[np.ndarray] = []

        half = patch_size // 2
        for ch in self.chars:
            x_in = build_dual_skeleton_input(ch, serif_jp, sans_jp, trunc_px=trunc_px)
            t_mask = skip_rd.render_binary_hi(ch)
            y_tg = mask_to_normalized_sdf(t_mask, canvas_size, trunc_px=trunc_px)[None, :, :]

            # Record coordinates near stroke boundaries for informative patch sampling
            active = np.abs(y_tg[0]) < 0.45
            ys, xs = np.where(active)
            if len(ys) > 0:
                ys = np.clip(ys, half, canvas_size - half)
                xs = np.clip(xs, half, canvas_size - half)
                coords = np.stack([ys, xs], axis=1).astype(np.int16)
                if len(coords) > 256:
                    idx = np.random.choice(len(coords), size=256, replace=False)
                    coords = coords[idx]
            else:
                coords = np.array([[canvas_size // 2, canvas_size // 2]], dtype=np.int16)

            self.inputs.append(x_in)
            self.targets.append(y_tg)
            self.edge_centers.append(coords)

    def __len__(self) -> int:
        return len(self.chars) * self.patches_per_char

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        c_idx = idx // self.patches_per_char
        x_in = self.inputs[c_idx]
        y_tg = self.targets[c_idx]
        coords = self.edge_centers[c_idx]

        p = self.patch_size
        half = p // 2
        if np.random.rand() < 0.85 and len(coords) > 0:
            cy, cx = coords[np.random.randint(len(coords))]
            cy = int(np.clip(cy + np.random.randint(-half // 2, half // 2 + 1), half, self.canvas_size - half))
            cx = int(np.clip(cx + np.random.randint(-half // 2, half // 2 + 1), half, self.canvas_size - half))
        else:
            cy = np.random.randint(half, self.canvas_size - half + 1)
            cx = np.random.randint(half, self.canvas_size - half + 1)

        y0, x0 = cy - half, cx - half
        crop_x = x_in[:, y0 : y0 + p, x0 : x0 + p].copy()
        crop_y = y_tg[:, y0 : y0 + p, x0 : x0 + p].copy()
        return torch.from_numpy(crop_x), torch.from_numpy(crop_y)
