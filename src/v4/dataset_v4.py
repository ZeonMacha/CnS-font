"""
v4.0 Full-Glyph 256×256 Dataset for Global-Context Training.

Unlike v3 (96×96 random crops), v4 trains on complete 256×256 glyph images
so the Transformer can capture whole-character structural context.

Data augmentation (applied on-the-fly):
  - Weight jitter: sdf_light shifted by dw ∈ [-0.08, +0.10]
  - Uniform half-width jitter: r_u ∈ [32, 48] UPEM
  - Random small rotation: ±2°
  - Random small scale: [0.95, 1.05]
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

import cv2
from scipy.ndimage import distance_transform_edt as edt
from skimage.morphology import skeletonize

from src.sdf_utils import mask_to_normalized_sdf
from src.v2.hb_outlines import HBOutlineExtractor, SkipProOutlineExtractor, rasterize_cmds


def build_deskinned_channels(
    cmds_light,
    canvas_size: int = 256,
    padding: float = 96.0,
    trunc_px: float = 16.0,
):
    """Build 4-channel deskinned input: sdf_light, dt_skel, cos2t, sin2t."""
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

    dt_smooth = cv2.GaussianBlur(dt_skel_u, (5, 5), sigmaX=1.2)
    gx = cv2.Sobel(dt_smooth, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(dt_smooth, cv2.CV_32F, 0, 1, ksize=3)
    norm2 = gx * gx + gy * gy + 1e-6
    decay = np.exp(-dt_skel_u / 65.0).astype(np.float32)
    cos2t = (((gy * gy) - (gx * gx)) / norm2 * decay).astype(np.float32)
    sin2t = ((-2.0 * gx * gy) / norm2 * decay).astype(np.float32)
    return sdf_light, dt_skel_u, cos2t, sin2t


def pack_input_tensor(
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
    """Assemble (4, S, S) float32 input tensor."""
    upem_per_px = (1000.0 + 2.0 * padding) / float(canvas_size)
    trunc_u = trunc_px * upem_per_px
    ch0 = np.clip(sdf_light + sdf_shift, -1.0, 1.0)
    ch1 = np.clip((uniform_half_width_u - dt_skel_u) / max(trunc_u, 1e-3), -1.0, 1.0)
    return np.stack([ch0, ch1, cos2t, sin2t], axis=0).astype(np.float32)


class FullGlyphDatasetV4(Dataset):
    """
    Full-glyph 256×256 deskinned Light dataset.

    Each sample is a complete character rendered at 256×256 with 4 input channels
    and 1 target SDF channel.
    """

    def __init__(
        self,
        skip_otf: Path,
        noto_sans: Path,
        cache_dir: Path,
        canvas_size: int = 256,
        holdout_chars: Optional[set] = None,
        use_cache: bool = True,
    ) -> None:
        self.skip_otf = skip_otf
        self.noto_sans = noto_sans
        self.cache_dir = cache_dir
        self.canvas_size = canvas_size
        self.holdout = holdout_chars or set()

        cache_dir.mkdir(parents=True, exist_ok=True)
        tag = f"v4_full_n{len(holdout_chars or set())}_s{canvas_size}"
        self.in_p = cache_dir / f"v4_inputs_{tag}.npy"
        self.tg_p = cache_dir / f"v4_targets_{tag}.npy"
        self.meta_p = cache_dir / f"v4_meta_{tag}.json"

        if use_cache and self.in_p.exists() and self.tg_p.exists():
            self._load_cache()
        else:
            self._build_cache()

    def _build_cache(self) -> None:
        """Build full-glyph 256×256 SDF cache from HarfBuzz outlines."""
        skip_ext = SkipProOutlineExtractor(self.skip_otf)
        train_chars = sorted(
            chr(cp) for cp in skip_ext.cmap if 0x4E00 <= cp <= 0x9FFF and chr(cp) not in self.holdout
        )
        skip_ext.close()

        n = len(train_chars)
        print(f"[v4-Dataset] Building full-glyph cache for {n} chars at {self.canvas_size}×{self.canvas_size}...", flush=True)

        # Storage: int8 for inputs (4ch), int8 for targets (1ch)
        inputs_i8 = np.empty((n, 4, self.canvas_size, self.canvas_size), dtype=np.int8)
        targets_i8 = np.empty((n, 1, self.canvas_size, self.canvas_size), dtype=np.int8)

        trunc_px = 16.0
        tls = threading.local()

        def _worker(item: Tuple[int, str]) -> Tuple[int, np.ndarray, np.ndarray]:
            i, ch = item
            if not hasattr(tls, "snL"):
                # Use HBOutlineExtractor directly with weight=None (default weight for non-VF fonts)
                # ttc_index=0 is JP, ttc_index=2 is SC (zh-Hans) in NotoSansCJK ttc files
                tls.snL = HBOutlineExtractor(self.noto_sans, ttc_index=0, weight=None)
                tls.sk = SkipProOutlineExtractor(self.skip_otf)

            c_l = tls.snL.get_gid_and_cmds(ch, "ja")[1]
            sl, dtu, c2, s2 = build_deskinned_channels(
                c_l, canvas_size=self.canvas_size, padding=96.0, trunc_px=trunc_px
            )
            x_f = pack_input_tensor(
                sl, dtu, c2, s2,
                uniform_half_width_u=40.0,
                canvas_size=self.canvas_size,
                padding=96.0,
                trunc_px=trunc_px,
            )

            # Target: SkipPro-B original glyph SDF
            c_sk = tls.sk.get_cmds(ch)
            if not c_sk:
                # Empty target for missing glyphs
                y_f = np.zeros((1, self.canvas_size, self.canvas_size), dtype=np.float32) - 1.0
            else:
                m_sk = rasterize_cmds(c_sk, canvas_size=self.canvas_size * 2, padding=96.0)
                y_f = mask_to_normalized_sdf(m_sk, self.canvas_size, trunc_px=trunc_px)[None, :, :]

            return (
                i,
                np.clip(np.round(x_f * 127.0), -127, 127).astype(np.int8),
                np.clip(np.round(y_f * 127.0), -127, 127).astype(np.int8),
            )

        workers = min(10, max(4, os.cpu_count() or 8))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, xi, yi in tqdm(
                ex.map(_worker, enumerate(train_chars)),
                total=n,
                desc="v4 Full-Glyph Cache",
                mininterval=5.0,
            ):
                inputs_i8[i] = xi
                targets_i8[i] = yi

        np.save(self.in_p, inputs_i8)
        np.save(self.tg_p, targets_i8)

        meta = {
            "n_chars": n,
            "canvas_size": self.canvas_size,
            "chars": train_chars,
            "holdout": sorted(self.holdout),
        }
        self.meta_p.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        self.chars = train_chars
        self.inputs_i8 = inputs_i8
        self.targets_i8 = targets_i8
        print(f"[v4-Dataset] Cache saved → {self.in_p.name} + {self.tg_p.name}", flush=True)

    def _load_cache(self) -> None:
        self.inputs_i8 = np.load(self.in_p)
        self.targets_i8 = np.load(self.tg_p)
        meta = json.loads(self.meta_p.read_text(encoding="utf-8"))
        self.chars = meta["chars"]
        print(f"[v4-Dataset] Loaded cache: {len(self.chars)} chars, {self.canvas_size}×{self.canvas_size}", flush=True)

    def __len__(self) -> int:
        return len(self.chars)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        """
        Returns:
            x: (4, 256, 256) float32 input tensor with augmentation
            y: (1, 256, 256) float32 target SDF
            char: character string
        """
        x_i8 = self.inputs_i8[idx]
        y_i8 = self.targets_i8[idx]

        x = torch.from_numpy(x_i8).float().mul_(1.0 / 127.0)
        y = torch.from_numpy(y_i8).float().mul_(1.0 / 127.0)

        # ── On-the-fly augmentation ─────────────────────────────────
        # Weight jitter: shift channel 0 (sdf_light)
        dw = float(np.random.uniform(-0.08, 0.10))
        x[0] = torch.clamp(x[0] + dw, -1.0, 1.0)

        # Uniform half-width jitter: rescale channel 1 (skel_sdf)
        # Original: r_u=40, trunc=16*4.656=74.5
        # New r_u: jitter around 40
        r_u_new = float(np.random.uniform(32.0, 48.0))
        r_u_orig = 40.0
        scale_factor = r_u_orig / r_u_new
        x[1] = torch.clamp(x[1] * scale_factor, -1.0, 1.0)

        # Random small rotation (±2°) — applied to all channels
        if np.random.random() < 0.5:
            angle = float(np.random.uniform(-2.0, 2.0))
            x = self._rotate_tensor(x, angle)

        # Random small scale ([0.95, 1.05])
        if np.random.random() < 0.5:
            s = float(np.random.uniform(0.95, 1.05))
            x = self._scale_tensor(x, s)

        return x, y, self.chars[idx]

    @staticmethod
    def _rotate_tensor(x: torch.Tensor, angle_deg: float) -> torch.Tensor:
        """Rotate (C, H, W) tensor by angle_deg degrees around center."""
        from torchvision.transforms.functional import rotate
        # (C, H, W) → (1, C, H, W) → rotate → (1, C, H, W) → (C, H, W)
        x_4d = x.unsqueeze(0)
        x_rot = rotate(x_4d, angle_deg, interpolation=2, fill=0)  # BILINEAR
        return x_rot.squeeze(0).clamp(-1.0, 1.0)

    @staticmethod
    def _scale_tensor(x: torch.Tensor, scale: float) -> torch.Tensor:
        """Scale (C, H, W) tensor around center."""
        from torchvision.transforms.functional import resize
        _, h, w = x.shape
        new_h = int(h * scale)
        new_w = int(w * scale)
        x_resized = resize(x.unsqueeze(0), [new_h, new_w], interpolation=2).squeeze(0)
        # Pad or crop back to original size
        if new_h < h or new_w < w:
            pad_h = h - new_h
            pad_w = w - new_w
            pad_top = pad_h // 2
            pad_left = pad_w // 2
            x_padded = torch.nn.functional.pad(
                x_resized,
                [pad_left, pad_w - pad_left, pad_top, pad_h - pad_top],
                mode="constant",
                value=-1.0,
            )
        else:
            crop_h = new_h - h
            crop_w = new_w - w
            crop_top = crop_h // 2
            crop_left = crop_w // 2
            x_padded = x_resized[:, crop_top : crop_top + h, crop_left : crop_left + w]
        return x_padded.clamp(-1.0, 1.0)


def create_v4_dataloaders(
    skip_otf: Path,
    noto_sans: Path,
    cache_dir: Path,
    canvas_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
    holdout_chars: Optional[set] = None,
) -> Tuple[FullGlyphDatasetV4, torch.utils.data.DataLoader]:
    """Create dataset and DataLoader for v4 training."""
    holdout = holdout_chars or set("志洳菜嘉改栅畴逡红语银饭伞丌习乡卫专书东言直令骨微道龙丝")
    dataset = FullGlyphDatasetV4(
        skip_otf=skip_otf,
        noto_sans=noto_sans,
        cache_dir=cache_dir,
        canvas_size=canvas_size,
        holdout_chars=holdout,
    )

    # Use RandomSampler for full-glyph training (no patch cropping needed)
    sampler = torch.utils.data.RandomSampler(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return dataset, dataloader
