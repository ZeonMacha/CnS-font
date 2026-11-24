"""
v4.5 Full-Glyph 256x256 Dataset with Cache Lineage, VF Weight Support,
Synchronized Augmentation, and Rare-Component Oversampling.

Fixes over v4.0-v4.3 (from 审计v4.md):
  1. Explicit `noto_weight` parameter (None for static TTC, 300/400 for VF TTC)
     embedded directly in cache filename so changing fonts/weights never loads stale cache.
  2. Unified 28-char holdout set (`UNIFIED_HOLDOUT_28`) shared across train & eval.
  3. Synchronized (x, y) rotation/scale with `-1.0` exterior fill and exact additive `r_u` shift.
  4. WeightedRandomSampler (3x oversampling) for rare bowed `子` side-radicals (`孔孖孜孤孩孫孺孾仔好耔`)
     and sparse dot/sweep characters (fixes `孓`/`义` under-representation).
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt as edt
from skimage.morphology import skeletonize
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from src.sdf_utils import mask_to_normalized_sdf
from src.v2.component_bank import parse_cjkvi_ids
from src.v2.hb_outlines import HBOutlineExtractor, SkipProOutlineExtractor, rasterize_cmds
from src.v45 import UNIFIED_HOLDOUT_28

# Rare high-bow 子 side-radicals + sparse dot/sweep archetypes to oversample (3x)
OVERSAMPLE_CHARS = set(
    "子孑孔孖孜孤孩孫孺孾仔好耔"
    "乂叉文夕勺凡丸刃寸太犬尤龙戈弋广厂飞习乡卫专书东"
    "义之乏乒乓丹主乃久么及久"
)


def build_deskinned_channels(
    cmds_light,
    canvas_size: int = 256,
    padding: float = 96.0,
    trunc_px: float = 16.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build 4-channel deskinned input: (sdf_light, dt_skel_u, cos2t, sin2t)."""
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


class FullGlyphDatasetV45(Dataset):
    """Full-glyph 256x256 dataset for v4.5 training or holdout validation."""

    def __init__(
        self,
        skip_otf: Path,
        noto_sans: Path,
        cache_dir: Path,
        *,
        noto_weight: Optional[int] = None,
        canvas_size: int = 256,
        holdout_chars: Optional[Set[str]] = None,
        split: str = "train",
        augment: bool = True,
        use_cache: bool = True,
    ) -> None:
        self.skip_otf = Path(skip_otf)
        self.noto_sans = Path(noto_sans)
        self.noto_weight = noto_weight
        self.cache_dir = Path(cache_dir)
        self.canvas_size = canvas_size
        self.holdout = set(holdout_chars) if holdout_chars is not None else set(UNIFIED_HOLDOUT_28)
        self.split = split
        self.augment = augment and (split == "train")

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        font_stem = self.noto_sans.stem.replace(" ", "_")
        w_tag = f"w{noto_weight}" if noto_weight is not None else "wstatic"
        tag = f"v45_{font_stem}_{w_tag}_{split}_h{len(self.holdout)}_s{canvas_size}"
        self.in_p = self.cache_dir / f"inputs_{tag}.npy"
        self.tg_p = self.cache_dir / f"targets_{tag}.npy"
        self.meta_p = self.cache_dir / f"meta_{tag}.json"

        if use_cache and self.in_p.exists() and self.tg_p.exists() and self.meta_p.exists():
            self._load_cache()
        else:
            self._build_cache()

    def _build_cache(self) -> None:
        skip_ext = SkipProOutlineExtractor(self.skip_otf)
        all_cjk = sorted(chr(cp) for cp in skip_ext.cmap if 0x4E00 <= cp <= 0x9FFF)
        skip_ext.close()

        if self.split == "val":
            target_chars = [c for c in all_cjk if c in self.holdout]
        else:
            target_chars = [c for c in all_cjk if c not in self.holdout]

        n = len(target_chars)
        print(
            f"[v45-Dataset] Building {self.split} cache: {n} chars "
            f"(font={self.noto_sans.name}, weight={self.noto_weight}, S={self.canvas_size})...",
            flush=True,
        )

        inputs_i8 = np.empty((n, 4, self.canvas_size, self.canvas_size), dtype=np.int8)
        targets_i8 = np.empty((n, 1, self.canvas_size, self.canvas_size), dtype=np.int8)

        trunc_px = 16.0
        tls = threading.local()

        def _worker(item: Tuple[int, str]) -> Tuple[int, np.ndarray, np.ndarray]:
            i, ch = item
            if not hasattr(tls, "snL"):
                tls.snL = HBOutlineExtractor(self.noto_sans, ttc_index=0, weight=self.noto_weight)
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

            c_sk = tls.sk.get_cmds(ch)
            if not c_sk:
                y_f = np.full((1, self.canvas_size, self.canvas_size), -1.0, dtype=np.float32)
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
                ex.map(_worker, enumerate(target_chars)),
                total=n,
                desc=f"v45 Cache ({self.split})",
                mininterval=5.0,
            ):
                inputs_i8[i] = xi
                targets_i8[i] = yi

        np.save(self.in_p, inputs_i8)
        np.save(self.tg_p, targets_i8)
        meta = {
            "split": self.split,
            "n_chars": n,
            "canvas_size": self.canvas_size,
            "noto_sans": self.noto_sans.name,
            "noto_weight": self.noto_weight,
            "chars": target_chars,
            "holdout": sorted(self.holdout),
        }
        self.meta_p.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
        self.chars = target_chars
        self.inputs_i8 = inputs_i8
        self.targets_i8 = targets_i8
        print(f"[v45-Dataset] Saved -> {self.in_p.name}", flush=True)

    def _load_cache(self) -> None:
        self.inputs_i8 = np.load(self.in_p)
        self.targets_i8 = np.load(self.tg_p)
        meta = json.loads(self.meta_p.read_text(encoding="utf-8"))
        self.chars = meta["chars"]
        print(
            f"[v45-Dataset] Loaded {self.split} cache: {len(self.chars)} chars "
            f"(font={meta.get('noto_sans')}, weight={meta.get('noto_weight')})",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.chars)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        x = torch.from_numpy(self.inputs_i8[idx]).float().mul_(1.0 / 127.0)
        y = torch.from_numpy(self.targets_i8[idx]).float().mul_(1.0 / 127.0)

        if not self.augment:
            return x, y, self.chars[idx]

        # 1. Weight jitter on ch0
        dw = float(np.random.uniform(-0.08, 0.10))
        x[0] = torch.clamp(x[0] + dw, -1.0, 1.0)

        # 2. Exact additive uniform half-width jitter on ch1
        upem_per_px = (1000.0 + 2.0 * 96.0) / float(self.canvas_size)
        trunc_u = 16.0 * upem_per_px
        r_u_new = float(np.random.uniform(32.0, 48.0))
        x[1] = torch.clamp(x[1] + (r_u_new - 40.0) / trunc_u, -1.0, 1.0)

        # 3. Synchronized rotation (±2°) on (x, y) with -1.0 exterior fill
        if np.random.random() < 0.5:
            angle = float(np.random.uniform(-2.0, 2.0))
            x_sdf = self._rotate_tensor(x[0:2], angle, fill=-1.0)
            x_ori = self._rotate_tensor(x[2:4], angle, fill=0.0)
            rad2 = float(np.deg2rad(-2.0 * angle))
            c2a, s2a = float(np.cos(rad2)), float(np.sin(rad2))
            ori_c = x_ori[0] * c2a - x_ori[1] * s2a
            ori_s = x_ori[0] * s2a + x_ori[1] * c2a
            x = torch.cat([x_sdf, torch.stack([ori_c, ori_s], dim=0)], dim=0)
            y = self._rotate_tensor(y, angle, fill=-1.0)

        # 4. Synchronized scale ([0.95, 1.05]) on (x, y) with -1.0 exterior fill
        if np.random.random() < 0.5:
            s = float(np.random.uniform(0.95, 1.05))
            x_sdf = self._scale_tensor(x[0:2], s, fill=-1.0)
            x_ori = self._scale_tensor(x[2:4], s, fill=0.0)
            x = torch.cat([x_sdf, x_ori], dim=0)
            y = self._scale_tensor(y, s, fill=-1.0)

        return x, y, self.chars[idx]

    @staticmethod
    def _rotate_tensor(x: torch.Tensor, angle_deg: float, fill: float = -1.0) -> torch.Tensor:
        from torchvision.transforms.functional import InterpolationMode, rotate
        return rotate(x.unsqueeze(0), angle_deg, interpolation=InterpolationMode.BILINEAR, fill=[fill]).squeeze(0).clamp(-1.0, 1.0)

    @staticmethod
    def _scale_tensor(x: torch.Tensor, scale: float, fill: float = -1.0) -> torch.Tensor:
        from torchvision.transforms.functional import InterpolationMode, resize
        _, h, w = x.shape
        new_h = max(2, int(round(h * scale)))
        new_w = max(2, int(round(w * scale)))
        x_resized = resize(x.unsqueeze(0), [new_h, new_w], interpolation=InterpolationMode.BILINEAR).squeeze(0)
        if new_h < h or new_w < w:
            pad_h, pad_w = h - new_h, w - new_w
            pt, pl = pad_h // 2, pad_w // 2
            x_out = torch.nn.functional.pad(x_resized, [pl, pad_w - pl, pt, pad_h - pt], mode="constant", value=fill)
        else:
            ct, cl = (new_h - h) // 2, (new_w - w) // 2
            x_out = x_resized[:, ct : ct + h, cl : cl + w]
        return x_out.clamp(-1.0, 1.0)


def build_sample_weights(chars: List[str], ids_path: Optional[Path] = None, boost: float = 3.0) -> List[float]:
    """Give 3x sampling weight to rare bowed `子` side-radicals and sparse dot/sweep glyphs."""
    ids_map: Dict[str, str] = parse_cjkvi_ids(ids_path) if (ids_path and ids_path.exists()) else {}
    weights: List[float] = []
    n_boosted = 0
    for ch in chars:
        seq = ids_map.get(ch, "")
        is_side_zi = seq.startswith("⿰") and ("子" in seq)
        if ch in OVERSAMPLE_CHARS or is_side_zi:
            weights.append(boost)
            n_boosted += 1
        else:
            weights.append(1.0)
    print(f"[v45-Dataset] WeightedSampler: boosted {n_boosted}/{len(chars)} rare/sparse chars by {boost:.1f}x", flush=True)
    return weights


def create_v45_dataloaders(
    skip_otf: Path,
    noto_sans: Path,
    cache_dir: Path,
    *,
    noto_weight: Optional[int] = None,
    ids_path: Optional[Path] = None,
    canvas_size: int = 256,
    batch_size: int = 16,
    num_workers: int = 4,
    holdout_chars: Optional[Set[str]] = None,
) -> Tuple[FullGlyphDatasetV45, DataLoader, FullGlyphDatasetV45, DataLoader]:
    """Create train (with rare-glyph oversampling) and holdout val DataLoaders."""
    holdout = set(holdout_chars) if holdout_chars is not None else set(UNIFIED_HOLDOUT_28)
    train_ds = FullGlyphDatasetV45(
        skip_otf, noto_sans, cache_dir,
        noto_weight=noto_weight, canvas_size=canvas_size,
        holdout_chars=holdout, split="train", augment=True,
    )
    val_ds = FullGlyphDatasetV45(
        skip_otf, noto_sans, cache_dir,
        noto_weight=noto_weight, canvas_size=canvas_size,
        holdout_chars=holdout, split="val", augment=False,
    )

    weights = build_sample_weights(train_ds.chars, ids_path=ids_path, boost=3.0)
    sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=min(batch_size, max(len(val_ds), 1)), shuffle=False,
        num_workers=0, pin_memory=True,
    )
    return train_ds, train_loader, val_ds, val_loader
